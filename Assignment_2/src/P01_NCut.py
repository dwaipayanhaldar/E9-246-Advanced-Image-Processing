import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from numpy import array, reshape, shape, matrix, ones, zeros, sqrt, sort, arange
from numpy import nonzero, fromfile, tile, append, prod, double, argsort, sign
from numpy import kron, multiply, divide, abs, reshape, asarray
from numpy.random import rand
from scipy.sparse import coo_matrix, csc_matrix, spdiags, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm, svd, LinAlgError
from scipy.spatial.distance import cdist
from skimage.filters import sobel



def adjacency_matrix(image, sigma_I, sigma_X, threshold=1e-3):
    """
    Full pairwise adjacency: every pixel connected to every other pixel.
    W(i,j) = exp(-||I_i - I_j||^2 / 2*sigma_I^2  -  ||X_i - X_j||^2 / 2*sigma_X^2)

    Trick: normalise features by their sigma and concatenate, then a single
    cdist call gives the combined exponent in one shot.
    """
    H, W_img = image.shape[:2]
    N = H * W_img
    pixels = image.reshape(N, -1).astype(np.float32)

    y_coords, x_coords = np.mgrid[:H, :W_img]
    coords = np.column_stack([y_coords.ravel(),
                              x_coords.ravel()]).astype(np.float32)

    # Concatenate [intensity/sigma_I, x/sigma_X, y/sigma_X]
    # cdist squared-euclidean on this gives  ||I_i-I_j||^2/sigma_I^2 + ||X_i-X_j||^2/sigma_X^2
    features = np.hstack([pixels / sigma_I, coords / sigma_X])
    combined_sq = cdist(features, features, 'sqeuclidean').astype(np.float32)

    W = np.exp(-combined_sq / 2, dtype=np.float32)
    np.fill_diagonal(W, 0)
    W[W < threshold] = 0          # drop near-zero entries for sparsity

    return csc_matrix(W)

def adjacency_matrix_modified_affinity(image, sigma_I, sigma_X, sigma_E, threshold=1e-3):
    """
    Modified affinity with pairwise edge term.

    W(i,j) = exp(
        -||I_i - I_j||^2 / (2*sigma_I^2)
        -||X_i - X_j||^2 / (2*sigma_X^2)
        -||G_i - G_j||^2 / (2*sigma_E^2)
    )

    where G is gradient magnitude (edge strength).
    """

    H, W_img = image.shape[:2]
    N = H * W_img

    # Flatten intensity
    pixels = image.reshape(N, 1).astype(np.float32)

    # Spatial coordinates
    y_coords, x_coords = np.mgrid[:H, :W_img]
    coords = np.column_stack([y_coords.ravel(), x_coords.ravel()]).astype(np.float32)

    edge_strength = sobel(image).reshape(N, 1).astype(np.float32)

    edge_strength /= (edge_strength.max() + 1e-8)


    features = np.hstack([
        pixels / sigma_I,
        coords / sigma_X,
        edge_strength / sigma_E
    ])

    combined_sq = cdist(features, features, 'sqeuclidean').astype(np.float32)

    # Affinity matrix
    W = np.exp(-combined_sq / 2.0).astype(np.float32)

    np.fill_diagonal(W, 0)
    W[W < threshold] = 0

    return csc_matrix(W)



def degree_matrix(W): 
    D = np.array(W.sum(axis=1)).flatten()
    return D

def compute_ncut_value(W, labels):
    A = np.where(labels == 1)[0]
    B = np.where(labels == 0)[0]

    if len(A) == 0 or len(B) == 0:
        return np.inf

    cut_AB = W[A][:, B].sum()
    assoc_A = W[A][:, :].sum()
    assoc_B = W[B][:, :].sum()

    ncut_value = cut_AB / (assoc_A + 1e-12) + cut_AB / (assoc_B + 1e-12)
    return float(ncut_value)

def choose_threshold(W, eigenvec):
    values_to_try = np.linspace(eigenvec.min(), eigenvec.max(), 40)

    best_ncut = np.inf
    best_thresh = 0

    for thresh in values_to_try:
        labels = (eigenvec > thresh).astype(int)
        ncut_value = compute_ncut_value(W, labels)

        if ncut_value < best_ncut:
            best_ncut = ncut_value
            best_thresh = thresh

    return best_thresh

def ncut_split(W_sub):
    N = W_sub.shape[0]

    D_sub = degree_matrix(W_sub)
    d_inv_sqrt = 1.0 / np.sqrt(D_sub + 1e-12)

    L_sym = np.eye(N) - diags(d_inv_sqrt) @ W_sub @ diags(d_inv_sqrt)

    _, eigenvecs = eigsh(L_sym, k=2, sigma=1e-10, which='LM')

    eig = eigenvecs[:, 1]

    # thresh = choose_threshold(W_sub, eig)
    labels = (eig > 0).astype(int)

    ncut_val = compute_ncut_value(W_sub, labels)

    return ncut_val, labels


def recursive_ncut(W, image_shape, threshold=0.001, min_size=500):
    """
    Iteratively split segments using NCut (BFS order).
    Stops splitting a segment when:
      - NCut value >= threshold  (cut is too expensive / segment is homogeneous)
      - segment has fewer than min_size pixels
    Returns an integer label map of shape image_shape[:2].
    """
    N = W.shape[0]
    label_map = np.zeros(N, dtype=int)
    next_label = [1]

    queue = [np.arange(N, dtype=np.intp)]

    while queue:
        indices = queue.pop(0)

        if len(indices) < min_size:
            continue

        W_sub = W[indices][:, indices]

        ncut_val, sub_labels = ncut_split(W_sub)
        print(f"  Segment size={len(indices):6d},  NCut={ncut_val:.4f}")

        if ncut_val >= threshold:
            continue  # homogeneous enough, stop splitting

        # Assign a new label to one half; the other keeps its current label
        idx_A = indices[sub_labels == 1]
        idx_B = indices[sub_labels == 0]
        label_map[idx_B] = next_label[0]
        next_label[0] += 1

        queue.append(idx_A)
        queue.append(idx_B)

    return label_map.reshape(image_shape[:2])


if __name__ == "__main__":
    from skimage.transform import resize

    script_dir = os.path.dirname(os.path.abspath(__file__))
    image = io.imread(os.path.join(script_dir, "../Data/P01/chess.jpg"))
    image_grayscale = color.rgb2gray(image)

    # Downsample: NCut is designed for small graphs (~100x100)
    scale = 0.15
    H, W_img = image_grayscale.shape
    image_small = resize(image_grayscale, (int(H * scale), int(W_img * scale)), anti_aliasing=True)
    print(f"Resized to {image_small.shape}")

    sigma_I = 0.1
    sigma_X = 15.0
    sigma_E = 10.0
    print("Building adjacency matrix...")
    W = adjacency_matrix_modified_affinity(image_small, sigma_I, sigma_X, sigma_E)

    print("Running recursive NCut...")
    label_map = recursive_ncut(W, image_small.shape, threshold=0.05, min_size=100)

    print(f"Number of segments: {label_map.max() + 1}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image_grayscale, cmap='gray')
    axes[0].set_title("Input (original)")
    axes[0].axis('off')
    axes[1].imshow(label_map, cmap='tab20')
    axes[1].set_title("Recursive NCut Segmentation")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    





# exception hander for singular value decomposition
class SVDError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)



def ncut(W, nbEigenValues):
    # parameters
    offset = .5
    maxiterations = 100
    eigsErrorTolerence = 1e-6
    truncMin = 1e-6
    eps = 2.2204e-16

    m = shape(W)[1]

    d = abs(W).sum(0)
    dr = 0.5 * (d - W.sum(0))
    d = d + offset * 2
    dr = dr + offset

    # calculation of the normalized LaPlacian
    W = W + spdiags(dr, [0], m, m, "csc")
    Dinvsqrt = spdiags((1.0 / sqrt(d + eps)), [0], m, m, "csc")
    P = Dinvsqrt * (W * Dinvsqrt);


    eigen_val, eigen_vec = eigsh(P, nbEigenValues, tol=eigsErrorTolerence, which='LA')

    # sort the eigen_vals so that the first
    # is the largest
    i = argsort(-eigen_val)
    eigen_val = eigen_val[i]
    eigen_vec = eigen_vec[:, i]

    # normalize the returned eigenvectors
    eigen_vec = Dinvsqrt * matrix(eigen_vec)
    norm_ones = norm(ones((m, 1)))
    for i in range(0, shape(eigen_vec)[1]):
        eigen_vec[:, i] = (eigen_vec[:, i] / norm(eigen_vec[:, i])) * norm_ones
        if eigen_vec[0, i] != 0:
            eigen_vec[:, i] = -1 * eigen_vec[:, i] * sign(eigen_vec[0, i])

    return (eigen_val, eigen_vec)


def discretisation(eigen_vec):
    eps = 2.2204e-16

    # normalize the eigenvectors
    [n, k] = shape(eigen_vec)
    vm = kron(ones((1, k)), sqrt(multiply(eigen_vec, eigen_vec).sum(1)))
    eigen_vec = divide(eigen_vec, vm)

    svd_restarts = 0
    exitLoop = 0

    ### if there is an exception we try to randomize and rerun SVD again
    ### do this 30 times
    while (svd_restarts < 30) and (exitLoop == 0):

        # initialize algorithm with a random ordering of eigenvectors
        c = zeros((n, 1))
        R = matrix(zeros((k, k)))
        R[:, 0] = eigen_vec[int(rand(1) * (n)), :].transpose()

        for j in range(1, k):
            c = c + abs(eigen_vec * R[:, j - 1])
            R[:, j] = eigen_vec[c.argmin(), :].transpose()

        lastObjectiveValue = 0
        nbIterationsDiscretisation = 0
        nbIterationsDiscretisationMax = 20


        while exitLoop == 0:
            nbIterationsDiscretisation = nbIterationsDiscretisation + 1

            # rotate the original eigen_vectors
            tDiscrete = eigen_vec * R

            # discretise the result by setting the max of each row=1 and
            # other values to 0
            j = reshape(asarray(tDiscrete.argmax(1)), n)
            eigenvec_discrete = csc_matrix((ones(len(j)), (range(0, n), array(j))), shape=(n, k))

            # calculate a rotation to bring the discrete eigenvectors cluster to the
            # original eigenvectors
            tSVD = eigenvec_discrete.transpose() * eigen_vec
            # catch a SVD convergence error and restart
            try:
                U, S, Vh = svd(tSVD)
                svd_restarts += 1
            except LinAlgError:
                # catch exception and go back to the beginning of the loop
                print("SVD did not converge, randomizing and trying again", file=sys.stderr)
                break

            # test for convergence
            NcutValue = 2 * (n - S.sum())
            if ((abs(NcutValue - lastObjectiveValue) < eps ) or
                    ( nbIterationsDiscretisation > nbIterationsDiscretisationMax )):
                exitLoop = 1
            else:
                # otherwise calculate rotation and continue
                lastObjectiveValue = NcutValue
                R = matrix(Vh).transpose() * matrix(U).transpose()

    if exitLoop == 0:
        raise SVDError("SVD did not converge after 30 retries")
    else:
        return (eigenvec_discrete)
    


# if __name__ == "__main__":
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     image = io.imread(os.path.join(script_dir, "../Data/P01/chess.jpg"))
#     image_grayscale = color.rgb2gray(image)

#     scale = 0.2
#     H, W_img = image_grayscale.shape
#     image_small = resize(image_grayscale, (int(H * scale), int(W_img * scale)), anti_aliasing=True)
#     print(f"Resized to {image_small.shape}")
#     # Resize to 100x100: full pairwise W is 10K x 10K — tractable
#     # image_small = resize(image_grayscale, (200, 200), anti_aliasing=True)
#     # print(f"Resized to {image_small.shape}")

#     # sigma_I: intensity sensitivity (grayscale in [0,1])
#     # sigma_X: spatial reach — set to ~15% of image width so distant
#     #          but similar pixels can still connect
#     print("Building full pairwise adjacency matrix...")
#     W = adjacency_matrix(image_small, sigma_I=0.05, sigma_X=30.0)
#     print(f"W nnz: {W.nnz}")

#     nbEigenValues = 2
#     print("Running NCut eigenvector decomposition...")
#     eigen_val, eigen_vec = ncut(W, nbEigenValues)
#     print("Eigenvalues:", eigen_val)

#     print("Discretising...")
#     eigenvec_discrete = discretisation(eigen_vec)

#     labels = asarray(eigenvec_discrete.argmax(1)).flatten()
#     label_map = labels.reshape(image_small.shape[:2])

#     print(f"Number of segments: {label_map.max() + 1}")
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].imshow(image_small, cmap='gray')
#     axes[0].set_title("Input (100×100)")
#     axes[0].axis('off')
#     axes[1].imshow(label_map, cmap='tab20')
#     axes[1].set_title("NCut Segmentation")
#     axes[1].axis('off')
#     plt.tight_layout()
#     plt.show()
