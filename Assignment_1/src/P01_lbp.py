import numpy as np
import matplotlib.pyplot as plt

def local_binary_pattern(image):
    height, width = image.shape
    histogram = np.zeros(256, dtype=int)

    for i in range(1, height-1):
        for j in range(1, width-1):
            c = image[i,j]
            b = ''
            b += '1' if image[i-1,j-1] >= c else '0'
            b += '1' if image[i-1,j]   >= c else '0'
            b += '1' if image[i-1,j+1] >= c else '0'
            b += '1' if image[i,j+1]   >= c else '0'
            b += '1' if image[i+1,j+1] >= c else '0'
            b += '1' if image[i+1,j]   >= c else '0'
            b += '1' if image[i+1,j-1] >= c else '0'
            b += '1' if image[i,j-1]   >= c else '0'


            idx = int(b, 2)
            histogram[idx] += 1

    histogram = histogram / ((height-2)*(width-2))
    return histogram

def local_binary_pattern_vectorised(image):
    center = image[1:-1, 1:-1]

    neighbors = [
        image[0:-2, 0:-2],  
        image[0:-2, 1:-1],   
        image[0:-2, 2:],     
        image[1:-1, 2:],    
        image[2:,   2:],    
        image[2:,   1:-1],   
        image[2:,   0:-2],  
        image[1:-1, 0:-2],  
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for i, neighbor in enumerate(neighbors):
        lbp += ((neighbor >= center).astype(np.uint8) << (7 - i))

    histogram = np.bincount(lbp.ravel(), minlength=256).astype(np.float64)
    histogram /= lbp.size

    return histogram

def local_binary_pattern_riu(image):
    center = image[1:-1, 1:-1]

    neighbors = [
        image[0:-2, 0:-2],  
        image[0:-2, 1:-1],   
        image[0:-2, 2:],     
        image[1:-1, 2:],    
        image[2:,   2:],    
        image[2:,   1:-1],   
        image[2:,   0:-2],  
        image[1:-1, 0:-2],  
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for i, neighbor in enumerate(neighbors):
        lbp += ((neighbor >= center).astype(np.uint8) << (7 - i))

    transitions = np.zeros_like(lbp, dtype=np.uint8)
    for i in range(8):
        bit_current = (lbp >> (7 - i)) & 1
        bit_next = (lbp >> (7 - (i + 1) % 8)) & 1
        transitions += np.abs(bit_current.astype(np.int8) - bit_next.astype(np.int8)).astype(np.uint8)

    num_ones = np.zeros_like(lbp, dtype=np.uint8)
    for i in range(8):
        num_ones += (lbp >> i) & 1
    
    lbp_riu2 = np.where(transitions <= 2, num_ones, 9)

    histogram = np.bincount(lbp_riu2.ravel(), minlength=10).astype(np.float64)
    histogram /= lbp_riu2.size

    return histogram