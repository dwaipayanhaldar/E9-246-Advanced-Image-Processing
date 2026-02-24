import json
import os
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)


class PersonSegDataset(Dataset):
    """
    Person semantic segmentation dataset in COCO format.

    Converts polygon annotations to binary pixel masks:
        0 = background
        1 = person

    Args:
        img_dir   : path to the folder containing JPEG images
        ann_file  : path to instances_default.json (COCO format)
        split_file: path to train_test_split.json
        split     : 'train' or 'test'
        img_size  : square output resolution (both image and mask are resized)
    """

    def __init__(self, img_dir: str, ann_file: str, split_file: str,
                 split: str = 'train', img_size: int = 512):
        self.img_dir  = img_dir
        self.img_size = img_size

        # ── Load COCO annotations ────────────────────────────────────────────
        with open(ann_file) as f:
            coco = json.load(f)

        # ── Load train / test split ──────────────────────────────────────────
        with open(split_file) as f:
            split_files = set(json.load(f)[split])

        # image_id → file name  &  image_id → (W, H)
        self.id_to_file: dict[int, str]         = {}
        self.id_to_size: dict[int, tuple[int, int]] = {}
        for img in coco['images']:
            if img['file_name'] in split_files:
                self.id_to_file[img['id']] = img['file_name']
                self.id_to_size[img['id']] = (img['width'], img['height'])

        # image_id → list of annotations (only for images in this split)
        self.id_to_anns: dict[int, list] = {k: [] for k in self.id_to_file}
        for ann in coco['annotations']:
            iid = ann['image_id']
            if iid in self.id_to_anns:
                self.id_to_anns[iid].append(ann)

        self.image_ids = sorted(self.id_to_file.keys())

        # ── Image pre-processing ─────────────────────────────────────────────
        # ImageNet normalisation (encoder is pre-trained on ImageNet)
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    # ── mask rasterisation ───────────────────────────────────────────────────
    def _make_mask(self, img_id: int) -> Image.Image:
        """Draw filled polygons onto a blank L-mode image (0=bg, 1=person)."""
        w, h = self.id_to_size[img_id]
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for ann in self.id_to_anns[img_id]:
            segs = ann.get('segmentation', [])
            if isinstance(segs, list):          # polygon format
                for seg in segs:
                    if len(seg) >= 6:           # at least 3 (x,y) pairs
                        poly = list(zip(seg[::2], seg[1::2]))
                        draw.polygon(poly, fill=1)
        return mask

    # ── Dataset interface ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]

        # load & transform image
        image = Image.open(
            os.path.join(self.img_dir, self.id_to_file[img_id])
        ).convert('RGB')
        image = self.img_transform(image)                    # [3, H, W]  float32

        # rasterise mask then resize with NEAREST to keep integer labels
        mask = self._make_mask(img_id)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # [H, W]  int64

        return image, mask



# FCN 32s with ResNet-18 backbone


class FCN32s(nn.Module):
    """
    Fully Convolutional Network — 32s  (Long, Shelhamer & Darrell, CVPR 2015).

    Modifications to ResNet18 (Section 3.1 of the FCN paper):
      1. avgpool and fc layers are REMOVED.
      2. A 1×1 convolution produces per-location class scores
         (mathematically equivalent to a fully-connected classifier applied
          densely at every spatial position — this is the key FCN insight).
      3. A single 32× transposed convolution up-samples the coarse score map
         back to full input resolution (FCN-32s variant).

    With a 512×512 input:
        conv1+bn+relu+maxpool : [B, 64,  128, 128]
        layer1                : [B, 64,  128, 128]
        layer2                : [B, 128,  64,  64]
        layer3                : [B, 256,  32,  32]
        layer4                : [B, 512,  16,  16]   ← deepest feature map
        score (1×1 conv)      : [B, C,    16,  16]
        upsample_32x          : [B, C,   512, 512]   ← back to input size

    Loss: per-pixel multinomial logistic loss = nn.CrossEntropyLoss()
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # ── Encoder: ResNet18 WITHOUT avgpool and fc ──────────────────────────
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet  = models.resnet18(weights=weights)

        self.encoder = nn.Sequential(
            resnet.conv1,    # stride 2  → H/2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # stride 2  → H/4
            resnet.layer1,   # stride 1  → H/4   (64 channels)
            resnet.layer2,   # stride 2  → H/8   (128 channels)
            resnet.layer3,   # stride 2  → H/16  (256 channels)
            resnet.layer4,   # stride 2  → H/32  (512 channels)
        )

        # ── Scorer: 1×1 conv (replaces FC) ───────────────────────────────────
        # This produces a C-channel "heat-map" at 1/32 of input resolution,
        # one score per class per spatial location.
        self.score = nn.Conv2d(512, num_classes, kernel_size=1)

        # ── Decoder: single 32× transposed convolution (FCN-32s) ─────────────
        # Output size formula: (in-1)*stride - 2*pad + kernel
        #   = (16-1)*32 - 2*16 + 64 = 480 - 32 + 64 = 512  ✓  (for 512 input)
        # groups=num_classes → each channel upsampled independently;
        # weight shape becomes [C, 1, 64, 64], allowing clean bilinear init.
        self.upsample_32x = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=64, stride=32, padding=16,
            groups=num_classes, bias=False,
        )
        self._init_bilinear()

    # ── bilinear initialisation (Section 3.3, FCN paper) ─────────────────────
    def _init_bilinear(self) -> None:
        """
        Fill the transposed-conv weight with a 2-D bilinear kernel so that,
        before any training, the decoder performs plain bilinear interpolation.
        This gives a much better starting point than random initialisation.
        """
        ks     = 64           # kernel size
        factor = ks // 2      # = 32  (== upsampling stride)
        center = factor - 0.5 # centre of the kernel (0-indexed, even kernel)
        rows, cols = np.ogrid[:ks, :ks]
        filt = ((1.0 - np.abs(rows - center) / factor) *
                (1.0 - np.abs(cols - center) / factor)).astype(np.float32)
        # weight shape: [C, 1, ks, ks]  (grouped conv, 1 in-ch per group)
        weight = torch.from_numpy(filt).unsqueeze(0).unsqueeze(0)
        weight = weight.repeat(self.num_classes, 1, 1, 1)
        self.upsample_32x.weight.data.copy_(weight)

    # ── forward pass ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)       # [B, 512, H/32, W/32]
        x = self.score(x)         # [B, C,   H/32, W/32]  — 1×1 conv scorer
        x = self.upsample_32x(x)  # [B, C,   H,    W   ]  — 32× upsample
        return x
    

# FCN8s with skip connections 


class FCN8s(nn.Module):
    """
    Fully Convolutional Network — 8s  (Long, Shelhamer & Darrell, CVPR 2015).

    Unlike FCN-32s (single 32× upsample), FCN-8s fuses predictions from
    three different encoder depths before the final upsample, recovering
    finer spatial detail via skip connections.

    Encoder feature maps used (ResNet18, 512×512 input):
        layer2 → [B, 128, H/8,  W/8 ]  ← pool3 skip
        layer3 → [B, 256, H/16, W/16]  ← pool4 skip
        layer4 → [B, 512, H/32, W/32]  ← pool5 (deepest)

    Decoder (FCN-8s):
        score_pool5  (1×1)  : [B, C, H/32, W/32]
        upsample_2x_a (×2)  : [B, C, H/16, W/16]
        + score_pool4 (1×1) : [B, C, H/16, W/16]  ← fuse pool4
        upsample_2x_b (×2)  : [B, C, H/8,  W/8 ]
        + score_pool3 (1×1) : [B, C, H/8,  W/8 ]  ← fuse pool3
        upsample_8x   (×8)  : [B, C, H,    W   ]  ← final output
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet  = models.resnet18(weights=weights)

        # ── Encoder (split to capture intermediate maps) ──────────────────────
        self.stem   = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )                    # [B, 64,  H/4,  W/4]
        self.layer1 = resnet.layer1   # [B, 64,  H/4,  W/4]
        self.layer2 = resnet.layer2   # [B, 128, H/8,  W/8 ]  ← pool3
        self.layer3 = resnet.layer3   # [B, 256, H/16, W/16]  ← pool4
        self.layer4 = resnet.layer4   # [B, 512, H/32, W/32]  ← pool5

        # ── 1×1 conv scorers at each depth ───────────────────────────────────
        self.score_pool5 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(128, num_classes, kernel_size=1)

        # ── Decoder transposed convolutions ──────────────────────────────────
        # 2× upsample: kernel=4, stride=2, padding=1
        #   output = (in-1)*2 - 2*1 + 4 = 2*in  ✓
        self.upsample_2x_a = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=4, stride=2, padding=1,
            groups=num_classes, bias=False,
        )
        self.upsample_2x_b = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=4, stride=2, padding=1,
            groups=num_classes, bias=False,
        )
        # 8× upsample: kernel=16, stride=8, padding=4
        #   output = (in-1)*8 - 2*4 + 16 = 8*in  ✓
        self.upsample_8x = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=16, stride=8, padding=4,
            groups=num_classes, bias=False,
        )
        self._init_bilinear()

    # ── bilinear initialisation ───────────────────────────────────────────────
    def _bilinear_weight(self, stride: int) -> torch.Tensor:
        """Return bilinear-interpolation weight for a given stride."""
        ks     = 2 * stride
        factor = stride
        center = factor - 0.5          # even kernel centre
        og     = np.ogrid[:ks, :ks]
        filt   = ((1.0 - np.abs(og[0] - center) / factor) *
                  (1.0 - np.abs(og[1] - center) / factor)).astype(np.float32)
        w = torch.from_numpy(filt).unsqueeze(0).unsqueeze(0)  # [1,1,ks,ks]
        return w.repeat(self.num_classes, 1, 1, 1)            # [C,1,ks,ks]

    def _init_bilinear(self) -> None:
        w2 = self._bilinear_weight(2)
        w8 = self._bilinear_weight(8)
        self.upsample_2x_a.weight.data.copy_(w2)
        self.upsample_2x_b.weight.data.copy_(w2)
        self.upsample_8x.weight.data.copy_(w8)

    # ── forward pass ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ──
        x     = self.stem(x)    # [B, 64,  H/4,  W/4]
        x     = self.layer1(x)  # [B, 64,  H/4,  W/4]
        pool3 = self.layer2(x)  # [B, 128, H/8,  W/8 ]
        pool4 = self.layer3(pool3)  # [B, 256, H/16, W/16]
        pool5 = self.layer4(pool4)  # [B, 512, H/32, W/32]

        # ── Decoder: pool5 → fuse pool4 ──
        s5      = self.score_pool5(pool5)        # [B, C, H/32, W/32]
        s5_up   = self.upsample_2x_a(s5)         # [B, C, H/16, W/16]
        s4      = self.score_pool4(pool4)         # [B, C, H/16, W/16]
        fuse4   = s5_up + s4                      # [B, C, H/16, W/16]

        # ── Decoder: fuse pool3 ──
        fuse4_up = self.upsample_2x_b(fuse4)     # [B, C, H/8,  W/8 ]
        s3       = self.score_pool3(pool3)        # [B, C, H/8,  W/8 ]
        fuse3    = fuse4_up + s3                  # [B, C, H/8,  W/8 ]

        # ── Final 8× upsample to input resolution ──
        out = self.upsample_8x(fuse3)             # [B, C, H,    W   ]
        return out



# Training utilities

def calculate_dice_loss(logits, mask, eps=1e-6):
    """
    Soft Dice loss for binary segmentation with a 2-channel output.

    logits : [B, 2, H, W]  — raw scores from the model
    mask   : [B, H, W]     — integer labels (0=background, 1=person)

    Steps:
      1. softmax over the class dimension → per-class probabilities
      2. take channel-1 (person) → [B, H, W]  probability of being 'person'
      3. cast mask to float and compute dice per sample, then average
    """
    # [B, 2, H, W] → softmax → take person channel → [B, H, W]
    probs = torch.softmax(logits, dim=1)[:, 1, :, :]   # [B, H, W]
    mask_f = mask.float()                               # [B, H, W]

    batch_size = probs.size(0)
    probs_flat = probs.contiguous().view(batch_size, -1)    # [B, H*W]
    mask_flat  = mask_f.contiguous().view(batch_size, -1)   # [B, H*W]

    intersection = torch.sum(probs_flat * mask_flat, dim=1)          # [B]
    union        = torch.sum(probs_flat, dim=1) + torch.sum(mask_flat, dim=1)  # [B]
    dice         = (2.0 * intersection + eps) / (union + eps)        # [B]

    return (1.0 - dice).mean()


def compute_confusion_matrix(preds_flat: np.ndarray,
                              targets_flat: np.ndarray,
                              num_classes: int) -> np.ndarray:
    """
    Vectorised confusion matrix using np.bincount.
    C[true, pred] = number of pixels.
    """
    valid  = (targets_flat >= 0) & (targets_flat < num_classes)
    p      = preds_flat[valid].astype(np.int64)
    t      = targets_flat[valid].astype(np.int64)
    counts = np.bincount(num_classes * t + p, minlength=num_classes ** 2)
    return counts.reshape(num_classes, num_classes)


def confusion_to_metrics(conf: np.ndarray) -> tuple[float, float]:
    """Derive pixel-accuracy and mean-IoU from a confusion matrix."""
    pixel_acc = np.diag(conf).sum() / (conf.sum() + 1e-8)
    iou_list  = []
    for c in range(conf.shape[0]):
        tp    = conf[c, c]
        denom = conf[c, :].sum() + conf[:, c].sum() - tp
        if denom > 0:
            iou_list.append(tp / denom)
    miou = float(np.mean(iou_list)) if iou_list else 0.0
    return float(pixel_acc), miou


def evaluate(model:       nn.Module,
             loader:      DataLoader,
             criterion,
             device:      torch.device,
             num_classes: int = 2,
             train_mode:  bool = False,
             optimizer:   torch.optim.Optimizer | None = None
             ) -> tuple[float, float, float, np.ndarray]:
    """
    Single-pass loop for both training and evaluation.

    train_mode=True  : model.train(), backward pass (optimizer required).
    train_mode=False : model.eval(),  inference only (no gradients).

    Returns (avg_loss, pixel_accuracy, mIoU, confusion_matrix).
    """
    if train_mode:
        model.train()
        assert optimizer is not None, "optimizer must be provided when train_mode=True"
    else:
        model.eval()

    running_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for images, masks in tqdm(loader,
                                  desc='  train' if train_mode else '  eval ',
                                  leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            if train_mode:
                optimizer.zero_grad()

            logits = model(images)                      # [B, C, H, W]
            loss   = calculate_dice_loss(logits, masks)

            if train_mode:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)

            preds = logits.detach().argmax(dim=1)       # [B, H, W]
            all_preds.append(preds.cpu().numpy().ravel())
            all_targets.append(masks.cpu().numpy().ravel())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    conf        = compute_confusion_matrix(all_preds, all_targets, num_classes)
    pixel_acc, miou = confusion_to_metrics(conf)

    return running_loss / len(loader.dataset), pixel_acc, miou, conf

