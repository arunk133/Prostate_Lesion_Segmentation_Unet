import os
import cv2
import numpy as np
import tensorflow as tf


class MriLesionDatasetTF(tf.keras.utils.Sequence):
    """
    TensorFlow Dataset
    EXACT match to PyTorch MriLesionDataset:
      - Same cropping (lesion-centered)
      - Same padding logic
      - Same normalization
      - Same channel order (converted to HWC)
    """

    def __init__(self, root_dir, batch_size=4, patch_size=144):
        self.root = root_dir
        self.batch = batch_size
        self.patch = patch_size

        cancer_ids = os.listdir(f"{root_dir}/images/cancer")
        non_cancer_ids = os.listdir(f"{root_dir}/images/non_cancer")

        # 2:1 cancer : non-cancer (same as PyTorch)
        non_cancer_ids = non_cancer_ids[:len(cancer_ids)//2]

        self.samples = (
            [("cancer", p) for p in cancer_ids] +
            [("non_cancer", p) for p in non_cancer_ids]
        )

    # --------------------------------------------------
    # REQUIRED by tf.keras.utils.Sequence
    # --------------------------------------------------
    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch))

    # --------------------------------------------------
    # Image loading (Z-score normalization)
    # --------------------------------------------------
    def load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-5)
        return img

    def load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return (mask > 0).astype(np.float32)

    # --------------------------------------------------
    # EXACT SAME crop logic as PyTorch
    # --------------------------------------------------
    def crop_patch(self, img, mask):
        patch = self.patch
        _, H, W = img.shape
        mask2d = mask[0]

        ys, xs = np.where(mask2d > 0)
        if len(xs) == 0:
            y, x = H // 2, W // 2
        else:
            y, x = int(ys.mean()), int(xs.mean())

        s = patch // 2
        y1, y2 = y - s, y + s
        x1, x2 = x - s, x + s

        pad_t = max(0, -y1)
        pad_l = max(0, -x1)
        pad_b = max(0, y2 - H)
        pad_r = max(0, x2 - W)

        img  = np.pad(img,  ((0,0),(pad_t,pad_b),(pad_l,pad_r)))
        mask = np.pad(mask, ((0,0),(pad_t,pad_b),(pad_l,pad_r)))

        y1 += pad_t; y2 += pad_t
        x1 += pad_l; x2 += pad_l

        return img[:, y1:y2, x1:x2], mask[:, y1:y2, x1:x2]

    # --------------------------------------------------
    # Fetch batch
    # --------------------------------------------------
    def __getitem__(self, idx):
        batch_samples = self.samples[
            idx * self.batch : (idx + 1) * self.batch
        ]

        images, masks = [], []

        for label, pid in batch_samples:
            base = f"{self.root}/images/{label}/{pid}/{pid}"

            t2w = self.load_img(base + "_t2w.png")
            hbv = self.load_img(base + "_hbv.png")
            adc = self.load_img(base + "_adc.png")

            img = np.stack([t2w, hbv, adc], axis=0)

            mask = self.load_mask(
                f"{self.root}/masks/{label}/{pid}.png"
            )[None]

            img, mask = self.crop_patch(img, mask)

            # CHW → HWC (TensorFlow format)
            images.append(np.transpose(img, (1, 2, 0)))
            masks.append(np.transpose(mask, (1, 2, 0)))

        return np.array(images, dtype=np.float32), \
               np.array(masks, dtype=np.float32)

