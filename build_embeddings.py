import os
import re

import cv2
import numpy as np

from sface_utils import analyse_image_sface

DATASET_DIR = "dataset"
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
COPY_PATTERN = re.compile(r"_[A-Za-z]+_temp$")      # edited copies like "10_bright_temp"

players = sorted(p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p)))
X, y, groups, is_original = [], [], [], []

print(f"{'Player':<20}{'Photos':>8}{'Faces found':>13}{'Originals':>11}{'No face':>9}")
print("-" * 61)
for label, player in enumerate(players):
    folder = os.path.join(DATASET_DIR, player)
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_TYPES)]
    found = originals = missing = 0

    for name in files:
        image = cv2.imread(os.path.join(folder, name))
        if image is None:
            missing += 1
            continue
        embedding, _, _, _ = analyse_image_sface(image)
        if embedding is None:
            missing += 1
            continue

        stem = os.path.splitext(name)[0]
        X.append(embedding)
        y.append(label)
        groups.append(f"{player}/{COPY_PATTERN.sub('', stem)}")   # a photo and its copies share a group
        original = COPY_PATTERN.search(stem) is None
        is_original.append(original)
        found += 1
        originals += original

    print(f"{player:<20}{len(files):>8}{found:>13}{originals:>11}{missing:>9}")

np.savez("embeddings.npz", X=np.array(X), y=np.array(y), groups=np.array(groups),
         is_original=np.array(is_original), classes=np.array(players))
print("-" * 61)
print(f"Saved {len(X)} embeddings to embeddings.npz. Next: python train_embeddings.py")
