import os

import cv2
import numpy as np

from sface_utils import SFACE_PATH, YUNET_PATH, analyse_image_sface

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

print("1. Model files")
for path, expected in ((YUNET_PATH, "about 0.2 MB"), (SFACE_PATH, "about 37 MB")):
    if os.path.isfile(path):
        print(f"   {path}: {os.path.getsize(path) / 1e6:.2f} MB (should be {expected})")
    else:
        print(f"   {path}: NOT FOUND")
        raise SystemExit("Put both downloaded .onnx files into the 'models' folder and run this again.")


def embeddings_from(folder, how_many):
    """Embed the first photos in a folder that contain a detectable face."""
    found = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMAGE_TYPES):
            continue
        image = cv2.imread(os.path.join(folder, name))
        if image is None:
            continue
        embedding, boxes, _, _ = analyse_image_sface(image)
        if embedding is not None:
            found.append((name, embedding, len(boxes)))
            if len(found) == how_many:
                break
    return found


print("\n2. Embedding test")
root = "test_photos" if os.path.isdir("test_photos") else "dataset"
folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
if len(folders) < 2:
    raise SystemExit("Need at least two player folders to run this test.")

person_a = embeddings_from(folders[0], 2)
person_b = embeddings_from(folders[1], 1)
if len(person_a) < 2 or len(person_b) < 1:
    raise SystemExit("Could not find enough photos with a detectable face in the first two folders.")

for folder, items in ((folders[0], person_a), (folders[1], person_b)):
    for name, embedding, n_faces in items:
        print(f"   {os.path.basename(folder)}/{name}: embedding of {embedding.shape[0]} numbers, {n_faces} face(s) found")

same = float(np.dot(person_a[0][1], person_a[1][1]))
different = float(np.dot(person_a[0][1], person_b[0][1]))
print(f"\n   similarity of two photos of {os.path.basename(folders[0])}: {same:.2f}")
print(f"   similarity of {os.path.basename(folders[0])} vs {os.path.basename(folders[1])}: {different:.2f}")
print("\nIf the first number is higher than the second, the models work. Next: python build_embeddings.py")
