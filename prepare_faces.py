import os
import re

import cv2
import numpy as np

DATASET_DIR = "dataset"      # your original photos, one folder per player
FACES_DIR = "faces"          # cropped faces will be saved here
FACE_SIZE = 128              # every face is resized to 128 x 128 pixels
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# This dataset already contains edited copies of many photos, named like
# "10_bright_temp.jpg", "10_flipped_temp.jpg", "10_dark_temp.jpg", "10_rotated_temp.jpg".
# We keep the file names so training can tell originals from copies of the same photo.
COPY_PATTERN = re.compile(r"_[A-Za-z]+_temp$")

# Two ready-made detectors that come with OpenCV. We try the first one, then the second.
CASCADES = [
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
]


def is_original(stem):
    return COPY_PATTERN.search(stem) is None


def detect_faces(gray):
    """Return a list of (x, y, w, h) boxes. Strict settings first, then more forgiving ones."""
    for min_neighbors in (5, 3):
        for detector in CASCADES:
            faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=min_neighbors, minSize=(60, 60))
            if len(faces) > 0:
                return list(faces)
    return []


def crop_face(image):
    """Return (face, reason). face is None when the photo should be skipped."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:
        return None, "no face"

    areas = sorted((w * h for (_, _, w, h) in faces), reverse=True)
    if len(areas) > 1 and areas[1] > 0.5 * areas[0]:
        return None, "several faces"         # two similar-sized faces: we cannot tell who is who

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    margin = int(0.15 * w)                   # include a bit of forehead and chin
    x1, y1 = max(x - margin, 0), max(y - margin, 0)
    x2, y2 = min(x + w + margin, gray.shape[1]), min(y + h + margin, gray.shape[0])
    face = cv2.resize(gray[y1:y2, x1:x2], (FACE_SIZE, FACE_SIZE))
    return cv2.equalizeHist(face), "ok"      # equalizeHist evens out lighting differences


def make_preview(originals):
    """One row per player with up to 10 different ORIGINAL photos, so you can check them by eye."""
    rows = []
    for player, faces in originals.items():
        if len(faces) > 10:
            picks = [faces[int(i * len(faces) / 10)] for i in range(10)]   # spread across the list
        else:
            picks = faces
        row = [cv2.resize(f, (80, 80)) for f in picks]
        row += [np.zeros((80, 80), np.uint8)] * (10 - len(row))
        strip = np.hstack(row)
        label = np.zeros((26, strip.shape[1]), np.uint8)
        cv2.putText(label, player, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        rows += [label, strip]
    cv2.imwrite("faces_preview.png", np.vstack(rows))


if not os.path.isdir(DATASET_DIR):
    raise SystemExit(f"Folder '{DATASET_DIR}' not found.")

players = sorted(p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p)))
originals_for_preview = {}

print(f"{'Player':<20}{'Photos':>7}{'Saved':>7}{'Originals':>11}{'Copies':>8}{'No face':>9}{'Several':>9}")
print("-" * 71)
for player in players:
    src = os.path.join(DATASET_DIR, player)
    dst = os.path.join(FACES_DIR, player)
    os.makedirs(dst, exist_ok=True)
    for old in os.listdir(dst):                  # start fresh so old results never mix in
        os.remove(os.path.join(dst, old))

    files = [f for f in sorted(os.listdir(src)) if f.lower().endswith(IMAGE_TYPES)]
    saved_originals, saved_copies, no_face, several = [], 0, 0, 0
    used_names = set()

    for name in files:
        image = cv2.imread(os.path.join(src, name))
        if image is None:
            continue
        face, reason = crop_face(image)
        if face is None:
            if reason == "no face":
                no_face += 1
            else:
                several += 1
            continue

        stem = os.path.splitext(name)[0]
        out_name, k = stem, 1
        while out_name in used_names:            # two files with the same name but different extension
            out_name, k = f"{stem}-{k}", k + 1
        used_names.add(out_name)
        cv2.imwrite(os.path.join(dst, out_name + ".png"), face)

        if is_original(stem):
            saved_originals.append(face)
        else:
            saved_copies += 1

    originals_for_preview[player] = saved_originals
    total = len(saved_originals) + saved_copies
    print(f"{player:<20}{len(files):>7}{total:>7}{len(saved_originals):>11}{saved_copies:>8}{no_face:>9}{several:>9}")

print("-" * 71)
make_preview({p: f for p, f in originals_for_preview.items() if f})
print("'Originals' = different photos. 'Copies' = edited versions (flipped, bright, dark, rotated) of those photos.")
print("Open 'faces_preview.png' and check that every row shows the right person's face.")
