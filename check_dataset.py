import os

import cv2

DATASET_DIR = "dataset"
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MIN_USABLE = 20   # photos with exactly one clear face needed per player

# OpenCV ships with a ready-made face detector, nothing to download
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def count_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return len(faces)


if not os.path.isdir(DATASET_DIR):
    raise SystemExit(f"Folder '{DATASET_DIR}' not found. Create it next to this script and add one folder per player.")

players = sorted(p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p)))
if not players:
    raise SystemExit("No player folders found inside 'dataset'.")

print(f"{'Player':<22}{'Images':>7}{'Unreadable':>12}{'No face':>9}{'Multi-face':>12}{'Usable':>8}   Status")
print("-" * 80)

all_ready = True
for player in players:
    folder = os.path.join(DATASET_DIR, player)
    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_TYPES)]
    unreadable, no_face, multi, usable = [], [], [], 0

    for name in files:
        image = cv2.imread(os.path.join(folder, name))
        if image is None:
            unreadable.append(name)
            continue
        n = count_faces(image)
        if n == 0:
            no_face.append(name)
        elif n == 1:
            usable += 1
        else:
            multi.append(name)

    ready = usable >= MIN_USABLE
    all_ready = all_ready and ready
    status = "OK" if ready else f"need {MIN_USABLE - usable} more good photos"
    print(f"{player:<22}{len(files):>7}{len(unreadable):>12}{len(no_face):>9}{len(multi):>12}{usable:>8}   {status}")

    # list a few problem files so you can delete or replace them
    for label, items in (("unreadable", unreadable), ("no face found", no_face), ("several faces", multi)):
        if items:
            shown = ", ".join(items[:6]) + (" ..." if len(items) > 6 else "")
            print(f"{'':<24}{label}: {shown}")

print("-" * 80)
print("Dataset is ready for training!" if all_ready else
      "Some players need more good photos. Delete the problem files above and add clearer ones.")
