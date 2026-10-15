import os

import cv2
import joblib

from face_utils import analyse_image, predict_probabilities

# Put NEW photos (not from your training dataset) in folders named exactly like the players:
#   test_photos/ms_dhoni/1.jpg, test_photos/virat_kohli/1.jpg, ...
TEST_DIR = "test_photos"
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

bundle = joblib.load("face_model.joblib")
classes = bundle["classes"]

if not os.path.isdir(TEST_DIR):
    raise SystemExit(f"Folder '{TEST_DIR}' not found. Create it and add one folder per player, e.g. {TEST_DIR}/{classes[0]}/")

stats = {c: {"tested": 0, "correct": 0, "no_face": 0} for c in classes}
mistakes = []

print(f"{'Photo':<40}{'Guess':<20}{'Conf':>6}  Result")
print("-" * 78)
for player in sorted(os.listdir(TEST_DIR)):
    folder = os.path.join(TEST_DIR, player)
    if not os.path.isdir(folder):
        continue
    if player not in classes:
        print(f"Skipping folder '{player}': it must be named exactly like one of {classes}")
        continue

    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMAGE_TYPES):
            continue
        image = cv2.imread(os.path.join(folder, name))
        if image is None:
            continue
        label = f"{player}/{name}"[:38]

        face, _, _ = analyse_image(image)
        if face is None:
            stats[player]["no_face"] += 1
            print(f"{label:<40}{'-':<20}{'-':>6}  no face found")
            continue

        guess, prob = predict_probabilities(bundle, face)[0]
        stats[player]["tested"] += 1
        ok = guess == player
        stats[player]["correct"] += ok
        if not ok:
            mistakes.append((player, guess, prob))
        print(f"{label:<40}{guess:<20}{prob * 100:>5.0f}%  {'correct' if ok else 'WRONG'}")

print("-" * 78)
print(f"{'Player':<20}{'Tested':>8}{'Correct':>9}{'No face':>9}{'Accuracy':>10}")
total_tested = total_correct = total_no_face = 0
for c in classes:
    s = stats[c]
    acc = f"{s['correct'] / s['tested'] * 100:.0f}%" if s["tested"] else "-"
    print(f"{c:<20}{s['tested']:>8}{s['correct']:>9}{s['no_face']:>9}{acc:>10}")
    total_tested += s["tested"]
    total_correct += s["correct"]
    total_no_face += s["no_face"]

print("-" * 78)
if total_tested:
    print(f"Overall: {total_correct} of {total_tested} photos correct = {total_correct / total_tested * 100:.0f}% "
          f"(plus {total_no_face} photos where no face was found)")
else:
    print("No photos with a detectable face were tested.")
for true, guess, prob in mistakes:
    print(f"  mistake: {true} was predicted as {guess} ({prob * 100:.0f}%)")
