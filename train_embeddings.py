import os

import cv2
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from sface_utils import analyse_image_sface

TEST_DIR = "test_photos"
MODEL_FILE = "face_model_sface.joblib"
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# After measuring accuracy on the test photos, also learn from them in the FINAL model
# (more variety = better model). The accuracy printed below is measured BEFORE this happens.
INCLUDE_TEST_PHOTOS_IN_FINAL = True


def new_classifier():
    return LogisticRegression(C=10, max_iter=3000)


data = np.load("embeddings.npz")
X, y, groups, is_original = data["X"], data["y"], data["groups"], data["is_original"]
classes = [str(c) for c in data["classes"]]
print(f"Loaded {len(X)} training embeddings for {len(classes)} players\n")

# ---------- 1. Cross-validation on the dataset (same method as before) ----------
folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups)
truth, pred = [], []
for train_idx, test_idx in folds:
    test_idx = test_idx[is_original[test_idx]]
    model = new_classifier().fit(X[train_idx], y[train_idx])
    truth += list(y[test_idx])
    pred += list(model.predict(X[test_idx]))
cv_accuracy = accuracy_score(truth, pred)
print(f"1) Cross-validation on dataset photos: {cv_accuracy * 100:.1f}% ({len(truth)} unseen original photos)")

# ---------- 2. The real test: new photos from the internet ----------
X_ext, y_ext, ext_files = [], [], []
no_face = 0
if os.path.isdir(TEST_DIR):
    for label, player in enumerate(classes):
        folder = os.path.join(TEST_DIR, player)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(IMAGE_TYPES):
                continue
            image = cv2.imread(os.path.join(folder, name))
            if image is None:
                continue
            embedding, _, _, _ = analyse_image_sface(image)
            if embedding is None:
                no_face += 1
                continue
            X_ext.append(embedding)
            y_ext.append(label)
            ext_files.append(f"{player}/{name}")

external_accuracy = None
if X_ext:
    X_ext, y_ext = np.array(X_ext), np.array(y_ext)
    dataset_only = new_classifier().fit(X, y)
    ext_pred = dataset_only.predict(X_ext)
    external_accuracy = accuracy_score(y_ext, ext_pred)
    print(f"\n2) New photos the model has never seen: {(ext_pred == y_ext).sum()} of {len(y_ext)} correct "
          f"= {external_accuracy * 100:.1f}%  ({no_face} photos had no detectable face)\n")
    print(f"{'Player':<20}{'Tested':>8}{'Correct':>9}{'Accuracy':>10}")
    for i, c in enumerate(classes):
        mask = y_ext == i
        if mask.any():
            print(f"{c:<20}{mask.sum():>8}{(ext_pred[mask] == i).sum():>9}{(ext_pred[mask] == i).mean() * 100:>9.0f}%")

    cm = confusion_matrix(y_ext, ext_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Purples", xticks_rotation=30, colorbar=False)
    ax.set_title(f"Confusion matrix on new photos ({external_accuracy * 100:.1f}% accuracy)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_external.png", dpi=150)
    print("\nSaved confusion_matrix_external.png")
else:
    print("\n2) No test photos found in 'test_photos', so only the cross-validation number is available.")

# ---------- 3. Final model ----------
X_final, y_final = X, y
if INCLUDE_TEST_PHOTOS_IN_FINAL and external_accuracy is not None:
    X_final = np.vstack([X, X_ext])
    y_final = np.concatenate([y, y_ext])
final = new_classifier().fit(X_final, y_final)
joblib.dump({"model": final, "classes": classes, "kind": "sface", "cv_accuracy": cv_accuracy,
             "external_accuracy": external_accuracy, "n_external": int(len(y_ext)) if external_accuracy is not None else 0},
            MODEL_FILE)
print(f"\nSaved the final model to {MODEL_FILE} (trained on {len(X_final)} faces)")
