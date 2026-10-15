import os
import re

import cv2
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")                      # draw charts into files, no window needed
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_utils import hog_features

FACES_DIR = "faces"
MODEL_FILE = "face_model.joblib"
COPY_PATTERN = re.compile(r"_[A-Za-z]+_temp$")     # e.g. "10_bright_temp"

# ---------- 1. Load the cropped faces ----------
players = sorted(p for p in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, p)))
faces, labels, groups, is_original = [], [], [], []

for label, player in enumerate(players):
    for name in sorted(os.listdir(os.path.join(FACES_DIR, player))):
        if not name.lower().endswith(".png"):
            continue
        face = cv2.imread(os.path.join(FACES_DIR, player, name), cv2.IMREAD_GRAYSCALE)
        if face is None:
            continue
        stem = os.path.splitext(name)[0]
        faces.append(face)
        labels.append(label)
        groups.append(f"{player}/{COPY_PATTERN.sub('', stem)}")   # a photo and its copies share a group
        is_original.append(COPY_PATTERN.search(stem) is None)

y = np.array(labels)
groups = np.array(groups)
is_original = np.array(is_original)

print(f"Loaded {len(faces)} faces from {len(players)} players "
      f"({is_original.sum()} original photos, {len(faces) - is_original.sum()} edited copies)\n")

# ---------- 2. Turn every face into numbers ----------
features = {kind: np.array([hog_features(f, kind) for f in faces]) for kind in ("coarse", "fine")}

# ---------- 3. Cross-validation that never lets a photo and its copies be split apart ----------
# Train on every variant of the training photos. Test only on ORIGINAL photos the model has never seen.
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(splitter.split(features["fine"], y, groups))


def make_model(svc_params, calibration_folds=None):
    """Scaler -> PCA -> SVM. With calibration_folds, wrap it so it can report probabilities."""
    base = make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"), SVC(**svc_params))
    if calibration_folds is None:
        return base
    return CalibratedClassifierCV(base, method="sigmoid", cv=calibration_folds, ensemble=False)


SVM_OPTIONS = {
    "linear C=1": dict(kernel="linear", C=1),
    "rbf C=10": dict(kernel="rbf", C=10, gamma="scale"),
    "rbf C=100": dict(kernel="rbf", C=100, gamma="scale"),
}


def cross_validate(X, svc_params):
    truth, pred = [], []
    for train_idx, test_idx in folds:
        test_idx = test_idx[is_original[test_idx]]         # test on original photos only
        model = make_model(svc_params)
        model.fit(X[train_idx], y[train_idx])
        truth += list(y[test_idx])
        pred += list(model.predict(X[test_idx]))
    return np.array(truth), np.array(pred)


print("Trying different settings (5-fold cross-validation, tested on original photos only):")
print(f"{'HOG detail':<12}{'SVM settings':<14}{'Accuracy':>9}")
best = None
for hog_kind in ("coarse", "fine"):
    for svm_name, svm_params in SVM_OPTIONS.items():
        truth, pred = cross_validate(features[hog_kind], svm_params)
        acc = accuracy_score(truth, pred)
        print(f"{hog_kind:<12}{svm_name:<14}{acc * 100:>8.1f}%")
        if best is None or acc > best["acc"]:
            best = dict(acc=acc, hog=hog_kind, svm_name=svm_name, svm=svm_params, truth=truth, pred=pred)

print(f"\nBest: HOG '{best['hog']}' + SVM {best['svm_name']}  ->  {best['acc'] * 100:.1f}% accuracy "
      f"on {len(best['truth'])} unseen original photos\n")
print(classification_report(best["truth"], best["pred"], target_names=players, zero_division=0))

# ---------- 4. Confusion matrix picture ----------
cm = confusion_matrix(best["truth"], best["pred"], labels=range(len(players)))
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=players).plot(ax=ax, cmap="Purples", xticks_rotation=30, colorbar=False)
ax.set_title(f"Confusion matrix ({best['acc'] * 100:.1f}% accuracy)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")

# ---------- 5. Train the final model on everything and save it ----------
# Probabilities are calibrated with the same photo-aware splitting, so copies never leak into calibration.
X_all = features[best["hog"]]
calibration_folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X_all, y, groups))
final = make_model(best["svm"], calibration_folds)
final.fit(X_all, y)
joblib.dump({"model": final, "classes": players, "hog": best["hog"], "cv_accuracy": best["acc"]}, MODEL_FILE)
print(f"Saved the trained model to {MODEL_FILE}")
