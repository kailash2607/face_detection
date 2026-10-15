import cv2

FACE_SIZE = 128

# Two ready-made face detectors that come with OpenCV
CASCADES = [
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
]

# Two ways of turning a face into numbers (HOG = histogram of oriented gradients).
# "coarse" looks at the overall shape of the face, "fine" looks at more detail.
HOG_SETTINGS = {
    "coarse": cv2.HOGDescriptor((FACE_SIZE, FACE_SIZE), (32, 32), (16, 16), (16, 16), 9),
    "fine": cv2.HOGDescriptor((FACE_SIZE, FACE_SIZE), (16, 16), (8, 8), (8, 8), 9),
}


def detect_faces(gray):
    """Return a list of (x, y, w, h) boxes. Strict settings first, then more forgiving ones."""
    for min_neighbors in (5, 3):
        for detector in CASCADES:
            faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=min_neighbors, minSize=(60, 60))
            if len(faces) > 0:
                return list(faces)
    return []


def crop_box(gray, box):
    """Cut one face out of a grayscale photo and standardise it (same recipe used for training)."""
    x, y, w, h = box
    margin = int(0.15 * w)                       # include a bit of forehead and chin
    x1, y1 = max(x - margin, 0), max(y - margin, 0)
    x2, y2 = min(x + w + margin, gray.shape[1]), min(y + h + margin, gray.shape[0])
    face = cv2.resize(gray[y1:y2, x1:x2], (FACE_SIZE, FACE_SIZE))
    return cv2.equalizeHist(face)                # evens out lighting differences


def crop_face(image):
    """Used when preparing training data. Return (face, reason); face is None when the photo is skipped."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:
        return None, "no face"

    areas = sorted((w * h for (_, _, w, h) in faces), reverse=True)
    if len(areas) > 1 and areas[1] > 0.5 * areas[0]:
        return None, "several faces"

    box = max(faces, key=lambda f: f[2] * f[3])
    return crop_box(gray, box), "ok"


def analyse_image(image):
    """Used by the app. Return (face, all_boxes, chosen_box); the largest face is the one classified."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:
        return None, [], None
    box = max(faces, key=lambda f: f[2] * f[3])
    return crop_box(gray, box), faces, box


def hog_features(face, kind="fine"):
    """Turn a 128x128 grayscale face into a list of numbers."""
    return HOG_SETTINGS[kind].compute(face).flatten()


def predict_probabilities(bundle, face):
    """Return [(player, probability), ...] sorted from most to least likely."""
    features = hog_features(face, bundle["hog"]).reshape(1, -1)
    probs = bundle["model"].predict_proba(features)[0]
    order = probs.argsort()[::-1]
    return [(bundle["classes"][i], float(probs[i])) for i in order]
