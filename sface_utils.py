import os

import cv2
import numpy as np

# Two small pre-trained neural networks from the OpenCV model zoo:
#   YuNet = finds faces (and 5 landmarks: eyes, nose, mouth corners)
#   SFace = turns an aligned face into 128 numbers (an "embedding") that describe WHO the face is
YUNET_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join("models", "face_recognition_sface_2021dec.onnx")
MAX_SIDE = 1280          # very large photos are shrunk to this size before detection

_detector = None
_recognizer = None


def _load_models():
    global _detector, _recognizer
    if _detector is None:
        for path in (YUNET_PATH, SFACE_PATH):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model file not found: {path}. Put both .onnx files in the 'models' folder.")
        _detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.7, 0.3, 5000)
        _recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    return _detector, _recognizer


def analyse_image_sface(image):
    """Return (embedding, all_boxes, chosen_box, aligned_face). Everything is None/[] when no face is found.
    Boxes are (x, y, w, h) in the coordinates of the ORIGINAL image. The largest face is the one used."""
    detector, recognizer = _load_models()

    h, w = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1.0 else image

    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        return None, [], None, None

    best = max(faces, key=lambda f: f[2] * f[3])
    aligned = recognizer.alignCrop(small, best)                 # straightens and crops the face to 112x112
    embedding = recognizer.feature(aligned).flatten()
    embedding = embedding / (np.linalg.norm(embedding) + 1e-9)  # length 1, so similarity is a simple dot product

    boxes = [tuple(int(v / scale) for v in f[:4]) for f in faces]
    chosen = tuple(int(v / scale) for v in best[:4])
    return embedding, boxes, chosen, aligned
