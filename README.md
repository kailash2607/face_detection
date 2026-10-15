# 🏏 Indian Cricket Team Member Face Identification

A machine learning and computer vision system that identifies a cricketer from a photo. It is trained on images of five Indian cricketers and classifies a new input image into the matching player's name, or reports "Unknown" when it is not confident.

## Problem Statement

Develop a machine learning/computer vision system that can identify a person from an image based on their facial features. The system should be trained using images of selected Indian cricket team members and classify a new input image into the corresponding person's name.

## Players

MS Dhoni · Virat Kohli · Rohit Sharma · Sachin Tendulkar · Ravindra Jadeja

## Dataset

- **Training data:** the *Celebrity Database (332 Names)* dataset by mhmaslam on Kaggle, which has one folder of images per celebrity. Five cricketers were selected, with up to 80 images each.
- The dataset already contains edited copies of many photos (flipped, brightened, darkened, rotated). After face detection this gave **390 faces: 98 distinct original photos and 292 edited copies**. Distinct photos per player: Dhoni 20, Jadeja 14, Rohit Sharma 28, Tendulkar 20, Kohli 16.
- **Independent test data:** **66 new photos collected from web image searches** (Dhoni 8, Jadeja 12, Rohit Sharma 14, Tendulkar 16, Kohli 16). They were kept in a separate folder and were not used for training when the accuracy was measured.

## Method

1. **Face detection and alignment:** YuNet, a small pre-trained neural network from the OpenCV Model Zoo, finds faces and facial landmarks (eyes, nose, mouth). The largest face is used, and it is straightened and cropped to 112×112.
2. **Face embedding:** SFace, a pre-trained face-recognition network from the OpenCV Model Zoo, converts the aligned face into a vector of 128 numbers describing the identity of the face. Vectors are scaled to length 1.
3. **Classification:** a logistic regression classifier is trained on the embeddings of the five players and outputs a probability for each player.
4. **Unknown handling:** if the highest probability is below a threshold (adjustable in the app), the app answers "Unknown" instead of guessing.

## Experiments and Results

### Baseline: classical features (HOG + PCA + SVM)

The first version detected faces with OpenCV Haar cascades, described each face with HOG features and classified with PCA and an SVM.

| Test | Result |
|---|---|
| Cross-validation on dataset photos | 90.2% (92 unseen original photos) |
| **66 new photos from the web** (measured on 62 faces that were detected) | **39% (24 of 62)** |

Cross-validation kept every photo and its edited copies on the same side of the split, so copies could not leak into the test set. Even so, the dataset photos share sources and styles, so the 90% did not carry over to new photos. This showed that testing on independent photos is essential. HOG features capture lighting, pose and background as much as the face, and with only 14-28 distinct photos per player the model did not generalise.

### Final model: pre-trained face embeddings (YuNet + SFace + logistic regression)

| Test | Result |
|---|---|
| Cross-validation on dataset photos | 100% (98 unseen original photos) |
| **66 new photos from the web** | **100% (66 of 66 correct)** |

| Player | New photos tested | Correct |
|---|---|---|
| MS Dhoni | 8 | 8 |
| Ravindra Jadeja | 12 | 12 |
| Rohit Sharma | 14 | 14 |
| Sachin Tendulkar | 16 | 16 |
| Virat Kohli | 16 | 16 |

The new detector also found a face in every one of the 66 photos, while the Haar cascade had missed 4 of them.

_Add `confusion_matrix_external.png` here._

**How to read these numbers:** 66 of 66 correct on independent photos is a strong result, but the test set is small, so it is consistent with a true accuracy of roughly 95% or higher rather than exactly 100%. The web photos may also share sources with the Kaggle images, which would make the test a little easier than real use. The final saved model was trained on the dataset photos **plus** the 66 test photos (456 faces) after the accuracy had been measured.

## Features of the App

- Upload a JPG, PNG or WEBP photo
- Shows the photo with a box around the classified face and the aligned face the model sees
- Predicted player with a confidence ring and the probability for every player
- Adjustable minimum-confidence slider for the "Unknown" result
- Friendly messages for photos with no detectable face or several faces (the largest face is classified)

## Tech Stack

Python 3 · OpenCV 4 (YuNet and SFace models) · scikit-learn · NumPy · Matplotlib · Streamlit

## Project Structure

```
face_detection/
├── dataset/                  # training photos, one folder per player
├── test_photos/              # independent test photos, one folder per player
├── models/                   # YuNet and SFace .onnx files (from the OpenCV Model Zoo)
├── check_sface.py            # checks that the two networks load and work
├── build_embeddings.py       # detects faces and computes embeddings for the dataset
├── train_embeddings.py       # cross-validation, test on new photos, final model
├── sface_utils.py            # detection and embedding functions
├── app.py                    # Streamlit web app
├── face_model_sface.joblib   # final trained model
├── confusion_matrix_external.png
│
├── check_dataset.py          # baseline (HOG) pipeline
├── prepare_faces.py
├── face_utils.py
├── train_model.py
├── test_folder.py
├── face_model.joblib
└── README.md
```

## How to Run

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install "opencv-python<5" scikit-learn numpy pillow matplotlib streamlit
```

Download the two model files from the OpenCV Model Zoo (github.com/opencv/opencv_zoo) into the `models` folder: `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`.

```bash
python check_sface.py
python build_embeddings.py
python train_embeddings.py
python -m streamlit run app.py
```

Use OpenCV 4.x, because version 5 does not include the Haar cascade detector used by the baseline code. If `face_model_sface.joblib` is present, the app uses it; otherwise it falls back to the baseline HOG model.

## Screenshots

_Add 3 screenshots: a correct identification, an "Unknown" result for a photo of someone else, and the "No face found" message._

## Limitations and Future Work

- The system only knows the five players it was trained on (a closed-set classifier). Anyone else gets the closest match, or "Unknown" if the confidence is low. How reliably it rejects people outside the five has not been measured.
- The test set of 66 photos is small, and all test photos came from web image searches.
- Extreme angles, sunglasses, heavy occlusion and very low-resolution faces can still cause errors.
- Possible improvements: more players and more photos, a dedicated "Unknown" class trained on other faces, and evaluation on photos from completely different sources.

## Dataset and Model Credit

Training images come from the "Celebrity Database (332 Names)" dataset on Kaggle by mhmaslam. The face detector (YuNet) and face recognition network (SFace) come from the OpenCV Model Zoo. Images are used only for an academic project and should not be redistributed.
