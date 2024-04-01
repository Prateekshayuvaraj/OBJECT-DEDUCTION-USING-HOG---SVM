# HOG & SVM Classifier for Image Orientation Detection

📷 This project utilizes Histogram of Oriented Gradients (HOG) feature extraction and Support Vector Machine (SVM) classification to detect whether an image captured from a webcam is horizontally or vertically oriented.

## Features
- Utilizes HOG feature extraction to capture shape and edge information from images.
- Trains an SVM classifier to distinguish between horizontal and vertical orientations.
- Real-time classification of webcam images.

## How to Use
1. **Dataset Preparation**: Organize your dataset with images of both horizontal and vertical orientations.
2. **Training**: Run `main.py` to train the SVM model using the provided dataset.
3. **Classification**: After training, the webcam classifier can be initiated using `classify_webcam()` function in `main.py`.
4. **Interpretation**: The webcam feed will display the orientation classification in real-time.

## Dependencies
- Python 3.x
- OpenCV
- scikit-image
- scikit-learn
- 
## Acknowledgements
- Inspired by the concepts of HOG and SVM for image classification.
