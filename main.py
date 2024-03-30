# main.py
from model import HOGSVMClassifier
import cv2
from skimage.feature import hog


def classify_webcam(classifier):
    # Open the webcam
    cap = cv2.VideoCapture(1)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Resize the frame to a fixed size (e.g., 64x64)
        resized_frame = cv2.resize(frame, (64, 64))

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Extract HOG features
        hog_features_frame = hog(gray_frame, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features_frame = hog_features_frame.reshape(1, -1)

        # Normalize HOG features using the same scaler used during training
        hog_features_frame_normalized = classifier.get_scaler().transform(hog_features_frame)

        # Make the prediction
        prediction = classifier.get_classifier().predict(hog_features_frame_normalized)

        # Display the frame with the classification result
        if prediction == 1:
            cv2.putText(frame, "Vertical", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Horizontal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('HOG SVM Classifier', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_path = r'C:\Users\Home\Downloads\image pre processing\HOG & SVM\dataset'

    classifier = HOGSVMClassifier(dataset_path)

    # Step 1: Load the Dataset
    X, y = classifier.load_dataset()

    # Step 2: Extract HOG Features
    X_hog = classifier.extract_hog_features(X)

    # Step 3: Split the Dataset
    X_train, X_test, y_train, y_test = classifier.split_dataset(X_hog, y)

    # Step 4: Normalize HOG Features
    X_train_normalized, X_test_normalized = classifier.normalize_hog_features(X_train, X_test)

    # Step 5: Train the SVM Model
    classifier.train_svm_model(X_train_normalized, y_train)

    # Step 6: Save the Model
    classifier.save_model()

    # Step 7: Classify Webcam
    classify_webcam(classifier)
