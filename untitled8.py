from keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model for image classification
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = [name.strip() for name in open("labels.txt", "r", encoding="utf-8").readlines()]

# Load MediaPipe Selfie Segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

def display_other_classes(predictions, threshold=0.01):
    """Displays classes with prediction confidence above a certain threshold."""
    for i, score in enumerate(predictions.flatten()):
        if score > threshold:
            print(f"Class: {class_names[i]}, Confidence Score: {np.round(score * 100, 2)}%")

while True:
    # Grab the webcam's image.
    ret, image = camera.read()

    # Flip the image horizontally
    image = cv2.flip(image, 1)

    # Convert image color space from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image using MediaPipe Selfie Segmentation
    results = selfie_segmentation.process(image_rgb)

    # Get segmentation mask
    segmentation_mask = results.segmentation_mask

    # Convert mask to binary image
    segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255

    # Create a background image (pure black background)
    background = np.zeros_like(image)

    # Combine foreground (person) and background
    output_image = np.where(segmentation_mask[:, :, None], image, background)

    # Make the image a numpy array and reshape it to the model's input shape
    image_for_classification = cv2.resize(output_image, (224, 224), interpolation=cv2.INTER_AREA)
    image_for_classification = np.asarray(image_for_classification, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_for_classification = (image_for_classification / 127.5) - 1

    # Predict using the classification model
    prediction = model.predict(image_for_classification)

    # Get the indices of the top two predictions
    top_indices = prediction.flatten().argsort()[-2:][::-1]

    # Extract top two predictions and their confidence scores
    top_predictions = [(class_names[i], prediction.flatten()[i]) for i in top_indices]

    # Prepare text for the top two predictions
    top_prediction_text = f"1st: {top_predictions[0][0]} - {np.round(top_predictions[0][1] * 100, 2)}%"
    second_prediction_text = f"2nd: {top_predictions[1][0]} - {np.round(top_predictions[1][1] * 100, 2)}%"

    # Draw the text on the image
    cv2.putText(output_image, top_prediction_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output_image, second_prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the image with predictions in a window
    cv2.imshow("Webcam Image with Predictions", output_image)

    # Listen for keyboard presses
    keyboard_input = cv2.waitKey(1)

    # If 'q' key is pressed, break from the loop
    if keyboard_input == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

