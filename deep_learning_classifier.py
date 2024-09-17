import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# Load the pre-trained deep learning model (e.g., Xception or EfficientNet)
model = load_model('deepfake_classifier_model.h5')

# Function to preprocess frame for the model
def preprocess_frame(frame, img_size=(299, 299)):
    # Resize the frame to match the input size of the model
    frame_resized = cv2.resize(frame, img_size)
    
    # Convert frame to array and preprocess it
    frame_array = np.array(frame_resized)
    frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension
    frame_preprocessed = preprocess_input(frame_array)  # Preprocess for Xception
    return frame_preprocessed

# Function to make predictions on video frames
def classify_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fake_probabilities = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        frame_preprocessed = preprocess_frame(frame)
        
        # Predict using the model
        prediction = model.predict(frame_preprocessed)
        fake_probabilities.append(prediction[0][0])  # Assuming output is probability of being fake
        
        # Display frame for debugging
        cv2.imshow('Deep Learning Classification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate average fake probability across frames
    avg_fake_prob = np.mean(fake_probabilities)
    
    if avg_fake_prob > 0.5:
        return "Deepfake Detected!"
    else:
        return "No Deepfake Detected."

