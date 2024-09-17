import cv2
import numpy as np

# Function to amplify subtle color changes for pulse detection
def magnify_color(frame, alpha=150):
    # Convert to YUV color space
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    # Amplify the U and V channels (color information)
    yuv[:, :, 1] = cv2.add(yuv[:, :, 1], alpha)  # U channel
    yuv[:, :, 2] = cv2.add(yuv[:, :, 2], alpha)  # V channel
    
    # Convert back to BGR
    amplified_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    return amplified_frame

# Function to detect pulse-related color changes
def detect_pulse(frame_sequence, threshold=5):
    # Assuming frame_sequence is a list of frames
    color_changes = []
    
    for i in range(1, len(frame_sequence)):
        # Calculate color difference between consecutive frames
        diff = cv2.absdiff(frame_sequence[i], frame_sequence[i-1])
        mean_diff = np.mean(diff)
        color_changes.append(mean_diff)
    
    # Analyze color changes (if variance is too low, pulse might be faked)
    variance = np.var(color_changes)
    if variance < threshold:
        return False  # Fake detected
    else:
        return True  # Real pulse detected

def detect_biological_signals(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_sequence = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        amplified_frame = magnify_color(frame)
        frame_sequence.append(amplified_frame)
        
        # Display the amplified frame for debugging
        cv2.imshow("Pulse Amplification", amplified_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    pulse_detected = detect_pulse(frame_sequence)
    
    if pulse_detected:
        return "Pulse Detected!"
    else:
        return "Fake Pulse Detected!"

