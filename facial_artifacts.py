import cv2
import dlib

# Load the pre-trained facial landmark detector
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def detect_facial_artifacts(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = detector(gray, 1)
    
    for face in faces:
        # Get the landmarks
        landmarks = predictor(gray, face)
        
        # Analyze landmarks (e.g., check symmetry, alignment)
        for i in range(1, 68):  # 68 landmarks
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
    # Return analysis results (True means no artifacts, False means detected issues)
    return frame, len(faces) > 0  # Return processed frame and detection status

def report_facial_artifacts(video_path):
    cap = cv2.VideoCapture(video_path)
    artifact_detected = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, detected = detect_facial_artifacts(frame)
        if not detected:
            artifact_detected = True
        # Optional: Display frame for debugging
        cv2.imshow('Facial Artifacts Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if artifact_detected:
        return "Facial Artifacts Detected!"
    else:
        return "No Facial Artifacts Detected."

