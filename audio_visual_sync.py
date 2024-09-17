import cv2
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import os
import subprocess

# Function to extract audio from the video
def extract_audio(video_path, output_audio_path):
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio_path} -y"
    subprocess.run(command, shell=True)
    return output_audio_path

# Function to detect lip movement using OpenCV
def detect_lip_movement(frame, face_cascade, lip_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    lip_detected = False

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        lips = lip_cascade.detectMultiScale(face_roi, 1.1, 4)
        if len(lips) > 0:
            lip_detected = True
            for (lx, ly, lw, lh) in lips:
                cv2.rectangle(frame, (x + lx, y + ly), (x + lx + lw, y + ly + lh), (255, 0, 0), 2)
    
    return frame, lip_detected

def detect_audio_visual_sync(video_path):
    # Load the cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    lip_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')  # Approximate lip detection
    
    # Extract audio from video
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Load the audio
    audio = AudioSegment.from_wav(audio_path)
    audio_frames = np.array(audio.get_array_of_samples())
    sample_rate, audio_data = wavfile.read(audio_path)
    
    total_lip_movements = 0
    total_audio_frames = len(audio_frames)
    sync_issue_detected = False
    lip_movement_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, lip_detected = detect_lip_movement(frame, face_cascade, lip_cascade)
        lip_movement_frames.append(lip_detected)
        
        # Display frame for debugging
        cv2.imshow("Lip Movement Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Analyze sync between lip movements and audio
    for i in range(min(len(lip_movement_frames), total_audio_frames)):
        if lip_movement_frames[i] and audio_frames[i] == 0:  # Lip movement without sound
            sync_issue_detected = True
            break
    
    os.remove(audio_path)  # Clean up temporary audio file
    
    if sync_issue_detected:
        return "Audio-Visual Sync Issues Detected!"
    else:
        return "No Audio-Visual Sync Issues Detected."

