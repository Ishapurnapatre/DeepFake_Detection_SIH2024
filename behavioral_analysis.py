import cv2
import subprocess
import os

# Assuming OpenFace is installed and available as a command-line tool
openface_path = "./OpenFace/build/bin/FeatureExtraction"

def detect_behavioral_inconsistencies(video_path):
    # Create a temporary directory to store OpenFace output
    temp_dir = "openface_output"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Run OpenFace to extract facial Action Units (AUs) from the video
    command = f"{openface_path} -f {video_path} -out_dir {temp_dir}"
    subprocess.run(command, shell=True)
    
    # Load the OpenFace output file (CSV format)
    output_file = os.path.join(temp_dir, "processed", f"{os.path.basename(video_path).replace('.mp4', '_au.csv')}")
    
    if not os.path.exists(output_file):
        return "Error: OpenFace processing failed."
    
    # Analyze AUs for inconsistencies (e.g., unnatural expressions)
    inconsistencies_detected = False
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the header
            au_values = line.strip().split(",")
            # Example: Check if specific AUs are behaving unnaturally
            au_intensity = list(map(float, au_values[5:]))  # AUs start from column 5
            if any(intensity < 0.1 or intensity > 5.0 for intensity in au_intensity):  # Thresholds for 'normal' AUs
                inconsistencies_detected = True
                break
    
    # Clean up
    subprocess.run(f"rm -r {temp_dir}", shell=True)

    if inconsistencies_detected:
        return "Behavioral Inconsistencies Detected!"
    else:
        return "No Behavioral Inconsistencies Detected."

