import facial_artifacts
import behavioral_analysis
import audio_visual_sync
import biological_signals
import deep_learning_classifier

def analyze_video(video_path):
    report = {}

    # Run each technique and collect feedback
    report['Facial Artifacts'] = facial_artifacts.report_facial_artifacts(video_path)
    report['Behavioral Analysis'] = behavioral_analysis.detect_behavioral_inconsistencies(video_path)
    report['Audio-Visual Sync'] = audio_visual_sync.detect_audio_visual_sync(video_path)
    report['Biological Signals'] = biological_signals.detect_biological_signals(video_path)
    report['Deep Learning Classifier'] = deep_learning_classifier.classify_frames(video_path)

    # Print the final report
    print("Deepfake Detection Report:")
    for technique, result in report.items():
        print(f"{technique}: {result}")
    
    return report

if __name__ == "__main__":
    video_path = 'recording.mp4'  # Example input video file
    analyze_video(video_path)
