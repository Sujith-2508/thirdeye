import cv2
import numpy as np
from face_detector import FaceDetector
from emotion_recognition import EmotionRecognizer
from engagement_analyzer import EngagementAnalyzer

def main():
    # Initialize components
    face_detector = FaceDetector()
    emotion_recognizer = EmotionRecognizer()
    engagement_analyzer = EngagementAnalyzer()
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    print("ThirdEye system started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Enlarge the frame (e.g., 1.5x)
        scale = 1.5
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        frame_height, frame_width = frame.shape[:2]

        # Detect faces
        faces = face_detector.detect_faces(frame)

        # Default values for display
        emotion = "Unknown"
        confidence = 0.0
        engagement_status = "Not Engaged"
        eyes_visible = False

        # If a face is detected, process the first face
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]
            emotion, confidence = emotion_recognizer.predict_emotion(face_img)
            engagement_score = engagement_analyzer.calculate_engagement(emotion, confidence, face_img)
            engagement_status = engagement_analyzer.get_engagement_status()
            eyes_visible = engagement_analyzer.detect_eyes(face_img)

        # Show warning at the top if eyes are not visible
        if not eyes_visible:
            warning_text = "Warning: Participant engagement is low"
            cv2.putText(frame, warning_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Prepare info for bottom left
        info_lines = [
            f"Participant is {emotion}",
            f"{'Engaged' if engagement_status.lower() == 'engaged' else 'Not Engaged'}",
            f"Eyes: {'Visible' if eyes_visible else 'Not Visible'}"
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (30, frame_height - 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if (i != 2 or eyes_visible) else (0, 0, 255), 2)

        # Show the frame with new tagline
        cv2.imshow('Third Eye - Transforming Presence into Participant', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()