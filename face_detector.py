import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        """
        Initialize the face detector with Haar Cascade classifier
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, frame):
        """
        Detect faces in the given frame
        Args:
            frame: numpy array containing the image
        Returns:
            list of face coordinates (x, y, w, h)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces 