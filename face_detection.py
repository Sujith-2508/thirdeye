import cv2
import numpy as np
import os
import urllib.request

class FaceDetector:
    def __init__(self):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Load the Haar Cascade classifier
        cascade_path = os.path.join('data', 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            print("Downloading Haar Cascade classifier...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            try:
                urllib.request.urlretrieve(url, cascade_path)
                print("Download complete!")
            except Exception as e:
                print(f"Error downloading classifier: {e}")
                raise
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise Exception("Error: Could not load Haar Cascade classifier")
    
    def detect_faces(self, frame):
        """
        Detect faces in the given frame
        Args:
            frame: numpy array containing the image
        Returns:
            list of tuples containing (x, y, w, h) for each detected face
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
    
    def draw_faces(self, frame, faces):
        """
        Draw rectangles around detected faces
        Args:
            frame: numpy array containing the image
            faces: list of tuples containing (x, y, w, h) for each detected face
        Returns:
            frame with rectangles drawn around faces
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return frame
    
    def extract_face(self, frame, face_coords):
        """
        Extract face region from the frame
        Args:
            frame: numpy array containing the image
            face_coords: tuple containing (x, y, w, h) for the face
        Returns:
            cropped face image
        """
        x, y, w, h = face_coords
        face = frame[y:y+h, x:x+w]
        return face 