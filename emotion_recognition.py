import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2

class EmotionRecognizer:
    def __init__(self):
        self.emotions = ['angry', 'happy', 'neutral', 'sad', 'surprised']
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_face(self, face_img):
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Resize to 48x48
        resized = cv2.resize(gray, (48, 48))
        # Normalize pixel values
        normalized = resized / 255.0
        # Reshape for model input
        reshaped = normalized.reshape(1, 48, 48, 1)
        return reshaped
    
    def predict_emotion(self, face_img):

        processed_face = self.preprocess_face(face_img)
        predictions = self.model.predict(processed_face)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        return self.emotions[emotion_idx], confidence
    
    def load_weights(self, weights_path):

        self.model.load_weights(weights_path) 