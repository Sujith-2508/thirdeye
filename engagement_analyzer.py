import numpy as np
from collections import deque
import time
import cv2
from datetime import datetime, timedelta

class EngagementAnalyzer:
    def __init__(self, window_size=30):
        """
        Initialize the engagement analyzer
        Args:
            window_size: size of the sliding window for emotion tracking (in seconds)
        """
        self.window_size = window_size
        self.emotion_history = deque(maxlen=window_size)
        self.engagement_scores = deque(maxlen=window_size)
        self.last_update = time.time()
        
        # Define emotion weights for engagement calculation
        self.emotion_weights = {
            'happy': 1.0,
            'surprise': 0.8,
            'neutral': 0.5,
            'sad': 0.3,
            'angry': 0.2,
            'fear': 0.2,
            'disgust': 0.1
        }
        
        self.engagement_threshold = 0.6
        self.attention_window = timedelta(minutes=5)
        self.engagement_history = []
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def update(self, emotion, confidence):
        """
        Update the engagement analysis with new emotion data
        Args:
            emotion: detected emotion
            confidence: confidence score of the emotion detection
        """
        current_time = time.time()
        if current_time - self.last_update >= 1.0:  # Update every second
            self.emotion_history.append((emotion, confidence))
            self.last_update = current_time
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score()
            self.engagement_scores.append(engagement_score)
    
    def _calculate_engagement_score(self):
        """
        Calculate engagement score based on recent emotions
        Returns:
            float: engagement score between 0 and 1
        """
        if not self.emotion_history:
            return 0.0
        
        # Calculate weighted average of emotions
        weighted_scores = []
        for emotion, confidence in self.emotion_history:
            weight = self.emotion_weights.get(emotion, 0.0)
            weighted_scores.append(weight * confidence)
        
        return np.mean(weighted_scores) if weighted_scores else 0.0
    
    def get_engagement_level(self):
        """
        Get the current engagement level
        Returns:
            tuple: (engagement_score, engagement_status)
        """
        if not self.engagement_scores:
            return 0.0, "No Data"
        
        current_score = self.engagement_scores[-1]
        
        if current_score >= 0.7:
            status = "Highly Engaged"
        elif current_score >= 0.4:
            status = "Moderately Engaged"
        else:
            status = "Disengaged"
        
        return current_score, status
    
    def get_engagement_trend(self):
        """
        Get the trend of engagement over time
        Returns:
            tuple: (trend_direction, trend_magnitude)
        """
        if len(self.engagement_scores) < 2:
            return "stable", 0.0
        
        recent_scores = list(self.engagement_scores)
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if abs(trend) < 0.01:
            direction = "stable"
        elif trend > 0:
            direction = "improving"
        else:
            direction = "declining"
        
        return direction, abs(trend)
    
    def get_alert_status(self):
        """
        Check if an alert should be triggered
        Returns:
            tuple: (should_alert, alert_message)
        """
        current_score, status = self.get_engagement_level()
        trend_direction, trend_magnitude = self.get_engagement_trend()
        
        if current_score < 0.3 and trend_direction == "declining":
            return True, "Critical: Participant showing signs of disengagement"
        elif current_score < 0.4:
            return True, "Warning: Participant engagement is low"
        elif trend_direction == "declining" and trend_magnitude > 0.1:
            return True, "Notice: Participant engagement is decreasing"
        
        return False, ""
    
    def detect_eyes(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)
        return len(eyes) > 0  # Returns True if eyes are detected
        
    def calculate_engagement(self, emotion, confidence, face_img):
        # Check if eyes are visible
        eyes_visible = self.detect_eyes(face_img)
        
        # Base engagement score from emotion
        if emotion == 'happy':
            base_score = 0.8
        elif emotion == 'neutral':
            base_score = 0.6
        elif emotion == 'surprised':
            base_score = 0.7
        elif emotion == 'sad':
            base_score = 0.4
        else:  # angry
            base_score = 0.3
            
        # Adjust score based on confidence
        adjusted_score = base_score * confidence
        
        # If eyes are not visible, reduce engagement score
        if not eyes_visible:
            adjusted_score *= 0.5
            
        # Record engagement with timestamp
        self.engagement_history.append({
            'timestamp': datetime.now(),
            'score': adjusted_score,
            'emotion': emotion,
            'confidence': confidence,
            'eyes_visible': eyes_visible
        })
        
        # Remove old records
        self._clean_history()
        
        return adjusted_score
    
    def get_engagement_status(self):
        if not self.engagement_history:
            return "No data available"
            
        recent_scores = [record['score'] for record in self.engagement_history]
        avg_score = np.mean(recent_scores)
        
        if avg_score >= self.engagement_threshold:
            return "High"
        elif avg_score >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _clean_history(self):
        current_time = datetime.now()
        self.engagement_history = [
            record for record in self.engagement_history
            if current_time - record['timestamp'] <= self.attention_window
        ] 