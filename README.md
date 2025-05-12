# ThirdEye: Real-Time Emotion & Engagement Recognition for Online Learning

ThirdEye is a deep learning-based system designed to monitor and enhance online student engagement by recognizing emotions and tracking attention in real time. It uses computer vision and neural networks to detect faces, classify emotions, and assess engagement based on both facial expressions and eye contact.

## Features
- **Face Detection:** Uses Haar Cascade Frontal Face algorithm for robust face detection.
- **Emotion Recognition:** Classifies emotions into five categories: `angry`, `happy`, `neutral`, `sad`, and `surprised`.
- **Engagement Analysis:** Calculates engagement level using both emotion and eye visibility (eye tracking). Engagement is considered low if the participant's eyes are not visible or if negative emotions are detected.
- **Real-Time Monitoring:** Processes webcam video in real time, displaying results instantly.
- **Automated Feedback:** Shows engagement status (High/Medium/Low) and eye visibility for each participant.
- **Participant Labeling:** All detected faces are labeled as "Participant" instead of "Student" for broader applicability.

## Requirements
- Python 3.10
- OpenCV
- TensorFlow
- Keras
- NumPy
- Pillow
- Matplotlib
- Pandas
- scikit-learn

Install all dependencies using:
```sh
pip install -r requirements.txt
```

## Setup Instructions
1. **Clone the repository** and navigate to the project directory.
2. **Create a virtual environment** (recommended):
   ```sh
   py -3.10 -m venv venv310
   .\venv310\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```sh
   python src/main.py
   ```

## Usage
- The webcam window will open and start detecting faces.
- For each detected face, the system will display:
  - `Participant: [emotion] ([confidence])`
  - `Engagement: [High/Medium/Low]`
  - `Eyes: Visible` or `Eyes: Not Visible`
- Press `q` to quit the application.

## Customization
- **Adding More Emotions:** To add new emotions (e.g., "laugh"), retrain the emotion recognition model with labeled data and update the `self.emotions` list in `src/emotion_recognition.py`.
- **Adjusting Engagement Logic:** You can modify engagement thresholds and logic in `src/engagement_analyzer.py`.

## Notes
- The default model is a placeholder. For best results, train the emotion recognition model on a suitable dataset and load the weights using the `load_weights` method in `EmotionRecognizer`.
- The system is designed for real-time use and works best in well-lit environments.

## License
This project is for educational and research purposes. 