from flask import Flask, render_template, jsonify
import sounddevice as sd
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import io
import soundfile as sf

# Initialize Flask application
app = Flask(__name__)

# Set parameters for audio recording
duration = 10  # Duration of the recording in seconds
fs = 16000     # Sampling frequency in Hz (16kHz)

# Function to extract audio features using MFCC and pitch
def extract_features(audio, sr):
    # Extract MFCC (Mel-frequency cepstral coefficients) features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    # Compute the mean of the MFCC features across time
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Extract pitch information from the audio signal
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    # Compute the average pitch (ignoring zero pitches)
    pitch = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
    
    # Combine MFCC mean and pitch into a single feature vector
    features = np.concatenate((mfcc_mean, [pitch]))
    return features

# Generate synthetic training data for emotion classification (this would typically come from real labeled data)
X_train = np.random.rand(100, 14)  # 100 samples with 14 features (13 MFCC + 1 pitch)
y_train = np.random.randint(0, 7, 100)  # Random emotion labels (0-6)

# Initialize the StandardScaler for feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Scale the training data

# Initialize the Support Vector Classifier (SVC) model with probability estimates
classifier = SVC(probability=True)
classifier.fit(X_train_scaled, y_train)  # Train the classifier on the scaled data

# Route for the homepage (renders a template with a button to start recording)
@app.route('/')
def index():
    return render_template('voice.html')  # Render the HTML template with the recording interface

# Route to handle recording and emotion prediction
@app.route('/record', methods=['POST'])
def record():
    # Record audio from the microphone
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished

    # Extract features from the recorded audio
    features = extract_features(audio_data.flatten(), fs).reshape(1, -1)
    features_scaled = scaler.transform(features)  # Scale the extracted features

    # Predict emotion probabilities and the predicted label
    emotion_probs = classifier.predict_proba(features_scaled)[0]  # Get the probability of each emotion
    emotion_label = classifier.predict(features_scaled)[0]  # Get the predicted emotion label

    # Define the list of possible emotions
    emotions = ["neutral", "happy", "sad", "anger", "fear", "disgust", "surprise"]
    predicted_emotion = emotions[emotion_label]  # Get the predicted emotion name
    confidence = emotion_probs[emotion_label]  # Get the confidence level for the prediction

    # Create a JSON response with the predicted emotion and confidence
    response = {
        'emotion': predicted_emotion,
        'confidence': f"{confidence * 100:.2f}%"  # Format the confidence as a percentage
    }

    return jsonify(response)  # Return the response as JSON

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)  # Run in debug mode for easier development

