from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained tokenizer and model for emotion classification
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Define the labels for the seven emotions
emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

# Function to analyze the emotion from a given text
def analyze_emotion(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    # Pass the tokenized input to the model
    outputs = model(**inputs)
    # Apply softmax to get the predicted emotion probabilities
    predictions = torch.softmax(outputs.logits, dim=-1)
    # Convert the predictions to a list of emotion scores
    emotion_scores = predictions[0].tolist()
    return emotion_scores

# Route to render the homepage (chatbox.html)
@app.route('/')
def home():
    return render_template('chatbox.html')

# Route to handle emotion analysis of user text input
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the input data from the request
    data = request.json
    user_text = data['text']  # Extract user input text
    question_index = data['question_index']  # Extract the question index (if needed)

    # Analyze the emotion scores for the user input text
    emotion_scores = analyze_emotion(user_text)
    
    # Return the emotion scores and the question index as JSON
    return jsonify({
        'emotion_scores': emotion_scores,
        'question_index': question_index
    })

# Route to finalize the emotion analysis and determine the final result
@app.route('/finalize', methods=['POST'])
def finalize():
    # Get the emotion scores from the request
    data = request.json
    emotion_scores = data['scores']  # List of emotion scores from multiple questions

    # Aggregate the emotion scores across all questions
    total_scores = [sum(x) for x in zip(*emotion_scores)]
    
    # Find the index of the highest total score (the predicted emotion)
    max_index = total_scores.index(max(total_scores))
    # Get the emotion label corresponding to the highest score
    emotion = emotion_labels[max_index]
    # Calculate the confidence as the percentage of the total score
    confidence = max(total_scores) * 100 / len(emotion_scores)  

    # Return the predicted emotion and confidence as JSON
    return jsonify({
        'emotion': emotion,
        'confidence': f"{confidence:.2f}%"  # Format the confidence to two decimal places
    })

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)


