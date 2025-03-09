from flask import Flask, request, jsonify, send_from_directory
import cv2
import base64
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN

# Initialize the Flask application
app = Flask(__name__, static_folder='static')

# Initialize the MTCNN face detector
face_detector = MTCNN()

# Function to decode base64 image data
def decode_base64_image(data):
    """
    Decodes base64-encoded image data into an OpenCV-compatible image.
    """
    try:
        # If data includes a comma (data URI format), split and decode the part after the comma
        if "," in data:
            image_data = base64.b64decode(data.split(',')[1])
        else:
            image_data = base64.b64decode(data)
        
        # Convert the byte data to a NumPy array
        np_img = np.frombuffer(image_data, np.uint8)
        # Decode the NumPy array into an image using OpenCV
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# Function to predict emotion from an image
def predict_emotion(img):
    """
    Detects faces in the image, crops the face, and predicts the emotion using DeepFace.
    """
    if img is None:
        return None, None, None
    
    # Convert the image from BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect faces in the image using MTCNN
    faces = face_detector.detect_faces(img_rgb)
    
    if len(faces) > 0:
        # If faces are detected, process the first face
        face = faces[0]
        # Get the bounding box coordinates of the face
        x, y, w, h = face['box']
        # Crop the detected face from the image
        face_crop = img_rgb[y:y+h, x:x+w]
        
        # Analyze the cropped face for emotions using DeepFace
        result = DeepFace.analyze(face_crop, actions=['emotion'], detector_backend='skip', enforce_detection=False)
        
        # If the result is a list (DeepFace may return a list), select the first result
        if isinstance(result, list):
            result = result[0]

        # Extract the dominant emotion and its probability
        dominant_emotion = result['dominant_emotion']
        dominant_probability = result['emotion'][dominant_emotion]
        # Store the face bounding box coordinates
        face_box_coords = {'x': x, 'y': y, 'width': w, 'height': h}
        return dominant_emotion, dominant_probability, face_box_coords

    # If no face is detected, return None
    return None, None, None

# Define the route for the home page
@app.route('/')
def index():
    """
    Serves the static HTML page (face.html) as the application's home page.
    """
    return send_from_directory(app.static_folder, 'face.html')

# Define the route to handle image uploads and predictions
@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles image uploads, processes the image, and returns emotion prediction results.
    """
    # Retrieve the base64 image data from the POST request's JSON payload
    data = request.json.get('image', '')
    # Decode the image data
    img = decode_base64_image(data)

    # Predict the emotion in the image
    emotion, confidence, face_box_coords = predict_emotion(img)
    
    # If no face is detected, return an error response
    if emotion is None:
        return jsonify({"error": "No face detected"}), 400
    
    # Return the predicted emotion, confidence score, and face bounding box coordinates
    return jsonify({
        "emotion": emotion,
        "confidence": str(confidence),
        "faceBoxCoords": face_box_coords
    })

# Main entry point for the Flask application
if __name__ == '__main__':
    # Start the application in debug mode for development purposes
    app.run(debug=True)
