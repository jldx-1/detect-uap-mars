"""
server.py

This script sets up a simple Flask server to serve the UAP detection model.
It loads the exported ONNX model and exposes a /predict endpoint for real-time inference.
Location: 
"""

import os
import io
import sys
import cv2
import numpy as np
import onnxruntime
from flask import Flask, request, jsonify
from PIL import Image

# Add the project src\train directory to the Python path for configuration access
sys.path.append(r"E:\projects\UAP\src\train")
import config

app = Flask(__name__)

# Load the ONNX model session globally
onnx_model_path = os.path.join(os.getcwd(), "uap_detection.onnx")
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")
session = onnxruntime.InferenceSession(onnx_model_path)

def preprocess_image_from_bytes(image_bytes, img_size=config.IMG_SIZE):
    """
    Preprocesses an image from bytes:
    - Loads image using PIL, converts to RGB.
    - Resizes to (img_size, img_size).
    - Normalizes to [0, 1] and reorders dimensions to (1, C, H, W).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((img_size, img_size))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects an image file in the POST request under the key 'image'.
    Returns the raw model output as JSON.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400
    
    try:
        image_bytes = file.read()
        input_image = preprocess_image_from_bytes(image_bytes, config.IMG_SIZE)
    except Exception as e:
        return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 500

    # Run inference using the ONNX model
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_image})
    
    # For simplicity, convert the first output to a list and return as JSON
    output_data = outputs[0].tolist()
    
    return jsonify({"output": output_data})

if __name__ == '__main__':
    # Run the server on port 5000, accessible externally
    app.run(host='0.0.0.0', port=5000, debug=True)
