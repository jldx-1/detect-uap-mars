"""
inference.py

This script loads the exported ONNX model and runs inference on new images.
Location:
"""

import os
import sys
import cv2
import numpy as np
import onnxruntime

# Add the project src\train directory to the Python path to import configuration
sys.path.append(r"E:\projects\UAP\src\train")
import config

def preprocess_image(image_path, img_size=config.IMG_SIZE):
    """
    Reads an image from disk, resizes it to the target size, and normalizes it.
    Returns a numpy array of shape (1, channels, img_size, img_size).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    # Normalize image: convert to float32 and scale to [0, 1]
    image = image.astype(np.float32) / 255.0
    # Change shape from (H, W, C) to (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension: (1, C, H, W)
    image = np.expand_dims(image, axis=0)
    return image

def run_inference(image_path, onnx_model_path="uap_detection.onnx"):
    """
    Loads the ONNX model and runs inference on a single image.
    """
    # Preprocess the input image
    input_image = preprocess_image(image_path, img_size=config.IMG_SIZE)
    
    # Initialize ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_model_path)
    
    # Get the input name for the model
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: input_image})
    
    # For this shell, we simply return the raw output
    print("Inference output shape:", outputs[0].shape)
    return outputs[0]

if __name__ == "__main__":
    # Path to a test image within the new structure; for example, in the train split's img folder:
    test_image_path = r"E:\projects\UAP\data\train\img\test_image.jpg"
    
    # Run inference and print the raw output
    output = run_inference(test_image_path)
    print("Inference complete. Raw output:", output)
