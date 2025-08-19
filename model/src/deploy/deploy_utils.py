"""
deploy_utils.py

Utility functions for deployment tasks for the UAP detection model.
This module provides common functions for image preprocessing and output postprocessing,
which can be used by both the inference script and the server application.

Location: 
"""

import io
import cv2
import numpy as np
from PIL import Image
import config

def preprocess_image_cv2(image_path, img_size=config.IMG_SIZE):
    """
    Preprocess an image from a file path using OpenCV.
    Reads the image, converts to RGB, resizes to (img_size, img_size),
    normalizes pixel values to [0,1], and reshapes to (1, C, H, W).

    Args:
        image_path (str): Path to the image file.
        img_size (int): Target image size (default from config).

    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format.
    image = np.expand_dims(image, axis=0)    # Add batch dimension.
    return image

def preprocess_image_pil(image_bytes, img_size=config.IMG_SIZE):
    """
    Preprocess an image from bytes using PIL.
    Converts the image to RGB, resizes it, normalizes pixel values,
    and converts it to a NumPy array with shape (1, C, H, W).

    Args:
        image_bytes (bytes): Image data in bytes.
        img_size (int): Target image size (default from config).

    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((img_size, img_size))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_output(output, conf_threshold=0.5):
    """
    Postprocess raw model outputs by filtering detections based on a confidence threshold.
    This is a placeholder function. In a complete implementation, this should decode the 
    raw output, apply non-max suppression, and return a list of detected bounding boxes with 
    associated class labels and confidence scores.

    Args:
        output (numpy.ndarray): Raw output from the model.
        conf_threshold (float): Confidence threshold for filtering detections.

    Returns:
        numpy.ndarray: Postprocessed predictions.
    """
    # Dummy implementation: for now, simply return the output.
    # Replace this with actual decoding and non-max suppression logic.
    return output

if __name__ == "__main__":
    # Quick test for preprocessing using cv2
    # Update the path below to reflect an existing image in your new folder structure.
    test_image_path = r"E:\projects\UAP\data\train\img\test_image.jpg"
    try:
        preprocessed = preprocess_image_cv2(test_image_path)
        print("Preprocessed image shape (cv2):", preprocessed.shape)
    except Exception as e:
        print("Error in cv2 preprocessing:", e)
    
    # Test dummy postprocessing function with a generated output
    dummy_output = np.random.rand(1, 30, 20, 20)
    processed_output = postprocess_output(dummy_output)
    print("Postprocessed output shape:", processed_output.shape)
