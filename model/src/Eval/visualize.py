"""
visualize.py

This module provides functions to visualize detection results.
It includes methods to overlay bounding boxes with labels and confidence
scores on an image and to plot precision-recall curves.

Location:
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def visualize_detections(image, boxes, labels=None, scores=None):
    """
    Overlays detection results on the input image.
    
    Args:
        image (numpy.ndarray): Input image in RGB format.
        boxes (list or array): List of bounding boxes [x1, y1, x2, y2].
        labels (list, optional): Class labels for each box.
        scores (list, optional): Confidence scores for each detection.
    """
    # Create a copy to draw on
    image_copy = image.copy()
    
    # Generate colors for labels
    if labels is not None:
        unique_labels = list(set(labels))
        colors = {label: [random.randint(0, 255) for _ in range(3)] for label in unique_labels}
    else:
        colors = {0: [255, 0, 0]}
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[i] if labels is not None else 0
        color = colors[label]
        # Draw bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        caption = f"{label}"
        if scores is not None:
            caption += f": {scores[i]:.2f}"
        cv2.putText(image_copy, caption, (x1, max(y1 - 10, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display image using matplotlib (convert BGR to RGB if needed)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_copy)
    plt.axis("off")
    plt.show()

def plot_precision_recall_curve(recalls, precisions):
    """
    Plots the precision-recall curve.
    
    Args:
        recalls (list or array): Recall values.
        precisions (list or array): Precision values.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Dummy test for visualize_detections
    dummy_image = np.ones((500, 500, 3), dtype=np.uint8) * 200  # Gray image
    dummy_boxes = [[50, 50, 150, 150], [200, 200, 300, 350]]
    dummy_labels = [0, 0]
    dummy_scores = [0.95, 0.87]
    visualize_detections(dummy_image, dummy_boxes, dummy_labels, dummy_scores)
    
    # Dummy test for plot_precision_recall_curve
    dummy_recalls = [0.0, 0.5, 0.7, 0.8, 1.0]
    dummy_precisions = [1.0, 0.9, 0.85, 0.8, 0.75]
    plot_precision_recall_curve(dummy_recalls, dummy_precisions)
