
"""
utils.py

Utility functions for the UAP detection model training pipeline.
Includes functions for seeding, directory creation, logging, checkpoint management,
and simple visualization of bounding boxes.
"""

import os
import random
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dir(dir_path):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def log_message(message):
    """
    Log a message with a timestamp.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save the model checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    log_message(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load the model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log_message(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint['epoch'], checkpoint['loss']

def visualize_boxes(image, boxes, labels=None):
    """
    Visualize bounding boxes on an image.
    
    Args:
        image (numpy array): Image array in RGB format.
        boxes (list or numpy array): List of bounding boxes [x1, y1, x2, y2].
        labels (list, optional): List of labels corresponding to the boxes.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        if labels is not None:
            ax.text(x1, y1, str(labels[i]), color='yellow', fontsize=12, weight='bold')
    plt.show()

if __name__ == "__main__":
    # Quick tests for utilities
    set_seed(42)
    test_dir = "./test_dir"
    create_dir(test_dir)
    log_message("Test log message: utils.py is working.")