"""
config.py

Configuration file for the sleek UAP detection model training pipeline.
This file contains hyperparameters, paths, and other settings.
Location: 
"""

import os

# ---------------------
# General Settings
# ---------------------
DEVICE = "cuda"  # Ensure CUDA is available; adjust if using CPU
NUM_WORKERS = 4
SEED = 42

# ---------------------
# Directories for Data, Checkpoints, and Logs
# ---------------------
# Data folder structure is expected to be:
# /app/data/
# ├── train/
# │   ├── img/
# │   ├── annotations/
# │   └── mask/
# ├── eval/
# │   ├── img/
# │   ├── annotations/
# │   └── mask/
# └── test/
#     ├── img/
#     ├── annotations/
#     └── mask/
DATA_DIR = "/app/data"

# Checkpoints and logs for the training pipeline (inside the container)
CHECKPOINT_DIR = "/app/src/train/checkpoints"
LOG_DIR = "/app/src/train/logs"

# ---------------------
# Data Preprocessing
# ---------------------
# (Data splitting is not needed here since the data is already split.)
IMG_SIZE = 640  # Resize images to 640x640 pixels

# ---------------------
# Training Parameters
# ---------------------
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9

# ---------------------
# Learning Rate Scheduler
# ---------------------
LR_SCHEDULER = "cosine"  # Options: "cosine", "step", etc.
STEP_SIZE = 10         # For step scheduler
GAMMA = 0.1            # Decay factor for step scheduler

# ---------------------
# Model Parameters
# ---------------------
MODEL_NAME = "yolov5s"   # Using YOLOv5 small model for the sleek approach
NUM_CLASSES = 1          # Number of classes (only UAP/anomaly)
PRETRAINED = True        # Use pre-trained weights (e.g., from COCO)
BACKBONE = "CSPDarknet"  # Backbone architecture used in YOLOv5
INPUT_CHANNELS = 3       # Number of image channels (RGB)

# ---------------------
# Checkpointing & Logging
# ---------------------
SAVE_INTERVAL = 5        # Save checkpoint every 5 epochs
BEST_MODEL_METRIC = "mAP"  # Metric to select the best model
VERBOSE = True           # Enable detailed logging

# ---------------------
# Miscellaneous
# ---------------------
# Additional custom parameters can be added here as needed.
