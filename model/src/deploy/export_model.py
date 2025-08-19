"""
export_model.py

This script converts the trained UAP detection model to ONNX format.
It loads the best model checkpoint, builds the model, and exports it to ONNX.
Location: 
"""

import os
import torch
import sys

# Ensure the project src directory is in the Python path to import modules from train folder
sys.path.append(r"E:\projects\UAP\src")

import config
from train.model import build_model  # Assuming model.py is in E:\projects\UAP\src\train

def export_to_onnx():
    # Build the model and load the best checkpoint
    model = build_model().to(config.DEVICE)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found:", checkpoint_path)
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()

    # Create a dummy input tensor with shape (1, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)
    dummy_input = torch.randn(1, config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE, device=config.DEVICE)

    # Define the output ONNX file path in the deploy directory
    onnx_path = os.path.join(os.getcwd(), "uap_detection.onnx")

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print("Model successfully exported to ONNX at:", onnx_path)

if __name__ == "__main__":
    export_to_onnx()
