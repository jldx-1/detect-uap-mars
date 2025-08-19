# src/train/train.py
from ultralytics import YOLO
import config

def train():
    model = YOLO('yolov5s.pt')
    model.train(
        data='data.yaml',
        epochs=config.NUM_EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        project='runs/train',
        name='uap_experiment'
    )

if __name__ == "__main__":
    train()
