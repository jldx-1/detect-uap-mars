from ultralytics import YOLO

# Load the best weights from the training run
model = YOLO('runs/train/uap_experiment2/weights/best.pt')

# Run inference on your test images and save annotated outputs
model.predict(
    source='data/test/images',
    save=True,
    project='data',
    name='predictions'
)
