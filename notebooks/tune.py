from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="1bJIstIKXufFtSkDGveO")
project = rf.workspace("davids-workspace-mtj3j").project("my-first-project-zirsv")
dataset = project.version(1).download("yolov8")
# --- 1. Download the Dataset from Roboflow ---
# (Make sure to put your actual API key back in)

print(f"Dataset downloaded to: {dataset.location}")

# --- 2. Train the YOLOv8 Model ---
# Load the pre-trained Nano model (it's the fastest and lightest)
model = YOLO("yolov8n.pt")

# Train the model
# We point 'data' to the data.yaml file that Roboflow just downloaded
# We use epochs=25 (how many times it looks at the data) for a fast test run
print("Starting training...")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=15,
    imgsz=640,
    device="cpu",  # Forces it to use your standard processor
)

print(
    "Training complete! Your custom model is saved in the 'runs/detect/train/weights' folder as 'best.pt'"
)
