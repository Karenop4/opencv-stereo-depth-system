from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/yolov26_faces/weights/best.pt')

# Export the model to ONNX format
success = model.export(format='onnx', opset=12)
print("Exported to:", success)
