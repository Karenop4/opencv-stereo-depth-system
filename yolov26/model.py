import cv2
from ultralytics import YOLO

import os
weights_path = 'runs/detect/yolov26_faces/weights/best.pt'

if os.path.exists(weights_path):
    print("Cargando modelo fine-tuneado para rostros...")
    model = YOLO(weights_path)
else:
    print("El entrenamiento aún no ha terminado. Cargando modelo base por ahora...")
    model = YOLO('yolo26n.pt')

cam = cv2.VideoCapture(1)  # USB CAM

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # classes=[0] => solo personas
    results = model(frame, classes=[0], conf=0.4)

    annotated = results[0].plot()

    cv2.imshow('YOLOv26 - Personas', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()