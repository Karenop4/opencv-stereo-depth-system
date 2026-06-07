import onnxruntime as ort

session = ort.InferenceSession('yolov26/runs/detect/yolov26_faces/weights/best.onnx')
for output in session.get_outputs():
    print(output.name, output.shape, output.type)
