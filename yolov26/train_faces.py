from ultralytics import YOLO

def main():
    print("Cargando el modelo base...")
    # Usamos el modelo que tienes en la carpeta actual
    model = YOLO('yolo26n.pt') 
    
    print("Iniciando fine-tuning...")
    # Comenzar el entrenamiento
    # Nota: Si tu PC no tiene mucha memoria VRAM o RAM, puedes reducir el 'batch' a 8 o 4.
    results = model.train(
        data='data.yaml',
        epochs=20,          # Puedes aumentar esto si quieres mayor precisión (e.g., 50)
        imgsz=640,          # Tamaño de imagen
        batch=16,           # Imágenes por lote
        name='yolov26_faces', # Nombre de la carpeta de resultados
        patience=10         # Si no mejora en 10 épocas, se detiene
    )
    
    print("¡Fine-tuning completado! Revisa la carpeta 'runs/detect/yolov26_faces' para ver los resultados.")

if __name__ == "__main__":
    main()
