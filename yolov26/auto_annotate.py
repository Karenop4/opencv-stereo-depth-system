import os
import cv2
import shutil
import random
from tqdm import tqdm

def create_dirs():
    base = 'datasets/human-faces'
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base, 'labels', split), exist_ok=True)
    return base

def yolo_format(bbox, img_w, img_h):
    # YOLO format: class x_center y_center width height (normalized)
    x, y, w, h = bbox
    cx = (x + w/2) / img_w
    cy = (y + h/2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def main():
    base_dir = 'Human Faces Dataset'
    out_dir = create_dirs()
    
    # Usar Haar Cascade para la detección automática de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    images = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, file))
                
    # Mezclar y dividir en train/val (80%/20%)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    def process_split(img_list, split_name):
        valid_count = 0
        for img_path in tqdm(img_list, desc=f"Procesando {split_name}"):
            img = cv2.imread(img_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            
            # Solo guardamos imágenes donde se detectó al menos un rostro
            if len(faces) > 0:
                h, w = img.shape[:2]
                filename = os.path.basename(img_path)
                parent = os.path.basename(os.path.dirname(img_path))
                new_filename = f"{parent}_{filename}" # evitar colisiones de nombres
                
                # Copiar imagen
                dest_img = os.path.join(out_dir, 'images', split_name, new_filename)
                shutil.copy(img_path, dest_img)
                
                # Crear archivo de etiquetas (labels)
                label_name = os.path.splitext(new_filename)[0] + '.txt'
                dest_label = os.path.join(out_dir, 'labels', split_name, label_name)
                with open(dest_label, 'w') as f:
                    for face in faces:
                        f.write(yolo_format(face, w, h) + '\n')
                valid_count += 1
        print(f"[{split_name}] Imágenes anotadas y guardadas: {valid_count}")

    print("Iniciando anotación automática...")
    process_split(train_imgs, 'train')
    process_split(val_imgs, 'val')
    print("¡Anotación completada! El dataset YOLO está en 'datasets/human-faces/'")

if __name__ == "__main__":
    main()
