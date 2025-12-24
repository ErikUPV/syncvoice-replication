import os
import glob
import cv2
import dlib
import numpy as np
import torch
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# --- CONFIGURACIÓN ---
GRID_ROOT = "data/unprocessed"
OUTPUT_ROOT = "data/face_feats"  # Guardaremos vectores .pt aquí
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
TARGET_FPS = 25.0
BATCH_SIZE = 32  # Procesar frames en lotes para acelerar la GPU

# --- MODELOS ---
# Usamos dlib para detectar (CPU) para ser consistentes con el script de labios
detector = dlib.get_frontal_face_detector()

# Usamos FaceNet para extraer features (GPU si es posible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preprocesamiento estándar de FaceNet (Normalización)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    # Normalización específica de FaceNet (whitening) se suele hacer internamente
    # o usando mean/std estándar. InceptionResnetV1 espera inputs normalizados si se entrena,
    # pero 'vggface2' suele ir bien con estandarización básica.
    # Usaremos una normalización fija simple:
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

def resample_frames(frames, original_fps, target_fps):
    if abs(original_fps - target_fps) < 0.1:
        return frames
    num_frames = len(frames)
    duration = num_frames / original_fps
    target_num_frames = int(duration * target_fps)
    indices = np.linspace(0, num_frames - 1, target_num_frames).round().astype(int)
    indices = indices[indices < num_frames]
    return [frames[i] for i in indices]

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 25.0
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    if not frames: return

    # 1. Downsample a 25fps
    frames = resample_frames(frames, fps, TARGET_FPS)
    if not frames: return

    # 2. Detectar cara en el frame 0 (Asumimos movimiento mínimo en GRID)
    # Convertimos a RGB para dlib/facenet
    img_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    rects = detector(img_rgb, 1)
    
    if len(rects) == 0:
        # Fallback frame central
        mid = len(frames)//2
        img_mid = cv2.cvtColor(frames[mid], cv2.COLOR_BGR2RGB)
        rects = detector(img_mid, 1)
        if len(rects) == 0: return

    rect = rects[0]
    
    # Coordenadas de la cara con un poco de margen
    l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
    h, w, _ = frames[0].shape
    
    # Margen de seguridad para no salirnos
    l, t = max(0, l), max(0, t)
    r, b = min(w, r), min(h, b)

    # 3. Extraer y preparar batch
    face_crops = []
    for f in frames:
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        crop = f_rgb[t:b, l:r]
        if crop.size == 0: continue
        
        # Transformar a tensor 160x160 normalizado
        tensor = transform(crop)
        face_crops.append(tensor)

    if not face_crops: return

    # Stackear en un tensor grande (Time, 3, 160, 160)
    input_tensor = torch.stack(face_crops).to(device)

    # 4. Inferencia por lotes (para no saturar VRAM si el video es largo)
    embeddings = []
    with torch.no_grad():
        # Procesar en mini-batches
        for i in range(0, len(input_tensor), BATCH_SIZE):
            batch = input_tensor[i : i + BATCH_SIZE]
            # Salida: (Batch, 512)
            emb = resnet(batch)
            embeddings.append(emb.cpu())

    # 5. Concatenar y guardar
    # Resultado final: Tensor (Time, 512)
    final_tensor = torch.cat(embeddings, dim=0)
    torch.save(final_tensor, save_path)

if __name__ == "__main__":
    speakers = glob.glob(os.path.join(GRID_ROOT, "s*"))
    print(f"Procesando {len(speakers)} speakers...")
    
    for spk_dir in speakers:
        spk_name = os.path.basename(spk_dir)
        out_dir = os.path.join(OUTPUT_ROOT, spk_name)
        os.makedirs(out_dir, exist_ok=True)
        
        videos = glob.glob(os.path.join(spk_dir, "*.mpg"))
        videos.sort()
        
        for vid in tqdm(videos, desc=f"Face {spk_name}"):
            fname = os.path.basename(vid).replace(".mpg", ".pt")
            save_p = os.path.join(out_dir, fname)
            if not os.path.exists(save_p):
                process_video(vid, save_p)