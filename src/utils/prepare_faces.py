import os
import glob
import cv2
import dlib
import numpy as np
import torch
import argparse
import re
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# --- ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description="Grid Face Feature Extractor")
parser.add_argument("--filter", type=str, default=".*", help="Regex para filtrar speakers (ej: 's[1-4]' o 's1|s2')")
args = parser.parse_args()

# --- CONFIGURACIÓN ---
GRID_ROOT = "data/unprocessed"
OUTPUT_ROOT = "data/face_feats"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
TARGET_FPS = 25.0
BATCH_SIZE = 256  # <--- SUBIDO A 256 PARA TU A30

# --- MODELOS ---
detector = dlib.get_frontal_face_detector()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicializar FaceNet
# Nota: Si lanzas 4 procesos, cada uno cargará una copia del modelo en VRAM.
# La A30 tiene 24GB, así que 4 copias x 0.5GB es insignificante.
print(f"[{args.filter}] Cargando modelo en {device}...")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

def resample_frames(frames, original_fps, target_fps):
    if abs(original_fps - target_fps) < 0.1: return frames
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

    frames = resample_frames(frames, fps, TARGET_FPS)
    if not frames: return

    # Detección en frame 0
    img_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    rects = detector(img_rgb, 1)
    
    if len(rects) == 0:
        mid = len(frames)//2
        img_mid = cv2.cvtColor(frames[mid], cv2.COLOR_BGR2RGB)
        rects = detector(img_mid, 1)
        if len(rects) == 0: return

    rect = rects[0]
    l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
    h, w, _ = frames[0].shape
    l, t = max(0, l), max(0, t)
    r, b = min(w, r), min(h, b)

    face_crops = []
    for f in frames:
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        crop = f_rgb[t:b, l:r]
        if crop.size == 0: continue
        tensor = transform(crop)
        face_crops.append(tensor)

    if not face_crops: return

    input_tensor = torch.stack(face_crops).to(device)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(input_tensor), BATCH_SIZE):
            batch = input_tensor[i : i + BATCH_SIZE]
            emb = resnet(batch)
            embeddings.append(emb.cpu())

    final_tensor = torch.cat(embeddings, dim=0)
    torch.save(final_tensor, save_path)

if __name__ == "__main__":
    # --- FILTRADO CON REGEX ---
    all_speakers = sorted(glob.glob(os.path.join(GRID_ROOT, "s*")))
    
    # Filtramos usando el argumento regex
    speakers = [s for s in all_speakers if re.search(args.filter, os.path.basename(s))]
    
    if not speakers:
        print(f"No se encontraron speakers con el filtro: {args.filter}")
        exit()

    print(f"[{args.filter}] Procesando {len(speakers)} speakers: {[os.path.basename(s) for s in speakers]}")
    
    for spk_dir in speakers:
        spk_name = os.path.basename(spk_dir)
        out_dir = os.path.join(OUTPUT_ROOT, spk_name)
        os.makedirs(out_dir, exist_ok=True)
        
        videos = glob.glob(os.path.join(spk_dir, "*.mpg"))
        videos.sort()
        
        # Barra de progreso personalizada
        for vid in tqdm(videos, desc=f"[{args.filter}] {spk_name}", position=0, leave=True):
            fname = os.path.basename(vid).replace(".mpg", ".pt")
            save_p = os.path.join(out_dir, fname)
            if not os.path.exists(save_p):
                process_video(vid, save_p)