import os
import glob
import cv2
import dlib
import numpy as np
import torch
from tqdm import tqdm

# Configuración
GRID_ROOT = "data/unprocessed"  # Donde tienes s1, s2...
OUTPUT_ROOT = "data/rois"  # Donde se guardarán los .npz o .pt
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if not frames: return

    # 1. Detectar landmarks en el primer frame (Suficiente para GRID)
    img = frames[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        print(f"Warning: No face in {video_path}")
        return

    shape = predictor(gray, rects[0])
    pts = np.array([[p.x, p.y] for p in shape.parts()])
    
    # Centro de la boca (puntos 48-68)
    mouth_center = np.mean(pts[48:68], axis=0)
    cx, cy = int(mouth_center[0]), int(mouth_center[1])

    # 2. Recortar todos los frames
    mouth_rois = []
    crop_w, crop_h = 96, 96
    
    for frame in frames:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Coordenadas de recorte asegurando que no salimos de la imagen
        y1 = max(0, cy - crop_h // 2)
        y2 = min(frame_gray.shape[0], cy + crop_h // 2)
        x1 = max(0, cx - crop_w // 2)
        x2 = min(frame_gray.shape[1], cx + crop_w // 2)

        crop = frame_gray[y1:y2, x1:x2]
        
        # Padding simple si el recorte es menor a 96x96 (bordes)
        if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
            padded = np.zeros((crop_h, crop_w), dtype=np.uint8)
            h, w = crop.shape
            padded[:h, :w] = crop
            crop = padded
            
        mouth_rois.append(crop)

    # 3. Guardar como tensor .pt (Compatible con PyTorch)
    # Shape: (Time, 96, 96)
    tensor_out = torch.tensor(np.array(mouth_rois), dtype=torch.float32) / 255.0
    torch.save(tensor_out, save_path)

# Loop Principal
speakers = glob.glob(os.path.join(GRID_ROOT, "s*"))
for spk_dir in speakers:
    spk_name = os.path.basename(spk_dir)
    out_dir = os.path.join(OUTPUT_ROOT, spk_name)
    os.makedirs(out_dir, exist_ok=True)
    
    videos = glob.glob(os.path.join(spk_dir, "*.mpg"))
    print(f"Procesando {spk_name}: {len(videos)} videos")
    
    for vid in tqdm(videos):
        fname = os.path.basename(vid).replace(".mpg", ".pt")
        save_p = os.path.join(out_dir, fname)
        if not os.path.exists(save_p):
            process_video(vid, save_p)