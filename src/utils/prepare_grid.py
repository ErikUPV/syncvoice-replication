import os
import glob
import cv2
import dlib
import numpy as np
import torch
from tqdm import tqdm

# Configuración
GRID_ROOT = "data/unprocessed"  # Tu ruta original
OUTPUT_ROOT = "data/rois"       # Tu ruta de salida
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
TARGET_FPS = 25.0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def resample_frames(frames, original_fps, target_fps):
    """
    Resamplea una lista de frames al target_fps usando Nearest Neighbor.
    Mantiene la nitidez de los labios (crucial para TCN) a costa de micro-jitter.
    """
    if abs(original_fps - target_fps) < 0.1:
        return frames # Ya es 25fps, no tocar.
    
    num_frames = len(frames)
    duration = num_frames / original_fps
    target_num_frames = int(duration * target_fps)
    
    # Índices de los frames que vamos a quedarnos
    indices = np.linspace(0, num_frames - 1, target_num_frames).round().astype(int)
    
    # Filtrar para no salirnos de rango por redondeo
    indices = indices[indices < num_frames]
    
    return [frames[i] for i in indices]

def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    
    # Obtener FPS originales del video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Fallback por si cv2 falla al leer los metadatos (común en .mpg viejos)
    if original_fps == 0 or np.isnan(original_fps):
        original_fps = 25.0 
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if not frames: return

    # --- NUEVO: Downsample a 25fps antes de procesar ---
    frames = resample_frames(frames, original_fps, TARGET_FPS)
    
    if not frames: return

    # 1. Detectar landmarks en el primer frame (del video ya resampleado)
    img = frames[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        # Fallback: Intentar en el frame central si el primero falla (a veces la intro es negra/ruido)
        mid_idx = len(frames) // 2
        gray_mid = cv2.cvtColor(frames[mid_idx], cv2.COLOR_BGR2GRAY)
        rects = detector(gray_mid, 1)
        
        if len(rects) == 0:
            # print(f"Warning: No face in {video_path}") # Spam en consola, mejor comentar
            return

    shape = predictor(gray, rects[0])
    pts = np.array([[p.x, p.y] for p in shape.parts()])
    
    # Centro de la boca (puntos 48-68 de dlib)
    mouth_center = np.mean(pts[48:68], axis=0)
    cx, cy = int(mouth_center[0]), int(mouth_center[1])

    # 2. Recortar todos los frames
    mouth_rois = []
    crop_w, crop_h = 96, 96
    
    for frame in frames:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Coordenadas seguras
        y1 = max(0, cy - crop_h // 2)
        y2 = min(frame_gray.shape[0], cy + crop_h // 2)
        x1 = max(0, cx - crop_w // 2)
        x2 = min(frame_gray.shape[1], cx + crop_w // 2)

        crop = frame_gray[y1:y2, x1:x2]
        
        # Padding si tocamos bordes
        if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
            padded = np.zeros((crop_h, crop_w), dtype=np.uint8)
            h, w = crop.shape
            padded[:h, :w] = crop
            crop = padded
            
        mouth_rois.append(crop)

    # 3. Guardar
    # Normalizamos 0-1 float32 para que PyTorch lo ingiera directo
    tensor_out = torch.tensor(np.array(mouth_rois), dtype=torch.float32) / 255.0
    torch.save(tensor_out, save_path)

# Loop Principal
if __name__ == "__main__":
    if not os.path.exists(PREDICTOR_PATH):
        print(f"ERROR: No se encuentra {PREDICTOR_PATH}")
        exit(1)

    speakers = glob.glob(os.path.join(GRID_ROOT, "s*"))
    print(f"Encontrados {len(speakers)} directorios de speakers.")

    for spk_dir in speakers:
        spk_name = os.path.basename(spk_dir)
        out_dir = os.path.join(OUTPUT_ROOT, spk_name)
        os.makedirs(out_dir, exist_ok=True)
        
        videos = glob.glob(os.path.join(spk_dir, "*.mpg"))
        # Ordenamos para tener progreso consistente
        videos.sort()
        
        # Usamos tqdm para barra de progreso por speaker
        for vid in tqdm(videos, desc=f"Speaker {spk_name}"):
            fname = os.path.basename(vid).replace(".mpg", ".pt")
            save_p = os.path.join(out_dir, fname)
            
            # Skip si ya existe
            if not os.path.exists(save_p):
                process_video(vid, save_p)