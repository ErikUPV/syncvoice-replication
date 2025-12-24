import os
import glob
import cv2
import dlib
import numpy as np
import torch
import argparse
import re
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from tqdm import tqdm

# --- CONFIGURACIÓN GLOBAL ---
GRID_ROOT = "data/unprocessed"
OUTPUT_ROOT = "data/face_feats"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
TARGET_FPS = 25.0

# --- DATASET CON CARGA PARALELA ---
class GridFaceDataset(Dataset):
    def __init__(self, video_paths, predictor_path):
        self.video_paths = video_paths
        self.predictor_path = predictor_path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

    def __len__(self):
        return len(self.video_paths)

    # Inicialización Lazy para evitar errores de Pickle en multiproceso
    def get_detector(self):
        if not hasattr(self, 'detector'):
            self.detector = dlib.get_frontal_face_detector()
        return self.detector

    def get_predictor(self):
        if not hasattr(self, 'predictor'):
            self.predictor = dlib.shape_predictor(self.predictor_path)
        return self.predictor

    def resample_frames(self, frames, original_fps):
        if abs(original_fps - TARGET_FPS) < 0.1: return frames
        num_frames = len(frames)
        duration = num_frames / original_fps
        target_num_frames = int(duration * TARGET_FPS)
        indices = np.linspace(0, num_frames - 1, target_num_frames).round().astype(int)
        indices = indices[indices < num_frames]
        return [frames[i] for i in indices]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # Leemos video (CPU INTENSIVO)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 25.0
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames: return None

        frames = self.resample_frames(frames, fps)
        if not frames: return None

        # Detección (CPU INTENSIVO)
        detector = self.get_detector()
        
        # Detectar en frame 0
        img_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        rects = detector(img_rgb, 1)
        
        if len(rects) == 0:
            # Fallback frame central
            mid = len(frames)//2
            img_mid = cv2.cvtColor(frames[mid], cv2.COLOR_BGR2RGB)
            rects = detector(img_mid, 1)
            if len(rects) == 0: return None

        rect = rects[0]
        predictor = self.get_predictor() # Si necesitas landmarks, aquí. Por ahora usamos rect.
        
        l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
        h, w, _ = frames[0].shape
        l, t = max(0, l), max(0, t)
        r, b = min(w, r), min(h, b)

        # Recorte y Transformación
        face_crops = []
        for f in frames:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            crop = f_rgb[t:b, l:r]
            
            # Check crop válido
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0: continue
            
            try:
                tensor = self.transform(crop)
                face_crops.append(tensor)
            except Exception:
                continue

        if not face_crops: return None

        # Stackeamos (Time, 3, 160, 160)
        # Nota: No enviamos a GPU aquí, lo hacemos en el main loop
        video_tensor = torch.stack(face_crops)
        
        # Calculamos ruta de salida para devolverla
        fname = os.path.basename(video_path).replace(".mpg", ".pt")
        # Asumimos estructura data/s1/video.mpg -> data/face_feats/s1/video.pt
        spk_name = os.path.basename(os.path.dirname(video_path))
        save_path = os.path.join(OUTPUT_ROOT, spk_name, fname)
        
        return video_tensor, save_path

# Función Collate personalizada para manejar videos de distinta duración
# (El DataLoader por defecto intenta apilar, y fallaría si T varía)
def collate_fn(batch):
    # Filtramos None (videos fallidos)
    batch = [b for b in batch if b is not None]
    return batch

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=".*")
    # Número de procesos CPU cargando datos. ¡Sube esto si la GPU duerme!
    # Prueba con 8, 12 o 16 dependiendo de tu CPU.
    parser.add_argument("--workers", type=int, default=8) 
    args = parser.parse_args()

    # 1. Buscar archivos
    all_speakers = glob.glob(os.path.join(GRID_ROOT, "s*"))
    speakers = [s for s in all_speakers if re.search(args.filter, os.path.basename(s))]
    
    if not speakers:
        print("No speakers found.")
        exit()

    video_files = []
    for spk in speakers:
        # Creamos directorios de salida
        spk_name = os.path.basename(spk)
        os.makedirs(os.path.join(OUTPUT_ROOT, spk_name), exist_ok=True)
        
        # Filtramos los que ya existen para no repetir trabajo
        vids = glob.glob(os.path.join(spk, "*.mpg"))
        for v in vids:
            fname = os.path.basename(v).replace(".mpg", ".pt")
            out_p = os.path.join(OUTPUT_ROOT, spk_name, fname)
            if not os.path.exists(out_p):
                video_files.append(v)
    
    print(f"Procesando {len(video_files)} videos nuevos con {args.workers} workers...")

    # 2. Configurar Dataset y Loader
    dataset = GridFaceDataset(video_files, PREDICTOR_PATH)
    
    # batch_size=1 porque cada "item" es un video entero (Tensor de ~75 frames)
    # y la GPU procesará ese video de golpe.
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.workers, 
        collate_fn=collate_fn,
        pin_memory=True # Acelera paso a GPU
    )

    # 3. Modelo en GPU
    device = torch.device('cuda')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 4. Loop de Inferencia
    # Usamos tqdm sobre el loader
    for batch in tqdm(loader, total=len(loader)):
        if not batch: continue
        
        # El batch es una lista de tuplas [(tensor, path), ...] debido a batch_size=1
        # Sacamos el primer (y único) elemento
        video_tensor, save_path = batch[0]
        
        # Mover a GPU y procesar
        # video_tensor shape: (T, 3, 160, 160)
        video_tensor = video_tensor.to(device, non_blocking=True)
        
        with torch.no_grad():
            # Si el video es muy largo (>200 frames), podríamos necesitar partirlo,
            # pero en GRID (75 frames) cabe sobrado en la A30.
            embeddings = resnet(video_tensor)
            
        # Guardar (Movemos a CPU para guardar)
        torch.save(embeddings.cpu(), save_path)