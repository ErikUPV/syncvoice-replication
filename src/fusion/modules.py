import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualAdapterConfig:
    face_dim: int = 512  # Dimensión de las características faciales
    lip_dim: int = 256   # Dimensión de las características de los labios
    bottleneck_dim: int = 128  # Dimensión del cuello de botella

class VisualAdapter(nn.Module):
    def __init__(self, visual_dim, text_dim, bottleneck_dim=128):
        super().__init__()
        # "bottleneck structure with LayerNorm, GELU" 
        self.norm = nn.LayerNorm(visual_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(visual_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, text_dim) # Proyecta al espacio del texto
        )
        
        # Para la conexión residual si las dimensiones no coinciden, 
        # a veces se usa una proyección lineal simple, pero el paper dice
        # "residual connection" en el contexto del adaptador. 
        # Asumiremos que es una proyección directa sumada a la salida proyectada.
        self.residual_proj = nn.Linear(visual_dim, text_dim) if visual_dim != text_dim else nn.Identity()

    def forward(self, x):
        # x shape: [Batch, Time_Video, Visual_Dim]
        residual = self.residual_proj(x)
        x = self.norm(x)
        x = self.projection(x)
        return x + residual # "residual connection"
    
    
    # Supongamos que ya tienes los modelos cargados
# lip_model = ... (ResNet18-3D)
# face_model = ... (ResNet o similar)

def get_video_feats(video_path):
    # 1. Cargar video y extraer frames (T frames, H, W, C)
    frames = load_video_frames(video_path) 
    
    # 2. Detectar landmarks y recortar
    # Retorna tensores: [T, 1, 96, 96] (labios BW) y [T, 3, 224, 224] (cara RGB)
    lip_crops, face_crops = extract_crops_with_mediapipe(frames)
    
    # 3. Pasar por los encoders (sin gradientes, solo inferencia)
    with torch.no_grad():
        # Lip Encoder suele requerir una ventana de tiempo, pero simplifiquemos:
        lip_emb = lip_model(lip_crops)   # Shape: [T, 512]
        face_emb = face_model(face_crops) # Shape: [T, 256]
    
    # 4. Concatenar
    video_feats = torch.cat([lip_emb, face_emb], dim=-1) # Shape: [T, 768]
    
    return video_feats