import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

class VisualAdapterConfig(BaseModel):
    face_dim: int = 512  # Dimensión de las características faciales
    lip_dim: int = 256   # Dimensión de las características de los labios
    bottleneck_dim: int = 128  # Dimensión del cuello de botella

class VisualAdapter(nn.Module):
    def __init__(self, visual_dim, text_dim, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        # "bottleneck structure with LayerNorm, GELU" 
        self.norm = nn.RMSNorm(visual_dim)
        
        self.temporal_mixer = nn.Sequential(
            nn.Conv1d(visual_dim, visual_dim, kernel_size=3, padding=1, groups=visual_dim),
            nn.GroupNorm(32, visual_dim),
            nn.Dropout(dropout)
        )

        self.projection = nn.Sequential(
            nn.Linear(visual_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, text_dim) # Proyecta al espacio del texto
        )
        
        self.residual_proj = nn.Linear(visual_dim, text_dim) if visual_dim != text_dim else nn.Identity()

    def forward(self, x):
        # x shape: [Batch, Time_Video, Visual_Dim]
        residual = self.residual_proj(x)

        x = x.transpose(1, 2) # [B, Visual_Dim, Time_Video]
        x = self.temporal_mixer(x)
        x = x.transpose(1, 2) # [B, Time_Video, Visual_Dim]

        x = self.norm(x)
        x = self.projection(x)
        return x + residual 
    
    
    # Supongamos que ya tienes los modelos cargados
# lip_model = ... (ResNet18-3D)
# face_model = ... (ResNet o similar)
class LipEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        # Simple ResNet-like 3D frontend
        # Input: [B, 1, T, 96, 96]
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False) # 
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Basic Blocks
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, out_dim)

        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def _make_layer(self, in_planes, planes):
        return BasicBlock3D(in_planes, planes, stride=1)

    def forward(self, x: torch.Tensor):
        # x: [B, T, 96, 96]
        x = x.unsqueeze(1) # [B, 1, T, 96, 96]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # [B, out_dim, T, H', W']
        print(x.shape)
        x = self.avgpool(x) # [B, out_dim, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1) # [B, out_dim, T]
        x = x.transpose(1, 2) # [B, T, out_dim]
        return x
    
class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, planes)
        
        # Esta es la conexión residual: si las dimensiones cambian (stride > 1),
        # necesitamos proyectar la identidad para que se pueda sumar.
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes),
            )

    def forward(self, x):
        identity = x  

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x) 

        out += identity  
        out = self.relu(out)

        return out