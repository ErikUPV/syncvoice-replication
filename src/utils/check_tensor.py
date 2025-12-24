import torch
import cv2
import numpy as np
import sys
import os

def tensor_to_video(pt_path, output_path):
    # 1. Cargar el tensor
    print(f"Cargando {pt_path}...")
    tensor = torch.load(pt_path)
    
    # 2. Verificar forma (Shape)
    print(f"Forma del tensor: {tensor.shape}")
    # Esperamos (T, 96, 96) o (T, 1, 96, 96)
    
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1) # Quitar canal si es (T, 1, H, W)
    
    # 3. Des-normalizar
    # Si guardaste float 0.0-1.0, multiplicamos por 255.
    # Si guardaste uint8 0-255, lo dejamos igual.
    data = tensor.numpy()
    if data.max() <= 1.0:
        print("Detectado rango 0-1. Multiplicando por 255.")
        data = (data * 255).astype(np.uint8)
    else:
        data = data.astype(np.uint8)

    # 4. Crear video
    T, H, W = data.shape
    fps = 25
    # isColor=False porque es escala de grises
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H), isColor=False)
    
    for i in range(T):
        frame = data[i]
        out.write(frame)
    
    out.release()
    print(f"Video guardado en: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python check_tensor.py <archivo.pt>")
    else:
        input_pt = sys.argv[1]
        output_mp4 = input_pt.replace(".pt", "_debug.mp4")
        tensor_to_video(input_pt, output_mp4)