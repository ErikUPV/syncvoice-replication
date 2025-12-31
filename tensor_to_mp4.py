def tensor_to_mp4(tensor_path, output_filename, fps=25):
    # 1. Cargar el tensor
    # Dimensiones esperadas: (T, 96, 96)
    data = torch.load(tensor_path)
    
    if data.ndim != 3:
        raise ValueError(f"Se esperaba un tensor de 3 dimensiones, pero se obtuvo {data.ndim}")

    # 2. Normalización
    # Convertimos a float para cálculos y escalamos al rango [0, 255]
    data = data.float()
    min_val = data.min()
    max_val = data.max()
    
    if max_val - min_val > 0:
        data = (data - min_val) / (max_val - min_val) * 255
    
    # Convertir a formato compatible con OpenCV (uint8 y numpy)
    frames = data.byte().cpu().numpy()

    # 3. Configurar el VideoWriter
    height, width = frames.shape[1], frames.shape[2]
    # 'mp4v' es un codec estándar para .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=False)

    # 4. Escribir los frames
    for i in range(frames.shape[0]):
        out.write(frames[i])

    out.release()
    print(f"Video guardado exitosamente como: {output_filename}")

if __name__ == "__main__":
    import torch
    import cv2
    import argparse

    parser = argparse.ArgumentParser(description="Convertir tensor de video a archivo MP4")
    parser.add_argument("--tensor_path", type=str, help="Ruta al archivo del tensor (.pt)")
    parser.add_argument("--output_filename", type=str, help="Nombre del archivo de salida (.mp4)")
    parser.add_argument("--fps", type=int, default=25, help="Frames por segundo del video de salida")

    args = parser.parse_args()

    tensor_to_mp4(args.tensor_path, args.output_filename, fps=args.fps)