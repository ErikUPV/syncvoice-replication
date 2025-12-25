import os
import glob
import torch
import wespeaker
import argparse
import re
import numpy as np
from tqdm import tqdm

# --- CONFIGURACIÓN ---
GRID_ROOT = "data/audio"
OUTPUT_ROOT = "data/speaker_embeddings"
SCP_FILE = "wav.scp"  # Archivo temporal que necesita WeSpeaker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default=".*", help="Regex para filtrar speakers (ej: 's[1-4]')")
    args = parser.parse_args()

    # 1. ESCANEAR ARCHIVOS Y GENERAR WAV.SCP
    # WeSpeaker necesita un archivo de texto con el formato: <ID_UNICO> <RUTA_ABSOLUTA>
    print("Escaneando archivos y generando wav.scp...")
    
    all_speakers = glob.glob(os.path.join(GRID_ROOT, "s*"))
    # Filtro de speakers
    speakers = [s for s in all_speakers if re.search(args.filter, os.path.basename(s))]
    
    if not speakers:
        print(f"No se encontraron speakers con el filtro '{args.filter}'")
        exit()

    audio_count = 0
    
    # Abrimos el archivo .scp para escritura
    with open(SCP_FILE, 'w') as f_scp:
        for spk_dir in speakers:
            spk_name = os.path.basename(spk_dir)
            
            # Crear carpeta de destino para los .pt
            os.makedirs(os.path.join(OUTPUT_ROOT, spk_name), exist_ok=True)
            
            wavs = glob.glob(os.path.join(spk_dir, "*.wav"))
            for wav_path in wavs:
                # Comprobamos si ya existe el embedding para saltarlo si quieres (opcional)
                # fname_pt = os.path.basename(wav_path).replace(".wav", ".pt")
                # if os.path.exists(os.path.join(OUTPUT_ROOT, spk_name, fname_pt)): continue

                # ID único: s1_bbaf2n (necesario para recuperar el nombre luego)
                file_id = f"{spk_name}_{os.path.basename(wav_path)}"
                abs_path = os.path.abspath(wav_path)
                
                f_scp.write(f"{file_id} {abs_path}\n")
                audio_count += 1

    if audio_count == 0:
        print("No hay archivos nuevos para procesar.")
        exit()

    print(f"Generado {SCP_FILE} con {audio_count} audios.")

    # 2. CARGAR MODELO WESPEAKER
    # 'english' descarga automáticamente el modelo CAM++ preentrenado en VoxCeleb
    print("Cargando modelo WeSpeaker (CAM++)...")
    model = wespeaker.load_model('english')
    model.set_device('cuda')
    
    # ¡IMPORTANTE! Activar GPU
    
    # 3. EXTRACCIÓN MASIVA (La parte eficiente)
    # WeSpeaker leerá el scp y procesará internamente (con su propio DataLoader C++)
    print("Extrayendo embeddings (esto puede tardar un poco, pero es muy rápido)...")
    
    # devuelve: lista de IDs, lista de Embeddings (numpy arrays)
    embeds_result = model.extract_embedding_list(SCP_FILE)
    
    # Si la versión de wespeaker devuelve una tupla (ids, embs) o lista de tuplas, nos adaptamos
    # La API estándar devuelve (list_of_keys, list_of_embeddings)
    if isinstance(embeds_result, tuple):
        keys, embeddings = embeds_result
    else:
        # Fallback por si cambia la versión
        keys = [x[0] for x in embeds_result]
        embeddings = [x[1] for x in embeds_result]

    print(f"Guardando {len(embeddings)} tensores en disco...")

    # 4. GUARDAR RESULTADOS
    # Recorremos y guardamos cada vector en su archivo .pt correspondiente
    for i, (key, emb) in enumerate(tqdm(zip(keys, embeddings), total=len(keys))):
        # Recuperamos info del ID: s1_bbaf2n.wav -> s1, bbaf2n.pt
        # El key era: s1_bbaf2n.wav
        
        # Ojo: mi ID generado arriba era f"{spk_name}_{os.path.basename(wav_path)}"
        # Ejemplo: s1_bbaf2n.wav
        
        # Separar speaker del archivo
        parts = key.split('_', 1) # s1, bbaf2n.wav
        spk_name = parts[0]
        filename = parts[1].replace(".wav", ".pt")
        
        save_path = os.path.join(OUTPUT_ROOT, spk_name, filename)
        
        # Convertir numpy a torch tensor
        tensor_emb = torch.from_numpy(emb).clone()
        
        torch.save(tensor_emb, save_path)

    # Limpieza
    if os.path.exists(SCP_FILE):
        os.remove(SCP_FILE)
    
    print("¡Proceso completado!")