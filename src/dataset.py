from torch.utils.data import Dataset
import requests
import os
import zipfile
import argparse
import tarfile

import os
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import random

def parse_grid_align(align_path):
    """
    Lee un archivo .align de GRID y extrae el texto limpio.
    Formato: start_time end_time word
    Ejemplo: 0 10250 sil -> Ignorar
             10250 16500 place -> 'place'
    """
    words = []
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: 
                continue # Línea malformada
            
            # El tercer elemento es la palabra
            word = parts[2]
            
            # Filtramos silencios ('sil') y pausas cortas ('sp')
            if word not in ['sil', 'sp']:
                words.append(word)
                
    # Unimos con espacios y normalizamos (opcional: upper() o lower())
    return " ".join(words).upper()


class GRIDEnTrainDataset(Dataset):
    def __init__(self, data_root: str):
        self.data_root = data_root
        # Rutas base
        self.dirs = {
            "audio": os.path.join(data_root, "audio"),
            "rois": os.path.join(data_root, "rois"),
            "face": os.path.join(data_root, "face_feats"),
            "align": os.path.join(data_root, "align"),
            "spk_emb": os.path.join(data_root, "speaker_embeddings")
        }
        
        self.samples = []
        self.resampler = T.Resample(orig_freq=50000, new_freq=48000)

        # 1. Escaneo Seguro: Usamos el audio como ancla
        # Solo speakers válidos (s1-s34, skip s21)
        speakers = [f"s{i}" for i in range(1, 35) if i != 21]

        print("Indexando dataset...")
        
        for spk in speakers:
            spk_audio_dir = os.path.join(self.dirs["audio"], spk)
            if not os.path.isdir(spk_audio_dir): continue
            
            # Listamos archivos y ordenamos para consistencia
            wavs = sorted([f for f in os.listdir(spk_audio_dir) if f.endswith(".wav")])
            
            for wav_file in wavs:
                # ID base: bbaf2n (sin extensión)
                file_id = os.path.splitext(wav_file)[0]
                
                # Construimos la entrada solo si existen todos los componentes
                # Asumimos estructura: data_root/align/s1/bbaf2n.align
                align_path = os.path.join(self.dirs["align"], spk, file_id + ".align")
                
                # Verificación opcional (puedes comentarla para acelerar el inicio si confías en tus datos)
                if os.path.exists(align_path):
                    self.samples.append({
                        "spk": spk,
                        "id": file_id,
                        "audio_path": os.path.join(spk_audio_dir, wav_file),
                        "align_path": align_path,
                    })

        print(f"Dataset cargado: {len(self.samples)} muestras válidas.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        spk = item["spk"]
        file_id = item["id"]
        
        # 1. TEXTO (Post-procesado aquí)
        text = parse_grid_align(item["align_path"])
        
        # 2. AUDIO
        # Carga y resampleo si es necesario (mejor hacerlo offline, pero aquí por seguridad)
        audio, _ = torchaudio.load(item["audio_path"]) # constant sampling rate
        audio = self.resampler(audio)

        # 3. VISUAL FEATURES (Calculamos rutas)
        # Nota: Tus scripts anteriores guardaban .pt, asegúrate de la extensión
        roi_path = os.path.join(self.dirs["rois"], spk, file_id + ".pt")
        face_path = os.path.join(self.dirs["face"], spk, file_id + ".pt")
        spk_emb_path = os.path.join(self.dirs["spk_emb"], spk)
        
        roi = torch.load(roi_path)
        face_feats = torch.load(face_path)
        speaker_embedding = random.choice([x for x in os.listdir(spk_emb_path) if x.endswith(".pt")])

        return audio, roi, face_feats, speaker_embedding
            


    def __len__(self):
        return len(self.audio_files) # Total number of samples in GRID
    
    def __getitem__(self, idx):
        # Implement loading logic here
        pass



# example: "https://spandh.dcs.shef.ac.uk/gridcorpus/s1/video/s1.mpg_vcd.zip"
def download_grid_dataset(output_dir: str, force_download: bool = False):
    output_dir = os.path.join(output_dir, "unprocessed")
    for i in range(1, 35):
        if i == 21: continue # Skip speaker 21 (no data)
        speaker_id = f"s{i}"
        url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/{speaker_id}/video/{speaker_id}.mpg_vcd.zip"
        speaker_dir = os.path.join(output_dir, speaker_id)
        if os.path.exists(speaker_dir) and not force_download:
            print(f"{speaker_dir} already exists, skipping download.")
            continue
        os.makedirs(speaker_dir, exist_ok=True)
        zip_path = os.path.join(speaker_dir, f"{speaker_id}.mpg_vcd.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading {url}...")
            response = requests.get(url)
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(speaker_dir)
            os.remove(zip_path)
        else:
            print(f"{zip_path} already exists, skipping download.")


# example: https://spandh.dcs.shef.ac.uk/gridcorpus/s1/align/s1.tar
def download_grid_alignments(output_dir: str, force_download: bool = False):
    output_dir = os.path.join(output_dir, "align")
    for i in range(1, 35):
        if i == 21: continue # Skip speaker 21 (no data)
        speaker_id = f"s{i}"
        url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/{speaker_id}/align/{speaker_id}.tar"
        speaker_dir = os.path.join(output_dir, speaker_id)
        if os.path.exists(speaker_dir) and not force_download:
            print(f"{speaker_dir} already exists, skipping download.")
            continue
        os.makedirs(speaker_dir, exist_ok=True)
        tar_path = os.path.join(speaker_dir, f"{speaker_id}.tar")
        if not os.path.exists(tar_path):
            print(f"Downloading {url}...")
            response = requests.get(url)
            with open(tar_path, 'wb') as f:
                f.write(response.content)
            print(f"Extracting {tar_path}...")
            with tarfile.open(tar_path, 'r') as tar_ref:
                tar_ref.extractall(speaker_dir)
            os.remove(tar_path)
        else:
            print(f"{tar_path} already exists, skipping download.")

# example: https://spandh.dcs.shef.ac.uk/gridcorpus/s1/audio/s1_50kHz.tar
def download_grid_audio(output_dir: str, force_download: bool = False):
    output_dir = os.path.join(output_dir, "audio")
    for i in range(1, 35):
        if i == 21: continue # Skip speaker 21 (no data)
        speaker_id = f"s{i}"
        url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/{speaker_id}/audio/{speaker_id}_50kHz.tar"
        speaker_dir = os.path.join(output_dir, speaker_id)
        if os.path.exists(speaker_dir) and not force_download:
            print(f"{speaker_dir} already exists, skipping download.")
            continue
        os.makedirs(speaker_dir, exist_ok=True)
        tar_path = os.path.join(speaker_dir, f"{speaker_id}_50kHz.tar")
        if not os.path.exists(tar_path):
            print(f"Downloading {url}...")
            response = requests.get(url)
            with open(tar_path, 'wb') as f:
                f.write(response.content)
            print(f"Extracting {tar_path}...")
            with tarfile.open(tar_path, 'r') as tar_ref:
                tar_ref.extractall(speaker_dir)
            os.remove(tar_path)
        else:
            print(f"{tar_path} already exists, skipping download.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GRID dataset")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the GRID dataset')
    args = parser.parse_args()
    download_grid_dataset(args.output_dir)
    download_grid_alignments(args.output_dir)
    download_grid_audio(args.output_dir)


