from torch.utils.data import Dataset
import requests
import os
import zipfile
import argparse
import tarfile

class GRIDEnDataset(Dataset):
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


