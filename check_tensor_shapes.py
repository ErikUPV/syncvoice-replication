import torch
import os
from tqdm import tqdm
DIR = "data/rois"

for dirpath, dirnames, filenames in os.walk(DIR):
    for speaker in dirnames:
        for file in tqdm(os.listdir(os.path.join(dirpath, speaker)), desc=f"Checking tensors for speaker {speaker}"):
            if file.endswith(".pt"):
                tensor = torch.load(os.path.join(dirpath, speaker, file))
                if list(tensor.shape) != [tensor.shape[0], 96, 96]:
                    print(f"{file}: {tensor.dtype}, shape: {tensor.shape}")