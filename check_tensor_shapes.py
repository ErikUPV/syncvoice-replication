import torch
import os

DIR = "data/rois"

for dirpath, dirnames, filenames in os.walk(DIR):
    for speaker in dirnames:
        for file in os.listdir(os.path.join(dirpath, speaker)):
            if file.endswith(".pt"):
                tensor = torch.load(os.path.join(dirpath, speaker, file))
                if list(tensor.shape) != [tensor.shape[0], 96, 96]:
                    print(f"{file}: {tensor.dtype}, shape: {tensor.shape}")