import pandas as pd
import argparse
import json
import os
import soundfile as sf
from tqdm.auto import tqdm

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

def save_to_jsonl(dataframe, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in dataframe.iterrows():
            line_data = {
                "audio": args.prefix + str(row["audio_file"]), 
                "text": row["text"], 
                "duration": row["duration"]
            }
            json_line = json.dumps(line_data, ensure_ascii=False)
            f.write(json_line + '\n')

def get_audio_duration(path):
    return sf.info(path).duration


parser = argparse.ArgumentParser()
# parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
# parser.add_argument('--prefix', type=str, default='', help='Prefix to add to audio file paths')
# parser.add_argument('--hf_dataset_path', type=str, default=None, help='Hugging Face dataset path')
# parser.add_argument('--accents', nargs='*', default=None, help='List of accents to filter by')
parser.add_argument('--output_dir', type=str, required=True, help="Directory in which to save the data")
parser.add_argument('--data_root', type=str, required=True, help='Root directory for audio files')
parser.add_argument('--valid_samples', type=int, help="Number of samples for the validation split (default: 200)", default=200)

args = parser.parse_args()

ids = []

# Get all ids from the rois directory (for example)
for spkr in os.listdir(f"{args.data_root}/rois"):
    if not spkr.startswith('s'): continue

    # get spkr and raw id without .pt
    ids.extend([(id.split('.')[0], spkr) for id in os.listdir(f"{args.data_root}/rois/{spkr}") if id.endswith(".pt")]) 

ids = set(ids)

samples = []
dirs =  ['align', 'audio', 'face_feats', 'rois', 'speaker_embeddings']
for id, spkr in tqdm(ids, desc="Processing samples"):
        samples.append({
            "text" : parse_grid_align(f"{args.data_root}/align/{spkr}/{id}.align"),
            "audio" : f"{args.data_root}/audio/{spkr}/{id}.wav",
            "duration" : get_audio_duration(f"{args.data_root}/audio/{spkr}/{id}.wav"),
            "lip_feats" : f"{args.data_root}/rois/{spkr}/{id}.pt",
            "face_feats" : f"{args.data_root}/face_feats/{spkr}/{id}.pt",
            "spk_embs" : f"{args.data_root}/speaker_embeddings/{spkr}/{id}.pt",
        })

# separate train and valid
train_df = pd.DataFrame(samples[:-args.valid_samples])
valid_df = pd.DataFrame(samples[-args.valid_samples:])

os.makedirs(args.output_dir, exist_ok=True)

save_to_jsonl(train_df, f"{args.output_dir}/train.jsonl")
save_to_jsonl(valid_df, f"{args.output_dir}/valid.jsonl")

print(f"Saved all samples to {args.output_dir}!")
print(f"Here's a training sample:\n {train_df.head(1)}")

