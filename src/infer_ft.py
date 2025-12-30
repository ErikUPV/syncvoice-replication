#!/usr/bin/env python3
"""
Full finetune inference script (no LoRA).

Checkpoint directory contains complete model files (pytorch_model.bin, config.json, audiovae.pth, etc.),
can be loaded directly via VoxCPM.

Usage:

    python scripts/test_voxcpm_ft_infer.py \
        --ckpt_dir /path/to/checkpoints/step_0001000 \
        --text "Hello, I am the finetuned VoxCPM." \
        --output ft_test.wav

With voice cloning:

    python scripts/test_voxcpm_ft_infer.py \
        --ckpt_dir /path/to/checkpoints/step_0001000 \
        --text "Hello, this is voice cloning result." \
        --prompt_audio path/to/ref.wav \
        --prompt_text "Reference audio transcript" \
        --output ft_clone.wav
"""

import argparse
from pathlib import Path

import soundfile as sf

from voxcpm.core import VoxCPM

from voxcpm.training import (
    Accelerator,
    BatchProcessor,
    TrainingTracker,
    build_dataloader,
    load_audio_text_datasets,
)

from tqdm import tqdm
import os
import random as rm
import subprocess
from dataset import parse_grid_align

def parse_args():
    parser = argparse.ArgumentParser("VoxCPM full-finetune inference test (no LoRA)")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Checkpoint directory (contains pytorch_model.bin, config.json, audiovae.pth, etc.)",
    )
    # parser.add_argument(
    #     "--text",
    #     type=str,
    #     required=True,
    #     help="Target text to synthesize",
    # )
    # parser.add_argument(
    #     "--prompt_audio",
    #     type=str,
    #     default="",
    #     help="Optional: reference audio path for voice cloning",
    # )
    # parser.add_argument(
    #     "--prompt_text",
    #     type=str,
    #     default="",
    #     help="Optional: transcript of reference audio",
    # )
    parser.add_argument(
        "--output",
        type=str,
        default="ft_test",
        help="Output directory path",
    )
    parser.add_argument(
        "--cfg_value",
        type=float,
        default=2.0,
        help="CFG scale (default: 2.0)",
    )
    parser.add_argument(
        "--inference_timesteps",
        type=int,
        default=10,
        help="Diffusion inference steps (default: 10)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=600,
        help="Max generation steps",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable text normalization",
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help="Path to the JSONL with validation samples"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model from checkpoint directory (no denoiser)
    print(f"[FT Inference] Loading model: {args.ckpt_dir}")
    model = VoxCPM.from_pretrained(
        hf_model_id=args.ckpt_dir,
        load_denoiser=False,
        optimize=True,
    )

    _, val_ds = load_audio_text_datasets(train_manifest=f"{args.data_root}/train.jsonl",
                                       val_manifest=f"{args.data_root}/valid.jsonl",
                                       sample_rate=44100)
    
    for i, item in enumerate(tqdm(val_ds, desc="Processing samples")):
        print(item)
        # Run inference
        prompt_wav_path = None  # item['audio']
        text = item["text"]
        lip_feats = item["lip_feats"]
        face_feats = item["face_feats"]

        id = lip_feats.split('/')[-1].replace('.pt','')
        speaker = lip_feats.split('/')[-2]

        speaker_dir = f'data/audio/{speaker}'
        chosen_ref = rm.choice([x for x in os.listdir(speaker_dir) if x.endswith('.wav') and not x.startswith(id)])
        prompt_wav_path = f'data/audio/{speaker}/{chosen_ref}'

        prompt_text = parse_grid_align(f"data/align/{speaker}/{chosen_ref.replace('.wav','.align')}")

        # prompt_wav_path = args.prompt_audio if args.prompt_audio else None
        # prompt_text = args.prompt_text if args.prompt_text else None

        print(f"[FT Inference] Synthesizing: text='{text}'")
        if prompt_wav_path:
            print(f"[FT Inference] Using reference audio: {prompt_wav_path}")
            print(f"[FT Inference] Reference text: {prompt_text}")

        audio_np = model.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            lip_path=lip_feats,
            face_path=face_feats,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
            max_len=args.max_len,
            normalize=args.normalize,
            denoise=False,
        )

        out_path = f"{args.output}/{speaker}/{id}.wav"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(str(out_path), audio_np, model.tts_model.sample_rate)

        # Combine with original video
        command = f"ffmpeg -i data/unprocessed/{speaker}/{id}.mpg -i {out_path} -c:v copy -map 0:v:0 -map 1:a:0 -shortest {out_path.replace('wav', '')}_2.mp4 -y"
        subprocess.run(command, shell=True, check=True)

        print(f"[FT Inference] Saved to: {out_path}, duration: {len(audio_np) / model.tts_model.sample_rate:.2f}s")


if __name__ == "__main__":
    main()
