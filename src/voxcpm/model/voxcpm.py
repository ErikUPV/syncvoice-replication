"""
VoxCPM: A Tokenizer-free speech generation model

This module contains the main VoxCPM model implementation, including configuration classes
and the core VoxCPMModel for text-to-speech generation.

Copyright 2025 OpenBMB
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from typing import Tuple, Union, Generator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import warnings
from einops import rearrange
from pydantic import BaseModel
from voxcpm.modules.audiovae import audio_vae

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
from tqdm import tqdm
from transformers import LlamaTokenizerFast

from ..modules.audiovae import AudioVAE, AudioVAEConfig
from ..modules.layers import ScalarQuantizationLayer
from ..modules.layers.lora import apply_lora_to_named_linear_modules
from ..modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiT
from ..modules.locenc import VoxCPMLocEnc
from ..modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from .utils import get_dtype, mask_multichar_chinese_tokens
from fusion.modules import LipEncoder, VisualAdapter, VisualAdapterConfig


class VoxCPMEncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: Optional[int] = None


class VoxCPMDitConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: Optional[int] = None

    cfm_config: CfmConfig


class VoxCPMConfig(BaseModel):
    lm_config: MiniCPM4Config
    patch_size: int = 2
    feat_dim: int = 64
    residual_lm_num_layers: int = 6
    scalar_quantization_latent_dim: int = 256
    scalar_quantization_scale: int = 9
    visual_resample_mode: str = "avgpooling"

    va_config: VisualAdapterConfig
    encoder_config: VoxCPMEncoderConfig
    dit_config: VoxCPMDitConfig
    audio_vae_config: Optional[AudioVAEConfig] = None

    max_length: int = 4096
    device: str = "cuda"
    dtype: str = "bfloat16"
    dit_mean_mode: bool = False


class LoRAConfig(BaseModel):
    enable_lm: bool = False        # Apply LoRA to base_lm + residual_lm
    enable_dit: bool = False       # Apply LoRA to VoxCPMLocDiT
    enable_proj: bool = False      # Apply LoRA to projection Linear layers

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0

    # Target linear layer names for LM & DiT (matched by attribute name)
    target_modules_lm: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    target_modules_dit: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    # Projection layer attribute names to find on VoxCPMModel
    target_proj_modules: list[str] = ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"]


VoxCPMConfig.model_rebuild()


class VoxCPMModel(nn.Module):
    def __init__(
        self,
        config: VoxCPMConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAE,
        lora_config: LoRAConfig = None,
    ):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.device = config.device
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        print(f"Running on device: {self.device}, dtype: {self.config.dtype}")

        # Text-Semantic LM
        self.base_lm = MiniCPMModel(config.lm_config)
        self.base_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101
        self.audio_end_token = 102

        # Residual Acoustic LM
        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        self.residual_lm = MiniCPMModel(residual_lm_config)
        self.residual_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        # Local Encoder
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiT(decoder_config, in_channels=config.feat_dim),
            mean_mode=config.dit_mean_mode,
        )

        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size, 
            config.lm_config.hidden_size, 
            config.scalar_quantization_latent_dim, 
            config.scalar_quantization_scale
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)

        # Stop Predictor
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)
        self.stop_loss = nn.CrossEntropyLoss(reduction="none")

        # Audio VAE
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.chunk_size
        self.sample_rate = audio_vae.sample_rate
        self.audio_vae.eval()

        # Visual Adapter
        self.lip_encoder = LipEncoder(out_dim=config.va_config.lip_dim)
        
        # 2. Visual Adapter (Fuses Lip + Face -> Text Dimension)
        # Note: We calculate total visual dim = lip_dim + face_dim
        visual_input_dim = config.va_config.lip_dim + config.va_config.face_dim
        
        self.visual_adapter = VisualAdapter(
            visual_dim=visual_input_dim,          # Corrected to match modules.py signature
            text_dim=config.lm_config.hidden_size, # Project to model's hidden dim
            bottleneck_dim=config.va_config.bottleneck_dim
        )

        # self.vis_gate = nn.Parameter(torch.tensor(-4.0))


        # self.multimodal_fusion_proj = nn.Linear(config.dit_config.hidden_dim, config.dit_config.hidden_dim)
        # nn.init.zeros_(self.multimodal_fusion_proj.weight)
        # nn.init.zeros_(self.multimodal_fusion_proj.bias)

        if self.lora_config is not None:
            self._apply_lora()

    def _apply_lora(self):
        """注入 LoRA 到 LM / DiT / 投影层"""
        cfg = self.lora_config
        lora_kwargs = dict(r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)

        # LM: base_lm + residual_lm
        if cfg.enable_lm:
            for lm in [self.base_lm, self.residual_lm]:
                apply_lora_to_named_linear_modules(
                    lm, target_submodule_names=cfg.target_modules_lm, **lora_kwargs
                )

        # DiT: feat_decoder.estimator
        if cfg.enable_dit:
            apply_lora_to_named_linear_modules(
                self.feat_decoder.estimator, target_submodule_names=cfg.target_modules_dit, **lora_kwargs
            )

        # 投影层
        if cfg.enable_proj:
            from ..modules.layers.lora import LoRALinear
            for attr_name in cfg.target_proj_modules:
                module = getattr(self, attr_name, None)
                if isinstance(module, nn.Linear):
                    setattr(self, attr_name, LoRALinear(base=module, **lora_kwargs))

    def optimize(self, disable: bool = False):
        if disable:
            return self
        try:
            if self.device != "cuda":
                raise ValueError("VoxCPMModel can only be optimized on CUDA device")
            try:
                import triton
            except:
                raise ValueError("triton is not installed")
            self.base_lm.forward_step = torch.compile(self.base_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.residual_lm.forward_step = torch.compile(self.residual_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.feat_encoder = torch.compile(self.feat_encoder, mode="reduce-overhead", fullgraph=True)
            self.feat_decoder.estimator = torch.compile(self.feat_decoder.estimator, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            print(f"Warning: torch.compile disabled - {e}")
        return self

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        lip_feats: torch.Tensor,   # [B, T_video, 96, 96]
        face_feats: torch.Tensor,  # [B, T_video, 512]
        *,
        progress: float = 0.0,
        sample_generate: bool = False,
    ):
        del position_ids  # not used yet

        text_tokens = text_tokens.to(self.device, dtype=torch.long)
        text_mask = text_mask.to(self.device, dtype=self._dtype())
        audio_feats = audio_feats.to(self.device, dtype=self._dtype())
        audio_mask = audio_mask.to(self.device, dtype=self._dtype())
        loss_mask = loss_mask.to(self.device, dtype=self._dtype())
        labels = labels.to(self.device, dtype=torch.long)

        lip_feats = lip_feats.to(self.device, dtype=self._dtype())
        face_feats = face_feats.to(self.device, dtype=self._dtype())


        lip_emb = self.lip_encoder(lip_feats)

        lip_lengths = (torch.sum(torch.any(lip_feats.abs() > 0, dim=(-1, -2)), dim=-1)).to(torch.int32)
        # lip_emb = self._resample_visuals(lip_emb, audio_feats.shape[1], mode='linear')
        # face_feats = self._resample_visuals(face_feats, audio_feats.shape[1], mode='linear')
        
        # 2. Concatenate: [B, T_video, lip_dim + face_dim]
        video_feats = torch.cat([lip_emb, face_feats], dim=-1)
        video_adapted = self.visual_adapter(video_feats)  # [B, T_video, hidden_size]
        
        B, T_total = audio_mask.shape
        visual_cond = torch.zeros((B, T_total, video_adapted.shape[-1]), device=self.device, dtype=video_adapted.dtype)

        for i in range(B):
            n_audio_tokens = int(audio_mask[i].sum().item())
            n_visual_valid = int(lip_lengths[i].item()) # Longitud REAL del video

            # RECORTAR ANTES DE RESAMPLEAR
            # Solo tomamos los frames válidos
            curr_vid = video_adapted[i, :n_visual_valid, :]

            # curr_vid = video_adapted[i] # [T_video, hidden_size]

            curr_vid = self._resample_visuals(
                curr_vid.unsqueeze(0),
                n_audio_tokens,
                mode=self.config.visual_resample_mode,
            ).squeeze(0)
            audio_start_idx = (audio_mask[i] == 1).nonzero(as_tuple=True)[0][0]
            visual_cond[i, audio_start_idx:audio_start_idx + n_audio_tokens, :] = curr_vid
        

        B, T, P, D = audio_feats.shape
        feat_embed = self.feat_encoder(audio_feats)
        feat_embed = self.enc_to_lm_proj(feat_embed)

        scale_emb = getattr(self.config.lm_config, "scale_emb", 1.0)
        if not getattr(self.config.lm_config, "use_mup", False):
            scale_emb = 1.0
        text_embed = self.base_lm.embed_tokens(text_tokens) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed

        enc_outputs, _ = self.base_lm(inputs_embeds=combined_embed, is_causal=True)
        enc_outputs = enc_outputs.to(self._dtype())
        enc_outputs = self.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = torch.cat((torch.zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1)

        residual_inputs = enc_outputs + audio_mask.unsqueeze(-1) * feat_embed
        residual_outputs, _ = self.residual_lm(inputs_embeds=residual_inputs, is_causal=True)
        residual_outputs = residual_outputs.to(self._dtype())
        residual_hidden = torch.cat(
            (torch.zeros_like(residual_outputs[:, 0:1, :]), residual_outputs[:, :-1, :]),
            dim=1,
        )

        dit_hidden = self.lm_to_dit_proj(lm_hidden) + self.res_to_dit_proj(residual_hidden)

        dit_hidden += visual_cond

        # Inject Visual Condition
        

        dit_hidden = rearrange(dit_hidden, "b t c -> (b t) c")

        # Keep diffusion inputs in the same dtype as the model (e.g., bfloat16)
        target_dtype = self._dtype()

        feat_gt = rearrange(audio_feats.to(target_dtype), "b t p d -> (b t) p d")
        feat_cond = torch.cat(
            (torch.zeros_like(audio_feats[:, 0:1, ...]), audio_feats[:, :-1, ...]),
            dim=1,
        )
        feat_cond = rearrange(feat_cond.to(target_dtype), "b t p d -> (b t) p d")

        loss_seq_mask = loss_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        loss_seq_mask = rearrange(loss_seq_mask, "b t p -> (b t) p 1").to(target_dtype)

        diff_loss = self.feat_decoder.compute_loss(
            feat_gt.transpose(1, 2).contiguous(),
            dit_hidden,
            cond=feat_cond.transpose(1, 2).contiguous(),
            tgt_mask=loss_seq_mask.transpose(1, 2).contiguous(),
            progress=progress,
        )

        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
        stop_losses = self.stop_loss(stop_logits.transpose(1, 2), labels)
        denom = torch.clamp(loss_mask.sum(), min=1.0)
        stop_loss = (stop_losses * loss_mask).sum() / denom

        feat_pred = None
        if sample_generate:
            feat_cond_for_sample = feat_cond.transpose(1, 2).contiguous()
            feat_pred_seq = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=feat_cond_for_sample,
                n_timesteps=self.config.dit_config.cfm_config.inference_cfg_rate
                if hasattr(self.config.dit_config.cfm_config, "inference_cfg_rate")
                else 10,
            )
            feat_pred = rearrange(feat_pred_seq.transpose(1, 2), "(b t) d p -> b d (t p)", b=B, p=self.patch_size)

        feat_gt_tensor = rearrange(feat_gt, "(b t) p d -> b d (t p)", b=B, p=self.patch_size)

        return {
            "loss/diff": diff_loss,
            "loss/stop": stop_loss,
            "feat_gt": feat_gt_tensor,
            "feat_pred": feat_pred,
        }

    def _dtype(self):
        return get_dtype(self.config.dtype)


    def generate(self, *args, **kwargs) -> torch.Tensor:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        return self._generate(*args, streaming=True, **kwargs)
    
    def _resample_visuals(self, tensor: torch.Tensor, target_len: int, mode: str = 'nearest') -> torch.Tensor:
        """Resamples [B, T, ...] visual features to [B, target_len, ...]"""
        # Tensor is [B, T, ...].
        src_len = tensor.shape[1]
        if src_len == target_len:
            return tensor
            
        if mode == 'nearest':
            tensor_in = tensor.transpose(1, 2)  # -> [B, D, T]
            tensor_out = F.interpolate(tensor_in, size=target_len, mode='nearest')
            return tensor_out.transpose(1, 2)  # -> [B, target_len, D]
        elif mode == 'avgpooling':
            return self._avg_pool_resample(tensor, target_len)
        elif mode == 'linear':
            # Interpolate expects [N, C, L], input is [B, T, D]
            tensor_in = tensor.transpose(1, 2) # -> [B, D, T]
            tensor_out = F.interpolate(tensor_in, size=target_len, mode='linear', align_corners=False)
            return tensor_out.transpose(1, 2) # -> [B, target_len, D]
        return tensor
    
    def _avg_pool_resample(self, tensor: torch.Tensor, target_len: int, mode: str = 'nearest') -> torch.Tensor:
        """
        Resamples [B, T, D] visual features to [B, target_len, D] using Adaptive Pooling.
        This preserves information from all frames by averaging them into the target bins.
        """
        # tensor: [B, T, D]
        print("Resampling visuals with avg pooling...")
        src_len = tensor.shape[1]
        if src_len == target_len:
            return tensor

        # Prepare for pooling: [B, D, T]
        tensor_in = tensor.transpose(1, 2)
        
        # Adaptive Avg Pool automatically calculates the window size and stride 
        # to squeeze T frames into target_len frames.
        tensor_out = F.adaptive_avg_pool1d(tensor_in, target_len)
        
        return tensor_out.transpose(1, 2) # -> [B, target_len, D]

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        lip_feats: torch.Tensor = None,
        face_feats: torch.Tensor = None,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0, # setting acceptable ratio of audio length to text length (for badcase detection)
        streaming: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False
        if len(prompt_wav_path) == 0:
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor(
                        [self.audio_start_token],
                        dtype=torch.int32,
                        device=text_token.device,
                    ),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_mask = torch.ones(text_length).type(torch.int32).to(text_token.device)
            audio_mask = torch.zeros(text_length).type(torch.int32).to(text_token.device)

        else:
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio, sr = torchaudio.load(prompt_wav_path)
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)    

            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

            patch_len = self.patch_size * self.chunk_size

            if audio.size(1) % patch_len != 0:
                # 左填充：在音频开头填充，保持有效音频数据在序列末尾
                padding_size = patch_len - audio.size(1) % patch_len
                audio = torch.nn.functional.pad(audio, (padding_size, 0))

            # (B, D, T)
            audio_feat = self.audio_vae.encode(audio.to(self.device), self.sample_rate).cpu()
            audio_feat = audio_feat.view(
                self.audio_vae.latent_dim,
                -1,
                self.patch_size,
            ).permute(1, 2, 0)
            audio_length = audio_feat.size(0)
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            text_token = torch.cat([text_token, text_pad_token])
            audio_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            audio_feat = torch.cat([audio_pad_feat, audio_feat], dim=0)
            text_mask = (
                torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
            )
            audio_mask = (
                torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)
            )

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)

        target_text_length = len(self.text_tokenizer(target_text))
        
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len), # avoid too long audio
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
            )
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, _ in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield decode_audio
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...")
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break   
                
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32)).squeeze(1).cpu()  
            yield decode_audio        
    
    @torch.inference_mode()
    def build_prompt_cache(
        self,
        prompt_text: str,
        prompt_wav_path: str,
    ):
        """
        Build prompt cache for subsequent fast generation.
        
        Args:
            prompt_text: prompt text (required)
            prompt_wav_path: prompt audio path (required)
            
        Returns:
            prompt_cache: dict with prompt_text (raw text) and audio features.
                         Text tokenization will be done during generation for consistency.
        """
        if not prompt_text or not prompt_wav_path:
            raise ValueError("prompt_text and prompt_wav_path are required")

        # load audio
        audio, sr = torchaudio.load(prompt_wav_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        patch_len = self.patch_size * self.chunk_size

        if audio.size(1) % patch_len != 0:
            # Left padding: pad at the beginning of the audio to keep valid audio data at the end of the sequence
            padding_size = patch_len - audio.size(1) % patch_len
            audio = torch.nn.functional.pad(audio, (padding_size, 0))

        # extract audio features
        audio_feat = self.audio_vae.encode(audio.to(self.device), self.sample_rate).cpu()

        audio_feat = audio_feat.view(
            self.audio_vae.latent_dim,
            -1,
            self.patch_size,
        ).permute(1, 2, 0) # (D, T, P)
        # build prompt cache - only save raw text and audio features
        prompt_cache = {
            "prompt_text": prompt_text,
            "audio_feat": audio_feat,
        }
        
        return prompt_cache

    
    def merge_prompt_cache(
        self,
        original_cache: dict,
        new_text: str,
        new_audio_feat: torch.Tensor,
    ):
        """
        Merge original prompt cache with newly generated content to stabilize voice.
        
        Args:
            original_cache: original prompt cache
            new_text: newly generated text 
            new_audio_feat: newly generated audio features
            
        Returns:
            merged_cache: merged cache with prompt_text and audio_feat
        """
        if original_cache is None:
            return {
                "prompt_text": new_text,
                "audio_feat": new_audio_feat,
            }
        original_prompt_text = original_cache["prompt_text"]
        original_audio_feat = original_cache["audio_feat"]
        # Merge text by concatenation
        merged_prompt_text = original_prompt_text + new_text
        merged_audio_feat = torch.cat([original_audio_feat, new_audio_feat], dim=0)

        # build new cache
        merged_cache = {
            "prompt_text": merged_prompt_text,
            "audio_feat": merged_audio_feat,
        }
        
        return merged_cache

            
    def generate_with_prompt_cache(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return next(self._generate_with_prompt_cache(*args, streaming=False, **kwargs))


    def generate_with_prompt_cache_streaming(
        self, *args, **kwargs
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], None, None]:
        return self._generate_with_prompt_cache(*args, streaming=True, **kwargs)


    @torch.inference_mode()
    @torch.inference_mode()
    def _generate_with_prompt_cache(
        self,
        target_text: str,
        prompt_cache: dict,
        # --- NEW: Visual Inputs ---
        lip_feats: torch.Tensor = None,
        face_feats: torch.Tensor = None,
        # --------------------------
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
        streaming_prefix_len: int = 3,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        Generate audio using pre-built prompt cache.
        
        Args:
            target_text: Text to convert to speech
            prompt_cache: Cache built by build_prompt_cache (can be None)
            min_len: Minimum audio length to avoid very short audio
            max_len: Maximum audio length
            inference_timesteps: Number of diffusion sampling steps
            cfg_value: Classifier-free guidance value
            retry_badcase: Whether to retry on bad cases
            retry_badcase_max_times: Maximum retry attempts
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio
            streaming: Whether to return a generator of audio chunks
            streaming_prefix_len: Number of prefix audio patches to use for streaming mode
            
        Returns:
            Generator of Tuple containing:
                - Decoded audio tensor for the current step if ``streaming=True``, else final decoded audio tensor
                - Tensor of new text tokens
                - New audio features up to the current step as a List if ``streaming=True``, else as a concatenated Tensor
        """
        
        # 1. Disable badcase retry if visuals are provided (Length is fixed by video)
        # if lip_feats is not None:
        #     max_len = lip_feats.shape[1]
        #     min_len = lip_feats.shape[1]
        #     retry_badcase = False

        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False

        # 2. Prepare Prompt Cache (Same as original)
        if prompt_cache is None:
            prompt_audio_feat = torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32)
            text = target_text
        else:
            prompt_audio_feat = prompt_cache["audio_feat"]
            prompt_text = prompt_cache["prompt_text"]
            text = prompt_text + target_text
        
        # 3. Tokenize Text (Same as original)
        text_token = torch.LongTensor(self.text_tokenizer(text))
        text_token = torch.cat(
            [text_token, torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device)],
            dim=-1,
        )
        target_text_token = torch.LongTensor(self.text_tokenizer(target_text))

        # 4. Prepare Audio/Text Masks (Same as original)
        audio_length = prompt_audio_feat.size(0)
        text_length = text_token.shape[0]
        text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
        audio_pad_feat = torch.zeros(
            (text_token.shape[0], self.patch_size, self.audio_vae.latent_dim),
            dtype=torch.float32,
            device=text_token.device,
        )
        text_token = torch.cat([text_token, text_pad_token])
        audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
        
        text_mask = torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
        audio_mask = torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)
    
        # 5. Run Inference Loop
        target_text_length = len(self.text_tokenizer(target_text))
        retry_badcase_times = 0
        
        while retry_badcase_times < retry_badcase_max_times:
            # --- UPDATED CALL: Pass visual features to _inference ---
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                # Pass visuals here
                lip_feats=lip_feats,
                face_feats=face_feats,
                # ----------------
                min_len=min_len,
                max_len=max_len if lip_feats is not None else min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            
            if streaming:
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, pred_audio_feat in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                    yield (decode_audio, target_text_token, pred_audio_feat)
                break
            else:
                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(f"  Badcase detected, retrying...")
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break
                    
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            patch_len = self.patch_size * self.chunk_size
            if audio_mask.sum().item() > 0:
                decode_audio = decode_audio[..., patch_len * (streaming_prefix_len - 1):].squeeze(1).cpu()
            else:
                decode_audio = decode_audio[..., :].squeeze(1).cpu()
            yield (decode_audio, target_text_token, pred_audio_feat)

    

    # def inference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    #     return next(self._inference(*args, streaming=False, **kwargs))
    
    def inference_streaming(self, *args, **kwargs) -> Generator[Tuple[torch.Tensor, List[torch.Tensor]], None, None]:
        return self._inference(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        lip_feats: torch.Tensor,
        face_feats: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 3,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """Core inference method for audio generation.
        
        This is the main inference loop that generates audio features
        using the language model and diffusion transformer.
        
        Args:
            text: Input text tokens
            text_mask: Mask for text tokens
            feat: Input audio features
            feat_mask: Mask for audio features
            min_len: Minimum generation length
            max_len: Maximum generation length
            inference_timesteps: Number of diffusion steps
            cfg_value: Classifier-free guidance value
            streaming: Whether to yield each step latent feature or just the final result
            
        Returns:
            Generator of Tuple containing:
                - Predicted latent feature at the current step if ``streaming=True``, else final latent features
                - Predicted audio feature sequence so far as a List if ``streaming=True``, else as a concatenated Tensor
        """
        B, T, P, D = feat.shape

        feat_embed = self.feat_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)
        
        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
            scale_emb = 1.0
       
        text_embed = self.base_lm.embed_tokens(text) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        prefix_feat_cond = feat[:, -1, ...]  # b, p, d
        pred_feat_seq = []  # b, t, p, d
        curr_embed = None

        visual_cond_seq = None
        if lip_feats is not None and face_feats is not None:
            # 1. Encode
            lip_emb = self.lip_encoder(lip_feats.to(self.device, dtype=self._dtype()))
            face_feats = face_feats.to(self.device, dtype=self._dtype())
            # 2. Concat & Adapt
            video_feats = torch.cat([lip_emb, face_feats], dim=-1)
            visual_cond_seq = self.visual_adapter(video_feats) # [B, T_vis, H]

            # 3. Resample
            audio_fps = self.audio_vae.sample_rate / self.audio_vae.hop_length
            scale_factor = audio_fps / 25.0 
            target_len = int(visual_cond_seq.shape[1] * scale_factor)  // self.patch_size
            visual_cond_seq = self._resample_visuals(
                visual_cond_seq,
                target_len=target_len,
                mode=self.config.visual_resample_mode,
            )
            
            # The length of generation is strictly the length of visual_cond_seq
            max_len = visual_cond_seq.shape[1]
        

        # Prepare prompt context patches for streaming mode
        # When there's a prompt audio, use its last (streaming_prefix_len - 1) patches as initial context
        prompt_context_patches = []
        audio_patch_count = int(feat_mask.sum().item())
        if audio_patch_count > 0:
            context_len = min(streaming_prefix_len - 1, audio_patch_count)
            # Take the last context_len patches from prompt audio as initial context
            # Split into list of [b, 1, p, d] tensors to match pred_feat_seq format
            prompt_context_patches = list(feat[:, -context_len:, :, :].split(1, dim=1))
            pred_feat_seq = prompt_context_patches + pred_feat_seq

        enc_outputs, kv_cache_tuple = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        self.base_lm.kv_cache.fill_caches(kv_cache_tuple)
        
        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

         
        residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
            inputs_embeds=enc_outputs + feat_mask.unsqueeze(-1) * feat_embed,
            is_causal=True,
        )
        self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]


        for i in tqdm(range(max_len)):
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
            dit_hidden = dit_hidden_1 + dit_hidden_2  # [b, h_dit]

            if visual_cond_seq is not None:
                # Get visual condition for current step i
                if i < visual_cond_seq.shape[1]:
                    curr_vis = visual_cond_seq[:, i, :] # [B, H]
                    dit_hidden += curr_vis * 2  # scale visual condition

                    print(f"lm_to_dit magnitude: {dit_hidden_1.abs().mean().item()}")
                    print(f"res_to_dit magnitude: {dit_hidden_2.abs().mean().item()}")
                    print("visual_cond mean abs:", curr_vis.abs().mean().item())
            
                    # print(f"Magnitudes:\n LM to DIT: {dit_hidden_1.abs().mean().item():.4f}, Residual to DIT: {dit_hidden_2.abs().mean().item():.4f}, Visual Cond: {curr_vis.abs().mean().item():.4f}")
                else:
                    # Should not happen if max_len is set correctly
                    pass

            # Sumar la condición visual al contexto del texto/audio

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(
                1, 2
            )  # [b, p, d]
            
            curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
            curr_embed = self.enc_to_lm_proj(curr_embed)
            
            pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
            prefix_feat_cond = pred_feat

            if streaming:
                # return the last three predicted latent features to provide enough context for smooth decoding
                pred_feat_chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)
                feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                
                yield feat_pred, pred_feat_seq
            
            if visual_cond_seq is None:
                stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
                if i > min_len and stop_flag == 1:
                    break
                
    
            lm_hidden = self.base_lm.forward_step(
                curr_embed[:, 0, :], torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()
           

            lm_hidden = self.fsq_layer(lm_hidden)
            residual_hidden = self.residual_lm.forward_step(
                lm_hidden + curr_embed[:, 0, :], torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()
                
        if not streaming:
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)  
            yield feat_pred, pred_feat_seq.squeeze(0).cpu()
            

    @classmethod
    def from_local(cls, path: str, optimize: bool = True, training: bool = False, lora_config: LoRAConfig = None):
        config = VoxCPMConfig.model_validate_json(open(os.path.join(path, "config.json")).read())
        tokenizer = LlamaTokenizerFast.from_pretrained(path)
        audio_vae_config = getattr(config, 'audio_vae_config', None)
        audio_vae = AudioVAE(config=audio_vae_config) if audio_vae_config else AudioVAE()
        vae_state_dict = torch.load(
            os.path.join(path, "audiovae.pth"),
            map_location="cpu",
            weights_only=True,
        )["state_dict"]
        model = cls(config, tokenizer, audio_vae, lora_config)
        if not training:
            lm_dtype = get_dtype(model.config.dtype)
            model = model.to(lm_dtype)
        else: # training mode
            for name, param in model.named_parameters():
                if "audio_vae" in name: # freeze VAE weights
                    param.requires_grad = False
                    continue
                # for param in model.base_lm.parameters(): # freeze main LM weights
                #     param.requires_grad = False
                # for param in model.residual_lm.parameters(): # freeze residual and main LM weights
                #     param.requires_grad = False
                if lora_config is not None:
                    if "lora" not in name: # freeze non-LoRA weights
                        param.requires_grad = False
        model.audio_vae = model.audio_vae.to(torch.float32)
        
        # Try to load from safetensors first, fallback to pytorch_model.bin
        safetensors_path = os.path.join(path, "model.safetensors")
        pytorch_model_path = os.path.join(path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading model from safetensors: {safetensors_path}")
            model_state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_model_path):
            print(f"Loading model from pytorch_model.bin: {pytorch_model_path}")
            checkpoint = torch.load(
                pytorch_model_path,
                map_location="cpu",
                weights_only=True,
            )
            model_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(
                f"Model file not found. Expected either {safetensors_path} or {pytorch_model_path}"
            )
            
        for kw, val in vae_state_dict.items():
            model_state_dict[f"audio_vae.{kw}"] = val
        
        # LoRALinear holds weight/bias directly, compatible with nn.Linear state_dict keys.
        # Using strict=False since pretrained weights don't contain lora_A/lora_B.
        model.load_state_dict(model_state_dict, strict=False)
        if training:
            return model
        return model.to(model.device).eval().optimize(disable=not optimize)

    # ------------------------------------------------------------------ #
    # LoRA Weight Management
    # ------------------------------------------------------------------ #
    def _iter_lora_modules(self):
        """Iterate over all LoRA modules."""
        from ..modules.layers.lora import LoRALinear
        for module in self.modules():
            if isinstance(module, LoRALinear):
                yield module

    def load_lora_weights(self, lora_path: str, device: str = None):
        """
        Load LoRA weights from file, supports calling after torch.compile.
        Uses named_parameters() to handle compile's _orig_mod wrapper.
        Supports both safetensors and pytorch formats.
        
        Args:
            lora_path: Checkpoint path (directory or .safetensors/.ckpt file)
            device: Target device, defaults to model's current device
        Returns:
            tuple: (loaded_keys, skipped_keys)
        """
        from pathlib import Path
        
        device = device or self.device
        lora_path = Path(lora_path)
        
        # Try safetensors first, then fallback to .ckpt
        if lora_path.is_dir():
            safetensors_file = lora_path / "lora_weights.safetensors"
            ckpt_file = lora_path / "lora_weights.ckpt"
        else:
            safetensors_file = lora_path if lora_path.suffix == ".safetensors" else None
            ckpt_file = lora_path if lora_path.suffix in [".ckpt", ".pth"] else None
        
        # Load from safetensors if available
        if safetensors_file and safetensors_file.exists() and SAFETENSORS_AVAILABLE:
            state_dict = load_file(str(safetensors_file), device=device)
        elif ckpt_file and ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            raise FileNotFoundError(
                f"LoRA checkpoint not found. Expected either {safetensors_file} or {ckpt_file}"
            )
        
        # Build param mapping (handle torch.compile's _orig_mod prefix)
        model_params = dict(self.named_parameters())
        key_mapping = {k.replace("._orig_mod.", "."): k for k in model_params if "._orig_mod." in k}
        
        loaded_keys, skipped_keys = [], []
        for key, value in state_dict.items():
            target_key = key if key in model_params else key_mapping.get(key)
            if target_key:
                model_params[target_key].data.copy_(value.to(device))
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        
        return loaded_keys, skipped_keys

    def set_lora_enabled(self, enabled: bool):
        """Enable/disable all LoRA layers."""
        for module in self._iter_lora_modules():
            module.set_enabled(enabled)

    def reset_lora_weights(self):
        """Reset all LoRA weights (A: kaiming, B: zeros), effectively unloading LoRA."""
        for module in self._iter_lora_modules():
            module.reset_lora_parameters()

    def get_lora_state_dict(self) -> dict:
        """Get all LoRA parameters (lora_A/lora_B)."""
        return {name: param.data.clone() 
                for name, param in self.named_parameters() 
                if "lora_" in name}
