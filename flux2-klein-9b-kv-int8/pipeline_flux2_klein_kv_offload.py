import os, gc, json, torch, numpy as np
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image

import PIL
import torch
from accelerate import init_empty_weights
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKLFlux2
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import logging
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel, 
    Flux2KVAttnProcessor, 
    Flux2KVParallelSelfAttnProcessor
)
from optimum.quanto import requantize
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

logger = logging.get_logger(__name__)

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
    if image_seq_len > 4300: return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)

def _pack_latents(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).permute(0, 2, 1)

def _patchify_latents(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return x.view(B, C, H//2, 2, W//2, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, C*4, H//2, W//2)

def _unpatchify_latents(x: torch.Tensor) -> torch.Tensor:
    B, C_patch, H, W = x.shape
    C = C_patch // 4
    return x.view(B, C, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C, H*2, W*2)

def _prepare_ids(B: int, H: int, W: int, t_offset: int, device: torch.device) -> torch.Tensor:
    t = torch.arange(1, device=device) + t_offset
    h = torch.arange(H, device=device)
    w = torch.arange(W, device=device)
    l = torch.arange(1, device=device)
    return torch.cartesian_prod(t, h, w, l).unsqueeze(0).expand(B, -1, -1)

def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
    B, seq_len, C = x.shape
    out_list = []
    for i in range(B):
        data = x[i]
        pos = x_ids[i]
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        flat_ids = h_ids * width + w_ids
        flat_out = torch.zeros((height * width, C), device=data.device, dtype=data.dtype)
        flat_out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, C), data)
        out_list.append(flat_out.view(height, width, C).permute(2, 0, 1))
    return torch.stack(out_list, dim=0)

class Flux2KleinKVOffloadPipeline(DiffusionPipeline):
    """
    Flux2 Klein KV Pipeline с ручным CPU↔GPU свопом для квантованных моделей.
    Поддерживает reference image conditioning через KV-cache.
    """
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        transformer: Flux2Transformer2DModel,
        default_device: str = "cuda"
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer
        )
        self._default_device = torch.device(default_device)
        self.vae_scale_factor = 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self._set_kv_processors()
        self._current_timestep = None

    def _set_kv_processors(self):
        for block in self.transformer.transformer_blocks:
            block.attn.set_processor(Flux2KVAttnProcessor())
        for block in self.transformer.single_transformer_blocks:
            block.attn.set_processor(Flux2KVParallelSelfAttnProcessor())

    @staticmethod
    def from_quanto(
        model_dir: Union[str, Path],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ) -> "Flux2KleinKVOffloadPipeline":
        """Загружает квантованные модели и собирает пайплайн."""
        model_dir = Path(model_dir).expanduser().resolve()
        torch.backends.cuda.enable_flash_sdp(True)
        
        # 1. Токенизатор & Scheduler & VAE
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(model_dir / "scheduler"), local_files_only=True)
        vae = AutoencoderKLFlux2.from_pretrained(str(model_dir / "vae"), torch_dtype=dtype, local_files_only=True).to(device)
        
        # 2. Text Encoder (quanto)
        cfg_te = AutoConfig.from_pretrained(str(model_dir / "text_encoder"), trust_remote_code=True, local_files_only=True)
        with init_empty_weights():
            text_encoder = AutoModelForCausalLM.from_config(cfg_te, dtype=dtype)
        te_map = json.load(open(model_dir / "text_encoder" / "quantization_map.json"))
        requantize(text_encoder, load_file(str(model_dir / "text_encoder" / "model.safetensors")), te_map, device="cpu")
        text_encoder.eval().to(dtype)
        
        # 3. Transformer (quanto)
        cfg_tr = Flux2Transformer2DModel.load_config(str(model_dir / "transformer"), local_files_only=True)
        with init_empty_weights():
            transformer = Flux2Transformer2DModel.from_config(cfg_tr, torch_dtype=dtype)
        tr_map = json.load(open(model_dir / "transformer" / "quantization_map.json"))
        requantize(transformer, load_file(str(model_dir / "transformer" / "model.safetensors")), tr_map, device="cpu")
        transformer.eval().to(dtype)
        
        gc.collect(); torch.cuda.empty_cache()
        
        pipe = Flux2KleinKVOffloadPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            default_device=device
        )

        pipe.warmup()

        return pipe

    def warmup(self):
        dummy = Image.new("RGB", (16, 16), "gray")
        _ = self(image=dummy, prompt="warmup", height=16, width=16, num_inference_steps=1)
        torch.cuda.synchronize()
        del _
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PIL.Image.Image] = None,
        prompt: Optional[str] = None,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 4,
        generator: Optional[torch.Generator] = None,
        max_sequence_length: int = 512,
    ):
        device = self._default_device

        # Pixel size (multiple of 16). Keep latents derived from the encoded tensor
        # so ref tokens and noise tokens always share the same spatial grid.
        if image is not None:
            pix_h = (int(image.height) // self.vae_scale_factor) * self.vae_scale_factor
            pix_w = (int(image.width) // self.vae_scale_factor) * self.vae_scale_factor
        else:
            pix_h = (int(height) // self.vae_scale_factor) * self.vae_scale_factor
            pix_w = (int(width) // self.vae_scale_factor) * self.vae_scale_factor

        # ═══ 1. Text Encoder ═══
        self.text_encoder.to(device, non_blocking=True)
        txt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inp = self.tokenizer(txt, return_tensors="pt", max_length=max_sequence_length, truncation=True, padding="max_length").to(device)
        out = self.text_encoder(input_ids=inp.input_ids, attention_mask=inp.attention_mask, output_hidden_states=True)
        hs = torch.stack([out.hidden_states[k] for k in (9, 18, 27)], dim=1)
        B, C, L, D = hs.shape
        prompt_embeds = hs.permute(0, 2, 1, 3).reshape(B, L, C * D)
        text_ids = torch.zeros((1, L, 4), device=device, dtype=torch.long)
        text_ids[..., 3] = torch.arange(L, device=device)

        self.text_encoder.to("cpu", non_blocking=True)
        del inp, out, hs
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

        # ═══ 2. Подготовка латентов ═══
        if image is not None:
            img_t = self.image_processor.preprocess(
                image, height=pix_h, width=pix_w, resize_mode="crop"
            ).to(device, dtype=prompt_embeds.dtype)
        else:
            # Neutral ref so t2i keeps the KV path dimensionally consistent.
            img_t = torch.zeros((1, 3, pix_h, pix_w), device=device, dtype=prompt_embeds.dtype)

        m = self.vae.bn.running_mean.view(1, -1, 1, 1).to(prompt_embeds.dtype)
        s = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(prompt_embeds.dtype)

        il = self.vae.encode(img_t).latent_dist.mode()
        il = _patchify_latents(il)
        il = (il - m) / s

        _, C_patch, pack_h, pack_w = il.shape
        il_packed = _pack_latents(il).squeeze(0)

        latents = torch.randn(
            (1, C_patch, pack_h, pack_w),
            generator=generator,
            device=device,
            dtype=prompt_embeds.dtype,
        )
        lat_p = _pack_latents(latents)

        t_ref, t_lat = torch.tensor([10], device=device), torch.tensor([0], device=device)
        h, w, l = (
            torch.arange(pack_h, device=device),
            torch.arange(pack_w, device=device),
            torch.tensor([0], device=device),
        )
        il_ids = torch.cartesian_prod(t_ref, h, w, l).unsqueeze(0)
        lat_ids = torch.cartesian_prod(t_lat, h, w, l).unsqueeze(0)

        inp_m = torch.cat([il_packed.unsqueeze(0), lat_p], dim=1)
        ids_m = torch.cat([il_ids, lat_ids], dim=1)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(
            num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=compute_empirical_mu(lat_p.shape[1], num_inference_steps),
        )
        self.scheduler.set_begin_index(0)
        kv_cache = None

        # ═══ 3. Transformer ═══
        self.transformer.to(device, non_blocking=True)
        try:
            for i, t in enumerate(self.scheduler.timesteps):
                self._current_timestep = t
                ts = t.expand(1).to(prompt_embeds.dtype) / 1000

                if i == 0:
                    pred, kv_cache = self.transformer(
                        hidden_states=inp_m,
                        timestep=ts,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=ids_m,
                        return_dict=False,
                        kv_cache_mode="extract",
                        num_ref_tokens=il_packed.shape[0],
                    )
                else:
                    pred = self.transformer(
                        hidden_states=lat_p,
                        timestep=ts,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=lat_ids,
                        return_dict=False,
                        kv_cache=kv_cache,
                        kv_cache_mode="cached",
                    )[0]

                lat_p = self.scheduler.step(pred, t, lat_p, return_dict=False)[0]
        finally:
            self.transformer.to("cpu", non_blocking=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self._current_timestep = None

        lat_d_patch = _unpack_latents_with_ids(lat_p, lat_ids, pack_h, pack_w)
        lat_d_patch = lat_d_patch * s + m
        lat_d = _unpatchify_latents(lat_d_patch)
        img_out = self.vae.decode(lat_d, return_dict=False)[0]
        return self.image_processor.postprocess(img_out, output_type="pil")[0]
