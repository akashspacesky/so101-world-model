"""
CogVideoX-2b LoRA fine-tuning on SO-101 data.

Fine-tunes the world model to understand SO-101's embodiment, camera perspective,
and workspace dynamics. Runs on cloud (A100/H100 40-80GB, ~8-16 hours).

Uses diffusers + PEFT for LoRA. Only trains the transformer attention layers;
VAE and text encoder stay frozen.

Reference: https://huggingface.co/docs/diffusers/training/cogvideox
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wm.data.dataset import WorldModelDataset


COGVIDEOX_MODEL_ID = "THUDM/CogVideoX-2b"


def train_world_model(
    data_dir: Path,
    output_dir: Path,
    # Model
    model_id: str = COGVIDEOX_MODEL_ID,
    # LoRA config
    lora_rank: int = 128,
    lora_alpha: int = 128,
    target_modules: list[str] | None = None,
    # Training
    epochs: int = 10,
    batch_size: int = 1,           # CogVideoX is large; batch 1-2 on A100 40GB
    gradient_accumulation: int = 4,
    lr: float = 2e-4,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,
    mixed_precision: str = "bf16",
    # Video
    clip_len: int = 13,
    stride: int = 2,
    # Misc
    device: str = "cuda",
    num_workers: int = 2,
    save_every: int = 1,
    log_every: int = 10,
    max_episodes: int | None = None,
    wandb_project: str | None = None,
    resume_from: str | Path | None = None,
) -> None:
    """
    Fine-tune CogVideoX-2b on SO-101 video data using LoRA.

    The world model learns:
    - SO-101 camera perspective and workspace layout
    - Robot arm appearance and motion dynamics
    - Task-relevant object interactions (pick/place, fold, etc.)

    After fine-tuning, the model generates physically plausible videos
    grounded in SO-101's actual kinematics and workspace.
    """
    try:
        from diffusers import CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
        from diffusers.training_utils import compute_snr
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, T5EncoderModel
        import accelerate
        from accelerate import Accelerator
    except ImportError as e:
        raise ImportError(
            f"Fine-tuning requires diffusers, peft, accelerate: {e}\n"
            "Run: uv pip install -e '.[dev]'"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accelerator handles mixed precision, gradient accumulation, multi-GPU
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation,
        log_with="wandb" if wandb_project else None,
    )

    if wandb_project and accelerator.is_main_process:
        accelerator.init_trackers(wandb_project, config={
            "lora_rank": lora_rank, "lr": lr, "epochs": epochs,
            "batch_size": batch_size, "clip_len": clip_len,
        })

    # Load pipeline components separately so we can freeze selectively
    print(f"Loading {model_id}...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float32,
    )

    # Freeze VAE and text encoder — only train the video transformer
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Apply LoRA to the transformer attention layers
    if target_modules is None:
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
        ]

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    transformer = get_peft_model(pipe.transformer, lora_config)
    transformer.print_trainable_parameters()

    # Dataset + dataloader
    dataset = WorldModelDataset(
        data_dir,
        clip_len=clip_len,
        stride=stride,
        max_episodes=max_episodes,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No episodes found in {data_dir}.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-8,
    )

    num_update_steps = math.ceil(len(loader) / gradient_accumulation) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_update_steps
    )

    # Prepare with accelerator
    transformer, optimizer, loader, scheduler = accelerator.prepare(
        transformer, optimizer, loader, scheduler
    )

    if resume_from:
        accelerator.load_state(str(resume_from))
        print(f"Resumed from {resume_from}")

    global_step = 0

    for epoch in range(1, epochs + 1):
        transformer.train()
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            with accelerator.accumulate(transformer):
                video = batch["video"]          # (B, T, C, H, W)
                text = batch["text"]            # list[str]
                first_frame = batch["first_frame"]  # (B, C, H, W)

                # Encode video frames into latents via VAE
                with torch.no_grad():
                    # CogVideoX VAE expects (B, C, T, H, W)
                    video_bctchw = video.permute(0, 2, 1, 3, 4)
                    latents = pipe.vae.encode(video_bctchw).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                    # Encode text with T5
                    text_inputs = pipe.tokenizer(
                        text,
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_inputs = {k: v.to(latents.device) for k, v in text_inputs.items()}
                    encoder_hidden_states = pipe.text_encoder(**text_inputs).last_hidden_state

                    # Encode conditioning image (first frame)
                    first_frame_latent = pipe.vae.encode(
                        first_frame.unsqueeze(2)  # add time dim
                    ).latent_dist.sample() * pipe.vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # Forward pass through transformer
                noise_pred = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    image_rotary_emb=None,
                    return_dict=False,
                )[0]

                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % log_every == 0 and accelerator.is_main_process:
                print(f"  Epoch {epoch} step {step} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                if wandb_project:
                    accelerator.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} | avg loss {avg_loss:.4f}")

        # Save LoRA weights
        if epoch % save_every == 0 and accelerator.is_main_process:
            save_path = output_dir / f"lora_epoch_{epoch:03d}"
            save_path.mkdir(exist_ok=True)
            unwrapped = accelerator.unwrap_model(transformer)
            unwrapped.save_pretrained(str(save_path))
            print(f"  Saved LoRA to {save_path}")

    if accelerator.is_main_process:
        final_path = output_dir / "lora_final"
        final_path.mkdir(exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(str(final_path))
        print(f"\nFine-tuning complete. Final weights at {final_path}")

    accelerator.end_training()
