# so101-world-model

## What This Is

A 1X-style world model for SO-101 robots. Given a current camera frame and a text instruction, it:
1. Generates a plausible future video of the task being completed (world model)
2. Extracts SO-101 joint actions from consecutive frame pairs (inverse dynamics model)
3. Executes the actions on the robot

Training data: all public SO-101/SO-100 datasets on HuggingFace + accumulated deployment data (flywheel).

**Paper inspiration**: [1X World Model](https://www.1x.tech/discover/world-model-self-learning)

---

## Architecture

```
Text + Current Frame
        │
   [CogVideoX-2b I2V]     ← fine-tuned on SO-101 data via LoRA
   (cloud inference)
        │
   Generated Video (13-49 frames)
        │
   Consecutive Frame Pairs (f_t, f_{t+1})
        │
   [Inverse Dynamics Model]  ← DINOv2-B encoder + transformer head
        │
   Action Sequence [T, 6]  ← SO-101 joint positions
        │
   SO-101 Robot Execution
```

### Key Components

| File | Purpose |
|------|---------|
| `wm/data/hub_downloader.py` | Download all SO-101/SO-100 HF datasets |
| `wm/data/dataset.py` | `IDMDataset` + `WorldModelDataset` |
| `wm/data/flywheel.py` | Deployment data collection + HF upload (MOAT) |
| `wm/models/video_encoder.py` | DINOv2-B frame encoder (frozen) |
| `wm/models/idm.py` | Inverse Dynamics Model: (f_t, f_{t+1}) → action |
| `wm/models/world_model.py` | CogVideoX-2b I2V wrapper + LoRA fine-tune |
| `wm/models/pipeline.py` | Full SO101Pipeline with test-time compute scaling |
| `wm/training/train_idm.py` | IDM supervised training loop |
| `wm/training/train_wm.py` | World model LoRA fine-tuning (cloud) |

---

## Repo Structure

```
so101-world-model/
├── wm/
│   ├── data/
│   │   ├── hub_downloader.py   # HF dataset search + download
│   │   ├── dataset.py          # IDMDataset, WorldModelDataset
│   │   ├── preprocessor.py     # Frame extraction, normalization
│   │   └── flywheel.py         # Deployment data → HF upload loop
│   ├── models/
│   │   ├── video_encoder.py    # DINOv2-B wrapper
│   │   ├── idm.py              # InverseDynamicsModel
│   │   ├── world_model.py      # CogVideoX-2b I2V fine-tune wrapper
│   │   └── pipeline.py         # SO101Pipeline (WM + IDM + scoring)
│   ├── training/
│   │   ├── train_idm.py        # IDM training loop
│   │   └── train_wm.py         # LoRA fine-tuning loop
│   └── utils/
│       └── logging.py
├── scripts/
│   ├── download_so101_data.py  # Pull all HF data locally
│   ├── train_idm.py            # IDM training entry point
│   ├── train_world_model.py    # WM fine-tuning entry point (cloud)
│   └── run_inference.py        # Deploy on SO-101
├── configs/
│   ├── default.yaml
│   ├── idm.yaml
│   └── world_model.yaml
└── tests/
```

---

## Quick Start

```bash
# Install
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"

# 1. Download all SO-101 data from HuggingFace
python scripts/download_so101_data.py --output_dir data/raw

# 2. Train the IDM (single GPU, ~2-4 hours on A100)
python scripts/train_idm.py --data_dir data/raw --output_dir checkpoints/idm

# 3. Fine-tune world model on SO-101 data (multi-GPU cloud, ~8-16 hours)
python scripts/train_world_model.py --data_dir data/raw --output_dir checkpoints/wm

# 4. Run on robot
python scripts/run_inference.py \
    --idm_ckpt checkpoints/idm/best.pt \
    --wm_lora checkpoints/wm/lora \
    --instruction "pick up the red block and place it in the box"
```

---

## The Moat: Data Flywheel

Every SO-101 deployment running this system can contribute data back:

```bash
# On your robot, after a session:
python -m wm.data.flywheel contribute \
    --episode_dir /path/to/episodes \
    --hf_repo your-org/so101-world-model-data
```

Successful episodes are auto-detected via CLIP goal scoring and uploaded to HuggingFace. Weekly retraining incorporates new data. More deployments → more data → better model → more deployments.

**Why this is a moat**: The IDM trained on 1000+ hours of SO-101 data is uniquely capable — anyone can run CogVideoX, but only you have this IDM.

---

## Notes for Claude

- **Always use `uv`** for dependency management, never `pip`
- IDM is the crown jewel — keep it standalone and well-tested
- World model runs on cloud (A100/H100), IDM runs anywhere (even M2 Air)
- CogVideoX-2b I2V HuggingFace ID: `THUDM/CogVideoX-2b`
- DINOv2-B HuggingFace ID: `facebook/dinov2-base`
- SO-101 action space: 6 DoF joint positions, typically in degrees or normalized [-1, 1]
- LeRobot dataset format: parquet files, images stored as bytes
- Test-time scaling: generate N=3-5 candidate videos, pick best by CLIP goal score
