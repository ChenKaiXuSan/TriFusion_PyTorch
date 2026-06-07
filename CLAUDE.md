# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Third-Person Multi-View Driver Action Analysis for Cooperative Perception under Visual Field Impairment**

This is a PyTorch-based research project analyzing driver head/actions from three external cameras (left/front/right). The goal is to understand how a multi-view sensing system can infer driver actions and provide compensation when the driver's perceptual access is limited.

**Key Goal**: Not just accuracy, but understanding how to robustly infer driver actions under visual constraints for assistive interfaces.

---

## Core Architecture

### Directory Structure

```
project/              # Main experiment codebase
├── main.py           # Entry point (Hydra-based training)
├── cross_validation.py
├── eval.py
├── dataloader/       # Data loading & preprocessing
│   ├── whole_video_dataset.py  # Core dataset class with chunked loading
│   ├── data_loader.py
│   └── annotation_dict.py
├── models/           # Model architectures
│   ├── base_model.py
│   ├── ts_cva_model.py       # TS-CVA fusion model
│   ├── res_3dcnn.py
│   ├── video_transformer.py
│   └── video_mamba.py
├── trainer/          # Training modules (single/multi-view)
│   ├── multi_selector.py     # Route to appropriate trainer
│   ├── single_selector.py
│   └── multi/              # Multi-view training logic
└── utils/            # Utilities & visualization
    ├── ts_cva_visualization.py

TriPoseFusion/        # Alternative 3D pose fusion codebase (similar structure)
pegasus/            # Shell scripts for experiments
analysis/           # Data analysis & figures
docs/               # Detailed documentation

tests/              # Pytest-based tests
configs/            # Hydra configuration files
```

---

## Core Concepts & Patterns

### 1. Multi-View vs Single-View Training

The project uses **Hydra config** to control training mode:

```yaml
# train.view = 'single' -> Single-view baseline (e.g., front camera only)
# train.view = 'multi'  -> Multi-view fusion (left/front/right combined)
```

**Trainer Selection** (`project/trainer/multi_selector.py`):
- `build_multi_trainer(hparams)` → multi-view training module
- `build_single_trainer(hparams)` → single-view training module
- Both return a PyTorch Lightning Module that handles training loop

### 2. Model Backbones (Shared Across Views)

Three backbone options in `project/models/`:
- **3D CNN** (`res_3dcnn.py`): Default choice, good balance of speed/performance
- **Video Transformer** (`video_transformer.py`): Higher capacity, slower
- **Video Mamba** (`video_mamba.py`): Linear complexity alternative

All views share backbone weights when `model.backbone_shared = true` (recommended).

### 3. Fusion Methods (`model.fuse_method`)

```python
# Late fusion (separate encoders, aggregate logits/probs)
fuse_method: late          # Logit averaging

# Early fusion (aggregate inputs before encoding)
fuse_method: add           # Element-wise sum
fuse_method: mul           # Element-wise multiply  
fuse_method: concat        # Concatenate views spatially
fuse_method: avg           # Average pooling across views

# Mid fusion (SE-Attention fusion - legacy, use ts_cva instead)
fuse_method: se_attn

# ⭐ TS-CVA (Temporal-Synchronous Cross-View Attention) - RECOMMENDED
fuse_method: ts_cva        # Frame-synchronous view interaction with dynamic gating
```

**TS-CVA Advantages**:
- Models complementary relationships between views at each timestep
- Learns time-varying weights to downweight occluded/unreliable views
- Provides interpretability via attention weights and gating scores
- Robust to view occlusions and lighting variations

### 4. Video Chunked Loading (OOM Prevention)

For long videos (1000+ frames), use chunked loading in `whole_video_dataset.py`:

```python
# Config parameter:
data:
  max_video_frames: 1000  # Max frames per chunk (tune based on resolution/memory)

# Long video automatically splits into multiple chunks:
# 5000-frame video + max_video_frames=1000 → 5 training samples of ~1000 frames each
```

**Chunk Size Guidelines**:
- 224×224 resolution: 500-1000 frames
- 112×112 resolution: 1000-2000 frames
- Larger resolutions: reduce further (300-500)

---

## Running Experiments

### Quick Start

```bash
# Install dependencies
pip install -e .

# Run single-view training with 3D CNN
python project/main.py \
    --config-name config \
    train.view=single \
    train.view_name='[front]' \
    model.backbone=3dcnn \
    model.fuse_method=late

# Run multi-view TS-CVA training
python project/main.py \
    --config-name config \
    train.view=multi \
    model.fuse_method=ts_cva \
    model.backbone=3dcnn
```

### Cross-Validation Mode

The project supports k-fold cross-validation in `cross_validation.py`:

```bash
python project/main.py \
    --config-name config \
    train.view=multi \
    data.fold=5  # 5-fold CV (default is single split)
```

---

## Test Suite

Run tests with pytest:

```bash
# All tests
pytest tests/ -v

# Specific test categories
pytest tests/test_ts_cva.py           # TS-CVA model tests
pytest tests/test_trainer_selection.py # Trainer routing tests  
pytest tests/test_chunked_loading.py   # Video chunking functionality
pytest tests/test_separable_resnet.py  # ResNet variants
```

**Key Tests**:
- `test_quick_integration.py`: Basic import/interface checks
- `test_integration_modalities.py`: RGB + keypoint fusion tests
- `test_zero_padding.py`: Padding handling validation

---

## Configuration (`configs/config.yaml`)

Key parameters to understand:

```yaml
# Paths
paths.root_path: /workspace/data/multi_view_driver_action
paths.video_path: /workspace/data/videos_split

# Data settings
data.img_size: 224
data.uniform_temporal_subsample_num: 8  # Frames extracted from each chunk
data.batch_size: 1
data.max_video_frames: 1000  # Chunk size (OOM prevention)

# Model choices
model.backbone: 3dcnn       # [3dcnn, transformer, mamba]
model.model_class_num: 4    # Number of action classes
model.fusion_mode: logit_mean  # [logit_mean, prob_mean, feature_mean, feature_concat]
model.fuse_method: ts_cva     # Fusion strategy

# TS-CVA specific (when fuse_method=ts_cva)
model.ts_cva_shared_backbone: true
model.ts_cva_use_view_embedding: true
model.ts_cva_use_gated_aggregation: true
model.ts_cva_num_heads: 4
model.ts_cva_temporal_dim: 512
model.ts_cva_temporal_layers: 2

# Training
train.max_epochs: 50
train.gpu: 0
```

---

## Key Files to Understand

### Entry Point Flow

1. `project/main.py` → Hydra config loads, trainer selection based on `train.view`
2. `project/cross_validation.py` → Handles k-fold CV logic
3. `project/trainer/multi_selector.py` → Routes to appropriate multi-view trainer
4. Training module returns PyTorch Lightning Module with:
   - Backbone encoders (shared across views)
   - Fusion layer (TS-CVA / late fusion / etc.)
   - Classification head

### Data Flow

1. `project/dataloader/data_loader.py` → DriverDataModule setup
2. `project/dataloader/whole_video_dataset.py` → LabeledVideoDataset class:
   - Handles video loading + chunking
   - SAM 3D body keypoint loading (optional)
   - Returns dict with keys: `video`, `sam3d_kpt`, `label`, `meta`

### Model Architecture

1. `project/models/base_model.py` → Base class for all models
2. `project/models/ts_cva_model.py` → TS-CVA implementation:
   - Cross-view attention per timestep
   - Learnable gated aggregation (dynamic view weights)
   - Temporal modeling via TCN
3. `project/models/res_3dcnn.py`, `video_transformer.py`, `video_mamba.py` → Backbones

---

## Documentation References

Detailed documentation is available in `doc/`:

- **[TS-CVA_README.md](doc/TS-CVA_README.md)** - Complete guide to TS-CVA architecture and usage
- **[VIDEO_CHUNKING_GUIDE.md](doc/VIDEO_CHUNKING_GUIDE.md)** - OOM prevention via chunked loading
- **[DATASET_USAGE.md](doc/DATASET_USAGE.md)** - How to use whole_video_dataset with multi-view videos
- **[OOM_SOLUTIONS_INDEX.md](doc/OOM_SOLUTIONS_INDEX.md)** - Memory error troubleshooting guide
- **[SEPARABLE_RESNET_README.md](doc/SEPARABLE_RESNET_README.md)** - Alternative ResNet variants

---

## Common Development Tasks

### Adding a New Fusion Method

1. Add method name to `model.fuse_method` choices in `config.yaml`
2. Implement fusion logic in relevant trainer module (`project/trainer/multi/`)
3. Add test case in `tests/`

### Adding a New Backbone

1. Create new model class inheriting from `BaseModel` in `project/models/`
2. Register in `project/models/make_model.py` factory function
3. Update config.yaml with new backbone name

### Visualization

TS-CVA provides visualization utilities:
- `project/utils/ts_cva_visualization.py`: Gate weights curves, attention heatmaps
- `project/utils/save_CAM.py`: CAM generation for interpretability

---

## Research Context

**SIGCHI-focused** - This work targets cooperative perception and assistive interfaces, not just classification accuracy. Key evaluation metrics include:

- **View contribution analysis**: How each view helps (single-view, LOVO, pairwise)
- **Robustness testing**: Random view dropout at inference
- **Interpretability**: Attention weights show which views are relied upon when
- **Per-class recall**: Identify failure modes

**Target Audience**: Human-centered computing researchers interested in:
- Assistive interfaces for visual field impairment
- Cooperative perception between human and system
- Robustness under partial sensing constraints

---

## Citation Format

```bibtex
@inproceedings{Chen2026TSCVA,
  title     = {Temporal-Synchronous Cross-View Attention for Multi-View Driver Action Recognition},
  author    = {Chen, Kaixu},
  booktitle = {CHI Conference on Human Factors in Computing Systems},
  year      = {2026}
}
```
