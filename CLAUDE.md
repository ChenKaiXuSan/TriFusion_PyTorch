# CLAUDE.md

This repository now keeps the active implementation focused on `TriPoseFusion`.
The legacy RGB/action-classification `project/` workflow has been moved out of
this repository and archived separately.

## Active Flow

`TriPoseFusion` trains a geometry-guided, self-supervised multi-view 3D pose
fusion model from synchronized SAM3D keypoints.

Main entry point:

```bash
python TriPoseFusion/train.py
```

Typical fold-index generation:

```bash
python TriPoseFusion/generate_index_mapping.py
```

Pegasus experiment scripts live in `pegasus/` and call `TriPoseFusion/train.py`
directly.

## Important Directories

```text
TriPoseFusion/
  train.py                         # Hydra/PyTorch Lightning training entry
  configs/train.yaml               # Main training config
  dataloader/                      # Keypoint dataset and Lightning datamodule
  models/keypoint_mlp.py           # Tri-view pose fusion model
  trainer/train_triple_fusion.py   # LightningModule and losses
  eval/                            # Evaluation scripts
  docs/                            # TriPoseFusion-specific docs
pegasus/                           # Experiment shell scripts
analysis/                          # Analysis outputs and figures
traingulation/                     # Triangulation utilities
```

## Development Notes

- Prefer `TriPoseFusion` imports and entry points for new work.
- Do not reintroduce dependencies on the legacy `project` package.
- If an old RGB/video classification component is needed for reference, use the
  archived copy outside this repository rather than restoring it here.
