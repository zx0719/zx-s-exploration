# MMLM

This directory refactors `Remote-Clip+Qwen` into a modular Hydra-style project.

## Structure

- `train.py`: thin training entrypoint
- `inference.py`: thin generation / evaluation entrypoint
- `src/model/`: shared Remote-Clip + Qwen model code
- `src/datasets/`: RSICD, AID, and OSVQA datasets with a shared dict-batch contract
- `src/trainer/`: stage1 trainer, stage2 trainer, and inference runners
- `src/configs/`: Hydra config groups for models, datasets, trainers, loggers, and inferencers

## Refactor mapping

- `Remote-Clip+Qwen/train_stage1/V2/train_stage1_rsicd.py`
  -> `src/model/remote_qwen.py`, `src/datasets/stage1.py`, `src/trainer/stage1.py`
- `Remote-Clip+Qwen/train_stage2/V1/train_stage2_osvqa.py`
  -> `src/model/remote_qwen.py`, `src/datasets/stage2.py`, `src/trainer/stage2.py`
- `Remote-Clip+Qwen/train_stage2/evaluate/eval_stage2_osvqa.py`
  -> `src/trainer/inferencer.py`

## Usage

Fill the `???` values in the selected config first.

Stage1:

```bash
python train.py model=stage1_align datasets=stage1_mixed trainer=stage1
```

Stage2:

```bash
python train.py model=stage2_connector_lora datasets=stage2_train_osvqa trainer=stage2
```

OSVQA evaluation:

```bash
python inference.py model=stage2_inference datasets=stage2_eval_osvqa inferencer=osvqa_eval
```

Single-image generation:

```bash
python inference.py inferencer=single_image model=stage2_inference datasets=none
```

## Notes

- `model.remoteclip_repo_path` should point to the RemoteClip source directory that exposes `RemoteVisionTower.py`.
- The project keeps the original dict-batch keys: `pixel_values`, `input_ids`, `attention_mask`, `labels`, and optional `meta`.
- `logger=swanlab` is optional. The default logger is a no-op.
