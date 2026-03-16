# Training Quickstart — Local torchrun

This document explains how to run training directly from the repo (no Kaggle notebook).

---

## Repo structure

| Component | Source |
|-----------|--------|
| This repo (`OLMoE`) | `https://github.com/iliasslasri/OLMoE.git` |
| OLMo submodule (`OLMo/`) | `https://github.com/Tristan22400/OLMo.git` branch `routing/moe-strategies` |
| Megablocks (pip, **not** a submodule) | `https://github.com/Tristan22400/megablocks.git` branch `routing/auxiliary-loss-free` |

Megablocks is not tracked as a git submodule — it is pip-installed directly from the fork URL, exactly as the Kaggle notebook does.

---

## Prerequisites

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/iliasslasri/OLMoE.git
cd OLMoE
```

If already cloned without submodules:
```bash
git submodule update --init --recursive
```

The OLMo submodule defaults to whatever commit `.gitmodules` points to. To use the active development branch:
```bash
cd OLMo
git fetch origin routing/moe-strategies
git checkout routing/moe-strategies
cd ..
```

### 2. Create and activate the conda environment

```bash
conda create -n olmoe_env python=3.10 -y
conda activate olmoe_env
```

### 3. Install OLMo

The submodule must be pip-installed in editable mode so `olmo` is importable:

```bash
pip install -e "OLMo/[train]"
```

This installs:
- `torch>=2.1,<2.5`, `ai2-olmo-core==0.1.0`, `omegaconf`, `rich`
- `wandb`, `torchmetrics`, `smashed`, `datasets`, `msgspec` (train extras)

### 4. Install megablocks (your fork)

```bash
pip install git+https://github.com/Tristan22400/megablocks.git@routing/auxiliary-loss-free
```

This is required whenever `moe_mlp_impl: sparse` is set in the config (the default for production runs). Megablocks provides the sparse GPU kernels for dropless MoE and the routing strategies (standard, loss-free, random).

> If megablocks fails to compile, ensure `nvcc` is available and matches your CUDA version:
> ```bash
> nvcc --version && python -c "import torch; print(torch.version.cuda)"
> ```

### 5. Install flash-attention (optional)

Only needed when `flash_attention: true` in the config (disabled in the debug config):

```bash
pip install flash-attn --no-build-isolation
```

### 6. Tokenizer

The tokenizer is bundled inside the OLMo submodule at:
```
OLMo/olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json
```
The config references it as `tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json` — resolved via `PYTHONPATH` pointing to `OLMo/` (set automatically by `run-local.sh`).

### 7. Data

Tokenized `.npy` files live in `data/tokenized/`. The production config (`configs/olmoe-small.yml`) currently has absolute cluster paths — update `data.paths` for local runs:

```yaml
data:
  paths:
    - data/tokenized/wiki/part-0-00000.npy
    - data/tokenized/pes2o/pes2o-0000.npy
```

To tokenize new raw data:

```bash
dolma tokens \
  --documents data/raw/wiki-0001.json.gz \
  --destination data/tokenized/wiki/wiki-0001.npy \
  --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
  --max_size '2_147_483_648' \
  --seed 0 \
  --tokenizer.eos_token_id 50279 \
  --tokenizer.pad_token_id 1 \
  --processes 2
```

### 8. W&B credentials

Create `~/.env` with:
```
WANDB_API_KEY=your_key_here
```

---

## Running training

### Option A — Use the existing script (recommended)

```bash
# From repo root
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env | cut -d '=' -f 2)
bash scripts/run-local.sh
```

This script:
- Sets `PYTHONPATH` to include `OLMo/`
- Sets `OLMO_TASK=model`, `PYTHONNOUSERSITE=1`
- Launches `torchrun` with 1 GPU on `configs/debug/olmoe-local.yml` (tiny debug model)
- Uses `~/miniconda3/envs/olmoe_env/bin/python` — **edit this path** if your env is elsewhere

### Option B — Direct torchrun command

```bash
export PYTHONPATH=$(pwd)/OLMo:$PYTHONPATH
export OLMO_TASK=model
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.env | cut -d '=' -f 2)

python -m torch.distributed.run \
  --nnodes=1 --node_rank=0 \
  --nproc-per-node=<NUM_GPUS> \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 \
  OLMo/scripts/train.py configs/olmoe-small.yml \
  --save_overwrite
```

Replace `<NUM_GPUS>` with: `nvidia-smi -L | wc -l`

### Option C — SLURM cluster

```bash
sbatch job.sh
```

`job.sh` activates the `olmoe` conda env, reads `WANDB_API_KEY` from `~/.env`, and calls `scripts/olmoe-gantry.sh`. Edit `--partition`, `--gres`, and `--time` as needed.

---

## Configs

| Config | Purpose |
|--------|---------|
| `configs/debug/olmoe-local.yml` | Tiny 2-layer, 128-dim model, no flash-attn, CPU init — fast local iteration |
| `configs/olmoe-small.yml` | Production: 1024-dim, 16 layers, 8 experts, seq len 4096 |

**Key fields to adjust before running locally:**

```yaml
# configs/olmoe-small.yml
save_folder: runs/${run_name}

data:
  paths:
    - data/tokenized/wiki/part-0-00000.npy   # must be local absolute or relative paths

wandb:
  entity: your-wandb-entity    # or set to null to disable

model:
  moe_routing_type: loss_free  # learned | loss_free | random
```

---

## Routing strategies

The megablocks fork adds three routing modes (set via `moe_routing_type` in the config):

| Value | Description |
|-------|-------------|
| `learned` | Standard auxiliary load-balancing loss (baseline) |
| `loss_free` | Auxiliary-loss-free additive bias (Wang et al., 2024) |
| `random` | Uniform random assignment (ablation) |

When using `loss_free`, ensure `moe_zloss_weight` is set (e.g. `0.001`) in the config.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'olmo'` | `pip install -e "OLMo/[train]"` and/or `export PYTHONPATH=$(pwd)/OLMo:$PYTHONPATH` |
| `ModuleNotFoundError: No module named 'megablocks'` | `pip install git+https://github.com/Tristan22400/megablocks.git@routing/auxiliary-loss-free` |
| megablocks compile error | Check `nvcc` version matches `torch.version.cuda`; try `pip install ninja` first |
| Flash attention error | Set `flash_attention: false` in config, or `pip install flash-attn --no-build-isolation` |
| OmegaConf error on `activation_checkpointing` | Must be `fine_grained` or `null` for MoE — not `whole_layer` or `one_in_two` |
| NCCL timeout on single GPU | Use `--nproc-per-node=1` and `fsdp.sharding_strategy: NO_SHARD` |
| W&B not logging | Check `WANDB_API_KEY`; set `WANDB_MODE=offline` to disable |
| OOM | Reduce `device_train_microbatch_size` or use the debug config |
