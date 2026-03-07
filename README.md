
# Papers
OLMoE: [OLMoE technical paper](https://arxiv.org/abs/2409.02060)

# Data
We train with the entire Wikipedia from Dolma 1 that is available at [HF OLMoE-mix-0924](https://huggingface.co/datasets/allenai/OLMoE-mix-0924), it's 3,689,204,525 tokens (3.689B).

And we later add [pes2o-0000](https://huggingface.co/datasets/allenai/OLMoE-mix-0924/tree/main/data/pes2o) (STEM papers), which makes the total number of tokens: 4.7B.

To download some of the data:

```bash
wget -O data/wiki-001.json.gz "https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/main/data/wiki/wiki-0001.json.gz?download=true"
```

Dowloading Eval data, for now we use the c4 dataset from [c4 HF dataset](https://huggingface.co/datasets/allenai/c4/tree/main/en), we take ```c4-train.00000-of-01024.json.gz``` and ```c4-train.00001-of-01024.json.gz```.

```bash
wget -O data/c4-train.00000-of-01024.json.gz "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz?download=true"
wget -O data/c4-train.00001-of-01024.json.gz "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00001-of-01024.json.gz?download=true"
```

check memory usage:
```bash
srun --jobid=744464 --overlap nvidia-smi
```

To tokenize the data:

```bash
dolma tokens \
--documents data/pes2o-0000.json.gz \
--destination data_all/pes2o-0000.npy \
--tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
--max_size '2_147_483_648' \
--seed 0 \
--tokenizer.eos_token_id 50279 \
--tokenizer.pad_token_id 1 \
--processes 2
```

- Scaling laws for tiny models: https://arxiv.org/pdf/2507.17702v1

OlMoE:

Ablations:
- Mixture-of-Experts vs. Dense
- Expert Granularity
- Shared Experts
- Load Balancing Loss
- Router Z-loss
- Loss free Routing
- Expert Choice vs. Token Choice X
- Sparse Upcycling X


MoE Analysis:
- Router Saturation
- Expert Co-activation
- Domain specialization
- Vocabulary specialization


### Evaluation

```bash
python -m torch.distributed.run --nproc-per-node 2 \
  OLMo/scripts/train.py configs/olmoe-small.yml \
  --load_path=runs/olmoe-small-multigpu/step1000-tmp \
  --eval_on_load \
  --max_duration=0 \
  --evaluators='[{label: hellaswag, type: downstream}, {label: piqa, type: downstream}, {label: arc_easy, type: downstream}]'

```

# Expert Granularity
- AI2 trains for up to 130B tokens, the OLMoE-mix is 4.060T tokens.
- Compare our plot with: [OlMoE reports, Plot: Granularity](https://wandb.ai/ai2-llm/olmoe/reports/Plot-Granularity--Vmlldzo4OTIxOTE4)

Expert granularity defines the trade-off between having a few large experts versus many smaller experts while keeping the total active parameters constant.

To determine the optimal expert size, OLMoE performed a controlled ablation study comparing different granularities while maintaining ~1.3B active parameters and ~6.9B total parameters.
Experiments

The authors compared three primary configurations of experts per MoE layer:

- 8 Experts (Top-1 Routing): The baseline configuration with coarse, large experts.
- 32 Experts (Top-4 Routing): Finer granularity where the hidden dimension of each expert is reduced by 4x.
- 64 Experts (Top-8 Routing): Even finer granularity where each expert is 8x smaller than the baseline.

#### Our model size and number of Tokens

#### Performance Comparison

| Configuration | Total Params | Active Params | Wiki | C4 |
|---------------|--------------|---------------|------------|----------|
| 8 Experts     | x         | x          | x       | x     |
| 32 Experts    | x         | x          | x       | x     |
| 64 Experts    | x         | x          | x       | x     |

Key Findings


# Timeline

- 2026-03-01 run ```iliass-lasri-team/olmoe-1/a69zc1qq```.
  - 24h training on 2 GPUs yields ~11k steps (350M tokens)
  - The training converges to a perplexity of ~20-30 range on the training set.
  - 350M tokens is 1/10th of the total training data, and is not sufficient to get any good results on the evals.
  - Next step is to increase the dataset size, by including a part of the C4 dataset, see dataset section, and run for more than 24h (we want at least to train on 1B) that will take ~4 days on 2 GPUs, with the current model size of ~320M parameters.
  - We might also try to increase the model size.

  - New run: ```iliass-lasri-team/olmoe-1/olmoe-sq_4096```
    - starting from ```iliass-lasri-team/olmoe-1/a69zc1qq``` last checkpoint.
    - Double the sequence length to 4096 tokens.
    - 3.6B tokens to 4.7B tokens.
  
- 2026-03-07 run ```iliass-lasri-team/olmoe-1/olmoe-sq_4096```
  - run path: ```iliass-lasri-team/olmoe-1/ce2uabgs```
  - from scratch (no checkpoint)
  - x5 throughput compared to previous run.

