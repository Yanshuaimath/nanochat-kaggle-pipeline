# Nanochat Kaggle Pipeline

This repository contains a Kaggle-ready training and post-training pipeline for
[`nanochat`](https://github.com/karpathy/nanochat). It packages the modified
nanochat source code as a Kaggle Dataset and provides notebook stages that can be
run on Kaggle, mostly targeting the `2x Tesla T4` GPU runtime.

The goal is practical pipeline validation on Kaggle: pretraining, SFT,
distillation, preference optimization, RL post-training, quantization, and
serving tests.

## Main D8 Notebook Order

Run the `d8` pipeline in this order:

1. [`nanochat_d8_pretraining.ipynb`](./nanochat_d8_pretraining.ipynb)
2. [`nanochat_d8_sft.ipynb`](./nanochat_d8_sft.ipynb)
3. [`nanochat_d8_distill.ipynb`](./nanochat_d8_distill.ipynb)
4. [`nanochat_d8_rl_and_dpo.ipynb`](./nanochat_d8_rl_and_dpo.ipynb)
5. [`nanochat_d8_grpo.ipynb`](./nanochat_d8_grpo.ipynb)
6. [`nanochat_d8_ppo_and_ppo-standard.ipynb`](./nanochat_d8_ppo_and_ppo-standard.ipynb)
7. [`nanochat_d8_universal-rl.ipynb`](./nanochat_d8_universal-rl.ipynb)
8. [`nanochat_d8_quant.ipynb`](./nanochat_d8_quant.ipynb)

The first two notebooks form the required base path:

- pretraining produces the tokenizer and `base:d8_kaggle` checkpoint cache
- SFT imports that pretraining cache and produces `sft:d8_kaggle`

The later notebooks start from the SFT cache and can be run independently after
SFT has been saved as a Kaggle Dataset.

## Other Notebooks

Additional training notebooks are included:

- [`d12_pretraining/nanochat-d12_pretraining.ipynb`](./d12_pretraining/nanochat-d12_pretraining.ipynb):
  real `d12` pretraining path. See the dedicated d12 section below for the
  two-part training flow.

## Repository Layout

```text
.
├── README.md
├── kernel-metadata.json
├── nanochat_d8_pretraining.ipynb
├── nanochat_d8_sft.ipynb
├── nanochat_d8_distill.ipynb
├── nanochat_d8_rl_and_dpo.ipynb
├── nanochat_d8_grpo.ipynb
├── nanochat_d8_ppo_and_ppo-standard.ipynb
├── nanochat_d8_universal-rl.ipynb
├── nanochat_d8_quant.ipynb
├── d12_pretraining/
└── kaggle_dataset/
    ├── dataset-metadata.json
    └── nanochat/
        ├── pyproject.toml
        ├── nanochat/
        ├── scripts/
        ├── tasks/
        └── tests/
```

The uploadable Kaggle Dataset package is [`kaggle_dataset/`](./kaggle_dataset).
The actual nanochat code lives inside
[`kaggle_dataset/nanochat/`](./kaggle_dataset/nanochat). After publishing, Kaggle
mounts that package as `nanochat-codes`, and the notebooks copy it into
`/kaggle/working/nanochat`.

## Kaggle Dataset Setup

All notebooks expect:

- GPU enabled
- internet enabled
- the `nanochat-codes` Kaggle Dataset attached

Most post-pretraining notebooks also expect one or more cache datasets created by
earlier stages.

The code dataset mapping is:

- local package directory: `kaggle_dataset/`
- code inside the package: `kaggle_dataset/nanochat/`
- Kaggle dataset slug: `nanochat-codes`
- mounted Kaggle path pattern:
  `/kaggle/input/datasets/<your-kaggle-username>/nanochat-codes`

For the author's Kaggle account, the code dataset id is:

```text
yshuaiqin/nanochat-codes
```

Each notebook auto-detects attached datasets from Kaggle input paths and copies
the required code/cache files into `/kaggle/working`.

## Publishing The Code Dataset

The Kaggle metadata for the code bundle is stored at
[`kaggle_dataset/dataset-metadata.json`](./kaggle_dataset/dataset-metadata.json).

Publish or update the code dataset with:

```bash
kaggle datasets version -p kaggle_dataset -m "update nanochat Kaggle dataset" --dir-mode zip
```

If publishing under a different Kaggle account, update the dataset id in
`kaggle_dataset/dataset-metadata.json` first.

After publishing, attach `nanochat-codes` to each Kaggle notebook.

## Stage Outputs

The notebooks pass work forward through Kaggle Dataset cache bundles:

| Stage | Notebook | Main output |
| --- | --- | --- |
| Pretraining | `nanochat_d8_pretraining.ipynb` | tokenizer and `base_checkpoints/d8_kaggle` |
| SFT | `nanochat_d8_sft.ipynb` | `chatsft_checkpoints/d8_kaggle` |
| Distill | `nanochat_d8_distill.ipynb` | `chatdistill_checkpoints/d8_kaggle` |
| RL + DPO | `nanochat_d8_rl_and_dpo.ipynb` | RL and DPO checkpoint caches |
| GRPO | `nanochat_d8_grpo.ipynb` | GRPO policy and reward checkpoints |
| PPO | `nanochat_d8_ppo_and_ppo-standard.ipynb` | PPO and PPO-standard checkpoints |
| Universal RL | `nanochat_d8_universal-rl.ipynb` | REINFORCE, RLOO-KL, and GSPO checkpoints |
| Quant | `nanochat_d8_quant.ipynb` | `chatquant_exports` artifacts |

The exact Kaggle Dataset ids for saved caches are set inside the notebook cells.
Change them before uploading if you are publishing under your own account.

## D12 Pretraining Notebook

[`d12_pretraining/nanochat-d12_pretraining.ipynb`](./d12_pretraining/nanochat-d12_pretraining.ipynb)
is a separate `d12` pretraining experiment. It downloads a moderate ClimbMix
pretraining slice, trains/evaluates the tokenizer, pretrains a `d12` base model,
saves `base_checkpoints/d12_kaggle`, and evaluates the base checkpoint with
reduced Kaggle-friendly settings.

Because the full `d12` pretraining run is too long for the expected T4 runtime,
the notebook is designed as two half-runs:

- first half: set `PRETRAIN_RUN_MODE = "first_half"` in the `D12 Run Settings`
  cell, then publish the resulting cache as `nanochat-d12-pretrain-cache-first-half`
- second half: attach `nanochat-d12-pretrain-cache-first-half`, keep
  `CACHE_DATASET_NAME = "nanochat-d12-pretrain-cache-first-half"` in the first
  code cell, and set `PRETRAIN_RUN_MODE = "second_half"` in `D12 Run Settings`

## Runtime Choices

The notebooks are tuned for Kaggle reliability rather than maximum throughput:

- `torchrun --nproc_per_node=2` when both T4 GPUs are available
- short training/evaluation runs for quick validation
- `--run=dummy` so Weights & Biases login is not required
- compile disabled in several post-training notebooks for Kaggle stability
- `NANOCHAT_DTYPE=float16` for most training runs and `float32` where the script
  is more stable without GradScaler

## What This Repo Validates

This repository validates the Kaggle execution path for:

- tokenizer training and evaluation
- `d8` base pretraining and evaluation
- SFT from a saved base checkpoint
- distillation data generation and student training
- DPO and RL-style post-training
- GRPO, PPO, and PPO-standard experiments
- checkpoint comparison reports
- optional quantized export/evaluation
- full-precision and quantized serving entrypoints

It does not claim:

- strong benchmark performance
- large-scale training quality
- production-ready checkpoints
- reliable model-quality conclusions from the Kaggle-scale runs

## Acknowledgement

This repository builds on
[`karpathy/nanochat`](https://github.com/karpathy/nanochat). The changes here are
focused on making nanochat runnable as a staged Kaggle workflow with dataset
packaging, cache handoff, post-training branches, quantization, and serving
entrypoints.
