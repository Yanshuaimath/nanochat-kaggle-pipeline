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

- [`d12_pretraining/nanochat-d12_pretraining_fp32_5_splits.ipynb`](./d12_pretraining/nanochat-d12_pretraining_fp32_5_splits.ipynb):
  successful `d12` fp32 pretraining path for Kaggle 2xT4. See the dedicated d12
  section below for the five-split resume flow.

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
| D12 pretraining | `d12_pretraining/nanochat-d12_pretraining_fp32_5_splits.ipynb` | tokenizer and `base_checkpoints/d12_kaggle` |
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

[`d12_pretraining/nanochat-d12_pretraining_fp32_5_splits.ipynb`](./d12_pretraining/nanochat-d12_pretraining_fp32_5_splits.ipynb)
is the current successful `d12` pretraining path. It targets Kaggle `2x Tesla T4`
with:

- `NANOCHAT_DTYPE=float32`
- `DEVICE_BATCH_SIZE=8`
- `TOTAL_BATCH_SIZE=524288`
- `TARGET_NUM_ITERATIONS=2520`
- five resumeable runs of 504 optimizer steps each

The notebook uses
[`scripts/base_train_updated.py`](./kaggle_dataset/nanochat/scripts/base_train_updated.py),
not the older `scripts/base_train.py`. The updated script is required because it
supports `--stop-at-step`, keeps the full 2520-step LR schedule while stopping at
split boundaries, restores optimizer shards on resume, and logs applied/skipped
optimizer steps.

Use one Kaggle run per split by setting `PRETRAIN_SPLIT_INDEX` in the first code
cell:

```python
PRETRAIN_SPLIT_INDEX = 1  # trains 0 -> 504
PRETRAIN_SPLIT_INDEX = 2  # resumes 504 -> 1008
PRETRAIN_SPLIT_INDEX = 3  # resumes 1008 -> 1512
PRETRAIN_SPLIT_INDEX = 4  # resumes 1512 -> 2016
PRETRAIN_SPLIT_INDEX = 5  # resumes 2016 -> 2520
```

Attach the previous split's cache dataset for splits 2-5:

| Split | Attach resume dataset | Output dataset |
| --- | --- | --- |
| 1 | none | `yshuaiqin/nanochat-d12-fp32-pretrain-cache-split-1` |
| 2 | `nanochat-d12-fp32-pretrain-cache-split-1` | `yshuaiqin/nanochat-d12-fp32-pretrain-cache-split-2` |
| 3 | `nanochat-d12-fp32-pretrain-cache-split-2` | `yshuaiqin/nanochat-d12-fp32-pretrain-cache-split-3` |
| 4 | `nanochat-d12-fp32-pretrain-cache-split-3` | `yshuaiqin/nanochat-d12-fp32-pretrain-cache-split-4` |
| 5 | `nanochat-d12-fp32-pretrain-cache-split-4` | `yshuaiqin/nanochat-d12-fp32-pretrain-cache` |

Intermediate split datasets keep the tokenizer, model checkpoint, metadata, and
both rank optimizer shards so the next split can resume. The final split keeps
the tokenizer plus final model/metadata checkpoint and prunes optimizer state.

The dataloader resume records parquet file, row group, and epoch state. This is
good enough for continuing training, but it is approximate because the internal
best-fit document buffer is not serialized.

### D12 fp16 failure note

The original d12 Kaggle attempt used `NANOCHAT_DTYPE=float16`. That path was not
reliable on 2xT4. D8 survived with float16, but d12 produced nonfinite behavior:
loss/gradients could overflow, GradScaler could skip optimizer steps, and earlier
checkpoints could advance in step number without meaningful weight updates.

In the diagnostic fp16 reruns, short tests could pass, but a longer 500-step test
failed with nonfinite loss at roughly step 376. This was enough to treat d12 fp16
on T4 as unstable rather than a dependable training recipe.

The likely pressure points are the d12 activation scale and fp16's limited
numeric range. In particular, the model uses squared-ReLU MLPs, projections, and
residual additions; in fp16, values above roughly 256 overflow when squared
because fp16 max finite value is about 65504. `base_train_updated.py` now logs
GradScaler state, skipped optimizer steps, and nonfinite losses so this kind of
failure is visible, but the working d12 recipe is fp32, not fp16.

The `gpt.py` fp16 clamp path is left as an opt-in debugging/stability experiment.
The successful five-split notebook sets `FP16_SAFE_MLP = False` and trains in
float32.

## Runtime Choices

The notebooks are tuned for Kaggle reliability rather than maximum throughput:

- `torchrun --nproc_per_node=2` when both T4 GPUs are available
- short training/evaluation runs for quick validation
- `--run=dummy` so Weights & Biases login is not required
- compile disabled in several post-training notebooks for Kaggle stability
- `NANOCHAT_DTYPE=float16` for the smaller d8 training path
- `NANOCHAT_DTYPE=float32` for d12 pretraining on T4, because the fp16 path was
  numerically unstable

## What This Repo Validates

This repository validates the Kaggle execution path for:

- tokenizer training and evaluation
- `d8` base pretraining and evaluation
- `d12` fp32 base pretraining across five Kaggle Dataset-backed resume runs
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
