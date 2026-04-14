# Nanochat Kaggle Pipeline

This repository packages a Kaggle-ready, end-to-end `nanochat` pipeline.

The project is built around two artifacts:

- [nanochat-run.ipynb](./nanochat-run.ipynb), the notebook you run on Kaggle
- [kaggle_dataset/](./kaggle_dataset), the local Kaggle dataset package included in this repo

The goal is pipeline validation, not training a strong final model. The notebook is intentionally small enough to exercise the full code path on Kaggle with `2x Tesla T4` GPUs.

## Acknowledgement

This repository builds on [karpathy/nanochat](https://github.com/karpathy/nanochat) and extends it with a Kaggle-oriented workflow, including dataset packaging, notebook execution, compact end-to-end validation, post-training branches, quantization, and serving entrypoints.

## What The Notebook Actually Does

The notebook is the source of truth for this repo. It runs the following stages:

1. verify the mounted Kaggle dataset and copy it into `/kaggle/working/nanochat`
2. configure caches under `/kaggle/working/nanochat_cache` and `/kaggle/working/huggingface_cache`
3. prompt for a Hugging Face token
4. install the Python dependencies needed in the Kaggle runtime
5. download a small pretraining slice with `python -m nanochat.dataset -n 8`
6. train and evaluate the tokenizer
7. pretrain a small `d8` base model
8. evaluate the base model on reduced Kaggle-friendly settings
9. download identity conversations and run chat SFT
10. evaluate the SFT chat model
11. install distillation extras and generate tiny teacher datasets
12. run a small distillation branch
13. generate preference data and run a tiny DPO branch
14. compare SFT, distill, and DPO outputs
15. quantize the distilled checkpoint and run a tiny AWQ smoke test
16. evaluate the quantized artifact
17. run tiny RL smoke tests for `chat_rl.py`, `chat_universal_rl.py`, and `chat_ppo.py`
18. print serving commands for full-precision and quantized web apps

The last two serving cells are intentionally non-blocking examples. They print the commands but do not launch the web servers automatically.

## Repo Layout

```text
.
├── README.md
├── nanochat-run.ipynb
└── kaggle_dataset/
    ├── dataset-metadata.json
    └── nanochat/
        ├── pyproject.toml
        ├── nanochat/
        ├── scripts/
        ├── tasks/
        └── tests/
```

One slightly unusual detail: the uploadable Kaggle dataset package is `kaggle_dataset/`, but the actual code lives inside `kaggle_dataset/nanochat/`. After publishing, Kaggle mounts that dataset as a folder named `nanochat-codes`, and the notebook copies that mounted content into `/kaggle/working/nanochat`.

## Dataset Setup

Before running the notebook, configure Kaggle with:

- GPU enabled
- internet enabled
- your uploaded `nanochat-codes` dataset attached

The dataset mapping is:

- local packaging directory in this repo: `kaggle_dataset/`
- code inside that package: `kaggle_dataset/nanochat/`
- Kaggle dataset name: `nanochat-codes`
- mounted Kaggle path pattern: `/kaggle/input/datasets/<your-kaggle-username>/nanochat-codes`

The notebook auto-detects the attached dataset from this mounted path pattern:

```python
/kaggle/input/datasets/<your-kaggle-username>/nanochat-codes
```

For example, with the author's Kaggle account:

```python
/kaggle/input/datasets/yshuaiqin/nanochat-codes
```

In this repository, the mounted Kaggle dataset `nanochat-codes` is expected to contain the same project files that live under [`kaggle_dataset/nanochat/`](./kaggle_dataset/nanochat).

Early in the notebook, you will also be prompted for `HF_TOKEN`.

## How To Publish The Dataset Bundle

This repo includes Kaggle dataset metadata at [kaggle_dataset/dataset-metadata.json](./kaggle_dataset/dataset-metadata.json).

For the author's Kaggle account, the dataset id is:

```text
yshuaiqin/nanochat-codes
```

If you publish under your own Kaggle account, the dataset id becomes:

```text
<your-kaggle-username>/nanochat-codes
```

Typical update flow:

```bash
kaggle datasets version -p kaggle_dataset -m "update nanochat Kaggle dataset" --dir-mode zip
```

This command publishes the local packaging directory `kaggle_dataset/`. After publishing, attach the resulting Kaggle dataset `nanochat-codes` to the notebook. The notebook then reads the mounted dataset from `/kaggle/input/datasets/<your-kaggle-username>/nanochat-codes`.

Then push the notebook separately as a Kaggle kernel that depends on that dataset.

## Key Runtime Choices In The Notebook

The notebook makes a few environment assumptions explicitly:

- `NANOCHAT_DTYPE=float16` for Tesla T4 compatibility
- `torchrun --nproc_per_node=2` for Kaggle's 2 GPUs
- short iteration counts and reduced eval sizes to keep the run practical
- `--run=dummy` in training stages so the notebook does not require a Weights & Biases login

The base-model stage uses a small `d8` configuration with conservative settings such as:

```text
--depth=8
--device-batch-size=2
--max-seq-len=1024
--num-iterations=50
```

The chat SFT stage uses:

```text
--device-batch-size=2
--total-batch-size=4096
--num-iterations=20
--mmlu-epochs=1
--gsm8k-epochs=1
```

That `--total-batch-size=4096` choice is important in this notebook because it gives the intended effective step count under gradient accumulation.

## Kaggle-Specific SFT Fix

The notebook does not use the default SFT entrypoint. It uses [kaggle_dataset/nanochat/scripts/chat_sft_updated.py](./kaggle_dataset/nanochat/scripts/chat_sft_updated.py) instead.

This variant adds a single-rank warmup for Hugging Face-backed SFT datasets before distributed workers begin loading them. In notebook environments like Kaggle, that reduces failures caused by multiple workers trying to create caches or download data at the same time.

## What This Repo Validates

This repository is best understood as a small-scale systems check for the full training and post-training pipeline.

Validated by the notebook:

- tokenizer training and evaluation
- base-model train and eval flow
- chat SFT and chat eval flow
- distillation data generation and student training
- preference data generation and DPO flow
- post-training comparison reports
- quantization and quantized evaluation
- RL code-path smoke tests
- serving entrypoints for full-precision and quantized models

Not claimed by this repo:

- strong benchmark performance
- large-scale training quality
- production-ready checkpoints
- quality conclusions from the tiny Kaggle-scale post-training runs

## If You Want To Inspect Or Modify The Pipeline

Start with [nanochat-run.ipynb](./nanochat-run.ipynb). It already contains the exact commands, paths, and reduced settings used for the Kaggle validation run, so it is the best place to adjust batch sizes, iteration counts, model tags, or which stages you want to keep.
