# Nanochat End-to-End LLM Pipeline

This repository packages a Kaggle-focused end-to-end LLM workflow built around the `nanochat` codebase.

The project is centered on two artifacts:

- [nanochat-run.ipynb](./nanochat-run.ipynb): the Kaggle notebook that runs the pipeline
- [kaggle_dataset/](./kaggle_dataset): the Kaggle dataset bundle mounted by the notebook

The goal is not a large-scale final model. This project is a working small-scale pipeline that validates the full code path on Kaggle with `2x Tesla T4` GPUs.

## What The Notebook Covers

The notebook walks through:

1. downloading a small pretraining slice
2. training and evaluating the tokenizer
3. base-model pretraining and evaluation
4. supervised fine-tuning
5. chat evaluation
6. distillation data generation and distillation
7. preference data generation and DPO
8. post-training comparison
9. quantization and quantized evaluation
10. RL branch smoke tests
11. serving entrypoints
12. AWQ code-path testing

## Repository Layout

```text
.
├── README.md
├── nanochat-run.ipynb
└── kaggle_dataset
    ├── dataset-metadata.json
    └── nanochat
        ├── pyproject.toml
        ├── nanochat
        ├── scripts
        ├── tasks
        └── tests
        
```

## Kaggle Workflow

### 1. Upload the dataset

The notebook expects the Kaggle dataset bundle contained in [kaggle_dataset/](./kaggle_dataset).

From a terminal with the Kaggle CLI configured:

```bash
kaggle datasets version -p kaggle_dataset -m "update nanochat Kaggle dataset" --dir-mode zip
```

This bundle is intended to be published as:

- dataset slug: `yshuaiqin/nanochat-codes`

In Kaggle, the mounted path used by the notebook is:

```python
/kaggle/input/datasets/yshuaiqin/nanochat-codes
```

### 2. Upload the notebook

Push [nanochat-run.ipynb](./nanochat-run.ipynb) as a Kaggle notebook that depends on the dataset above.

Typical CLI flow:

```bash
kaggle kernels push -p <your-notebook-folder>
```

The notebook should be configured with:

- GPU enabled
- internet enabled
- dataset source `yshuaiqin/nanochat-codes`

### 3. Run the notebook on Kaggle

Inside Kaggle, the notebook:

- copies the mounted dataset repo into `/kaggle/working/nanochat`
- uses `/kaggle/working/nanochat_cache` as the main cache/checkpoint directory
- uses `/kaggle/working/huggingface_cache` for Hugging Face caches
- sets `NANOCHAT_DTYPE=float16` for `Tesla T4`

## Kaggle-Specific Notes

### SFT startup fix

The standard SFT path was fragile in the Kaggle notebook environment because distributed workers could stall during Hugging Face dataset loading.

To make this reliable, the dataset bundle includes:

- [kaggle_dataset/nanochat/scripts/chat_sft_updated.py](./kaggle_dataset/nanochat/scripts/chat_sft_updated.py)

This version adds a Kaggle-safe dataset warmup path before distributed SFT starts.

### Important SFT batch setting

For the validated `d8` / `2x T4` setup, the notebook uses:

```bash
--device-batch-size=2
--total-batch-size=4096
--num-iterations=20
```

The `--total-batch-size=4096` setting matters because it makes the effective SFT iteration count behave as intended in this environment.

## Validated Small-Scale Pipeline

This repository captures a successful small-scale validation run, not a fully trained production model.

What is validated:

- each major pipeline stage runs successfully on Kaggle
- checkpoints are written correctly
- evaluation reports are saved correctly
- post-training branches execute successfully at smoke-test scale

What is not claimed:

- strong benchmark performance
- large-scale training quality
- final-model readiness

