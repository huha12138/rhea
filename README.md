

# Rhea

This repository contains the anonymous implementation of **Rhea: Role-aware Heuristic Episodic Attention for Conversational LLMs**, a role-aware memory framework for improving long-horizon multi-turn conversations with large language models.

Rhea separates persistent global instructions from episodic dialogue history and reconstructs a compact context for inference.

## Repository Layout

- `inference.py`: runs model inference for supported MTEval tasks.
- `evaluate.py`: performs LLM-as-judge evaluation for generated outputs.
- `calculate_score.py`: aggregates metrics and writes report tables.
- `create_data.py`: builds task-specific data files from raw sources.
- `utils/`: contains prompt templates, model configuration, parsing, logging, and metric helpers.
- `prompts/`: contains reusable prompts for data construction and evaluation.
- `data/`: contains packaged benchmark data used by the scripts.

## Requirements

### Hardware

- NVIDIA RTX 3090 or higher is recommended.
- For long-context evaluation, GPUs with larger memory are preferred.

### Software

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Preparation

Download the required models from Hugging Face or use your local checkpoints.

Example directory structure:

```text
models/
├── Qwen3-0.6B/
└── Mistral-7B-Instruct-v0.2/
```

Before running the code, please check and modify the model paths in `run.sh` if necessary.

Model and judge paths can also be configured with environment variables:

```bash
export RHEA_INSTRUCTION_MODEL=Qwen/Qwen3-0.6B
export RHEA_MODEL_PATH=checkpoints/rhea-only-assist-max
export RHEA_LOCAL_JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
export RHEA_JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct
```

For API-hosted judge models such as GPT models, set:

```bash
export OPENAI_API_KEY=<your-api-key>
export RHEA_JUDGE_MODEL=gpt-4o-mini
```

If your API endpoint is OpenAI-compatible but self-hosted or proxied, set `OPENAI_BASE_URL` as needed.

## Data

The packaged `data/MTEval/` directory contains the benchmark splits used by inference. Some data-construction and judge-evaluation utilities additionally expect anonymized source files under `raw_data/`, including `raw_data/documents.jsonl`.

If `raw_data/` is not distributed with the review package, use the packaged splits for inference only, or place the anonymized raw source files in the expected paths before running `create_data.py` or `evaluate.py`.

## Run

The main entry script is:

```bash
bash run.sh
```

You can also specify visible GPUs manually:

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

Common runtime options can be overridden through environment variables:

```bash
MODEL_NAME=rhea-only-assist-max \
TASKS="refinement_multi expansion_multi" \
MAX_NEW_TOKENS=512 \
bash run.sh
```

## Results

Generated outputs and evaluation results are written to the directories configured in `utils/constants.py`:

- `inf/` for inference outputs.
- `eval/` for judge-model evaluations.
- `results/` for aggregated score reports.

Typical output structure:

```text
results/
├── raw_result.csv
└── result.md
```

Please check the `results/` directory after running `run.sh`.

## Notes

- Make sure model identifiers or local checkpoint paths are correctly configured.
- The Rhea checkpoint is expected at `RHEA_MODEL_PATH`; update this variable if your checkpoint is stored elsewhere.
- Long-context experiments may require substantial GPU memory.
- The default entry point for reproducing the main experiment is `run.sh`.
- Runtime directories such as `inf/`, `eval/`, `log/`, `output/`, and `results/` are ignored by default and should not be included in double-blind submissions.
