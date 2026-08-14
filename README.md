# Academic Knowledge-Network Learning

This repository contains the implementation of a dual-graph neural network for personalized future topic prediction. It builds a global knowledge network (GKN) from an observation window and combines it with each scholar's local knowledge network (LKN) to score candidate future topics.

The public release contains code and documentation only. It intentionally excludes datasets, scholar identifiers, model checkpoints, experimental outputs, notebooks with saved outputs, and manuscript materials.

## Included Workflows

- Controlled scholar-holdout or pair-stratified GNN evaluation.
- Shared candidate-pair structural-feature baselines.
- Validation-set threshold selection and scholar-cluster bootstrap summaries.
- Checkpoint-faithful node-occlusion XAI for the top and worst scholar-level cases.

## Installation

Install a PyTorch build appropriate for your CPU, CUDA, or ROCm environment first. Then install the core packages:

```bash
pip install -r requirements.txt
```

The optional feature-engineered baselines require additional packages:

```bash
pip install -r requirements-baselines.txt
```

For ROCm, install the ROCm-compatible PyTorch wheel before installing `torch-geometric`; consult the official PyTorch and PyG installation instructions for matching versions.

## Input Format

Provide two directories: one for the observation window and one for the prediction window. Each must contain files named `<scholar_id>.txt`. A file contains one undirected topic edge per line:

```text
TOPIC_A TOPIC_B
TOPIC_A TOPIC_C
```

Only scholars present in both directories are considered. The repository does not include example records because even small graph samples may expose research histories. See [input-format.md](docs/input-format.md) for details.

## Train and Evaluate

Run a GNN-only experiment without optional baselines:

```bash
python scripts/run_experiment.py \
  --obs-window 3 \
  --lkn-obs-dir /path/to/observation_lkns \
  --lkn-pred-dir /path/to/prediction_lkns \
  --lkn-scale 3000 \
  --output-dir outputs/obs3y_lkn3000 \
  --split-unit scholar \
  --pooling candidate-attention \
  --fusion rich \
  --skip-baselines \
  --device auto
```

Use `--excluded-topics-file` to remove generic terms and `--topic-id-regex` when input identifiers should be validated. Hyperparameters are exposed as CLI options, so the script has no data-specific path or tuning defaults.

To run the common-pair baselines as well, omit `--skip-baselines` after installing `requirements-baselines.txt`.

## Node-Occlusion XAI

After a completed GNN run, generate top-five and worst-five scholar explanations:

```bash
python scripts/run_occlusion_xai.py \
  --evaluation-root outputs \
  --windows 3 \
  --lkn-scale 3000 \
  --top-k 5 \
  --device auto
```

The XAI script reconstructs the dataset from the paths stored inside the ignored run directory, reloads the saved checkpoint, and measures the probability change caused by removing each node from the selected scholar LKN. Add `--topic-map /path/to/topic_map.csv --topic-id-column topic_id --topic-name-column topic_name` to show readable topic labels.

## Reproducibility and Privacy

- Train/validation/test splits, model settings, and thresholds are written under `outputs/`.
- `outputs/`, data files, checkpoints, logs, and common result formats are ignored by Git by default.
- Do not commit a completed run directory or raw LKN inputs to a public repository.

See [privacy.md](docs/privacy.md) before publishing derived artifacts.
