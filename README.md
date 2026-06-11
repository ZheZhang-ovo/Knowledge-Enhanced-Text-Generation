# Knowledge-Enhanced Text Generation

This repository contains my project on knowledge-enhanced text generation.
It explores graph-enhanced BART models for generating text with citation and
concept knowledge.

## Features

- BART-based text generation
- Citation graph integration
- Concept graph integration
- Oracle-based training and evaluation

## Project Structure

- `new_finetune.py`: trains the knowledge-enhanced generation model
- `new_generate.py`: generates predictions from a trained model
- `new_graphbart.py`: implements the graph-enhanced BART model
- `args.py`: defines command-line configuration
- `utils/`: contains data, model, and generation utilities

## Data and Models

Datasets, cached files, generated outputs, and model checkpoints are excluded
from this repository because some of them exceed GitHub's file-size limits.
Place downloaded data in `data/` and trained checkpoints in `trained_models/`.

## Training

Train the baseline model:

```bash
python new_finetune.py --source oracle --gpu 0
```

Train with a citation graph:

```bash
python new_finetune.py --source oracle --gpu 0 --citation_graph --prepend
```

Train with a concept graph:

```bash
python new_finetune.py --source oracle --gpu 0 --concept_graph
```

Train with both citation and concept graphs:

```bash
python new_finetune.py --source oracle --gpu 0 --citation_graph --prepend --concept_graph
```

## Generation

```bash
python new_generate.py --source oracle --gpu 0 --model_name bart.pthbest
```

The generated predictions are written to the configured output directory.

## Author

Zhe Zhang
