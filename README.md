# Prot2Motif

Prot2Motif predicts a motif profile for each protein sequence in a FASTA file.
The repository contains the trained network checkpoint, its binary classifier,
and the inference implementation.

## Repository contents

- `model/prot2motif.pt` — trained prot2motif checkpoint.
- `prot2motif.pt` — checkpoint explicitly used by the example notebook.
- `model/classifier_LogisticRegression_decoder.pkl` — trained Logistic Regression
  classifier used to select proteins for motif generation.
- `inference/` — network architecture, amino-acid tokenizer, and inference CLI.
- `results/` — default destination for generated results (created on first run).

## Installation

Use Python 3.9.21. From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The installed PyTorch build can use CUDA when it is available; otherwise the
inference runs on CPU.

## Run inference

Provide a FASTA file containing protein sequences:

```bash
python -m inference.run_generate --fasta path/to/proteins.fasta
```

By default, the command loads the two model artifacts from `model/` and writes
`results/generated_results.npz`. Paths and runtime parameters can be overridden:

```bash
python -m inference.run_generate \
  --fasta path/to/proteins.fasta \
  --output results/my_predictions.npz \
  --batch-size 64 \
  --threshold 0.5
```

`--max-proteins 10` can be used for a small test run. If the model artifacts are
stored elsewhere, pass their paths with `--model` and `--classifier`.

The compressed NPZ output contains:

- `protein_ids`: FASTA record identifiers.
- `probabilities`: classifier probability for each sequence.
- `positive_mask` and `positive_indices`: sequences meeting the threshold.
- `motifs`: predicted `(26, 5)` probability profile for each sequence. Profiles
  for sequences below the threshold are zero-filled.

The classifier artifact is a Python pickle. Only load it from a trusted source.
