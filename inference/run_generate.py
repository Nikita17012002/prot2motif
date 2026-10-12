import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm

from inference.model import AttnDecoderRNN, EncoderRNN, Seq2Seq
from inference.tokenizer import AminoAcidTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "model" / "prot2motif_model.pt"
DEFAULT_CLASSIFIER_PATH = ROOT / "model" / "classifier_LogisticRegression_decoder.pkl"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "generated_results.npz"


def load_model(model_path: Path, device: torch.device) -> Seq2Seq:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a model state dict: {model_path}")

    model = Seq2Seq(
        EncoderRNN(hidden_size=256),
        AttnDecoderRNN(hidden_size=256, output_size=5),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_sequences(
    fasta_path: Path, limit: Optional[int] = None
) -> List[Tuple[str, str]]:
    sequences = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        sequences.append((record.id, str(record.seq)))
        if limit is not None and len(sequences) >= limit:
            break
    return sequences


def classify_batch(
    model: Seq2Seq, batch_tokens: torch.Tensor, classifier, device: torch.device
) -> np.ndarray:
    with torch.inference_mode():
        encoder_output, _, _ = model.encoder(batch_tokens.to(device))
        features = encoder_output.mean(dim=1)

    return classifier.predict_proba(features.cpu().numpy().astype(np.float32))[:, 1].astype(
        np.float32
    )


def generate_motifs_batch(
    model: Seq2Seq, batch_tokens: torch.Tensor, device: torch.device
) -> np.ndarray:
    with torch.inference_mode():
        output, _, _ = model(batch_tokens.to(device))
        motifs = torch.softmax(output, dim=-1)
    return motifs.cpu().numpy().astype(np.float16)


def save_compact_results(
    output_path: Path,
    protein_ids: List[str],
    probabilities: np.ndarray,
    motifs: np.ndarray,
    positive_mask: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        protein_ids=np.asarray(protein_ids, dtype="U"),
        probabilities=np.asarray(probabilities, dtype=np.float16),
        motifs=np.asarray(motifs, dtype=np.float16),
        positive_mask=np.asarray(positive_mask, dtype=bool),
        positive_indices=np.flatnonzero(positive_mask).astype(np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict protein motif profiles using the pretrained Prot2Motif models."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Model checkpoint.")
    parser.add_argument(
        "--classifier", type=Path, default=DEFAULT_CLASSIFIER_PATH, help="Classifier pickle."
    )
    parser.add_argument("--fasta", type=Path, required=True, help="Input protein FASTA file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output .npz file.")
    parser.add_argument("--batch-size", type=int, default=64, help="Proteins per inference batch.")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold in [0, 1]."
    )
    parser.add_argument(
        "--max-proteins", type=int, default=None, help="Optional limit for a quick test run."
    )
    args = parser.parse_args()

    for label, path in (("Model", args.model), ("Classifier", args.classifier), ("FASTA", args.fasta)):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero.")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")
    if args.max_proteins is not None and args.max_proteins < 1:
        parser.error("--max-proteins must be greater than zero.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    classifier = joblib.load(args.classifier)
    records = load_sequences(args.fasta, limit=args.max_proteins)
    if not records:
        parser.error(f"No protein sequences found in FASTA file: {args.fasta}")

    protein_ids = [protein_id for protein_id, _ in records]
    sequences = [sequence for _, sequence in records]
    tokenizer = AminoAcidTokenizer()
    probabilities = np.empty(len(sequences), dtype=np.float32)
    positive_mask = np.zeros(len(sequences), dtype=bool)
    motifs = np.zeros((len(sequences), 26, 5), dtype=np.float16)

    for start in tqdm(range(0, len(sequences), args.batch_size), desc="Generating motifs"):
        end = min(start + args.batch_size, len(sequences))
        batch_tokens = tokenizer.batch_encode(sequences[start:end], max_len=1024, padding=True)
        batch_probabilities = classify_batch(model, batch_tokens, classifier, device)
        probabilities[start:end] = batch_probabilities

        batch_positive_mask = batch_probabilities >= args.threshold
        positive_mask[start:end] = batch_positive_mask
        if np.any(batch_positive_mask):
            positive_positions = np.flatnonzero(batch_positive_mask)
            motifs[start + positive_positions] = generate_motifs_batch(
                model, batch_tokens[positive_positions], device
            )

    save_compact_results(args.output, protein_ids, probabilities, motifs, positive_mask)
    print(f"Saved {len(protein_ids)} proteins to {args.output}")
    print(f"Positive proteins above threshold {args.threshold}: {int(positive_mask.sum())}")


if __name__ == "__main__":
    main()
