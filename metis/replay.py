"""Offline JSONL replay for user-provided QAC/CES state exports."""

import argparse
import json
from pathlib import Path

import torch

from .inference import MetisPredictor
from .model import ModelConfig, build_model
from .schema import CognitiveState


def replay(checkpoint: Path, input_path: Path, output_path: Path, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["state_dict"])
    model.choice_temperature = saved.get("choice_temperature", 1.0)
    predictor = MetisPredictor(model, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    compared = 0
    agreements = 0
    for line_number, line in enumerate(input_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        state = CognitiveState.from_dict(record.get("state", record))
        result = predictor.predict(state)
        historical = record.get("historical_choice")
        if historical is not None:
            compared += 1
            agreements += int(str(historical).upper() == result.choice.name)
        rows.append({"id": record.get("id", line_number), "choice": result.choice.name,
                     "choice_probabilities": dict(result.choice_probabilities),
                     "scores": dict(result.scores), "uncertainty": result.uncertainty,
                     "historical_choice": historical,
                     "agrees_with_history": None if historical is None else str(historical).upper() == result.choice.name})
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"input_count": len(rows), "output": str(output_path),
            "historical_comparison_count": compared,
            "historical_agreement_rate": agreements / compared if compared else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", type=Path, help="JSONL records with a state object")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(replay(args.checkpoint, args.input, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
