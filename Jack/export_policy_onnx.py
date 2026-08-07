#!/usr/bin/env python3
"""Export the distilled PyTorch policy checkpoint to ONNX for Android/desktop ONNX Runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from jack_game.bc import FEATURE_DIM
from jack_game.policy import PolicyNet


def _embed_onnx_weights(output_path: Path) -> None:
    """Merge external .onnx.data into a single file for Android packaging."""
    sidecar = Path(f"{output_path}.data")
    model = onnx.load(str(output_path), load_external_data=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)
    if sidecar.exists():
        sidecar.unlink()


def export(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_dim = int(checkpoint.get("feature_dim", FEATURE_DIM))
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    num_layers = int(checkpoint.get("num_layers", 2))
    model = PolicyNet(input_dim=feature_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, feature_dim, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    _embed_onnx_weights(output_path)

    meta_path = output_path.with_suffix(output_path.suffix + ".json")
    meta_path.write_text(
        json.dumps(
            {
                "feature_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "source_checkpoint": str(checkpoint_path),
                "embedded_weights": True,
            },
            indent=2,
        )
    )
    size_kb = output_path.stat().st_size / 1024
    print(f"Wrote {output_path} ({size_kb:.0f} KB, weights embedded)")
    print(f"Wrote {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Jack5 policy checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/policy_distill800_l3.pt",
        help="PyTorch policy checkpoint",
    )
    parser.add_argument(
        "--out",
        default="checkpoints/policy_distill800_l3.onnx",
        help="Output ONNX path",
    )
    args = parser.parse_args()
    export(Path(args.checkpoint), Path(args.out))


if __name__ == "__main__":
    main()
