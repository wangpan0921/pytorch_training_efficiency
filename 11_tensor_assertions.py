from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lab_utils import best_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Tensor contracts used by a training batch.")
    parser.add_argument("--output", default="artifacts/week02/tensor_assertions.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(2026)
    device = best_device()
    batch, channels, height, width = 4, 3, 8, 8
    images = torch.randn(batch, channels, height, width, device=device, dtype=torch.float32)
    labels = torch.randint(0, 10, (batch,), device=device, dtype=torch.long)
    logits = torch.randn(batch, 10, device=device)
    checks = {
        "images_shape": tuple(images.shape) == (batch, channels, height, width),
        "images_float": images.dtype.is_floating_point,
        "labels_shape": tuple(labels.shape) == (batch,),
        "labels_integer": labels.dtype == torch.long,
        "same_device_images_labels": images.device == labels.device,
        "same_device_images_logits": images.device == logits.device,
        "batch_dimension_matches": images.shape[0] == labels.shape[0] == logits.shape[0],
        "class_dimension_matches": logits.shape[1] == 10,
        "finite_images": bool(torch.isfinite(images).all()),
        "finite_logits": bool(torch.isfinite(logits).all()),
    }
    assert all(checks.values()), checks
    report = {"device": str(device), "checks": checks, "image_bytes": images.numel() * images.element_size()}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
