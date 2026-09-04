from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lab_utils import best_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ten Tensor layout and dtype assertions.")
    parser.add_argument("--output", default="artifacts/week02/tensor_layout.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(2026)
    device = best_device()
    base = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
    checks = {
        "numel_matches_shape": base.numel() == 2 * 3 * 4,
        "dtype_is_float32": base.dtype == torch.float32,
        "device_is_expected": base.device == device,
        "transpose_changes_shape": base.transpose(1, 2).shape == (2, 4, 3),
        "transpose_preserves_numel": base.transpose(1, 2).numel() == base.numel(),
        "slice_reduces_dimension": base[:, :, ::2].shape == (2, 3, 2),
        "view_preserves_storage": base.view(6, 4).data_ptr() == base.data_ptr(),
        "contiguous_has_dense_stride": base.transpose(1, 2).contiguous().is_contiguous(),
        "clone_is_independent": base.clone().data_ptr() != base.data_ptr(),
        "flatten_preserves_numel": base.flatten().numel() == base.numel(),
    }
    assert all(checks.values()), checks
    report = {"device": str(device), "shape": list(base.shape), "stride": list(base.stride()), "checks": checks}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
