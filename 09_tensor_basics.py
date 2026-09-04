from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lab_utils import best_device, seed_everything


def tensor_info(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "is_contiguous": value.is_contiguous(),
        "data_ptr": value.data_ptr(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect tensor layout and copy behavior.")
    parser.add_argument("--output", default="artifacts/week02/tensor_basics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(2026)
    device = best_device()
    base = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    transposed = base.t()
    viewed = base.view(2, 6)
    contiguous = transposed.contiguous()
    cloned = base.clone()
    assert viewed.data_ptr() == base.data_ptr()
    assert transposed.data_ptr() == base.data_ptr()
    assert not transposed.is_contiguous()
    assert contiguous.data_ptr() != transposed.data_ptr()
    assert cloned.data_ptr() != base.data_ptr()
    assert torch.equal(contiguous, transposed)
    base[0, 0] = 99
    assert viewed[0, 0].item() == 99
    assert cloned[0, 0].item() == 0
    report = {
        "device": str(device),
        "base": tensor_info(base),
        "transposed": tensor_info(transposed),
        "viewed": tensor_info(viewed),
        "contiguous": tensor_info(contiguous),
        "cloned": tensor_info(cloned),
        "checks": {
            "view_shares_storage": viewed.data_ptr() == base.data_ptr(),
            "transpose_shares_storage": transposed.data_ptr() == base.data_ptr(),
            "contiguous_copies": contiguous.data_ptr() != transposed.data_ptr(),
            "clone_copies": cloned.data_ptr() != base.data_ptr(),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
