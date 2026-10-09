from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lab_utils import best_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare autograd with central finite differences.")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", default="artifacts/week03/finite_difference.json")
    return parser.parse_args()


def objective(parameters: torch.Tensor, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = parameters[:2]
    bias = parameters[2]
    residual = features @ weight + bias - target
    return 0.5 * (residual.square()).mean()


def main() -> None:
    args = parse_args()
    if args.epsilon <= 0 or args.tolerance <= 0:
        raise ValueError("epsilon and tolerance must be positive")
    seed = 2026
    seed_everything(seed)
    device = best_device()
    dtype = torch.float64
    features = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]], device=device, dtype=dtype)
    target = torch.tensor([2.0, -1.0, 0.5], device=device, dtype=dtype)
    parameters = torch.tensor([0.75, -0.4, 0.2], device=device, dtype=dtype, requires_grad=True)

    value = objective(parameters, features, target)
    value.backward()
    autograd_gradient = parameters.grad.detach().clone()

    finite_gradient = torch.empty_like(parameters)
    for index in range(parameters.numel()):
        plus = parameters.detach().clone()
        minus = parameters.detach().clone()
        plus[index] += args.epsilon
        minus[index] -= args.epsilon
        finite_gradient[index] = (objective(plus, features, target) - objective(minus, features, target)) / (2 * args.epsilon)

    absolute_error = (finite_gradient - autograd_gradient).abs()
    max_absolute_error = float(absolute_error.max())
    assert max_absolute_error <= args.tolerance, {
        "max_absolute_error": max_absolute_error,
        "tolerance": args.tolerance,
        "epsilon": args.epsilon,
    }

    report = {
        "seed": seed,
        "device": str(device),
        "dtype": str(dtype),
        "epsilon": args.epsilon,
        "tolerance": args.tolerance,
        "parameters": parameters.detach().tolist(),
        "features": features.tolist(),
        "target": target.tolist(),
        "objective": float(value.detach()),
        "autograd_gradient": autograd_gradient.tolist(),
        "finite_difference_gradient": finite_gradient.tolist(),
        "absolute_error": absolute_error.tolist(),
        "max_absolute_error": max_absolute_error,
        "passed": True,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
