from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lab_utils import best_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify autograd gradients and accumulation semantics.")
    parser.add_argument("--output", default="artifacts/week03/autograd_basics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = 2026
    seed_everything(seed)
    device = best_device()

    # A one-example linear model keeps the chain rule visible by hand.
    x = torch.tensor(2.0, device=device)
    target = torch.tensor(5.0, device=device)
    weight = torch.tensor(1.5, device=device, requires_grad=True)
    bias = torch.tensor(0.25, device=device, requires_grad=True)
    prediction = weight * x + bias
    prediction.retain_grad()
    loss = 0.5 * (prediction - target) ** 2
    loss.backward()

    residual = prediction.detach() - target
    manual_weight_grad = residual * x
    manual_bias_grad = residual
    autograd_weight_grad = weight.grad.detach().clone()
    autograd_bias_grad = bias.grad.detach().clone()
    assert torch.allclose(autograd_weight_grad, manual_weight_grad)
    assert torch.allclose(autograd_bias_grad, manual_bias_grad)
    assert prediction.grad is not None
    assert torch.allclose(prediction.grad, residual)
    assert weight.is_leaf and bias.is_leaf
    assert not prediction.is_leaf

    # A second backward pass adds to .grad until the caller clears it.
    weight.grad.zero_()
    bias.grad.zero_()
    first_loss = (weight * x + bias - target).pow(2)
    first_loss.backward()
    first_weight_grad = weight.grad.detach().clone()
    first_bias_grad = bias.grad.detach().clone()
    second_loss = (weight * x + bias - (target + 1.0)).pow(2)
    second_loss.backward()
    accumulated_weight_grad = weight.grad.detach().clone()
    accumulated_bias_grad = bias.grad.detach().clone()
    assert torch.allclose(accumulated_weight_grad, first_weight_grad + (2 * (weight.detach() * x + bias.detach() - (target + 1.0)) * x))
    assert torch.allclose(accumulated_bias_grad, first_bias_grad + (2 * (weight.detach() * x + bias.detach() - (target + 1.0))))

    weight.grad = None
    bias.grad = None
    assert weight.grad is None and bias.grad is None

    report = {
        "seed": seed,
        "device": str(device),
        "inputs": {"x": float(x), "target": float(target), "weight": 1.5, "bias": 0.25},
        "loss": float(loss.detach()),
        "prediction": float(prediction.detach()),
        "manual_grad": {"weight": float(manual_weight_grad), "bias": float(manual_bias_grad)},
        "autograd_grad": {"weight": float(autograd_weight_grad), "bias": float(autograd_bias_grad)},
        "leaf_and_non_leaf": {
            "weight_is_leaf": weight.is_leaf,
            "bias_is_leaf": bias.is_leaf,
            "prediction_is_leaf": prediction.is_leaf,
            "prediction_grad_after_retain": float(prediction.grad),
        },
        "accumulation": {
            "first_weight_grad": float(first_weight_grad),
            "first_bias_grad": float(first_bias_grad),
            "after_two_backward_weight_grad": float(accumulated_weight_grad),
            "after_two_backward_bias_grad": float(accumulated_bias_grad),
            "cleared_with_none": weight.grad is None and bias.grad is None,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
