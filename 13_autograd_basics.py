from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lab_utils import best_device, seed_everything


def parse_args() -> argparse.Namespace:
    # 通过命令行参数指定实验报告的输出路径；不传参数时使用默认路径。
    parser = argparse.ArgumentParser(description="Verify autograd gradients and accumulation semantics.")
    parser.add_argument("--output", default="artifacts/week03/autograd_basics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 固定随机种子，保证实验在重复运行时尽量得到一致的结果。
    # 本实验没有随机张量，但统一设置种子有助于保持实验模板的一致性。
    seed = 2026
    seed_everything(seed)

    # 优先选择可用的加速设备；没有加速设备时回退到 CPU。
    # 所有张量放在同一设备上，避免设备不一致错误。
    device = best_device()

    # A one-example linear model keeps the chain rule visible by hand.
    # 这个最小例子对应：prediction = weight * x + bias，
    # loss = 1/2 * (prediction - target)^2。每一步都可以手算，
    # 便于把 Autograd 的结果与链式法则进行核对。
    x = torch.tensor(2.0, device=device)
    target = torch.tensor(5.0, device=device)

    # requires_grad=True 表示需要记录相关计算，并在 backward() 后
    # 计算这些参数对 loss 的梯度；weight 和 bias 是叶子张量。
    weight = torch.tensor(1.5, device=device, requires_grad=True)
    bias = torch.tensor(0.25, device=device, requires_grad=True)

    # 前向计算会动态记录计算图：weight、bias、x -> prediction -> loss。
    prediction = weight * x + bias

    # prediction 是运算产生的非叶子张量。默认只保留叶子张量的 .grad；
    # retain_grad() 让我们也能在反向传播后观察这个中间节点的梯度。
    prediction.retain_grad()
    loss = 0.5 * (prediction - target) ** 2

    # 从 loss 出发沿计算图反向应用链式法则，结果写入 weight.grad、
    # bias.grad，以及因 retain_grad() 而保留下来的 prediction.grad。
    loss.backward()

    # detach() 只保留 residual 的数值，不让这段手算过程连接回原计算图。

    residual = prediction.detach() - target
    manual_weight_grad = residual * x
    manual_bias_grad = residual

    # 根据链式法则，dloss/dprediction = residual，
    # dloss/dweight = residual * x，dloss/dbias = residual。
    autograd_weight_grad = weight.grad.detach().clone()
    autograd_bias_grad = bias.grad.detach().clone()
    assert torch.allclose(autograd_weight_grad, manual_weight_grad)
    assert torch.allclose(autograd_bias_grad, manual_bias_grad)

    # allclose 允许浮点数存在微小误差，用于验证自动微分与手算结果一致。
    assert prediction.grad is not None
    assert torch.allclose(prediction.grad, residual)
    assert weight.is_leaf and bias.is_leaf
    assert not prediction.is_leaf

    # A second backward pass adds to .grad until the caller clears it.
    # backward() 默认累加梯度而不是覆盖；先清零，保证下面从干净状态开始。
    weight.grad.zero_()
    bias.grad.zero_()

    # 第一次损失使用原目标值 target。
    first_loss = (weight * x + bias - target).pow(2)
    first_loss.backward()

    # 保存第一次反向传播的梯度，供后面验证累加结果。
    first_weight_grad = weight.grad.detach().clone()
    first_bias_grad = bias.grad.detach().clone()
    # 第二次 backward() 前没有清零，因此它的梯度会加到第一次结果上。
    second_loss = (weight * x + bias - (target + 1.0)).pow(2)
    second_loss.backward()
    accumulated_weight_grad = weight.grad.detach().clone()
    accumulated_bias_grad = bias.grad.detach().clone()

    # 下面两个断言显式验证：累计梯度 = 第一次梯度 + 第二次梯度。
    assert torch.allclose(accumulated_weight_grad, first_weight_grad + (2 * (weight.detach() * x + bias.detach() - (target + 1.0)) * x))
    assert torch.allclose(accumulated_bias_grad, first_bias_grad + (2 * (weight.detach() * x + bias.detach() - (target + 1.0))))

    # 设为 None 表示当前没有保存梯度，与数值全为 0 的张量不同。
    weight.grad = None
    bias.grad = None
    assert weight.grad is None and bias.grad is None

    # 组织可读的实验报告；float(...) 将标量 Tensor 转成 JSON 可序列化的数字。
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

    # 确保输出目录存在，再把报告写入 JSON 文件。
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
