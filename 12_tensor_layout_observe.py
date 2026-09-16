from __future__ import annotations

import argparse
import json
import time

import torch

from lab_utils import best_device, format_bytes, percentile, seed_everything, synchronize, write_json


def observe_slice_contiguous(device: torch.device) -> dict[str, object]:
    """练习一：把非连续张量传给需要连续输入的算子，比较调用前后的布局。"""
    base = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
    sliced = base[:, :, ::2]  # 步长切片 -> 非连续，但仍共享 storage。
    transposed = base.transpose(1, 2)  # 转置 -> 非连续，stride 跨越不连续子空间。

    # ``view`` 并非总在非连续时失败：只要目标形状能用现有 stride 表达就会成功。
    # transpose 后的布局无法用 ``view`` 展平，会抛 RuntimeError，需改用 ``reshape``。
    view_on_transposed_failed = False
    try:
        transposed.view(-1)
    except RuntimeError:
        view_on_transposed_failed = True

    # ``reshape`` 总能成功，但对非连续输入会复制到新 storage。
    reshaped = transposed.reshape(-1)
    dense = sliced.contiguous()  # 复制到新的密集 storage。

    return {
        "sliced_shape": list(sliced.shape),
        "sliced_stride": list(sliced.stride()),
        "sliced_is_contiguous": sliced.is_contiguous(),
        "sliced_shares_storage_with_base": sliced.data_ptr() == base.data_ptr(),
        "transposed_is_contiguous": transposed.is_contiguous(),
        "view_on_transposed_failed": view_on_transposed_failed,
        "reshape_shares_storage_with_base": reshaped.data_ptr() == base.data_ptr(),
        "dense_is_contiguous": dense.is_contiguous(),
        "dense_shares_storage_with_base": dense.data_ptr() == base.data_ptr(),
        "conclusion": (
            "view 不总在非连续时失败，但 transpose 后的展平会失败并要求 reshape；"
            "reshape/contiguous 对非连续输入会复制：is_contiguous 变 True，data_ptr 随之改变。"
        ),
    }


def observe_byte_budget(device: torch.device) -> dict[str, object]:
    """练习二：比较逻辑字节数与 contiguous 需要搬运的字节数。"""
    base = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
    transposed = base.transpose(1, 2)  # 非连续，numel 不变。

    logical_bytes = base.element_size() * base.numel()
    moved_bytes = transposed.element_size() * transposed.numel()

    return {
        "element_size": base.element_size(),
        "numel": base.numel(),
        "logical_bytes": logical_bytes,
        "transposed_stride": list(transposed.stride()),
        "contiguous_stride": list(transposed.contiguous().stride()),
        "bytes_read_plus_write": moved_bytes * 2,
        "conclusion": (
            "搬运字节数等于逻辑字节数（每元素读写一遍），numel 不变；"
            "真正的成本来自非连续 stride 导致的跨步访存。"
        ),
    }


def observe_copy_scaling(device: torch.device, scales: list[int], repeats: int) -> list[dict[str, object]]:
    """练习三：放大 shape，观察 contiguous 复制耗时如何变化。"""
    rows: list[dict[str, object]] = []
    for scale in scales:
        base = torch.randn(2, 3 * scale, 4 * scale, device=device)
        transposed = base.transpose(1, 2)  # 非连续。
        assert not transposed.is_contiguous()

        transposed.contiguous()  # 预热，避免首次分配影响计时。
        synchronize(device)

        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            transposed.contiguous()
            synchronize(device)
            samples.append(time.perf_counter() - start)

        logical_bytes = base.element_size() * base.numel()
        rows.append(
            {
                "scale": scale,
                "shape": list(base.shape),
                "numel": base.numel(),
                "logical_bytes": logical_bytes,
                "logical_size": format_bytes(logical_bytes),
                "median_ms": round(percentile(samples, 0.5) * 1e3, 4),
                "p90_ms": round(percentile(samples, 0.9) * 1e3, 4),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run three Tensor layout observation experiments.")
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="artifacts/week02/tensor_layout_observe.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(2026)
    device = best_device()

    report = {
        "device": str(device),
        "exercise_1_slice_contiguous": observe_slice_contiguous(device),
        "exercise_2_byte_budget": observe_byte_budget(device),
        "exercise_3_copy_scaling": observe_copy_scaling(device, args.scales, args.repeats),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    write_json(args.output, report)


if __name__ == "__main__":
    main()

