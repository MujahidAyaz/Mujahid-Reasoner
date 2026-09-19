from __future__ import annotations

import pytest
import torch

from src.training.runtime import TrainingRuntime


def test_auto_runtime_uses_available_device() -> None:
    runtime = TrainingRuntime(
        device="auto",
        precision="auto",
    )

    if torch.cuda.is_available():
        assert runtime.device.type == "cuda"
    elif runtime._mps_available():
        assert runtime.device.type == "mps"
    else:
        assert runtime.device.type == "cpu"


def test_cpu_fp32_runtime() -> None:
    runtime = TrainingRuntime(
        device="cpu",
        precision="fp32",
    )

    assert runtime.device.type == "cpu"
    assert runtime.precision == torch.float32
    assert runtime.autocast_enabled is False
    assert runtime.scaler_enabled is False


def test_cpu_auto_precision_is_fp32() -> None:
    runtime = TrainingRuntime(
        device="cpu",
        precision="auto",
    )

    assert runtime.precision == torch.float32
    assert runtime.autocast_enabled is False
    assert runtime.scaler_enabled is False


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        TrainingRuntime(
            device="invalid",
            precision="auto",
        )


def test_invalid_precision_is_rejected() -> None:
    with pytest.raises(ValueError):
        TrainingRuntime(
            device="cpu",
            precision="invalid",
        )


def test_cpu_fp16_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="FP16"):
        TrainingRuntime(
            device="cpu",
            precision="fp16",
        )


def test_cuda_request_requires_cuda() -> None:
    if torch.cuda.is_available():
        runtime = TrainingRuntime(
            device="cuda",
            precision="fp32",
        )

        assert runtime.device.type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA"):
            TrainingRuntime(
                device="cuda",
                precision="fp32",
            )


def test_mps_request_requires_mps() -> None:
    if TrainingRuntime._mps_available():
        runtime = TrainingRuntime(
            device="mps",
            precision="fp32",
        )

        assert runtime.device.type == "mps"
    else:
        with pytest.raises(RuntimeError, match="MPS"):
            TrainingRuntime(
                device="mps",
                precision="fp32",
            )


def test_runtime_info_is_consistent() -> None:
    runtime = TrainingRuntime(
        device="cpu",
        precision="fp32",
    )

    info = runtime.info()

    assert info.device == runtime.device
    assert info.precision == runtime.precision
    assert info.requested_device == "cpu"
    assert info.requested_precision == "fp32"
    assert info.is_cpu is True
    assert info.is_cuda is False
    assert info.is_mps is False


def test_cpu_autocast_context_is_disabled() -> None:
    runtime = TrainingRuntime(
        device="cpu",
        precision="fp32",
    )

    with runtime.autocast_context():
        tensor = torch.randn(4, 4)
        result = tensor @ tensor

    assert result.dtype == torch.float32


def test_cpu_has_no_grad_scaler() -> None:
    runtime = TrainingRuntime(
        device="cpu",
        precision="fp32",
    )

    assert runtime.create_grad_scaler() is None


def test_runtime_summary_contains_core_information() -> None:
    runtime = TrainingRuntime(
        device="cpu",
        precision="fp32",
    )

    summary = runtime.summary()

    assert "device=cpu" in summary
    assert "precision=torch.float32" in summary
    assert "autocast=False" in summary
    assert "grad_scaler=False" in summary