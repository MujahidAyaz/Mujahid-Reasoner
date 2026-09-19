from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeInfo:
    """Resolved hardware and numerical runtime configuration."""

    device: torch.device
    requested_device: str

    precision: torch.dtype
    requested_precision: str

    autocast_enabled: bool
    scaler_enabled: bool

    cuda_available: bool
    mps_available: bool

    device_name: str

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"


class TrainingRuntime:
    """
    Resolve and configure the hardware/numerical runtime.

    Design goals:
        - explicit device selection
        - safe automatic device detection
        - precision resolution
        - BF16/FP16 capability checks
        - autocast support
        - GradScaler support
        - CUDA performance configuration

    The runtime deliberately does not silently downgrade an explicitly
    requested precision when that would change the user's experiment.
    """

    VALID_DEVICES = {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }

    VALID_PRECISIONS = {
        "auto",
        "fp32",
        "fp16",
        "bf16",
    }

    def __init__(
        self,
        *,
        device: str = "auto",
        precision: str = "auto",
        allow_tf32: bool = True,
        cudnn_benchmark: bool = True,
    ) -> None:
        self.requested_device = device
        self.requested_precision = precision

        self._validate_arguments()

        self.device = self._resolve_device()

        self.precision = self._resolve_precision()

        self._configure_cuda(
            allow_tf32=allow_tf32,
            cudnn_benchmark=cudnn_benchmark,
        )

        self.autocast_enabled = (
            self._should_enable_autocast()
        )

        self.scaler_enabled = (
            self.device.type == "cuda"
            and self.precision == torch.float16
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def info(self) -> RuntimeInfo:
        """Return an immutable description of the resolved runtime."""

        return RuntimeInfo(
            device=self.device,
            requested_device=self.requested_device,
            precision=self.precision,
            requested_precision=self.requested_precision,
            autocast_enabled=self.autocast_enabled,
            scaler_enabled=self.scaler_enabled,
            cuda_available=torch.cuda.is_available(),
            mps_available=self._mps_available(),
            device_name=self._device_name(),
        )

    def autocast_context(self):
        """
        Return the appropriate PyTorch autocast context.

        FP32 returns a no-op context.
        CUDA BF16/FP16 uses torch.autocast.
        MPS BF16/FP16 uses torch.autocast where supported.
        """

        if not self.autocast_enabled:
            return torch.autocast(
                device_type=self.device.type,
                enabled=False,
            )

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.precision,
            enabled=True,
        )

    def create_grad_scaler(self):
        """
        Create a GradScaler when required.

        GradScaler is useful for CUDA FP16 because FP16 has a narrower
        numerical range. BF16 normally does not require gradient scaling.
        """

        if not self.scaler_enabled:
            return None

        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

    def synchronize(self) -> None:
        """Synchronize asynchronous accelerator work when applicable."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        elif self.device.type == "mps":
            synchronize = getattr(
                torch.mps,
                "synchronize",
                None,
            )

            if synchronize is not None:
                synchronize()

    def summary(self) -> str:
        """Return a concise human-readable runtime summary."""

        info = self.info()

        return (
            f"device={info.device} "
            f"device_name={info.device_name} "
            f"precision={info.precision} "
            f"autocast={info.autocast_enabled} "
            f"grad_scaler={info.scaler_enabled}"
        )

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_device(self) -> torch.device:
        if self.requested_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")

            if self._mps_available():
                return torch.device("mps")

            return torch.device("cpu")

        if self.requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was explicitly requested, but CUDA "
                    "is not available."
                )

            return torch.device("cuda")

        if self.requested_device == "mps":
            if not self._mps_available():
                raise RuntimeError(
                    "MPS was explicitly requested, but MPS "
                    "is not available."
                )

            return torch.device("mps")

        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Precision resolution
    # ------------------------------------------------------------------

    def _resolve_precision(self) -> torch.dtype:
        if self.requested_precision == "fp32":
            return torch.float32

        if self.requested_precision == "fp16":
            self._validate_fp16_support()

            return torch.float16

        if self.requested_precision == "bf16":
            self._validate_bf16_support()

            return torch.bfloat16

        # Automatic precision selection.
        #
        # CUDA:
        #   Prefer BF16 on hardware that supports it because it has
        #   FP32-like exponent range and normally avoids GradScaler.
        #
        # MPS:
        #   Prefer FP16 for broad compatibility.
        #
        # CPU:
        #   Use FP32 as the conservative development default.
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16

            return torch.float16

        if self.device.type == "mps":
            return torch.float16

        return torch.float32

    def _validate_fp16_support(self) -> None:
        if self.device.type == "cpu":
            raise RuntimeError(
                "FP16 training is not supported by the current "
                "CPU runtime. Use fp32 or a supported accelerator."
            )

    def _validate_bf16_support(self) -> None:
        if self.device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "BF16 was explicitly requested, but the current "
                    "CUDA device does not report BF16 support."
                )

            return

        if self.device.type == "mps":
            # MPS BF16 support varies significantly by hardware and
            # PyTorch version. Let PyTorch determine support at runtime.
            return

        # CPU BF16 support is hardware/backend dependent. PyTorch can
        # execute BF16 CPU operations on supported systems, but it is
        # not a useful default for this project's current environment.
        if not hasattr(
            torch,
            "set_float32_matmul_precision",
        ):
            raise RuntimeError(
                "BF16 CPU support cannot be safely validated "
                "with this PyTorch runtime."
            )

    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------

    def _configure_cuda(
        self,
        *,
        allow_tf32: bool,
        cudnn_benchmark: bool,
    ) -> None:
        if self.device.type != "cuda":
            return

        # TF32 can significantly accelerate matrix multiplications on
        # supported NVIDIA GPUs while retaining FP32 accumulation-style
        # numerical behavior appropriate for deep-learning workloads.
        torch.backends.cuda.matmul.allow_tf32 = (
            allow_tf32
        )

        torch.backends.cudnn.allow_tf32 = (
            allow_tf32
        )

        torch.backends.cudnn.benchmark = (
            cudnn_benchmark
        )

        # Modern PyTorch recommends setting the matmul precision policy
        # explicitly rather than relying entirely on global defaults.
        if allow_tf32:
            torch.set_float32_matmul_precision(
                "high"
            )

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mps_available() -> bool:
        mps = getattr(
            torch.backends,
            "mps",
            None,
        )

        if mps is None:
            return False

        return bool(
            mps.is_available()
        )

    def _device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(
                self.device
            )

        if self.device.type == "mps":
            return "Apple Metal Performance Shaders"

        return "CPU"

    def _should_enable_autocast(self) -> bool:
        return (
            self.precision != torch.float32
            and self.device.type in {
                "cuda",
                "mps",
            }
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_arguments(self) -> None:
        if self.requested_device not in self.VALID_DEVICES:
            raise ValueError(
                "device must be one of: "
                "auto, cpu, cuda, mps."
            )

        if (
            self.requested_precision
            not in self.VALID_PRECISIONS
        ):
            raise ValueError(
                "precision must be one of: "
                "auto, fp32, fp16, bf16."
            )