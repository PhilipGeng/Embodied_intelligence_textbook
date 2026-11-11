"""Explicit compute-device selection shared by training and inference."""
import torch


def resolve_device(requested="auto"):
    if str(requested) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in this Python/PyTorch environment. "
            "Use --device cpu or install a CUDA-enabled PyTorch with a working NVIDIA driver."
        )
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, auto")
    return device


def device_metadata(device, requested="auto"):
    device = torch.device(device)
    return dict(requested_device=str(requested), device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                torch_version=torch.__version__)
