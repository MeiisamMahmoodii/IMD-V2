def detect_device(prefer_gpu: bool = True) -> str:
    if not prefer_gpu:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def suggested_batch_size(vram_gb: float | None = None) -> int:
    if vram_gb is None:
        return 2
    if vram_gb >= 40:
        return 8
    if vram_gb >= 24:
        return 4
    if vram_gb >= 16:
        return 2
    return 1
