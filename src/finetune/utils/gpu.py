"""GPU view used before training. Extracted verbatim from notebook cell 27."""


def force_single_gpu_view():
    import torch
    # CUDA wurde in einer früheren Zelle bereits mit beiden GPUs
    # initialisiert, daher greift CUDA_VISIBLE_DEVICES nicht mehr.
    # Wir gaukeln dem HF Trainer vor, dass nur eine GPU sichtbar ist,
    # damit er kein DataParallel-Wrapping vornimmt.
    torch.cuda.device_count = lambda: 1
    torch.cuda.set_device(0)
