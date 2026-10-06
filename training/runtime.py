import os
import random
import numpy as np
import torch


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def choose_device(name):
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else "cpu"
        if name == "auto"
        else name
    )


@torch.inference_mode()
def predict_loader(model, loader, device, criterion=None):
    model.eval()
    targets, scores, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        if criterion:
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            loss_sum += loss.item() * len(x)
        targets.append(y.cpu().numpy())
        scores.append(logits.sigmoid().cpu().numpy())
    return (
        np.concatenate(targets),
        np.concatenate(scores),
        loss_sum / len(loader.dataset),
    )
