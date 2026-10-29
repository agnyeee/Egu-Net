"""Ordinary sequential, single-device training and recovery."""
from pathlib import Path
import math
import numpy as np
import torch
from torch.optim import Adam
from .checkpoint import save_checkpoint, load_checkpoint
from .losses import total_loss


def select_device(requested="auto"):
    if requested == "cpu":
        return torch.device("cpu"), "CPU explicitly selected"
    if requested not in ("auto", "cuda"):
        raise ValueError("DEVICE must be auto, cuda, or cpu")
    if not torch.cuda.is_available():
        msg = "CUDA unavailable; using CPU for a small smoke test."
        return torch.device("cpu"), msg
    try:
        torch.cuda.init()
        device = torch.device("cuda:0")
        a = torch.ones((2, 2), device=device)
        _ = a @ a
        torch.cuda.synchronize()
        props = torch.cuda.get_device_properties(device)
        free, _ = torch.cuda.mem_get_info(device)
        return device, f"CUDA usable: {props.name}; free memory {free / 2**30:.2f} GiB"
    except Exception as exc:
        msg = ("CUDA was reported but initialization/test operation failed. The installed PyTorch "
               "CUDA build and NVIDIA driver are incompatible or unavailable. GPU training stopped; "
               f"diagnostic: {type(exc).__name__}: {exc}")
        if requested == "cuda":
            raise RuntimeError(msg) from exc
        return torch.device("cpu"), msg


def train(model, bundles, labels, cube, output_dir, config, device, epochs,
          batch_size=128, checkpoint_every=20, resume=True, endmembers=None,
          abundance_targets=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    x = torch.as_tensor(bundles, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    scene = torch.as_tensor(cube, dtype=torch.float32, device=device)
    endmember_tensor = (None if endmembers is None else
                        torch.as_tensor(endmembers, dtype=torch.float32, device=device))
    abundance_target_tensor = (None if abundance_targets is None else
                               torch.as_tensor(abundance_targets, dtype=torch.float32,
                                               device=device))
    if abundance_target_tensor is not None and abundance_target_tensor.ndim == 3:
        abundance_target_tensor = abundance_target_tensor.unsqueeze(0)
    if scene.ndim == 3:
        scene = scene.permute(2, 0, 1).unsqueeze(0)
    elif scene.ndim == 4 and scene.shape[-1] == model.bands:
        scene = scene.permute(0, 3, 1, 2)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Bundle and pseudo-label counts do not match")
    optimizer = Adam(model.parameters(), lr=float(config["learning_rate"]), betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(x) / batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(1.0 - step / max(total_steps, 1), 0.0) ** float(config["poly_power"]))
    state = {"epoch": 0, "batch_cursor": 0, "global_step": 0, "history": [], "order": None}
    latest = output_dir / "latest.pt"
    best = output_dir / "best.pt"
    if resume and latest.exists():
        payload = load_checkpoint(latest, model, optimizer, scheduler, config, device)
        state = payload["state"]
        print(f"Resuming compatible checkpoint at epoch {state['epoch'] + 1}, step {state['global_step']}")
    elif resume and best.exists():
        raise ValueError("best.pt exists but latest.pt is missing; refusing a silent new run")
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("mixed_precision", False) and device.type == "cuda"))
    model.train()
    best_loss = min((r["total"] for r in state["history"]), default=float("inf"))
    while state["epoch"] < epochs:
        epoch = state["epoch"]
        if state["order"] is None:
            state["order"] = np.random.permutation(len(x)).tolist()
            state["batch_cursor"] = 0
            epoch_sums = {"total": 0.0, "classification": 0.0,
                          "reconstruction": 0.0, "mixture": 0.0,
                          "abundance": 0.0}
        else:
            epoch_sums = state.get("partial_sums", {
                "total": 0.0, "classification": 0.0,
                "reconstruction": 0.0, "mixture": 0.0, "abundance": 0.0
            })
        order = state["order"]
        while state["batch_cursor"] < len(order):
            start = state["batch_cursor"]
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                result = model(x[indices], scene)
                loss, cls, rec, mix, abundance, _ = total_loss(
                    result["class_logits"], y[indices], result["reconstruction"],
                    scene, model, config["l2_coefficient"],
                    config["reconstruction_weight"], result["abundances"],
                    endmember_tensor, config["mixture_reconstruction_weight"],
                    abundance_target_tensor, config["abundance_supervision_weight"]
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state["global_step"] += 1
            state["batch_cursor"] += len(indices)
            for key, val in (("total", loss), ("classification", cls),
                             ("reconstruction", rec), ("mixture", mix),
                             ("abundance", abundance)):
                epoch_sums[key] += float(val.detach())
            state["partial_sums"] = epoch_sums
            if state["global_step"] % checkpoint_every == 0:
                save_checkpoint(latest, model, optimizer, scheduler, state, config)
        denom = max(1, math.ceil(len(order) / batch_size))
        row = {"epoch": epoch + 1, **{k: v / denom for k, v in epoch_sums.items()}}
        state["history"].append(row)
        state["epoch"] += 1
        state["batch_cursor"] = 0
        state["order"] = None
        state.pop("partial_sums", None)
        save_checkpoint(latest, model, optimizer, scheduler, state, config)
        if row["total"] < best_loss:
            best_loss = row["total"]
            save_checkpoint(best, model, optimizer, scheduler, state, config)
        print(f"Epoch {state['epoch']}/{epochs}: total={row['total']:.6f}, "
              f"E-Net={row['classification']:.6f}, UR-Net={row['reconstruction']:.6f}, "
              f"mixture={row['mixture']:.6f}, "
              f"abundance={row['abundance']:.6f}")
    return state["history"]
