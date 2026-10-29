"""Atomic, device-independent checkpoint operations."""
from pathlib import Path
import os
import random
import tempfile
import numpy as np
import torch

MODEL_ID = "EGU-Net-ss-Simple-v1"


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # load_checkpoint may map the whole payload to CUDA for model restoration;
    # PyTorch's CPU RNG API still requires its state tensor on the CPU.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())


def save_checkpoint(path, model, optimizer, scheduler, state, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": MODEL_ID, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict() if scheduler else None,
               "state": state, "history": state.get("history", []),
               "config": config, "rng": rng_state()}
    handle, temp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(payload, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def load_checkpoint(path, model, optimizer=None, scheduler=None, expected_config=None,
                    map_location="cpu", restore_random=True):
    path = Path(path)
    if not path.is_file():
        return None
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("model_id") != MODEL_ID:
        raise ValueError(f"Incompatible checkpoint model identity: {payload.get('model_id')}")
    saved = payload.get("config", {})
    if expected_config:
        mismatches = {k: (saved.get(k), expected_config.get(k))
                      for k in set(saved) | set(expected_config)
                      if saved.get(k) != expected_config.get(k)}
        if mismatches:
            raise ValueError(f"Checkpoint architecture mismatch: {mismatches}")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("Checkpoint has no optimizer state; refusing a silent restart")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_random:
        restore_rng(payload.get("rng"))
    return payload
