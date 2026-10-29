"""MATLAB loader with explicit spatial and pixel-order conventions."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def _pick(mat, names, predicate):
    for key in names:
        if key in mat and isinstance(mat[key], np.ndarray) and predicate(mat[key]):
            return key, np.asarray(mat[key])
    candidates = [(k, np.asarray(v)) for k, v in mat.items()
                  if not k.startswith("__") and isinstance(v, np.ndarray) and predicate(v)]
    if len(candidates) != 1:
        raise ValueError(f"Could not uniquely identify MATLAB variable; candidates: {[k for k, _ in candidates]}")
    return candidates[0]


def load_samson(hsi_path, gt_path, expected_shape=(95, 95, 156), materials=3):
    """Load HSI as H×W×B; GT A as C×(H*W), E as B×C.

    MATLAB pixel vectors use column-major spatial order. The explicit reshape
    below preserves that order when converting the HSI cube to V.
    """
    hsi_path, gt_path = Path(hsi_path), Path(gt_path)
    if not hsi_path.is_file() or not gt_path.is_file():
        missing = [str(p) for p in (hsi_path, gt_path) if not p.is_file()]
        raise FileNotFoundError("Missing Samson MATLAB dataset file(s): " + ", ".join(missing))
    hmat = loadmat(hsi_path)
    _, raw = _pick(hmat, ("V", "Y", "data", "Samson", "img", "cube"), lambda a: a.ndim in (2, 3))
    if raw.ndim == 3:
        if raw.shape == expected_shape:
            cube = raw
        elif raw.shape == (expected_shape[2], expected_shape[0], expected_shape[1]):
            cube = np.transpose(raw, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected HSI cube shape {raw.shape}; expected {expected_shape} or bands-first")
        # MATLAB's [H,W,B] is turned into [B,N] in column-major spatial order.
        spectra = np.reshape(np.transpose(cube, (2, 0, 1)), (cube.shape[2], -1), order="F")
    else:
        if raw.shape == (expected_shape[2], expected_shape[0] * expected_shape[1]):
            spectra = raw
        elif raw.shape == (expected_shape[0] * expected_shape[1], expected_shape[2]):
            spectra = raw.T
        else:
            raise ValueError(f"Unexpected HSI matrix shape {raw.shape}; expected (156,9025) or transpose")
        cube = np.reshape(spectra, (expected_shape[2], expected_shape[0], expected_shape[1]), order="F").transpose(1, 2, 0)
    gt = loadmat(gt_path)
    _, abund = _pick(gt, ("A", "abundance", "abundances", "GT_A"), lambda a: a.ndim == 2 and materials in a.shape)
    if abund.shape == (expected_shape[0] * expected_shape[1], materials):
        abund = abund.T
    _, endmembers = _pick(gt, ("M", "E", "endmembers", "GT_M"), lambda a: a.ndim == 2 and materials in a.shape)
    if endmembers.shape == (materials, expected_shape[2]):
        endmembers = endmembers.T
    if abund.shape != (materials, expected_shape[0] * expected_shape[1]):
        raise ValueError(f"GT abundance shape {abund.shape}; expected {(materials, expected_shape[0]*expected_shape[1])}")
    if endmembers.shape != (expected_shape[2], materials):
        raise ValueError(f"GT endmember shape {endmembers.shape}; expected {(expected_shape[2], materials)}")
    if not (np.isfinite(cube).all() and np.isfinite(abund).all() and np.isfinite(endmembers).all()):
        raise ValueError("MATLAB data contain NaN or infinity")
    return {"cube": cube.astype(np.float32), "spectra": spectra.astype(np.float32),
            "abundances_gt": abund.astype(np.float32), "endmembers_gt": endmembers.astype(np.float32),
            "shape": expected_shape}
