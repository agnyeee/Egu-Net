"""Evaluation, endmember alignment, metrics, and prediction artifact export."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment, nnls


def align_materials(predicted, reference):
    """Reorder predicted channels to reference channels by minimum RMSE."""
    c = reference.shape[0]
    cost = np.zeros((c, c), dtype=np.float64)
    for ref in range(c):
        for pred in range(c):
            cost[ref, pred] = np.sqrt(np.mean((reference[ref] - predicted[pred]) ** 2))
    rows, cols = linear_sum_assignment(cost)
    order = cols[np.argsort(rows)]
    return predicted[order], order, cost[rows, cols]


def recover_endmembers(spectra, abundances):
    """Nonnegative least squares E estimate from observed X and predicted A."""
    x = np.asarray(spectra, dtype=np.float64)
    a = np.asarray(abundances, dtype=np.float64)
    out = np.empty((x.shape[0], a.shape[0]), dtype=np.float32)
    for band in range(x.shape[0]):
        out[band], _ = nnls(a.T, x[band], maxiter=1000)
    return out


def abundance_maps_to_pixels(abundance_maps, height, width):
    """Convert C×H×W predictions to C×N MATLAB column-major pixel order."""
    maps = np.asarray(abundance_maps, dtype=np.float32)
    if maps.ndim == 4:
        if maps.shape[0] != 1:
            raise ValueError("Expected one abundance-map batch")
        maps = maps[0]
    if maps.ndim == 3:
        if maps.shape[1:] != (height, width):
            raise ValueError(f"Abundance map shape {maps.shape} does not match {(height, width)}")
        return np.stack([maps[c].reshape(-1, order="F") for c in range(maps.shape[0])])
    if maps.ndim == 2 and maps.shape[1] == height * width:
        return maps
    raise ValueError("Predicted abundances must be C×H×W or C×(H*W)")


def metrics(reference_abundances, predicted_abundances, reference_endmembers,
            predicted_endmembers):
    a, order, _ = align_materials(predicted_abundances, reference_abundances)
    endmembers = predicted_endmembers[:, order]
    # Paper Eq. (9): sqrt(mean over pixels of mean squared error over materials).
    armse = float(np.sqrt(np.mean(np.mean((reference_abundances - a) ** 2, axis=0))))
    per_material = np.sqrt(np.mean((reference_abundances - a) ** 2, axis=1))
    dot = np.sum(reference_endmembers * endmembers, axis=0)
    denom = np.linalg.norm(reference_endmembers, axis=0) * np.linalg.norm(endmembers, axis=0)
    sad = np.arccos(np.clip(dot / np.maximum(denom, 1e-12), -1.0, 1.0))
    return {"aRMSE": armse, "per_material_RMSE": per_material.tolist(),
            "SAD_radians": sad.tolist(), "average_SAD_radians": float(sad.mean()),
            "predicted_channel_order": order.tolist()}, a, endmembers


def evaluate_and_save(spectra, cube_shape, predicted_abundances, reconstruction,
                      reference_abundances, reference_endmembers, output_dir,
                      material_names=None):
    output_dir = Path(output_dir)
    metrics_dir, features_dir = output_dir / "metrics", output_dir / "features"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    h, w, bands = cube_shape
    pred = abundance_maps_to_pixels(predicted_abundances, h, w)
    recovered = recover_endmembers(spectra, pred)
    summary, pred, recovered = metrics(reference_abundances, pred,
                                        reference_endmembers, recovered)
    names = list(material_names or [f"Material {i+1}" for i in range(len(pred))])
    summary["material_names"] = names
    summary["abundance_rmse_by_material"] = dict(zip(names, summary["per_material_RMSE"]))
    summary["sad_radians_by_material"] = dict(zip(names, summary["SAD_radians"]))
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (metrics_dir / "summary.csv").write_text(
        "metric,value\n" + "\n".join([f"aRMSE,{summary['aRMSE']}",
                                          f"average_SAD_radians,{summary['average_SAD_radians']}"] +
                                         [f"RMSE_{names[i]},{summary['per_material_RMSE'][i]}" for i in range(len(names))] +
                                         [f"SAD_radians_{names[i]},{summary['SAD_radians'][i]}" for i in range(len(names))]),
        encoding="utf-8")
    recon = np.asarray(reconstruction, dtype=np.float32)
    if recon.ndim == 4:
        recon = recon[0].transpose(1, 2, 0)
    elif recon.shape[0] == bands:
        recon = recon.transpose(1, 2, 0)
    maps = np.stack([pred[c].reshape(h, w, order="F") for c in range(len(pred))])
    np.savez_compressed(features_dir / "predictions.npz", abundances=pred,
                         abundance_maps=maps, reconstruction=recon,
                         recovered_endmembers=recovered,
                         aligned_order=np.asarray(summary["predicted_channel_order"]))
    return summary, maps, recovered, recon
