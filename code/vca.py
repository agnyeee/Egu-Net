"""Vertex Component Analysis (VCA), following the published SNR branches."""
import numpy as np


def _estimate_snr_db(y, n_endmembers):
    """Estimate SNR using the signal-subspace criterion used by VCA."""
    bands, n_pixels = y.shape
    mean = y.mean(axis=1, keepdims=True)
    centered = y - mean
    u, _, _ = np.linalg.svd(centered / np.sqrt(n_pixels), full_matrices=False)
    projected_centered = u[:, :n_endmembers].T @ centered

    power_y = float(np.square(y).sum() / n_pixels)
    power_x = float(np.square(projected_centered).sum() / n_pixels
                    + np.square(mean).sum())
    signal = max(power_x - (n_endmembers / bands) * power_y, 1e-30)
    noise = max(power_y - power_x, 1e-30)
    return abs(10.0 * np.log10(signal / noise))


def vca(spectra, n_endmembers, seed=1, snr_db=None):
    """Return ``(bands, endmembers)`` signatures and selected pixel indices.

    ``spectra`` has shape bands-by-pixels. At high SNR, VCA uses the
    projective projection of the uncentered signal subspace. At low SNR, it
    uses a centered ``p-1`` dimensional affine projection with an augmented
    constant row. The selected output signatures are the original observed
    pixels, rather than their lower-dimensional projected coordinates.
    """
    y = np.asarray(spectra, dtype=np.float64)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("VCA expects a finite bands-by-pixels matrix")

    bands, n_pixels = y.shape
    p = int(n_endmembers)
    if not 1 <= p <= min(bands, n_pixels):
        raise ValueError("n_endmembers must be between 1 and min(bands, pixels)")

    if p == 1:
        centered = y - y.mean(axis=1, keepdims=True)
        idx = int(np.argmax(np.linalg.norm(centered, axis=0)))
        return y[:, [idx]].astype(np.float32), np.asarray([idx], dtype=np.int64)

    if snr_db is None:
        snr_db = _estimate_snr_db(y, p)
    snr_threshold = 15.0 + 10.0 * np.log10(p)
    rng = np.random.default_rng(seed)

    if snr_db > snr_threshold:
        # High-SNR branch: project the uncentered data and apply the VCA
        # projective normalization. Centering here changes the simplex and
        # can prevent the true signal vertices from being selected.
        u, _, _ = np.linalg.svd(y, full_matrices=False)
        ud = u[:, :p]
        x = ud.T @ y
        center = x.mean(axis=1, keepdims=True)
        denominator = np.sum(x * center, axis=0, keepdims=True)
        eps = 1e-12
        denominator = np.where(
            np.abs(denominator) < eps,
            np.where(denominator < 0.0, -eps, eps),
            denominator,
        )
        projected = x / denominator
    else:
        # Low-SNR branch: center, project onto p-1 principal components,
        # then augment with the maximum projected norm.
        mean = y.mean(axis=1, keepdims=True)
        centered = y - mean
        u, _, _ = np.linalg.svd(centered / np.sqrt(n_pixels), full_matrices=False)
        ud = u[:, : p - 1]
        x = ud.T @ centered
        max_norm = max(float(np.linalg.norm(x, axis=0).max()), 1e-12)
        projected = np.vstack((x, np.full((1, n_pixels), max_norm)))

    # Random orthogonal projections select one extreme pixel per vertex.
    auxiliary = np.zeros((p, p), dtype=np.float64)
    auxiliary[-1, 0] = 1.0
    selected = []
    for i in range(p):
        direction = rng.standard_normal(p)
        previous = auxiliary[:, : max(i, 1)]
        direction -= previous @ np.linalg.pinv(previous) @ direction
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = rng.standard_normal(p)
            direction -= previous @ np.linalg.pinv(previous) @ direction
            norm = float(np.linalg.norm(direction))
        direction /= max(norm, 1e-12)

        scores = direction @ projected
        idx = int(np.argmax(np.abs(scores)))
        selected.append(idx)
        auxiliary[:, i] = projected[:, idx]

    return y[:, selected].astype(np.float32), np.asarray(selected, dtype=np.int64)
