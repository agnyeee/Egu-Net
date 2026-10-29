"""Unlabeled spectral-bundle and pseudo-label preparation."""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import nnls
from .vca import vca


def spectral_bundles(spectra, height, width, n_materials, block_size=5,
                     local_vertices=None, clusters=None, seed=1):
    """Extract local VCA candidates, then cluster duplicate/variable spectra."""
    cube = np.reshape(spectra, (spectra.shape[0], height, width), order="F").transpose(1, 2, 0)
    candidates = []
    local_vertices = local_vertices or n_materials
    for r in range(0, height, block_size):
        for c in range(0, width, block_size):
            tile = cube[r:min(r + block_size, height), c:min(c + block_size, width)]
            tile_spectra = tile.reshape(-1, spectra.shape[0]).T
            if tile_spectra.shape[1] < n_materials:
                continue
            count = min(local_vertices, tile_spectra.shape[1], spectra.shape[0])
            em, _ = vca(tile_spectra, count, seed=seed + r * width + c)
            candidates.append(em.T)
    pool = np.concatenate(candidates, axis=0)
    k = int(clusters or max(n_materials, round(0.20 * height * width)))
    k = min(max(k, n_materials), len(pool))
    if k == len(pool):
        bundles = pool.copy()
    else:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=min(1024, max(32, k)),
                             n_init=3, max_iter=100)
        bundles = km.fit(pool).cluster_centers_.astype(np.float32)
    return bundles, pool.astype(np.float32)


def pseudo_labels(bundle_spectra, dictionary, max_iter=500, tol=1e-5):
    """Fit nonnegative least squares against scene-derived VCA prototypes.

    The fitted coefficients are normalized to pseudo-abundances. GT variables
    are deliberately absent from this API.
    """
    e = np.asarray(dictionary, dtype=np.float64)
    x = np.asarray(bundle_spectra, dtype=np.float64)
    labels = np.empty((len(x), e.shape[1]), dtype=np.float32)
    for i, spectrum in enumerate(x):
        coeff, _ = nnls(e, spectrum, maxiter=max_iter)
        total = coeff.sum()
        labels[i] = (coeff / total if total > tol else np.full(e.shape[1], 1.0 / e.shape[1]))
    return labels


def make_training_targets(spectra, height, width, n_materials, block_size=5,
                          local_vertices=None, clusters=None, seed=1):
    prototypes, _ = vca(spectra, n_materials, seed=seed)
    bundles, pool = spectral_bundles(spectra, height, width, n_materials,
                                     block_size, local_vertices, clusters, seed)
    labels = pseudo_labels(bundles, prototypes)
    # Pixelwise targets use only observed spectra and scene-derived VCA
    # prototypes. Ground-truth variables are not passed to this function.
    pixel_abundances = pseudo_labels(spectra.T, prototypes)
    return {"prototypes": prototypes, "bundles": bundles, "candidate_pool": pool,
            "pseudo_labels": labels, "pixel_abundances": pixel_abundances}
