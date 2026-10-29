"""Cached, GT-independent Samson preprocessing."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .pseudo_endmembers import make_training_targets

PREPROCESS_VERSION = "vca_projective_v3_pixel_targets"


def preprocess(spectra, shape, cache_dir, n_materials=3, block_size=5,
               local_vertices=3, clusters=None, seed=1):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(np.ascontiguousarray(spectra).view(np.uint8)).hexdigest()
    path = cache_dir / f"samson_preprocess_{PREPROCESS_VERSION}_{digest[:16]}.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        meta = json.loads(str(saved["metadata"]))
        if (meta.get("sha256") == digest and meta.get("shape") == list(shape)
                and meta.get("version") == PREPROCESS_VERSION):
            return {k: saved[k] for k in (
                "prototypes", "bundles", "candidate_pool", "pseudo_labels",
                "pixel_abundances"
            )}
    targets = make_training_targets(spectra, shape[0], shape[1], n_materials,
                                    block_size, local_vertices, clusters, seed)
    np.savez_compressed(path, **targets,
                        metadata=json.dumps({"sha256": digest, "shape": list(shape),
                                             "version": PREPROCESS_VERSION,
                                             "method": f"{PREPROCESS_VERSION}: local VCA + MiniBatchKMeans + VCA-dictionary NNLS"}))
    return targets
