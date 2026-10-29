"""Required figure generation."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_evaluation_figures(reference_abundances, abundance_maps, reference_endmembers,
                            recovered_endmembers, history, shape, names, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    h, w, _ = shape
    gt = np.stack([
        np.asarray(reference_abundances[i]).reshape(h, w, order="F")
        for i in range(len(names))
    ])
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 3.5 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        vmax = float(max(gt[i].max(), abundance_maps[i].max()))
        im = axes[i, 0].imshow(gt[i], vmin=0, vmax=vmax, cmap="viridis")
        axes[i, 0].set_title(f"{name}: reference")
        fig.colorbar(im, ax=axes[i, 0], fraction=0.046)
        im = axes[i, 1].imshow(abundance_maps[i], vmin=0, vmax=vmax, cmap="viridis")
        axes[i, 1].set_title(f"{name}: predicted")
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046)
        im = axes[i, 2].imshow(np.abs(gt[i] - abundance_maps[i]), cmap="magma")
        axes[i, 2].set_title(f"{name}: absolute error")
        fig.colorbar(im, ax=axes[i, 2], fraction=0.046)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(figure_dir / "abundance_and_error_maps.png", dpi=180)

    fig2, axes2 = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for i, name in enumerate(names):
        ax = axes2[0, i]
        ax.plot(reference_endmembers[:, i], label="Reference", lw=2)
        ax.plot(recovered_endmembers[:, i], label="Recovered", lw=1.6)
        ax.set_title(name); ax.set_xlabel("Spectral band"); ax.set_ylabel("Reflectance")
        ax.legend()
    fig2.tight_layout()
    fig2.savefig(figure_dir / "endmember_spectra.png", dpi=180)

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    keys = (("total", "Total training loss"), ("classification", "E-Net classification loss"),
            ("reconstruction", "UR-Net reconstruction loss"))
    for ax, (key, title) in zip(axes3, keys):
        ax.plot([r[key] for r in history])
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    fig3.tight_layout()
    fig3.savefig(figure_dir / "training_losses.png", dpi=180)
    plt.close(fig); plt.close(fig2); plt.close(fig3)
    return [figure_dir / "abundance_and_error_maps.png",
            figure_dir / "endmember_spectra.png", figure_dir / "training_losses.png"]
