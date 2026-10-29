"""EGU-Net-ss losses."""
import torch
import torch.nn.functional as F


def classification_loss(logits, labels):
    # Matches the official EGU-Net-ss implementation (softmax cross entropy).
    return -(labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def reconstruction_loss(reconstruction, cube):
    return F.mse_loss(reconstruction, cube)


def linear_mixture_loss(abundances, endmembers, cube):
    """Reconstruct spectra from abundance fractions and bands-by-materials E."""
    if endmembers.ndim != 2 or abundances.ndim != 4 or cube.ndim != 4:
        raise ValueError("Expected endmembers BxC, abundances NxCxHxW, cube NxBxHxW")
    if abundances.shape[1] != endmembers.shape[1]:
        raise ValueError("Abundance channels must match endmember columns")
    mixed_spectra = torch.einsum("bc,nchw->nbhw", endmembers, abundances)
    return F.mse_loss(mixed_spectra, cube)


def abundance_target_loss(abundances, abundance_targets):
    """Match full-scene network abundances to scene-derived NNLS targets."""
    if abundances.shape != abundance_targets.shape:
        raise ValueError(
            f"Abundance shapes differ: {abundances.shape} vs {abundance_targets.shape}"
        )
    return F.mse_loss(abundances, abundance_targets)


def l2_regularization(model):
    return 0.5 * sum(p.square().sum() for n, p in model.named_parameters()
                     if p.requires_grad and p.ndim > 1 and "bn" not in n.lower())


def total_loss(logits, labels, reconstruction, cube, model,
               l2_coefficient=0.005, reconstruction_weight=1.0,
               abundances=None, endmembers=None, mixture_weight=0.0,
               abundance_targets=None, abundance_supervision_weight=0.0):
    cls = classification_loss(logits, labels)
    rec = reconstruction_loss(reconstruction, cube)
    reg = l2_regularization(model)
    if mixture_weight:
        if abundances is None or endmembers is None:
            raise ValueError("Mixture loss requires abundances and endmembers")
        mix = linear_mixture_loss(abundances, endmembers, cube)
    else:
        mix = rec.new_zeros(())
    if abundance_supervision_weight:
        if abundances is None or abundance_targets is None:
            raise ValueError("Abundance supervision requires predictions and targets")
        abundance = abundance_target_loss(abundances, abundance_targets)
    else:
        abundance = rec.new_zeros(())
    total = (cls + reconstruction_weight * rec + mixture_weight * mix
             + abundance_supervision_weight * abundance + l2_coefficient * reg)
    return total, cls, rec, mix, abundance, reg
