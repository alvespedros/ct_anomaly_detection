"""
visualize_slices.py
-------------------
Visualize random transversal (axial) slices of a 3D/4D image tensor
side-by-side with its reconstructed counterpart.

Supports tensors with shapes:
  - (D, H, W)        — single-channel 3-D volume
  - (C, D, H, W)     — multi-channel 3-D volume  (first channel used)
  - (B, C, D, H, W)  — batched multi-channel      (first batch item used)

All tensors are expected to live on a CUDA GPU; the module moves them
to CPU internally for plotting.
"""

import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_volume(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalise an arbitrary tensor to shape (D, H, W) on CPU.

    Accepted input shapes
    ---------------------
    (D, H, W)          → returned as-is
    (C, D, H, W)       → first channel selected
    (B, C, D, H, W)    → first batch + first channel selected
    """
    t = tensor.detach().float().cpu()

    if t.ndim == 3:
        return t                         # (D, H, W)
    if t.ndim == 4:
        return t[0]                      # (D, H, W)
    if t.ndim == 5:
        return t[0, 0]                   # (D, H, W)

    raise ValueError(
        f"Unsupported tensor shape {tuple(tensor.shape)}. "
        "Expected (D,H,W), (C,D,H,W), or (B,C,D,H,W)."
    )


def _norm(volume: torch.Tensor) -> torch.Tensor:
    """Min-max normalise a volume to [0, 1] for display."""
    vmin, vmax = volume.min(), volume.max()
    if (vmax - vmin).abs() < 1e-8:
        return torch.zeros_like(volume)
    return (volume - vmin) / (vmax - vmin)


# ── public API ─────────────────────────────────────────────────────────────────

def visualize_slices(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    n_slices: int = 6,
    cmap: str = "gray",
    figsize_per_slice: tuple[float, float] = (4.0, 4.0),
    suptitle: str = "Original vs Reconstructed",
    save_path: str | None = None,
    seed: int | None = None,
) -> plt.Figure:
    """
    Plot *n_slices* random transversal (axial) slices comparing
    *original* and *reconstructed* tensors side-by-side.

    Parameters
    ----------
    original        : GPU tensor — the ground-truth image volume.
    reconstructed   : GPU tensor — the model's output volume.
    n_slices        : Number of random axial slices to show (default 6).
    cmap            : Matplotlib colormap (default "gray").
    figsize_per_slice: (width, height) in inches per column-pair.
    suptitle        : Figure-level title.
    save_path       : If given, the figure is saved to this path.
    seed            : Optional random seed for reproducibility.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if seed is not None:
        random.seed(seed)

    # ── normalise to (D, H, W) on CPU ──────────────────────────────────────
    vol_orig  = _norm(_to_volume(original))
    vol_recon = _norm(_to_volume(reconstructed))

    if vol_recon.shape != vol_orig.shape:
        raise ValueError(
            f"Shape mismatch: original {tuple(vol_orig.shape)} vs "
            f"reconstructed {tuple(vol_recon.shape)}."
        )

    D, H, W = vol_orig.shape

    # ── pick random axial (transversal) slice indices ───────────────────────
    n_slices = min(n_slices, D)
    indices  = sorted(random.sample(range(D), k=n_slices))

    # ── layout: 2 columns (orig | recon) × n_slices rows ───────────────────
    fig_w = figsize_per_slice[0] * 2
    fig_h = figsize_per_slice[1] * n_slices

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d0d0d")
    fig.suptitle(suptitle, color="white", fontsize=14, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(
        n_slices, 2,
        figure=fig,
        hspace=0.35,
        wspace=0.05,
    )

    col_headers = ["Original", "Reconstructed"]
    col_colors  = ["#4fc3f7", "#81c784"]

    for row, idx in enumerate(indices):
        for col, vol in enumerate((vol_orig, vol_recon)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(vol[idx].numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=1)
            ax.set_facecolor("#1a1a1a")

            # column header on the first row only
            if row == 0:
                ax.set_title(
                    col_headers[col],
                    color=col_colors[col],
                    fontsize=12, fontweight="bold", pad=8,
                )

            # slice index label on the left column only
            if col == 0:
                ax.set_ylabel(f"z = {idx}", color="#aaaaaa", fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Figure saved → {save_path}")

    return fig


# ── convenience wrapper ────────────────────────────────────────────────────────

def show_slices(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    **kwargs,
) -> None:
    """
    Like `visualize_slices` but calls plt.show() automatically.
    Useful in interactive sessions / Jupyter notebooks.
    """
    fig = visualize_slices(original, reconstructed, **kwargs)
    plt.show()
    plt.close(fig)


