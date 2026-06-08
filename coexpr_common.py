#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coexpr_common.py -- shared helpers for the spatial gene / module / co-expression scripts
=========================================================================================

Imports gw_common to REUSE the validated chip loading, log1p-CPM normalization and
boundary I/O (does not modify any existing script). Plotting lives here in one
place (draw_map) so it is easy to tweak. Backgrounds are coloured spot scatters
(no H&E) so per-sample rotation is clean.
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import gw_common as gw

REGION_COLORS = gw.REGION_COLORS
log = gw.log


def load_samples(data_dir, info_path, normalize=True):
    """Reload every chip from raw matrices (any gene) -> dict sample -> AnnData
    with log1p-CPM X, raw counts in layers['counts'], obs['region'], obsm['spatial']."""
    info = pd.read_csv(info_path, sep="\t", dtype=str)
    info.columns = [c.strip() for c in info.columns]
    out = {}
    for _, row in info.iterrows():
        folder = row["Folder"].strip()
        sample = row.get("ExternalSampleId") or folder
        sample = sample.strip() if isinstance(sample, str) and sample.strip() else folder
        ad_ = gw.load_chip(data_dir, folder, sample, row["Number of cluster"],
                           row["Grey"], row["White"])
        if ad_ is None:
            continue
        if normalize:
            gw.normalize(ad_)
        out[sample] = ad_
    log.info("loaded %d sample(s) from %s", len(out), data_dir)
    return out


def read_rotations(path, default=0.0):
    """rotations file: tab-separated with columns 'sample' and 'degrees' (may be
    negative). Returns {sample: degrees}; samples absent use the global default."""
    rot = {}
    if path and os.path.isfile(path):
        df = pd.read_csv(path, sep="\t")
        cols = {c.lower(): c for c in df.columns}
        scol = cols.get("sample"); dcol = cols.get("degrees", cols.get("deg"))
        if scol and dcol:
            for _, r in df.iterrows():
                rot[str(r[scol])] = float(r[dcol])
        log.info("rotations: %d sample(s) from %s", len(rot), path)
    return rot


def rotate_xy(xy, deg, center):
    """Rotate points clockwise (in the displayed, y-inverted view) by deg about
    center. Uses a math-CCW matrix which, under ax.invert_yaxis(), reads as a
    clockwise turn on screen -- matching the H&E image rotated by PIL.rotate(-deg)."""
    xy = np.asarray(xy, float)
    if not deg:
        return xy
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s], [s, c]])
    return (xy - center) @ R.T + center


def expr_of(adata, gene):
    x = adata[:, gene].X
    return x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()


def counts_of(adata, genes):
    idx = [adata.var_names.get_loc(g) for g in genes]
    C = adata.layers["counts"][:, idx]
    return C.toarray() if hasattr(C, "toarray") else np.asarray(C)


def scale01(x):
    """Min-max scale to [0, 1]; constant vectors map to 0."""
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def scale01_robust(x, lo_pct=1.0, hi_pct=99.0):
    """Robust min-max scaling to [0, 1] using percentiles (clipped), so a few
    outlier spots cannot set the scale. Constant vectors map to 0."""
    x = np.asarray(x, float)
    lo = np.nanpercentile(x, lo_pct)
    hi = np.nanpercentile(x, hi_pct)
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def draw_he(ax, chip_dir, deg=0.0):
    """H&E panel: read tissue_hires_image.png DIRECTLY (no spatial_summary needed),
    rotate clockwise by deg to match the rotated spots."""
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("H&E")
    name = os.path.basename(chip_dir)
    path = os.path.join(chip_dir, "spatial", "tissue_hires_image.png")
    if not os.path.isfile(path):
        log.warning("[%s] no H&E image: %s", name, path)
        ax.text(0.5, 0.5, "no H&E\n(image missing)", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        return
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB").rotate(-deg, expand=True, fillcolor=(255, 255, 255))
        ax.imshow(np.asarray(im)); ax.set_aspect("equal")
    except Exception as e:
        log.warning("[%s] H&E load failed: %s", name, e)
        ax.text(0.5, 0.5, "H&E error", ha="center", va="center", transform=ax.transAxes)


def draw_map(ax, xy, center, deg=0.0, segments=(), *,
             region=None, values=None, highlight=None,
             cmap="viridis", vmin=None, vmax=None, norm=None, point_size=8,
             bg_color="#e8e8e8", hi_color="#d62728", region_colors=None,
             boundary_color="#FFD700", boundary_width=1.4, colorbar=True, title=""):
    """One spatial panel. Choose ONE colour mode:
       region=<array>     -> categorical grey/white(/none)
       highlight=<bool>   -> two colours (highlighted vs background)
       values=<array>     -> continuous colour map
    Spots and boundary are rotated together about the same `center`."""
    P = rotate_xy(xy, deg, center)
    if region is not None:
        rc = region_colors or REGION_COLORS
        for reg, col in rc.items():
            m = np.asarray(region) == reg
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], s=point_size, c=col, linewidths=0, label=reg)
        ax.legend(frameon=False, markerscale=2, fontsize=6, loc="best")
    elif highlight is not None:
        h = np.asarray(highlight, bool)
        ax.scatter(P[~h, 0], P[~h, 1], s=point_size, c=bg_color, linewidths=0)
        if h.any():
            ax.scatter(P[h, 0], P[h, 1], s=point_size, c=hi_color, linewidths=0)
    else:
        kw = dict(s=point_size, c=values, cmap=cmap, linewidths=0)
        if norm is not None:
            kw["norm"] = norm
        else:
            kw["vmin"] = vmin; kw["vmax"] = vmax
        sc_ = ax.scatter(P[:, 0], P[:, 1], **kw)
        if colorbar:
            cb = ax.figure.colorbar(sc_, ax=ax, fraction=0.046, pad=0.04)
            if norm is not None and hasattr(norm, "boundaries"):
                b = np.asarray(norm.boundaries)
                centers = (b[:-1] + b[1:]) / 2
                cb.set_ticks(centers)
                cb.set_ticklabels([str(int(round(c))) for c in centers])
    for seg in segments:
        s = rotate_xy(np.asarray(seg, float), deg, center)
        ax.plot(s[:, 0], s[:, 1], "--", color=boundary_color, lw=boundary_width, zorder=6,
                path_effects=[pe.Stroke(linewidth=boundary_width + 1.4, foreground="black"),
                              pe.Normal()])
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
