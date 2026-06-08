#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gw_common.py -- shared library for the spinal-cord grey/white Visium pipeline
=============================================================================

Used by:
  01_compute.py        build all source data (no plotting)
  02_plot_spatial.py   spatial maps with grey/white boundary
  03_plot_summary.py   dot plot / violin / single-gene box / log2FC / paired

Nothing here runs on import; it only defines functions and constants.
"""

import gzip
import json
import logging
import os
import sys
from math import ceil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                       # headless default; 02 --draw overrides
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

import scanpy as sc
import anndata as ad
from scipy.stats import mannwhitneyu
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

log = logging.getLogger("gw")

# --------------------------------------------------------------------------- #
# Constants & gene panel                                                       #
# --------------------------------------------------------------------------- #
REGION_COLORS = {"grey": "#7f7f7f", "white": "#d62728", "none": "#e8e8e8"}
TARGET_SUM = 1e4

DEFAULT_PANEL = [
    ("Neuron",                ["NEFM", "SNAP25"]),
    ("Oligo",                 ["MBP", "PLP1"]),
    ("Homeostatic_microglia", ["CX3CR1", "CSF1R", "P2RY12"]),
    ("Core_DAMs",             ["TREM2", "APOE", "TYROBP"]),
    ("Single_DAM",            ["CLEC7A", "ITGAX", "LGALS3", "SPP1", "LPL", "CD9"]),
]
DEFAULT_SINGLE_SETS = "Single_DAM"

MODULES = {}        # set_name -> genes (co-expression modules)
SINGLE_GROUPS = {}  # set_name -> genes (per-gene sets, e.g. Single_DAM)
SINGLE_DAM = []     # flat list of all single-gene-set genes


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "?"


def setup_logging(outdir, filename="run.log"):
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(outdir, filename),
                                      mode="w", encoding="utf-8")])
    log.info("scanpy %s | anndata %s", pkg_version("scanpy"), pkg_version("anndata"))


def parse_gene_list(path):
    """'<set>\\t<g1,g2,...>' -> [(set, [genes]), ...]."""
    panel = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(None, 1)
            if len(parts) < 2:
                log.warning("skipping malformed gene-list line: %r", line)
                continue
            name = parts[0].strip()
            genes = [g.strip() for g in parts[1].replace(",", " ").split() if g.strip()]
            if name and genes:
                panel.append((name, genes))
    return panel


def configure_panel(panel, single_sets):
    global MODULES, SINGLE_GROUPS, SINGLE_DAM
    single_sets = {s.strip() for s in single_sets if s and s.strip()}
    MODULES = {n: list(g) for n, g in panel if n not in single_sets}
    SINGLE_GROUPS = {n: list(g) for n, g in panel if n in single_sets}
    SINGLE_DAM = [g for genes in SINGLE_GROUPS.values() for g in genes]
    unknown = single_sets - {n for n, _ in panel}
    if unknown:
        log.warning("--single-gene-sets not in panel: %s", sorted(unknown))
    log.info("panel: %d module(s) %s | %d single-gene set(s) %s",
             len(MODULES), list(MODULES), len(SINGLE_GROUPS), list(SINGLE_GROUPS))


def panel_genes_present(var_names):
    return [g for g in dict.fromkeys(sum(MODULES.values(), []) + SINGLE_DAM) if g in var_names]


def rename_modules_by_genes(sep="+"):
    """Rename each co-expression module to '<gene1>+<gene2>+...'. Single-gene
    sets are left untouched. Call right after configure_panel()."""
    global MODULES
    MODULES = {sep.join(genes): genes for genes in MODULES.values()}
    log.info("modules renamed by genes -> %s", list(MODULES))


# --------------------------------------------------------------------------- #
# Region assignment                                                            #
# --------------------------------------------------------------------------- #
def clustering_column(number_of_cluster):
    val = str(number_of_cluster).strip().lower()
    if "graph" in val:
        return "graph_based"
    return f"gene_expression_k_means_k_{int(float(val))}"


def parse_cluster_ids(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return {f"Cluster {int(float(tok.strip()))}"
            for tok in str(cell).split(",") if tok.strip()}


def load_chip(data_dir, folder, sample, number_of_cluster, grey_cell, white_cell):
    """Load one chip with ALL tissue spots (region grey/white/none) + raw counts
    + spatial coords. Returns None if grey or white is empty."""
    chip_dir = os.path.join(data_dir, folder)
    mtx_dir = os.path.join(chip_dir, "matrix")
    meta_path = os.path.join(chip_dir, "metadata.tsv.gz")
    if not os.path.isdir(mtx_dir) or not os.path.isfile(meta_path):
        log.warning("[%s] missing matrix dir or metadata -- skipped", sample)
        return None

    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=True)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    adata.obs["sample"] = sample
    adata.obs["folder"] = folder

    col = clustering_column(number_of_cluster)
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)
    if col not in meta.columns:
        log.warning("[%s] clustering column %r not found -- skipped", sample, col)
        return None
    meta = meta.reindex(adata.obs_names)
    labels = meta[col].astype("string")
    grey_set, white_set = parse_cluster_ids(grey_cell), parse_cluster_ids(white_cell)
    overlap = grey_set & white_set
    if overlap:
        log.warning("[%s] clusters in BOTH grey and white: %s", sample, overlap)
    region = np.where(labels.isin(grey_set), "grey",
             np.where(labels.isin(white_set), "white", "none"))
    adata.obs["clustering_used"] = col
    adata.obs["cluster_label"] = labels.values
    adata.obs["region"] = region

    n = {r: int((region == r).sum()) for r in ("grey", "white", "none")}
    log.info("[%s] clustering=%s | grey=%s%d | white=%s%d | none=%d",
             sample, col, sorted(grey_set), n["grey"],
             sorted(white_set), n["white"], n["none"])
    if n["grey"] == 0 or n["white"] == 0:
        log.warning("[%s] grey or white empty -- skipped", sample)
        return None

    xy_path = os.path.join(chip_dir, "spatial", "spatial_xy_matrix.tsv.gz")
    if os.path.isfile(xy_path):
        xy = pd.read_csv(xy_path, sep="\t", index_col=0).reindex(adata.obs_names)
        adata.obsm["spatial"] = xy[["x", "y"]].to_numpy(float)
    else:
        log.warning("[%s] no spatial coords", sample)
    return adata


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def present_genes(adata, genes, label):
    have = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        log.warning("genes absent for %s: %s", label, missing)
    return have


def normalize(adata):
    """log1p CPM: per-spot total-count normalise to TARGET_SUM, then log1p.
    Raw counts must already be in layer 'counts'."""
    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata)


def add_module_scores(adata):
    counts = adata.layers["counts"]
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    for name, genes in MODULES.items():
        have = present_genes(adata, genes, name)
        if not have:
            adata.obs[f"score_{name}"] = np.nan
            adata.obs[f"allpos_{name}"] = False
            continue
        sc.tl.score_genes(adata, have, score_name=f"score_{name}", ctrl_size=50)
        idx = [adata.var_names.get_loc(g) for g in have]
        adata.obs[f"allpos_{name}"] = (counts[:, idx] > 0).all(axis=1)


def _writeback_scores(dst, src):
    for name in MODULES:
        for col in (f"score_{name}", f"allpos_{name}"):
            if col in src.obs:
                dst.obs.loc[src.obs_names, col] = src.obs[col].values


def compute_module_scores(combined, per_chip=False, on_grey_white=False):
    """Module scores either once on the pooled object (default; common
    background, comparable across chips) or within each chip (own background;
    suits per-sample spatial maps). on_grey_white restricts the scoring
    background to grey/white spots (excludes 'none'). per_chip + on_grey_white
    reproduces the old grey_white_marker_analysis.py behaviour."""
    if per_chip or on_grey_white:
        for name in MODULES:
            combined.obs[f"score_{name}"] = np.nan
            combined.obs[f"allpos_{name}"] = False
    if not per_chip:
        if on_grey_white:
            sub = combined[np.isin(combined.obs["region"], ["grey", "white"])].copy()
            add_module_scores(sub); _writeback_scores(combined, sub)
        else:
            add_module_scores(combined)
        log.info("module scores: POOLED%s", " (grey/white background)" if on_grey_white else "")
        return
    for s in combined.obs["sample"].unique().tolist():
        sub = combined[combined.obs["sample"] == s].copy()
        if on_grey_white:
            sub = sub[np.isin(sub.obs["region"], ["grey", "white"])].copy()
        if sub.n_obs == 0:
            continue
        add_module_scores(sub); _writeback_scores(combined, sub)
    log.info("module scores: PER-CHIP%s", " (grey/white background)" if on_grey_white else "")


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #
def expr_vector(adata, gene):
    x = adata[:, gene].X
    return x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan, float)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]; n = pv.size; order = np.argsort(pv)
    ranked = np.minimum.accumulate((pv[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    adj = np.empty(n); adj[order] = np.clip(ranked, 0, 1); out[ok] = adj
    return out


def stars(p):
    if pd.isna(p):
        return "ns"
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def compare_grey_white(grey_vals, white_vals, is_expression):
    g = np.asarray(grey_vals, float); w = np.asarray(white_vals, float)
    mg, mw = float(g.mean()), float(w.mean())
    res = {"n_grey": int(g.size), "n_white": int(w.size),
           "mean_grey": mg, "mean_white": mw, "mean_diff_grey_minus_white": mg - mw}
    if is_expression:
        res["log2FC_grey_over_white"] = float(
            np.log2((np.expm1(g).mean() + 1e-9) / (np.expm1(w).mean() + 1e-9)))
        res["pct_grey"] = float((g > 0).mean()); res["pct_white"] = float((w > 0).mean())
    else:
        res["log2FC_grey_over_white"] = np.nan
        res["pct_grey"] = np.nan; res["pct_white"] = np.nan
    if g.size >= 3 and w.size >= 3 and np.ptp(np.concatenate([g, w])) > 0:
        try:
            _, p = mannwhitneyu(g, w, alternative="two-sided")
        except ValueError:
            p = np.nan
    else:
        p = np.nan
    res["pval"] = float(p) if p == p else np.nan
    res["enriched_in"] = "grey" if (mg - mw) > 0 else "white"
    return res


def build_stats(adata, scope_label):
    """Grey-vs-white stats. Uses only region==grey/white spots (none ignored)."""
    rows = []
    grey = adata.obs["region"].values == "grey"
    white = adata.obs["region"].values == "white"
    for name in MODULES:
        col = f"score_{name}"
        if col not in adata.obs or adata.obs[col].isna().all():
            continue
        s = adata.obs[col].values.astype(float)
        r = compare_grey_white(s[grey], s[white], is_expression=False)
        r.update(scope=scope_label, feature=name, feature_kind="module_score", gene_set=name)
        rows.append(r)
    for name, genes in MODULES.items():
        for gname in genes:
            if gname not in adata.var_names:
                continue
            x = expr_vector(adata, gname)
            r = compare_grey_white(x[grey], x[white], is_expression=True)
            r.update(scope=scope_label, feature=gname,
                     feature_kind="module_member_gene", gene_set=name)
            rows.append(r)
    for name, genes in SINGLE_GROUPS.items():
        for gname in genes:
            if gname not in adata.var_names:
                continue
            x = expr_vector(adata, gname)
            r = compare_grey_white(x[grey], x[white], is_expression=True)
            r.update(scope=scope_label, feature=gname,
                     feature_kind="single_gene", gene_set=name)
            rows.append(r)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["FDR"] = bh_fdr(df["pval"].values)
        df["sig"] = df["FDR"].map(stars)
    return df


def allpos_fraction_table(adata, scope_label):
    rows = []
    for name in MODULES:
        col = f"allpos_{name}"
        if col not in adata.obs:
            continue
        for region in ("grey", "white"):
            sub = adata.obs.loc[adata.obs["region"] == region, col]
            rows.append({"scope": scope_label, "module": name, "region": region,
                         "n_spots": int(sub.size), "n_allpos": int(sub.sum()),
                         "frac_allpos": float(sub.mean()) if sub.size else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Source-data (tidy) tables                                                    #
# --------------------------------------------------------------------------- #
def sd_module_scores_long(adata):
    """spot-level: sample, region, module, score (grey/white only)."""
    gw = adata[np.isin(adata.obs["region"], ["grey", "white"])]
    frames = []
    for name in MODULES:
        col = f"score_{name}"
        if col in gw.obs:
            frames.append(pd.DataFrame({
                "sample": gw.obs["sample"].values, "region": gw.obs["region"].values,
                "module": name, "score": gw.obs[col].values.astype(float)}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sd_gene_expr_long(adata):
    """spot-level: sample, region, gene_set, gene, expr (log1p; grey/white only)."""
    gw = adata[np.isin(adata.obs["region"], ["grey", "white"])]
    sample = gw.obs["sample"].values; region = gw.obs["region"].values
    set_of = {}
    for s, genes in {**MODULES, **SINGLE_GROUPS}.items():
        for g in genes:
            set_of.setdefault(g, s)
    frames = []
    for g in panel_genes_present(gw.var_names):
        frames.append(pd.DataFrame({
            "sample": sample, "region": region, "gene_set": set_of.get(g, ""),
            "gene": g, "expr": expr_vector(gw, g)}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sd_dotplot(adata):
    """region x gene: mean log1p expr + fraction expressing (grey/white only)."""
    gw = adata[np.isin(adata.obs["region"], ["grey", "white"])]
    rows = []
    for g in panel_genes_present(gw.var_names):
        x = expr_vector(gw, g); reg = gw.obs["region"].values
        for region in ("grey", "white"):
            m = reg == region
            rows.append({"gene": g, "region": region,
                         "mean_expr": float(x[m].mean()),
                         "pct_expressing": float((x[m] > 0).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Boundary                                                                     #
# --------------------------------------------------------------------------- #
def median_spacing(xy):
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def knn_smooth(xy, value, knn):
    tree = cKDTree(xy); k = min(knn + 1, len(xy))
    _, idx = tree.query(xy, k=k)
    with np.errstate(invalid="ignore"):
        sm = np.nanmean(value[idx], axis=1)
    sm[np.isnan(sm)] = np.nanmean(value)
    return sm


def auto_boundary(df, knn=12, grid_res=400, sigma=1.0, min_seg_frac=0.05):
    """df has columns x, y, region (grey/white/none). Returns list of Nx2 polylines."""
    xy = df[["x", "y"]].to_numpy(float)
    spacing = median_spacing(xy)
    value = np.where(df.region.values == "grey", 1.0,
            np.where(df.region.values == "white", 0.0, np.nan))
    smooth = knn_smooth(xy, value, knn)

    x0, x1, y0, y1 = xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()
    gx = np.linspace(x0 - spacing, x1 + spacing, grid_res)
    gy = np.linspace(y0 - spacing, y1 + spacing, grid_res)
    GX, GY = np.meshgrid(gx, gy)
    grid = griddata(xy, smooth, (GX, GY), method="linear")
    dist, _ = cKDTree(xy).query(np.column_stack([GX.ravel(), GY.ravel()]), k=1)
    grid.ravel()[dist > 1.5 * spacing] = np.nan
    mask = ~np.isnan(grid)
    if sigma > 0:
        filled = np.where(mask, grid, 0.5)
        grid = gaussian_filter(filled, sigma); grid[~mask] = np.nan

    fig = plt.figure(); ax = fig.add_subplot(111)
    cs = ax.contour(GX, GY, grid, levels=[0.5])
    try:
        segs = list(cs.allsegs[0]) if cs.allsegs else []
    except AttributeError:
        segs = []
        for p in cs.get_paths():
            segs.extend(p.to_polygons(closed_only=False))
    plt.close(fig)

    min_len = min_seg_frac * (x1 - x0 + y1 - y0)
    kept = [s for s in segs if np.sum(np.hypot(np.diff(s[:, 0]), np.diff(s[:, 1]))) >= min_len]
    log.info("  boundary: %d/%d segment(s) kept (min_len=%.0f px)", len(kept), len(segs), min_len)
    return kept


def save_segments(path, segments):
    rows = [{"path_id": pid, "x": x, "y": y}
            for pid, seg in enumerate(segments) for x, y in seg]
    pd.DataFrame(rows, columns=["path_id", "x", "y"]).to_csv(path, sep="\t", index=False)


def load_segments(path):
    df = pd.read_csv(path, sep="\t")
    return [g[["x", "y"]].to_numpy(float) for _, g in df.groupby("path_id")]


def load_boundary(bdir, sample):
    """Prefer manual_<sample>.tsv, else auto_<sample>.tsv. Returns (segments, source)."""
    manual = os.path.join(bdir, f"manual_{sample}.tsv")
    auto = os.path.join(bdir, f"auto_{sample}.tsv")
    if os.path.isfile(manual):
        return load_segments(manual), "manual"
    if os.path.isfile(auto):
        return load_segments(auto), "auto"
    return [], "none"


def hires_scalef(chip_dir):
    """(image_array, hires/fullres scale). Derived without scalefactors_json."""
    img_path = os.path.join(chip_dir, "spatial", "tissue_hires_image.png")
    summ_path = os.path.join(chip_dir, "spatial", "spatial_summary.json")
    if not (os.path.isfile(img_path) and os.path.isfile(summ_path)):
        return None, None
    try:
        from PIL import Image
    except ImportError:
        log.warning("Pillow not installed -> no H&E overlay")
        return None, None
    img = np.asarray(Image.open(img_path))
    with open(summ_path) as fh:
        full_w = json.load(fh).get("image_dims", [None])[0]
    if not full_w:
        return None, None
    return img, img.shape[1] / float(full_w)


# --------------------------------------------------------------------------- #
# Plotting primitives                                                          #
# --------------------------------------------------------------------------- #
def savefig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s.{png,pdf}", os.path.join(outdir, name))


def _ordered_panel_genes(var_names):
    ordered, groups, pos, start = [], [], [], 0
    for name, genes in {**MODULES, **SINGLE_GROUPS}.items():
        have = [g for g in genes if g in var_names]
        if not have:
            continue
        ordered += have; groups.append(name)
        pos.append((start, start + len(have) - 1)); start += len(have)
    return ordered, groups, pos


def _dotplot_sig_map(stats_pooled):
    if stats_pooled is None or stats_pooled.empty:
        return {}
    sub = stats_pooled[stats_pooled.feature_kind.isin(["single_gene", "module_member_gene"])]
    return dict(zip(sub.feature, sub.sig))


def _style_and_save_dotplot(dp, outdir, name, genes_sig, swapped, italic):
    ax = dp.get_axes()["mainplot_ax"]
    # [新增] 调整 vertical 图的 X 轴标签角度 (仅当 swapped=True 时，X轴才是 region 分组)
    if swapped:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    labels = ax.get_yticklabels() if swapped else ax.get_xticklabels()
    ticks = ax.get_yticks() if swapped else ax.get_xticks()
    if italic:
        for lbl in labels:
            lbl.set_style("italic")
    for t, lbl in zip(ticks, labels):
        s = genes_sig.get(lbl.get_text())
        if not s or s == "ns":
            continue
        if swapped:
            ax.annotate(s, xy=(1.0, t), xycoords=("axes fraction", "data"),
                        xytext=(25, 0), textcoords="offset points", va="center", ha="left",
                        fontsize=8, fontweight="bold", clip_on=False)
#            ax.annotate(s, xy=(1.0, t), xycoords=("axes fraction", "data"),
#                        xytext=(6, 0), textcoords="offset points", va="center", ha="left",
#                        fontsize=8, fontweight="bold", clip_on=False)
        else:
            ax.annotate(s, xy=(t, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", clip_on=False)
    savefig(ax.figure, outdir, name)


def plot_dotplot(adata, outdir, stats_pooled=None, region_order=("grey", "white"), italic=True):
    ad_gw = adata[np.isin(adata.obs["region"], ["grey", "white"])].copy()
    present = [r for r in region_order if r in set(ad_gw.obs["region"].astype(str))]
    ad_gw.obs["region"] = pd.Categorical(ad_gw.obs["region"].astype(str),
                                         categories=present, ordered=True)
    ordered, groups, pos = _ordered_panel_genes(ad_gw.var_names)
    if not ordered:
        log.warning("no panel genes for dot plot"); return
    genes_sig = _dotplot_sig_map(stats_pooled)
    common = dict(groupby="region", standard_scale="var", var_group_positions=pos,
                  var_group_labels=groups, title="Marker expression: grey vs white",
                  show=False, return_fig=True)
#    dp = sc.pl.dotplot(ad_gw, ordered, **common)                       # genes as columns
    dp = sc.pl.dotplot(ad_gw, ordered, figsize=(len(ordered)*0.3, 3.5), **common)
    _style_and_save_dotplot(dp, outdir, "fig_dotplot", genes_sig, swapped=False, italic=italic)
#    dp2 = sc.pl.dotplot(ad_gw, ordered, swap_axes=True, **common)      # genes as rows
    dp2 = sc.pl.dotplot(ad_gw, ordered, swap_axes=True, figsize=(3.0, len(ordered)*0.3), **common)
    _style_and_save_dotplot(dp2, outdir, "fig_dotplot_vertical", genes_sig, swapped=True, italic=italic)


def plot_module_violin(scores_long, stats_pooled, outdir, region_order=("grey", "white")):
    if scores_long is None or scores_long.empty:
        return
    mods = list(dict.fromkeys(scores_long["module"]))
    order_r = [r for r in region_order if r in set(scores_long["region"])]
    fig, ax = plt.subplots(figsize=(1.8 * len(mods) + 2, 5))
    sns.violinplot(data=scores_long, x="module", y="score", hue="region", split=True,
                   inner="quartile", palette=REGION_COLORS, order=mods, hue_order=order_r,
                   ax=ax, cut=0)
    ax.set_xlabel(""); ax.set_ylabel("Module score"); ax.set_title("Module scores: grey vs white (pooled)")
    ymin, ymax = float(scores_long["score"].min()), float(scores_long["score"].max())
    rng = (ymax - ymin) or 1.0
    ax.set_ylim(ymin - 0.05 * rng, ymax + 0.18 * rng)   # headroom for stars
    if stats_pooled is not None and not stats_pooled.empty:
        for i, m in enumerate(mods):
            row = stats_pooled[(stats_pooled.feature == m) & (stats_pooled.feature_kind == "module_score")]
            if not row.empty:
                ax.text(i, ymax + 0.06 * rng, row.iloc[0]["sig"], ha="center", va="bottom", fontsize=11)
    ax.legend(title="region", frameon=False, loc="lower right")
    savefig(fig, outdir, "fig_module_violin")


def plot_single_gene_box(expr_long, stats_pooled, outdir, region_order=("grey", "white"),
                         drop_zeros=True, one_row=True, max_points=1500, gene_subset=None):
    """Single-gene grey-vs-white boxes. By default only expressing spots (expr>0)
    are boxed (Visium is sparse); the detection rate is annotated as 'xx%+'.
    Overlays a faint jitter sample and per-chip mean line. gene_subset overrides
    which genes to draw (default: the configured Single_DAM set)."""
    base = gene_subset if gene_subset is not None else SINGLE_DAM
    genes = [g for g in base if g in set(expr_long["gene"])]
    if not genes:
        log.warning("no single-gene-set genes for boxplot"); return
    order_r = [r for r in region_order if r in set(expr_long["region"])]
    ncols, nrows = (len(genes), 1) if one_row else (min(3, len(genes)), ceil(len(genes) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 4.0 * nrows), squeeze=False)
    sig_map = {}
    if stats_pooled is not None and not stats_pooled.empty:
        s = stats_pooled[stats_pooled.feature_kind == "single_gene"]
        sig_map = dict(zip(s.feature, s.sig))
    for j, (ax, g) in enumerate(zip(axes.ravel(), genes)):
        d_all = expr_long[expr_long.gene == g]
        d = d_all[d_all.expr > 0] if drop_zeros else d_all
        if d.empty:
            ax.set_title(g, style="italic"); ax.axis("off"); continue
        sns.boxplot(data=d, x="region", y="expr", order=order_r, hue="region",
                    hue_order=order_r, palette=REGION_COLORS, legend=False,
                    showfliers=False, width=0.6, ax=ax)
        samp = d.sample(min(max_points, len(d)), random_state=0) if len(d) > max_points else d
        sns.stripplot(data=samp, x="region", y="expr", order=order_r, ax=ax,
                      size=1.5, color="0.25", alpha=0.25, jitter=0.25)
        cm = d.groupby(["sample", "region"])["expr"].mean().reset_index()
        for s_ in cm["sample"].unique():
            sub = cm[cm["sample"] == s_].set_index("region").reindex(order_r)
            ax.plot(range(len(order_r)), sub["expr"].values, "-o", color="black",
                    lw=0.6, ms=3, alpha=0.6, zorder=5)
        ymax = float(d["expr"].max()); ax.set_ylim(0, ymax * 1.22)
        for i, region in enumerate(order_r):
            pct = (d_all.loc[d_all.region == region, "expr"] > 0).mean() * 100
            ax.text(i, ymax * 1.04, f"{pct:.0f}%+", ha="center", va="bottom", fontsize=7, color="0.3")
        sg = sig_map.get(g, "")
        if sg and sg != "ns":
            ax.text((len(order_r) - 1) / 2, ymax * 1.13, sg, ha="center", va="bottom", fontsize=11)
        ax.set_title(g, style="italic"); ax.set_xlabel("")
        ax.set_ylabel("log1p expr" + (" (expressing)" if drop_zeros else "") if j == 0 else "")
    for ax in axes.ravel()[len(genes):]:
        ax.axis("off")
    note = "expressing spots only, %+=detection" if drop_zeros else "all spots"
    fig.suptitle(f"Single-gene expression grey vs white ({note}; line=per-chip mean)")
    savefig(fig, outdir, "fig_single_gene_box")


def plot_log2fc_bar(stats_pooled, outdir):
    if stats_pooled is None or stats_pooled.empty:
        return
    df = stats_pooled.dropna(subset=["log2FC_grey_over_white"]).copy()
    if df.empty:
        log.warning("no expression features for log2FC bar"); return
    df = df.sort_values("log2FC_grey_over_white")
    colors = [REGION_COLORS["grey"] if v > 0 else REGION_COLORS["white"]
              for v in df["log2FC_grey_over_white"]]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(df) + 1.5))
    ax.barh(df["feature"] + " (" + df["feature_kind"] + ")", df["log2FC_grey_over_white"], color=colors)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("log2 FC (grey / white)")
    ax.set_title("Marker enrichment (pooled)\ngrey>0 -> grey   white<0 -> white")
    for y, (_, r) in enumerate(df.iterrows()):
        ax.text(r["log2FC_grey_over_white"], y, " " + r["sig"], va="center",
                ha="left" if r["log2FC_grey_over_white"] >= 0 else "right", fontsize=8)
    savefig(fig, outdir, "fig_log2FC_bar")


def plot_paired_perchip(perchip_stats, outdir):
    if perchip_stats is None or perchip_stats.empty:
        return
    mods = list(dict.fromkeys(
        perchip_stats.loc[perchip_stats.feature_kind == "module_score", "feature"]))
    if not mods:
        return
    fig, axes = plt.subplots(1, len(mods), figsize=(3.2 * len(mods), 4.5), squeeze=False)
    for ax, m in zip(axes[0], mods):
        sub = perchip_stats[(perchip_stats.feature == m) &
                            (perchip_stats.feature_kind == "module_score")]
        for _, r in sub.iterrows():
            ax.plot([0, 1], [r["mean_grey"], r["mean_white"]], color="0.6", lw=0.9)
            ax.plot(0, r["mean_grey"], "o", color=REGION_COLORS["grey"], ms=5)
            ax.plot(1, r["mean_white"], "o", color=REGION_COLORS["white"], ms=5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["grey", "white"])
        ax.set_xlim(-0.3, 1.3); ax.set_title(m); ax.set_ylabel("mean module score")
    fig.suptitle("Per-chip grey vs white (each line = one chip)")
    fig.subplots_adjust(wspace=0.4)
    savefig(fig, outdir, "fig_module_paired_perchip")


def plot_allpos_fraction(allpos_pooled, outdir, region_order=("grey", "white")):
    if allpos_pooled is None or allpos_pooled.empty:
        return
    order_r = [r for r in region_order if r in set(allpos_pooled["region"])]
    fig, ax = plt.subplots(figsize=(1.6 * allpos_pooled.module.nunique() + 2, 4.5))
    sns.barplot(data=allpos_pooled, x="module", y="frac_allpos", hue="region",
                hue_order=order_r, palette=REGION_COLORS, ax=ax)
    ax.set_ylabel("fraction of all-positive spots"); ax.set_xlabel("")
    ax.set_title("All-positive (co-expressing) spot fraction (pooled)")
    ax.legend(title="region", frameon=False)
    savefig(fig, outdir, "fig_allpositive_fraction")


# --------------------------------------------------------------------------- #
# Spatial maps with boundary                                                   #
# --------------------------------------------------------------------------- #
def _panel_keys(adata, include_single):
    keys = ["region"]
    keys += [f"score_{m}" for m in MODULES
             if f"score_{m}" in adata.obs and not adata.obs[f"score_{m}"].isna().all()]
    if include_single:
        keys += [g for g in SINGLE_DAM if g in adata.var_names]
    return keys


def _draw_spatial_panel(ax, sub, coords, key, img, point_size):
    if key == "region":
        for region, color in REGION_COLORS.items():
            m = sub.obs["region"].values == region
            if m.any():
                ax.scatter(coords[m, 0], coords[m, 1], s=point_size, c=color,
                           label=region, linewidths=0)
        ax.legend(frameon=False, markerscale=2, fontsize=6, loc="best")
        ax.set_title("region")
    else:
        vals = (sub.obs[key].values.astype(float) if key.startswith("score_")
                else expr_vector(sub, key))
        sc_ = ax.scatter(coords[:, 0], coords[:, 1], s=point_size, c=vals,
                         cmap="viridis", linewidths=0)
        ax.figure.colorbar(sc_, ax=ax, fraction=0.046, pad=0.04)
        if key.startswith("score_"):
            ax.set_title(key.replace("score_", ""))      # module / group name
        else:
            ax.set_title(key, style="italic")            # gene symbol -> italic


def plot_spatial_all(adata, bdir, sample_dirs, outdir, include_single=True,
                     backgrounds=("scatter", "he"), ncol=4, point_size=6, samples=None,
                     boundary_color="#FFD700", boundary_width=1.8, show_none=False):
    spatial_dir = os.path.join(outdir, "spatial_maps")
    keys = _panel_keys(adata, include_single)
    all_samples = adata.obs["sample"].unique().tolist()
    todo = [s for s in all_samples if s in set(samples)] if samples else all_samples
    for sample in todo:
        sub = adata[adata.obs["sample"] == sample]
        if not show_none:                       # keep only grey/white spots
            sub = sub[np.isin(sub.obs["region"], ["grey", "white"])]
        if "spatial" not in sub.obsm or sub.n_obs == 0:
            log.warning("[%s] no spatial coords / no grey-white spots -- skipped", sample); continue
        xy = sub.obsm["spatial"]
        segments, source = load_boundary(bdir, sample)
        for bg in backgrounds:
            img, sf = (None, 1.0)
            if bg == "he":
                img, sf = hires_scalef(sample_dirs.get(sample, ""))
                if img is None:
                    continue
            coords = xy * sf
            ncols = min(ncol, len(keys)); nrows = ceil(len(keys) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False)
            for ax, key in zip(axes.ravel(), keys):
                if img is not None:
                    ax.imshow(img)
                _draw_spatial_panel(ax, sub, coords, key, img, point_size)
                for seg in segments:
                    ax.plot(seg[:, 0] * sf, seg[:, 1] * sf, "--", color=boundary_color,
                            lw=boundary_width, zorder=6,
                            path_effects=[pe.Stroke(linewidth=boundary_width + 1.6,
                                                    foreground="black"), pe.Normal()])
                if img is None:
                    ax.invert_yaxis()
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for ax in axes.ravel()[len(keys):]:
                ax.axis("off")
            fig.suptitle(f"{sample}  ({bg}; boundary={source})")
            savefig(fig, spatial_dir, f"spatial_{bg}_{sample}")


def draw_interactive(adata, sample, chip_dir, bdir, on_he=True, show_guide=True):
    """Trace a grey/white boundary by clicking; saves boundaries/manual_<sample>.tsv.
    Clicked points are joined in order into one polyline and saved verbatim
    (no smoothing/refit); manual overrides auto at plot time. The current
    boundary is shown as a faint guide unless show_guide=False.
    Needs an interactive matplotlib backend (a display)."""
    sub = adata[adata.obs["sample"] == sample]
    if sub.n_obs == 0 or "spatial" not in sub.obsm:
        print(f"[draw] no data/coords for {sample}"); return False
    xy = sub.obsm["spatial"]
    img, sf = (hires_scalef(chip_dir) if on_he else (None, None))
    if img is not None:
        coords = xy * sf; back = 1.0 / sf
    else:
        coords = xy; sf = 1.0; back = 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    if img is not None:
        ax.imshow(img)
    for region, color in REGION_COLORS.items():
        m = sub.obs["region"].values == region
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=7, c=color, alpha=0.5,
                       linewidths=0, label=region)
    # faint guide = current boundary
    if show_guide:
        guide, src = load_boundary(bdir, sample)
        for seg in guide:
            ax.plot(seg[:, 0] * sf, seg[:, 1] * sf, "--", color="cyan", lw=1.0, alpha=0.7)
    else:
        src = "hidden"
    if img is None:
        ax.invert_yaxis()
    ax.set_aspect("equal"); ax.legend(frameon=False, markerscale=2, fontsize=8)
    ax.set_title(f"DRAW boundary: {sample}\n"
                 f"left=add  right=undo  middle/Enter=finish   (cyan = current {src})")
    print(f"[draw] {sample}: left-click add point, right-click undo, "
          "middle-click or Enter to finish. (cyan dashed = current boundary)")
    plt.show(block=False)
    plt.pause(0.1)
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    if len(pts) < 2:
        print("[draw] <2 points clicked -> nothing saved, existing boundary kept.")
        return False
    os.makedirs(bdir, exist_ok=True)
    out = os.path.join(bdir, f"manual_{sample}.tsv")
    save_segments(out, [np.array(pts) * back])
    print(f"[draw] saved {len(pts)} points -> {out}")
    return True


# --------------------------------------------------------------------------- #
# README generator                                                             #
# --------------------------------------------------------------------------- #
def write_readme(outdir, ctx):
    """ctx: dict with keys data_dir, info, n_chips_total, n_chips_used, n_spots,
    n_grey, n_white, n_none, target_sum, knn, grid_res, sigma, min_seg_frac,
    modules, single_groups."""
    mods = "\n".join(f"  - **{k}** (co-expression): {', '.join(v)}" for k, v in ctx["modules"].items())
    sing = "\n".join(f"  - **{k}** (per-gene): {', '.join(v)}" for k, v in ctx["single_groups"].items())
    md = f"""# Grey / White matter marker analysis — results

Generated by `01_compute.py`. Inputs: `{ctx['data_dir']}` + `{ctx['info']}`.

## What was computed

**Region assignment.** For each chip the clustering named in `spatial_info.txt`
column *Number of cluster* is used: `graph based` -> `graph_based`; a number *N*
-> `gene_expression_k_means_k_N`. Cluster ids in the *Grey* / *White* columns are
mapped to `Cluster N` and labelled `grey` / `white`; spots in neither are `none`
(kept only for tissue context and to anchor the boundary, never in statistics).

**Normalisation.** Raw counts -> `normalize_total(target_sum={ctx['target_sum']:g})`
-> `log1p` (log1p CPM). Raw counts kept in layer `counts`.

**Gene panel.**
{mods}
{sing}

- Module score: `scanpy.tl.score_genes` (ctrl_size=50, n_bins=25) over the set's
  genes. Scoring mode: **{ctx['score_mode']}**.
- All-positive fraction: fraction of spots detecting **every** gene in the set (counts>0).
- Single-gene sets are tested / drawn per gene.

**Statistics (grey vs white).** Mann–Whitney U (two-sided) per feature; BH FDR
across features within a scope. `log2FC_grey_over_white` is computed on linear
(expm1) expression for genes (NaN for module scores, which can be negative);
direction `enriched_in` uses the mean difference. `pct_grey`/`pct_white` =
fraction of spots expressing the gene.

**Boundary.** grey=1/white=0 label field -> KNN smoothing (k={ctx['knn']}) ->
grid interpolation ({ctx['grid_res']}²) -> off-tissue mask -> gaussian
(sigma={ctx['sigma']}) -> 0.5 iso-contour; segments shorter than
{ctx['min_seg_frac']:.0%} of the field are dropped. Coordinates are
full-resolution pixels.

## Cohort

- chips used: **{ctx['n_chips_used']} / {ctx['n_chips_total']}**
- pooled spots: **{ctx['n_spots']}** (grey={ctx['n_grey']}, white={ctx['n_white']}, none={ctx['n_none']})

## Files

| file | meaning |
|------|---------|
| `plot_bundle.h5ad` | all tissue spots × panel genes (log1p), obs region/sample/score/allpos, `obsm['spatial']`, panel in `uns` |
| `sd_module_scores_long.tsv.gz` | spot-level module scores (grey/white) — violin source |
| `sd_gene_expr_long.tsv.gz` | spot-level log1p expression per panel gene (grey/white) — box/dotplot source |
| `sd_dotplot.tsv` | region×gene mean expr + % expressing |
| `sd_stats_pooled.tsv` | pooled grey-vs-white stats (mean, log2FC, p, FDR, sig, %expr) |
| `sd_stats_per_chip.tsv` | same, per chip |
| `sd_allpos_fraction.tsv` | all-positive spot fraction per region (pooled) |
| `sd_allpos_fraction_per_chip.tsv` | same, per chip |
| `boundaries/auto_<sample>.tsv` | automatic boundary polyline(s) — editable |
| `boundaries/manual_<sample>.tsv` | your override (used by `02_plot_spatial.py` if present) |

## How to (re)draw

```bash
# 1. compute once (produces everything above)
python 01_compute.py --data-dir {ctx['data_dir']} --info {ctx['info']} --gene-list gene.list.txt --outdir {os.path.basename(outdir)}

# 2. spatial maps with boundary (all panels, both backgrounds)
python 02_plot_spatial.py --bundle {os.path.basename(outdir)}/plot_bundle.h5ad --data-dir {ctx['data_dir']} --outdir {os.path.basename(outdir)}

# 3. adjust boundaries you don't like, then RE-RUN step 2 only:
#    cp {os.path.basename(outdir)}/boundaries/auto_<sample>.tsv {os.path.basename(outdir)}/boundaries/manual_<sample>.tsv  # then edit
#    or:  python 02_plot_spatial.py --draw <sample> --bundle ... --data-dir ...

# 4. summary figures (dotplot / violin / single-gene box / log2FC / paired)
python 03_plot_summary.py --sd-dir {os.path.basename(outdir)} --bundle {os.path.basename(outdir)}/plot_bundle.h5ad --outdir {os.path.basename(outdir)}/figures
```

Sanity check: Neuron (NEFM/SNAP25) should enrich grey matter, Oligo (MBP/PLP1) white matter.
"""
    path = os.path.join(outdir, "README_results.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    log.info("wrote %s", path)
