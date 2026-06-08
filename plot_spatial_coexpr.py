#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spatial_coexpr.py -- per-group gene co-expression spatial maps
===================================================================

Reads a gene-list file (groups), reloads raw matrices, and for each group maps
where its genes are CO-EXPRESSED, three ways (detection = raw counts > 0):
  * co     : strict AND (all genes detected) -> binary highlight;
             with --k K, switches to k-of-N (>=K detected) and does NOT run AND
  * count  : number of genes detected per spot (0..N) -> colour map
  * score  : min over genes of per-gene 0-1 scaled log1p expression -> colour map
Each panel overlays the precomputed grey/white boundary and supports per-sample
rotation. Also writes a grey-vs-white co-expression fraction table.

PDF organisation:
  default            -> one PDF per metric  (panels per page = groups)
  --by-group         -> one PDF per group   (panels per page = metrics)   [for comparison]

  python plot_spatial_coexpr.py --data-dir matrix --info spatial_info.txt \
      --gene-list gene.list.txt --boundary-dir results_gw/boundaries \
      --outdir results_coexpr/coexpr --rotate-file rotations.tsv --range both
"""

import argparse
import os
import re
import sys
from math import ceil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm

import coexpr_common as cc
import gw_common as gw

log = gw.log
HE = "__HE__"


def metric_label(metric, k):
    if metric == "co":
        return f"co (>={k})" if k else "co (AND all)"
    return {"count": "# genes detected", "score": "min co-score"}[metric]


def sanitize(s):
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")


def discrete_count_cmap(count_max, base_cmap, zero_color):
    """Fixed colour blocks for integer counts 0..count_max. 0 -> zero_color,
    1..count_max -> evenly sampled from base_cmap. A given count maps to the SAME
    colour in every panel (count_max is shared across groups), so 2-/3-/6-gene
    groups are comparable; smaller groups simply never reach the top colours."""
    base = plt.get_cmap(base_cmap)
    cols = [zero_color] + [base(i / max(1, count_max)) for i in range(1, count_max + 1)]
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(np.arange(-0.5, count_max + 1.5, 1.0), cmap.N)
    return cmap, norm


def draw_metric(ax, sub, center, deg, segments, genes, metric, k, args, title, count_max):
    genes = [g for g in genes if g in sub.var_names]
    ax.set_title(title)
    if not genes:
        ax.text(0.5, 0.5, "no genes", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([]); return
    xy = sub.obsm["spatial"]
    common = dict(point_size=args.point_size, boundary_color=args.boundary_color,
                  boundary_width=args.boundary_width, title=title)
    det = cc.counts_of(sub, genes) > 0
    npos = det.sum(axis=1)
    if metric == "co":
        co = (npos >= k) if k else det.all(axis=1)
        cc.draw_map(ax, xy, center, deg, segments, highlight=co,
                    bg_color=args.bg_color, hi_color=args.hi_color, **common)
    elif metric == "count":
        cmap_d, norm_d = discrete_count_cmap(count_max, args.cmap, args.bg_color)
        cc.draw_map(ax, xy, center, deg, segments, values=npos,
                    cmap=cmap_d, norm=norm_d, **common)
    else:  # score (robust per-gene scaling, then min across genes)
        Xs = np.column_stack([cc.scale01_robust(cc.expr_of(sub, g),
                                                args.score_lo_pct, args.score_hi_pct)
                              for g in genes])
        cc.draw_map(ax, xy, center, deg, segments, values=Xs.min(axis=1),
                    cmap=args.cmap, vmin=0, vmax=1, **common)


def render_pdf(pdf_path, chunk, panels, fill_panel, suptitle_fn, args, rot, ncols, rng):
    nrows = ceil(len(panels) / ncols)
    with PdfPages(pdf_path) as pdf:
        for sample, adata in chunk:
            center = adata.obsm["spatial"].mean(axis=0)
            deg = rot.get(sample, args.rotate)
            chip_dir = os.path.join(args.data_dir, str(adata.obs["folder"].iloc[0]))
            segments, _ = gw.load_boundary(args.boundary_dir, sample)
            sub = adata if rng == "all" else adata[np.isin(adata.obs["region"], ["grey", "white"])]
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.2), squeeze=False)
            for ax, panel in zip(axes.ravel(), panels):
                if panel == HE:
                    cc.draw_he(ax, chip_dir, deg)
                else:
                    fill_panel(ax, panel, sub, center, deg, segments)
            for ax in axes.ravel()[len(panels):]:
                ax.axis("off")
            fig.suptitle(suptitle_fn(sample, deg, rng))
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, dpi=200)
            if args.png:
                fig.savefig(pdf_path[:-4] + f"_{sample}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    log.info("wrote %s (%d sample(s))", pdf_path, len(chunk))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="matrix")
    ap.add_argument("--info", default="spatial_info.txt")
    ap.add_argument("--gene-list", required=True)
    ap.add_argument("--outdir", default="results_coexpr/coexpr")
    ap.add_argument("--boundary-dir", default="results_gw/boundaries")
    ap.add_argument("--range", choices=["all", "grey_white", "both"], default="all")
    ap.add_argument("--metrics", default="co,count,score", help="comma subset of co,count,score")
    ap.add_argument("--k", type=int, default=0, help="k-of-N: detected in >=k genes (0 = strict AND)")
    ap.add_argument("--by-group", action="store_true",
                    help="one PDF per group (panels = metrics) instead of one PDF per metric")
    ap.add_argument("--no-he", action="store_true")
    ap.add_argument("--ncols", type=int, default=0, help="panels per row (0 = auto)")
    ap.add_argument("--samples-per-pdf", type=int, default=3)
    ap.add_argument("--rotate-file", default=None)
    ap.add_argument("--rotate", type=float, default=0.0)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--count-max", type=int, default=0,
                    help="fixed top of the discrete 'count' colour scale (0 = auto: "
                         "largest group size, so counts are comparable across groups)")
    ap.add_argument("--score-lo-pct", type=float, default=1.0,
                    help="lower percentile for robust per-gene scaling of the co-score")
    ap.add_argument("--score-hi-pct", type=float, default=99.0,
                    help="upper percentile for robust per-gene scaling of the co-score")
    ap.add_argument("--point-size", type=float, default=8.0)
    ap.add_argument("--bg-color", default="#e8e8e8", help="non-co-expressing spot colour")
    ap.add_argument("--hi-color", default="#d62728", help="co-expressing spot colour")
    ap.add_argument("--boundary-color", default="#FFD700")
    ap.add_argument("--boundary-width", type=float, default=1.4)
    ap.add_argument("--name", default="coexpr")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()

    gw.setup_logging(args.outdir, filename="plot_spatial_coexpr.log")
    panel = gw.parse_gene_list(args.gene_list)
    if not panel:
        log.error("no gene sets parsed from %s", args.gene_list); sys.exit(1)
    group_genes = {name: list(genes) for name, genes in panel}
    groups = list(group_genes)
    metrics = [m for m in args.metrics.split(",") if m.strip() in ("co", "count", "score")]
    k = args.k if args.k > 0 else None
    log.info("groups=%s | metrics=%s | k=%s | by_group=%s", groups, metrics, k, args.by_group)

    samples = cc.load_samples(args.data_dir, args.info)
    if not samples:
        log.error("no samples loaded"); sys.exit(1)
    rot = cc.read_rotations(args.rotate_file, default=args.rotate)

    # shared top of the discrete count scale: largest group size (genes present)
    var0 = set(next(iter(samples.values())).var_names)
    group_N = {g: len([x for x in genes if x in var0]) for g, genes in group_genes.items()}
    count_max = args.count_max if args.count_max > 0 else max([1] + list(group_N.values()))
    log.info("count scale 0..%d (group sizes: %s)", count_max, group_N)

    # ---- grey-vs-white co-expression fraction table ---------------------- #
    rows = []
    for sample, adata in samples.items():
        reg = adata.obs["region"].values
        for grp, genes in group_genes.items():
            gp = [g for g in genes if g in adata.var_names]
            if not gp:
                continue
            det = cc.counts_of(adata, gp) > 0
            npos = det.sum(axis=1)
            co = (npos >= k) if k else det.all(axis=1)
            for region in ("grey", "white"):
                m = reg == region
                rows.append({"sample": sample, "group": grp, "region": region,
                             "n_spots": int(m.sum()), "n_coexpr": int(co[m].sum()),
                             "frac_coexpr": float(co[m].mean()) if m.sum() else np.nan,
                             "definition": (f">={k}of{len(gp)}" if k else f"AND_of_{len(gp)}"),
                             "n_genes_present": len(gp)})
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, f"{args.name}_fraction.tsv"),
                              sep="\t", index=False)
    log.info("wrote %s_fraction.tsv", args.name)

    ranges = ["all", "grey_white"] if args.range == "both" else [args.range]
    items = list(samples.items())
    spp = max(1, args.samples_per_pdf)
    chunks = [items[i:i + spp] for i in range(0, len(items), spp)]

    for rng in ranges:
        if not args.by_group:
            # one PDF per metric; panels = groups
            panels = ([HE] if not args.no_he else []) + groups
            ncols = args.ncols if args.ncols > 0 else len(panels)
            for metric in metrics:
                def fill(ax, grp, sub, center, deg, segments, _m=metric):
                    draw_metric(ax, sub, center, deg, segments, group_genes[grp], _m, k, args,
                                grp, count_max)
                stitle = (lambda s, d, r, _m=metric:
                          f"{s}  ({metric_label(_m, k)}; range={r}; rotate={d:g})")
                for ci, chunk in enumerate(chunks, 1):
                    out = os.path.join(args.outdir, f"{args.name}_{metric}_{rng}_part{ci:02d}.pdf")
                    render_pdf(out, chunk, panels, fill, stitle, args, rot, ncols, rng)
        else:
            # one PDF per group; panels = metrics
            for grp in groups:
                panels = ([HE] if not args.no_he else []) + metrics
                ncols = args.ncols if args.ncols > 0 else len(panels)
                def fill(ax, met, sub, center, deg, segments, _g=grp):
                    draw_metric(ax, sub, center, deg, segments, group_genes[_g], met, k, args,
                                metric_label(met, k), count_max)
                stitle = (lambda s, d, r, _g=grp: f"{s}  (group {_g}; range={r}; rotate={d:g})")
                for ci, chunk in enumerate(chunks, 1):
                    out = os.path.join(args.outdir, f"{args.name}_{sanitize(grp)}_{rng}_part{ci:02d}.pdf")
                    render_pdf(out, chunk, panels, fill, stitle, args, rot, ncols, rng)


if __name__ == "__main__":
    main()
