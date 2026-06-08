#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spatial_module.py -- per-group module-score spatial maps
=============================================================

Reads a gene-list file (groups), reloads raw matrices, normalizes to log1p CPM,
and computes a module score per group WITHIN each chip (scanpy score_genes;
per-chip background -> suited to per-sample maps). Each sample gets one page:
an H&E panel + one panel per group (colour = module score), with the precomputed
grey/white boundary overlaid and optional per-sample rotation. Samples split into
PDFs of N each (default 3). Reuses gw_common via coexpr_common; no existing
script is modified.

  python plot_spatial_module.py --data-dir matrix --info spatial_info.txt \
      --gene-list gene.list.txt --boundary-dir results_gw/boundaries \
      --outdir results_coexpr/module --rotate-file rotations.tsv

Module score can be negative (enrichment vs a matched-expression background);
try --cmap RdBu_r --center0 for a diverging, zero-centred colour scale.
"""

import argparse
import os
import sys
from math import ceil

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import coexpr_common as cc
import gw_common as gw

log = gw.log
HE = "__HE__"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="matrix")
    ap.add_argument("--info", default="spatial_info.txt")
    ap.add_argument("--gene-list", required=True, help="groups file: <name>\\t<g1,g2,...>")
    ap.add_argument("--outdir", default="results_coexpr/module")
    ap.add_argument("--boundary-dir", default="results_gw/boundaries")
    ap.add_argument("--range", choices=["all", "grey_white", "both"], default="all")
    ap.add_argument("--no-he", action="store_true")
    ap.add_argument("--ncols", type=int, default=0, help="panels per row (0 = auto: HE+groups)")
    ap.add_argument("--samples-per-pdf", type=int, default=3)
    ap.add_argument("--rotate-file", default=None)
    ap.add_argument("--rotate", type=float, default=0.0)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--center0", action="store_true",
                    help="symmetric colour limits centred at 0 (use with a diverging cmap)")
    ap.add_argument("--point-size", type=float, default=8.0)
    ap.add_argument("--boundary-color", default="#FFD700")
    ap.add_argument("--boundary-width", type=float, default=1.4)
    ap.add_argument("--shared-scale", action="store_true",
                    help="one colour scale per group across all samples")
    ap.add_argument("--name", default="spatial_module")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()

    gw.setup_logging(args.outdir, filename="plot_spatial_module.log")
    panel = gw.parse_gene_list(args.gene_list)
    if not panel:
        log.error("no gene sets parsed from %s", args.gene_list); sys.exit(1)
    gw.configure_panel(panel, [])                 # treat every group as a module
    groups = list(gw.MODULES)
    log.info("module groups: %s", groups)

    samples = cc.load_samples(args.data_dir, args.info)
    if not samples:
        log.error("no samples loaded"); sys.exit(1)
    for adata in samples.values():
        gw.add_module_scores(adata)               # per-chip module scores -> obs[score_<group>]

    rot = cc.read_rotations(args.rotate_file, default=args.rotate)
    ranges = ["all", "grey_white"] if args.range == "both" else [args.range]

    def sub_for(adata, rng):
        return adata if rng == "all" else adata[np.isin(adata.obs["region"], ["grey", "white"])]

    panels = ([HE] if not args.no_he else []) + groups
    ncols = args.ncols if args.ncols > 0 else len(panels)
    nrows = ceil(len(panels) / ncols)

    for rng in ranges:
        vlim = {}
        if args.shared_scale or args.center0:
            for grp in groups:
                vals = [sub_for(a, rng).obs[f"score_{grp}"].to_numpy(float) for a in samples.values()]
                allv = np.concatenate(vals)
                if args.center0:
                    a = float(np.nanmax(np.abs(allv))); vlim[grp] = (-a, a)
                else:
                    vlim[grp] = (float(np.nanmin(allv)), float(np.nanmax(allv)))

        items = list(samples.items())
        spp = max(1, args.samples_per_pdf)
        chunks = [items[i:i + spp] for i in range(0, len(items), spp)]
        for ci, chunk in enumerate(chunks, 1):
            pdf_path = os.path.join(args.outdir, f"{args.name}_{rng}_part{ci:02d}.pdf")
            with PdfPages(pdf_path) as pdf:
                for sample, adata in chunk:
                    center = adata.obsm["spatial"].mean(axis=0)
                    deg = rot.get(sample, args.rotate)
                    chip_dir = os.path.join(args.data_dir, str(adata.obs["folder"].iloc[0]))
                    segments, _ = gw.load_boundary(args.boundary_dir, sample)
                    sub = sub_for(adata, rng)
                    xy = sub.obsm["spatial"]
                    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.2),
                                             squeeze=False)
                    for ax, panel_name in zip(axes.ravel(), panels):
                        if panel_name == HE:
                            cc.draw_he(ax, chip_dir, deg); continue
                        vmin, vmax = vlim.get(panel_name, (None, None))
                        cc.draw_map(ax, xy, center, deg, segments,
                                    values=sub.obs[f"score_{panel_name}"].to_numpy(float),
                                    cmap=args.cmap, vmin=vmin, vmax=vmax, point_size=args.point_size,
                                    boundary_color=args.boundary_color, boundary_width=args.boundary_width,
                                    title=panel_name)
                    for ax in axes.ravel()[len(panels):]:
                        ax.axis("off")
                    fig.suptitle(f"{sample}  (module score, per-chip; range={rng}; rotate={deg:g})")
                    fig.tight_layout(rect=(0, 0, 1, 0.96))
                    pdf.savefig(fig, dpi=200)
                    if args.png:
                        fig.savefig(os.path.join(args.outdir, f"{args.name}_{rng}_{sample}.png"),
                                    dpi=200, bbox_inches="tight")
                    plt.close(fig)
            log.info("wrote %s (%d sample(s))", pdf_path, len(chunk))


if __name__ == "__main__":
    main()
