#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_spatial_genes.py -- single-gene spatial expression maps (separate PDFs)
============================================================================

Reloads raw matrices (ANY gene), normalizes to log1p CPM, and for each sample
draws one page: an H&E tissue panel + one panel per requested gene (colour =
log1p expression), the precomputed grey/white boundary overlaid, and optional
per-sample clockwise rotation for orientation alignment. Samples are split into
PDFs of N samples each (default 3). Reuses gw_common via coexpr_common; does NOT
modify any existing script.

  python plot_spatial_genes.py --data-dir matrix --info spatial_info.txt \
      --gene CLEC7A,ITGAX,LGALS3,SPP1,LPL,CD9 --ncols 7 \
      --boundary-dir results_gw/boundaries --outdir results_coexpr/genes \
      --rotate-file rotations.tsv --samples-per-pdf 3

H&E + 6 genes = 7 panels per sample -> use --ncols 7.
rotations.tsv: tab-separated, columns 'sample' and 'degrees' (negative allowed).
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


def parse_genes(args):
    genes = []
    if args.gene_file and os.path.isfile(args.gene_file):
        with open(args.gene_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0]
                genes += [g.strip() for g in line.replace(",", " ").split() if g.strip()]
    if args.gene:
        genes += [g.strip() for g in args.gene.replace(",", " ").split() if g.strip()]
    return list(dict.fromkeys(genes))


def draw_he(ax, chip_dir, deg):
    """H&E panel: read tissue_hires_image.png DIRECTLY (no dependence on
    spatial_summary.json / scale factor -- the panel only needs the image)."""
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("H&E")
    name = os.path.basename(chip_dir)
    img_path = os.path.join(chip_dir, "spatial", "tissue_hires_image.png")
    if not os.path.isfile(img_path):
        log.warning("[%s] no H&E: image missing -> %s", name, img_path)
        ax.text(0.5, 0.5, "no H&E\n(image missing)", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        return
    try:
        from PIL import Image
        im = Image.open(img_path).convert("RGB").rotate(-deg, expand=True, fillcolor=(255, 255, 255))
        ax.imshow(np.asarray(im)); ax.set_aspect("equal")
    except Exception as e:
        log.warning("[%s] H&E load/rotate failed: %s", name, e)
        ax.text(0.5, 0.5, "H&E error", ha="center", va="center", transform=ax.transAxes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="matrix")
    ap.add_argument("--info", default="spatial_info.txt")
    ap.add_argument("--outdir", default="results_coexpr/genes")
    ap.add_argument("--gene", default=None, help="comma/space separated gene symbols")
    ap.add_argument("--gene-file", default=None, help="file of gene symbols (comma/newline)")
    ap.add_argument("--boundary-dir", default="results_gw/boundaries")
    ap.add_argument("--range", choices=["all", "grey_white"], default="all",
                    help="which spots to colour (default: all tissue)")
    ap.add_argument("--no-he", action="store_true", help="omit the H&E tissue panel")
    ap.add_argument("--ncols", type=int, default=7, help="panels per row (H&E + genes)")
    ap.add_argument("--samples-per-pdf", type=int, default=3)
    ap.add_argument("--rotate-file", default=None, help="rotations.tsv (sample<TAB>degrees)")
    ap.add_argument("--rotate", type=float, default=0.0, help="global default rotation (deg, CW)")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--point-size", type=float, default=8.0)
    ap.add_argument("--boundary-color", default="#FFD700")
    ap.add_argument("--boundary-width", type=float, default=1.4)
    ap.add_argument("--shared-scale", action="store_true",
                    help="one colour scale per gene across all samples (for comparison)")
    ap.add_argument("--name", default="spatial_genes")
    ap.add_argument("--png", action="store_true", help="also write one PNG per page")
    args = ap.parse_args()

    gw.setup_logging(args.outdir, filename="plot_spatial_genes.log")
    genes = parse_genes(args)
    if not genes:
        log.error("no genes given (use --gene or --gene-file)"); sys.exit(1)
    log.info("genes: %s", genes)

    samples = cc.load_samples(args.data_dir, args.info)
    if not samples:
        log.error("no samples loaded; check --data-dir / --info"); sys.exit(1)
    rot = cc.read_rotations(args.rotate_file, default=args.rotate)
    rng = args.range

    def sub_for(adata):
        return adata if rng == "all" else adata[np.isin(adata.obs["region"], ["grey", "white"])]

    # optional shared colour scale per gene across samples
    vlim = {}
    if args.shared_scale:
        for g in genes:
            vals = [cc.expr_of(sub_for(a), g) for a in samples.values() if g in a.var_names]
            if vals:
                allv = np.concatenate(vals)
                vlim[g] = (float(allv.min()), float(allv.max()))

    panels = ([HE] if not args.no_he else []) + genes
    ncols = max(1, args.ncols)
    nrows = ceil(len(panels) / ncols)

    items = list(samples.items())
    spp = max(1, args.samples_per_pdf)
    chunks = [items[i:i + spp] for i in range(0, len(items), spp)]

    for ci, chunk in enumerate(chunks, 1):
        pdf_path = os.path.join(args.outdir, f"{args.name}_{rng}_part{ci:02d}.pdf")
        with PdfPages(pdf_path) as pdf:
            for sample, adata in chunk:
                center = adata.obsm["spatial"].mean(axis=0)        # full-tissue centroid
                deg = rot.get(sample, args.rotate)
                folder = str(adata.obs["folder"].iloc[0])
                chip_dir = os.path.join(args.data_dir, folder)
                segments, _ = gw.load_boundary(args.boundary_dir, sample)
                sub = sub_for(adata)
                xy = sub.obsm["spatial"]
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.2),
                                         squeeze=False)
                for ax, panel in zip(axes.ravel(), panels):
                    if panel == HE:
                        draw_he(ax, chip_dir, deg)
                        continue
                    if panel not in sub.var_names:
                        ax.set_title(panel, style="italic")
                        ax.text(0.5, 0.5, "absent", ha="center", va="center", transform=ax.transAxes)
                        ax.set_xticks([]); ax.set_yticks([]); continue
                    vmin, vmax = vlim.get(panel, (None, None))
                    cc.draw_map(ax, xy, center, deg, segments, values=cc.expr_of(sub, panel),
                                cmap=args.cmap, vmin=vmin, vmax=vmax, point_size=args.point_size,
                                boundary_color=args.boundary_color, boundary_width=args.boundary_width)
                    ax.set_title(panel, style="italic")
                for ax in axes.ravel()[len(panels):]:
                    ax.axis("off")
                fig.suptitle(f"{sample}  (log1p expr; range={rng}; rotate={deg:g} deg)")
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                pdf.savefig(fig, dpi=200)
                if args.png:
                    fig.savefig(os.path.join(args.outdir, f"{args.name}_{rng}_{sample}.png"),
                                dpi=200, bbox_inches="tight")
                plt.close(fig)
        log.info("wrote %s (%d sample(s))", pdf_path, len(chunk))


if __name__ == "__main__":
    main()
