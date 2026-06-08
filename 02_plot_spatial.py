#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_plot_spatial.py -- spatial maps with grey/white boundary
============================================================

Reads plot_bundle.h5ad and the boundaries/ folder produced by 01_compute.py and
draws, per sample, every panel (region + module scores + single genes) on both
backgrounds (coloured spot scatter and H&E hires image), with the dashed
grey/white boundary overlaid on every panel.

Boundary precedence per sample: boundaries/manual_<sample>.tsv (your edit) if it
exists, otherwise boundaries/auto_<sample>.tsv. So: run once, inspect, adjust the
manual file (or use --draw), then RE-RUN this script -- no recomputation.

    # batch (headless)
    python 02_plot_spatial.py --bundle results_grey_white/plot_bundle.h5ad \
        --data-dir matrix --outdir results_grey_white

    # interactively (re)draw one boundary (needs a display)
    python 02_plot_spatial.py --draw CGND-HRA-02230 \
        --bundle results_grey_white/plot_bundle.h5ad --data-dir matrix \
        --outdir results_grey_white
"""

import argparse
import os
import sys

import anndata as ad

import gw_common as gw

log = gw.log


def configure_from_bundle(adata, gene_list, single_sets):
    if gene_list:
        panel = gw.parse_gene_list(gene_list)
        single = (single_sets or gw.DEFAULT_SINGLE_SETS).split(",")
    else:
        modules = {k: list(v) for k, v in adata.uns.get("modules", {}).items()}
        singles = {k: list(v) for k, v in adata.uns.get("single_groups", {}).items()}
        panel = [(k, v) for k, v in {**modules, **singles}.items()]
        single = list(singles) if single_sets is None else single_sets.split(",")
        if not panel:
            log.error("bundle has no panel info; pass --gene-list."); sys.exit(1)
    gw.configure_panel(panel, single)


def sample_dirs_from(adata, data_dir):
    """sample -> chip_dir, using the 'folder' column stored in the bundle."""
    if "folder" not in adata.obs:
        return {}
    pairs = adata.obs[["sample", "folder"]].drop_duplicates()
    return {r["sample"]: os.path.join(data_dir, r["folder"]) for _, r in pairs.iterrows()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="results_grey_white/plot_bundle.h5ad")
    ap.add_argument("--data-dir", default="matrix", help="for H&E images (chip folders)")
    ap.add_argument("--outdir", default="results_grey_white")
    ap.add_argument("--gene-list", default=None, help="override panel (default: from bundle)")
    ap.add_argument("--single-gene-sets", default=None)
    ap.add_argument("--no-single", action="store_true",
                    help="omit the single-gene panels (region + modules only)")
    ap.add_argument("--show-none", action="store_true",
                    help="also plot spots not assigned grey/white (default: drop them)")
    ap.add_argument("--backgrounds", default="scatter,he",
                    help="comma list: scatter, he")
    ap.add_argument("--ncol", type=int, default=4)
    ap.add_argument("--point-size", type=float, default=6.0)
    ap.add_argument("--boundary-color", default="#FFD700",
                    help="boundary line colour, same on both backgrounds (default gold)")
    ap.add_argument("--boundary-width", type=float, default=1.8)
    ap.add_argument("--draw", metavar="SAMPLE", default=None,
                    help="interactively (re)draw one sample's boundary, then exit")
    ap.add_argument("--no-guide", action="store_true",
                    help="with --draw: hide the cyan reference line (clean canvas)")
    ap.add_argument("--clear", metavar="SAMPLE", default=None,
                    help="delete manual_<SAMPLE>.tsv (revert to auto), re-render, exit")
    args = ap.parse_args()

    gw.setup_logging(args.outdir, filename="plot_spatial.log")
    if not os.path.isfile(args.bundle):
        log.error("bundle not found: %s (run 01_compute.py first)", args.bundle); sys.exit(1)
    adata = ad.read_h5ad(args.bundle)
    log.info("loaded %s (%d spots x %d genes)", args.bundle, adata.n_obs, adata.n_vars)
    configure_from_bundle(adata, args.gene_list, args.single_gene_sets)

    bdir = os.path.join(args.outdir, "boundaries")
    os.makedirs(bdir, exist_ok=True)
    sample_dirs = sample_dirs_from(adata, args.data_dir)
    backgrounds = tuple(b.strip() for b in args.backgrounds.split(",") if b.strip())

    if args.clear:
        mp = os.path.join(bdir, f"manual_{args.clear}.tsv")
        if os.path.isfile(mp):
            os.remove(mp); log.info("removed %s (reverted to auto)", mp)
        else:
            log.info("no manual boundary for %s -- nothing to clear", args.clear)
        gw.plot_spatial_all(adata, bdir, sample_dirs, args.outdir,
                            include_single=not args.no_single, backgrounds=backgrounds,
                            ncol=args.ncol, point_size=args.point_size, samples=[args.clear],
                            boundary_color=args.boundary_color, boundary_width=args.boundary_width,
                            show_none=args.show_none)
        return

    if args.draw:
        if args.draw not in set(adata.obs["sample"].unique().tolist()):
            log.error("sample %r not in bundle. Available: %s",
                      args.draw, sorted(adata.obs["sample"].unique().tolist())); sys.exit(1)
        import matplotlib.pyplot as plt
        backend_ok = False
        for be in ("QtAgg", "Qt5Agg", "TkAgg", "GTK3Agg"):
            try:
                plt.switch_backend(be); backend_ok = True
                log.info("interactive backend: %s", be); break
            except Exception:
                continue
        if not backend_ok:
            log.error("no interactive backend available (need PyQt5 or python3-tk). "
                      "Install one, run with X display, or edit "
                      "boundaries/manual_%s.tsv by hand.", args.draw); sys.exit(1)
        on_he = "he" in backgrounds
        saved = gw.draw_interactive(adata, args.draw, sample_dirs.get(args.draw, ""),
                                    bdir, on_he=on_he, show_guide=not args.no_guide)
        if saved:
            plt.switch_backend("Agg")          # re-render that sample's figures
            gw.plot_spatial_all(adata, bdir, sample_dirs, args.outdir,
                                include_single=not args.no_single, backgrounds=backgrounds,
                                ncol=args.ncol, point_size=args.point_size, samples=[args.draw],
                                boundary_color=args.boundary_color,
                                boundary_width=args.boundary_width, show_none=args.show_none)
            log.info("updated figures for %s in %s/spatial_maps", args.draw, args.outdir)
        return

    gw.plot_spatial_all(
        adata, bdir, sample_dirs, args.outdir,
        include_single=not args.no_single,
        backgrounds=tuple(b.strip() for b in args.backgrounds.split(",") if b.strip()),
        ncol=args.ncol, point_size=args.point_size,
        boundary_color=args.boundary_color, boundary_width=args.boundary_width,
        show_none=args.show_none)
    log.info("spatial plotting done -> %s/spatial_maps", args.outdir)


if __name__ == "__main__":
    main()
