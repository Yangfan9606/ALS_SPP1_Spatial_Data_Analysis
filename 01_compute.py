#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_compute.py -- build all source data for the grey/white analysis (no plotting)
=================================================================================

Per chip: load matrix (all tissue spots), assign grey/white/none, log1p-CPM
normalise, module scores, all-positive flags, auto grey/white boundary.
Pooled: grey-vs-white statistics. Everything is written to --outdir as
re-loadable source data (see README_results.md). Figures are produced
separately by 02_plot_spatial.py and 03_plot_summary.py.

    python 01_compute.py --data-dir matrix --info spatial_info.txt \
        --gene-list gene.list.txt --outdir results_grey_white
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad

import gw_common as gw

log = gw.log


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="matrix")
    ap.add_argument("--info", default="spatial_info.txt")
    ap.add_argument("--outdir", default="results_grey_white")
    ap.add_argument("--gene-list", default=None)
    ap.add_argument("--single-gene-sets", default=gw.DEFAULT_SINGLE_SETS)
    ap.add_argument("--name-modules-by-genes", action="store_true",
                    help="rename each co-expression module to 'gene1+gene2+...' "
                         "(single-gene sets unchanged)")
    ap.add_argument("--score-per-chip", action="store_true",
                    help="compute module scores within each chip (own background); "
                         "suits per-sample spatial maps")
    ap.add_argument("--score-on-grey-white", action="store_true",
                    help="use only grey/white spots as scoring background (exclude none); "
                         "with --score-per-chip this matches the old monolithic script")
    # boundary params
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--grid-res", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--min-seg-frac", type=float, default=0.05)
    ap.add_argument("--no-boundary", action="store_true", help="skip boundary computation")
    args = ap.parse_args()

    gw.setup_logging(args.outdir)
    log.info("data-dir=%s | info=%s | outdir=%s", args.data_dir, args.info, args.outdir)
    bdir = os.path.join(args.outdir, "boundaries")
    os.makedirs(bdir, exist_ok=True)

    if args.gene_list:
        panel = gw.parse_gene_list(args.gene_list)
        if not panel:
            log.error("no usable gene sets in %s", args.gene_list); sys.exit(1)
        log.info("loaded %d gene set(s) from %s", len(panel), args.gene_list)
    else:
        panel = gw.DEFAULT_PANEL
        log.info("using built-in default panel")
    gw.configure_panel(panel, args.single_gene_sets.split(","))
    if args.name_modules_by_genes:
        gw.rename_modules_by_genes()

    info = pd.read_csv(args.info, sep="\t", dtype=str)
    info.columns = [c.strip() for c in info.columns]
    log.info("spatial_info has %d chips", len(info))

    adatas = []
    for _, row in info.iterrows():
        folder = row["Folder"].strip()
        sample = row.get("ExternalSampleId") or folder
        sample = sample.strip() if isinstance(sample, str) and sample.strip() else folder
        log.info("==== %s (folder=%s) ====", sample, folder)
        adata = gw.load_chip(args.data_dir, folder, sample,
                             row["Number of cluster"], row["Grey"], row["White"])
        if adata is None:
            continue

        # auto boundary from all tissue spots (grey/white/none)
        if not args.no_boundary and "spatial" in adata.obsm:
            df_xy = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"])
            df_xy["region"] = adata.obs["region"].values
            segs = gw.auto_boundary(df_xy, knn=args.knn, grid_res=args.grid_res,
                                    sigma=args.sigma, min_seg_frac=args.min_seg_frac)
            gw.save_segments(os.path.join(bdir, f"auto_{sample}.tsv"), segs)
        adatas.append(adata)

    if not adatas:
        log.error("No chip processed. Check --data-dir contains the 'Folder' dirs."); sys.exit(1)
    log.info("processed %d / %d chips", len(adatas), len(info))

    combined = ad.concat(adatas, join="inner", label="batch",
                         keys=[a.obs["sample"].iloc[0] for a in adatas], index_unique="_")
    n = {r: int((combined.obs.region == r).sum()) for r in ("grey", "white", "none")}
    log.info("pooled: %d spots (grey=%d white=%d none=%d) x %d genes",
             combined.n_obs, n["grey"], n["white"], n["none"], combined.n_vars)

    gw.normalize(combined)                         # log1p CPM (per-spot)
    gw.compute_module_scores(combined, per_chip=args.score_per_chip,
                             on_grey_white=args.score_on_grey_white)

    # ---- statistics ------------------------------------------------------- #
    stats_pooled = gw.build_stats(combined, "pooled")
    per_chip = []
    for s in combined.obs["sample"].unique().tolist():
        st = gw.build_stats(combined[combined.obs["sample"] == s], s)
        st.insert(0, "sample", s)
        per_chip.append(st)
    stats_per_chip = pd.concat(per_chip, ignore_index=True) if per_chip else pd.DataFrame()

    allpos_pooled = gw.allpos_fraction_table(combined, "pooled")
    allpos_perchip = []
    for s in combined.obs["sample"].unique().tolist():
        t = gw.allpos_fraction_table(combined[combined.obs["sample"] == s], s)
        allpos_perchip.append(t)
    allpos_perchip = pd.concat(allpos_perchip, ignore_index=True) if allpos_perchip else pd.DataFrame()

    # ---- source-data tables ---------------------------------------------- #
    def out(name): return os.path.join(args.outdir, name)
    gw.sd_module_scores_long(combined).to_csv(out("sd_module_scores_long.tsv.gz"), sep="\t", index=False)
    gw.sd_gene_expr_long(combined).to_csv(out("sd_gene_expr_long.tsv.gz"), sep="\t", index=False)
    gw.sd_dotplot(combined).to_csv(out("sd_dotplot.tsv"), sep="\t", index=False)
    stats_pooled.to_csv(out("sd_stats_pooled.tsv"), sep="\t", index=False)
    stats_per_chip.to_csv(out("sd_stats_per_chip.tsv"), sep="\t", index=False)
    allpos_pooled.to_csv(out("sd_allpos_fraction.tsv"), sep="\t", index=False)
    allpos_perchip.to_csv(out("sd_allpos_fraction_per_chip.tsv"), sep="\t", index=False)
    log.info("source-data tables written")

    # ---- plot bundle (all tissue spots, panel genes only) ---------------- #
    genes = gw.panel_genes_present(combined.var_names)
    slim = combined[:, genes].copy()
    keep = [c for c in slim.obs.columns
            if c in ("sample", "folder", "region", "cluster_label", "clustering_used")
            or c.startswith("score_") or c.startswith("allpos_")]
    slim.obs = slim.obs[keep].copy()
    slim.uns["modules"] = {k: list(v) for k, v in gw.MODULES.items()}
    slim.uns["single_groups"] = {k: list(v) for k, v in gw.SINGLE_GROUPS.items()}
    slim.write_h5ad(out("plot_bundle.h5ad"))
    log.info("wrote plot_bundle.h5ad (%d spots x %d panel genes)", slim.n_obs, slim.n_vars)

    # ---- README ----------------------------------------------------------- #
    gw.write_readme(args.outdir, {
        "data_dir": args.data_dir, "info": args.info,
        "n_chips_total": len(info), "n_chips_used": len(adatas),
        "n_spots": combined.n_obs, "n_grey": n["grey"], "n_white": n["white"], "n_none": n["none"],
        "target_sum": gw.TARGET_SUM, "knn": args.knn, "grid_res": args.grid_res,
        "sigma": args.sigma, "min_seg_frac": args.min_seg_frac,
        "score_mode": ("per-chip" if args.score_per_chip else "pooled")
                      + (", grey/white background" if args.score_on_grey_white else ", all-spot background"),
        "modules": gw.MODULES, "single_groups": gw.SINGLE_GROUPS})
    log.info("compute done.")


if __name__ == "__main__":
    main()
