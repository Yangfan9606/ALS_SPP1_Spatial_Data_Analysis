#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fig_single_gene_box.py -- standalone per-gene boxplots (grey vs white)
===========================================================================

Self-contained: pandas + numpy + matplotlib + seaborn only (no gw_common).

Draws a grey-vs-white boxplot for EVERY panel gene (all module-member genes
AND the single-gene set), 6 genes per page, uniform subplot size, into one
multi-page PDF (plus one PNG per page for quick viewing).

Test == display: significance is recomputed here on EXACTLY the spots shown.
By default only expressing spots (expr > 0) are boxed and tested (Visium is
sparse; use --box-include-zeros to keep zeros). Per gene: two-sided Mann-Whitney
U (Wilcoxon rank-sum, grey vs white spots), then Benjamini-Hochberg FDR across
the genes drawn -> padj shown as stars + value. Detection rate is annotated
'xx%+'; a faint jitter sample and the per-chip mean line are overlaid.

Source data
  --sd-dir/sd_gene_expr_long.tsv.gz   sample, region, gene_set, gene, expr (log1p)
(writes <name>_stats.tsv with per-gene pval/padj.)

    python plot_fig_single_gene_box.py --sd-dir results_gw --outdir results_gw/figures_custom
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import mannwhitneyu

REGION_COLORS = {"grey": "#7f7f7f", "white": "#d62728"}


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "ns"
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


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


def gene_pvalue(d_disp, region_order):
    """Two-sided Mann-Whitney U on the displayed spots (grey vs white)."""
    g = d_disp.loc[d_disp.region == region_order[0], "expr"].to_numpy(float)
    w = d_disp.loc[d_disp.region == region_order[1], "expr"].to_numpy(float)
    if g.size < 3 or w.size < 3 or np.ptp(np.concatenate([g, w])) == 0:
        return np.nan
    try:
        _, p = mannwhitneyu(g, w, alternative="two-sided")
    except ValueError:
        return np.nan
    return float(p)


def draw_gene(ax, d_all, region_order, padj, drop_zeros, max_points):
    d = d_all[d_all.expr > 0] if drop_zeros else d_all
    gene = d_all["gene"].iloc[0]
    gene_set = d_all["gene_set"].iloc[0] if "gene_set" in d_all else ""
    if d.empty:
        ax.set_title(gene, style="italic"); ax.text(0.5, 0.5, "no expressing spots",
                                                     ha="center", va="center", transform=ax.transAxes)
        return
    sns.boxplot(data=d, x="region", y="expr", order=region_order, hue="region",
                hue_order=region_order, palette=REGION_COLORS, legend=False,
                showfliers=False, width=0.3, ax=ax)
    samp = d.sample(min(max_points, len(d)), random_state=0) if len(d) > max_points else d
    sns.stripplot(data=samp, x="region", y="expr", order=region_order, ax=ax,
                  size=1.5, color="0.25", alpha=0.25, jitter=0.25)
    cm = d.groupby(["sample", "region"])["expr"].mean().reset_index()
    for s_ in cm["sample"].unique():
        sub = cm[cm["sample"] == s_].set_index("region").reindex(region_order)
        ax.plot(range(len(region_order)), sub["expr"].values, "-o", color="black",
                lw=0.6, ms=3, alpha=0.6, zorder=5)
    ymax = float(d["expr"].max()) or 1.0
    ax.set_ylim(0, ymax * 1.22)
    for i, region in enumerate(region_order):
        pct = (d_all.loc[d_all.region == region, "expr"] > 0).mean() * 100
        plabel = f"{pct:.2f}%+" if 0 < pct < 1 else f"{pct:.0f}%+"
        ax.text(i, ymax * 1.04, plabel, ha="center", va="bottom", fontsize=7, color="0.3")
    pj = padj.get(gene, np.nan)
    if not np.isnan(pj):
        ptxt = f"padj={pj:.1e}" if pj < 1e-3 else f"padj={pj:.2f}"
        ax.text((len(region_order) - 1) / 2, ymax * 1.13, f"{stars(pj)}  {ptxt}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(gene, style="italic"); ax.set_xlabel(""); ax.set_ylabel("log1p expr")
#    if gene_set:
#        ax.text(0.02, 0.98, gene_set, transform=ax.transAxes, ha="left", va="top",
#                fontsize=6.5, color="0.45")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sd-dir", default="results_gw")
    ap.add_argument("--outdir", default="results_gw/figures_custom")
    ap.add_argument("--region-order", default="grey,white")
    ap.add_argument("--per-page", type=int, default=6)
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--box-include-zeros", action="store_true")
    ap.add_argument("--max-points", type=int, default=1500)
    ap.add_argument("--name", default="fig_gene_box_all")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    expr_path = os.path.join(args.sd_dir, "sd_gene_expr_long.tsv.gz")
    if not os.path.isfile(expr_path):
        sys.exit(f"missing {expr_path} (run 01_compute.py first)")
    expr = pd.read_csv(expr_path, sep="\t")

    region_order = [r.strip() for r in args.region_order.split(",")
                    if r.strip() in ("grey", "white")]
    genes = list(dict.fromkeys(expr["gene"]))          # panel order from the file
    drop_zeros = not args.box_include_zeros

    # significance: test on EXACTLY the displayed spots, then BH-FDR across genes
    pvals = []
    for g in genes:
        d_all = expr[expr.gene == g]
        d_disp = d_all[d_all.expr > 0] if drop_zeros else d_all
        pvals.append(gene_pvalue(d_disp, region_order))
    padj_arr = bh_fdr(pvals)
    padj = {g: padj_arr[i] for i, g in enumerate(genes)}
    pd.DataFrame({"gene": genes, "pval": pvals, "padj": [padj[g] for g in genes],
                  "test": "Mann-Whitney U two-sided",
                  "spots": "expr>0" if drop_zeros else "all"}).to_csv(
        os.path.join(args.outdir, f"{args.name}_stats.tsv"), sep="\t", index=False)

    per_page, ncols = args.per_page, args.ncols
    nrows = int(np.ceil(per_page / ncols))             # fixed grid -> uniform subplot size
    pages = [genes[i:i + per_page] for i in range(0, len(genes), per_page)]
    note = "expressing spots only, %+=detection" if drop_zeros else "all spots"

    pdf_path = os.path.join(args.outdir, f"{args.name}.pdf")
    with PdfPages(pdf_path) as pdf:
        for pi, page_genes in enumerate(pages, 1):
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 3.6),
                                     squeeze=False)
            for ax, gene in zip(axes.ravel(), page_genes):
                draw_gene(ax, expr[expr.gene == gene], region_order, padj,
                          drop_zeros, args.max_points)
            for ax in axes.ravel()[len(page_genes):]:
                ax.axis("off")
            fig.suptitle(f"Gene expression grey vs white ({note}; line=per-chip mean; "
                         f"padj=BH-FDR of Mann-Whitney U) -- page {pi}/{len(pages)}")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, dpi=200)
            fig.savefig(os.path.join(args.outdir, f"{args.name}_p{pi:02d}.png"),
                        dpi=200, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {pdf_path} ({len(genes)} genes, {len(pages)} page(s)) + per-page PNGs + "
          f"{args.name}_stats.tsv")


if __name__ == "__main__":
    main()
