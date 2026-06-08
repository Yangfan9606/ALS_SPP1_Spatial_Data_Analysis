#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fig_paired.py -- standalone per-chip paired module plot (grey vs white)
============================================================================

Self-contained: pandas + matplotlib + scipy only (does NOT import gw_common).

Each line = one chip, joining that chip's mean grey-spot module score to its
mean white-spot module score. Significance is a PAIRED test ACROSS CHIPS:
for each module the per-chip (mean_grey, mean_white) pairs are compared with a
Wilcoxon signed-rank test (default) or paired t-test. n = number of chips.
This asks "across samples, is the module score consistently higher in grey vs
white?" -- different from the spot-level p-value in sd_stats_per_chip.tsv.

Source data
  --sd-dir/sd_stats_per_chip.tsv   rows with feature_kind=="module_score";
                                   columns: sample, feature, mean_grey, mean_white

    python plot_fig_paired.py --sd-dir results_gw --outdir results_gw/figures_custom
    python plot_fig_paired.py --sd-dir results_gw --test ttest

    # put two pairs of modules each on a shared y-axis (names OR 1-based numbers):
    python plot_fig_paired.py --sd-dir results_gw \
        --share-y "NEFM+SNAP25,MBP+PLP1; TREM2+APOE+TYROBP,SPP1+CD44"
    python plot_fig_paired.py --sd-dir results_gw --share-y "1,2; 4,5" --wspace 0.6
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel

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


def paired_test(grey, white, kind):
    grey, white = np.asarray(grey, float), np.asarray(white, float)
    ok = ~(np.isnan(grey) | np.isnan(white))
    grey, white = grey[ok], white[ok]
    n = grey.size
    if n < 2 or np.allclose(grey, white):
        return np.nan, n
    try:
        if kind == "ttest":
            _, p = ttest_rel(grey, white)
        else:
            _, p = wilcoxon(grey, white)          # paired, two-sided
    except ValueError:
        p = np.nan
    return float(p), n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sd-dir", default="results_gw")
    ap.add_argument("--outdir", default="results_gw/figures_custom")
    ap.add_argument("--region-order", default="grey,white")
    ap.add_argument("--test", choices=["wilcoxon", "ttest"], default="wilcoxon")
    ap.add_argument("--no-fdr", action="store_true",
                    help="show raw p instead of BH-FDR padj across modules")
    ap.add_argument("--name", default="fig_module_paired_perchip")
    ap.add_argument("--share-y", default=None,
                    help="modules that should share one y-axis for comparison. "
                         "Use module (feature) names or 1-based panel positions, "
                         "comma-separated; separate independent groups with ';'. "
                         "'all' shares a single y-axis across every panel. "
                         "Example: --share-y 'NEFM+SNAP25,MBP+PLP1; TREM2+APOE+TYROBP,SPP1+CD44'")
    ap.add_argument("--wspace", type=float, default=0.5,
                    help="horizontal spacing between panels (raise to stop y-axis "
                         "labels being covered; default 0.5)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.sd_dir, "sd_stats_per_chip.tsv")
    if not os.path.isfile(path):
        sys.exit(f"missing {path} (run 01_compute.py first)")
    df = pd.read_csv(path, sep="\t")
    df = df[df.feature_kind == "module_score"]
    if df.empty:
        sys.exit("no module_score rows in sd_stats_per_chip.tsv")

    order = [r.strip() for r in args.region_order.split(",") if r.strip() in ("grey", "white")]
    mods = list(dict.fromkeys(df.feature))
    print("modules (panel order): " + ", ".join(f"{i+1}:{m}" for i, m in enumerate(mods)))

    # per-module paired test, then BH-FDR across the modules
    pvals, ns = [], []
    for m in mods:
        sub = df[df.feature == m]
        p, n = paired_test(sub["mean_grey"], sub["mean_white"], args.test)
        pvals.append(p); ns.append(n)
    padj = bh_fdr(pvals)
    shown = pvals if args.no_fdr else padj
    label = "p" if args.no_fdr else "padj"
    test_name = "paired t-test" if args.test == "ttest" else "Wilcoxon signed-rank (paired)"
    pd.DataFrame({"module": mods, "n_chips": ns, "pval": pvals, "padj": padj,
                  "test": test_name}).to_csv(
        os.path.join(args.outdir, f"{args.name}_stats.tsv"), sep="\t", index=False)

    # ---- y-axis sharing groups (compare chosen modules on a common scale) -- #
    def resolve_token(tok):
        tok = tok.strip()
        if not tok:
            return None
        if tok.isdigit():                        # 1-based position in panel order
            i = int(tok) - 1
            return mods[i] if 0 <= i < len(mods) else None
        return tok if tok in mods else None      # else an exact module (feature) name

    share_groups = []
    if args.share_y:
        if args.share_y.strip().lower() == "all":
            share_groups = [list(mods)]
        else:
            for grp in args.share_y.split(";"):
                members = [m for m in (resolve_token(t) for t in grp.split(",")) if m]
                if members:
                    share_groups.append(members)
    if share_groups:
        print("shared y-axis group(s): " + " | ".join(", ".join(g) for g in share_groups))

    # per-module data and its own (data) range, then the axis range to use
    data, mlim = {}, {}
    for m in mods:
        sub = df[df.feature == m]
        g = sub["mean_grey"].to_numpy(float)
        w = sub["mean_white"].to_numpy(float)
        data[m] = (g, w)
        mlim[m] = (np.nanmin([g.min(), w.min()]), np.nanmax([g.max(), w.max()]))
    axis_rng = dict(mlim)                         # module -> (lo, hi) for the y-axis
    for grp in share_groups:                      # widen members to the group extent
        lo = min(mlim[m][0] for m in grp); hi = max(mlim[m][1] for m in grp)
        for m in grp:
            axis_rng[m] = (lo, hi)

    fig, axes = plt.subplots(1, len(mods), figsize=(3.2 * len(mods), 4.8), squeeze=False)
    for ax, m, val, n in zip(axes[0], mods, shown, ns):
        g, w = data[m]
        means = {"grey": g, "white": w}
        xpos = {r: i for i, r in enumerate(order)}
        # per-chip lines + points
        for gv, wv in zip(g, w):
            vals = {"grey": gv, "white": wv}
            ax.plot([xpos[r] for r in order], [vals[r] for r in order],
                    color="0.6", lw=0.9, zorder=1)
        for r in order:
            ax.plot([xpos[r]] * len(means[r]), means[r], "o",
                    color=REGION_COLORS[r], ms=5, zorder=2)

        alo, ahi = axis_rng[m]                     # possibly shared across a group
        rng = (ahi - alo) or 1.0
        ax.set_ylim(alo - 0.08 * rng, ahi + 0.22 * rng)
        # annotation sits above THIS module's own points, offset by the axis range
        ytop = np.nanmax([g.max(), w.max()])
        vtxt = f"{label}=NA" if np.isnan(val) else \
            (f"{label}={val:.1e}" if val < 1e-3 else f"{label}={val:.3f}")
        ax.text(0.5, ytop + 0.10 * rng, f"{stars(val)}\n{vtxt} (n={n})",
                ha="center", va="bottom", fontsize=9)
        # bracket
        ax.plot([0, 0, 1, 1], [ytop + 0.05 * rng, ytop + 0.08 * rng,
                               ytop + 0.08 * rng, ytop + 0.05 * rng], color="k", lw=0.8)

        ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
        ax.set_xlim(-0.4, len(order) - 0.6)
        ax.set_title(m); ax.set_ylabel("mean module score")
    corr = "" if args.no_fdr else "; BH-FDR across modules"
    fig.suptitle(f"Per-chip grey vs white (each line = one chip; {test_name} across chips{corr})")
    fig.subplots_adjust(wspace=args.wspace)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"{args.name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(args.outdir, args.name)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
