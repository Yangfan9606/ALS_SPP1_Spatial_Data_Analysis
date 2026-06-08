#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_dotplot_vertical.py -- standalone vertical dot plot (genes as rows)
========================================================================

Self-contained: does NOT import gw_common. It uses scanpy ONLY to compute the
dot values (fraction expressing + standard-scaled mean expression), then draws
the figure manually with matplotlib for full control over:
  * dot sizes that actually fill the cells (paper-style)
  * square-ish cells (panel width tied to #groups, not stretched)
  * significance stars OUTSIDE the frame (right of each gene row)
  * size legend + colour bar stacked on the RIGHT, never overlapping
  * no title, no gene-group brackets

Colours match the original scanpy dotplot (default cmap = 'Reds').

Inputs
  --bundle    plot_bundle.h5ad        (expression + region)
  --sd-dir/sd_dotplot.tsv             gene order (panel order)
  --sd-dir/sd_stats_pooled.tsv        significance (feature, feature_kind, sig)

    python plot_dotplot_vertical.py --bundle results_gw/plot_bundle.h5ad \
        --sd-dir results_gw --outdir results_gw/figures_custom
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import scanpy as sc
import anndata as ad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bundle", default="results_gw/plot_bundle.h5ad")
    ap.add_argument("--sd-dir", default="results_gw")
    ap.add_argument("--outdir", default="results_gw/figures_custom")
    ap.add_argument("--region-order", default="grey,white")
    ap.add_argument("--cmap", default="Reds",
                    help="colour map (default: Reds, matches scanpy dotplot)")
    ap.add_argument("--no-italic", action="store_true")
    ap.add_argument("--name", default="fig_dotplot_vertical")
    # tunables for dot rendering
    ap.add_argument("--cell-in", type=float, default=0.46,
                    help="cell size in inches (controls dot spacing & size)")
    ap.add_argument("--dot-max-frac", type=float, default=0.80,
                    help="largest dot diameter as fraction of a cell")
    ap.add_argument("--dot-min-frac", type=float, default=0.10,
                    help="smallest dot diameter as fraction of a cell")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if not os.path.isfile(args.bundle):
        sys.exit(f"missing {args.bundle} (run 01_compute.py first)")
    adata = ad.read_h5ad(args.bundle)

    # ── subset to grey / white in requested order ───────────────────────────
    adata = adata[np.isin(adata.obs["region"], ["grey", "white"])].copy()
    order = [r.strip() for r in args.region_order.split(",") if r.strip()]
    present = [r for r in order if r in set(adata.obs["region"].astype(str))]
    adata.obs["region"] = pd.Categorical(
        adata.obs["region"].astype(str), categories=present, ordered=True
    )

    # ── gene order ───────────────────────────────────────────────────────────
    dot_path = os.path.join(args.sd_dir, "sd_dotplot.tsv")
    if os.path.isfile(dot_path):
        genes = list(dict.fromkeys(pd.read_csv(dot_path, sep="\t")["gene"]))
        genes = [g for g in genes if g in adata.var_names]
    else:
        genes = list(adata.var_names)
    if not genes:
        sys.exit("no panel genes found in bundle")

    # ── significance per gene ────────────────────────────────────────────────
    sig = {}
    stats_path = os.path.join(args.sd_dir, "sd_stats_pooled.tsv")
    if os.path.isfile(stats_path):
        st = pd.read_csv(stats_path, sep="\t")
        st = st[st.feature_kind.isin(["single_gene", "module_member_gene"])]
        sig = dict(zip(st.feature, st.sig))

    # ── let scanpy COMPUTE the dot values (then we draw ourselves) ───────────
    dp = sc.pl.dotplot(
        adata, genes, groupby="region", standard_scale="var",
        swap_axes=True, return_fig=True, show=False,
    )
    frac = dp.dot_size_df    # fraction of cells expressing (0..1)
    color = dp.dot_color_df  # standard-scaled mean expression (0..1)

    # normalise orientation -> index = groups, columns = genes
    if not set(genes).issubset(set(frac.columns)):
        frac, color = frac.T, color.T
    frac = frac.loc[present, genes]
    color = color.loc[present, genes]

    n_genes = len(genes)
    n_groups = len(present)

    # ── geometry (inches) ────────────────────────────────────────────────────
    cell = args.cell_in
    left_margin = 1.35              # room for italic gene labels
    panel_w = n_groups * cell
    panel_h = n_genes * cell
    star_gap = 0.55                 # column for significance stars
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ 调图例总宽度就改这里(英寸)。垂直图例比横版窄,1.1~1.4 比较合适。   │
    # │ 数字越大 → 右侧给图例留的空间越宽。改这里会同时正确扩大画布。       │
    # └─────────────────────────────────────────────────────────────────────┘
    legend_w = 1.45                 # <<< 图例总宽度 (inches)
    top_margin = 0.30
    bottom_margin = 0.75            # room for x labels

    fig_w = left_margin + panel_w + star_gap + legend_w
    fig_h = top_margin + panel_h + bottom_margin

    fig = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_axes([
        left_margin / fig_w,
        bottom_margin / fig_h,
        panel_w / fig_w,
        panel_h / fig_h,
    ])

    # ── dot size mapping: diameter (in points) -> area for scatter `s` ───────
    pts_per_inch = 72.0
    d_max = args.dot_max_frac * cell * pts_per_inch   # max diameter in points
    d_min = args.dot_min_frac * cell * pts_per_inch   # min diameter in points

    cmap = plt.get_cmap(args.cmap)
    norm = Normalize(vmin=0.0, vmax=1.0)

    # build coordinate / value arrays (gene row 0 at TOP)
    xs, ys, sizes, colors = [], [], [], []
    for gi, g in enumerate(genes):
        y = n_genes - 1 - gi      # invert so first gene is on top
        for xi, grp in enumerate(present):
            f = float(frac.loc[grp, g])
            c = float(color.loc[grp, g])
            d = d_min + f * (d_max - d_min)   # diameter in points
            xs.append(xi)
            ys.append(y)
            sizes.append((d / 2.0) ** 2 * np.pi)   # area in points^2
            colors.append(cmap(norm(c)))

    ax.scatter(xs, ys, s=sizes, c=colors,
               edgecolors="black", linewidths=0.45, zorder=3)

    # ── axis cosmetics ───────────────────────────────────────────────────────
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(-0.5, n_genes - 0.5)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(present, rotation=0, ha="center", fontsize=9)
    ax.set_yticks([n_genes - 1 - i for i in range(n_genes)])
    ax.set_yticklabels(
        genes,
        fontsize=9,
        fontstyle=("normal" if args.no_italic else "italic"),
    )
    ax.tick_params(axis="both", length=3, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_aspect("equal")          # keep cells square

    # ── significance stars just right of the frame ──────────────────────────
    for gi, g in enumerate(genes):
        s = sig.get(g)
        if s and s != "ns":
            y = n_genes - 1 - gi
            ax.annotate(
                s,
                xy=(1.0, y), xycoords=("axes fraction", "data"),
                xytext=(7, 0), textcoords="offset points",
                va="center", ha="left",
                fontsize=8.5, fontweight="bold", color="#222222",
                clip_on=False,
            )

    # ══════════════════════════════════════════════════════════════════════
    # ║                    右侧图例(垂直排版) — 可调区                     ║
    # ══════════════════════════════════════════════════════════════════════
    # 图例整体的左右位置:legend_x0 越大越往右。
    legend_x0 = (left_margin + panel_w + star_gap) / fig_w
    legend_x0 += 0.015               # <<< 图例左右微调 (越大越往右)

    # ── (1) 点大小图例:圆点【垂直堆叠】(小点在上,大点在下) ──────────────
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ size_ax 的四个值 = [左, 下, 宽, 高](都是 0~1 的画布比例)。          │
    # │   · 第2个值 (0.50) 调上下位置:越大越靠上。                          │
    # │   · 第4个值 (0.42) 调这块图例的高度:越大圆点间距越松。              │
    # └─────────────────────────────────────────────────────────────────────┘
    size_ax = fig.add_axes([
        legend_x0,
        (bottom_margin + panel_h * 0.50) / fig_h,   # <<< 大小图例 上下位置
        legend_w / fig_w,
        (panel_h * 0.42) / fig_h,                    # <<< 大小图例 高度
    ])
    size_ax.set_axis_off()
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)

    # 标题(放最上方)。改文字 / 字号就在这一行。
    size_ax.text(0.0, 1.02, "Fraction of cells\nin group (%)",
                 ha="left", va="bottom", fontsize=12,   # <<< 标题字号
                 transform=size_ax.transAxes)

    # 示例比例值(图例里显示哪几档就改这个列表)
    fracs_demo = [20, 40, 60, 80, 100]                  # <<< 图例档位
    n_demo = len(fracs_demo)
    # 垂直坐标:从上(大值)到下(小值)。想反过来就把 0.82 和 0.08 对调。
    y_positions = np.linspace(0.7, 0.15, n_demo)
    for yp, fd in zip(y_positions, fracs_demo):
        d = d_min + (fd / 100.0) * (d_max - d_min)
        size_ax.scatter([0.2], [yp], s=(d / 2.0) ** 2 * np.pi,   # 0.18 = 圆点列的水平位置
                        c=[[0.5, 0.5, 0.5]], edgecolors="black",
                        linewidths=0.45, transform=size_ax.transAxes,
                        clip_on=False, zorder=3)
        size_ax.text(0.35, yp, str(fd), ha="left", va="center",   # 0.42 = 数字的水平位置
                     fontsize=10, transform=size_ax.transAxes, color="#222")

    # ── (2) 颜色条:【垂直】瘦高型 ────────────────────────────────────────
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ cbar_ax 的四个值 = [左, 下, 宽, 高]。                                 │
    # │   · 第3个值 (0.16) 调色条【粗细】:越大越粗。                        │
    # │   · 第4个值 (0.30) 调色条【高度】。                                  │
    # │   · 第2个值 (0.06) 调上下位置。                                      │
    # └─────────────────────────────────────────────────────────────────────┘
    cbar_ax = fig.add_axes([
        legend_x0 + (legend_w / fig_w) * 0.3,           # <<< 色条 左右位置
        (bottom_margin + panel_h * 0.06) / fig_h,        # <<< 色条 上下位置
        (legend_w / fig_w) * 0.15,                       # <<< 色条 粗细
        (panel_h * 0.30) / fig_h,                        # <<< 色条 高度
    ])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])                        # <<< 色条刻度
    cb.ax.tick_params(labelsize=10, length=3)             # <<< 刻度字号
    cb.outline.set_linewidth(0.6)

    # 颜色条标题(放正上方)。改文字 / 字号就在这一行。
    cbar_ax.text(0.5, 1.04, "Mean expression\nin group",
                 ha="center", va="bottom", fontsize=12,  # <<< 标题字号
                 transform=cbar_ax.transAxes)
    # ══════════════════════════════════════════════════════════════════════

    # ── save ─────────────────────────────────────────────────────────────────
    plt.close(dp.fig) if hasattr(dp, "fig") and dp.fig is not fig else None
    for ext in ("png", "pdf"):
        out = os.path.join(args.outdir, f"{args.name}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.join(args.outdir, args.name)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
