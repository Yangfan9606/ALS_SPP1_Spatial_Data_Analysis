# ALS_SPP1_Spatial_Data_Analysis
Scripts of spatial data analysis for ALS-SPP1 project

# Spinal-cord Visium — grey/white matter marker analysis

Python pipeline for the spatial transcriptomics (10x Visium) analysis in **Microglial heterogeneity and SPP1‑mediated glial crosstalk underlie ALS neuroinflammation**. For each tissue section it assigns spots to grey vs white matter (from Loupe clusters), draws the grey/white boundary, computes marker-set module / co-expression scores, and produces the spatial maps and grey-vs-white statistical figures.

## Input data

This repository contains **code only**. The Visium count matrices, spatial coordinates and the sample metadata are **not** redistributed here and are **available from the corresponding/first author on reasonable request**.

Per-section inputs are expected under `--data-dir` (one sub-directory per section, named by the `Folder` column of `spatial_info.txt`):

```
matrix/<Folder>/
├── matrix/                       # 10x mtx (matrix.mtx, features/genes.tsv, barcodes.tsv)
├── metadata.tsv.gz               # per-spot Loupe clustering metadata
└── spatial/spatial_xy_matrix.tsv.gz   # per-spot x/y coordinates
```

These per-spot tables (clusters + coordinates) are exported from the `.cloupe` file with **cloupe-extract** (https://github.com/Coleliao/cloupe-extract).

The section table `spatial_info.txt` (columns: `Folder`, `ExternalSampleId`, `Number of cluster`, `Grey`, `White`, …) is built from a private `metadata_info.txt`; see `spatial_info.template.txt` for the format and the one-line `awk` recipe. The grey/white cluster IDs (`Grey`, `White`) are chosen by manual inspection in 10x Loupe Browser.

Marker panels (`gene.list*.txt`) and display rotations (`rotations.tsv`) are small text configs; add your own or the ones used in the paper.

## Install

```bash
pip install -r requirements.txt   # Python >= 3.10
```

## Run

`run.sh` is the canonical end-to-end recipe. In brief:

```bash
# 1. compute: module scores, grey/white stats, auto boundaries, plot bundle
python 01_compute.py --data-dir matrix --info spatial_info.txt \
    --gene-list gene.list.txt --name-modules-by-genes \
    --score-per-chip --score-on-grey-white --outdir results_gw

# 2. spatial maps (auto boundaries; re-trace poor ones with --draw <Folder>)
python 02_plot_spatial.py --bundle results_gw/plot_bundle.h5ad \
    --data-dir matrix --outdir results_gw

# 3. grey-vs-white figures
python plot_fig_paired.py        --sd-dir results_gw --outdir results_gw/figures_custom --share-y "3,4,5"
python plot_dotplot_vertical.py  --bundle results_gw/plot_bundle.h5ad --sd-dir results_gw --outdir results_gw/figures_custom
python plot_fig_single_gene_box.py --sd-dir results_gw --outdir results_gw/figures_custom

# co-expression / module / single-gene spatial maps
python plot_spatial_coexpr.py  --data-dir matrix --info spatial_info.txt --gene-list gene.list.for_coExp.v2.txt --boundary-dir results_gw/boundaries --outdir results_coexpr/coexpr
python plot_spatial_module.py  --data-dir matrix --info spatial_info.txt --gene-list gene.list.for_module_score.txt --boundary-dir results_gw/boundaries --outdir results_coexpr/module
python plot_spatial_genes.py   --data-dir matrix --info spatial_info.txt --gene CLEC7A,ITGAX,LGALS3,SPP1,LPL,CD9 --boundary-dir results_gw/boundaries --outdir results_coexpr/genes
```

## Scripts

| File | Purpose |
|---|---|
| `gw_common.py` | shared core: chip loading, log1p-CPM normalisation, module scores, grey/white boundary, stats, I/O |
| `coexpr_common.py` | shared plotting helpers for the spatial map scripts |
| `01_compute.py` | build all source-data tables + plot bundle + auto boundaries (no plotting) |
| `02_plot_spatial.py` | per-section spatial maps; `--draw` re-traces a boundary interactively |
| `plot_spatial_coexpr.py` | co-expression spatial maps (min co-score / k-of-N / count) |
| `plot_spatial_module.py` | module-score spatial maps |
| `plot_spatial_genes.py` | single-gene spatial maps |
| `plot_fig_paired.py` | per-section paired grey-vs-white module test (Wilcoxon/t-test) |
| `plot_dotplot_vertical.py` | grey-vs-white dot plot (Mann–Whitney U per gene) |
| `plot_fig_single_gene_box.py` | per-gene grey-vs-white box plots on expressing spots |

## Citation / data availability

Please cite **Microglial heterogeneity and SPP1-mediated glial crosstalk underlie ALS neuroinflammation**. Processed and raw data are available from the authors on request: Jianing Lin (jianing.lin@ki.se, first author) and Sebastian Lewandowski (sebastian.lewandowski@ki.se, corresponding author), Karolinska Institutet.

## License

MIT — see [LICENSE](LICENSE).
