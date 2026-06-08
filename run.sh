#!/bin/bash

### cat metadata_info.txt|awk -F "\t" '{print $2"-"$1"-"$3"_loupe\t"$0}'|sed "s/Chip-ExternalSampleId-ID_loupe/Folder/" > spatial_info.txt

# 1. compute: module scores, grey/white stats, auto boundaries, plot bundle
python 01_compute.py --data-dir matrix --info spatial_info.txt \
      --gene-list gene.list.txt --name-modules-by-genes \
      --score-per-chip --outdir results_gw --score-on-grey-white

# 2. spatial maps (auto boundaries; re-trace poor ones with --draw <Folder>)
python 02_plot_spatial.py --bundle results_gw/plot_bundle.h5ad --data-dir matrix --outdir results_gw

# 3. grey-vs-white figures
python plot_fig_paired.py --sd-dir results_gw --outdir results_gw/figures_custom --share-y "3,4,5" --wspace 0.6
python plot_dotplot_vertical.py --bundle results_gw/plot_bundle.h5ad --sd-dir results_gw --outdir results_gw/figures_custom
python plot_fig_single_gene_box.py --sd-dir results_gw --outdir results_gw/figures_custom --per-page 6 --ncols 6

# co-expression / module / single-gene spatial maps
python plot_spatial_genes.py --data-dir matrix --info spatial_info.txt \
    --gene CLEC7A,ITGAX,LGALS3,SPP1,LPL,CD9 --ncols 7 \
    --boundary-dir results_gw/boundaries --outdir results_coexpr/genes \
    --rotate-file rotations.tsv
  
python plot_spatial_module.py --data-dir matrix --info spatial_info.txt \
    --gene-list gene.list.for_module_score.txt --boundary-dir results_gw/boundaries \
    --outdir results_coexpr/module --rotate-file rotations.tsv
  
python plot_spatial_coexpr.py --data-dir matrix --info spatial_info.txt \
    --gene-list gene.list.for_coExp.txt --boundary-dir results_gw/boundaries \
    --outdir results_coexpr/coexpr --rotate-file rotations.tsv
