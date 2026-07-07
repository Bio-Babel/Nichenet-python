---
name: use-nichenetr
description: Predicting which sender-cell ligands explain a receiver cell's differential expression, target genes, or receptors, using NicheNet's prior-knowledge networks (ligand-target, ligand-receptor, signaling).
---

# nichenetr

Python port of the R nichenetr package. NicheNet links ligands expressed by
"sender" cells to genes differentially expressed in "receiver" cells, using
a prior model built from curated ligand-receptor, signaling, and
gene-regulatory networks — not from expression co-occurrence alone. Use it
when the question is "which ligands could explain this observed change,"
not merely "which receptors are co-expressed with these ligands."

Do not reach for nichenetr for a plain per-cell-type differential expression
test (use scanpy directly) or for spatial/co-localization based
ligand-receptor scoring — nichenetr has no spatial component and always
requires organism-matched prior networks (ligand_target_matrix, lr_network,
weighted_networks) loaded via the `load_*` functions.

## Mental model

1. Load the prior networks for one organism (`load_ligand_target_matrix`,
   `load_lr_network`, `load_weighted_networks`, ...). Never mix organisms
   across these calls in one analysis — gene overlaps break silently.
2. Canonicalize gene symbols with `alias_to_symbol_anndata` (AnnData input)
   or `convert_alias_to_symbols` (a plain gene list) immediately after
   loading data, before any expressed-gene or DE computation.
3. Core step: `predict_ligand_activities(geneset, background_expressed_genes,
   ligand_target_matrix, potential_ligands)` ranks candidate ligands by how
   well their target-gene regulatory potentials predict membership in your
   gene set of interest (AUROC/AUPR/Pearson).
4. Downstream, top-ranked ligands feed `get_weighted_ligand_target_links` /
   `get_weighted_ligand_receptor_links` for target/receptor inference, then
   `prepare_ligand_target_visualization` + `make_heatmap_ggplot` (or the
   circos/mushroom-plot family) for visualization.
5. `nichenet_seuratobj_aggregate` is a one-call wrapper around the entire
   manual recipe for an AnnData object with a cell-type column and a
   two-level condition column.

## Quick reference

```python
import nichenetr as nn

lr_network = nn.load_lr_network(organism="human")
ligand_target_matrix = nn.load_ligand_target_matrix(organism="human")
weighted_networks = nn.load_weighted_networks(organism="human")

adata = nn.alias_to_symbol_anndata(adata, organism="human")
background = nn.get_expressed_genes(adata, "celltype", "Malignant")
potential_ligands = nn.get_expressed_genes(adata, "celltype", "CAF")

ligand_activities = nn.predict_ligand_activities(
    geneset=geneset_oi,
    background_expressed_genes=background,
    ligand_target_matrix=ligand_target_matrix,
    potential_ligands=potential_ligands,
)
```

For more: `biobabel.describe_package(import_name="nichenetr")`.
