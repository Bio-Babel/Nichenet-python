# nichenetr

Python port of the R [nichenetr](https://github.com/saeyslab/nichenetr) package (v2.2.1.1).

NicheNet models intercellular communication by linking ligands to target genes. It predicts which ligands from sender cells are most likely to affect gene expression in receiver cells, using an integrated model of ligand-receptor interactions, signaling pathways, and gene regulatory networks.

## Installation

```bash
pip install -e .
```

For circos plot support:

```bash
pip install -e ".[circos]"
```

## Quick Start

```python
import nichenetr as nn

# Load pre-built networks (mouse)
lr_network = nn.load_lr_network(organism="mouse")
ligand_target_matrix = nn.load_ligand_target_matrix(organism="mouse")
weighted_networks = nn.load_weighted_networks(organism="mouse")

# Predict ligand activities
activities = nn.predict_ligand_activities(
    geneset=de_genes,
    background_expressed_genes=expressed_genes,
    ligand_target_matrix=ligand_target_matrix,
    potential_ligands=potential_ligands,
)
```

## Key Features

- **Ligand activity prediction**: Rank candidate ligands by their predicted ability to regulate a gene set of interest
- **Target gene inference**: Identify which target genes are most likely regulated by top-ranked ligands
- **Signaling path analysis**: Trace ligand-to-target signaling paths through intermediate signaling and regulatory networks
- **Prioritization**: Combine ligand activity with expression and differential expression evidence
- **Visualization**: Heatmaps, circos plots, mushroom plots for ligand-receptor-target relationships
- **Wrappers**: High-level functions for AnnData-based workflows (`nichenet_seuratobj_aggregate`)

## Data

The package uses pre-built NicheNet models hosted on Zenodo. Data is downloaded automatically on first use and cached locally. Both human and mouse networks are available.

## Tutorials

- [Step-by-step NicheNet analysis](tutorials/seurat_steps.ipynb)
- [NicheNet wrapper](tutorials/seurat_wrapper.ipynb)
- [Gene set-based ligand activity](tutorials/ligand_activity_geneset.ipynb)
- [Prioritization with multiple senders/receivers](tutorials/seurat_steps_prioritization.ipynb)
- [Signaling path inference](tutorials/ligand_target_signaling_path.ipynb)
- [Target prediction evaluation](tutorials/target_prediction_evaluation_geneset.ipynb)
- [Single-cell ligand activity](tutorials/ligand_activity_single_cell.ipynb)
- [Circos visualization (Seurat wrapper)](tutorials/seurat_wrapper_circos.ipynb)
- [Advanced circos visualization](tutorials/circos.ipynb)

## Differences from the R Package

- Uses **AnnData** instead of Seurat objects as the single-cell data container
- Uses **scanpy** for differential expression (`rank_genes_groups` with Wilcoxon test)
- Uses **matplotlib/seaborn** for visualization instead of ggplot2
- Uses **pycirclize** for circos plots instead of R circlize
- Uses **networkx** for graph algorithms instead of igraph
- Uses **scikit-learn** for random forest and evaluation metrics
- Function `alias_to_symbol_seurat` is renamed to `alias_to_symbol_anndata`
- Function `diagrammer_format_signaling_graph` is renamed to `format_signaling_graph`
