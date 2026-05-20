# nichenet-python

[![PyPI](https://img.shields.io/pypi/v/nichenet-python)](https://pypi.org/project/nichenet-python/)

Python [**nichenetr**](https://github.com/saeyslab/nichenetr) package — modeling intercellular communication by linking ligands to target genes.

## Installation

```bash
pip install nichenet-python                # from PyPI
```

For local development:

```bash
git clone https://github.com/Bio-Babel/Nichenet-python.git
cd Nichenet-python
pip install -e ".[dev]"
```

## Tutorials

Runnable notebooks reproducing the R nichenetr vignettes live under [`tutorials/`](tutorials/):

| Notebook | Topic |
|---|---|
| `ligand_activity_geneset.ipynb` | Ligand activity analysis on a gene set of interest |
| `ligand_activity_single_cell.ipynb` | Per-cell ligand activity prediction |
| `ligand_target_signaling_path.ipynb` | Inferring ligand→target signaling paths |
| `target_prediction_evaluation_geneset.ipynb` | Evaluating target gene predictions |
| `seurat_steps.ipynb` | Step-by-step NicheNet analysis on a Seurat-style object |
| `seurat_steps_prioritization.ipynb` | Adds the prioritization scheme on top of `seurat_steps` |
| `seurat_wrapper.ipynb` | One-call wrapper for the Seurat-style workflow |
| `seurat_wrapper_circos.ipynb` | Wrapper workflow with circos visualization |
| `circos.ipynb` | Standalone circos plots of ligand→target / ligand→receptor links |

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```

## License

GPL-3.0-only, matching the upstream R nichenetr.
