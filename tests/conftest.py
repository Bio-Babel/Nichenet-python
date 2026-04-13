"""Shared pytest fixtures for nichenetr_py."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from nichenetr.datasets import NamedMatrix


@pytest.fixture
def small_lr_network():
    """A small DataFrame with ~5 LR pairs."""
    return pd.DataFrame(
        {
            "from": ["L1", "L2", "L3", "L1", "L2"],
            "to": ["R1", "R2", "R3", "R2", "R3"],
            "source": ["src1", "src1", "src2", "src2", "src1"],
            "database": ["db1", "db1", "db2", "db2", "db1"],
        }
    )


@pytest.fixture
def small_ligand_target_matrix():
    """A NamedMatrix with 10 target genes and 5 ligands."""
    rng = np.random.RandomState(42)
    genes = [f"G{i}" for i in range(1, 11)]
    ligands = ["L1", "L2", "L3", "L4", "L5"]
    data = rng.rand(10, 5).astype(np.float64)
    # Make it sparse
    data[data < 0.3] = 0.0
    sparse_data = scipy.sparse.csr_matrix(data)
    return NamedMatrix(data=sparse_data, rownames=genes, colnames=ligands)


@pytest.fixture
def small_geneset():
    """List of 4 gene names that overlap with small_ligand_target_matrix rows."""
    return ["G1", "G3", "G5", "G7"]


@pytest.fixture
def small_adata():
    """A small AnnData with 100 cells, 50 genes, 3 cell types."""
    import anndata

    rng = np.random.RandomState(99)
    n_cells = 100
    n_genes = 50
    X = rng.rand(n_cells, n_genes).astype(np.float32)

    cell_types = np.array(["TypeA"] * 34 + ["TypeB"] * 33 + ["TypeC"] * 33)
    obs = pd.DataFrame(
        {"celltype": cell_types},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    return adata
