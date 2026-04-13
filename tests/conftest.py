"""Shared pytest fixtures for nichenetr_py."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import anndata

from nichenetr.datasets import NamedMatrix


# ---------------------------------------------------------------------------
# Basic fixtures (original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Extended fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adata_with_conditions():
    """AnnData with 200 cells, 80 genes, 3 cell types, and 2 conditions.

    Gene names include L1-L5 (ligands) and R1-R5 (receptors) to match
    lr_network_named fixture.
    """
    rng = np.random.RandomState(123)
    n_cells = 200
    gene_names = (
        [f"L{i}" for i in range(1, 6)]
        + [f"R{i}" for i in range(1, 6)]
        + [f"G{i}" for i in range(1, 71)]
    )
    n_genes = len(gene_names)
    X = rng.rand(n_cells, n_genes).astype(np.float32)

    # Make some genes differentially expressed between conditions
    # In condition "treated", L1-L3 and G1-G10 have higher expression in TypeA
    cell_types = (
        ["TypeA"] * 50 + ["TypeB"] * 50 + ["TypeC"] * 50 + ["TypeA"] * 25 + ["TypeB"] * 25
    )
    conditions = (
        ["treated"] * 50 + ["treated"] * 50 + ["treated"] * 50
        + ["control"] * 25 + ["control"] * 25
    )

    # Boost expression of L1-L3 and G1-G10 in treated TypeA
    for i in range(50):  # treated TypeA cells
        X[i, :3] += 2.0  # L1, L2, L3
        X[i, 10:20] += 1.5  # G1-G10

    obs = pd.DataFrame(
        {"celltype": cell_types, "condition": conditions},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=gene_names)
    return anndata.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def lr_network_named():
    """LR network with ligand/receptor column names."""
    return pd.DataFrame({
        "from": ["L1", "L2", "L3", "L1", "L2", "L4", "L5"],
        "to": ["R1", "R2", "R3", "R2", "R3", "R4", "R5"],
        "source": ["s1"] * 7,
        "database": ["d1"] * 7,
    })


@pytest.fixture
def weighted_networks_fixture():
    """Weighted networks with lr_sig and gr DataFrames."""
    lr_sig = pd.DataFrame({
        "from": ["L1", "L1", "L2", "L2", "L3", "TF1", "TF2"],
        "to": ["TF1", "R1", "TF2", "R2", "R3", "TF2", "TF3"],
        "weight": [0.8, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5],
    })
    gr = pd.DataFrame({
        "from": ["TF1", "TF2", "TF3", "TF1", "TF2"],
        "to": ["G1", "G1", "G2", "G3", "G2"],
        "weight": [0.9, 0.4, 0.6, 0.3, 0.7],
    })
    return {"lr_sig": lr_sig, "gr": gr}


@pytest.fixture
def ligand_tf_matrix():
    """NamedMatrix for TFs (rows) x Ligands (cols)."""
    tfs = ["TF1", "TF2", "TF3"]
    ligands = ["L1", "L2", "L3"]
    data = np.array([
        [0.5, 0.0, 0.2],
        [0.3, 0.7, 0.1],
        [0.0, 0.4, 0.6],
    ])
    return NamedMatrix(
        data=scipy.sparse.csr_matrix(data),
        rownames=tfs,
        colnames=ligands,
    )


@pytest.fixture
def large_ligand_target_matrix():
    """Larger NamedMatrix matching gene names from adata_with_conditions."""
    rng = np.random.RandomState(55)
    gene_names = (
        [f"L{i}" for i in range(1, 6)]
        + [f"R{i}" for i in range(1, 6)]
        + [f"G{i}" for i in range(1, 71)]
    )
    ligand_names = [f"L{i}" for i in range(1, 6)]
    n_genes = len(gene_names)
    n_ligands = len(ligand_names)
    data = rng.rand(n_genes, n_ligands).astype(np.float64)
    data[data < 0.3] = 0.0
    return NamedMatrix(
        data=scipy.sparse.csr_matrix(data),
        rownames=gene_names,
        colnames=ligand_names,
    )


@pytest.fixture
def rf_prediction_df():
    """A typical output of assess_rf_class_probabilities."""
    rng = np.random.RandomState(10)
    n = 50
    genes = [f"gene_{i}" for i in range(n)]
    response = [True] * 15 + [False] * 35
    prediction = rng.rand(n)
    # Make true targets have higher predictions
    prediction[:15] += 0.3
    prediction = np.clip(prediction, 0, 1)
    return pd.DataFrame({
        "gene": genes,
        "response": response,
        "prediction": prediction,
    })


@pytest.fixture
def prioritization_table():
    """A minimal prioritization table for mushroom/circos tests."""
    n = 20
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "sender": ["TypeA"] * 10 + ["TypeB"] * 10,
        "receiver": ["TypeC"] * 20,
        "ligand": [f"L{i % 5 + 1}" for i in range(n)],
        "receptor": [f"R{i % 5 + 1}" for i in range(n)],
        "lfc_ligand": rng.randn(n),
        "lfc_receptor": rng.randn(n),
        "p_val_ligand": rng.uniform(0.001, 0.1, n),
        "p_val_receptor": rng.uniform(0.001, 0.1, n),
        "p_adj_ligand": rng.uniform(0.001, 0.1, n),
        "p_adj_receptor": rng.uniform(0.001, 0.1, n),
        "pct_expressed_sender": rng.rand(n),
        "pct_expressed_receiver": rng.rand(n),
        "avg_ligand": rng.rand(n),
        "avg_receptor": rng.rand(n),
        "ligand_receptor_prod": rng.rand(n),
        "scaled_avg_exprs_ligand": rng.rand(n),
        "scaled_avg_exprs_receptor": rng.rand(n),
        "scaled_p_val_adapted_ligand": rng.rand(n),
        "scaled_p_val_adapted_receptor": rng.rand(n),
        "scaled_activity": rng.rand(n),
        "prioritization_score": rng.rand(n),
    })


@pytest.fixture
def circos_links():
    """DataFrame for circos visualization tests."""
    return pd.DataFrame({
        "ligand": ["L1", "L1", "L2", "L2", "L3"],
        "target": ["G1", "G2", "G1", "G3", "G2"],
        "weight": [0.8, 0.5, 0.6, 0.3, 0.7],
        "target_type": ["TypeC", "TypeC", "TypeC", "TypeC", "TypeC"],
        "ligand_type": ["TypeA", "TypeA", "TypeB", "TypeB", "General"],
    })


@pytest.fixture
def expression_scaled():
    """Scaled expression DataFrame for single-cell ligand activity tests."""
    rng = np.random.RandomState(88)
    cell_ids = [f"cell_{i}" for i in range(10)]
    gene_names = [f"G{i}" for i in range(1, 11)]
    data = rng.rand(10, 10)
    return pd.DataFrame(data, index=cell_ids, columns=gene_names)
