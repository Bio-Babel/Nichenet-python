"""Tests for nichenetr.wrappers."""

import numpy as np
import pandas as pd
import pytest
import anndata
import scipy.sparse

from nichenetr.datasets import NamedMatrix
from nichenetr.wrappers import (
    nichenet_seuratobj_aggregate,
    nichenet_seuratobj_cluster_de,
    _resolve_sender_celltypes,
    _get_expressed_genes_for_celltypes,
    _compute_de_between_conditions,
    _build_lr_receptor_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def wrapper_fixtures():
    """Build self-consistent small data for wrapper tests."""
    rng = np.random.RandomState(42)
    # Gene names
    ligand_names = ["LIG1", "LIG2", "LIG3"]
    receptor_names = ["REC1", "REC2", "REC3"]
    target_names = [f"TGT{i}" for i in range(1, 31)]
    all_genes = ligand_names + receptor_names + target_names

    n_cells = 120
    n_genes = len(all_genes)
    X = rng.rand(n_cells, n_genes).astype(np.float32)

    # Boost ligands in TypeA treated
    X[:30, :3] += 2.0  # LIG1-3 in treated TypeA
    # Boost targets in TypeB treated
    X[30:60, 6:20] += 1.5  # TGT1-14 in treated TypeB

    cell_types = (
        ["TypeA"] * 30 + ["TypeB"] * 30 +  # treated
        ["TypeA"] * 30 + ["TypeB"] * 30     # control
    )
    conditions = (
        ["treated"] * 60 +
        ["control"] * 60
    )

    obs = pd.DataFrame(
        {"celltype": cell_types, "condition": conditions},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=all_genes)
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # LR network
    lr_network = pd.DataFrame({
        "from": ["LIG1", "LIG2", "LIG3", "LIG1"],
        "to": ["REC1", "REC2", "REC3", "REC2"],
        "source": ["s"] * 4,
        "database": ["d"] * 4,
    })

    # Ligand-target matrix: all_genes rows x ligand_names columns
    lt_data = rng.rand(n_genes, 3).astype(np.float64)
    lt_data[lt_data < 0.3] = 0.0
    # Boost target gene weights for LIG1
    for i in range(6, 20):  # TGT1-14
        lt_data[i, 0] += 0.5
    lt_matrix = NamedMatrix(
        data=scipy.sparse.csr_matrix(lt_data),
        rownames=all_genes,
        colnames=ligand_names,
    )

    # Weighted networks
    lr_sig_rows = []
    for lig in ligand_names:
        for rec in receptor_names:
            lr_sig_rows.append({"from": lig, "to": rec, "weight": rng.rand()})
    weighted_networks = {
        "lr_sig": pd.DataFrame(lr_sig_rows),
        "gr": pd.DataFrame({
            "from": ["TF1"], "to": ["TGT1"], "weight": [0.5],
        }),
    }

    return {
        "adata": adata,
        "lr_network": lr_network,
        "lt_matrix": lt_matrix,
        "weighted_networks": weighted_networks,
    }


# ---------------------------------------------------------------------------
# _resolve_sender_celltypes
# ---------------------------------------------------------------------------

class TestResolveSenderCelltypes:
    def test_all(self, small_adata):
        result = _resolve_sender_celltypes("all", small_adata, "celltype")
        assert set(result) == {"TypeA", "TypeB", "TypeC"}

    def test_undefined(self, small_adata):
        result = _resolve_sender_celltypes("undefined", small_adata, "celltype")
        assert result == []

    def test_single_string(self, small_adata):
        result = _resolve_sender_celltypes("TypeA", small_adata, "celltype")
        assert result == ["TypeA"]

    def test_list(self, small_adata):
        result = _resolve_sender_celltypes(["TypeA", "TypeB"], small_adata, "celltype")
        assert result == ["TypeA", "TypeB"]


# ---------------------------------------------------------------------------
# _get_expressed_genes_for_celltypes
# ---------------------------------------------------------------------------

class TestGetExpressedGenesForCelltypes:
    def test_basic(self, small_adata):
        result = _get_expressed_genes_for_celltypes(
            ["TypeA", "TypeB"], small_adata, "celltype", 0.10, None
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_union(self, small_adata):
        a = _get_expressed_genes_for_celltypes(["TypeA"], small_adata, "celltype", 0.10, None)
        b = _get_expressed_genes_for_celltypes(["TypeB"], small_adata, "celltype", 0.10, None)
        both = _get_expressed_genes_for_celltypes(
            ["TypeA", "TypeB"], small_adata, "celltype", 0.10, None
        )
        assert len(both) >= max(len(a), len(b))


# ---------------------------------------------------------------------------
# _compute_de_between_conditions
# ---------------------------------------------------------------------------

class TestComputeDeBetweenConditions:
    def test_basic(self, wrapper_fixtures):
        result = _compute_de_between_conditions(
            wrapper_fixtures["adata"],
            celltype_col="celltype",
            receiver="TypeB",
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            expression_pct=0.10,
            assay_oi=None,
        )
        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns
        assert "p_val" in result.columns
        assert "avg_log2FC" in result.columns
        assert "pct.1" in result.columns
        assert "pct.2" in result.columns

    def test_with_list_receiver(self, wrapper_fixtures):
        result = _compute_de_between_conditions(
            wrapper_fixtures["adata"],
            celltype_col="celltype",
            receiver=["TypeA", "TypeB"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            expression_pct=0.10,
            assay_oi=None,
        )
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _build_lr_receptor_matrix
# ---------------------------------------------------------------------------

class TestBuildLrReceptorMatrix:
    def test_basic(self):
        df = pd.DataFrame({
            "from": ["L1", "L1", "L2", "L2"],
            "to": ["R1", "R2", "R1", "R2"],
            "weight": [0.5, 0.8, 0.3, 0.7],
        })
        mat, receptors, ligands = _build_lr_receptor_matrix(df)
        assert mat.shape == (2, 2)
        assert len(receptors) == 2
        assert len(ligands) == 2

    def test_single_entry(self):
        df = pd.DataFrame({"from": ["L1"], "to": ["R1"], "weight": [0.5]})
        mat, receptors, ligands = _build_lr_receptor_matrix(df)
        assert mat.shape == (1, 1)


# ---------------------------------------------------------------------------
# nichenet_seuratobj_aggregate
# ---------------------------------------------------------------------------

class TestNichenetSeuratObjAggregate:
    def test_basic(self, wrapper_fixtures):
        result = nichenet_seuratobj_aggregate(
            receiver="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            top_n_ligands=3,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)
        assert "ligand_activities" in result
        assert "top_ligands" in result
        assert "geneset_oi" in result
        assert isinstance(result["ligand_activities"], pd.DataFrame)

    def test_no_celltype_col_raises(self, wrapper_fixtures):
        with pytest.raises(ValueError, match="celltype_col must be provided"):
            nichenet_seuratobj_aggregate(
                receiver="TypeB",
                adata=wrapper_fixtures["adata"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
                celltype_col=None,
                ligand_target_matrix=wrapper_fixtures["lt_matrix"],
                lr_network=wrapper_fixtures["lr_network"],
                weighted_networks=wrapper_fixtures["weighted_networks"],
            )

    def test_invalid_geneset_raises(self, wrapper_fixtures):
        with pytest.raises(ValueError, match="geneset must be"):
            nichenet_seuratobj_aggregate(
                receiver="TypeB",
                adata=wrapper_fixtures["adata"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
                celltype_col="celltype",
                geneset="invalid",
                ligand_target_matrix=wrapper_fixtures["lt_matrix"],
                lr_network=wrapper_fixtures["lr_network"],
                weighted_networks=wrapper_fixtures["weighted_networks"],
            )

    def test_undefined_sender(self, wrapper_fixtures):
        result = nichenet_seuratobj_aggregate(
            receiver="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="undefined",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)

    def test_geneset_up(self, wrapper_fixtures):
        result = nichenet_seuratobj_aggregate(
            receiver="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            geneset="up",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)

    def test_filter_top_false(self, wrapper_fixtures):
        result = nichenet_seuratobj_aggregate(
            receiver="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            filter_top_ligands=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)

    def test_geneset_down(self, wrapper_fixtures):
        try:
            result = nichenet_seuratobj_aggregate(
                receiver="TypeB",
                adata=wrapper_fixtures["adata"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
                sender="TypeA",
                celltype_col="celltype",
                geneset="down",
                ligand_target_matrix=wrapper_fixtures["lt_matrix"],
                lr_network=wrapper_fixtures["lr_network"],
                weighted_networks=wrapper_fixtures["weighted_networks"],
                verbose=False,
                lfc_cutoff=0.1,
            )
            assert isinstance(result, dict)
        except ValueError:
            # May raise "No genes were differentially expressed" for down
            pass

    def test_list_receiver(self, wrapper_fixtures):
        result = nichenet_seuratobj_aggregate(
            receiver=["TypeB"],
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender=["TypeA"],
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)

    def test_all_senders(self, wrapper_fixtures):
        result = nichenet_seuratobj_aggregate(
            receiver="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="all",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# nichenet_seuratobj_cluster_de
# ---------------------------------------------------------------------------

class TestComputeDeWithLayer:
    def test_with_layer(self, wrapper_fixtures):
        adata = wrapper_fixtures["adata"]
        adata.layers["custom"] = adata.X.copy()
        result = _compute_de_between_conditions(
            adata, celltype_col="celltype", receiver="TypeB",
            condition_col="condition", condition_oi="treated",
            condition_ref="control", expression_pct=0.1, assay_oi="custom",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestAggregateWithLrNetLigandCol:
    """Test when lr_network already has 'ligand'/'receptor' columns."""

    def test_ligand_receptor_cols(self, wrapper_fixtures):
        lr = wrapper_fixtures["lr_network"].copy()
        lr = lr.rename(columns={"from": "ligand", "to": "receptor"})
        result = nichenet_seuratobj_aggregate(
            receiver="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=lr,
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)


class TestNichenetSeuratObjClusterDe:
    def test_basic(self, wrapper_fixtures):
        result = nichenet_seuratobj_cluster_de(
            receiver_affected="TypeB",
            receiver_reference="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)
        assert "ligand_activities" in result
        assert "top_ligands" in result
        assert isinstance(result["ligand_activities"], pd.DataFrame)

    def test_no_celltype_col_raises(self, wrapper_fixtures):
        with pytest.raises(ValueError, match="celltype_col must be provided"):
            nichenet_seuratobj_cluster_de(
                receiver_affected="TypeB",
                receiver_reference="TypeB",
                adata=wrapper_fixtures["adata"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
                celltype_col=None,
                ligand_target_matrix=wrapper_fixtures["lt_matrix"],
                lr_network=wrapper_fixtures["lr_network"],
                weighted_networks=wrapper_fixtures["weighted_networks"],
            )

    def test_invalid_geneset_raises(self, wrapper_fixtures):
        with pytest.raises(ValueError, match="geneset must be"):
            nichenet_seuratobj_cluster_de(
                receiver_affected="TypeB",
                receiver_reference="TypeB",
                adata=wrapper_fixtures["adata"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
                celltype_col="celltype",
                geneset="invalid",
                ligand_target_matrix=wrapper_fixtures["lt_matrix"],
                lr_network=wrapper_fixtures["lr_network"],
                weighted_networks=wrapper_fixtures["weighted_networks"],
            )

    def test_undefined_sender(self, wrapper_fixtures):
        result = nichenet_seuratobj_cluster_de(
            receiver_affected="TypeB",
            receiver_reference="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="undefined",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)

    def test_geneset_up(self, wrapper_fixtures):
        result = nichenet_seuratobj_cluster_de(
            receiver_affected="TypeB",
            receiver_reference="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            geneset="up",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)

    def test_filter_top_false(self, wrapper_fixtures):
        result = nichenet_seuratobj_cluster_de(
            receiver_affected="TypeB",
            receiver_reference="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            filter_top_ligands=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)

    def test_list_receivers(self, wrapper_fixtures):
        result = nichenet_seuratobj_cluster_de(
            receiver_affected=["TypeB"],
            receiver_reference=["TypeB"],
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender=["TypeA"],
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)

    def test_geneset_down_cluster(self, wrapper_fixtures):
        try:
            result = nichenet_seuratobj_cluster_de(
                receiver_affected="TypeB",
                receiver_reference="TypeB",
                adata=wrapper_fixtures["adata"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
                sender="TypeA",
                celltype_col="celltype",
                geneset="down",
                ligand_target_matrix=wrapper_fixtures["lt_matrix"],
                lr_network=wrapper_fixtures["lr_network"],
                weighted_networks=wrapper_fixtures["weighted_networks"],
                verbose=False,
                lfc_cutoff=0.1,
            )
            assert isinstance(result, dict)
        except ValueError:
            pass

    def test_with_lr_ligand_receptor_cols(self, wrapper_fixtures):
        """Test cluster_de when LR network uses ligand/receptor cols."""
        lr = wrapper_fixtures["lr_network"].copy()
        lr = lr.rename(columns={"from": "ligand", "to": "receptor"})
        result = nichenet_seuratobj_cluster_de(
            receiver_affected="TypeB",
            receiver_reference="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=lr,
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)

    def test_with_dataframe_lt_matrix(self, wrapper_fixtures):
        """Test when ligand_target_matrix is a DataFrame instead of NamedMatrix."""
        lt = wrapper_fixtures["lt_matrix"]
        lt_df = pd.DataFrame(
            lt.data.toarray(),
            index=lt.rownames,
            columns=lt.colnames,
        )
        result = nichenet_seuratobj_cluster_de(
            receiver_affected="TypeB",
            receiver_reference="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=lt_df,
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)


class TestAggregateVerbose:
    def test_verbose_output(self, wrapper_fixtures, capsys):
        result = nichenet_seuratobj_aggregate(
            receiver="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=True,
            lfc_cutoff=0.1,
        )
        captured = capsys.readouterr()
        assert "ligand activity" in captured.out.lower() or "receptor" in captured.out.lower()


class TestClusterDeVerbose:
    def test_verbose_output(self, wrapper_fixtures, capsys):
        result = nichenet_seuratobj_cluster_de(
            receiver_affected="TypeB",
            receiver_reference="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=True,
            lfc_cutoff=0.1,
        )
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestAggregateWithDataFrameLtMatrix:
    """Test aggregate when ligand_target_matrix is a DataFrame."""

    def test_basic(self, wrapper_fixtures):
        lt = wrapper_fixtures["lt_matrix"]
        lt_df = pd.DataFrame(
            lt.data.toarray(),
            index=lt.rownames,
            columns=lt.colnames,
        )
        result = nichenet_seuratobj_aggregate(
            receiver="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=lt_df,
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wrapper_fixtures["weighted_networks"],
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)


class TestAggregateWeightedNetworkColumns:
    """Test when weighted_networks lr_sig uses ligand/receptor cols."""

    def test_ligand_receptor_in_weighted_networks(self, wrapper_fixtures):
        wn = wrapper_fixtures["weighted_networks"].copy()
        wn["lr_sig"] = wn["lr_sig"].rename(columns={"from": "ligand", "to": "receptor"})
        result = nichenet_seuratobj_aggregate(
            receiver="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wn,
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)


class TestAutoLoadResources:
    """Test that None resources trigger auto-loading."""

    def test_aggregate_auto_load(self, wrapper_fixtures):
        from unittest.mock import patch
        with patch("nichenetr.wrappers.load_ligand_target_matrix", return_value=wrapper_fixtures["lt_matrix"]), \
             patch("nichenetr.wrappers.load_lr_network", return_value=wrapper_fixtures["lr_network"]), \
             patch("nichenetr.wrappers.load_weighted_networks", return_value=wrapper_fixtures["weighted_networks"]):
            result = nichenet_seuratobj_aggregate(
                receiver="TypeB",
                adata=wrapper_fixtures["adata"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
                sender="TypeA",
                celltype_col="celltype",
                ligand_target_matrix=None,
                lr_network=None,
                weighted_networks=None,
                verbose=False,
                lfc_cutoff=0.1,
            )
        assert isinstance(result, dict)

    def test_cluster_de_auto_load(self, wrapper_fixtures):
        from unittest.mock import patch
        with patch("nichenetr.wrappers.load_ligand_target_matrix", return_value=wrapper_fixtures["lt_matrix"]), \
             patch("nichenetr.wrappers.load_lr_network", return_value=wrapper_fixtures["lr_network"]), \
             patch("nichenetr.wrappers.load_weighted_networks", return_value=wrapper_fixtures["weighted_networks"]):
            result = nichenet_seuratobj_cluster_de(
                receiver_affected="TypeB",
                receiver_reference="TypeB",
                adata=wrapper_fixtures["adata"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
                sender="TypeA",
                celltype_col="celltype",
                ligand_target_matrix=None,
                lr_network=None,
                weighted_networks=None,
                verbose=False,
                lfc_cutoff=0.1,
            )
        assert isinstance(result, dict)


class TestClusterDeWeightedNetworkColumns:
    """Test cluster_de when weighted_networks lr_sig uses ligand/receptor cols."""

    def test_ligand_receptor_in_weighted_networks(self, wrapper_fixtures):
        wn = wrapper_fixtures["weighted_networks"].copy()
        wn["lr_sig"] = wn["lr_sig"].rename(columns={"from": "ligand", "to": "receptor"})
        result = nichenet_seuratobj_cluster_de(
            receiver_affected="TypeB",
            receiver_reference="TypeB",
            adata=wrapper_fixtures["adata"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            sender="TypeA",
            celltype_col="celltype",
            ligand_target_matrix=wrapper_fixtures["lt_matrix"],
            lr_network=wrapper_fixtures["lr_network"],
            weighted_networks=wn,
            verbose=False,
            lfc_cutoff=0.1,
        )
        assert isinstance(result, dict)
