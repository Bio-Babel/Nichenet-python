"""Tests for nichenetr.datasets (bundled resources and remote loaders)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from nichenetr.datasets import (
    NamedMatrix,
    _load_named_matrix,
    load_geneinfo,
    load_geneinfo_alias,
    load_hyperparameter_list,
    load_lr_network,
    load_ligand_target_matrix,
    load_weighted_networks,
    load_sig_network,
    load_gr_network,
    load_ligand_tf_matrix,
    load_seurat_obj,
    load_hnscc_expression,
    load_pemt_signature,
    load_source_weights_df,
    load_optimized_source_weights_df,
)


# ---------------------------------------------------------------------------
# Bundled resource tests
# ---------------------------------------------------------------------------

class TestLoadSourceWeightsDf:
    def test_returns_dataframe(self):
        result = load_source_weights_df()
        assert isinstance(result, pd.DataFrame)

    def test_not_empty(self):
        result = load_source_weights_df()
        assert len(result) > 0

    def test_has_expected_columns(self):
        result = load_source_weights_df()
        assert result.shape[1] >= 1


class TestLoadOptimizedSourceWeightsDf:
    def test_returns_dataframe(self):
        result = load_optimized_source_weights_df()
        assert isinstance(result, pd.DataFrame)

    def test_not_empty(self):
        result = load_optimized_source_weights_df()
        assert len(result) > 0


class TestLoadGeneinfo:
    def test_returns_dataframe(self):
        result = load_geneinfo()
        assert isinstance(result, pd.DataFrame)

    def test_not_empty(self):
        result = load_geneinfo()
        assert len(result) > 0

    def test_human_version(self):
        result = load_geneinfo(version="human")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_invalid_version_raises(self):
        with pytest.raises(FileNotFoundError):
            load_geneinfo(version="nonexistent_version")


class TestLoadGeneInfoAlias:
    def test_returns_dataframe_human(self):
        result = load_geneinfo_alias(organism="human")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "alias" in result.columns
        assert "symbol" in result.columns

    def test_invalid_organism_raises(self):
        with pytest.raises(FileNotFoundError):
            load_geneinfo_alias(organism="fish")


class TestLoadHyperparameterList:
    def test_returns_list_or_dict(self):
        result = load_hyperparameter_list()
        assert isinstance(result, (dict, list))

    def test_not_empty(self):
        result = load_hyperparameter_list()
        assert len(result) > 0

    def test_entries_have_parameter_key(self):
        result = load_hyperparameter_list()
        if isinstance(result, list):
            assert all("parameter" in entry for entry in result)


class TestNamedMatrix:
    def test_named_tuple_fields(self):
        data = scipy.sparse.csr_matrix(np.eye(3))
        nm = NamedMatrix(data=data, rownames=["a", "b", "c"], colnames=["x", "y", "z"])
        assert nm.rownames == ["a", "b", "c"]
        assert nm.colnames == ["x", "y", "z"]
        assert nm.data.shape == (3, 3)


# ---------------------------------------------------------------------------
# Remote loader tests (mocked)
# ---------------------------------------------------------------------------

class TestLoadLrNetwork:
    def test_with_mocked_path(self, tmp_path):
        # Create a fake parquet
        df = pd.DataFrame({"from": ["A"], "to": ["B"], "source": ["s"], "database": ["d"]})
        path = tmp_path / "lr_network_mouse.parquet"
        df.to_parquet(path)

        with patch("nichenetr.datasets.resolve_data_path", return_value=path):
            result = load_lr_network(organism="mouse")
        assert isinstance(result, pd.DataFrame)
        assert "from" in result.columns

    def test_human_organism(self, tmp_path):
        df = pd.DataFrame({"from": ["X"], "to": ["Y"], "source": ["s"], "database": ["d"]})
        path = tmp_path / "lr_network_human.parquet"
        df.to_parquet(path)

        with patch("nichenetr.datasets.resolve_data_path", return_value=path):
            result = load_lr_network(organism="human")
        assert len(result) == 1


class TestLoadLigandTargetMatrix:
    def test_with_mocked_files(self, tmp_path):
        data = scipy.sparse.csr_matrix(np.array([[0.1, 0.2], [0.3, 0.4]]))
        npz_path = tmp_path / "ligand_target_matrix_mouse.npz"
        scipy.sparse.save_npz(npz_path, data)

        row_path = tmp_path / "ligand_target_matrix_mouse_rownames.json"
        col_path = tmp_path / "ligand_target_matrix_mouse_colnames.json"
        row_path.write_text(json.dumps(["gene1", "gene2"]))
        col_path.write_text(json.dumps(["lig1", "lig2"]))

        def mock_resolve(filename):
            return tmp_path / filename

        with patch("nichenetr.datasets.resolve_data_path", side_effect=mock_resolve):
            result = load_ligand_target_matrix(organism="mouse")
        assert isinstance(result, NamedMatrix)
        assert result.data.shape == (2, 2)
        assert result.rownames == ["gene1", "gene2"]
        assert result.colnames == ["lig1", "lig2"]


class TestLoadWeightedNetworks:
    def test_with_mocked_files(self, tmp_path):
        lr_sig = pd.DataFrame({"from": ["A"], "to": ["B"], "weight": [0.5]})
        gr = pd.DataFrame({"from": ["C"], "to": ["D"], "weight": [0.8]})
        lr_sig.to_parquet(tmp_path / "weighted_networks_mouse_lr_sig.parquet")
        gr.to_parquet(tmp_path / "weighted_networks_mouse_gr.parquet")

        def mock_resolve(filename):
            return tmp_path / filename

        with patch("nichenetr.datasets.resolve_data_path", side_effect=mock_resolve):
            result = load_weighted_networks(organism="mouse")
        assert "lr_sig" in result
        assert "gr" in result
        assert isinstance(result["lr_sig"], pd.DataFrame)


class TestLoadSigNetwork:
    def test_with_mocked_path(self, tmp_path):
        df = pd.DataFrame({"from": ["A"], "to": ["B"], "source": ["s"]})
        path = tmp_path / "sig_network_human.parquet"
        df.to_parquet(path)

        with patch("nichenetr.datasets.resolve_data_path", return_value=path):
            result = load_sig_network()
        assert isinstance(result, pd.DataFrame)


class TestLoadGrNetwork:
    def test_with_mocked_path(self, tmp_path):
        df = pd.DataFrame({"from": ["X"], "to": ["Y"], "source": ["s"]})
        path = tmp_path / "gr_network_human.parquet"
        df.to_parquet(path)

        with patch("nichenetr.datasets.resolve_data_path", return_value=path):
            result = load_gr_network()
        assert isinstance(result, pd.DataFrame)


class TestLoadLigandTfMatrix:
    def test_with_mocked_files(self, tmp_path):
        data = scipy.sparse.csr_matrix(np.eye(2))
        scipy.sparse.save_npz(tmp_path / "ligand_tf_matrix.npz", data)
        (tmp_path / "ligand_tf_matrix_rownames.json").write_text('["TF1","TF2"]')
        (tmp_path / "ligand_tf_matrix_colnames.json").write_text('["L1","L2"]')

        with patch("nichenetr.datasets.resolve_data_path", side_effect=lambda f: tmp_path / f):
            result = load_ligand_tf_matrix()
        assert isinstance(result, NamedMatrix)
        assert result.rownames == ["TF1", "TF2"]


class TestLoadSeuratObj:
    def test_with_mocked_path(self, tmp_path):
        import anndata
        adata = anndata.AnnData(X=np.zeros((3, 3)))
        path = tmp_path / "seuratObj.h5ad"
        adata.write_h5ad(path)

        with patch("nichenetr.datasets.resolve_data_path", return_value=path):
            result = load_seurat_obj()
        assert isinstance(result, anndata.AnnData)
        assert result.shape == (3, 3)


class TestLoadHnsccExpression:
    def test_with_mocked_files(self, tmp_path):
        data = scipy.sparse.csr_matrix(np.eye(3))
        scipy.sparse.save_npz(tmp_path / "hnscc_expression.npz", data)
        (tmp_path / "hnscc_expression_rownames.json").write_text('["c1","c2","c3"]')
        (tmp_path / "hnscc_expression_colnames.json").write_text('["g1","g2","g3"]')
        sample_info = pd.DataFrame({"cell": ["c1", "c2", "c3"], "type": ["A", "B", "C"]})
        sample_info.to_parquet(tmp_path / "hnscc_sample_info.parquet")

        with patch("nichenetr.datasets.resolve_data_path", side_effect=lambda f: tmp_path / f):
            result = load_hnscc_expression()
        assert "expression" in result
        assert "sample_info" in result
        assert isinstance(result["expression"], NamedMatrix)
        assert isinstance(result["sample_info"], pd.DataFrame)


class TestLoadPemtSignature:
    def test_with_mocked_path(self, tmp_path):
        path = tmp_path / "pemt_signature.txt"
        path.write_text("GENE1\nGENE2\nGENE3\n")

        with patch("nichenetr.datasets.resolve_data_path", return_value=path):
            result = load_pemt_signature()
        assert result == ["GENE1", "GENE2", "GENE3"]

    def test_empty_lines_skipped(self, tmp_path):
        path = tmp_path / "pemt_signature.txt"
        path.write_text("GENE1\n\nGENE2\n  \nGENE3\n")

        with patch("nichenetr.datasets.resolve_data_path", return_value=path):
            result = load_pemt_signature()
        assert len(result) == 3


class TestLoadNamedMatrix:
    def test_converts_non_csr_to_csr(self, tmp_path):
        """If loaded data is COO, it should be converted to CSR."""
        data = scipy.sparse.coo_matrix(np.array([[1, 0], [0, 2]]))
        scipy.sparse.save_npz(tmp_path / "test.npz", data)
        (tmp_path / "test_rownames.json").write_text('["r1","r2"]')
        (tmp_path / "test_colnames.json").write_text('["c1","c2"]')

        with patch("nichenetr.datasets.resolve_data_path", side_effect=lambda f: tmp_path / f):
            result = _load_named_matrix("test")
        assert isinstance(result.data, scipy.sparse.csr_matrix)
