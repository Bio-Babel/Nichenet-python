"""Tests for nichenetr.targets."""

import numpy as np
import pandas as pd
import pytest

from nichenetr.targets import (
    get_weighted_ligand_target_links,
    prepare_ligand_target_visualization,
    get_weighted_ligand_receptor_links,
)


class TestGetWeightedLigandTargetLinks:
    def test_output_structure(self, small_ligand_target_matrix, small_geneset):
        result = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"ligand", "target", "weight"}

    def test_ligand_column_value(self, small_ligand_target_matrix, small_geneset):
        result = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        # All rows should have ligand == "L1"
        assert (result["ligand"] == "L1").all()

    def test_missing_ligand_returns_nan(self, small_ligand_target_matrix, small_geneset):
        result = get_weighted_ligand_target_links(
            ligand_oi="NONEXISTENT",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert len(result) == 1
        assert pd.isna(result["target"].iloc[0])

    def test_targets_in_geneset(self, small_ligand_target_matrix, small_geneset):
        result = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        non_nan_targets = result.dropna(subset=["target"])
        if len(non_nan_targets) > 0:
            assert set(non_nan_targets["target"]).issubset(set(small_geneset))


class TestPrepareLigandTargetVisualization:
    def test_matrix_shape(self, small_ligand_target_matrix, small_geneset):
        # First get the links
        links = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        result = prepare_ligand_target_visualization(
            ligand_target_df=links,
            ligand_target_matrix=small_ligand_target_matrix,
            cutoff=0.0,
        )
        assert result.ndim == 2
        # Result should have rownames and colnames attributes
        assert hasattr(result, "rownames")
        assert hasattr(result, "colnames")

    def test_empty_input(self, small_ligand_target_matrix):
        empty_df = pd.DataFrame(
            {"ligand": [np.nan], "target": [np.nan], "weight": [np.nan]}
        )
        result = prepare_ligand_target_visualization(
            ligand_target_df=empty_df,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert result.shape == (0, 0)


class TestGetWeightedLigandReceptorLinks:
    def test_output_structure(self):
        lr_network = pd.DataFrame({
            "from": ["L1", "L2", "L3"],
            "to": ["R1", "R2", "R3"],
        })
        weighted_lr_sig = pd.DataFrame({
            "from": ["L1", "L2", "L3", "L1"],
            "to": ["R1", "R2", "R3", "R2"],
            "weight": [0.5, 0.8, 0.3, 0.6],
        })
        result = get_weighted_ligand_receptor_links(
            best_upstream_ligands=["L1", "L2"],
            expressed_receptors=["R1", "R2", "R3"],
            lr_network=lr_network,
            weighted_networks_lr_sig=weighted_lr_sig,
        )
        assert isinstance(result, pd.DataFrame)
        assert "from" in result.columns
        assert "to" in result.columns
        assert "weight" in result.columns

    def test_filters_to_requested_ligands(self):
        lr_network = pd.DataFrame({
            "from": ["L1", "L2", "L3"],
            "to": ["R1", "R2", "R3"],
        })
        weighted_lr_sig = pd.DataFrame({
            "from": ["L1", "L2", "L3"],
            "to": ["R1", "R2", "R3"],
            "weight": [0.5, 0.8, 0.3],
        })
        result = get_weighted_ligand_receptor_links(
            best_upstream_ligands=["L1"],
            expressed_receptors=["R1", "R2", "R3"],
            lr_network=lr_network,
            weighted_networks_lr_sig=weighted_lr_sig,
        )
        if len(result) > 0:
            assert set(result["from"].unique()).issubset({"L1"})
