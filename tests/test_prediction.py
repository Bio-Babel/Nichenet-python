"""Tests for nichenetr.prediction."""

import numpy as np
import pandas as pd
import pytest

from nichenetr.prediction import predict_ligand_activities


class TestPredictLigandActivities:
    def test_returns_correct_columns(self, small_ligand_target_matrix, small_geneset):
        background = small_ligand_target_matrix.rownames  # G1..G10
        ligands = small_ligand_target_matrix.colnames[:3]  # L1, L2, L3

        result = predict_ligand_activities(
            geneset=small_geneset,
            background_expressed_genes=background,
            ligand_target_matrix=small_ligand_target_matrix,
            potential_ligands=ligands,
        )

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "test_ligand", "auroc", "aupr", "aupr_corrected", "pearson"
        ]

    def test_one_row_per_ligand(self, small_ligand_target_matrix, small_geneset):
        background = small_ligand_target_matrix.rownames
        ligands = ["L1", "L2"]

        result = predict_ligand_activities(
            geneset=small_geneset,
            background_expressed_genes=background,
            ligand_target_matrix=small_ligand_target_matrix,
            potential_ligands=ligands,
        )

        assert len(result) == 2
        assert set(result["test_ligand"]) == {"L1", "L2"}

    def test_aupr_in_range(self, small_ligand_target_matrix, small_geneset):
        background = small_ligand_target_matrix.rownames
        ligands = small_ligand_target_matrix.colnames

        result = predict_ligand_activities(
            geneset=small_geneset,
            background_expressed_genes=background,
            ligand_target_matrix=small_ligand_target_matrix,
            potential_ligands=ligands,
        )

        # AUPR must be in [0, 1]
        valid_aupr = result["aupr"].dropna()
        assert (valid_aupr >= 0).all()
        assert (valid_aupr <= 1).all()

    def test_auroc_in_range(self, small_ligand_target_matrix, small_geneset):
        background = small_ligand_target_matrix.rownames
        ligands = small_ligand_target_matrix.colnames

        result = predict_ligand_activities(
            geneset=small_geneset,
            background_expressed_genes=background,
            ligand_target_matrix=small_ligand_target_matrix,
            potential_ligands=ligands,
        )

        valid_auroc = result["auroc"].dropna()
        assert (valid_auroc >= 0).all()
        assert (valid_auroc <= 1).all()

    def test_no_overlap_raises(self, small_ligand_target_matrix):
        result = None
        with pytest.raises(ValueError, match="None of the potential_ligands"):
            predict_ligand_activities(
                geneset=["G1"],
                background_expressed_genes=["G1", "G2"],
                ligand_target_matrix=small_ligand_target_matrix,
                potential_ligands=["NONEXISTENT"],
            )
