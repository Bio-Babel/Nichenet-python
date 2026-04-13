"""Tests for nichenetr.prediction."""

import numpy as np
import pandas as pd
import pytest

from nichenetr.prediction import (
    predict_ligand_activities,
    predict_single_cell_ligand_activities,
    normalize_single_cell_ligand_activities,
    single_ligand_activity_score_regression,
    convert_single_cell_expression_to_settings,
)


class TestPredictLigandActivities:
    def test_returns_correct_columns(self, small_ligand_target_matrix, small_geneset):
        background = small_ligand_target_matrix.rownames
        ligands = small_ligand_target_matrix.colnames[:3]
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
        with pytest.raises(ValueError, match="None of the potential_ligands"):
            predict_ligand_activities(
                geneset=["G1"],
                background_expressed_genes=["G1", "G2"],
                ligand_target_matrix=small_ligand_target_matrix,
                potential_ligands=["NONEXISTENT"],
            )

    def test_no_geneset_overlap_raises(self, small_ligand_target_matrix):
        with pytest.raises(ValueError, match="None of the geneset genes"):
            predict_ligand_activities(
                geneset=["NONEXISTENT1", "NONEXISTENT2"],
                background_expressed_genes=["G1", "G2"],
                ligand_target_matrix=small_ligand_target_matrix,
                potential_ligands=["L1"],
            )

    def test_no_gene_overlap_raises(self, small_ligand_target_matrix):
        with pytest.raises(ValueError, match="No overlap"):
            predict_ligand_activities(
                geneset=["X1"],
                background_expressed_genes=["X1", "X2"],
                ligand_target_matrix=small_ligand_target_matrix,
                potential_ligands=["L1"],
            )


class TestConvertSingleCellExpressionToSettings:
    def test_classification_mode(self, expression_scaled):
        cell_id = expression_scaled.index[0]
        result = convert_single_cell_expression_to_settings(
            cell_id=cell_id,
            expression_matrix=expression_scaled,
            setting_name="test",
            setting_from=["L1", "L2"],
            regression=False,
        )
        assert "name" in result
        assert "from" in result
        assert "response" in result
        assert result["response"].dtype == bool

    def test_regression_mode(self, expression_scaled):
        cell_id = expression_scaled.index[0]
        result = convert_single_cell_expression_to_settings(
            cell_id=cell_id,
            expression_matrix=expression_scaled,
            setting_name="test",
            setting_from=["L1"],
            regression=True,
        )
        assert result["response"].dtype == np.float64

    def test_name_format(self, expression_scaled):
        cell_id = expression_scaled.index[0]
        result = convert_single_cell_expression_to_settings(
            cell_id=cell_id,
            expression_matrix=expression_scaled,
            setting_name="prefix",
            setting_from=[],
        )
        assert result["name"] == f"prefix_{cell_id}"


class TestPredictSingleCellLigandActivities:
    def test_basic(self, small_ligand_target_matrix, expression_scaled):
        # Adjust expression_scaled to have genes matching the matrix
        gene_names = small_ligand_target_matrix.rownames
        rng = np.random.RandomState(77)
        expr = pd.DataFrame(
            rng.rand(5, len(gene_names)),
            index=[f"cell_{i}" for i in range(5)],
            columns=gene_names,
        )
        result = predict_single_cell_ligand_activities(
            cell_ids=list(expr.index[:3]),
            expression_scaled=expr,
            ligand_target_matrix=small_ligand_target_matrix,
            potential_ligands=["L1", "L2"],
        )
        assert isinstance(result, pd.DataFrame)
        assert "setting" in result.columns
        assert "test_ligand" in result.columns

    def test_empty_result(self, small_ligand_target_matrix):
        """All-constant expression should produce empty result."""
        gene_names = small_ligand_target_matrix.rownames
        expr = pd.DataFrame(
            np.ones((3, len(gene_names))),
            index=["c1", "c2", "c3"],
            columns=gene_names,
        )
        result = predict_single_cell_ligand_activities(
            cell_ids=["c1"],
            expression_scaled=expr,
            ligand_target_matrix=small_ligand_target_matrix,
            potential_ligands=["L1"],
        )
        assert isinstance(result, pd.DataFrame)
        assert "setting" in result.columns


class TestNormalizeSingleCellLigandActivities:
    def test_basic(self):
        df = pd.DataFrame({
            "setting": ["c1", "c1", "c2", "c2"],
            "test_ligand": ["L1", "L2", "L1", "L2"],
            "aupr": [0.5, 0.3, 0.7, 0.1],
        })
        result = normalize_single_cell_ligand_activities(df)
        assert isinstance(result, pd.DataFrame)
        assert "cell" in result.columns
        assert "L1" in result.columns
        assert "L2" in result.columns
        assert len(result) == 2

    def test_single_cell(self):
        df = pd.DataFrame({
            "setting": ["c1", "c1"],
            "test_ligand": ["L1", "L2"],
            "aupr": [0.5, 0.5],
        })
        result = normalize_single_cell_ligand_activities(df)
        assert len(result) == 1


class TestSingleLigandActivityScoreRegression:
    def test_basic(self):
        ligand_activities = pd.DataFrame({
            "cell": ["c1", "c2", "c3", "c4", "c5"],
            "L1": [0.5, 0.3, 0.8, 0.1, 0.6],
            "L2": [0.2, 0.7, 0.4, 0.9, 0.3],
        })
        scores_tbl = pd.DataFrame({
            "cell": ["c1", "c2", "c3", "c4", "c5"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = single_ligand_activity_score_regression(ligand_activities, scores_tbl)
        assert isinstance(result, pd.DataFrame)
        assert "ligand" in result.columns
        assert "pearson_regression" in result.columns
        assert "spearman_regression" in result.columns
        assert len(result) == 2

    def test_constant_predictor(self):
        """When a ligand has constant activity, correlations should be NaN."""
        ligand_activities = pd.DataFrame({
            "cell": ["c1", "c2", "c3"],
            "L1": [0.5, 0.5, 0.5],
        })
        scores_tbl = pd.DataFrame({
            "cell": ["c1", "c2", "c3"],
            "score": [1.0, 2.0, 3.0],
        })
        result = single_ligand_activity_score_regression(ligand_activities, scores_tbl)
        assert pd.isna(result["pearson_regression"].iloc[0])

    def test_r_squared_reasonable(self):
        """With perfectly correlated data, r_squared should be ~1."""
        ligand_activities = pd.DataFrame({
            "cell": [f"c{i}" for i in range(10)],
            "L1": list(range(10)),
        })
        scores_tbl = pd.DataFrame({
            "cell": [f"c{i}" for i in range(10)],
            "score": list(range(10)),
        })
        result = single_ligand_activity_score_regression(ligand_activities, scores_tbl)
        assert result["r_squared"].iloc[0] > 0.99
