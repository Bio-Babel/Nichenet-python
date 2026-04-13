"""Tests for nichenetr.evaluation."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from nichenetr.datasets import NamedMatrix
from nichenetr.evaluation import (
    assess_rf_class_probabilities,
    classification_evaluation_continuous_pred_wrapper,
)


class TestClassificationEvaluationContinuousPredWrapper:
    def test_returns_correct_columns(self):
        df = pd.DataFrame({
            "prediction": [0.9, 0.8, 0.3, 0.1, 0.2, 0.7, 0.6, 0.05, 0.15, 0.4],
            "response":   [True, True, False, False, False, True, True, False, False, False],
        })
        result = classification_evaluation_continuous_pred_wrapper(df)
        expected_cols = {
            "auroc", "aupr", "aupr_corrected",
            "pearson", "pearson_log_pval",
            "spearman", "spearman_log_pval",
        }
        assert set(result.columns) == expected_cols

    def test_single_row_result(self):
        df = pd.DataFrame({
            "prediction": [0.9, 0.1, 0.8, 0.2],
            "response": [True, False, True, False],
        })
        result = classification_evaluation_continuous_pred_wrapper(df)
        assert len(result) == 1

    def test_auroc_in_range(self):
        df = pd.DataFrame({
            "prediction": [0.9, 0.8, 0.3, 0.1, 0.7, 0.05],
            "response": [True, True, False, False, True, False],
        })
        result = classification_evaluation_continuous_pred_wrapper(df)
        auroc = result["auroc"].iloc[0]
        assert 0 <= auroc <= 1

    def test_perfect_prediction(self):
        df = pd.DataFrame({
            "prediction": [1.0, 1.0, 0.0, 0.0],
            "response": [True, True, False, False],
        })
        result = classification_evaluation_continuous_pred_wrapper(df)
        assert result["auroc"].iloc[0] == 1.0

    def test_degenerate_constant_prediction(self):
        df = pd.DataFrame({
            "prediction": [0.5, 0.5, 0.5, 0.5],
            "response": [True, True, False, False],
        })
        result = classification_evaluation_continuous_pred_wrapper(df)
        # Constant prediction -> degenerate
        assert pd.isna(result["auroc"].iloc[0])


class TestAssessRfClassProbabilities:
    @pytest.fixture
    def rf_fixtures(self):
        """Build a small ligand-target matrix and gene sets for RF testing."""
        rng = np.random.RandomState(7)
        n_genes = 40
        n_ligands = 3
        genes = [f"gene{i}" for i in range(n_genes)]
        ligands = ["lig1", "lig2", "lig3"]
        data = rng.rand(n_genes, n_ligands)
        # Make geneset genes have higher values in column 0
        geneset_idx = list(range(0, 10))
        data[geneset_idx, 0] += 0.5
        data[data < 0.2] = 0.0
        sparse = scipy.sparse.csr_matrix(data)
        ltm = NamedMatrix(data=sparse, rownames=genes, colnames=ligands)
        geneset = [genes[i] for i in geneset_idx]
        background = genes
        return {
            "ltm": ltm,
            "geneset": geneset,
            "background": background,
            "ligands": ligands,
        }

    def test_output_columns(self, rf_fixtures):
        result = assess_rf_class_probabilities(
            round_num=1,
            folds=3,
            geneset=rf_fixtures["geneset"],
            background_expressed_genes=rf_fixtures["background"],
            ligands_oi=rf_fixtures["ligands"],
            ligand_target_matrix=rf_fixtures["ltm"],
        )
        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns
        assert "response" in result.columns
        assert "prediction" in result.columns

    def test_predictions_in_range(self, rf_fixtures):
        result = assess_rf_class_probabilities(
            round_num=1,
            folds=3,
            geneset=rf_fixtures["geneset"],
            background_expressed_genes=rf_fixtures["background"],
            ligands_oi=rf_fixtures["ligands"],
            ligand_target_matrix=rf_fixtures["ltm"],
        )
        assert (result["prediction"] >= 0).all()
        assert (result["prediction"] <= 1).all()
