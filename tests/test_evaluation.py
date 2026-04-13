"""Tests for nichenetr.evaluation."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from nichenetr.datasets import NamedMatrix
from nichenetr.evaluation import (
    assess_rf_class_probabilities,
    classification_evaluation_continuous_pred_wrapper,
    calculate_fraction_top_predicted,
    calculate_fraction_top_predicted_fisher,
    get_top_predicted_genes,
    convert_settings_ligand_prediction,
)


# ---------------------------------------------------------------------------
# classification_evaluation_continuous_pred_wrapper
# ---------------------------------------------------------------------------

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
        assert pd.isna(result["auroc"].iloc[0])

    def test_empty_input(self):
        df = pd.DataFrame({"prediction": [], "response": []})
        result = classification_evaluation_continuous_pred_wrapper(df)
        assert pd.isna(result["auroc"].iloc[0])

    def test_constant_response(self):
        df = pd.DataFrame({
            "prediction": [0.5, 0.8, 0.3],
            "response": [True, True, True],
        })
        result = classification_evaluation_continuous_pred_wrapper(df)
        assert pd.isna(result["auroc"].iloc[0])

    def test_inf_values_handled(self):
        df = pd.DataFrame({
            "prediction": [np.inf, 0.5, 0.3, 0.1],
            "response": [True, True, False, False],
        })
        result = classification_evaluation_continuous_pred_wrapper(df)
        assert not pd.isna(result["auroc"].iloc[0])


# ---------------------------------------------------------------------------
# assess_rf_class_probabilities
# ---------------------------------------------------------------------------

class TestAssessRfClassProbabilities:
    @pytest.fixture
    def rf_fixtures(self):
        rng = np.random.RandomState(7)
        n_genes = 40
        n_ligands = 3
        genes = [f"gene{i}" for i in range(n_genes)]
        ligands = ["lig1", "lig2", "lig3"]
        data = rng.rand(n_genes, n_ligands)
        geneset_idx = list(range(0, 10))
        data[geneset_idx, 0] += 0.5
        data[data < 0.2] = 0.0
        sparse = scipy.sparse.csr_matrix(data)
        ltm = NamedMatrix(data=sparse, rownames=genes, colnames=ligands)
        geneset = [genes[i] for i in geneset_idx]
        background = genes
        return {"ltm": ltm, "geneset": geneset, "background": background, "ligands": ligands}

    def test_output_columns(self, rf_fixtures):
        result = assess_rf_class_probabilities(
            round_num=1, folds=3,
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
            round_num=1, folds=3,
            geneset=rf_fixtures["geneset"],
            background_expressed_genes=rf_fixtures["background"],
            ligands_oi=rf_fixtures["ligands"],
            ligand_target_matrix=rf_fixtures["ltm"],
        )
        assert (result["prediction"] >= 0).all()
        assert (result["prediction"] <= 1).all()

    def test_reproducible_with_same_seed(self, rf_fixtures):
        r1 = assess_rf_class_probabilities(
            round_num=42, folds=2,
            geneset=rf_fixtures["geneset"],
            background_expressed_genes=rf_fixtures["background"],
            ligands_oi=rf_fixtures["ligands"],
            ligand_target_matrix=rf_fixtures["ltm"],
        )
        r2 = assess_rf_class_probabilities(
            round_num=42, folds=2,
            geneset=rf_fixtures["geneset"],
            background_expressed_genes=rf_fixtures["background"],
            ligands_oi=rf_fixtures["ligands"],
            ligand_target_matrix=rf_fixtures["ltm"],
        )
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# calculate_fraction_top_predicted
# ---------------------------------------------------------------------------

class TestCalculateFractionTopPredicted:
    def test_basic(self, rf_prediction_df, small_ligand_target_matrix):
        result = calculate_fraction_top_predicted(
            round_num=1,
            response_prediction_df=rf_prediction_df,
            ligands_oi=["L1"],
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert isinstance(result, pd.DataFrame)
        assert "true_target" in result.columns
        assert "n" in result.columns
        assert "positive_prediction" in result.columns
        assert "fraction_positive_predicted" in result.columns

    def test_fractions_in_range(self, rf_prediction_df, small_ligand_target_matrix):
        result = calculate_fraction_top_predicted(
            round_num=1,
            response_prediction_df=rf_prediction_df,
            ligands_oi=["L1"],
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert (result["fraction_positive_predicted"] >= 0).all()
        assert (result["fraction_positive_predicted"] <= 1).all()

    def test_custom_quantile(self, rf_prediction_df, small_ligand_target_matrix):
        result = calculate_fraction_top_predicted(
            round_num=1,
            response_prediction_df=rf_prediction_df,
            ligands_oi=["L1"],
            ligand_target_matrix=small_ligand_target_matrix,
            quantile_cutoff=0.80,
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# calculate_fraction_top_predicted_fisher
# ---------------------------------------------------------------------------

class TestCalculateFractionTopPredictedFisher:
    def test_returns_float(self, rf_prediction_df, small_ligand_target_matrix):
        result = calculate_fraction_top_predicted_fisher(
            round_num=1,
            response_prediction_df=rf_prediction_df,
            ligands_oi=["L1"],
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_returns_dict(self, rf_prediction_df, small_ligand_target_matrix):
        result = calculate_fraction_top_predicted_fisher(
            round_num=1,
            response_prediction_df=rf_prediction_df,
            ligands_oi=["L1"],
            ligand_target_matrix=small_ligand_target_matrix,
            p_value_output=False,
        )
        assert isinstance(result, dict)
        assert "oddsratio" in result
        assert "p_value" in result


# ---------------------------------------------------------------------------
# get_top_predicted_genes
# ---------------------------------------------------------------------------

class TestGetTopPredictedGenes:
    def test_basic(self, rf_prediction_df, small_ligand_target_matrix):
        result = get_top_predicted_genes(
            round_num=1,
            response_prediction_df=rf_prediction_df,
            ligands_oi=["L1"],
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns
        assert "true_target" in result.columns
        assert "predicted_top_target_round1" in result.columns

    def test_respects_n(self, rf_prediction_df, small_ligand_target_matrix):
        result = get_top_predicted_genes(
            round_num=1,
            response_prediction_df=rf_prediction_df,
            ligands_oi=["L1"],
            ligand_target_matrix=small_ligand_target_matrix,
            n=3,
        )
        assert len(result) <= 3

    def test_round_num_in_column_name(self, rf_prediction_df, small_ligand_target_matrix):
        result = get_top_predicted_genes(
            round_num=5,
            response_prediction_df=rf_prediction_df,
            ligands_oi=["L1"],
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert "predicted_top_target_round5" in result.columns


# ---------------------------------------------------------------------------
# convert_settings_ligand_prediction
# ---------------------------------------------------------------------------

class TestConvertSettingsLigandPrediction:
    def test_validation_single(self):
        settings = [
            {"name": "s1", "from": "LIG1", "response": pd.Series([True, False])},
        ]
        result = convert_settings_ligand_prediction(
            settings, all_ligands=["A", "B"], validation=True, single=True
        )
        assert len(result) == 2
        assert all("ligand" in r for r in result)
        assert result[0]["from"] == "A"
        assert result[1]["from"] == "B"

    def test_validation_not_single(self):
        settings = [
            {"name": "s1", "from": "LIG1", "response": pd.Series([True])},
        ]
        result = convert_settings_ligand_prediction(
            settings, all_ligands=["A", "B"], validation=True, single=False
        )
        assert len(result) == 1
        assert result[0]["from"] == ["A", "B"]

    def test_no_validation_single(self):
        settings = [
            {"name": "s1", "from": "LIG1", "response": pd.Series([True])},
        ]
        result = convert_settings_ligand_prediction(
            settings, all_ligands=["A"], validation=False, single=True
        )
        assert len(result) == 1
        assert "ligand" not in result[0]

    def test_no_validation_not_single(self):
        settings = [
            {"name": "s1", "from": "LIG1", "response": pd.Series([True])},
        ]
        result = convert_settings_ligand_prediction(
            settings, all_ligands=["A", "B"], validation=False, single=False
        )
        assert len(result) == 1
        assert result[0]["from"] == ["A", "B"]

    def test_multiple_from_collapsed(self):
        settings = [
            {"name": "s1", "from": ["L1", "L2"], "response": pd.Series([True])},
        ]
        result = convert_settings_ligand_prediction(
            settings, all_ligands=["A"], validation=True, single=True
        )
        assert result[0]["ligand"] == "L1-L2"

    def test_type_errors(self):
        with pytest.raises(TypeError, match="settings should be a list"):
            convert_settings_ligand_prediction("not_list", ["A"])
        with pytest.raises(TypeError, match="all_ligands should be a list"):
            convert_settings_ligand_prediction([], "not_list")
        with pytest.raises(TypeError, match="validation should be"):
            convert_settings_ligand_prediction([], [], validation="yes")
        with pytest.raises(TypeError, match="single should be"):
            convert_settings_ligand_prediction([], [], single="yes")
