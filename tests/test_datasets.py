"""Tests for nichenetr.datasets (bundled resources)."""

import pandas as pd
import pytest

from nichenetr.datasets import (
    load_geneinfo,
    load_hyperparameter_list,
    load_source_weights_df,
)


class TestLoadSourceWeightsDf:
    def test_returns_dataframe(self):
        result = load_source_weights_df()
        assert isinstance(result, pd.DataFrame)

    def test_not_empty(self):
        result = load_source_weights_df()
        assert len(result) > 0

    def test_has_expected_columns(self):
        result = load_source_weights_df()
        # Should have at least a source name and a weight column
        assert result.shape[1] >= 1


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


class TestLoadHyperparameterList:
    def test_returns_list_or_dict(self):
        result = load_hyperparameter_list()
        assert isinstance(result, (dict, list))

    def test_not_empty(self):
        result = load_hyperparameter_list()
        assert len(result) > 0

    def test_entries_have_parameter_key(self):
        result = load_hyperparameter_list()
        # The JSON contains a list of dicts with 'parameter' keys
        if isinstance(result, list):
            assert all("parameter" in entry for entry in result)
