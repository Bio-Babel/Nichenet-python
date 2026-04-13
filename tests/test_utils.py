"""Tests for nichenetr.utils."""

import numpy as np
import pandas as pd
import pytest

from nichenetr.utils import (
    mapper,
    scale_quantile,
    scale_quantile_adapted,
    scaling_modified_zscore,
    scaling_zscore,
)


# ---------------------------------------------------------------------------
# scaling_zscore
# ---------------------------------------------------------------------------

class TestScalingZscore:
    def test_known_output(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scaling_zscore(x)
        # mean=3, std(ddof=1)=sqrt(2.5)
        expected = (x - 3.0) / np.std(x, ddof=1)
        np.testing.assert_allclose(result, expected)

    def test_single_value_returns_zero(self):
        result = scaling_zscore(np.array([42.0]))
        assert result.shape == (1,) or result.shape == ()
        np.testing.assert_allclose(result, 0.0)

    def test_zero_std(self):
        x = np.array([5.0, 5.0, 5.0])
        result = scaling_zscore(x)
        # std==0 so result = x - mean = 0
        np.testing.assert_allclose(result, np.zeros(3))

    def test_preserves_shape(self):
        x = np.array([10.0, 20.0, 30.0])
        assert scaling_zscore(x).shape == x.shape

    def test_list_input(self):
        result = scaling_zscore([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# scaling_modified_zscore
# ---------------------------------------------------------------------------

class TestScalingModifiedZscore:
    def test_known_output(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scaling_modified_zscore(x)
        median = 3.0
        mad = np.median(np.abs(x - median)) * 1.4826
        expected = 0.6745 * (x - median) / mad
        np.testing.assert_allclose(result, expected)

    def test_zero_mad(self):
        x = np.array([3.0, 3.0, 3.0])
        result = scaling_modified_zscore(x)
        # mad==0, so result = 0.6745 * (x - median) = 0
        np.testing.assert_allclose(result, np.zeros(3))


# ---------------------------------------------------------------------------
# scale_quantile
# ---------------------------------------------------------------------------

class TestScaleQuantile:
    def test_range_zero_one(self):
        rng = np.random.RandomState(0)
        x = rng.randn(100)
        result = scale_quantile(x)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_known_output(self):
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = scale_quantile(x, outlier_cutoff=0.0)
        # With cutoff=0, q_low=0, q_high=1, so linear rescale is identity
        np.testing.assert_allclose(result, x)

    def test_2d(self):
        x = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        result = scale_quantile(x, outlier_cutoff=0.0)
        assert result.shape == (3, 2)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# scale_quantile_adapted
# ---------------------------------------------------------------------------

class TestScaleQuantileAdapted:
    def test_pseudovalue_added(self):
        x = np.array([0.0, 0.5, 1.0])
        result = scale_quantile_adapted(x, outlier_cutoff=0.0)
        expected = scale_quantile(x, outlier_cutoff=0.0) + 0.001
        np.testing.assert_allclose(result, expected)

    def test_minimum_is_above_zero(self):
        rng = np.random.RandomState(1)
        x = rng.rand(50)
        result = scale_quantile_adapted(x)
        assert result.min() >= 0.001


# ---------------------------------------------------------------------------
# mapper
# ---------------------------------------------------------------------------

class TestMapper:
    def test_basic(self):
        df = pd.DataFrame({"name": ["a", "b", "c"], "value": [1, 2, 3]})
        result = mapper(df, "value", "name")
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_duplicate_keys_last_wins(self):
        df = pd.DataFrame({"name": ["a", "a"], "value": [1, 2]})
        result = mapper(df, "value", "name")
        # dict(zip(...)) keeps last value for duplicate keys
        assert result == {"a": 2}
