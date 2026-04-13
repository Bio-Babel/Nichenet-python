"""Tests for nichenetr.visualization."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nichenetr.visualization import (
    make_heatmap_ggplot,
    make_threecolor_heatmap_ggplot,
    make_line_plot,
    make_mushroom_plot,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


class TestMakeHeatmapGgplot:
    def test_returns_figure(self):
        matrix = pd.DataFrame(
            np.random.rand(4, 3),
            index=["g1", "g2", "g3", "g4"],
            columns=["L1", "L2", "L3"],
        )
        fig = make_heatmap_ggplot(matrix)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_accepts_numpy_array(self):
        arr = np.random.rand(3, 3)
        fig = make_heatmap_ggplot(arr)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestMakeThreecolorHeatmapGgplot:
    def test_returns_figure(self):
        matrix = pd.DataFrame(
            np.random.randn(4, 3),
            index=["g1", "g2", "g3", "g4"],
            columns=["L1", "L2", "L3"],
        )
        fig = make_threecolor_heatmap_ggplot(matrix)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_colors(self):
        matrix = pd.DataFrame(
            np.random.randn(3, 2),
            index=["a", "b", "c"],
            columns=["x", "y"],
        )
        fig = make_threecolor_heatmap_ggplot(
            matrix, low_color="green", mid_color="white", high_color="purple"
        )
        assert isinstance(fig, matplotlib.figure.Figure)


class TestMakeLinePlot:
    def test_returns_figure(self):
        df = pd.DataFrame({
            "test_ligand": [f"L{i}" for i in range(20)],
            "aupr_corrected": np.random.rand(20),
        })
        fig = make_line_plot(df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_potential_ligands(self):
        df = pd.DataFrame({
            "test_ligand": [f"L{i}" for i in range(20)],
            "aupr_corrected": np.random.rand(20),
        })
        fig = make_line_plot(df, potential_ligands=["L0", "L1", "L2"])
        assert isinstance(fig, matplotlib.figure.Figure)


class TestMakeMushroomPlot:
    def test_returns_figure(self):
        n = 10
        df = pd.DataFrame({
            "sender": ["TypeA"] * n,
            "ligand": [f"L{i}" for i in range(n)],
            "receptor": [f"R{i}" for i in range(n)],
            "scaled_avg_exprs_ligand": np.random.rand(n),
            "scaled_avg_exprs_receptor": np.random.rand(n),
            "scaled_p_val_adapted_ligand": np.random.rand(n),
            "scaled_p_val_adapted_receptor": np.random.rand(n),
            "prioritization_score": np.random.rand(n),
        })
        fig = make_mushroom_plot(df, top_n=5)
        assert isinstance(fig, matplotlib.figure.Figure)
