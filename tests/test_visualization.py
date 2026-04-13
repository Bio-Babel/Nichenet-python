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
    make_circos_plot,
    make_circos_lr,
    prepare_circos_visualization,
    _matrix_to_dataframe,
    _resolve_legend_loc,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestMatrixToDataframe:
    def test_dataframe_passthrough(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _matrix_to_dataframe(df)
        assert isinstance(result, pd.DataFrame)

    def test_numpy_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = _matrix_to_dataframe(arr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_named_array(self):
        from nichenetr.targets import _attach_names
        arr = _attach_names(np.array([[1, 2], [3, 4]]), ["r1", "r2"], ["c1", "c2"])
        result = _matrix_to_dataframe(arr)
        assert list(result.index) == ["r1", "r2"]
        assert list(result.columns) == ["c1", "c2"]

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            _matrix_to_dataframe(np.array([1, 2, 3]))


class TestResolveLegendLoc:
    def test_known_positions(self):
        assert _resolve_legend_loc("top") == "upper center"
        assert _resolve_legend_loc("bottom") == "lower center"
        assert _resolve_legend_loc("left") == "center left"
        assert _resolve_legend_loc("right") == "center right"
        assert _resolve_legend_loc("none") == ""

    def test_unknown_falls_back(self):
        assert _resolve_legend_loc("whatever") == "best"


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

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

    def test_with_axes(self):
        fig, ax = plt.subplots()
        matrix = pd.DataFrame(np.random.rand(2, 2))
        result = make_heatmap_ggplot(matrix, ax=ax)
        assert result is fig

    def test_x_axis_bottom(self):
        matrix = pd.DataFrame(np.random.rand(2, 2))
        fig = make_heatmap_ggplot(matrix, x_axis_position="bottom")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_axes(self):
        matrix = pd.DataFrame(np.random.rand(2, 2))
        fig = make_heatmap_ggplot(matrix, x_axis=False, y_axis=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_legend_none(self):
        matrix = pd.DataFrame(np.random.rand(2, 2))
        fig = make_heatmap_ggplot(matrix, legend_position="none")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_invalid_x_axis_position(self):
        with pytest.raises(ValueError, match="x_axis_position"):
            make_heatmap_ggplot(np.random.rand(2, 2), x_axis_position="middle")

    def test_invalid_legend_position(self):
        with pytest.raises(ValueError, match="legend_position"):
            make_heatmap_ggplot(np.random.rand(2, 2), legend_position="center")


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
        matrix = pd.DataFrame(np.random.randn(3, 2))
        fig = make_threecolor_heatmap_ggplot(
            matrix, low_color="green", mid_color="white", high_color="purple"
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_axes(self):
        fig, ax = plt.subplots()
        matrix = pd.DataFrame(np.random.randn(2, 2))
        result = make_threecolor_heatmap_ggplot(matrix, ax=ax)
        assert result is fig

    def test_no_axes(self):
        matrix = pd.DataFrame(np.random.randn(2, 2))
        fig = make_threecolor_heatmap_ggplot(matrix, x_axis=False, y_axis=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_all_positive_values(self):
        """All positive with mid=0 triggers vmin adjustment."""
        matrix = pd.DataFrame(np.array([[1, 2], [3, 4]]).astype(float))
        fig = make_threecolor_heatmap_ggplot(matrix, mid=0.0)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_all_negative_values(self):
        """All negative with mid=0 triggers vmax adjustment."""
        matrix = pd.DataFrame(np.array([[-1, -2], [-3, -4]]).astype(float))
        fig = make_threecolor_heatmap_ggplot(matrix, mid=0.0)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_x_axis_bottom(self):
        matrix = pd.DataFrame(np.random.randn(2, 2))
        fig = make_threecolor_heatmap_ggplot(matrix, x_axis_position="bottom")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_legend_none(self):
        matrix = pd.DataFrame(np.random.randn(2, 2))
        fig = make_threecolor_heatmap_ggplot(matrix, legend_position="none")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_invalid_x_axis_position(self):
        with pytest.raises(ValueError, match="x_axis_position"):
            make_threecolor_heatmap_ggplot(np.random.randn(2, 2), x_axis_position="middle")


# ---------------------------------------------------------------------------
# Line plot
# ---------------------------------------------------------------------------

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

    def test_missing_column_raises(self):
        df = pd.DataFrame({"wrong": [1]})
        with pytest.raises(ValueError, match="not found"):
            make_line_plot(df)

    def test_missing_score_column_raises(self):
        df = pd.DataFrame({"test_ligand": ["L1"], "wrong": [1]})
        with pytest.raises(ValueError, match="not found"):
            make_line_plot(df)

    def test_custom_ranking_range(self):
        df = pd.DataFrame({
            "test_ligand": [f"L{i}" for i in range(30)],
            "aupr_corrected": np.random.rand(30),
        })
        fig = make_line_plot(df, ranking_range=(5, 15))
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# Mushroom plot
# ---------------------------------------------------------------------------

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

    def test_show_rankings(self):
        n = 5
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
        fig = make_mushroom_plot(df, top_n=5, show_rankings=True)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_pct_expressed_size(self):
        n = 5
        df = pd.DataFrame({
            "sender": ["TypeA"] * n,
            "ligand": [f"L{i}" for i in range(n)],
            "receptor": [f"R{i}" for i in range(n)],
            "pct_expressed_sender": np.random.rand(n),
            "pct_expressed_receiver": np.random.rand(n),
            "scaled_p_val_adapted_ligand": np.random.rand(n),
            "scaled_p_val_adapted_receptor": np.random.rand(n),
            "prioritization_score": np.random.rand(n),
        })
        fig = make_mushroom_plot(df, top_n=5, size="pct_expressed")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_pct_expressed_color(self):
        n = 5
        df = pd.DataFrame({
            "sender": ["TypeA"] * n,
            "ligand": [f"L{i}" for i in range(n)],
            "receptor": [f"R{i}" for i in range(n)],
            "scaled_avg_exprs_ligand": np.random.rand(n),
            "scaled_avg_exprs_receptor": np.random.rand(n),
            "pct_expressed_sender": np.random.rand(n),
            "pct_expressed_receiver": np.random.rand(n),
            "prioritization_score": np.random.rand(n),
        })
        fig = make_mushroom_plot(df, top_n=5, color="pct_expressed")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_false_color_range(self):
        n = 5
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
        fig = make_mushroom_plot(df, top_n=5, true_color_range=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"sender": ["A"], "ligand": ["L1"]})
        with pytest.raises(ValueError, match="Missing columns"):
            make_mushroom_plot(df)

    def test_missing_priority_score_raises(self):
        n = 3
        df = pd.DataFrame({
            "sender": ["A"] * n,
            "ligand": [f"L{i}" for i in range(n)],
            "receptor": [f"R{i}" for i in range(n)],
            "scaled_avg_exprs_ligand": np.random.rand(n),
            "scaled_avg_exprs_receptor": np.random.rand(n),
            "scaled_p_val_adapted_ligand": np.random.rand(n),
            "scaled_p_val_adapted_receptor": np.random.rand(n),
        })
        with pytest.raises(ValueError, match="prioritization_score"):
            make_mushroom_plot(df)

    def test_multiple_senders(self):
        n = 10
        df = pd.DataFrame({
            "sender": ["TypeA"] * 5 + ["TypeB"] * 5,
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


# ---------------------------------------------------------------------------
# Circos preparation and rendering
# ---------------------------------------------------------------------------

class TestPrepareCircosVisualization:
    def test_basic(self, circos_links):
        result = prepare_circos_visualization(circos_links)
        assert isinstance(result, dict)
        assert "links_circle" in result
        assert "ligand_colors" in result
        assert "order" in result
        assert "gaps" in result

    def test_custom_colors(self, circos_links):
        result = prepare_circos_visualization(
            circos_links,
            ligand_colors={"TypeA": "#FF0000", "TypeB": "#00FF00", "General": "#0000FF"},
            target_colors={"TypeC": "#888888"},
        )
        assert result["ligand_colors"] is not None

    def test_custom_widths(self, circos_links):
        widths = {
            "width_same_cell_same_ligand_type": 1.0,
            "width_different_cell": 10,
            "width_ligand_target": 20,
            "width_same_cell_same_target_type": 1.0,
        }
        result = prepare_circos_visualization(circos_links, widths=widths)
        assert len(result["gaps"]) > 0

    def test_celltype_order(self, circos_links):
        result = prepare_circos_visualization(
            circos_links,
            celltype_order=["General", "TypeA", "TypeB"],
        )
        assert isinstance(result["order"], list)

    def test_missing_columns_raises(self):
        bad_df = pd.DataFrame({"ligand": ["L1"]})
        with pytest.raises(ValueError, match="must have columns"):
            prepare_circos_visualization(bad_df)

    def test_missing_ligand_color_raises(self, circos_links):
        with pytest.raises(ValueError, match="ligand_colors must contain"):
            prepare_circos_visualization(
                circos_links,
                ligand_colors={"TypeA": "#FF0000"},  # Missing TypeB, General
                target_colors={"TypeC": "#888888"},
            )

    def test_missing_celltype_order_raises(self, circos_links):
        with pytest.raises(ValueError, match="celltype_order must contain"):
            prepare_circos_visualization(
                circos_links,
                celltype_order=["TypeA"],  # Missing TypeB, General
            )

    def test_missing_widths_raises(self, circos_links):
        with pytest.raises(ValueError, match="widths must contain"):
            prepare_circos_visualization(circos_links, widths={"bad": 1})


class TestMakeCircosPlot:
    def test_basic(self, circos_links):
        vis = prepare_circos_visualization(circos_links)
        fig = make_circos_plot(vis)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_transparency(self, circos_links):
        vis = prepare_circos_visualization(circos_links)
        fig = make_circos_plot(vis, transparency=True)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="must contain keys"):
            make_circos_plot({"links_circle": pd.DataFrame()})


class TestMakeCircosLr:
    def test_basic(self, prioritization_table):
        fig = make_circos_lr(
            prioritization_table,
            colors_sender={"TypeA": "#FF0000", "TypeB": "#0000FF"},
            colors_receiver={"TypeC": "#00FF00"},
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_cutoff(self, prioritization_table):
        fig = make_circos_lr(
            prioritization_table,
            colors_sender={"TypeA": "#FF0000", "TypeB": "#0000FF"},
            colors_receiver={"TypeC": "#00FF00"},
            cutoff=0.5,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_scale(self, prioritization_table):
        fig = make_circos_lr(
            prioritization_table,
            colors_sender={"TypeA": "#FF0000", "TypeB": "#0000FF"},
            colors_receiver={"TypeC": "#00FF00"},
            scale=True,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"sender": ["A"]})
        with pytest.raises(ValueError, match="must contain columns"):
            make_circos_lr(df, colors_sender={}, colors_receiver={})


class TestCircosPycirclize:
    """Test the pycirclize backend directly."""

    def test_direct_call(self, circos_links):
        from nichenetr.visualization import _circos_pycirclize
        vis = prepare_circos_visualization(circos_links)
        fig = _circos_pycirclize(
            vis["links_circle"],
            vis["ligand_colors"],
            vis["order"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_order(self, circos_links):
        from nichenetr.visualization import _circos_pycirclize
        vis = prepare_circos_visualization(circos_links)
        fig = _circos_pycirclize(
            vis["links_circle"],
            vis["ligand_colors"],
            [],  # empty order triggers fallback to sectors_in_links
        )
        assert isinstance(fig, matplotlib.figure.Figure)


class TestCircosFallback:
    """Test fallback renderer when pycirclize is not available."""

    def test_fallback_lt(self, circos_links):
        from nichenetr.visualization import _circos_fallback
        vis = prepare_circos_visualization(circos_links)
        fig = _circos_fallback(
            vis["links_circle"],
            vis["ligand_colors"],
            vis["order"],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_fallback_with_transparency(self, circos_links):
        from nichenetr.visualization import _circos_fallback
        vis = prepare_circos_visualization(circos_links)
        fig = _circos_fallback(
            vis["links_circle"],
            vis["ligand_colors"],
            vis["order"],
            transparency=True,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_fallback_with_axes(self, circos_links):
        from nichenetr.visualization import _circos_fallback
        vis = prepare_circos_visualization(circos_links)
        fig_pre, ax = plt.subplots()
        fig = _circos_fallback(
            vis["links_circle"],
            vis["ligand_colors"],
            vis["order"],
            ax=ax,
        )
        assert fig is fig_pre

    def test_fallback_empty_order(self, circos_links):
        from nichenetr.visualization import _circos_fallback
        vis = prepare_circos_visualization(circos_links)
        fig = _circos_fallback(
            vis["links_circle"],
            vis["ligand_colors"],
            [],  # empty order
        )
        assert isinstance(fig, matplotlib.figure.Figure)


class TestHeatmapWithoutSeaborn:
    """Test imshow fallback when seaborn is temporarily patched away."""

    def test_heatmap_without_seaborn(self):
        import nichenetr.visualization as viz
        old = viz._HAS_SEABORN
        try:
            viz._HAS_SEABORN = False
            matrix = pd.DataFrame(np.random.rand(3, 3), index=["a", "b", "c"], columns=["x", "y", "z"])
            fig = make_heatmap_ggplot(matrix)
            assert isinstance(fig, matplotlib.figure.Figure)

            fig2 = make_heatmap_ggplot(matrix, legend_position="none")
            assert isinstance(fig2, matplotlib.figure.Figure)
        finally:
            viz._HAS_SEABORN = old

    def test_threecolor_without_seaborn(self):
        import nichenetr.visualization as viz
        old = viz._HAS_SEABORN
        try:
            viz._HAS_SEABORN = False
            matrix = pd.DataFrame(np.random.randn(3, 3), index=["a", "b", "c"], columns=["x", "y", "z"])
            fig = make_threecolor_heatmap_ggplot(matrix)
            assert isinstance(fig, matplotlib.figure.Figure)

            fig2 = make_threecolor_heatmap_ggplot(matrix, legend_position="none")
            assert isinstance(fig2, matplotlib.figure.Figure)
        finally:
            viz._HAS_SEABORN = old


class TestMakeCircosLrDuplicates:
    """Test circos LR with duplicate ligand/receptor names across senders."""

    def test_duplicate_ligand_names(self):
        df = pd.DataFrame({
            "sender": ["TypeA", "TypeA", "TypeB", "TypeB"],
            "receiver": ["TypeC", "TypeC", "TypeC", "TypeC"],
            "ligand": ["L1", "L1", "L1", "L2"],
            "receptor": ["R1", "R2", "R1", "R2"],
            "prioritization_score": [0.9, 0.8, 0.7, 0.6],
        })
        fig = make_circos_lr(
            df,
            colors_sender={"TypeA": "#FF0000", "TypeB": "#0000FF"},
            colors_receiver={"TypeC": "#00FF00"},
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_overlapping_ligand_receptor_names(self):
        """Edge case: same name appears as both ligand and receptor."""
        df = pd.DataFrame({
            "sender": ["TypeA", "TypeA"],
            "receiver": ["TypeB", "TypeB"],
            "ligand": ["SHARED", "L2"],
            "receptor": ["SHARED", "R2"],
            "prioritization_score": [0.9, 0.8],
        })
        fig = make_circos_lr(
            df,
            colors_sender={"TypeA": "#FF0000"},
            colors_receiver={"TypeB": "#0000FF"},
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_duplicate_receptor_multiple_receivers(self):
        """Receptor R1 appears with two different receivers -> disambiguation."""
        df = pd.DataFrame({
            "sender": ["TypeA", "TypeA", "TypeA", "TypeA"],
            "receiver": ["TypeB", "TypeB", "TypeC", "TypeC"],
            "ligand": ["L1", "L2", "L1", "L2"],
            "receptor": ["R1", "R2", "R1", "R2"],
            "prioritization_score": [0.9, 0.8, 0.7, 0.6],
        })
        fig = make_circos_lr(
            df,
            colors_sender={"TypeA": "#FF0000"},
            colors_receiver={"TypeB": "#0000FF", "TypeC": "#00FF00"},
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_explicit_transparency(self):
        df = pd.DataFrame({
            "sender": ["TypeA", "TypeA"],
            "receiver": ["TypeB", "TypeB"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "prioritization_score": [0.9, 0.8],
        })
        fig = make_circos_lr(
            df,
            colors_sender={"TypeA": "#FF0000"},
            colors_receiver={"TypeB": "#0000FF"},
            transparency=[0.2, 0.5],
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_equal_weights_transparency(self):
        """When all weights are equal, transparency branch changes."""
        df = pd.DataFrame({
            "sender": ["TypeA", "TypeA"],
            "receiver": ["TypeB", "TypeB"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "prioritization_score": [0.5, 0.5],
        })
        fig = make_circos_lr(
            df,
            colors_sender={"TypeA": "#FF0000"},
            colors_receiver={"TypeB": "#0000FF"},
        )
        assert isinstance(fig, matplotlib.figure.Figure)
