"""Visualization helpers for NicheNet results.

Provides matplotlib/seaborn-based equivalents of the ggplot2 plotting functions
in the R *nichenetr* package, including heatmaps, mushroom dot-plots, line
ranking plots, and circos (chord) diagrams.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    _HAS_SEABORN = True
except ImportError:  # pragma: no cover
    _HAS_SEABORN = False

try:
    from pycirclize import Circos

    _HAS_PYCIRCLIZE = True
except ImportError:  # pragma: no cover
    _HAS_PYCIRCLIZE = False

__all__ = [
    "make_heatmap_ggplot",
    "make_threecolor_heatmap_ggplot",
    "make_line_plot",
    "make_mushroom_plot",
    "make_circos_plot",
    "make_circos_lr",
    "prepare_circos_visualization",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _matrix_to_dataframe(
    matrix_oi: Any,
) -> pd.DataFrame:
    """Convert a matrix-like object to a pandas DataFrame.

    Supports ``numpy.ndarray`` (including ``_NamedArray`` with ``.rownames``
    and ``.colnames`` attributes), ``pandas.DataFrame``, and any 2-D
    array-like.

    Parameters
    ----------
    matrix_oi : array-like or DataFrame
        Input matrix.

    Returns
    -------
    pandas.DataFrame
        DataFrame representation of *matrix_oi*.
    """
    if isinstance(matrix_oi, pd.DataFrame):
        return matrix_oi.copy()

    arr = np.asarray(matrix_oi)
    if arr.ndim != 2:
        raise ValueError("matrix_oi must be 2-dimensional")

    rownames = getattr(matrix_oi, "rownames", None)
    colnames = getattr(matrix_oi, "colnames", None)

    if rownames is None:
        rownames = [str(i) for i in range(arr.shape[0])]
    if colnames is None:
        colnames = [str(i) for i in range(arr.shape[1])]

    return pd.DataFrame(arr, index=rownames, columns=colnames)


def _resolve_legend_loc(position: str) -> str:
    """Map R-style legend positions to matplotlib legend location strings."""
    mapping = {
        "top": "upper center",
        "bottom": "lower center",
        "left": "center left",
        "right": "center right",
        "none": "",
    }
    return mapping.get(position, "best")


# ---------------------------------------------------------------------------
# Heatmap functions
# ---------------------------------------------------------------------------

def make_heatmap_ggplot(
    matrix_oi: Any,
    y_name: str = "y",
    x_name: str = "x",
    y_axis: bool = True,
    x_axis: bool = True,
    x_axis_position: str = "top",
    legend_position: str = "top",
    color: str = "blue",
    legend_title: str = "score",
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    figsize: Tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Create a two-color heatmap (white to *color*) from a matrix.

    This is the Python equivalent of R ``nichenetr::make_heatmap_ggplot``.
    Uses ``seaborn.heatmap`` when available, falling back to
    ``matplotlib.pyplot.imshow``.

    Parameters
    ----------
    matrix_oi : array-like or pandas.DataFrame
        2-D matrix of continuous values.  ``_NamedArray`` objects (with
        ``.rownames`` / ``.colnames``) are also accepted.
    y_name : str
        Label for the y-axis.
    x_name : str
        Label for the x-axis.
    y_axis : bool
        Whether to display the y-axis tick labels and title.
    x_axis : bool
        Whether to display the x-axis tick labels and title.
    x_axis_position : str
        ``"top"`` or ``"bottom"``; only relevant when *x_axis* is True.
    legend_position : str
        One of ``"top"``, ``"bottom"``, ``"left"``, ``"right"``, ``"none"``.
    color : str
        High-end colour of the two-colour gradient (low is white).
    legend_title : str
        Title for the colour-bar legend.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw on.  If *None* a new figure is created.
    show : bool
        If True, call ``plt.show()`` before returning.
    figsize : tuple of float, optional
        Figure size in inches ``(width, height)``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the heatmap.
    """
    df = _matrix_to_dataframe(matrix_oi)

    if x_axis_position not in ("top", "bottom"):
        raise ValueError("x_axis_position must be 'top' or 'bottom'")
    if legend_position not in ("top", "bottom", "left", "right", "none"):
        raise ValueError(
            "legend_position must be one of 'top', 'bottom', 'left', 'right', 'none'"
        )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "white_to_color", ["whitesmoke", color]
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(4, df.shape[1] * 0.5), max(3, df.shape[0] * 0.4)))
    else:
        fig = ax.get_figure()

    if _HAS_SEABORN:
        cbar_kws = {"label": legend_title}
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            cbar=legend_position != "none",
            cbar_kws=cbar_kws,
        )
    else:
        im = ax.imshow(df.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(df.shape[0]))
        ax.set_yticklabels(df.index)
        if legend_position != "none":
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(legend_title)

    # Axis visibility
    if not x_axis:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel(x_name)
        if x_axis_position == "top":
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
        ax.tick_params(axis="x", rotation=90)

    if not y_axis:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    else:
        ax.set_ylabel(y_name)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def make_threecolor_heatmap_ggplot(
    matrix_oi: Any,
    y_name: str = "y",
    x_name: str = "x",
    y_axis: bool = True,
    x_axis: bool = True,
    x_axis_position: str = "top",
    legend_position: str = "top",
    low_color: str = "blue",
    mid_color: str = "whitesmoke",
    high_color: str = "red",
    mid: float = 0.0,
    legend_title: str = "score",
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    figsize: Tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Create a diverging three-color heatmap from a matrix.

    This is the Python equivalent of R
    ``nichenetr::make_threecolor_heatmap_ggplot``.  Ideal for plotting
    log-fold-change expression.

    Parameters
    ----------
    matrix_oi : array-like or pandas.DataFrame
        2-D matrix of continuous values.
    y_name : str
        Label for the y-axis.
    x_name : str
        Label for the x-axis.
    y_axis : bool
        Whether to display y-axis tick labels and title.
    x_axis : bool
        Whether to display x-axis tick labels and title.
    x_axis_position : str
        ``"top"`` or ``"bottom"``.
    legend_position : str
        One of ``"top"``, ``"bottom"``, ``"left"``, ``"right"``, ``"none"``.
    low_color : str
        Colour for the lowest value.
    mid_color : str
        Colour at the *mid* value.
    high_color : str
        Colour for the highest value.
    mid : float
        Midpoint value that receives *mid_color*.
    legend_title : str
        Colour-bar legend title.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes.
    show : bool
        If True, call ``plt.show()``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the heatmap.
    """
    df = _matrix_to_dataframe(matrix_oi)

    if x_axis_position not in ("top", "bottom"):
        raise ValueError("x_axis_position must be 'top' or 'bottom'")
    if legend_position not in ("top", "bottom", "left", "right", "none"):
        raise ValueError(
            "legend_position must be one of 'top', 'bottom', 'left', 'right', 'none'"
        )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "three_color", [low_color, mid_color, high_color]
    )

    vmin = df.values.min()
    vmax = df.values.max()
    # TwoSlopeNorm requires vmin < vcenter < vmax
    if vmin >= mid:
        vmin = mid - 1.0
    if vmax <= mid:
        vmax = mid + 1.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=mid, vmax=vmax)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(4, df.shape[1] * 0.5), max(3, df.shape[0] * 0.4)))
    else:
        fig = ax.get_figure()

    if _HAS_SEABORN:
        cbar_kws = {"label": legend_title}
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            norm=norm,
            linewidths=0.5,
            linecolor="white",
            cbar=legend_position != "none",
            cbar_kws=cbar_kws,
        )
    else:
        im = ax.imshow(df.values, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(df.shape[0]))
        ax.set_yticklabels(df.index)
        if legend_position != "none":
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(legend_title)

    if not x_axis:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel(x_name)
        if x_axis_position == "top":
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
        ax.tick_params(axis="x", rotation=90)

    if not y_axis:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    else:
        ax.set_ylabel(y_name)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Line / bar plot
# ---------------------------------------------------------------------------

def make_line_plot(
    ligand_activities: pd.DataFrame,
    potential_ligands: Optional[Sequence[str]] = None,
    ranking_range: Tuple[int, int] = (1, 20),
    agnostic_color: str = "tomato",
    focused_color: str = "black",
    *,
    score_column: str = "aupr_corrected",
    ligand_column: str = "test_ligand",
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    figsize: Tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Create a ranked bar/lollipop plot of ligand activity scores.

    Mirrors the R ``nichenetr::make_line_plot`` which compares sender-agnostic
    and sender-focused ligand rankings.  When *potential_ligands* is provided
    the plot highlights those ligands in *focused_color*; otherwise a simple
    horizontal bar chart of the top ligands is drawn.

    Parameters
    ----------
    ligand_activities : pandas.DataFrame
        Must contain columns named by *ligand_column* and *score_column*.
    potential_ligands : sequence of str, optional
        Ligands expressed in the sender cell type (sender-focused set).
        If *None*, all ligands are plotted as a simple bar chart.
    ranking_range : tuple of int
        ``(start, end)`` ranks to display (1-indexed, inclusive).
    agnostic_color : str
        Colour for the sender-agnostic ranking points.
    focused_color : str
        Colour for the sender-focused ligands.
    score_column : str
        Name of the numeric score column in *ligand_activities*.
    ligand_column : str
        Name of the column containing ligand names.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes.
    show : bool
        If True, call ``plt.show()``.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the ranking plot.
    """
    if ligand_column not in ligand_activities.columns:
        raise ValueError(f"Column '{ligand_column}' not found in ligand_activities")
    if score_column not in ligand_activities.columns:
        raise ValueError(f"Column '{score_column}' not found in ligand_activities")

    df = ligand_activities.copy()
    df["rank"] = df[score_column].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("rank")

    start, end = ranking_range
    df_plot = df[(df["rank"] >= start) & (df["rank"] <= end)].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6, max(3, len(df_plot) * 0.35)))
    else:
        fig = ax.get_figure()

    if potential_ligands is not None:
        potential_set = set(potential_ligands)
        colors = [
            focused_color if lig in potential_set else agnostic_color
            for lig in df_plot[ligand_column]
        ]
    else:
        colors = agnostic_color

    y_positions = range(len(df_plot))
    ax.barh(
        y_positions,
        df_plot[score_column].values,
        color=colors,
        edgecolor="white",
        height=0.7,
    )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df_plot[ligand_column].values)
    ax.invert_yaxis()
    ax.set_xlabel(score_column)
    ax.set_ylabel("Ligand")
    ax.set_title(f"Ligand activity (ranks {start}-{end})")

    # Add legend when potential_ligands provided
    if potential_ligands is not None:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=agnostic_color, label="Sender-agnostic"),
            Patch(facecolor=focused_color, label="Sender-focused"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Mushroom plot
# ---------------------------------------------------------------------------

def make_mushroom_plot(
    prioritization_table: pd.DataFrame,
    top_n: int = 30,
    show_rankings: bool = False,
    show_all_datapoints: bool = False,
    true_color_range: bool = True,
    size: str = "scaled_avg_exprs",
    color: str = "scaled_p_val_adapted",
    ligand_fill_colors: Tuple[str, str] = ("#DEEBF7", "#08306B"),
    receptor_fill_colors: Tuple[str, str] = ("#FEE0D2", "#A50F15"),
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    figsize: Tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Create a mushroom dot-plot of prioritised ligand-receptor interactions.

    Each glyph consists of two semicircles (ligand above, receptor below).
    Semicircle *size* encodes one metric (default: scaled average expression)
    and *colour* encodes another (default: scaled adjusted p-value).

    This mirrors R ``nichenetr::make_mushroom_plot`` which relies on
    ``ggforce::geom_arc_bar``.

    Parameters
    ----------
    prioritization_table : pandas.DataFrame
        Must contain ``sender``, ``ligand``, ``receptor``, and columns
        ``<size>_ligand``, ``<size>_receptor``, ``<color>_ligand``,
        ``<color>_receptor`` (or ``_sender`` / ``_receiver`` variants for
        ``pct_expressed``).
    top_n : int
        Number of top-ranked interactions to display.
    show_rankings : bool
        Whether to annotate each glyph with its rank number.
    show_all_datapoints : bool
        If True, also show interactions beyond *top_n* in muted colours.
    true_color_range : bool
        If True, let the colour scale adapt to the data range of the top-n
        interactions.  If False, fix to [0, 1].
    size : str
        Column base name for semicircle size.
    color : str
        Column base name for semicircle colour.
    ligand_fill_colors : tuple of str
        ``(low, high)`` gradient colours for ligand semicircles.
    receptor_fill_colors : tuple of str
        ``(low, high)`` gradient colours for receptor semicircles.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes.
    show : bool
        If True, call ``plt.show()``.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the mushroom plot.
    """
    from matplotlib.patches import Wedge
    from matplotlib.collections import PatchCollection

    df = prioritization_table.copy()

    # Determine column suffixes
    size_ext = ["ligand", "receptor"]
    color_ext = ["ligand", "receptor"]
    if size == "pct_expressed":
        size_ext = ["sender", "receiver"]
    if color == "pct_expressed":
        color_ext = ["sender", "receiver"]

    required = [
        "sender", "ligand", "receptor",
        f"{size}_{size_ext[0]}", f"{size}_{size_ext[1]}",
        f"{color}_{color_ext[0]}", f"{color}_{color_ext[1]}",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in prioritization_table: {missing}")

    # Add ranking if not present
    if "prioritization_score" not in df.columns:
        raise ValueError("prioritization_table must contain 'prioritization_score'")
    if "prioritization_rank" not in df.columns:
        df["prioritization_rank"] = df["prioritization_score"].rank(ascending=False, method="min").astype(int)

    df = df.sort_values("prioritization_score", ascending=False)

    # Create lr_interaction label
    df["lr_interaction"] = df["ligand"] + " - " + df["receptor"]

    # Get unique top interactions
    top_interactions = df["lr_interaction"].unique()[:top_n]
    df_top = df[df["lr_interaction"].isin(top_interactions)].copy()

    if df_top.empty:
        raise ValueError("No ligand-receptor interactions found in the top_n.")

    # Order senders
    senders = sorted(df_top["sender"].unique())
    sender_to_x = {s: i for i, s in enumerate(senders)}

    # Interaction order (reversed so first is at top)
    interactions = list(top_interactions)
    interaction_to_y = {inter: len(interactions) - i for i, inter in enumerate(interactions)}

    n_celltypes = len(senders)
    n_interactions = len(interactions)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (max(4, n_celltypes * 1.5 + 2), max(4, n_interactions * 0.45))
        )
    else:
        fig = ax.get_figure()

    scale = 0.4  # max radius of semicircle

    # Colour maps
    cmap_ligand = mcolors.LinearSegmentedColormap.from_list(
        "ligand_cmap", [ligand_fill_colors[0], ligand_fill_colors[1]]
    )
    cmap_receptor = mcolors.LinearSegmentedColormap.from_list(
        "receptor_cmap", [receptor_fill_colors[0], receptor_fill_colors[1]]
    )

    # Colour normalisation
    if true_color_range:
        lig_color_vals = df_top[f"{color}_{color_ext[0]}"]
        rec_color_vals = df_top[f"{color}_{color_ext[1]}"]
        norm_lig = mcolors.Normalize(vmin=lig_color_vals.min(), vmax=lig_color_vals.max())
        norm_rec = mcolors.Normalize(vmin=rec_color_vals.min(), vmax=rec_color_vals.max())
    else:
        norm_lig = mcolors.Normalize(vmin=0, vmax=1)
        norm_rec = mcolors.Normalize(vmin=0, vmax=1)

    # Draw semicircles
    for _, row in df_top.iterrows():
        inter = row["lr_interaction"]
        if inter not in interaction_to_y:
            continue
        x = sender_to_x.get(row["sender"])
        if x is None:
            continue
        y = interaction_to_y[inter]

        size_lig = row[f"{size}_{size_ext[0]}"]
        size_rec = row[f"{size}_{size_ext[1]}"]
        color_lig = row[f"{color}_{color_ext[0]}"]
        color_rec = row[f"{color}_{color_ext[1]}"]

        r_lig = np.sqrt(np.clip(size_lig, 0, 1)) * scale
        r_rec = np.sqrt(np.clip(size_rec, 0, 1)) * scale

        # Ligand: upper semicircle (180 to 360 degrees)
        wedge_lig = Wedge(
            (x, y), r_lig, 180, 360,
            facecolor=cmap_ligand(norm_lig(color_lig)),
            edgecolor="white", linewidth=0.5,
        )
        ax.add_patch(wedge_lig)

        # Receptor: lower semicircle (0 to 180 degrees)
        wedge_rec = Wedge(
            (x, y), r_rec, 0, 180,
            facecolor=cmap_receptor(norm_rec(color_rec)),
            edgecolor="white", linewidth=0.5,
        )
        ax.add_patch(wedge_rec)

        # Rank annotation
        if show_rankings:
            ax.text(x, y, str(int(row["prioritization_rank"])),
                    ha="center", va="center", fontsize=6, color="white",
                    fontweight="bold")

    # Axes setup
    ax.set_xlim(-0.6, n_celltypes - 0.4)
    ax.set_ylim(0.3, n_interactions + 0.7)
    ax.set_xticks(range(n_celltypes))
    ax.set_xticklabels(senders, rotation=45, ha="right")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Sender cell types")
    ax.set_yticks(list(range(1, n_interactions + 1)))
    ax.set_yticklabels([interactions[n_interactions - i] for i in range(1, n_interactions + 1)])
    ax.set_ylabel("Ligand-receptor interaction")
    ax.set_aspect("equal", adjustable="box")

    # Grid
    for xi in np.arange(-0.5, n_celltypes + 0.5, 1):
        ax.axvline(xi, color="grey", linewidth=0.3, alpha=0.5)
    for yi in np.arange(0.5, n_interactions + 1.5, 1):
        ax.axhline(yi, color="grey", linewidth=0.3, alpha=0.5)

    # Colour bars
    sm_lig = plt.cm.ScalarMappable(cmap=cmap_ligand, norm=norm_lig)
    sm_lig.set_array([])
    sm_rec = plt.cm.ScalarMappable(cmap=cmap_receptor, norm=norm_rec)
    sm_rec.set_array([])

    # Humanise column names for legend
    _keywords = {
        "lfc": "LFC", "p": "pval", "val": "", "prod": "product",
        "avg": "mean", "adj": "adjusted", "exprs": "expression",
    }

    def _humanise(col: str) -> str:
        parts = col.split("_")
        return " ".join(_keywords.get(p, p) for p in parts).strip().capitalize()

    color_title = _humanise(color)

    cbar_lig = fig.colorbar(sm_lig, ax=ax, fraction=0.02, pad=0.04)
    cbar_lig.set_label(f"{color_title} (ligand)")
    cbar_rec = fig.colorbar(sm_rec, ax=ax, fraction=0.02, pad=0.08)
    cbar_rec.set_label(f"{color_title} (receptor)")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Circos / chord diagram helpers
# ---------------------------------------------------------------------------

def prepare_circos_visualization(
    circos_links: pd.DataFrame,
    ligand_colors: Optional[Dict[str, str]] = None,
    target_colors: Optional[Dict[str, str]] = None,
    widths: Optional[Dict[str, float]] = None,
    celltype_order: Optional[List[str]] = None,
) -> dict:
    """Prepare data structures for a ligand-target circos plot.

    This mirrors R ``nichenetr::prepare_circos_visualization``.

    Parameters
    ----------
    circos_links : pandas.DataFrame
        Must contain columns ``ligand``, ``target``, ``weight``,
        ``target_type``, and ``ligand_type``.
    ligand_colors : dict of str to str, optional
        Mapping from ``ligand_type`` values to hex/named colours.  If *None*,
        colours are auto-generated.
    target_colors : dict of str to str, optional
        Mapping from ``target_type`` values to hex/named colours.
    widths : dict of str to float, optional
        Gap widths with keys ``width_same_cell_same_ligand_type``,
        ``width_different_cell``, ``width_ligand_target``,
        ``width_same_cell_same_target_type``.
    celltype_order : list of str, optional
        Explicit ordering of cell types (ligand types) around the ring.

    Returns
    -------
    dict
        Keys: ``"links_circle"`` (DataFrame with ``ligand``, ``target``,
        ``weight``), ``"ligand_colors"`` (dict sector->colour),
        ``"order"`` (list of sector names), ``"gaps"`` (list of gap degrees).
    """
    required_cols = {"ligand", "target", "weight", "target_type", "ligand_type"}
    if not required_cols.issubset(circos_links.columns):
        raise ValueError(
            f"circos_links must have columns {required_cols}, "
            f"missing: {required_cols - set(circos_links.columns)}"
        )

    df = circos_links.copy()

    # Auto-generate colours if needed
    ligand_types = df["ligand_type"].unique().tolist()
    target_types = df["target_type"].unique().tolist()

    if ligand_colors is None or target_colors is None:
        n_auto = (0 if ligand_colors else len(ligand_types)) + (
            0 if target_colors else len(target_types)
        )
        auto_cmap = plt.cm.get_cmap("tab20", n_auto)
        idx = 0
        if ligand_colors is None:
            ligand_colors = {}
            for lt in ligand_types:
                ligand_colors[lt] = mcolors.to_hex(auto_cmap(idx))
                idx += 1
        if target_colors is None:
            target_colors = {}
            for tt in target_types:
                target_colors[tt] = mcolors.to_hex(auto_cmap(idx))
                idx += 1

    # Validate colour dicts
    for lt in ligand_types:
        if lt not in ligand_colors:
            raise ValueError(f"ligand_colors must contain '{lt}'")
    for tt in target_types:
        if tt not in target_colors:
            raise ValueError(f"target_colors must contain '{tt}'")

    # Filter extra colours
    ligand_colors = {k: v for k, v in ligand_colors.items() if k in ligand_types}
    target_colors = {k: v for k, v in target_colors.items() if k in target_types}

    # Validate celltype_order
    if celltype_order is not None:
        for lt in ligand_types:
            if lt not in celltype_order:
                raise ValueError(f"celltype_order must contain '{lt}'")
        celltype_order = [c for c in celltype_order if c in ligand_types]

    # Default widths
    if widths is None:
        widths = {
            "width_same_cell_same_ligand_type": 0.5,
            "width_different_cell": 6,
            "width_ligand_target": 15,
            "width_same_cell_same_target_type": 0.5,
        }

    required_widths = {
        "width_same_cell_same_ligand_type",
        "width_different_cell",
        "width_ligand_target",
        "width_same_cell_same_target_type",
    }
    if not required_widths.issubset(widths.keys()):
        raise ValueError(f"widths must contain keys: {required_widths}")

    # Add trailing space to ligands (R does this to distinguish ligand/target sectors)
    df["ligand"] = df["ligand"] + " "

    # Build sector colour mappings
    grid_col: Dict[str, str] = {}
    for _, row in df[["ligand", "ligand_type"]].drop_duplicates().iterrows():
        grid_col[row["ligand"]] = ligand_colors[row["ligand_type"]]
    for _, row in df[["target", "target_type"]].drop_duplicates().iterrows():
        grid_col[row["target"]] = target_colors[row["target_type"]]

    # Order
    target_type_order = df.sort_values("target_type")["target_type"].unique().tolist()
    target_order = df.sort_values(["target_type", "target"])["target"].unique().tolist()

    if celltype_order is None:
        # Put "General" first if present, then descending alpha
        def _sort_key(lt: str) -> tuple:
            return (0 if lt == "General" else 1, lt)

        ligand_type_order = sorted(ligand_types, key=_sort_key, reverse=False)
        # Re-sort: General first, then reverse-alpha (as in R)
        ligand_type_order_r: list[str] = []
        for lt in ligand_types:
            if lt == "General":
                ligand_type_order_r.insert(0, lt)
            else:
                ligand_type_order_r.append(lt)
        ligand_type_order = ligand_type_order_r
        # Sort ligands within each type
        ligand_order: list[str] = []
        for lt in ligand_type_order:
            ligs = df[df["ligand_type"] == lt]["ligand"].unique().tolist()
            ligand_order.extend(sorted(ligs))
    else:
        ligand_type_order = celltype_order
        ligand_order = []
        for lt in celltype_order:
            ligs = df[df["ligand_type"] == lt].sort_values("ligand")["ligand"].unique().tolist()
            ligand_order.extend(ligs)

    order = ligand_order + target_order

    # Gaps
    gaps_sender: list[float] = []
    for i, lt in enumerate(ligand_type_order):
        n_ligs = df[df["ligand_type"] == lt]["ligand"].nunique()
        gaps_sender.extend(
            [widths["width_same_cell_same_ligand_type"]] * max(0, n_ligs - 1)
        )
        if i < len(ligand_type_order) - 1:
            gaps_sender.append(widths["width_different_cell"])

    gaps_target: list[float] = []
    for i, tt in enumerate(target_type_order):
        n_tgts = df[df["target_type"] == tt]["target"].nunique()
        gaps_target.extend(
            [widths["width_same_cell_same_target_type"]] * max(0, n_tgts - 1)
        )
        if i < len(target_type_order) - 1:
            gaps_target.append(widths["width_different_cell"])

    gaps = (
        gaps_sender
        + [widths["width_ligand_target"]]
        + gaps_target
        + [widths["width_ligand_target"]]
    )

    links_circle = df[["ligand", "target", "weight"]].drop_duplicates()

    return {
        "links_circle": links_circle,
        "ligand_colors": grid_col,
        "order": order,
        "gaps": gaps,
    }


def make_circos_plot(
    vis_circos_obj: dict,
    transparency: bool = False,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    figsize: Tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Draw a circos (chord) plot for ligand-target links.

    Requires the ``pycirclize`` package.  Falls back to a simple
    matplotlib-based arc diagram when ``pycirclize`` is not installed.

    Parameters
    ----------
    vis_circos_obj : dict
        Object returned by :func:`prepare_circos_visualization`.  Must contain
        keys ``"links_circle"``, ``"ligand_colors"``, ``"order"``, ``"gaps"``.
    transparency : bool
        If True, link transparency scales with link weight (stronger links
        are more opaque).
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes (only used for the fallback renderer).
    show : bool
        If True, call ``plt.show()``.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the circos plot.
    """
    required_keys = {"links_circle", "ligand_colors", "order", "gaps"}
    if not required_keys.issubset(vis_circos_obj.keys()):
        raise ValueError(f"vis_circos_obj must contain keys {required_keys}")

    links = vis_circos_obj["links_circle"]
    grid_col = vis_circos_obj["ligand_colors"]
    order = vis_circos_obj["order"]

    if _HAS_PYCIRCLIZE:
        return _circos_pycirclize(
            links, grid_col, order, transparency=transparency,
            figsize=figsize, show=show,
        )
    else:
        return _circos_fallback(
            links, grid_col, order, transparency=transparency,
            ax=ax, figsize=figsize, show=show,
        )


def make_circos_lr(
    prioritized_tbl: pd.DataFrame,
    colors_sender: Dict[str, str],
    colors_receiver: Dict[str, str],
    cutoff: float = 0,
    scale: bool = False,
    transparency: Optional[Sequence[float]] = None,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = False,
    figsize: Tuple[float, float] | None = None,
) -> matplotlib.figure.Figure:
    """Create a circos plot for ligand-receptor interactions.

    Mirrors R ``nichenetr::make_circos_lr``.  Requires ``pycirclize`` for the
    full chord-diagram; falls back to a simple arc diagram otherwise.

    Parameters
    ----------
    prioritized_tbl : pandas.DataFrame
        Must contain ``sender``, ``receiver``, ``ligand``, ``receptor``, and
        ``prioritization_score``.
    colors_sender : dict of str to str
        Mapping from sender cell-type names to colours.
    colors_receiver : dict of str to str
        Mapping from receiver cell-type names to colours.
    cutoff : float
        Minimum ``prioritization_score`` for a link to be visible.
    scale : bool
        If True, min-max scale the weights to [0, 1].
    transparency : sequence of float, optional
        Per-link transparency values (0 = opaque, 1 = fully transparent).
        If *None*, transparency is derived from the weight.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes (fallback renderer only).
    show : bool
        If True, call ``plt.show()``.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the circos plot.
    """
    required_cols = {"sender", "receiver", "ligand", "receptor", "prioritization_score"}
    if not required_cols.issubset(prioritized_tbl.columns):
        raise ValueError(
            f"prioritized_tbl must contain columns {required_cols}, "
            f"missing: {required_cols - set(prioritized_tbl.columns)}"
        )

    df = prioritized_tbl.copy()
    df["weight"] = df["prioritization_score"]

    if "ligand_receptor" not in df.columns:
        df["ligand_receptor"] = df["ligand"] + "--" + df["receptor"]

    # Disambiguate duplicate sector names (same ligand from different senders)
    df["_lr_sr"] = df["sender"] + df["receiver"] + df["ligand_receptor"]

    for lig in df["ligand"].unique():
        mask_lig = df["ligand"] == lig
        for j, sender in enumerate(sorted(df.loc[mask_lig, "sender"].unique())):
            if j > 0:
                mask = mask_lig & (df["sender"] == sender)
                df.loc[mask, "ligand"] = lig + " " * j

    for rec in df["receptor"].unique():
        mask_rec = df["receptor"] == rec
        for j, receiver in enumerate(sorted(df.loc[mask_rec, "receiver"].unique())):
            if j > 0:
                mask = mask_rec & (df["receiver"] == receiver)
                df.loc[mask, "receptor"] = rec + " " * j

    # Ensure no overlap between ligand and receptor sector names
    while set(df["ligand"].unique()) & set(df["receptor"].unique()):
        overlap = set(df["ligand"].unique()) & set(df["receptor"].unique())
        for name in overlap:
            df.loc[df["receptor"] == name, "receptor"] = name + " "

    # Build colour map
    grid_col: Dict[str, str] = {}
    for _, row in df[["ligand", "sender"]].drop_duplicates().iterrows():
        grid_col[row["ligand"]] = colors_sender.get(row["sender"], "#999999")
    for _, row in df[["receptor", "receiver"]].drop_duplicates().iterrows():
        grid_col[row["receptor"]] = colors_receiver.get(row["receiver"], "#999999")

    # Order: ligands grouped by sender, then receptors grouped by receiver
    ligand_order: list[str] = []
    for sender in sorted(df["sender"].unique()):
        ligs = sorted(df[df["sender"] == sender]["ligand"].unique())
        ligand_order.extend(ligs)

    receptor_order: list[str] = []
    for receiver in sorted(df["receiver"].unique()):
        recs = sorted(df[df["receiver"] == receiver]["receptor"].unique())
        receptor_order.extend(recs)

    order = ligand_order + receptor_order

    links = df[["ligand", "receptor", "weight"]].drop_duplicates()
    links.columns = ["ligand", "target", "weight"]

    if scale and links["weight"].max() != links["weight"].min():
        w = links["weight"]
        links["weight"] = (w - w.min()) / (w.max() - w.min())

    # Filter by cutoff
    links_vis = links[links["weight"] >= cutoff]

    # Transparency
    use_transparency = transparency is not None
    if transparency is None and links_vis["weight"].max() > links_vis["weight"].min():
        _w = links_vis["weight"].values
        trans_vals = 1.0 - _w
    else:
        trans_vals = np.zeros(len(links_vis)) if transparency is None else np.asarray(transparency)
        if transparency is not None:
            use_transparency = True

    if _HAS_PYCIRCLIZE:
        return _circos_pycirclize(
            links_vis, grid_col, order,
            transparency=use_transparency,
            transparency_values=trans_vals,
            figsize=figsize, show=show,
        )
    else:
        return _circos_fallback(
            links_vis, grid_col, order,
            transparency=use_transparency,
            ax=ax, figsize=figsize, show=show,
        )


# ---------------------------------------------------------------------------
# Circos backends
# ---------------------------------------------------------------------------

def _circos_pycirclize(
    links: pd.DataFrame,
    grid_col: Dict[str, str],
    order: List[str],
    *,
    transparency: bool = False,
    transparency_values: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] | None = None,
    show: bool = False,
) -> matplotlib.figure.Figure:
    """Render a chord diagram with pycirclize.

    Parameters
    ----------
    links : pandas.DataFrame
        Columns ``ligand``, ``target``, ``weight``.
    grid_col : dict
        Sector name to colour mapping.
    order : list of str
        Sector order around the ring.
    transparency : bool
        Scale link alpha by weight.
    transparency_values : numpy.ndarray, optional
        Explicit per-link transparency (0 = opaque).
    figsize : tuple, optional
        Figure size.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Build a "from-to" matrix for pycirclize
    sectors_in_links = set(links["ligand"].unique()) | set(links["target"].unique())
    # Only keep sectors that appear in links
    order_filtered = [s for s in order if s in sectors_in_links]
    if not order_filtered:
        order_filtered = list(sectors_in_links)

    # Create matrix
    matrix_df = pd.DataFrame(0.0, index=order_filtered, columns=order_filtered)
    for _, row in links.iterrows():
        src, tgt, w = row["ligand"], row["target"], row["weight"]
        if src in matrix_df.index and tgt in matrix_df.columns:
            matrix_df.loc[src, tgt] += w

    circos = Circos.initialize_from_matrix(
        matrix_df,
        space=3,
        cmap="tab20",
        label_kws=dict(size=7, r=110),
        link_kws=dict(direction=1),
    )

    # Override sector colours
    for sector in circos.sectors:
        colour = grid_col.get(sector.name, "#CCCCCC")
        # Colour the track
        track = sector.tracks[0] if sector.tracks else None
        if track is not None:
            track.axis(fc=colour, ec="none")

    fig = circos.plotfig(figsize=figsize or (8, 8))
    if show:
        plt.show()
    return fig


def _circos_fallback(
    links: pd.DataFrame,
    grid_col: Dict[str, str],
    order: List[str],
    *,
    transparency: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Tuple[float, float] | None = None,
    show: bool = False,
) -> matplotlib.figure.Figure:
    """Simple arc-diagram fallback when pycirclize is not available.

    Parameters
    ----------
    links : pandas.DataFrame
        Columns ``ligand``, ``target``, ``weight``.
    grid_col : dict
        Node name to colour.
    order : list of str
        Node order.
    transparency : bool
        Scale link alpha by weight.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes.
    figsize : tuple, optional
        Figure size.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.patches import Arc as MplArc

    sectors = set(links["ligand"].unique()) | set(links["target"].unique())
    order_filtered = [s for s in order if s in sectors]
    if not order_filtered:
        order_filtered = sorted(sectors)

    node_pos = {name: i for i, name in enumerate(order_filtered)}
    n = len(order_filtered)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(8, n * 0.4), max(4, n * 0.25)))
    else:
        fig = ax.get_figure()

    # Draw links as arcs
    w_max = links["weight"].max() if len(links) > 0 else 1
    w_min = links["weight"].min() if len(links) > 0 else 0
    w_range = w_max - w_min if w_max != w_min else 1

    for _, row in links.iterrows():
        src, tgt = row["ligand"], row["target"]
        if src not in node_pos or tgt not in node_pos:
            continue
        x1, x2 = node_pos[src], node_pos[tgt]
        if x1 == x2:
            continue
        mid = (x1 + x2) / 2
        width = abs(x2 - x1)
        height = width * 0.5

        alpha = 1.0
        if transparency:
            alpha = 0.2 + 0.8 * (row["weight"] - w_min) / w_range

        colour = grid_col.get(src, "#999999")
        arc = MplArc(
            (mid, 0), width, height, angle=0,
            theta1=0, theta2=180,
            color=colour, linewidth=1.5, alpha=alpha,
        )
        ax.add_patch(arc)

    # Draw nodes
    for name, pos in node_pos.items():
        colour = grid_col.get(name, "#CCCCCC")
        ax.plot(pos, 0, "o", color=colour, markersize=8, zorder=5)
        ax.text(pos, -0.15, name.strip(), ha="center", va="top",
                fontsize=6, rotation=90)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-1, max(2, n * 0.3))
    ax.axis("off")
    ax.set_title("Ligand-target chord diagram (arc layout)")

    fig.tight_layout()
    if show:
        plt.show()
    return fig
