"""Functions for extracting and visualizing ligand-target and ligand-receptor links.

Provides utilities to identify weighted ligand-target and ligand-receptor
relationships from NicheNet prior models, and to prepare matrices suitable
for heatmap visualization with optional hierarchical clustering.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .datasets import NamedMatrix

__all__ = [
    "get_weighted_ligand_target_links",
    "prepare_ligand_target_visualization",
    "get_weighted_ligand_receptor_links",
    "prepare_ligand_receptor_visualization",
    "get_ligand_target_links_oi",
]


def get_weighted_ligand_target_links(
    ligand_oi: str,
    geneset: list[str] | set[str],
    ligand_target_matrix: NamedMatrix,
    n: int = 250,
) -> pd.DataFrame:
    """Get the top weighted ligand-target links for a ligand of interest.

    For a given ligand, identify the top *n* target genes by regulatory
    potential score and return those that overlap with a gene set of interest.

    Parameters
    ----------
    ligand_oi : str
        Ligand of interest (must be present in *ligand_target_matrix* column
        names).
    geneset : list[str] or set[str]
        Gene set of interest (e.g., differentially expressed genes).
    ligand_target_matrix : NamedMatrix
        Sparse matrix with target genes as rows and ligands as columns,
        together with corresponding row and column name lists.
    n : int, optional
        Number of top target genes to consider per ligand. Default is 250.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``ligand``, ``target``, and ``weight``.
        If no targets overlap with *geneset*, ``target`` and ``weight`` are
        ``NaN``.
    """
    geneset = set(geneset)

    # Locate ligand column
    if ligand_oi not in ligand_target_matrix.colnames:
        return pd.DataFrame(
            {"ligand": [ligand_oi], "target": [np.nan], "weight": [np.nan]}
        )

    col_idx = ligand_target_matrix.colnames.index(ligand_oi)
    col_dense: np.ndarray = np.asarray(
        ligand_target_matrix.data[:, col_idx].todense()
    ).ravel()

    # Get threshold: minimum score among top-n targets
    sorted_scores = np.sort(col_dense)[::-1]
    top_n_score = sorted_scores[min(n, len(sorted_scores)) - 1]

    # Targets meeting threshold that are also in geneset
    mask = col_dense >= top_n_score
    candidate_names = [
        ligand_target_matrix.rownames[i] for i, m in enumerate(mask) if m
    ]
    targets = [t for t in candidate_names if t in geneset]

    if len(targets) == 0:
        return pd.DataFrame(
            {"ligand": [ligand_oi], "target": [np.nan], "weight": [np.nan]}
        )

    # Build row-index lookup for selected targets
    row_idx_map = {name: i for i, name in enumerate(ligand_target_matrix.rownames)}
    weights = [col_dense[row_idx_map[t]] for t in targets]

    return pd.DataFrame({"ligand": ligand_oi, "target": targets, "weight": weights})


def prepare_ligand_target_visualization(
    ligand_target_df: pd.DataFrame,
    ligand_target_matrix: NamedMatrix,
    cutoff: float = 0.25,
) -> np.ndarray:
    """Prepare a matrix of ligand-target regulatory potential scores for heatmap visualization.

    Constructs a dense sub-matrix of *ligand_target_matrix* restricted to
    the ligands and targets present in *ligand_target_df*.  A quantile-based
    cutoff zeroes out weak links, and rows/columns with all-zero entries are
    dropped.  Finally, hierarchical clustering is applied to reorder rows
    (targets) and columns (ligands).

    Parameters
    ----------
    ligand_target_df : pd.DataFrame
        DataFrame with columns ``ligand``, ``target``, and ``weight``
        (typically the output of :func:`get_weighted_ligand_target_links`).
    ligand_target_matrix : NamedMatrix
        Full ligand-target prior model as a sparse named matrix (targets as
        rows, ligands as columns).
    cutoff : float, optional
        Quantile cutoff applied to the weights in *ligand_target_df*.
        Scores below this quantile are set to zero. Default is 0.25.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_targets, n_ligands)`` with row and column
        labels accessible as ``array.dtype.metadata`` is not used; instead
        the returned array is a standard ``ndarray`` whose rows and columns
        correspond to the names stored in the ``rownames`` and ``colnames``
        attributes attached to the array (see attributes below).

        The array has two custom attributes:

        * ``rownames`` -- list[str] of target gene names (ordered by
          clustering).
        * ``colnames`` -- list[str] of ligand names (ordered by clustering).
    """
    # Drop NaN rows
    df = ligand_target_df.dropna(subset=["target", "weight"])
    if df.empty:
        arr = np.empty((0, 0))
        arr = _attach_names(arr, [], [])
        return arr

    # Quantile cutoff
    cutoff_value = np.quantile(df["weight"].values, cutoff)

    all_targets = df["target"].unique().tolist()
    all_ligands = df["ligand"].unique().tolist()

    # Build index maps
    row_map = {name: i for i, name in enumerate(ligand_target_matrix.rownames)}
    col_map = {name: i for i, name in enumerate(ligand_target_matrix.colnames)}

    # Filter to names that actually exist in the matrix
    all_targets = [t for t in all_targets if t in row_map]
    all_ligands = [l for l in all_ligands if l in col_map]

    if not all_targets or not all_ligands:
        arr = np.empty((0, 0))
        arr = _attach_names(arr, [], [])
        return arr

    row_indices = [row_map[t] for t in all_targets]
    col_indices = [col_map[l] for l in all_ligands]

    # Extract sub-matrix (targets x ligands)
    sub = ligand_target_matrix.data[np.ix_(row_indices, col_indices)].toarray()

    # Apply cutoff
    sub[sub < cutoff_value] = 0.0

    # Remove rows/cols that are all zero
    row_sums = sub.sum(axis=1)
    col_sums = sub.sum(axis=0)
    keep_row = row_sums > 0
    keep_col = col_sums > 0

    keep_targets = [t for t, k in zip(all_targets, keep_row) if k]
    keep_ligands = [l for l, k in zip(all_ligands, keep_col) if k]
    sub = sub[np.ix_(keep_row, keep_col)]

    if sub.shape[0] == 0 or sub.shape[1] == 0:
        arr = np.empty((0, 0))
        arr = _attach_names(arr, [], [])
        return arr

    # Hierarchical clustering for ordering
    if sub.shape[0] > 1 and sub.shape[1] > 1:
        # Cluster rows (targets): distance = 1 - cor(t(matrix))
        corr_rows = np.corrcoef(sub)
        corr_rows = np.nan_to_num(corr_rows, nan=0.0)
        dist_rows = 1.0 - corr_rows
        np.fill_diagonal(dist_rows, 0.0)
        dist_rows = np.clip(dist_rows, 0.0, None)
        condensed_rows = dist_rows[np.triu_indices(dist_rows.shape[0], k=1)]
        link_rows = linkage(condensed_rows, method="ward")
        order_rows = _hclust_order(link_rows)

        # Cluster cols (ligands): distance = 1 - cor(matrix)
        corr_cols = np.corrcoef(sub.T)
        corr_cols = np.nan_to_num(corr_cols, nan=0.0)
        dist_cols = 1.0 - corr_cols
        np.fill_diagonal(dist_cols, 0.0)
        dist_cols = np.clip(dist_cols, 0.0, None)
        condensed_cols = dist_cols[np.triu_indices(dist_cols.shape[0], k=1)]
        link_cols = linkage(condensed_cols, method="ward")
        order_cols = _hclust_order(link_cols)
    else:
        order_rows = list(range(sub.shape[0]))
        order_cols = list(range(sub.shape[1]))

    ordered_targets = [keep_targets[i] for i in order_rows]
    ordered_ligands = [keep_ligands[i] for i in order_cols]
    result = sub[np.ix_(order_rows, order_cols)]

    result = _attach_names(result, ordered_targets, ordered_ligands)
    return result


def get_weighted_ligand_receptor_links(
    best_upstream_ligands: list[str],
    expressed_receptors: list[str],
    lr_network: pd.DataFrame,
    weighted_networks_lr_sig: pd.DataFrame,
) -> pd.DataFrame:
    """Get weighted ligand-receptor links for ligands and receptors of interest.

    Joins the ligand-receptor network with the weighted signaling network to
    retrieve interaction weights, filtering to ligands in
    *best_upstream_ligands* and receptors in *expressed_receptors*.

    Parameters
    ----------
    best_upstream_ligands : list[str]
        Ligands of interest.
    expressed_receptors : list[str]
        Receptors expressed in the target cell type.
    lr_network : pd.DataFrame
        Ligand-receptor network with columns ``from`` and ``to``.
    weighted_networks_lr_sig : pd.DataFrame
        Weighted ligand-receptor signaling network with columns ``from``,
        ``to``, and ``weight``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``from``, ``to``, and ``weight`` containing
        the weighted ligand-receptor interactions.
    """
    lr_distinct = lr_network[["from", "to"]].drop_duplicates()

    # Inner join weighted network with lr network
    weighted_lr = weighted_networks_lr_sig.merge(
        lr_distinct, on=["from", "to"], how="inner"
    )

    # Filter lr_network to best ligands and expressed receptors
    lr_top = lr_distinct[
        lr_distinct["from"].isin(best_upstream_ligands)
        & lr_distinct["to"].isin(expressed_receptors)
    ].drop_duplicates(subset=["from", "to"])
    best_upstream_receptors = lr_top["to"].unique().tolist()

    # Filter weighted network
    result = weighted_lr[
        weighted_lr["from"].isin(best_upstream_ligands)
        & weighted_lr["to"].isin(best_upstream_receptors)
    ].copy()

    return result.reset_index(drop=True)


def prepare_ligand_receptor_visualization(
    lr_network_top_df: pd.DataFrame,
    best_upstream_ligands: list[str],
    order_hclust: Literal["both", "ligands", "receptors", "none"] = "both",
) -> np.ndarray:
    """Prepare a ligand-receptor interaction matrix for heatmap visualization.

    Pivots the long-format weighted ligand-receptor links into a matrix
    (receptors as rows, ligands as columns), optionally applying hierarchical
    clustering to reorder rows and/or columns.

    Parameters
    ----------
    lr_network_top_df : pd.DataFrame
        Long-format DataFrame with columns ``from`` (ligand), ``to``
        (receptor), and ``weight``.
    best_upstream_ligands : list[str]
        Ordered list of upstream ligands.  Used as column order when
        hierarchical clustering is not applied to ligands.
    order_hclust : {'both', 'ligands', 'receptors', 'none'}, optional
        Which axes to reorder by hierarchical clustering.  Default is
        ``'both'``.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_receptors, n_ligands)`` with custom
        attributes ``rownames`` (receptor names) and ``colnames`` (ligand
        names), ordered according to *order_hclust*.
    """
    valid_options = {"both", "ligands", "receptors", "none"}
    if order_hclust not in valid_options:
        raise ValueError(
            f"order_hclust must be one of {valid_options!r}, got {order_hclust!r}"
        )

    # Pivot: receptors (rows) x ligands (cols)
    pivot = lr_network_top_df.pivot_table(
        index="to", columns="from", values="weight", fill_value=0.0, aggfunc="first"
    )
    matrix = pivot.values.astype(float)
    receptor_names = pivot.index.tolist()
    ligand_names = pivot.columns.tolist()

    if matrix.size == 0:
        arr = np.empty((0, 0))
        arr = _attach_names(arr, [], [])
        return arr

    # Cluster receptors (rows)
    if order_hclust in ("both", "receptors"):
        if matrix.shape[0] > 1:
            dist_r = pdist(matrix, metric="matching")
            link_r = linkage(dist_r, method="ward")
            order_r = _hclust_order(link_r)
        else:
            order_r = [0]
        ordered_receptors = [receptor_names[i] for i in order_r]
    else:
        # No clustering: alphabetical order for receptors
        ordered_receptors = sorted(receptor_names)

    # Cluster ligands (cols)
    if order_hclust in ("both", "ligands"):
        if matrix.shape[1] > 1:
            dist_l = pdist(matrix.T, metric="matching")
            link_l = linkage(dist_l, method="ward")
            order_l = _hclust_order(link_l)
        else:
            order_l = [0]
        ordered_ligands = [ligand_names[i] for i in order_l]
    else:
        # No clustering: use reversed best_upstream_ligands order
        ordered_ligands = [
            l for l in reversed(best_upstream_ligands) if l in ligand_names
        ]

    # Intersect to handle any mismatches
    ordered_receptors = [r for r in ordered_receptors if r in receptor_names]
    ordered_ligands = [l for l in ordered_ligands if l in ligand_names]

    # Reindex
    r_idx = [receptor_names.index(r) for r in ordered_receptors]
    l_idx = [ligand_names.index(l) for l in ordered_ligands]
    result = matrix[np.ix_(r_idx, l_idx)]

    result = _attach_names(result, ordered_receptors, ordered_ligands)
    return result


def get_ligand_target_links_oi(
    ligand_type_indication_df: pd.DataFrame,
    active_ligand_target_links_df: pd.DataFrame,
    cutoff: float = 0.40,
) -> pd.DataFrame:
    """Filter ligand-target links by a quantile cutoff for circos visualization.

    Joins ligand type annotations with active ligand-target links, applies a
    weight quantile cutoff, and removes ligands and targets that have no
    links above the threshold.

    Parameters
    ----------
    ligand_type_indication_df : pd.DataFrame
        DataFrame with at least columns ``ligand_type`` and ``ligand``,
        indicating the cell type expressing each ligand.
    active_ligand_target_links_df : pd.DataFrame
        DataFrame with columns ``ligand``, ``target``, ``weight``, and
        ``target_type``.
    cutoff : float, optional
        Quantile cutoff on the weight column.  Links with weight at or below
        this quantile are used to determine which ligands and targets to
        remove. Default is 0.40.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing ligand-target links.  The quantile
        cutoff value is stored in ``df.attrs['cutoff_include_all_ligands']``.

    Raises
    ------
    ValueError
        If required columns are missing or *cutoff* is outside [0, 1].
    """
    required_lt = {"ligand_type", "ligand"}
    if not required_lt.issubset(ligand_type_indication_df.columns):
        raise ValueError(
            "ligand_type_indication_df must have columns 'ligand_type' and 'ligand'"
        )

    required_al = {"ligand", "target", "weight", "target_type"}
    if not required_al.issubset(active_ligand_target_links_df.columns):
        raise ValueError(
            "active_ligand_target_links_df must have columns "
            "'ligand', 'target', 'weight', and 'target_type'"
        )

    if not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between 0 and 1")

    # Join ligand type information
    merged = active_ligand_target_links_df.merge(
        ligand_type_indication_df, on="ligand", how="inner"
    )

    cutoff_value = np.quantile(merged["weight"].values, cutoff)

    # Identify links above cutoff
    above_cutoff = merged[merged["weight"] > cutoff_value]
    ligands_above = set(above_cutoff["ligand"].unique())
    targets_above = set(above_cutoff["target"].unique())

    # Remove ligands and targets not represented above cutoff
    ligands_to_remove = set(merged["ligand"].unique()) - ligands_above
    targets_to_remove = set(merged["target"].unique()) - targets_above

    result = merged[
        ~merged["target"].isin(targets_to_remove)
        & ~merged["ligand"].isin(ligands_to_remove)
    ].copy()

    result.attrs["cutoff_include_all_ligands"] = cutoff_value
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hclust_order(linkage_matrix: np.ndarray) -> list[int]:
    """Return the leaf ordering from a scipy linkage matrix.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix as returned by :func:`scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    list[int]
        Indices of the original observations in dendrogram order.
    """
    from scipy.cluster.hierarchy import leaves_list

    return leaves_list(linkage_matrix).tolist()


class _NamedArray(np.ndarray):
    """Thin ndarray subclass carrying *rownames* and *colnames* attributes."""

    rownames: list[str]
    colnames: list[str]

    def __new__(
        cls,
        input_array: np.ndarray,
        rownames: list[str],
        colnames: list[str],
    ) -> "_NamedArray":
        obj = np.asarray(input_array).view(cls)
        obj.rownames = rownames
        obj.colnames = colnames
        return obj

    def __array_finalize__(self, obj: object) -> None:
        if obj is None:
            return
        self.rownames = getattr(obj, "rownames", [])
        self.colnames = getattr(obj, "colnames", [])


def _attach_names(
    arr: np.ndarray, rownames: list[str], colnames: list[str]
) -> np.ndarray:
    """Wrap a plain ndarray with *rownames* and *colnames* attributes."""
    return _NamedArray(arr, rownames=rownames, colnames=colnames)
