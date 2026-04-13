"""Prioritization of cell-cell communication interactions.

Provides functions to calculate differential expression, average expression,
and combine multiple evidence sources into a prioritized ranking of
sender-ligand-receiver-receptor interactions.  Ported from the R
``nichenetr`` package with semantic parity.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from .utils import scale_quantile_adapted, scaling_zscore

__all__ = [
    "calculate_de",
    "get_exprs_avg",
    "process_table_to_ic",
    "generate_info_tables",
    "generate_prioritization_tables",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank_scale(series: pd.Series) -> pd.Series:
    """Rank-scale a Series to [0, 1] using average tie-breaking.

    Mirrors R's ``rank(x, ties.method="average", na.last=FALSE) /
    max(rank(...))``.

    Parameters
    ----------
    series : pd.Series
        Numeric values to rank-scale.

    Returns
    -------
    pd.Series
        Rank-scaled values.
    """
    ranked = series.rank(method="average", na_option="top")
    return ranked / ranked.max()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_de(
    adata: ad.AnnData,
    celltype_col: str,
    condition_oi: Optional[str] = None,
    condition_col: Optional[str] = None,
    assay_oi: Optional[str] = None,
    *,
    features: Optional[Sequence[str]] = None,
    min_pct: float = 0.0,
    logfc_threshold: float = 0.0,
) -> pd.DataFrame:
    """Calculate differential expression of each cell type versus all others.

    Uses ``scanpy.tl.rank_genes_groups`` (Wilcoxon test) as the Python
    equivalent of Seurat's ``FindAllMarkers``.  When *condition_oi* is
    provided, the data is first subset to cells belonging to that condition.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated single-cell data matrix (cells x genes).
    celltype_col : str
        Column in ``adata.obs`` with cell-type labels.
    condition_oi : str or None, optional
        If given, subset to cells from this condition before DE.
    condition_col : str or None, optional
        Column in ``adata.obs`` indicating the condition.  Required when
        *condition_oi* is not ``None``.
    assay_oi : str or None, optional
        Layer name to use.  ``None`` uses ``adata.X``.
    features : sequence of str or None, optional
        If given, restrict the analysis to these genes.
    min_pct : float, optional
        Minimum fraction of cells expressing the gene in either group.
        Corresponds to ``min.pct`` in Seurat.  Default ``0.0``.
    logfc_threshold : float, optional
        Minimum absolute log-fold change threshold.  Corresponds to
        ``logfc.threshold`` in Seurat.  Default ``0.0``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``gene``, ``p_val``, ``avg_log2FC``,
        ``pct.1``, ``pct.2``, ``p_val_adj``, and ``cluster_id``.

    Raises
    ------
    ValueError
        If only one of *condition_col* and *condition_oi* is provided.
    """
    if (condition_oi is None) != (condition_col is None):
        raise ValueError("Please provide both condition_col and condition_oi.")

    sub = adata.copy()

    # Subset to condition of interest
    if condition_oi is not None:
        mask = sub.obs[condition_col].astype(str) == str(condition_oi)
        sub = sub[mask].copy()

    # Optionally restrict to specific features
    if features is not None:
        available = [f for f in features if f in sub.var_names]
        if len(available) == 0:
            raise ValueError("None of the requested features are in the data.")
        sub = sub[:, available].copy()

    # Use specified layer
    if assay_oi is not None:
        if assay_oi in sub.layers:
            sub.X = sub.layers[assay_oi]

    # Set grouping
    sub.obs["_ct"] = sub.obs[celltype_col].astype(str).values
    celltypes = sorted(sub.obs["_ct"].unique())

    sc.tl.rank_genes_groups(
        sub,
        groupby="_ct",
        method="wilcoxon",
        pts=True,
    )

    results: list[pd.DataFrame] = []
    for ct in celltypes:
        res = sc.get.rank_genes_groups_df(sub, group=ct)
        res = res.rename(
            columns={
                "names": "gene",
                "pvals": "p_val",
                "logfoldchanges": "avg_log2FC",
                "pvals_adj": "p_val_adj",
            }
        )

        # Percentage expressed
        pts_df = pd.DataFrame(sub.uns["rank_genes_groups"]["pts"])
        pts_rest = pd.DataFrame(sub.uns["rank_genes_groups"]["pts_rest"])
        if ct in pts_df.columns:
            pct1_map = pts_df[ct].to_dict()
            pct2_map = pts_rest[ct].to_dict()
            res["pct.1"] = res["gene"].map(pct1_map).fillna(0.0)
            res["pct.2"] = res["gene"].map(pct2_map).fillna(0.0)
        else:
            res["pct.1"] = 0.0
            res["pct.2"] = 0.0

        res["cluster_id"] = ct

        # Apply min_pct filter
        if min_pct > 0:
            res = res[(res["pct.1"] >= min_pct) | (res["pct.2"] >= min_pct)]

        # Apply logfc threshold
        if logfc_threshold > 0:
            res = res[res["avg_log2FC"].abs() >= logfc_threshold]

        # Ensure column order
        res = res[
            ["gene", "p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj", "cluster_id"]
        ]
        results.append(res)

    if len(results) == 0:
        return pd.DataFrame(
            columns=["gene", "p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj", "cluster_id"]
        )

    return pd.concat(results, ignore_index=True)


def get_exprs_avg(
    adata: ad.AnnData,
    celltype_col: str,
    condition_oi: Optional[str] = None,
    condition_col: Optional[str] = None,
    assay_oi: Optional[str] = None,
    *,
    features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Calculate average gene expression per cell type.

    If *condition_oi* is provided, only cells from that condition are
    considered.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated single-cell data matrix (cells x genes).
    celltype_col : str
        Column in ``adata.obs`` with cell-type labels.
    condition_oi : str or None, optional
        If given, subset to cells from this condition.
    condition_col : str or None, optional
        Column in ``adata.obs`` indicating the condition.  Required when
        *condition_oi* is not ``None``.
    assay_oi : str or None, optional
        Layer name to use.  ``None`` uses ``adata.X``.
    features : sequence of str or None, optional
        If given, restrict output to these genes.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns ``gene``, ``cluster_id``, and
        ``avg_expr``.

    Raises
    ------
    ValueError
        If only one of *condition_col* and *condition_oi* is provided.
    """
    if (condition_oi is None) != (condition_col is None):
        raise ValueError("Please provide both condition_col and condition_oi.")

    sub = adata.copy()

    # Subset to condition
    if condition_oi is not None:
        mask = sub.obs[condition_col].astype(str) == str(condition_oi)
        sub = sub[mask].copy()

    # Optional layer
    if assay_oi is not None and assay_oi in sub.layers:
        sub.X = sub.layers[assay_oi]

    # Optional gene filtering
    if features is not None:
        available = [f for f in features if f in sub.var_names]
        if len(available) == 0:
            return pd.DataFrame(columns=["gene", "cluster_id", "avg_expr"])
        sub = sub[:, available].copy()

    celltypes = sorted(sub.obs[celltype_col].astype(str).unique())

    records: list[dict] = []
    for ct in celltypes:
        mask_ct = sub.obs[celltype_col].astype(str).values == ct
        mat = sub[mask_ct].X
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        means = np.asarray(mat).mean(axis=0).ravel()
        genes = sub.var_names.tolist()
        for g, m in zip(genes, means):
            records.append({"gene": g, "cluster_id": ct, "avg_expr": float(m)})

    return pd.DataFrame(records)


def process_table_to_ic(
    table_obj: pd.DataFrame,
    table_type: str = "expression",
    lr_network: Optional[pd.DataFrame] = None,
    senders_oi: Optional[Sequence[str]] = None,
    receivers_oi: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Process DE or expression information for intercellular communication.

    Retains ligand information for senders and receptor information for
    receivers, then joins through the prior-knowledge ligand-receptor network.

    Parameters
    ----------
    table_obj : pd.DataFrame
        Output of :func:`get_exprs_avg`, :func:`calculate_de`, or a
        condition-level DE table.
    table_type : str, optional
        One of ``"expression"``, ``"celltype_DE"``, or ``"group_DE"``.
    lr_network : pd.DataFrame or None
        Ligand-receptor network with columns ``ligand`` and ``receptor``
        (or ``from`` and ``to``).
    senders_oi : sequence of str or None, optional
        Sender cell types to retain.
    receivers_oi : sequence of str or None, optional
        Receiver cell types to retain.

    Returns
    -------
    pd.DataFrame
        Combined sender-receiver table linked through the LR network.
    """
    if lr_network is None:
        raise ValueError("lr_network must be provided.")

    lr = lr_network.copy()
    if {"from", "to"}.issubset(lr.columns) and not {"ligand", "receptor"}.issubset(
        lr.columns
    ):
        lr = lr.rename(columns={"from": "ligand", "to": "receptor"})

    ligands = lr["ligand"].unique()
    receptors = lr["receptor"].unique()

    if table_type == "expression":
        if senders_oi is not None:
            warnings.warn(
                "senders_oi is given. The expression data will be scaled with "
                "all remaining cell types, so it is recommended that "
                "senders_oi = None"
            )
        if receivers_oi is not None:
            warnings.warn(
                "receivers_oi is given. The expression data will be scaled "
                "with all remaining cell types, so it is recommended that "
                "receivers_oi = None"
            )

        sender_table = table_obj.rename(
            columns={"cluster_id": "sender", "gene": "ligand", "avg_expr": "avg_ligand"}
        )
        receiver_table = table_obj.rename(
            columns={"cluster_id": "receiver", "gene": "receptor", "avg_expr": "avg_receptor"}
        )
        columns_select = [
            "sender", "receiver", "ligand", "receptor",
            "avg_ligand", "avg_receptor", "ligand_receptor_prod",
        ]

    elif table_type == "celltype_DE":
        if senders_oi is None:
            warnings.warn(
                "senders_oi is None. For DE filtering, it is best if this "
                "parameter is given."
            )
        if receivers_oi is None:
            warnings.warn(
                "receivers_oi is None. For DE filtering, it is best if this "
                "parameter is given."
            )

        sender_table = table_obj.rename(
            columns={
                "cluster_id": "sender",
                "gene": "ligand",
                "avg_log2FC": "avg_ligand",
                "p_val": "p_val_ligand",
                "p_val_adj": "p_adj_ligand",
                "pct.1": "pct_expressed_sender",
            }
        )
        receiver_table = table_obj.rename(
            columns={
                "cluster_id": "receiver",
                "gene": "receptor",
                "avg_log2FC": "avg_receptor",
                "p_val": "p_val_receptor",
                "p_val_adj": "p_adj_receptor",
                "pct.1": "pct_expressed_receiver",
            }
        )
        columns_select = [
            "sender", "receiver", "ligand", "receptor",
            "lfc_ligand", "lfc_receptor", "ligand_receptor_lfc_avg",
            "p_val_ligand", "p_adj_ligand",
            "p_val_receptor", "p_adj_receptor",
            "pct_expressed_sender", "pct_expressed_receiver",
        ]

    elif table_type == "group_DE":
        if senders_oi is not None:
            raise ValueError(
                "senders_oi is given. Since we do not consider cell type "
                "specificity with group DE, please change this to None."
            )
        if receivers_oi is not None:
            raise ValueError(
                "receivers_oi is given. Since we do not consider cell type "
                "specificity with group DE, please change this to None."
            )

        sender_table = table_obj.rename(
            columns={
                "gene": "ligand",
                "avg_log2FC": "avg_ligand",
                "p_val": "p_val_ligand",
                "p_val_adj": "p_adj_ligand",
            }
        )
        receiver_table = table_obj.rename(
            columns={
                "gene": "receptor",
                "avg_log2FC": "avg_receptor",
                "p_val": "p_val_receptor",
                "p_val_adj": "p_adj_receptor",
            }
        )
        columns_select = [
            "ligand", "receptor",
            "lfc_ligand", "lfc_receptor", "ligand_receptor_lfc_avg",
            "p_val_ligand", "p_adj_ligand",
            "p_val_receptor", "p_adj_receptor",
        ]
    else:
        raise ValueError(
            f"table_type must be 'expression', 'celltype_DE', or 'group_DE', "
            f"got {table_type!r}"
        )

    # Filter senders/receivers
    if senders_oi is not None and "sender" in sender_table.columns:
        sender_table = sender_table[sender_table["sender"].isin(senders_oi)]
    if receivers_oi is not None and "receiver" in receiver_table.columns:
        receiver_table = receiver_table[receiver_table["receiver"].isin(receivers_oi)]

    # Filter to known ligands / receptors
    sender_table = sender_table[sender_table["ligand"].isin(ligands)]
    receiver_table = receiver_table[receiver_table["receptor"].isin(receptors)]

    # Join sender -> lr_network -> receiver
    merged = sender_table.merge(lr[["ligand", "receptor"]], on="ligand", how="inner")
    # Need to handle suffix for receptor columns; receiver_table has "receptor"
    # as a gene column, so join on receptor
    merged = merged.merge(receiver_table, on="receptor", how="inner")

    # Handle duplicate suffixes from merge
    # Clean up: if columns ended with _x or _y from merge, prefer the sender/receiver one
    for col in list(merged.columns):
        if col.endswith("_x") or col.endswith("_y"):
            base = col[:-2]
            if base not in merged.columns:
                merged = merged.rename(columns={col: base})

    # Calculate combined metric
    if table_type == "expression":
        merged["ligand_receptor_prod"] = merged["avg_ligand"] * merged["avg_receptor"]
        merged = merged.sort_values("ligand_receptor_prod", ascending=False)
    else:
        # DE types: rename avg to lfc and compute average
        merged = merged.rename(
            columns={"avg_ligand": "lfc_ligand", "avg_receptor": "lfc_receptor"}
        )
        merged["ligand_receptor_lfc_avg"] = (
            merged["lfc_ligand"] + merged["lfc_receptor"]
        ) / 2.0
        merged = merged.sort_values("ligand_receptor_lfc_avg", ascending=False)

    # Select and deduplicate
    available_cols = [c for c in columns_select if c in merged.columns]
    merged = merged[available_cols].drop_duplicates()

    return merged.reset_index(drop=True)


def generate_info_tables(
    adata: ad.AnnData,
    celltype_col: str,
    senders_oi: Sequence[str],
    receivers_oi: Sequence[str],
    lr_network: pd.DataFrame,
    condition_col: Optional[str] = None,
    condition_oi: Optional[str] = None,
    condition_ref: Optional[str] = None,
    scenario: str = "case_control",
    assay_oi: Optional[str] = None,
) -> Dict[str, Optional[pd.DataFrame]]:
    """Generate the information tables required for prioritization.

    Computes cell-type DE, average expression, and (optionally) condition-level
    DE for ligands and receptors, then processes each through
    :func:`process_table_to_ic`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated single-cell data matrix.
    celltype_col : str
        Column in ``adata.obs`` with cell-type labels.
    senders_oi : sequence of str
        Sender cell types.
    receivers_oi : sequence of str
        Receiver cell types.
    lr_network : pd.DataFrame
        Ligand-receptor network (columns ``ligand``, ``receptor`` or
        ``from``, ``to``).
    condition_col : str or None, optional
        Column for condition labels.
    condition_oi : str or None, optional
        Condition of interest.
    condition_ref : str or None, optional
        Reference condition.
    scenario : str, optional
        ``"case_control"`` or ``"one_condition"``.  Default
        ``"case_control"``.
    assay_oi : str or None, optional
        Layer to use.  ``None`` uses ``adata.X``.

    Returns
    -------
    dict
        Dictionary with keys ``"sender_receiver_de"``,
        ``"sender_receiver_info"``, and ``"lr_condition_de"``.

    Raises
    ------
    ValueError
        On invalid parameter combinations.
    """
    senders_oi = list(senders_oi)
    receivers_oi = list(receivers_oi)

    # Validate
    if celltype_col not in adata.obs.columns:
        raise KeyError(
            f"celltype_col {celltype_col!r} not found in adata.obs."
        )

    cond_args = [condition_col, condition_oi, condition_ref]
    n_none = sum(x is None for x in cond_args)
    if 0 < n_none < 3:
        raise ValueError(
            "condition_col, condition_oi, and condition_ref must be either "
            "all None or all provided."
        )

    if n_none == 3 and scenario == "case_control":
        raise ValueError(
            "condition_* arguments are not provided but the 'case_control' "
            "scenario is selected.  Provide condition arguments or change "
            "scenario to 'one_condition'."
        )

    if scenario not in ("case_control", "one_condition"):
        raise ValueError("scenario must be 'case_control' or 'one_condition'.")

    ct_vals = set(adata.obs[celltype_col].astype(str).unique())
    missing_s = [s for s in senders_oi if s not in ct_vals]
    if missing_s:
        raise ValueError(f"Senders not in data: {missing_s}")
    missing_r = [r for r in receivers_oi if r not in ct_vals]
    if missing_r:
        raise ValueError(f"Receivers not in data: {missing_r}")

    # Normalise lr_network columns
    lr = lr_network.copy()
    if {"from", "to"}.issubset(lr.columns) and not {"ligand", "receptor"}.issubset(
        lr.columns
    ):
        lr = lr.rename(columns={"from": "ligand", "to": "receptor"})

    lr_features = list(set(lr["ligand"].tolist() + lr["receptor"].tolist()))

    # Cell-type DE
    de_table = calculate_de(
        adata,
        celltype_col=celltype_col,
        condition_oi=condition_oi,
        condition_col=condition_col,
        assay_oi=assay_oi,
        features=lr_features,
    )

    # Average expression
    expr_info = get_exprs_avg(
        adata,
        celltype_col=celltype_col,
        condition_oi=condition_oi,
        condition_col=condition_col,
        assay_oi=assay_oi,
        features=lr_features,
    )

    # Condition-specific DE
    processed_condition_markers: Optional[pd.DataFrame] = None
    if scenario == "case_control" and condition_col is not None:
        # Pseudo-bulk condition DE using rank_genes_groups between conditions
        sub = adata.copy()
        if assay_oi is not None and assay_oi in sub.layers:
            sub.X = sub.layers[assay_oi]

        available_feats = [f for f in lr_features if f in sub.var_names]
        if available_feats:
            sub = sub[:, available_feats].copy()

        sub.obs["_condition"] = sub.obs[condition_col].astype(str).values
        sc.tl.rank_genes_groups(
            sub,
            groupby="_condition",
            groups=[str(condition_oi)],
            reference=str(condition_ref),
            method="wilcoxon",
        )
        cond_df = sc.get.rank_genes_groups_df(sub, group=str(condition_oi))
        cond_df = cond_df.rename(
            columns={
                "names": "gene",
                "pvals": "p_val",
                "logfoldchanges": "avg_log2FC",
                "pvals_adj": "p_val_adj",
            }
        )
        cond_df = cond_df[["gene", "p_val", "avg_log2FC", "p_val_adj"]]

        processed_condition_markers = process_table_to_ic(
            cond_df, table_type="group_DE", lr_network=lr
        )
    elif scenario == "one_condition" and condition_col is not None:
        warnings.warn(
            "condition_* arguments are provided but the 'one_condition' "
            "scenario is selected.  Only cells from condition_oi will be "
            "used for cell type specificity; condition specificity will not "
            "be calculated."
        )

    # Process
    processed_de = process_table_to_ic(
        de_table,
        table_type="celltype_DE",
        lr_network=lr,
        senders_oi=senders_oi,
        receivers_oi=receivers_oi,
    )

    processed_expr = process_table_to_ic(
        expr_info,
        table_type="expression",
        lr_network=lr,
    )

    return {
        "sender_receiver_de": processed_de,
        "sender_receiver_info": processed_expr,
        "lr_condition_de": processed_condition_markers,
    }


def generate_prioritization_tables(
    sender_receiver_info: pd.DataFrame,
    sender_receiver_de: pd.DataFrame,
    ligand_activities: pd.DataFrame,
    lr_condition_de: Optional[pd.DataFrame] = None,
    prioritizing_weights: Optional[Dict[str, float]] = None,
    scenario: str = "case_control",
) -> pd.DataFrame:
    """Prioritize cell-cell interactions by combining multiple evidence sources.

    Combines cell-type differential expression, expression specificity,
    NicheNet ligand-activity scores, and (optionally) condition-level DE into
    a single prioritization score using a weighted average.

    Parameters
    ----------
    sender_receiver_info : pd.DataFrame
        Expression information from :func:`process_table_to_ic` with
        ``table_type="expression"``.
    sender_receiver_de : pd.DataFrame
        DE information from :func:`process_table_to_ic` with
        ``table_type="celltype_DE"``.
    ligand_activities : pd.DataFrame
        Output of ``predict_ligand_activities`` with columns
        ``test_ligand``, ``aupr_corrected``, and optionally ``rank``.
    lr_condition_de : pd.DataFrame or None, optional
        Condition-level DE from :func:`process_table_to_ic` with
        ``table_type="group_DE"``.
    prioritizing_weights : dict or None, optional
        Dict with keys ``"de_ligand"``, ``"de_receptor"``,
        ``"activity_scaled"``, ``"exprs_ligand"``, ``"exprs_receptor"``,
        ``"ligand_condition_specificity"``,
        ``"receptor_condition_specificity"``.  If ``None``, weights are
        set by *scenario*.
    scenario : str, optional
        ``"case_control"`` (all weights = 1) or ``"one_condition"``
        (condition specificity weights = 0).

    Returns
    -------
    pd.DataFrame
        Prioritized interactions sorted by ``prioritization_score``.
    """
    weight_names = [
        "de_ligand",
        "de_receptor",
        "activity_scaled",
        "exprs_ligand",
        "exprs_receptor",
        "ligand_condition_specificity",
        "receptor_condition_specificity",
    ]

    if prioritizing_weights is None:
        if scenario not in ("case_control", "one_condition"):
            raise ValueError(
                "scenario must be 'case_control' or 'one_condition'."
            )
        if scenario == "case_control":
            if lr_condition_de is None:
                raise ValueError(
                    "lr_condition_de is None.  Provide it or change scenario "
                    "to 'one_condition'."
                )
            weights = {k: 1.0 for k in weight_names}
        else:
            if lr_condition_de is not None:
                warnings.warn(
                    "lr_condition_de is provided but will not be used.  "
                    "Change scenario to 'case_control' if condition "
                    "specificity should be considered."
                )
            weights = {k: 1.0 for k in weight_names}
            weights["ligand_condition_specificity"] = 0.0
            weights["receptor_condition_specificity"] = 0.0
    else:
        missing = [k for k in weight_names if k not in prioritizing_weights]
        if missing:
            raise ValueError(
                f"prioritizing_weights is missing keys: {missing}"
            )
        weights = {k: float(prioritizing_weights[k]) for k in weight_names}

    # Ensure rank column
    la = ligand_activities.copy()
    if "rank" not in la.columns:
        la["rank"] = la["aupr_corrected"].rank(ascending=False, method="average")

    # --- Ligand DE prioritization ---
    sender_ligand = (
        sender_receiver_de[["sender", "ligand", "lfc_ligand", "p_val_ligand"]]
        .drop_duplicates()
        .copy()
    )
    sender_ligand["lfc_pval_ligand"] = (
        -np.log10(sender_ligand["p_val_ligand"].clip(lower=1e-300))
        * sender_ligand["lfc_ligand"]
    )
    sender_ligand["p_val_adapted_ligand"] = (
        -np.log10(sender_ligand["p_val_ligand"].clip(lower=1e-300))
        * np.sign(sender_ligand["lfc_ligand"])
    )
    sender_ligand["scaled_lfc_ligand"] = _rank_scale(sender_ligand["lfc_ligand"])
    sender_ligand["scaled_p_val_ligand"] = _rank_scale(
        -sender_ligand["p_val_ligand"]
    )
    sender_ligand["scaled_lfc_pval_ligand"] = _rank_scale(
        sender_ligand["lfc_pval_ligand"]
    )
    sender_ligand["scaled_p_val_adapted_ligand"] = _rank_scale(
        sender_ligand["p_val_adapted_ligand"]
    )

    # --- Receptor DE prioritization ---
    recv_receptor = (
        sender_receiver_de[["receiver", "receptor", "lfc_receptor", "p_val_receptor"]]
        .drop_duplicates()
        .copy()
    )
    recv_receptor["lfc_pval_receptor"] = (
        -np.log10(recv_receptor["p_val_receptor"].clip(lower=1e-300))
        * recv_receptor["lfc_receptor"]
    )
    recv_receptor["p_val_adapted_receptor"] = (
        -np.log10(recv_receptor["p_val_receptor"].clip(lower=1e-300))
        * np.sign(recv_receptor["lfc_receptor"])
    )
    recv_receptor["scaled_lfc_receptor"] = _rank_scale(
        recv_receptor["lfc_receptor"]
    )
    recv_receptor["scaled_p_val_receptor"] = _rank_scale(
        -recv_receptor["p_val_receptor"]
    )
    recv_receptor["scaled_lfc_pval_receptor"] = _rank_scale(
        recv_receptor["lfc_pval_receptor"]
    )
    recv_receptor["scaled_p_val_adapted_receptor"] = _rank_scale(
        recv_receptor["p_val_adapted_receptor"]
    )

    # --- Ligand activity prioritization ---
    la_prio = la[["test_ligand", "aupr_corrected", "rank"]].copy()
    if "receiver" in la.columns:
        la_prio["receiver"] = la["receiver"]
    la_prio = la_prio.rename(
        columns={"test_ligand": "ligand", "aupr_corrected": "activity"}
    )
    la_prio["activity_zscore"] = scaling_zscore(la_prio["activity"].values)
    la_prio["scaled_activity"] = scale_quantile_adapted(
        la_prio["activity"].values, outlier_cutoff=0.01
    )

    # --- Expression specificity of ligand ---
    lig_spec = (
        sender_receiver_info[["sender", "ligand", "avg_ligand"]]
        .drop_duplicates()
        .copy()
    )
    lig_spec["scaled_avg_exprs_ligand"] = lig_spec.groupby("ligand")[
        "avg_ligand"
    ].transform(lambda x: scale_quantile_adapted(x.values))

    # --- Expression specificity of receptor ---
    rec_spec = (
        sender_receiver_info[["receiver", "receptor", "avg_receptor"]]
        .drop_duplicates()
        .copy()
    )
    rec_spec["scaled_avg_exprs_receptor"] = rec_spec.groupby("receptor")[
        "avg_receptor"
    ].transform(lambda x: scale_quantile_adapted(x.values))

    # --- Condition-level prioritization ---
    lig_cond: Optional[pd.DataFrame] = None
    rec_cond: Optional[pd.DataFrame] = None
    if lr_condition_de is not None:
        # Ligand condition
        lig_cond = (
            lr_condition_de[["ligand", "lfc_ligand", "p_val_ligand"]]
            .drop_duplicates()
            .copy()
        )
        lig_cond["lfc_pval_ligand"] = (
            -np.log10(lig_cond["p_val_ligand"].clip(lower=1e-300))
            * lig_cond["lfc_ligand"]
        )
        lig_cond["p_val_adapted_ligand"] = (
            -np.log10(lig_cond["p_val_ligand"].clip(lower=1e-300))
            * np.sign(lig_cond["lfc_ligand"])
        )
        lig_cond["scaled_lfc_ligand"] = _rank_scale(lig_cond["lfc_ligand"])
        lig_cond["scaled_p_val_ligand"] = _rank_scale(-lig_cond["p_val_ligand"])
        lig_cond["scaled_lfc_pval_ligand"] = _rank_scale(
            lig_cond["lfc_pval_ligand"]
        )
        lig_cond["scaled_p_val_adapted_ligand"] = _rank_scale(
            lig_cond["p_val_adapted_ligand"]
        )
        # Suffix with _group
        lig_cond = lig_cond.rename(
            columns={
                c: c + "_group"
                for c in lig_cond.columns
                if c != "ligand"
            }
        )

        # Receptor condition
        rec_cond = (
            lr_condition_de[["receptor", "lfc_receptor", "p_val_receptor"]]
            .drop_duplicates()
            .copy()
        )
        rec_cond["lfc_pval_receptor"] = (
            -np.log10(rec_cond["p_val_receptor"].clip(lower=1e-300))
            * rec_cond["lfc_receptor"]
        )
        rec_cond["p_val_adapted_receptor"] = (
            -np.log10(rec_cond["p_val_receptor"].clip(lower=1e-300))
            * np.sign(rec_cond["lfc_receptor"])
        )
        rec_cond["scaled_lfc_receptor"] = _rank_scale(
            rec_cond["lfc_receptor"]
        )
        rec_cond["scaled_p_val_receptor"] = _rank_scale(
            -rec_cond["p_val_receptor"]
        )
        rec_cond["scaled_lfc_pval_receptor"] = _rank_scale(
            rec_cond["lfc_pval_receptor"]
        )
        rec_cond["scaled_p_val_adapted_receptor"] = _rank_scale(
            rec_cond["p_val_adapted_receptor"]
        )
        rec_cond = rec_cond.rename(
            columns={
                c: c + "_group"
                for c in rec_cond.columns
                if c != "receptor"
            }
        )
    else:
        cond_weights = [
            weights.get("ligand_condition_specificity", 0),
            weights.get("receptor_condition_specificity", 0),
        ]
        if any(w > 0 for w in cond_weights):
            raise ValueError(
                "No lr_condition_de table given, yet the relevant weights are "
                "nonzero.  Either set weights of 'ligand_condition_specificity'"
                " and 'receptor_condition_specificity' to zero or provide "
                "lr_condition_de."
            )

    # --- Assemble final table ---
    tbl = sender_receiver_de.merge(sender_receiver_info, how="inner")

    if weights["de_ligand"] > 0:
        tbl = tbl.merge(sender_ligand, how="inner")
    if weights["activity_scaled"] > 0:
        tbl = tbl.merge(la_prio, how="inner")
    if weights["de_receptor"] > 0:
        tbl = tbl.merge(recv_receptor, how="inner")
    if weights["exprs_ligand"] > 0:
        tbl = tbl.merge(lig_spec, how="inner")
    if weights["exprs_receptor"] > 0:
        tbl = tbl.merge(rec_spec, how="inner")
    if weights["ligand_condition_specificity"] > 0 and lig_cond is not None:
        tbl = tbl.merge(lig_cond, on="ligand", how="inner")
    if weights["receptor_condition_specificity"] > 0 and rec_cond is not None:
        tbl = tbl.merge(rec_cond, on="receptor", how="inner")

    # --- Compute prioritization score ---
    sum_weights = (
        0.5 * weights["de_ligand"]
        + 0.5 * weights["de_receptor"]
        + weights["activity_scaled"]
        + 0.5 * weights["exprs_ligand"]
        + 0.5 * weights["exprs_receptor"]
        + weights["ligand_condition_specificity"]
        + weights["receptor_condition_specificity"]
    )

    def _get_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return df[col].fillna(default)
        return pd.Series(default, index=df.index)

    tbl["prioritization_score"] = (
        0.5 * weights["de_ligand"] * _get_col(tbl, "scaled_p_val_adapted_ligand")
        + 0.5 * weights["de_receptor"] * _get_col(tbl, "scaled_p_val_adapted_receptor")
        + weights["activity_scaled"] * _get_col(tbl, "scaled_activity")
        + 0.5 * weights["exprs_ligand"] * _get_col(tbl, "scaled_avg_exprs_ligand")
        + 0.5 * weights["exprs_receptor"] * _get_col(tbl, "scaled_avg_exprs_receptor")
        + weights["ligand_condition_specificity"]
        * _get_col(tbl, "scaled_p_val_adapted_ligand_group")
        + weights["receptor_condition_specificity"]
        * _get_col(tbl, "scaled_p_val_adapted_receptor_group")
    ) / sum_weights

    tbl = tbl.sort_values("prioritization_score", ascending=False)
    tbl["prioritization_rank"] = tbl["prioritization_score"].rank(
        ascending=False, method="average"
    )

    return tbl.reset_index(drop=True)
