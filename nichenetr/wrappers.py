"""High-level NicheNet analysis wrappers for AnnData objects.

Provides end-to-end NicheNet pipelines that mirror the R ``nichenetr``
wrapper functions ``nichenet_seuratobj_aggregate`` and
``nichenet_seuratobj_cluster_de``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .prediction import predict_ligand_activities
from .symbols import get_expressed_genes, get_lfc_celltype
from .targets import (
    get_weighted_ligand_target_links,
    prepare_ligand_target_visualization,
    get_weighted_ligand_receptor_links,
    prepare_ligand_receptor_visualization,
)
from .datasets import load_lr_network, load_ligand_target_matrix, load_weighted_networks

__all__ = [
    "nichenet_seuratobj_aggregate",
    "nichenet_seuratobj_cluster_de",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_sender_celltypes(
    sender: Union[str, Sequence[str]],
    adata: ad.AnnData,
    celltype_col: str,
) -> list[str]:
    """Resolve the sender argument into a list of cell type labels.

    Parameters
    ----------
    sender : str or sequence of str
        ``"all"`` to use every cell type, ``"undefined"`` for no specific
        sender, or explicit cell type name(s).
    adata : anndata.AnnData
        The annotated data object.
    celltype_col : str
        Column in ``adata.obs`` that stores cell-type identities.

    Returns
    -------
    list[str]
        Resolved sender cell types (empty list for ``"undefined"``).
    """
    if isinstance(sender, str):
        if sender == "all":
            return sorted(adata.obs[celltype_col].astype(str).unique().tolist())
        elif sender == "undefined":
            return []
        else:
            return [sender]
    return list(sender)


def _get_expressed_genes_for_celltypes(
    celltypes: Sequence[str],
    adata: ad.AnnData,
    celltype_col: str,
    pct: float,
    layer: Optional[str],
) -> list[str]:
    """Collect union of expressed genes across multiple cell types.

    Parameters
    ----------
    celltypes : sequence of str
        Cell types to query.
    adata : anndata.AnnData
        Annotated data.
    celltype_col : str
        Column holding cell type labels.
    pct : float
        Minimum expression fraction.
    layer : str or None
        Layer to use.

    Returns
    -------
    list[str]
        Union of expressed genes across all *celltypes*.
    """
    genes: set[str] = set()
    for ct in celltypes:
        ct_genes = get_expressed_genes(
            adata, celltype_col=celltype_col, celltype=ct, pct=pct, assay_oi=layer
        )
        genes.update(ct_genes)
    return sorted(genes)


def _compute_de_between_conditions(
    adata: ad.AnnData,
    celltype_col: str,
    receiver: Union[str, Sequence[str]],
    condition_col: str,
    condition_oi: str,
    condition_ref: str,
    expression_pct: float,
    assay_oi: Optional[str],
) -> pd.DataFrame:
    """Run DE between two conditions within receiver cell(s).

    Parameters
    ----------
    adata : anndata.AnnData
        Full dataset.
    celltype_col : str
        Cell-type column.
    receiver : str or sequence of str
        Receiver cell type(s) to subset.
    condition_col : str
        Column with condition labels.
    condition_oi : str
        Condition of interest.
    condition_ref : str
        Reference condition.
    expression_pct : float
        Minimum fraction of cells expressing a gene.
    assay_oi : str or None
        Layer to use.

    Returns
    -------
    pd.DataFrame
        DE results with columns ``gene``, ``p_val``, ``avg_log2FC``,
        ``pct.1``, ``pct.2``, ``p_val_adj``.
    """
    receivers = [receiver] if isinstance(receiver, str) else list(receiver)
    mask = adata.obs[celltype_col].astype(str).isin(receivers)
    sub = adata[mask].copy()

    if assay_oi is not None and assay_oi in sub.layers:
        sub.X = sub.layers[assay_oi]

    sub.obs["_cond"] = sub.obs[condition_col].astype(str).values

    sc.tl.rank_genes_groups(
        sub,
        groupby="_cond",
        groups=[str(condition_oi)],
        reference=str(condition_ref),
        method="wilcoxon",
        pts=True,
    )

    res = sc.get.rank_genes_groups_df(sub, group=str(condition_oi))
    res = res.rename(
        columns={
            "names": "gene",
            "pvals": "p_val",
            "logfoldchanges": "avg_log2FC",
            "pvals_adj": "p_val_adj",
        }
    )

    # Percentage expressed
    rgg = sub.uns["rank_genes_groups"]
    grp = str(condition_oi)
    if "pts" in rgg:
        pts_df = pd.DataFrame(rgg["pts"])
        if grp in pts_df.columns:
            res["pct.1"] = res["gene"].map(pts_df[grp].to_dict()).fillna(0.0)
        else:
            res["pct.1"] = 0.0
        if "pts_rest" in rgg:
            pts_rest = pd.DataFrame(rgg["pts_rest"])
            if grp in pts_rest.columns:
                res["pct.2"] = res["gene"].map(pts_rest[grp].to_dict()).fillna(0.0)
            else:
                res["pct.2"] = 0.0
        else:
            res["pct.2"] = 0.0
    else:
        res["pct.1"] = 0.0
        res["pct.2"] = 0.0

    return res[["gene", "p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj"]]


def _build_lr_receptor_matrix(
    lr_network_top_df_large: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Pivot ligand-receptor weight table into a matrix and hierarchically
    cluster rows (receptors) and columns (ligands).

    Parameters
    ----------
    lr_network_top_df_large : pd.DataFrame
        Filtered weighted LR network with ``from``, ``to``, ``weight``.

    Returns
    -------
    tuple
        ``(matrix, order_receptors, order_ligands)`` after hierarchical
        clustering.
    """
    pivot = lr_network_top_df_large.pivot_table(
        index="to", columns="from", values="weight", fill_value=0.0
    )
    mat = pivot.values

    # Cluster receptors
    if mat.shape[0] > 1:
        dist_r = pdist(mat, metric="jaccard")
        dist_r = np.nan_to_num(dist_r, nan=0.0)
        link_r = linkage(dist_r, method="ward")
        from scipy.cluster.hierarchy import leaves_list
        order_r = leaves_list(link_r)
        order_receptors = [pivot.index[i] for i in order_r]
    else:
        order_receptors = pivot.index.tolist()

    # Cluster ligands
    if mat.shape[1] > 1:
        dist_l = pdist(mat.T, metric="jaccard")
        dist_l = np.nan_to_num(dist_l, nan=0.0)
        link_l = linkage(dist_l, method="ward")
        from scipy.cluster.hierarchy import leaves_list
        order_l = leaves_list(link_l)
        order_ligands = [pivot.columns[i] for i in order_l]
    else:
        order_ligands = pivot.columns.tolist()

    vis_mat = pivot.loc[order_receptors, order_ligands].values
    return vis_mat, order_receptors, order_ligands


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nichenet_seuratobj_aggregate(
    receiver: Union[str, Sequence[str]],
    adata: ad.AnnData,
    condition_col: str,
    condition_oi: str,
    condition_ref: str,
    sender: Union[str, Sequence[str]] = "all",
    celltype_col: Optional[str] = None,
    ligand_target_matrix: Optional[np.ndarray] = None,
    lr_network: Optional[pd.DataFrame] = None,
    weighted_networks: Optional[Dict[str, pd.DataFrame]] = None,
    expression_pct: float = 0.10,
    lfc_cutoff: float = 0.25,
    top_n_ligands: int = 30,
    top_n_targets: int = 200,
    cutoff_visualization: float = 0.33,
    geneset: str = "DE",
    filter_top_ligands: bool = True,
    assay_oi: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full NicheNet pipeline on an AnnData object.

    Explains differential expression in *receiver* cells between two
    conditions by ligands expressed by *sender* cells.  This is the Python
    equivalent of R ``nichenet_seuratobj_aggregate``.

    Parameters
    ----------
    receiver : str or sequence of str
        Receiver cell type(s).
    adata : anndata.AnnData
        Annotated single-cell data matrix.
    condition_col : str
        Column in ``adata.obs`` indicating the experimental condition.
    condition_oi : str
        Condition of interest (e.g. treated).
    condition_ref : str
        Reference/control condition.
    sender : str or sequence of str, optional
        ``"all"``, ``"undefined"``, or explicit cell type(s).  Default
        ``"all"``.
    celltype_col : str or None, optional
        Column in ``adata.obs`` with cell-type labels.  If ``None``, defaults
        to ``adata.obs`` column used by ``scanpy`` (i.e. must be set via
        ``adata.obs``).  Typically the active identity column.
    ligand_target_matrix : np.ndarray or None, optional
        Pre-loaded ligand-target matrix.  Loaded automatically if ``None``.
    lr_network : pd.DataFrame or None, optional
        Ligand-receptor network.  Loaded automatically if ``None``.
    weighted_networks : dict or None, optional
        Dict with key ``"lr_sig"`` holding a weighted LR network DataFrame.
        Loaded automatically if ``None``.
    expression_pct : float, optional
        Minimum fraction of cells expressing a gene.  Default ``0.10``.
    lfc_cutoff : float, optional
        Minimum absolute log2 fold-change for DE gene set.  Default ``0.25``.
    top_n_ligands : int, optional
        Number of top ligands to select.  Default ``30``.
    top_n_targets : int, optional
        Number of top targets per ligand.  Default ``200``.
    cutoff_visualization : float, optional
        Regulatory potential cutoff for target visualization.  Default
        ``0.33``.
    geneset : str, optional
        ``"DE"`` (both up and down), ``"up"``, or ``"down"``.
        Default ``"DE"``.
    filter_top_ligands : bool, optional
        Whether to filter to top *n* ligands.  Default ``True``.
    assay_oi : str or None, optional
        Layer in *adata* to use.  Default ``None`` (uses ``adata.X``).
    verbose : bool, optional
        Print progress messages.  Default ``True``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``ligand_activities`` : pd.DataFrame
        - ``top_ligands`` : list[str]
        - ``top_targets`` : list[str]
        - ``top_receptors`` : list[str]
        - ``ligand_target_matrix`` : np.ndarray or None
        - ``ligand_target_df`` : pd.DataFrame
        - ``ligand_receptor_matrix`` : np.ndarray
        - ``ligand_receptor_df`` : pd.DataFrame
        - ``geneset_oi`` : list[str]
        - ``background_expressed_genes`` : list[str]

    Raises
    ------
    ValueError
        On invalid inputs or when no DE genes / ligands are found.
    """
    if geneset not in ("DE", "up", "down"):
        raise ValueError("geneset must be 'DE', 'up', or 'down'.")

    # Resolve cell-type column
    if celltype_col is None:
        # Try to infer a sensible default
        raise ValueError(
            "celltype_col must be provided.  Specify the column in "
            "adata.obs that contains cell-type labels."
        )

    # Ensure receiver is a list
    receivers = [receiver] if isinstance(receiver, str) else list(receiver)

    # Auto-load resources
    if ligand_target_matrix is None:
        ligand_target_matrix = load_ligand_target_matrix()
    if lr_network is None:
        lr_network = load_lr_network()
    if weighted_networks is None:
        weighted_networks = load_weighted_networks()

    # Normalise lr_network column names
    lr_net = lr_network.copy()
    if "from" in lr_net.columns and "to" in lr_net.columns:
        if "ligand" not in lr_net.columns:
            lr_net = lr_net.rename(columns={"from": "ligand", "to": "receptor"})

    # Determine ligand-target matrix gene names
    # Keep/create a NamedMatrix for predict_ligand_activities
    if isinstance(ligand_target_matrix, pd.DataFrame):
        lt_matrix = ligand_target_matrix
        lt_target_genes = lt_matrix.index.tolist()
        lt_ligand_genes = lt_matrix.columns.tolist()
        from .datasets import NamedMatrix as _NM
        lt_matrix_original = _NM(
            data=scipy.sparse.csr_matrix(lt_matrix.values),
            rownames=lt_target_genes,
            colnames=lt_ligand_genes,
        )
    else:
        # Assume a NamedMatrix (has .rownames, .colnames, .data)
        lt_matrix_original = ligand_target_matrix
        lt_target_genes = list(ligand_target_matrix.rownames)
        lt_ligand_genes = list(ligand_target_matrix.colnames)
        _lt_data = ligand_target_matrix.data
        if hasattr(_lt_data, "toarray"):
            _lt_data = _lt_data.toarray()
        lt_matrix = pd.DataFrame(
            _lt_data,
            index=lt_target_genes,
            columns=lt_ligand_genes,
        )

    weighted_networks_lr = weighted_networks["lr_sig"].copy()
    # Filter to known LR pairs
    lr_pairs = lr_net[["ligand", "receptor"]].drop_duplicates()
    if "from" in weighted_networks_lr.columns:
        weighted_networks_lr = weighted_networks_lr.merge(
            lr_pairs.rename(columns={"ligand": "from", "receptor": "to"}),
            on=["from", "to"],
            how="inner",
        )
    else:
        weighted_networks_lr = weighted_networks_lr.rename(
            columns={"ligand": "from", "receptor": "to"}
        )
        weighted_networks_lr = weighted_networks_lr.merge(
            lr_pairs.rename(columns={"ligand": "from", "receptor": "to"}),
            on=["from", "to"],
            how="inner",
        )

    ligands = lr_net["ligand"].unique().tolist()
    receptors = lr_net["receptor"].unique().tolist()

    if verbose:
        print("Define expressed ligands and receptors in receiver and sender cells")

    # Step 1: Get expressed genes
    sender_celltypes = _resolve_sender_celltypes(sender, adata, celltype_col)

    expressed_genes_receiver = _get_expressed_genes_for_celltypes(
        receivers, adata, celltype_col, expression_pct, assay_oi
    )

    if sender == "undefined" or (isinstance(sender, str) and sender == "undefined"):
        all_genes = set(adata.var_names.tolist())
        all_genes.update(lt_target_genes)
        all_genes.update(lt_ligand_genes)
        expressed_genes_sender = sorted(all_genes)
    else:
        expressed_genes_sender = _get_expressed_genes_for_celltypes(
            sender_celltypes, adata, celltype_col, expression_pct, assay_oi
        )

    # Step 2: DE in receiver between conditions
    if verbose:
        print("Perform DE analysis in receiver cell")

    de_table = _compute_de_between_conditions(
        adata, celltype_col, receivers, condition_col,
        condition_oi, condition_ref, expression_pct, assay_oi,
    )

    if geneset == "DE":
        geneset_oi = de_table.loc[
            (de_table["p_val_adj"] <= 0.05) & (de_table["avg_log2FC"].abs() >= lfc_cutoff),
            "gene",
        ].tolist()
    elif geneset == "up":
        geneset_oi = de_table.loc[
            (de_table["p_val_adj"] <= 0.05) & (de_table["avg_log2FC"] >= lfc_cutoff),
            "gene",
        ].tolist()
    else:  # down
        geneset_oi = de_table.loc[
            (de_table["p_val_adj"] <= 0.05) & (de_table["avg_log2FC"] <= -lfc_cutoff),
            "gene",
        ].tolist()

    geneset_oi = [g for g in geneset_oi if g in lt_target_genes]
    if len(geneset_oi) == 0:
        raise ValueError("No genes were differentially expressed.")

    background_expressed_genes = [
        g for g in expressed_genes_receiver if g in lt_target_genes
    ]

    # Step 3: Define potential ligands
    expressed_ligands = sorted(set(ligands) & set(expressed_genes_sender))
    expressed_receptors = sorted(set(receptors) & set(expressed_genes_receiver))
    if len(expressed_ligands) == 0:
        raise ValueError("No ligands expressed in sender cells.")
    if len(expressed_receptors) == 0:
        raise ValueError("No receptors expressed in receiver cells.")

    potential_ligands = sorted(
        lr_net.loc[
            lr_net["ligand"].isin(expressed_ligands)
            & lr_net["receptor"].isin(expressed_receptors),
            "ligand",
        ].unique()
    )
    if len(potential_ligands) == 0:
        raise ValueError("No potentially active ligands.")

    # Step 4: Ligand activity analysis
    if verbose:
        print("Perform NicheNet ligand activity analysis")

    lig_activities = predict_ligand_activities(
        geneset=geneset_oi,
        background_expressed_genes=background_expressed_genes,
        ligand_target_matrix=lt_matrix_original,
        potential_ligands=potential_ligands,
    )
    lig_activities = lig_activities.sort_values("aupr_corrected", ascending=False)
    lig_activities["rank"] = lig_activities["aupr_corrected"].rank(
        ascending=False, method="average"
    )

    if filter_top_ligands:
        best_upstream_ligands = (
            lig_activities.nlargest(top_n_ligands, "aupr_corrected")["test_ligand"]
            .unique()
            .tolist()
        )
    else:
        best_upstream_ligands = lig_activities["test_ligand"].unique().tolist()

    # Step 5: Infer target genes
    if verbose:
        print("Infer active target genes of the prioritized ligands")

    target_links: list[pd.DataFrame] = []
    for lig in best_upstream_ligands:
        tl = get_weighted_ligand_target_links(
            lig,
            geneset=geneset_oi,
            ligand_target_matrix=lt_matrix_original,
            n=top_n_targets,
        )
        if tl is not None and len(tl) > 0:
            target_links.append(tl)

    if target_links:
        active_lt_df = pd.concat(target_links, ignore_index=True).dropna()
    else:
        active_lt_df = pd.DataFrame(columns=["ligand", "target", "weight"])

    vis_ligand_target = None
    if len(active_lt_df) > 0:
        active_lt_raw = prepare_ligand_target_visualization(
            ligand_target_df=active_lt_df,
            ligand_target_matrix=lt_matrix_original,
            cutoff=cutoff_visualization,
        )
        # Convert _NamedArray to DataFrame for indexing
        if active_lt_raw is not None and active_lt_raw.shape[0] > 0:
            _rn = getattr(active_lt_raw, "rownames", None)
            _cn = getattr(active_lt_raw, "colnames", None)
            if _rn is not None and _cn is not None:
                active_lt_links = pd.DataFrame(
                    np.asarray(active_lt_raw), index=_rn, columns=_cn
                )
            else:
                active_lt_links = pd.DataFrame(np.asarray(active_lt_raw))
        else:
            active_lt_links = pd.DataFrame()

        if active_lt_links is not None and active_lt_links.shape[0] > 0:
            order_ligands = [
                l for l in reversed(best_upstream_ligands)
                if l in active_lt_links.columns
            ]
            order_targets = [
                t
                for t in active_lt_df["target"].unique()
                if t in active_lt_links.index
            ]
            if order_targets and order_ligands:
                vis_ligand_target = active_lt_links.loc[order_targets, order_ligands].T

    # Ligand-receptor network
    if verbose:
        print("Infer receptors of the prioritized ligands")

    lr_network_top = lr_net.loc[
        lr_net["ligand"].isin(best_upstream_ligands)
        & lr_net["receptor"].isin(expressed_receptors)
    ][["ligand", "receptor"]].drop_duplicates()
    best_upstream_receptors = lr_network_top["receptor"].unique().tolist()

    lr_top_large = weighted_networks_lr.loc[
        weighted_networks_lr["from"].isin(best_upstream_ligands)
        & weighted_networks_lr["to"].isin(best_upstream_receptors)
    ]

    vis_lr_network = None
    order_receptors_final: list[str] = []
    order_ligands_receptor: list[str] = []
    if len(lr_top_large) > 0:
        vis_lr_network, order_receptors_final, order_ligands_receptor = (
            _build_lr_receptor_matrix(lr_top_large)
        )

    # Sender DE analysis (LFC per sender cell type)
    ligand_activities_de = None
    lfc_matrix = None
    are_there_senders = len(sender_celltypes) > 0

    if are_there_senders:
        if verbose:
            print("Perform DE analysis in sender cells")

        lfc_table = get_lfc_celltype(
            adata,
            celltype_col=celltype_col,
            senders=sender_celltypes,
            condition_col=condition_col,
            condition_oi=condition_oi,
            condition_ref=condition_ref,
            ligands_oi=potential_ligands,
            celltype_specificity=True,
            layer=assay_oi,
        )
        lfc_table = lfc_table.fillna(0.0)

        # Combine with ligand activities
        la_subset = lig_activities[["test_ligand", "pearson"]].rename(
            columns={"test_ligand": "gene"}
        )
        ligand_activities_de = la_subset.merge(lfc_table, on="gene", how="left").fillna(0.0)

        # Build LFC matrix
        lfc_cols = [c for c in lfc_table.columns if c != "gene"]
        if lfc_cols:
            lfc_matrix = ligand_activities_de.set_index("gene")[lfc_cols]

    # Build ligand-receptor DataFrame
    lr_df = lr_top_large.rename(columns={"from": "ligand", "to": "receptor"})

    return {
        "ligand_activities": lig_activities,
        "top_ligands": best_upstream_ligands,
        "top_targets": active_lt_df["target"].unique().tolist() if len(active_lt_df) > 0 else [],
        "top_receptors": lr_top_large["to"].unique().tolist() if len(lr_top_large) > 0 else [],
        "ligand_target_matrix": vis_ligand_target,
        "ligand_target_df": active_lt_df,
        "ligand_receptor_matrix": vis_lr_network,
        "ligand_receptor_df": lr_df,
        "ligand_receptor_heatmap": None,  # No ggplot equivalent; users can use seaborn/matplotlib
        "ligand_target_heatmap": None,
        "ligand_expression_dotplot": None,
        "ligand_differential_expression_heatmap": lfc_matrix,
        "ligand_activity_target_heatmap": None,
        "geneset_oi": geneset_oi,
        "background_expressed_genes": background_expressed_genes,
    }


def nichenet_seuratobj_cluster_de(
    receiver_affected: Union[str, Sequence[str]],
    receiver_reference: Union[str, Sequence[str]],
    adata: ad.AnnData,
    condition_col: str,
    condition_oi: str,
    condition_ref: str,
    sender: Union[str, Sequence[str]] = "all",
    celltype_col: Optional[str] = None,
    ligand_target_matrix: Optional[Any] = None,
    lr_network: Optional[pd.DataFrame] = None,
    weighted_networks: Optional[Dict[str, pd.DataFrame]] = None,
    expression_pct: float = 0.10,
    lfc_cutoff: float = 0.25,
    top_n_ligands: int = 30,
    top_n_targets: int = 200,
    cutoff_visualization: float = 0.33,
    geneset: str = "DE",
    filter_top_ligands: bool = True,
    assay_oi: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run NicheNet where DE is between two cell populations from different
    conditions.

    ``receiver_affected`` cells from *condition_oi* are compared against
    ``receiver_reference`` cells from *condition_ref*.  This is the Python
    equivalent of R ``nichenet_seuratobj_aggregate_cluster_de``.

    Parameters
    ----------
    receiver_affected : str or sequence of str
        Cell type(s) in the affected/treated condition.
    receiver_reference : str or sequence of str
        Cell type(s) in the reference/control condition.
    adata : anndata.AnnData
        Annotated single-cell data matrix.
    condition_col : str
        Column in ``adata.obs`` indicating the experimental condition.
    condition_oi : str
        Condition of interest.
    condition_ref : str
        Reference condition.
    sender : str or sequence of str, optional
        ``"all"``, ``"undefined"``, or explicit cell type(s).
    celltype_col : str or None, optional
        Column in ``adata.obs`` with cell-type labels.
    ligand_target_matrix : array-like or None, optional
        Pre-loaded ligand-target matrix.
    lr_network : pd.DataFrame or None, optional
        Ligand-receptor network.
    weighted_networks : dict or None, optional
        Weighted networks dict.
    expression_pct : float, optional
        Minimum expression fraction.  Default ``0.10``.
    lfc_cutoff : float, optional
        Minimum absolute log2 fold-change.  Default ``0.25``.
    top_n_ligands : int, optional
        Number of top ligands.  Default ``30``.
    top_n_targets : int, optional
        Number of top targets per ligand.  Default ``200``.
    cutoff_visualization : float, optional
        Regulatory potential cutoff.  Default ``0.33``.
    geneset : str, optional
        ``"DE"``, ``"up"``, or ``"down"``.  Default ``"DE"``.
    filter_top_ligands : bool, optional
        Whether to keep only top ligands.  Default ``True``.
    assay_oi : str or None, optional
        Layer to use.
    verbose : bool, optional
        Print progress.  Default ``True``.

    Returns
    -------
    dict
        Same structure as :func:`nichenet_seuratobj_aggregate`.

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    if geneset not in ("DE", "up", "down"):
        raise ValueError("geneset must be 'DE', 'up', or 'down'.")

    if celltype_col is None:
        raise ValueError(
            "celltype_col must be provided.  Specify the column in "
            "adata.obs that contains cell-type labels."
        )

    affected = (
        [receiver_affected] if isinstance(receiver_affected, str) else list(receiver_affected)
    )
    reference = (
        [receiver_reference] if isinstance(receiver_reference, str) else list(receiver_reference)
    )

    # Auto-load
    if ligand_target_matrix is None:
        ligand_target_matrix = load_ligand_target_matrix()
    if lr_network is None:
        lr_network = load_lr_network()
    if weighted_networks is None:
        weighted_networks = load_weighted_networks()

    lr_net = lr_network.copy()
    if "from" in lr_net.columns and "to" in lr_net.columns:
        if "ligand" not in lr_net.columns:
            lr_net = lr_net.rename(columns={"from": "ligand", "to": "receptor"})

    if isinstance(ligand_target_matrix, pd.DataFrame):
        lt_matrix = ligand_target_matrix
        lt_target_genes = lt_matrix.index.tolist()
        lt_ligand_genes = lt_matrix.columns.tolist()
        from .datasets import NamedMatrix as _NM
        lt_matrix_original = _NM(
            data=scipy.sparse.csr_matrix(lt_matrix.values),
            rownames=lt_target_genes,
            colnames=lt_ligand_genes,
        )
    else:
        # Assume a NamedMatrix (has .rownames, .colnames, .data)
        lt_matrix_original = ligand_target_matrix
        lt_target_genes = list(ligand_target_matrix.rownames)
        lt_ligand_genes = list(ligand_target_matrix.colnames)
        _lt_data = ligand_target_matrix.data
        if hasattr(_lt_data, "toarray"):
            _lt_data = _lt_data.toarray()
        lt_matrix = pd.DataFrame(
            _lt_data,
            index=lt_target_genes,
            columns=lt_ligand_genes,
        )

    weighted_networks_lr = weighted_networks["lr_sig"].copy()
    lr_pairs = lr_net[["ligand", "receptor"]].drop_duplicates()
    if "from" in weighted_networks_lr.columns:
        weighted_networks_lr = weighted_networks_lr.merge(
            lr_pairs.rename(columns={"ligand": "from", "receptor": "to"}),
            on=["from", "to"],
            how="inner",
        )
    else:
        weighted_networks_lr = weighted_networks_lr.rename(
            columns={"ligand": "from", "receptor": "to"}
        )
        weighted_networks_lr = weighted_networks_lr.merge(
            lr_pairs.rename(columns={"ligand": "from", "receptor": "to"}),
            on=["from", "to"],
            how="inner",
        )

    ligands = lr_net["ligand"].unique().tolist()
    receptors = lr_net["receptor"].unique().tolist()

    sender_celltypes = _resolve_sender_celltypes(sender, adata, celltype_col)

    if verbose:
        print("Define expressed ligands and receptors in receiver and sender cells")

    # Expressed genes: reference receivers (for determining receptors)
    expressed_genes_receiver_ref = _get_expressed_genes_for_celltypes(
        reference, adata, celltype_col, expression_pct, assay_oi
    )

    # Expressed genes: both affected + reference receivers (for background)
    all_receivers = sorted(set(affected + reference))
    expressed_genes_receiver = _get_expressed_genes_for_celltypes(
        all_receivers, adata, celltype_col, expression_pct, assay_oi
    )

    # Expressed genes: senders
    if sender == "undefined" or (isinstance(sender, str) and sender == "undefined"):
        all_genes = set(adata.var_names.tolist())
        all_genes.update(lt_target_genes)
        all_genes.update(lt_ligand_genes)
        expressed_genes_sender = sorted(all_genes)
    else:
        expressed_genes_sender = _get_expressed_genes_for_celltypes(
            sender_celltypes, adata, celltype_col, expression_pct, assay_oi
        )

    # Step 2: DE between the two receiver populations across conditions
    if verbose:
        print("Perform DE analysis between two receiver cell clusters")

    # Subset: affected cells from condition_oi
    mask_aff = (
        adata.obs[celltype_col].astype(str).isin(affected)
        & (adata.obs[condition_col].astype(str) == str(condition_oi))
    )
    # Subset: reference cells from condition_ref
    mask_ref = (
        adata.obs[celltype_col].astype(str).isin(reference)
        & (adata.obs[condition_col].astype(str) == str(condition_ref))
    )

    sub = adata[mask_aff | mask_ref].copy()
    if assay_oi is not None and assay_oi in sub.layers:
        sub.X = sub.layers[assay_oi]

    sub.obs["_cond"] = sub.obs[condition_col].astype(str).values

    sc.tl.rank_genes_groups(
        sub,
        groupby="_cond",
        groups=[str(condition_oi)],
        reference=str(condition_ref),
        method="wilcoxon",
        pts=True,
    )
    de_res = sc.get.rank_genes_groups_df(sub, group=str(condition_oi))
    de_res = de_res.rename(
        columns={
            "names": "gene",
            "pvals": "p_val",
            "logfoldchanges": "avg_log2FC",
            "pvals_adj": "p_val_adj",
        }
    )

    if geneset == "DE":
        geneset_oi = de_res.loc[
            (de_res["p_val_adj"] <= 0.05) & (de_res["avg_log2FC"].abs() >= lfc_cutoff),
            "gene",
        ].tolist()
    elif geneset == "up":
        geneset_oi = de_res.loc[
            (de_res["p_val_adj"] <= 0.05) & (de_res["avg_log2FC"] >= lfc_cutoff),
            "gene",
        ].tolist()
    else:
        geneset_oi = de_res.loc[
            (de_res["p_val_adj"] <= 0.05) & (de_res["avg_log2FC"] <= -lfc_cutoff),
            "gene",
        ].tolist()

    geneset_oi = [g for g in geneset_oi if g in lt_target_genes]
    if len(geneset_oi) == 0:
        raise ValueError("No genes were differentially expressed.")

    background_expressed_genes = [
        g for g in expressed_genes_receiver if g in lt_target_genes
    ]

    # Step 3: Potential ligands
    expressed_ligands = sorted(set(ligands) & set(expressed_genes_sender))
    expressed_receptors = sorted(set(receptors) & set(expressed_genes_receiver))
    if not expressed_ligands:
        raise ValueError("No ligands expressed in sender cells.")
    if not expressed_receptors:
        raise ValueError("No receptors expressed in receiver cells.")

    potential_ligands = sorted(
        lr_net.loc[
            lr_net["ligand"].isin(expressed_ligands)
            & lr_net["receptor"].isin(expressed_receptors),
            "ligand",
        ].unique()
    )
    if not potential_ligands:
        raise ValueError("No potentially active ligands.")

    # Step 4: Ligand activity
    if verbose:
        print("Perform NicheNet ligand activity analysis")

    lig_activities = predict_ligand_activities(
        geneset=geneset_oi,
        background_expressed_genes=background_expressed_genes,
        ligand_target_matrix=lt_matrix_original,
        potential_ligands=potential_ligands,
    )
    lig_activities = lig_activities.sort_values("aupr_corrected", ascending=False)
    lig_activities["rank"] = lig_activities["aupr_corrected"].rank(
        ascending=False, method="average"
    )

    if filter_top_ligands:
        best_upstream_ligands = (
            lig_activities.nlargest(top_n_ligands, "aupr_corrected")["test_ligand"]
            .unique()
            .tolist()
        )
    else:
        best_upstream_ligands = lig_activities["test_ligand"].unique().tolist()

    # Step 5: Target genes
    if verbose:
        print("Infer active target genes of the prioritized ligands")

    target_links: list[pd.DataFrame] = []
    for lig in best_upstream_ligands:
        tl = get_weighted_ligand_target_links(
            lig,
            geneset=geneset_oi,
            ligand_target_matrix=lt_matrix_original,
            n=top_n_targets,
        )
        if tl is not None and len(tl) > 0:
            target_links.append(tl)

    if target_links:
        active_lt_df = pd.concat(target_links, ignore_index=True).dropna()
    else:
        active_lt_df = pd.DataFrame(columns=["ligand", "target", "weight"])

    vis_ligand_target = None
    if len(active_lt_df) > 0:
        active_lt_raw = prepare_ligand_target_visualization(
            ligand_target_df=active_lt_df,
            ligand_target_matrix=lt_matrix_original,
            cutoff=cutoff_visualization,
        )
        if active_lt_raw is not None and active_lt_raw.shape[0] > 0:
            _rn = getattr(active_lt_raw, "rownames", None)
            _cn = getattr(active_lt_raw, "colnames", None)
            if _rn is not None and _cn is not None:
                active_lt_links = pd.DataFrame(
                    np.asarray(active_lt_raw), index=_rn, columns=_cn
                )
            else:
                active_lt_links = pd.DataFrame(np.asarray(active_lt_raw))
        else:
            active_lt_links = pd.DataFrame()

        if active_lt_links is not None and active_lt_links.shape[0] > 0:
            order_ligands = [
                l for l in reversed(best_upstream_ligands)
                if l in active_lt_links.columns
            ]
            order_targets = [
                t
                for t in active_lt_df["target"].unique()
                if t in active_lt_links.index
            ]
            if order_targets and order_ligands:
                vis_ligand_target = active_lt_links.loc[order_targets, order_ligands].T

    # Ligand-receptor network
    if verbose:
        print("Infer receptors of the prioritized ligands")

    lr_network_top = lr_net.loc[
        lr_net["ligand"].isin(best_upstream_ligands)
        & lr_net["receptor"].isin(expressed_receptors)
    ][["ligand", "receptor"]].drop_duplicates()
    best_upstream_receptors = lr_network_top["receptor"].unique().tolist()

    lr_top_large = weighted_networks_lr.loc[
        weighted_networks_lr["from"].isin(best_upstream_ligands)
        & weighted_networks_lr["to"].isin(best_upstream_receptors)
    ]

    vis_lr_network = None
    if len(lr_top_large) > 0:
        vis_lr_network, _, _ = _build_lr_receptor_matrix(lr_top_large)

    lr_df = lr_top_large.rename(columns={"from": "ligand", "to": "receptor"})

    # Sender dotplot / LFC  -- only if senders are defined
    lfc_matrix = None
    are_there_senders = len(sender_celltypes) > 0
    if are_there_senders:
        lfc_table = get_lfc_celltype(
            adata,
            celltype_col=celltype_col,
            senders=sender_celltypes,
            condition_col=condition_col,
            condition_oi=condition_oi,
            condition_ref=condition_ref,
            ligands_oi=potential_ligands,
            celltype_specificity=True,
            layer=assay_oi,
        )
        lfc_table = lfc_table.fillna(0.0)
        lfc_cols = [c for c in lfc_table.columns if c != "gene"]
        if lfc_cols:
            lfc_matrix = lfc_table.set_index("gene")[lfc_cols]

    return {
        "ligand_activities": lig_activities,
        "top_ligands": best_upstream_ligands,
        "top_targets": active_lt_df["target"].unique().tolist() if len(active_lt_df) > 0 else [],
        "top_receptors": lr_top_large["to"].unique().tolist() if len(lr_top_large) > 0 else [],
        "ligand_target_matrix": vis_ligand_target,
        "ligand_target_df": active_lt_df,
        "ligand_receptor_matrix": vis_lr_network,
        "ligand_receptor_df": lr_df,
        "ligand_receptor_heatmap": None,
        "ligand_target_heatmap": None,
        "ligand_expression_dotplot": None,
        "ligand_differential_expression_heatmap": lfc_matrix,
        "ligand_activity_target_heatmap": None,
        "geneset_oi": geneset_oi,
        "background_expressed_genes": background_expressed_genes,
    }
