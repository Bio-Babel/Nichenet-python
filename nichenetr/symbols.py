"""Gene symbol utilities: alias conversion, expressed-gene detection, and
cell-type--ligand assignment.

This module is the Python equivalent of symbol- and expression-related
helpers found in the R *nichenetr* package (``supporting_functions.R`` and
``application_prediction.R``).  Seurat objects are replaced by
:class:`anndata.AnnData` throughout.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from .datasets import load_geneinfo_alias

__all__ = [
    "get_expressed_genes",
    "alias_to_symbol_anndata",
    "convert_alias_to_symbols",
    "assign_ligands_to_celltype",
    "get_lfc_celltype",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_expression_matrix(
    adata: ad.AnnData,
    layer: Optional[str] = None,
) -> "sparse.spmatrix | np.ndarray":
    """Return the expression matrix (cells x genes) from *adata*.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    layer : str or None
        Layer to use.  ``None`` means ``adata.X``.

    Returns
    -------
    matrix
        Expression matrix with shape ``(n_obs, n_vars)``.
    """
    if layer is not None:
        return adata.layers[layer]
    return adata.X


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_expressed_genes(
    adata: ad.AnnData,
    celltype_col: str,
    celltype: str,
    pct: float = 0.10,
    assay_oi: Optional[str] = None,
) -> list[str]:
    """Return genes expressed in at least *pct* fraction of cells of a given
    cell type.

    This is the Python equivalent of ``get_expressed_genes`` in R nichenetr,
    which operates on Seurat objects.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated single-cell data matrix (cells x genes).
    celltype_col : str
        Column in ``adata.obs`` that stores cell-type labels.
    celltype : str
        Cell type of interest.  Must be a value present in
        ``adata.obs[celltype_col]``.
    pct : float, optional
        Minimum fraction of cells that must express a gene (expression > 0)
        for it to be considered "expressed".  Default is ``0.10``.
    assay_oi : str or None, optional
        Layer in *adata* to use for the expression matrix.  ``None`` (the
        default) uses ``adata.X``.

    Returns
    -------
    list[str]
        Gene names that pass the expression threshold.

    Raises
    ------
    KeyError
        If *celltype_col* is not in ``adata.obs``.
    ValueError
        If *celltype* is not found in ``adata.obs[celltype_col]``.
    """
    if celltype_col not in adata.obs.columns:
        raise KeyError(
            f"Column {celltype_col!r} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    if celltype not in adata.obs[celltype_col].values:
        raise ValueError(
            f"Cell type {celltype!r} is not present in "
            f"adata.obs[{celltype_col!r}]."
        )

    mask = adata.obs[celltype_col].values == celltype
    sub = adata[mask]
    mat = _get_expression_matrix(sub, layer=assay_oi)

    n_cells = mat.shape[0]
    if n_cells == 0:
        return []

    # Fraction of cells with expression > 0 per gene
    if sparse.issparse(mat):
        nonzero_counts = np.asarray((mat > 0).sum(axis=0)).ravel()
    else:
        nonzero_counts = np.asarray((np.asarray(mat) > 0).sum(axis=0)).ravel()

    fractions = nonzero_counts / n_cells
    gene_mask = fractions >= pct
    return list(adata.var_names[gene_mask])


def convert_alias_to_symbols(
    genes: Sequence[str],
    organism: str = "human",
    verbose: bool = True,
) -> list[str]:
    """Convert gene aliases to official gene symbols.

    Uses the bundled ``geneinfo_alias`` annotation table.  Genes that are not
    found in the alias table are kept as-is.

    Parameters
    ----------
    genes : sequence of str
        Gene names (symbols or aliases) to convert.
    organism : str, optional
        ``"human"`` or ``"mouse"``.  Default is ``"human"``.
    verbose : bool, optional
        If ``True``, print information about symbols that could not be
        mapped or that were converted.  Default is ``True``.

    Returns
    -------
    list[str]
        Official gene symbols in the same order as *genes*.
    """
    genes = list(genes)
    if not genes:
        return []

    geneinfo = load_geneinfo_alias(organism=organism)

    # Build alias -> symbol mapping (first occurrence wins, matching R
    # behaviour of ``mapper`` which uses ``setNames``).
    alias_col = "alias"
    symbol_col = "symbol"
    # Keep first mapping per alias (duplicates are resolved to the first row)
    deduped = geneinfo.drop_duplicates(subset=alias_col, keep="first")
    alias2symbol: dict[str, str] = dict(
        zip(deduped[alias_col], deduped[symbol_col])
    )

    # Identify orphan aliases (not in annotation table)
    orphan_aliases = [g for g in genes if g not in alias2symbol]
    if orphan_aliases:
        if verbose:
            print(
                "There are provided symbols that are not in the alias "
                "annotation table:"
            )
            print(orphan_aliases)
            print(
                "They are added to the alias annotation table so they "
                "don't get lost."
            )
        # Map orphans to themselves
        for alias in orphan_aliases:
            alias2symbol[alias] = alias

    converted = [alias2symbol.get(g, g) for g in genes]

    if verbose:
        changed = [g for g, c in zip(genes, converted) if g != c]
        if not changed:
            print("All input symbols were official symbols.")
        else:
            rows = geneinfo[geneinfo[alias_col].isin(changed)][
                [symbol_col, alias_col]
            ].drop_duplicates()
            print("Following are the official gene symbols of input aliases:")
            print(rows.to_string(index=False))

    return converted


def alias_to_symbol_anndata(
    adata: ad.AnnData,
    organism: str = "mouse",
) -> ad.AnnData:
    """Convert gene aliases to official symbols in an AnnData object.

    This is the Python equivalent of ``alias_to_symbol_seurat``.  Gene names
    in ``adata.var_names`` are converted using :func:`convert_alias_to_symbols`.
    When a conversion would create duplicate gene names, the original alias is
    retained to avoid collisions.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix whose ``var_names`` may contain gene aliases.
    organism : str, optional
        ``"human"`` or ``"mouse"``.  Default is ``"mouse"``.

    Returns
    -------
    anndata.AnnData
        A **copy** of *adata* with updated ``var_names``.
    """
    adata = adata.copy()
    feature_names = list(adata.var_names)
    newnames = convert_alias_to_symbols(
        feature_names, organism=organism, verbose=False
    )

    # Resolve duplicates: if a converted name already appears more than once,
    # revert the alias entries back to their original names (matching R logic).
    name_counts: dict[str, int] = {}
    for n in newnames:
        name_counts[n] = name_counts.get(n, 0) + 1
    doubles = {n for n, c in name_counts.items() if c > 1}

    if doubles:
        for i, (old, new) in enumerate(zip(feature_names, newnames)):
            if new in doubles and old != new:
                newnames[i] = old

    adata.var_names = pd.Index(newnames)
    adata.var_names_make_unique()
    return adata


def assign_ligands_to_celltype(
    adata: ad.AnnData,
    celltype_col: str,
    sender_celltypes: Sequence[str],
    ligands_oi: Sequence[str],
    celltype_specificity: bool = True,
    layer: Optional[str] = None,
    func_agg: Callable[[np.ndarray], float] = np.mean,
    func_assign: Optional[Callable[[np.ndarray], float]] = None,
) -> pd.DataFrame:
    """Assign each ligand to the sender cell type where it is most highly
    expressed.

    Ligands whose expression exceeds a threshold (default:
    ``mean + sd`` of expression across sender cell types) in exactly one
    cell type are assigned to that cell type.  Ligands that exceed the
    threshold in zero or multiple cell types are labelled ``"General"``.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated single-cell data matrix.
    celltype_col : str
        Column in ``adata.obs`` containing cell-type labels.
    sender_celltypes : sequence of str
        Sender cell types to consider.
    ligands_oi : sequence of str
        Ligands of interest (must be present in ``adata.var_names``).
    celltype_specificity : bool, optional
        If ``True`` (default), assign ligands to specific cell types.  If
        ``False``, all ligands are labelled ``"General"``.
    layer : str or None, optional
        Layer to use for expression values.  ``None`` uses ``adata.X``.
    func_agg : callable, optional
        Aggregation function applied per gene across cells of a cell type.
        Default is :func:`numpy.mean`.
    func_assign : callable or None, optional
        Threshold function applied to the vector of aggregated expression
        values across cell types.  A ligand is assigned to a cell type if
        its aggregated expression exceeds this threshold.  Default is
        ``lambda x: np.mean(x) + np.std(x)``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``ligand_type`` and ``ligand``.
    """
    if func_assign is None:
        func_assign = lambda x: np.mean(x) + np.std(x)  # noqa: E731

    ligands_oi = list(ligands_oi)
    sender_celltypes = list(sender_celltypes)

    # Validate ligands
    missing = [l for l in ligands_oi if l not in adata.var_names]
    if missing:
        raise ValueError(
            f"The following ligands are not in adata.var_names: {missing}"
        )

    if not celltype_specificity:
        return pd.DataFrame(
            {"ligand_type": "General", "ligand": ligands_oi}
        )

    # Subset to sender cell types
    mask = adata.obs[celltype_col].isin(sender_celltypes)
    sub = adata[mask]

    # Compute average expression per cell type for each ligand
    ligand_indices = [list(adata.var_names).index(l) for l in ligands_oi]

    avg_expr = {}  # celltype -> array of avg expression per ligand
    for ct in sender_celltypes:
        ct_mask = sub.obs[celltype_col].values == ct
        ct_sub = sub[ct_mask]
        mat = _get_expression_matrix(ct_sub, layer=layer)
        # Extract ligand columns
        if sparse.issparse(mat):
            ligand_mat = mat[:, ligand_indices].toarray()
        else:
            ligand_mat = np.asarray(mat[:, ligand_indices])
        avg_expr[ct] = np.array([func_agg(ligand_mat[:, i]) for i in range(len(ligands_oi))])

    # Build matrix: rows = ligands, cols = celltypes
    ct_names = list(avg_expr.keys())
    avg_matrix = np.column_stack([avg_expr[ct] for ct in ct_names])

    # For each ligand, determine which cell types exceed the threshold
    assignment = {}  # celltype -> list of ligand names
    for i, ligand in enumerate(ligands_oi):
        row = avg_matrix[i, :]
        threshold = func_assign(row)
        exceeds = row > threshold
        for j, ct in enumerate(ct_names):
            if exceeds[j]:
                assignment.setdefault(ct, []).append(ligand)

    # Find ligands assigned to exactly one cell type
    all_assigned: dict[str, int] = {}
    for ct, ligs in assignment.items():
        for lig in ligs:
            all_assigned[lig] = all_assigned.get(lig, 0) + 1

    unique_ligands = {lig for lig, cnt in all_assigned.items() if cnt == 1}
    general_ligands = [l for l in ligands_oi if l not in unique_ligands]

    rows = []
    for ct, ligs in assignment.items():
        for lig in ligs:
            if lig in unique_ligands:
                rows.append({"ligand_type": ct, "ligand": lig})

    for lig in general_ligands:
        rows.append({"ligand_type": "General", "ligand": lig})

    return pd.DataFrame(rows, columns=["ligand_type", "ligand"])


def get_lfc_celltype(
    adata: ad.AnnData,
    celltype_col: str,
    senders: Sequence[str],
    condition_col: str,
    condition_oi: str,
    condition_ref: str,
    ligands_oi: Optional[Sequence[str]] = None,
    celltype_specificity: bool = True,
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """Compute log-fold changes of genes between two conditions per sender
    cell type.

    For each sender cell type, ``scanpy.tl.rank_genes_groups`` is used to
    perform a Wilcoxon test between *condition_oi* and *condition_ref*.  The
    resulting log-fold change values are collected into a single DataFrame
    with one column per sender cell type.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated single-cell data matrix.
    celltype_col : str
        Column in ``adata.obs`` storing cell-type labels.
    senders : sequence of str
        Sender cell types to compute LFCs for.
    condition_col : str
        Column in ``adata.obs`` indicating the experimental condition.
    condition_oi : str
        Condition of interest (numerator of fold change).
    condition_ref : str
        Reference condition (denominator of fold change).
    ligands_oi : sequence of str or None, optional
        If given, restrict the output to these genes.  Default is ``None``
        (return all genes).
    celltype_specificity : bool, optional
        If ``True`` (default), compute LFCs separately for each sender cell
        type and return a wide DataFrame (one column per cell type).  If
        ``False``, compute LFCs across all sender cells pooled together,
        returning a single ``"lfc"`` column.
    layer : str or None, optional
        Layer to use.  ``None`` uses ``adata.X``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``"gene"`` column and one LFC column per sender
        cell type (or a single ``"lfc"`` column when
        ``celltype_specificity=False``).
    """
    senders = list(senders)

    def _lfc_for_subset(sub: ad.AnnData, group_col: str, label: str) -> pd.DataFrame:
        """Run rank_genes_groups and extract LFCs."""
        sub = sub.copy()
        if layer is not None and layer != "X":
            sub.X = sub.layers[layer]

        # Ensure the grouping column is categorical-friendly
        sub.obs[group_col] = sub.obs[group_col].astype(str)

        # Check that both conditions are present
        present = set(sub.obs[group_col].unique())
        if condition_oi not in present or condition_ref not in present:
            warnings.warn(
                f"Not both conditions ({condition_oi!r}, {condition_ref!r}) "
                f"are present in the subset for {label!r}. Returning empty "
                f"DataFrame."
            )
            return pd.DataFrame(columns=["gene", label])

        sc.tl.rank_genes_groups(
            sub,
            groupby=group_col,
            groups=[condition_oi],
            reference=condition_ref,
            method="wilcoxon",
            use_raw=False,
        )

        result = sc.get.rank_genes_groups_df(sub, group=condition_oi)
        df = result[["names", "logfoldchanges"]].rename(
            columns={"names": "gene", "logfoldchanges": label}
        )
        return df

    if not celltype_specificity:
        # Pool all sender cells together
        mask = adata.obs[celltype_col].isin(senders)
        sub = adata[mask]
        df = _lfc_for_subset(sub, condition_col, "lfc")
        if ligands_oi is not None:
            df = df[df["gene"].isin(ligands_oi)]
        return df.reset_index(drop=True)

    # Per-celltype LFCs
    dfs: list[pd.DataFrame] = []
    for ct in senders:
        ct_mask = adata.obs[celltype_col].values == ct
        if not ct_mask.any():
            warnings.warn(f"Cell type {ct!r} not found; skipping.")
            continue
        sub = adata[ct_mask]
        df = _lfc_for_subset(sub, condition_col, ct)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["gene"])

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="gene", how="outer")

    if ligands_oi is not None:
        merged = merged[merged["gene"].isin(ligands_oi)]

    return merged.reset_index(drop=True)
