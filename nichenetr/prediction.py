"""Ligand activity prediction functions.

Core NicheNet ligand activity analysis: predict which ligands best
explain an observed transcriptional response by computing AUROC, AUPR,
and Pearson correlation between a ligand's target‐gene regulatory
potential scores and the binary indicator of membership in a gene set
of interest.

Also provides single‐cell extensions that compute per‐cell ligand
activities, normalize them across cells, and correlate activities
with cell‐level properties.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from .datasets import NamedMatrix
from .utils import scaling_modified_zscore

__all__ = [
    "predict_ligand_activities",
    "predict_single_cell_ligand_activities",
    "normalize_single_cell_ligand_activities",
    "single_ligand_activity_score_regression",
    "convert_single_cell_expression_to_settings",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_ligand_column(
    ligand_target_matrix: NamedMatrix,
    ligand: str,
    gene_names: List[str],
) -> np.ndarray:
    """Extract a dense column vector for *ligand*, aligned to *gene_names*.

    Parameters
    ----------
    ligand_target_matrix : NamedMatrix
        Sparse matrix with target genes as rows and ligands as columns.
    ligand : str
        Name of the ligand whose column is requested.
    gene_names : list[str]
        Ordered gene names that define the desired row alignment.

    Returns
    -------
    np.ndarray
        1‑D float64 array of regulatory potential scores, one per gene
        in *gene_names*.
    """
    col_idx = ligand_target_matrix.colnames.index(ligand)
    row_lookup = {g: i for i, g in enumerate(ligand_target_matrix.rownames)}
    row_indices = [row_lookup[g] for g in gene_names]

    mat = ligand_target_matrix.data
    if scipy.sparse.issparse(mat):
        col_vec = mat[:, col_idx].toarray().ravel()
    else:
        col_vec = np.asarray(mat[:, col_idx]).ravel()

    return col_vec[row_indices].astype(np.float64)


def _single_ligand_activity(
    ligand: str,
    response: np.ndarray,
    ligand_target_matrix: NamedMatrix,
    background_genes: List[str],
    n_geneset: int,
) -> dict:
    """Compute AUROC, AUPR, corrected AUPR and Pearson for one ligand.

    Mirrors the R path: ``evaluate_target_prediction`` ->
    ``evaluate_target_prediction_strict`` ->
    ``classification_evaluation_continuous_pred``.

    Parameters
    ----------
    ligand : str
        Ligand gene symbol.
    response : np.ndarray
        Binary int array (1 = gene in geneset, 0 = background only),
        ordered to match *background_genes*.
    ligand_target_matrix : NamedMatrix
        The NicheNet ligand‐target matrix (targets x ligands).
    background_genes : list[str]
        Gene names corresponding to elements of *response*.
    n_geneset : int
        Number of genes in the gene set of interest (used for AUPR
        correction).

    Returns
    -------
    dict
        Keys: ``test_ligand``, ``auroc``, ``aupr``, ``aupr_corrected``,
        ``pearson``.
    """
    prediction = _get_ligand_column(ligand_target_matrix, ligand, background_genes)

    # Degenerate cases: constant prediction or constant response
    if np.std(prediction) == 0 or np.std(response) == 0:
        return {
            "test_ligand": ligand,
            "auroc": np.nan,
            "aupr": np.nan,
            "aupr_corrected": np.nan,
            "pearson": np.nan,
        }

    # AUROC -- sklearn uses the same definition as ROCR::performance("auc")
    auroc = roc_auc_score(response, prediction)

    # AUPR -- sklearn average_precision_score uses step‑function
    # interpolation which differs slightly from the trapezoidal rule
    # used by R caTools::trapz on the ROCR precision-recall curve.
    # We use it as the closest sklearn equivalent; the correction
    # dominates the ranking anyway.
    aupr = average_precision_score(response, prediction)

    # Corrected AUPR (subtract random baseline = geneset fraction)
    aupr_random = n_geneset / len(background_genes)
    aupr_corrected = aupr - aupr_random

    # Pearson correlation between prediction scores and binary response
    pearson, _ = pearsonr(prediction, response.astype(np.float64))

    return {
        "test_ligand": ligand,
        "auroc": float(auroc),
        "aupr": float(aupr),
        "aupr_corrected": float(aupr_corrected),
        "pearson": float(pearson),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_ligand_activities(
    geneset: Sequence[str],
    background_expressed_genes: Sequence[str],
    ligand_target_matrix: NamedMatrix,
    potential_ligands: Sequence[str],
) -> pd.DataFrame:
    """Predict ligand activities for a gene set of interest.

    For every candidate ligand, assess how well its target‐gene
    regulatory potential scores (from the NicheNet ligand‐target
    matrix) distinguish genes in *geneset* from the remaining
    *background_expressed_genes*.  Three complementary metrics are
    computed per ligand: AUROC, AUPR (with a corrected variant that
    subtracts the random baseline), and Pearson correlation.

    This is the core NicheNet analysis step.

    Parameters
    ----------
    geneset : sequence of str
        Gene symbols of the genes of interest (e.g. differentially
        expressed genes in the receiver cell).
    background_expressed_genes : sequence of str
        Gene symbols of all expressed background genes.  May contain
        genes from *geneset* as well -- they will be included on the
        positive side of the response vector.
    ligand_target_matrix : NamedMatrix
        NicheNet ligand‐target matrix with target genes as rows and
        ligands as columns.  Regulatory potential scores are used as
        the predictor for each ligand.
    potential_ligands : sequence of str
        Candidate ligand gene symbols to evaluate.

    Returns
    -------
    pandas.DataFrame
        One row per ligand with columns ``test_ligand``, ``auroc``,
        ``aupr``, ``aupr_corrected``, and ``pearson``.
    """
    geneset = list(geneset)
    background_expressed_genes = list(background_expressed_genes)
    potential_ligands = list(potential_ligands)

    # Build the response vector: union of background + geneset genes,
    # with TRUE (1) for geneset members and FALSE (0) otherwise.
    # Mirrors R convert_gene_list_settings_evaluation: background genes
    # that are also in geneset are moved to the positive class.
    geneset_set = set(geneset)
    available_rows = set(ligand_target_matrix.rownames)

    # All genes that participate: background union geneset, restricted
    # to those present in the matrix rows.
    all_genes_set = set(background_expressed_genes) | geneset_set
    all_genes = [g for g in sorted(all_genes_set) if g in available_rows]

    if not all_genes:
        raise ValueError(
            "No overlap between background/geneset genes and "
            "ligand_target_matrix row names."
        )

    response = np.array(
        [1 if g in geneset_set else 0 for g in all_genes], dtype=np.int32
    )
    n_geneset = int(response.sum())

    if n_geneset == 0:
        raise ValueError(
            "None of the geneset genes were found among the background "
            "genes / ligand_target_matrix rows."
        )

    # Filter ligands to those present in the matrix columns
    available_cols = set(ligand_target_matrix.colnames)
    ligands_to_test = [lig for lig in potential_ligands if lig in available_cols]

    if not ligands_to_test:
        raise ValueError(
            "None of the potential_ligands are present in "
            "ligand_target_matrix columns."
        )

    records = [
        _single_ligand_activity(
            lig, response, ligand_target_matrix, all_genes, n_geneset
        )
        for lig in ligands_to_test
    ]

    return pd.DataFrame(records)[
        ["test_ligand", "auroc", "aupr", "aupr_corrected", "pearson"]
    ]


def convert_single_cell_expression_to_settings(
    cell_id: str,
    expression_matrix: pd.DataFrame,
    setting_name: str,
    setting_from: Sequence[str],
    regression: bool = False,
) -> dict:
    """Convert single‐cell expression to settings format for ligand activity analysis.

    For classification mode (default), genes with expression at or
    above the 0.975 quantile in the given cell are treated as the gene
    set of interest.

    Parameters
    ----------
    cell_id : str
        Row identifier of the cell in *expression_matrix*.
    expression_matrix : pandas.DataFrame
        Scaled expression matrix with cells as rows and genes as
        columns.
    setting_name : str
        Name prefix for the setting (the cell id is appended).
    setting_from : sequence of str
        Gene symbols of the potentially active ligands.
    regression : bool, optional
        If ``True``, return continuous expression as the response
        instead of a binary indicator.  Default is ``False``.

    Returns
    -------
    dict
        Dictionary with keys ``name``, ``from``, and ``response``.
        ``response`` is a :class:`pandas.Series` indexed by gene name
        containing either ``bool`` or ``float`` values.
    """
    row = expression_matrix.loc[cell_id]

    if regression:
        response = row.astype(np.float64)
    else:
        threshold = np.quantile(row.values, 0.975)
        response = row >= threshold

    return {
        "name": f"{setting_name}_{cell_id}",
        "from": list(setting_from),
        "response": response,
    }


def predict_single_cell_ligand_activities(
    cell_ids: Sequence[str],
    expression_scaled: pd.DataFrame,
    ligand_target_matrix: NamedMatrix,
    potential_ligands: Sequence[str],
) -> pd.DataFrame:
    """Predict ligand activities for individual cells.

    For each cell, genes above the 0.975 quantile of expression are
    treated as the gene set of interest.  Ligand activity (AUROC, AUPR,
    Pearson) is then computed exactly as in
    :func:`predict_ligand_activities`.

    Parameters
    ----------
    cell_ids : sequence of str
        Cell identifiers (must be rows of *expression_scaled*).
    expression_scaled : pandas.DataFrame
        Scaled expression matrix (cells x genes).  High values indicate
        that a gene is more strongly expressed in that cell relative to
        others.
    ligand_target_matrix : NamedMatrix
        NicheNet ligand‐target matrix (targets x ligands).
    potential_ligands : sequence of str
        Candidate ligand gene symbols.

    Returns
    -------
    pandas.DataFrame
        Columns: ``setting`` (cell id), ``test_ligand``, ``auroc``,
        ``aupr``, ``pearson``.
    """
    frames: list[pd.DataFrame] = []

    for cell_id in cell_ids:
        setting = convert_single_cell_expression_to_settings(
            cell_id, expression_scaled, "", potential_ligands
        )

        # Binary response: genes above the 0.975 quantile
        response_series: pd.Series = setting["response"]
        geneset = list(response_series.index[response_series.astype(bool)])
        background = list(response_series.index)

        if len(geneset) == 0 or len(geneset) == len(background):
            # Degenerate: skip this cell (all same class)
            continue

        df = predict_ligand_activities(
            geneset=geneset,
            background_expressed_genes=background,
            ligand_target_matrix=ligand_target_matrix,
            potential_ligands=potential_ligands,
        )
        df = df.rename(columns={"aupr_corrected": "aupr_corrected"})
        df.insert(0, "setting", cell_id)
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["setting", "test_ligand", "auroc", "aupr", "pearson"]
        )

    result = pd.concat(frames, ignore_index=True)
    # Match R output column selection
    return result[["setting", "test_ligand", "auroc", "aupr", "pearson"]]


def normalize_single_cell_ligand_activities(
    ligand_activities: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize per‐cell ligand activities for cross‐cell comparison.

    Within each cell (``setting``), AUPR scores are transformed via
    the modified z‐score (median / MAD).  The result is pivoted into a
    wide table with cells as rows and ligands as columns.

    Parameters
    ----------
    ligand_activities : pandas.DataFrame
        Output of :func:`predict_single_cell_ligand_activities`.
        Must contain columns ``setting``, ``test_ligand``, and
        ``aupr``.

    Returns
    -------
    pandas.DataFrame
        Wide‐format DataFrame with a ``cell`` column followed by one
        column per ligand containing normalized AUPR values.
    """
    df = ligand_activities[["setting", "test_ligand", "aupr"]].copy()

    # Modified z‑score per cell
    df["aupr"] = df.groupby("setting")["aupr"].transform(
        lambda x: scaling_modified_zscore(x.values)
    )
    df = df.drop_duplicates(subset=["setting", "test_ligand"])
    df = df.rename(columns={"setting": "cell", "test_ligand": "ligand"})

    # Pivot to wide format: cells as rows, ligands as columns
    wide = df.pivot(index="cell", columns="ligand", values="aupr")

    # Fill missing values with the global minimum (matches R fill logic)
    global_min = wide.min().min()
    if pd.isna(global_min):
        global_min = 0.0
    wide = wide.fillna(global_min)

    # Reset index so 'cell' is a regular column
    wide = wide.reset_index()
    wide.columns.name = None

    return wide


def single_ligand_activity_score_regression(
    ligand_activities: pd.DataFrame,
    scores_tbl: pd.DataFrame,
) -> pd.DataFrame:
    """Correlate per‐cell ligand activities with a cell‐level property.

    For each ligand column in *ligand_activities*, compute Pearson and
    Spearman correlations (and simple linear‐regression statistics)
    against the numeric scores in *scores_tbl*.

    Parameters
    ----------
    ligand_activities : pandas.DataFrame
        Output of :func:`normalize_single_cell_ligand_activities`.
        Must have a ``cell`` column and one column per ligand.
    scores_tbl : pandas.DataFrame
        Two‐column DataFrame with ``cell`` (str) and ``score``
        (numeric) columns.

    Returns
    -------
    pandas.DataFrame
        One row per ligand with regression / correlation metrics
        including ``ligand``, ``pearson_regression``,
        ``spearman_regression``, ``pearson_log_pval``,
        ``spearman_log_pval``, and standard linear‐model summary
        statistics.
    """
    combined = ligand_activities.merge(scores_tbl, on="cell", how="inner")
    score = combined["score"].values.astype(np.float64)

    ligand_cols = [c for c in combined.columns if c not in ("cell", "score")]

    records: list[dict] = []
    for lig in ligand_cols:
        pred = combined[lig].values.astype(np.float64)

        # Pearson
        if np.std(pred) == 0 or np.std(score) == 0:
            pear_r, pear_p = np.nan, np.nan
            spear_r, spear_p = np.nan, np.nan
        else:
            pear_r, pear_p = pearsonr(pred, score)
            spear_r, spear_p = spearmanr(pred, score)

        # Simple linear model: score ~ pred
        n = len(pred)
        if n > 2 and np.std(pred) > 0:
            slope, intercept = np.polyfit(pred, score, 1)
            fitted = slope * pred + intercept
            residuals = score - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((score - np.mean(score)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            adj_r_squared = (
                1.0 - (1.0 - r_squared) * (n - 1) / (n - 2)
                if ss_tot > 0
                else np.nan
            )
            sigma = np.sqrt(ss_res / (n - 2)) if n > 2 else np.nan
        else:
            r_squared = np.nan
            adj_r_squared = np.nan
            sigma = np.nan

        # Clamp p‑values away from zero before log10
        min_pval = np.finfo(np.float64).tiny
        pear_p_safe = max(pear_p, min_pval) if not np.isnan(pear_p) else np.nan
        spear_p_safe = max(spear_p, min_pval) if not np.isnan(spear_p) else np.nan

        records.append(
            {
                "ligand": lig,
                "pearson_regression": float(pear_r) if not np.isnan(pear_r) else np.nan,
                "spearman_regression": float(spear_r) if not np.isnan(spear_r) else np.nan,
                "pearson_log_pval": (
                    -np.log10(pear_p_safe) if not np.isnan(pear_p_safe) else np.nan
                ),
                "spearman_log_pval": (
                    -np.log10(spear_p_safe) if not np.isnan(spear_p_safe) else np.nan
                ),
                "r_squared": r_squared,
                "adj_r_squared": adj_r_squared,
                "inverse_rmse": 1.0 / sigma if sigma and sigma > 0 else np.nan,
            }
        )

    return pd.DataFrame(records)
