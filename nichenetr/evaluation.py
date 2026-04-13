"""Evaluation utilities for NicheNet target-gene and ligand predictions.

Provides functions for assessing how well a set of ligands predicts
membership in a target gene set via random-forest cross-validation,
classification metrics (AUROC, AUPR, Pearson), fraction-of-top-predicted
analysis, Fisher enrichment testing, and settings conversion for ligand
activity prediction benchmarking.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.stats import fisher_exact, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .datasets import NamedMatrix

__all__ = [
    "assess_rf_class_probabilities",
    "classification_evaluation_continuous_pred_wrapper",
    "calculate_fraction_top_predicted",
    "calculate_fraction_top_predicted_fisher",
    "get_top_predicted_genes",
    "convert_settings_ligand_prediction",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_submatrix(
    ligand_target_matrix: NamedMatrix,
    genes: List[str],
    ligands: List[str],
) -> pd.DataFrame:
    """Extract a dense DataFrame from *ligand_target_matrix* for the given
    *genes* (rows) and *ligands* (columns), keeping only genes present in the
    matrix.

    Parameters
    ----------
    ligand_target_matrix : NamedMatrix
        Full ligand-target matrix.
    genes : list[str]
        Row (gene) names to select.
    ligands : list[str]
        Column (ligand) names to select.

    Returns
    -------
    pd.DataFrame
        Dense matrix with genes as index and ligands as columns.
    """
    row_idx_map = {name: i for i, name in enumerate(ligand_target_matrix.rownames)}
    col_idx_map = {name: i for i, name in enumerate(ligand_target_matrix.colnames)}

    col_indices = [col_idx_map[l] for l in ligands if l in col_idx_map]
    valid_ligands = [l for l in ligands if l in col_idx_map]
    row_indices = [row_idx_map[g] for g in genes if g in row_idx_map]
    valid_genes = [g for g in genes if g in row_idx_map]

    mat = ligand_target_matrix.data
    if scipy.sparse.issparse(mat):
        sub = mat[np.ix_(row_indices, col_indices)].toarray()
    else:
        sub = np.asarray(mat[np.ix_(row_indices, col_indices)])

    return pd.DataFrame(sub, index=valid_genes, columns=valid_ligands)


def _build_response_vector(
    geneset: List[str],
    background: List[str],
) -> pd.Series:
    """Return a boolean Series: True for geneset genes, False for background.

    Parameters
    ----------
    geneset : list[str]
        Genes in the target gene set.
    background : list[str]
        Background genes (those not in *geneset* will be labelled False).

    Returns
    -------
    pd.Series
        Boolean series indexed by gene name.
    """
    strict_bg = [g for g in background if g not in set(geneset)]
    genes = list(geneset) + strict_bg
    labels = [True] * len(geneset) + [False] * len(strict_bg)
    return pd.Series(labels, index=genes, name="response")


def _train_rf(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 1000,
    mtry_divisor: int = 2,
    random_state: int = 1,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier mirroring R's randomForest defaults.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (genes x ligands).
    y : pd.Series
        Binary response (True / False).
    n_estimators : int
        Number of trees.
    mtry_divisor : int
        ``max_features`` is set to ``ceil(n_features ** (1 / mtry_divisor))``.
    random_state : int
        Random seed for the forest.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    max_features = int(math.ceil(X.shape[1] ** (1.0 / mtry_divisor)))
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
    )
    clf.fit(X.values, y.values)
    return clf


def _predict_rf_prob(
    clf: RandomForestClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return predicted probability for the positive (True) class.

    Parameters
    ----------
    clf : RandomForestClassifier
        Fitted model.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        Probability of True class for each sample.
    """
    probs = clf.predict_proba(X.values)
    # Find the index of the True class
    classes = list(clf.classes_)
    true_idx = classes.index(True)
    return probs[:, true_idx]


def _rf_target_prediction_fold(
    fold_idx: int,
    geneset_grouped: List[List[str]],
    background_grouped: List[List[str]],
    ligands_oi: List[str],
    ligand_target_matrix: NamedMatrix,
) -> pd.DataFrame:
    """Run one fold of the RF cross-validation.

    Parameters
    ----------
    fold_idx : int
        Index of the held-out fold (0-based).
    geneset_grouped : list[list[str]]
        Geneset genes split into folds.
    background_grouped : list[list[str]]
        Background genes split into folds.
    ligands_oi : list[str]
        Ligands of interest (feature columns).
    ligand_target_matrix : NamedMatrix
        Full ligand-target matrix.

    Returns
    -------
    pd.DataFrame
        Columns: gene, response (bool), prediction (float).
    """
    n_folds = len(geneset_grouped)
    training_indices = [j for j in range(n_folds) if j != fold_idx]

    # Training genes
    train_geneset: list[str] = []
    for j in training_indices:
        train_geneset.extend(geneset_grouped[j])
    train_geneset = list(dict.fromkeys(train_geneset))  # unique, order-preserving

    train_background: list[str] = list(train_geneset)
    for j in training_indices:
        train_background.extend(background_grouped[j])
    train_background = list(dict.fromkeys(train_background))

    # Test genes
    test_geneset = geneset_grouped[fold_idx]
    test_background = list(dict.fromkeys(
        list(geneset_grouped[fold_idx]) + list(background_grouped[fold_idx])
    ))

    # Build training data
    train_response = _build_response_vector(train_geneset, train_background)
    train_X = _extract_submatrix(ligand_target_matrix, list(train_response.index), ligands_oi)
    # Align
    common_train = train_X.index.intersection(train_response.index)
    train_X = train_X.loc[common_train]
    train_y = train_response.loc[common_train]

    # Build test data
    test_response = _build_response_vector(test_geneset, test_background)
    test_X = _extract_submatrix(ligand_target_matrix, list(test_response.index), ligands_oi)
    common_test = test_X.index.intersection(test_response.index)
    test_X = test_X.loc[common_test]
    test_y = test_response.loc[common_test]

    # Train and predict
    clf = _train_rf(train_X, train_y)
    preds = _predict_rf_prob(clf, test_X)

    return pd.DataFrame({
        "gene": common_test.tolist(),
        "response": test_y.values.astype(bool),
        "prediction": preds,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_rf_class_probabilities(
    round_num: int,
    folds: int,
    geneset: List[str],
    background_expressed_genes: List[str],
    ligands_oi: List[str],
    ligand_target_matrix: NamedMatrix,
) -> pd.DataFrame:
    """Assess gene-set membership probability via cross-validated random forest.

    Target genes and background genes are shuffled (seeded by *round_num*),
    split into *folds* stratified groups, and a random-forest classifier is
    trained on each fold combination. Predicted class probabilities for the
    held-out fold are collected and returned.

    Parameters
    ----------
    round_num : int
        Cross-validation round number, used as the random seed for
        reproducible shuffling.
    folds : int
        Number of cross-validation folds.
    geneset : list[str]
        Genes in the target gene set (positive class).
    background_expressed_genes : list[str]
        Background expressed genes. Genes also in *geneset* are removed
        before splitting.
    ligands_oi : list[str]
        Ligands of interest (used as features from the ligand-target matrix).
    ligand_target_matrix : NamedMatrix
        Ligand-target regulatory potential matrix.

    Returns
    -------
    pd.DataFrame
        Columns: ``gene`` (str), ``response`` (bool), ``prediction`` (float).
        Each row represents a gene with its true class and predicted
        probability of belonging to the gene set.
    """
    rng = np.random.RandomState(round_num)

    # Shuffle and split geneset
    geneset_arr = list(geneset)
    rng.shuffle(geneset_arr)
    geneset_grouped = [list(g) for g in np.array_split(geneset_arr, folds)]

    # Strict background: remove geneset members
    geneset_set = set(geneset)
    strict_bg = [g for g in background_expressed_genes if g not in geneset_set]
    rng2 = np.random.RandomState(round_num)
    strict_bg_arr = list(strict_bg)
    rng2.shuffle(strict_bg_arr)
    background_grouped = [list(g) for g in np.array_split(strict_bg_arr, folds)]

    # Run each fold
    results: list[pd.DataFrame] = []
    for i in range(len(geneset_grouped)):
        fold_df = _rf_target_prediction_fold(
            i, geneset_grouped, background_grouped, ligands_oi, ligand_target_matrix
        )
        results.append(fold_df)

    return pd.concat(results, ignore_index=True)


def classification_evaluation_continuous_pred_wrapper(
    response_prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute classification evaluation metrics for continuous predictions.

    Calculates AUROC, AUPR (and corrected AUPR), Pearson and Spearman
    correlations, and their log-transformed p-values from a DataFrame of
    predicted probabilities and true binary responses.

    Parameters
    ----------
    response_prediction_df : pd.DataFrame
        Must contain columns ``prediction`` (float) and ``response``
        (bool or 0/1). May also contain a ``gene`` column (ignored).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns: ``auroc``, ``aupr``,
        ``aupr_corrected``, ``pearson``, ``pearson_log_pval``,
        ``spearman``, ``spearman_log_pval``.
    """
    prediction = response_prediction_df["prediction"].values.astype(float)
    response = response_prediction_df["response"].values.astype(float)

    # Handle degenerate cases
    if (
        len(prediction) == 0
        or np.std(response) == 0
        or np.std(prediction) == 0
    ):
        return pd.DataFrame(
            {
                "auroc": [np.nan],
                "aupr": [np.nan],
                "aupr_corrected": [np.nan],
                "pearson": [np.nan],
                "pearson_log_pval": [np.nan],
                "spearman": [np.nan],
                "spearman_log_pval": [np.nan],
            }
        )

    # Replace Inf values in prediction
    if np.any(np.isinf(prediction)):
        max_val = np.nanmax(prediction[np.isfinite(prediction)])
        prediction[np.isinf(prediction)] = max_val + 0.1

    # AUROC
    auroc = roc_auc_score(response, prediction)

    # AUPR
    aupr = average_precision_score(response, prediction)
    pos_class = np.sum(response)
    total = len(response)
    aupr_random = pos_class / total
    aupr_corrected = aupr - aupr_random

    # Pearson
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cor_p, cor_p_pval = pearsonr(prediction, response)
        cor_s, cor_s_pval = spearmanr(prediction, response)

    # Cap p-values at machine epsilon to avoid log(0)
    min_pval = np.finfo(float).tiny
    if not np.isnan(cor_p_pval) and cor_p_pval < min_pval:
        cor_p_pval = min_pval
    if not np.isnan(cor_s_pval) and cor_s_pval < min_pval:
        cor_s_pval = min_pval

    pearson_log_pval = -np.log10(cor_p_pval) if not np.isnan(cor_p_pval) else np.nan
    spearman_log_pval = -np.log10(cor_s_pval) if not np.isnan(cor_s_pval) else np.nan

    return pd.DataFrame(
        {
            "auroc": [auroc],
            "aupr": [aupr],
            "aupr_corrected": [aupr_corrected],
            "pearson": [cor_p],
            "pearson_log_pval": [pearson_log_pval],
            "spearman": [cor_s],
            "spearman_log_pval": [spearman_log_pval],
        }
    )


def calculate_fraction_top_predicted(
    round_num: int,
    response_prediction_df: pd.DataFrame,
    ligands_oi: List[str],
    ligand_target_matrix: NamedMatrix,
    classification_model: Optional[Any] = None,
    quantile_cutoff: float = 0.95,
) -> pd.DataFrame:
    """Calculate fraction of gene-set members among top-predicted genes.

    Genes are ranked by predicted probability. Those at or above the
    *quantile_cutoff* quantile are considered "top predicted". The function
    then computes, for each class (gene-set members vs. background), the
    number and fraction that fall in the top-predicted set.

    Parameters
    ----------
    round_num : int
        Cross-validation round number (stored but not used for computation
        when *response_prediction_df* is already provided).
    response_prediction_df : pd.DataFrame
        Must contain columns ``gene``, ``prediction``, and ``response``.
    ligands_oi : list[str]
        Ligands of interest (retained for API compatibility).
    ligand_target_matrix : NamedMatrix
        Ligand-target matrix (retained for API compatibility).
    classification_model : optional
        Pre-trained model. Currently unused; reserved for future extensions
        where predictions are computed on the fly.
    quantile_cutoff : float
        Quantile threshold for top-predicted genes. Default is 0.95
        (top 5 percent).

    Returns
    -------
    pd.DataFrame
        Columns: ``true_target`` (bool), ``n`` (int, total genes in class),
        ``positive_prediction`` (int, number in top-predicted set),
        ``fraction_positive_predicted`` (float).
    """
    df = response_prediction_df.copy()
    df = df.sort_values("prediction", ascending=False)
    threshold = np.quantile(df["prediction"].values, quantile_cutoff)

    top_predicted = df[df["prediction"] >= threshold]

    # Count positive predictions per class
    predicted_positive = (
        top_predicted.groupby("response")
        .size()
        .reset_index(name="positive_prediction")
        .rename(columns={"response": "true_target"})
    )

    # Count all per class
    all_counts = (
        df.groupby("response")
        .size()
        .reset_index(name="n")
        .rename(columns={"response": "true_target"})
    )

    merged = all_counts.merge(predicted_positive, on="true_target", how="inner")
    merged["fraction_positive_predicted"] = (
        merged["positive_prediction"] / merged["n"]
    )
    return merged


def calculate_fraction_top_predicted_fisher(
    round_num: int,
    response_prediction_df: pd.DataFrame,
    ligands_oi: List[str],
    ligand_target_matrix: NamedMatrix,
    classification_model: Optional[Any] = None,
    quantile_cutoff: float = 0.95,
    p_value_output: bool = True,
) -> Union[float, Dict[str, Any]]:
    """Fisher exact test for enrichment of gene-set members in top predictions.

    Constructs a 2x2 contingency table (gene-set vs. background) x
    (top-predicted vs. not-top-predicted) and performs a one-sided
    Fisher exact test (alternative = "greater").

    Parameters
    ----------
    round_num : int
        Cross-validation round number (retained for API compatibility).
    response_prediction_df : pd.DataFrame
        Must contain columns ``gene``, ``prediction``, and ``response``.
    ligands_oi : list[str]
        Ligands of interest (retained for API compatibility).
    ligand_target_matrix : NamedMatrix
        Ligand-target matrix (retained for API compatibility).
    classification_model : optional
        Pre-trained model. Currently unused; reserved for future extensions.
    quantile_cutoff : float
        Quantile threshold for top-predicted genes. Default is 0.95.
    p_value_output : bool
        If True (default), return only the p-value. If False, return a
        dict with ``oddsratio`` and ``p_value`` keys.

    Returns
    -------
    float or dict
        The Fisher exact test p-value (if *p_value_output* is True) or a
        dict with keys ``oddsratio`` and ``p_value``.
    """
    df = response_prediction_df.copy()
    df = df.sort_values("prediction", ascending=False)
    threshold = np.quantile(df["prediction"].values, quantile_cutoff)

    top_predicted = df[df["prediction"] >= threshold]

    # Counts per class among top-predicted
    pos_counts = top_predicted.groupby("response").size()
    all_counts = df.groupby("response").size()

    tp = int(pos_counts.get(True, 0))
    fp = int(pos_counts.get(False, 0))
    fn = int(all_counts.get(True, 0)) - tp
    tn = int(all_counts.get(False, 0)) - fp

    # Contingency table:
    #              top-predicted   not-top-predicted
    # geneset          tp                fn
    # background       fp                tn
    contingency = np.array([[tp, fn], [fp, tn]])
    oddsratio, p_value = fisher_exact(contingency, alternative="greater")

    if p_value_output:
        return p_value
    return {"oddsratio": oddsratio, "p_value": p_value}


def get_top_predicted_genes(
    round_num: int,
    response_prediction_df: pd.DataFrame,
    ligands_oi: List[str],
    ligand_target_matrix: NamedMatrix,
    classification_model: Optional[Any] = None,
    n: int = 250,
    quantile_cutoff: float = 0.95,
) -> pd.DataFrame:
    """Get the top-predicted genes from a cross-validation round.

    Genes whose predicted probability is at or above the *quantile_cutoff*
    quantile are returned, annotated with whether they are true gene-set
    members. If fewer than *n* genes pass the quantile threshold the
    result may contain fewer rows.

    Parameters
    ----------
    round_num : int
        Cross-validation round number (used to label the output column).
    response_prediction_df : pd.DataFrame
        Must contain columns ``gene``, ``prediction``, and ``response``.
    ligands_oi : list[str]
        Ligands of interest (retained for API compatibility).
    ligand_target_matrix : NamedMatrix
        Ligand-target matrix (retained for API compatibility).
    classification_model : optional
        Pre-trained model. Currently unused; reserved for future extensions.
    n : int
        Maximum number of top genes to return. Default is 250.
    quantile_cutoff : float
        Quantile threshold for top-predicted genes. Default is 0.95.

    Returns
    -------
    pd.DataFrame
        Columns: ``gene`` (str), ``true_target`` (bool),
        ``predicted_top_target_round{round_num}`` (bool).
        Sorted by descending prediction score, limited to *n* rows.
    """
    df = response_prediction_df.copy()
    df = df.sort_values("prediction", ascending=False)
    threshold = np.quantile(df["prediction"].values, quantile_cutoff)

    top = df[df["prediction"] >= threshold].head(n)

    col_name = f"predicted_top_target_round{round_num}"
    result = pd.DataFrame({
        "gene": top["gene"].values,
        "true_target": top["response"].values.astype(bool),
        col_name: True,
    })
    return result


def convert_settings_ligand_prediction(
    settings: List[Dict[str, Any]],
    all_ligands: List[str],
    validation: bool = True,
    single: bool = True,
) -> List[Dict[str, Any]]:
    """Convert settings to the format required for ligand activity prediction.

    Transforms a list of experimental settings into the format used by
    ligand-activity scoring functions. The output format varies depending
    on whether the goal is model validation (true active ligand known) or
    application (true ligand unknown), and whether ligands are evaluated
    individually or collectively.

    Parameters
    ----------
    settings : list[dict]
        Each dict must contain:

        - ``name`` (str): setting name.
        - ``from`` (str or list[str]): active ligand(s) in this setting.
        - ``response`` : observed target response (gene-named array/series
          indicating target membership).
    all_ligands : list[str]
        All candidate ligands to evaluate.
    validation : bool
        If True, the true active ligand is preserved in the output
        (``ligand`` key). If False, it is omitted (application mode).
    single : bool
        If True, one output entry is created per (setting, ligand) pair.
        If False, one output entry is created per setting with all ligands
        grouped together in ``from``.

    Returns
    -------
    list[dict]
        Each dict contains:

        - ``name`` (str): setting name.
        - ``from`` (str or list[str]): ligand(s) to test.
        - ``response``: observed target response.
        - ``ligand`` (str, only when *validation* is True): the true
          active ligand(s), collapsed with ``"-"`` if multiple.
    """
    if not isinstance(settings, list):
        raise TypeError("settings should be a list")
    if not isinstance(all_ligands, (list, tuple)):
        raise TypeError("all_ligands should be a list of strings")
    if not isinstance(validation, bool):
        raise TypeError("validation should be True or False")
    if not isinstance(single, bool):
        raise TypeError("single should be True or False")

    new_settings: list[dict] = []

    for setting in settings:
        from_field = setting["from"]
        # Collapse multiple ligands into a single hyphen-separated string
        if isinstance(from_field, (list, tuple)) and len(from_field) > 1:
            from_collapsed = "-".join(from_field)
        elif isinstance(from_field, (list, tuple)):
            from_collapsed = from_field[0] if from_field else ""
        else:
            from_collapsed = from_field

        if validation and single:
            for ligand in all_ligands:
                new_settings.append({
                    "name": setting["name"],
                    "ligand": from_collapsed,
                    "from": ligand,
                    "response": setting["response"],
                })
        elif validation and not single:
            new_settings.append({
                "name": setting["name"],
                "ligand": from_collapsed,
                "from": list(all_ligands),
                "response": setting["response"],
            })
        elif not validation and single:
            for ligand in all_ligands:
                new_settings.append({
                    "name": setting["name"],
                    "from": ligand,
                    "response": setting["response"],
                })
        else:  # not validation and not single
            new_settings.append({
                "name": setting["name"],
                "from": list(all_ligands),
                "response": setting["response"],
            })

    return new_settings
