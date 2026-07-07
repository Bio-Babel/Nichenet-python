"""Runnable skeleton for the nichenetr.target_prediction_evaluation workflow.

Uses a tiny synthetic ligand-target matrix and gene set (no network access,
no bundled dataset download), running 2 small cross-validation rounds.

Step mapping (nichenetr.target_prediction_evaluation workflow, run per round):
    1. assess_rf_class_probabilities              -> cross-validated RF predictions
    2. classification_evaluation_continuous_pred_wrapper -> AUROC/AUPR/Pearson
    3. calculate_fraction_top_predicted           -> fraction of geneset in top-predicted
    4. calculate_fraction_top_predicted_fisher    -> Fisher exact enrichment p-value
    5. get_top_predicted_genes                    -> merged across rounds on "gene"
"""

from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd
import scipy.sparse as sp

import nichenetr as nn


def main() -> None:
    rng = np.random.default_rng(0)

    background_expressed_genes = [f"T{i}" for i in range(1, 21)]
    geneset_oi = background_expressed_genes[:6]
    ligands_oi = ["L1", "L2", "L3"]
    ligand_target_matrix = nn.NamedMatrix(
        data=sp.csr_matrix(rng.uniform(0, 1, size=(len(background_expressed_genes), len(ligands_oi)))),
        rownames=background_expressed_genes,
        colnames=ligands_oi,
    )

    top_gene_tables = []
    for round_num in range(1, 3):
        # 1. assess_rf_class_probabilities
        predictions = nn.assess_rf_class_probabilities(
            round_num=round_num,
            folds=2,
            geneset=geneset_oi,
            background_expressed_genes=background_expressed_genes,
            ligands_oi=ligands_oi,
            ligand_target_matrix=ligand_target_matrix,
        )

        # 2. classification_evaluation_continuous_pred_wrapper
        metrics = nn.classification_evaluation_continuous_pred_wrapper(predictions)
        print(f"round {round_num} metrics:\n{metrics}")

        # 3. calculate_fraction_top_predicted
        fraction = nn.calculate_fraction_top_predicted(
            round_num=round_num,
            response_prediction_df=predictions,
            ligands_oi=ligands_oi,
            ligand_target_matrix=ligand_target_matrix,
            quantile_cutoff=0.75,
        )
        print(f"round {round_num} fraction top predicted:\n{fraction}")

        # 4. calculate_fraction_top_predicted_fisher
        fisher_p = nn.calculate_fraction_top_predicted_fisher(
            round_num=round_num,
            response_prediction_df=predictions,
            ligands_oi=ligands_oi,
            ligand_target_matrix=ligand_target_matrix,
            quantile_cutoff=0.75,
        )
        print(f"round {round_num} fisher p-value: {fisher_p}")

        # 5. get_top_predicted_genes
        top_genes = nn.get_top_predicted_genes(
            round_num=round_num,
            response_prediction_df=predictions,
            ligands_oi=ligands_oi,
            ligand_target_matrix=ligand_target_matrix,
            n=10,
            quantile_cutoff=0.75,
        )
        top_gene_tables.append(top_genes)

    merged = reduce(lambda left, right: pd.merge(left, right, on="gene", how="outer"), top_gene_tables)
    print(merged)


if __name__ == "__main__":
    main()
