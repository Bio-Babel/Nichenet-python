"""Runnable skeleton for the nichenetr.single_cell_ligand_activity workflow.

Uses a tiny synthetic raw expression matrix and ligand-target matrix (no
network access, no bundled dataset download).

Step mapping (nichenetr.single_cell_ligand_activity workflow):
    1. scale_quantile                          -> quantile-scale raw expression
    2. predict_single_cell_ligand_activities   -> per-cell ligand activity
    3. normalize_single_cell_ligand_activities -> cross-cell normalization
    4. single_ligand_activity_score_regression -> correlate with a phenotype score
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import nichenetr as nn


def main() -> None:
    rng = np.random.default_rng(0)

    target_genes = [f"T{i}" for i in range(1, 5)]
    ligand_genes = ["L1", "L2"]
    genes = target_genes + ligand_genes
    cell_ids = [f"C{i}" for i in range(1, 9)]

    raw_expression_matrix = pd.DataFrame(
        rng.uniform(0, 10, size=(len(cell_ids), len(genes))), index=cell_ids, columns=genes
    )

    # 1. scale_quantile
    expression_scaled = pd.DataFrame(
        nn.scale_quantile(raw_expression_matrix.to_numpy()),
        index=raw_expression_matrix.index,
        columns=raw_expression_matrix.columns,
    )

    ligand_target_matrix = nn.NamedMatrix(
        data=sp.csr_matrix(rng.uniform(0, 1, size=(len(target_genes), len(ligand_genes)))),
        rownames=target_genes,
        colnames=ligand_genes,
    )

    # 2. predict_single_cell_ligand_activities
    single_cell_activities = nn.predict_single_cell_ligand_activities(
        cell_ids=cell_ids,
        expression_scaled=expression_scaled,
        ligand_target_matrix=ligand_target_matrix,
        potential_ligands=ligand_genes,
    )

    if single_cell_activities.empty:
        print("no valid per-cell gene-set split with this synthetic data; try a different seed")
        return

    # 3. normalize_single_cell_ligand_activities
    normalized = nn.normalize_single_cell_ligand_activities(single_cell_activities)

    # 4. single_ligand_activity_score_regression
    cell_scores_tbl = pd.DataFrame({"cell": cell_ids, "score": rng.uniform(0, 1, size=len(cell_ids))})
    regression = nn.single_ligand_activity_score_regression(normalized, cell_scores_tbl)

    print(regression)


if __name__ == "__main__":
    main()
