"""Smoke test for nichenetr._biobabel.

Exercises one real, dependency-light path through the public API using a
tiny synthetic NamedMatrix (no network access, no bundled dataset
download): predict_ligand_activities -> get_weighted_ligand_target_links.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def main() -> None:
    import nichenetr as nn

    target_genes = ["T1", "T2", "T3", "T4"]
    ligand_genes = ["L1", "L2"]
    rng = np.random.default_rng(0)

    ligand_target_matrix = nn.NamedMatrix(
        data=sp.csr_matrix(rng.uniform(0, 1, size=(len(target_genes), len(ligand_genes)))),
        rownames=target_genes,
        colnames=ligand_genes,
    )

    ligand_activities = nn.predict_ligand_activities(
        geneset=["T1", "T2"],
        background_expressed_genes=target_genes,
        ligand_target_matrix=ligand_target_matrix,
        potential_ligands=ligand_genes,
    )
    assert set(ligand_activities.columns) >= {"test_ligand", "auroc", "aupr", "aupr_corrected", "pearson"}

    top_ligand = ligand_activities.sort_values("aupr_corrected", ascending=False)["test_ligand"].iloc[0]
    links = nn.get_weighted_ligand_target_links(
        ligand_oi=top_ligand, geneset=["T1", "T2"], ligand_target_matrix=ligand_target_matrix, n=4
    )
    assert list(links.columns) == ["ligand", "target", "weight"]

    print("nichenetr smoke test passed:", top_ligand)


if __name__ == "__main__":
    main()
