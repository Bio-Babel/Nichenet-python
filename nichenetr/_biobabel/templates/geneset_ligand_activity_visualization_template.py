"""Runnable skeleton for the nichenetr.geneset_ligand_activity_visualization workflow.

Uses a tiny synthetic raw expression matrix, gene set, and ligand-target
matrix in place of load_hnscc_expression()/load_pemt_signature()/
load_ligand_target_matrix() (no network access, no bundled dataset
download).

Step mapping (nichenetr.geneset_ligand_activity_visualization workflow):
    1. convert_alias_to_symbols        -> canonicalize raw expression column names
    2. predict_ligand_activities       -> rank candidate ligands against geneset_oi
    3. get_weighted_ligand_target_links + prepare_ligand_target_visualization
       + make_heatmap_ggplot           -> ligand-target heatmap
    4. get_weighted_ligand_receptor_links + prepare_ligand_receptor_visualization
       + make_heatmap_ggplot           -> ligand-receptor heatmap
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

    # 1. convert_alias_to_symbols (organism="mouse" bundled alias table; unmapped
    #    names are kept as-is, so this synthetic set mostly passes through)
    raw_columns = target_genes + ligand_genes
    canonical_columns = nn.convert_alias_to_symbols(raw_columns, organism="mouse", verbose=False)

    background_expressed_genes = target_genes
    potential_ligands = ligand_genes
    geneset_oi = target_genes[:2]

    ligand_target_matrix = nn.NamedMatrix(
        data=sp.csr_matrix(rng.uniform(0, 1, size=(len(target_genes), len(ligand_genes)))),
        rownames=target_genes,
        colnames=ligand_genes,
    )

    # 2. predict_ligand_activities
    ligand_activities = nn.predict_ligand_activities(
        geneset=geneset_oi,
        background_expressed_genes=background_expressed_genes,
        ligand_target_matrix=ligand_target_matrix,
        potential_ligands=potential_ligands,
    )
    top_ligands = ligand_activities.sort_values("aupr_corrected", ascending=False)["test_ligand"].tolist()

    # 3. ligand-target heatmap
    ligand_target_df = pd.concat(
        [
            nn.get_weighted_ligand_target_links(
                ligand_oi=lig, geneset=geneset_oi, ligand_target_matrix=ligand_target_matrix, n=4
            )
            for lig in top_ligands
        ]
    )
    ligand_target_vis = nn.prepare_ligand_target_visualization(
        ligand_target_df=ligand_target_df, ligand_target_matrix=ligand_target_matrix, cutoff=0.0
    )
    nn.make_heatmap_ggplot(ligand_target_vis, y_name="Targets", x_name="Ligands", show=False)

    # 4. ligand-receptor heatmap
    lr_network = pd.DataFrame({"from": ligand_genes, "to": ["R1", "R2"]})
    weighted_networks_lr_sig = pd.DataFrame({"from": ligand_genes, "to": ["R1", "R2"], "weight": [0.6, 0.4]})
    lr_top_df = nn.get_weighted_ligand_receptor_links(
        best_upstream_ligands=top_ligands,
        expressed_receptors=["R1", "R2"],
        lr_network=lr_network,
        weighted_networks_lr_sig=weighted_networks_lr_sig,
    )
    lr_vis = nn.prepare_ligand_receptor_visualization(lr_network_top_df=lr_top_df, best_upstream_ligands=top_ligands)
    nn.make_heatmap_ggplot(lr_vis, y_name="Receptors", x_name="Ligands", show=False)

    print(f"canonicalized columns: {canonical_columns}")
    print(ligand_activities)


if __name__ == "__main__":
    main()
