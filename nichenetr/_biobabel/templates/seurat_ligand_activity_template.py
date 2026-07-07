"""Runnable skeleton for the nichenetr.seurat_ligand_activity workflow.

Uses a tiny synthetic AnnData and synthetic prior networks (no network
access, no bundled dataset download) so it can run instantly in CI. Replace
the synthetic adata/ligand_target_matrix/lr_network/weighted_networks with
real objects (e.g. from nichenetr.load_seurat_obj() and
nichenetr.load_ligand_target_matrix()) for a real analysis.

Step mapping (nichenetr.seurat_ligand_activity workflow):
    1. alias_to_symbol_anndata      -> canonicalize gene symbols
    2. get_expressed_genes (x2)     -> background_expressed_genes, potential_ligands
    3. predict_ligand_activities    -> rank candidate ligands
    4. get_weighted_ligand_target_links + prepare_ligand_target_visualization
       + make_heatmap_ggplot        -> ligand-target heatmap
    5. get_weighted_ligand_receptor_links + prepare_ligand_receptor_visualization
       + make_heatmap_ggplot        -> ligand-receptor heatmap
    6. get_lfc_celltype + make_threecolor_heatmap_ggplot -> sender LFC heatmap
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import nichenetr as nn


def main() -> None:
    rng = np.random.default_rng(0)

    target_genes = [f"T{i}" for i in range(1, 7)]
    ligand_genes = ["L1", "L2", "L3"]
    receptor_genes = ["R1", "R2"]
    genes = target_genes + ligand_genes + receptor_genes

    n_receiver, n_sender = 10, 10
    n_cells = n_receiver + n_sender
    X = np.zeros((n_cells, len(genes)))
    receiver_idx = slice(0, n_receiver)
    sender_idx = slice(n_receiver, n_cells)
    X[receiver_idx, : len(target_genes)] = rng.uniform(1, 5, size=(n_receiver, len(target_genes)))
    X[receiver_idx, len(target_genes) + len(ligand_genes):] = rng.uniform(1, 5, size=(n_receiver, len(receptor_genes)))
    X[sender_idx, len(target_genes):len(target_genes) + len(ligand_genes)] = rng.uniform(1, 5, size=(n_sender, len(ligand_genes)))

    celltype = ["Receiver"] * n_receiver + ["Sender"] * n_sender
    condition = ["A"] * n_receiver + ["A"] * 5 + ["B"] * 5
    obs = pd.DataFrame({"celltype": celltype, "condition": condition})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))

    # 1. alias_to_symbol_anndata
    adata = nn.alias_to_symbol_anndata(adata, organism="mouse")

    # 2. get_expressed_genes
    background_expressed_genes = nn.get_expressed_genes(adata, "celltype", "Receiver", pct=0.10)
    potential_ligands = nn.get_expressed_genes(adata, "celltype", "Sender", pct=0.10)

    ligand_target_matrix = nn.NamedMatrix(
        data=sp.csr_matrix(rng.uniform(0, 1, size=(len(target_genes), len(ligand_genes)))),
        rownames=target_genes,
        colnames=ligand_genes,
    )
    geneset_oi = ["T1", "T2"]

    # 3. predict_ligand_activities
    ligand_activities = nn.predict_ligand_activities(
        geneset=geneset_oi,
        background_expressed_genes=background_expressed_genes,
        ligand_target_matrix=ligand_target_matrix,
        potential_ligands=potential_ligands,
    )
    top_ligand = ligand_activities.sort_values("aupr_corrected", ascending=False)["test_ligand"].iloc[0]

    # 4. ligand-target heatmap
    ligand_target_df = nn.get_weighted_ligand_target_links(
        ligand_oi=top_ligand, geneset=geneset_oi, ligand_target_matrix=ligand_target_matrix, n=6
    )
    ligand_target_vis = nn.prepare_ligand_target_visualization(
        ligand_target_df=ligand_target_df, ligand_target_matrix=ligand_target_matrix, cutoff=0.0
    )
    nn.make_heatmap_ggplot(ligand_target_vis, y_name="Targets", x_name="Ligands", show=False)

    # 5. ligand-receptor heatmap
    lr_network = pd.DataFrame({"from": ["L1", "L2", "L3"], "to": ["R1", "R1", "R2"]})
    weighted_networks_lr_sig = pd.DataFrame(
        {"from": ["L1", "L2", "L3"], "to": ["R1", "R1", "R2"], "weight": [0.5, 0.3, 0.8]}
    )
    lr_top_df = nn.get_weighted_ligand_receptor_links(
        best_upstream_ligands=[top_ligand],
        expressed_receptors=receptor_genes,
        lr_network=lr_network,
        weighted_networks_lr_sig=weighted_networks_lr_sig,
    )
    lr_vis = nn.prepare_ligand_receptor_visualization(
        lr_network_top_df=lr_top_df, best_upstream_ligands=[top_ligand]
    )
    nn.make_heatmap_ggplot(lr_vis, y_name="Receptors", x_name="Ligands", show=False)

    # 6. sender log-fold-change heatmap
    sender_lfc = nn.get_lfc_celltype(
        adata,
        celltype_col="celltype",
        senders=["Sender"],
        condition_col="condition",
        condition_oi="B",
        condition_ref="A",
        ligands_oi=[top_ligand],
    )
    if not sender_lfc.empty:
        nn.make_threecolor_heatmap_ggplot(
            sender_lfc.set_index("gene").to_numpy(), y_name="Ligands", x_name="Senders", show=False
        )

    print(f"top ligand: {top_ligand}")


if __name__ == "__main__":
    main()
