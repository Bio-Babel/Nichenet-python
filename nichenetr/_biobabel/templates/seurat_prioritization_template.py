"""Runnable skeleton for the nichenetr.seurat_prioritization workflow.

Uses a tiny synthetic AnnData, a synthetic lr_network, and a mocked
ligand_activities table (standing in for the real output of
nichenetr.predict_ligand_activities from a prior ligand-activity run) so it
runs instantly with no network access.

Step mapping (nichenetr.seurat_prioritization workflow):
    1. generate_info_tables         -> per-cell-type DE, expression, condition DE
    2. generate_prioritization_tables -> combine into one ranked table
    3. make_mushroom_plot           -> visualize the top pairs
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

import nichenetr as nn


def main() -> None:
    rng = np.random.default_rng(0)

    genes = ["L1", "R1"]
    celltypes = ["Sender", "Receiver"]
    conditions = ["High", "Low"]
    n_per_group = 5

    rows = []
    for ct in celltypes:
        for cond in conditions:
            for _ in range(n_per_group):
                rows.append((ct, cond))
    obs = pd.DataFrame(rows, columns=["celltype", "tumor"])
    X = rng.uniform(1, 5, size=(len(obs), len(genes)))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))

    lr_network = pd.DataFrame({"from": ["L1"], "to": ["R1"]})
    # Stand-in for nichenetr.predict_ligand_activities output from a prior
    # ligand-activity run (see the nichenetr.seurat_ligand_activity workflow).
    ligand_activities = pd.DataFrame({"test_ligand": ["L1"], "aupr_corrected": [0.6]})

    # 1. generate_info_tables
    info_tables = nn.generate_info_tables(
        adata,
        celltype_col="celltype",
        senders_oi=["Sender"],
        receivers_oi=["Receiver"],
        lr_network=lr_network,
        condition_col="tumor",
        condition_oi="High",
        condition_ref="Low",
        scenario="case_control",
    )

    # 2. generate_prioritization_tables
    prioritization_table = nn.generate_prioritization_tables(
        sender_receiver_info=info_tables["sender_receiver_info"],
        sender_receiver_de=info_tables["sender_receiver_de"],
        ligand_activities=ligand_activities,
        lr_condition_de=info_tables["lr_condition_de"],
        scenario="case_control",
    )

    # 3. visualize
    if not prioritization_table.empty:
        nn.make_mushroom_plot(prioritization_table, top_n=min(10, len(prioritization_table)), show=False)

    print(f"prioritized {len(prioritization_table)} ligand-receptor pairs")


if __name__ == "__main__":
    main()
