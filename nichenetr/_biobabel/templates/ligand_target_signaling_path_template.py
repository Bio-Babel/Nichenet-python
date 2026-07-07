"""Runnable skeleton for the nichenetr.ligand_target_signaling_path workflow.

Uses tiny synthetic prior networks (no network access, no bundled dataset
download): a one-ligand, one-regulator, one-target signaling chain
L1 -> TF1 -> G1.

Step mapping (nichenetr.ligand_target_signaling_path workflow):
    1. get_ligand_signaling_path  -> trace the path from L1 to G1 through TF1
    2. format_signaling_graph     -> nodes/edges table for visualization
    3. infer_supporting_datasources -> which prior sources support each edge
"""

from __future__ import annotations

import pandas as pd
import scipy.sparse as sp

import nichenetr as nn


def main() -> None:
    ligands_all = ["L1"]
    targets_all = ["G1"]

    ligand_tf_matrix = nn.NamedMatrix(
        data=sp.csr_matrix([[0.8]]), rownames=["TF1"], colnames=["L1"]
    )
    weighted_networks = {
        "lr_sig": pd.DataFrame({"from": ["L1"], "to": ["TF1"], "weight": [0.5]}),
        "gr": pd.DataFrame({"from": ["TF1"], "to": ["G1"], "weight": [0.7]}),
    }

    # 1. get_ligand_signaling_path
    signaling_graph_list = nn.get_ligand_signaling_path(
        ligand_tf_matrix=ligand_tf_matrix,
        ligands_all=ligands_all,
        targets_all=targets_all,
        top_n_regulators=1,
        weighted_networks=weighted_networks,
    )

    # 2. format_signaling_graph
    formatted = nn.format_signaling_graph(
        signaling_graph_list=signaling_graph_list, ligands_all=ligands_all, targets_all=targets_all
    )

    # 3. infer_supporting_datasources
    lr_network = pd.DataFrame({"from": ["L1"], "to": ["TF1"], "source": ["test_lr"]})
    sig_network = pd.DataFrame({"from": ["L1"], "to": ["TF1"], "source": ["test_sig"]})
    gr_network = pd.DataFrame({"from": ["TF1"], "to": ["G1"], "source": ["test_gr"]})
    datasource_table = nn.infer_supporting_datasources(
        signaling_graph_list=signaling_graph_list,
        lr_network=lr_network,
        sig_network=sig_network,
        gr_network=gr_network,
    )

    print(formatted["nodes"])
    print(datasource_table)


if __name__ == "__main__":
    main()
