"""Tests for nichenetr.networks."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from nichenetr.datasets import NamedMatrix
from nichenetr.networks import get_ligand_signaling_path, infer_supporting_datasources


@pytest.fixture
def network_fixtures():
    """Build a minimal set of weighted networks and ligand-TF matrix
    that are mutually consistent for testing."""
    # TFs as rows, ligands as columns
    tfs = ["TF1", "TF2", "TF3"]
    ligands = ["L1", "L2"]
    targets = ["T1", "T2"]

    # ligand_tf_matrix: TFs x ligands
    ltm_data = np.array([
        [0.5, 0.0],
        [0.3, 0.7],
        [0.0, 0.4],
    ])
    ligand_tf_matrix = NamedMatrix(
        data=scipy.sparse.csr_matrix(ltm_data),
        rownames=tfs,
        colnames=ligands,
    )

    # Weighted networks
    lr_sig = pd.DataFrame({
        "from": ["L1", "L1", "TF1", "L2", "TF2", "TF3"],
        "to":   ["TF1", "TF2", "TF2", "TF2", "TF3", "TF1"],
        "weight": [0.8, 0.3, 0.5, 0.6, 0.4, 0.2],
    })
    gr = pd.DataFrame({
        "from": ["TF1", "TF2", "TF3", "TF2"],
        "to":   ["T1",  "T1",  "T2",  "T2"],
        "weight": [0.9, 0.4, 0.6, 0.3],
    })
    weighted_networks = {"lr_sig": lr_sig, "gr": gr}

    return {
        "ligand_tf_matrix": ligand_tf_matrix,
        "weighted_networks": weighted_networks,
        "ligands": ligands,
        "targets": targets,
        "tfs": tfs,
    }


class TestGetLigandSignalingPath:
    def test_returns_dict_with_sig_and_gr(self, network_fixtures):
        result = get_ligand_signaling_path(
            ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
            ligands_all=["L1"],
            targets_all=["T1"],
            top_n_regulators=2,
            weighted_networks=network_fixtures["weighted_networks"],
        )
        assert isinstance(result, dict)
        assert "sig" in result
        assert "gr" in result
        assert isinstance(result["sig"], pd.DataFrame)
        assert isinstance(result["gr"], pd.DataFrame)

    def test_sig_has_from_to_weight(self, network_fixtures):
        result = get_ligand_signaling_path(
            ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
            ligands_all=["L1"],
            targets_all=["T1"],
            top_n_regulators=2,
            weighted_networks=network_fixtures["weighted_networks"],
        )
        for key in ("sig", "gr"):
            df = result[key]
            if len(df) > 0:
                assert "from" in df.columns
                assert "to" in df.columns
                assert "weight" in df.columns

    def test_missing_weighted_networks_raises(self, network_fixtures):
        with pytest.raises(ValueError, match="weighted_networks must be provided"):
            get_ligand_signaling_path(
                ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
                ligands_all=["L1"],
                targets_all=["T1"],
                weighted_networks=None,
            )


class TestInferSupportingDatasources:
    def test_output_columns(self, network_fixtures):
        # First get the signaling path
        sig_path = get_ligand_signaling_path(
            ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
            ligands_all=["L1"],
            targets_all=["T1"],
            top_n_regulators=2,
            weighted_networks=network_fixtures["weighted_networks"],
        )

        lr_network = pd.DataFrame({
            "from": ["L1", "L1"],
            "to": ["TF1", "TF2"],
            "source": ["lr_src1", "lr_src2"],
        })
        sig_network = pd.DataFrame({
            "from": ["TF1", "TF2"],
            "to": ["TF2", "TF3"],
            "source": ["sig_src1", "sig_src2"],
        })
        gr_network = pd.DataFrame({
            "from": ["TF1", "TF2"],
            "to": ["T1", "T1"],
            "source": ["gr_src1", "gr_src2"],
        })

        result = infer_supporting_datasources(
            signaling_graph_list=sig_path,
            lr_network=lr_network,
            sig_network=sig_network,
            gr_network=gr_network,
        )

        assert isinstance(result, pd.DataFrame)
        assert "from" in result.columns
        assert "to" in result.columns
        assert "source" in result.columns
        assert "layer" in result.columns
