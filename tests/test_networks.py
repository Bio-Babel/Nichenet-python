"""Tests for nichenetr.networks."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from nichenetr.datasets import NamedMatrix
from nichenetr.networks import (
    get_ligand_signaling_path,
    format_signaling_graph,
    infer_supporting_datasources,
)


@pytest.fixture
def network_fixtures():
    """Build a minimal set of weighted networks and ligand-TF matrix."""
    tfs = ["TF1", "TF2", "TF3"]
    ligands = ["L1", "L2"]
    targets = ["T1", "T2"]

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

    def test_invalid_weighted_networks_type(self, network_fixtures):
        with pytest.raises(ValueError, match="must be a dict"):
            get_ligand_signaling_path(
                ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
                ligands_all=["L1"],
                targets_all=["T1"],
                weighted_networks="not_a_dict",
            )

    def test_missing_key_in_weighted_networks(self, network_fixtures):
        with pytest.raises(ValueError, match="must be a DataFrame"):
            get_ligand_signaling_path(
                ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
                ligands_all=["L1"],
                targets_all=["T1"],
                weighted_networks={"lr_sig": "not_df", "gr": pd.DataFrame({"from": [], "to": [], "weight": []})},
            )

    def test_missing_weight_column(self, network_fixtures):
        wn = {
            "lr_sig": pd.DataFrame({"from": ["A"], "to": ["B"]}),
            "gr": pd.DataFrame({"from": ["C"], "to": ["D"], "weight": [1]}),
        }
        with pytest.raises(ValueError, match="weight"):
            get_ligand_signaling_path(
                ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
                ligands_all=["L1"],
                targets_all=["T1"],
                weighted_networks=wn,
            )

    def test_invalid_ligands_position(self, network_fixtures):
        with pytest.raises(ValueError, match="ligands_position"):
            get_ligand_signaling_path(
                ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
                ligands_all=["L1"],
                targets_all=["T1"],
                weighted_networks=network_fixtures["weighted_networks"],
                ligands_position="invalid",
            )

    def test_missing_ligand_raises(self, network_fixtures):
        with pytest.raises(ValueError, match="Ligands not found"):
            get_ligand_signaling_path(
                ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
                ligands_all=["MISSING"],
                targets_all=["T1"],
                weighted_networks=network_fixtures["weighted_networks"],
            )

    def test_missing_target_raises(self, network_fixtures):
        with pytest.raises(ValueError, match="Target genes not in"):
            get_ligand_signaling_path(
                ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
                ligands_all=["L1"],
                targets_all=["MISSING"],
                weighted_networks=network_fixtures["weighted_networks"],
            )

    def test_invalid_top_n(self, network_fixtures):
        with pytest.raises(ValueError, match="top_n_regulators"):
            get_ligand_signaling_path(
                ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
                ligands_all=["L1"],
                targets_all=["T1"],
                weighted_networks=network_fixtures["weighted_networks"],
                top_n_regulators=-1,
            )

    def test_ligands_in_rows(self, network_fixtures):
        """Test ligands_position='rows' (transposed matrix)."""
        # Transpose the matrix: ligands as rows, TFs as cols
        ltm = network_fixtures["ligand_tf_matrix"]
        transposed = NamedMatrix(
            data=ltm.data.T.tocsr(),
            rownames=ltm.colnames,  # ligands
            colnames=ltm.rownames,  # TFs
        )
        result = get_ligand_signaling_path(
            ligand_tf_matrix=transposed,
            ligands_all=["L1"],
            targets_all=["T1"],
            top_n_regulators=2,
            weighted_networks=network_fixtures["weighted_networks"],
            ligands_position="rows",
        )
        assert "sig" in result
        assert "gr" in result

    def test_minmax_scaling(self, network_fixtures):
        result = get_ligand_signaling_path(
            ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
            ligands_all=["L1"],
            targets_all=["T1"],
            top_n_regulators=2,
            weighted_networks=network_fixtures["weighted_networks"],
            minmax_scaling=True,
        )
        for key in ("sig", "gr"):
            df = result[key]
            if len(df) > 0:
                assert df["weight"].min() >= 0.74  # 0.75 - epsilon
                assert df["weight"].max() <= 1.76  # 1.75 + epsilon

    def test_multiple_ligands_and_targets(self, network_fixtures):
        result = get_ligand_signaling_path(
            ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
            ligands_all=["L1", "L2"],
            targets_all=["T1", "T2"],
            top_n_regulators=2,
            weighted_networks=network_fixtures["weighted_networks"],
        )
        assert len(result["sig"]) > 0 or len(result["gr"]) > 0


class TestFormatSignalingGraph:
    def _get_sig_path(self, network_fixtures):
        return get_ligand_signaling_path(
            ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
            ligands_all=["L1"],
            targets_all=["T1"],
            top_n_regulators=2,
            weighted_networks=network_fixtures["weighted_networks"],
        )

    def test_returns_nodes_and_edges(self, network_fixtures):
        sig_path = self._get_sig_path(network_fixtures)
        result = format_signaling_graph(sig_path, ["L1"], ["T1"])
        assert "nodes" in result
        assert "edges" in result
        assert isinstance(result["nodes"], pd.DataFrame)
        assert isinstance(result["edges"], pd.DataFrame)

    def test_node_types(self, network_fixtures):
        sig_path = self._get_sig_path(network_fixtures)
        result = format_signaling_graph(sig_path, ["L1"], ["T1"])
        types = set(result["nodes"]["type"])
        assert "ligand" in types
        assert "target" in types

    def test_edge_colors(self, network_fixtures):
        sig_path = self._get_sig_path(network_fixtures)
        result = format_signaling_graph(
            sig_path, ["L1"], ["T1"],
            sig_color="blue", gr_color="red",
        )
        assert "color" in result["edges"].columns

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            format_signaling_graph("not_dict", ["L1"], ["T1"])

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="must be a DataFrame"):
            format_signaling_graph({"sig": pd.DataFrame()}, ["L1"], ["T1"])

    def test_missing_ligand_in_sig_raises(self, network_fixtures):
        sig_path = self._get_sig_path(network_fixtures)
        with pytest.raises(ValueError, match="Ligands not found"):
            format_signaling_graph(sig_path, ["MISSING"], ["T1"])

    def test_missing_target_in_gr_raises(self, network_fixtures):
        sig_path = self._get_sig_path(network_fixtures)
        with pytest.raises(ValueError, match="Targets not found"):
            format_signaling_graph(sig_path, ["L1"], ["MISSING"])

    def test_invalid_color_type(self, network_fixtures):
        sig_path = self._get_sig_path(network_fixtures)
        with pytest.raises(ValueError, match="sig_color must be"):
            format_signaling_graph(sig_path, ["L1"], ["T1"], sig_color=123)


class TestInferSupportingDatasources:
    def test_output_columns(self, network_fixtures):
        sig_path = get_ligand_signaling_path(
            ligand_tf_matrix=network_fixtures["ligand_tf_matrix"],
            ligands_all=["L1"],
            targets_all=["T1"],
            top_n_regulators=2,
            weighted_networks=network_fixtures["weighted_networks"],
        )

        lr_network = pd.DataFrame({
            "from": ["L1", "L1"], "to": ["TF1", "TF2"], "source": ["lr1", "lr2"],
        })
        sig_network = pd.DataFrame({
            "from": ["TF1", "TF2"], "to": ["TF2", "TF3"], "source": ["sig1", "sig2"],
        })
        gr_network = pd.DataFrame({
            "from": ["TF1", "TF2"], "to": ["T1", "T1"], "source": ["gr1", "gr2"],
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

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="must be a DataFrame"):
            infer_supporting_datasources(
                signaling_graph_list={"sig": pd.DataFrame(), "gr": pd.DataFrame()},
                lr_network="not_df",
                sig_network=pd.DataFrame(),
                gr_network=pd.DataFrame(),
            )

    def test_invalid_graph_list(self):
        with pytest.raises(ValueError, match="must be a dict"):
            infer_supporting_datasources(
                signaling_graph_list="not_dict",
                lr_network=pd.DataFrame(),
                sig_network=pd.DataFrame(),
                gr_network=pd.DataFrame(),
            )
