"""Tests for nichenetr.prioritization."""

import numpy as np
import pandas as pd
import pytest
import anndata

from nichenetr.prioritization import (
    calculate_de,
    get_exprs_avg,
    process_table_to_ic,
    generate_info_tables,
    generate_prioritization_tables,
    _rank_scale,
)


# ---------------------------------------------------------------------------
# _rank_scale
# ---------------------------------------------------------------------------

class TestRankScale:
    def test_basic(self):
        s = pd.Series([10, 20, 30])
        result = _rank_scale(s)
        assert result.max() == 1.0
        assert result.min() > 0

    def test_ties(self):
        s = pd.Series([1, 1, 3])
        result = _rank_scale(s)
        assert result.iloc[0] == result.iloc[1]  # tied values get same rank


# ---------------------------------------------------------------------------
# calculate_de
# ---------------------------------------------------------------------------

class TestCalculateDe:
    def test_basic(self, adata_with_conditions):
        result = calculate_de(
            adata_with_conditions,
            celltype_col="celltype",
            condition_oi="treated",
            condition_col="condition",
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"gene", "p_val", "avg_log2FC", "pct.1", "pct.2", "p_val_adj", "cluster_id"}
        assert expected_cols.issubset(set(result.columns))
        assert len(result) > 0

    def test_without_condition(self, small_adata):
        result = calculate_de(small_adata, celltype_col="celltype")
        assert isinstance(result, pd.DataFrame)
        assert "cluster_id" in result.columns

    def test_mismatched_condition_args(self, adata_with_conditions):
        with pytest.raises(ValueError, match="both condition_col and condition_oi"):
            calculate_de(
                adata_with_conditions,
                celltype_col="celltype",
                condition_oi="treated",
            )

    def test_with_features(self, adata_with_conditions):
        result = calculate_de(
            adata_with_conditions,
            celltype_col="celltype",
            condition_oi="treated",
            condition_col="condition",
            features=["L1", "L2", "L3"],
        )
        assert set(result["gene"]).issubset({"L1", "L2", "L3"})

    def test_no_valid_features_raises(self, adata_with_conditions):
        with pytest.raises(ValueError, match="None of the requested features"):
            calculate_de(
                adata_with_conditions,
                celltype_col="celltype",
                features=["NONEXISTENT1", "NONEXISTENT2"],
            )

    def test_with_min_pct(self, adata_with_conditions):
        result = calculate_de(
            adata_with_conditions,
            celltype_col="celltype",
            min_pct=0.5,
        )
        assert isinstance(result, pd.DataFrame)

    def test_with_logfc_threshold(self, adata_with_conditions):
        result = calculate_de(
            adata_with_conditions,
            celltype_col="celltype",
            logfc_threshold=0.5,
        )
        if len(result) > 0:
            assert (result["avg_log2FC"].abs() >= 0.5).all()

    def test_with_layer(self, adata_with_conditions):
        adata_with_conditions.layers["test"] = adata_with_conditions.X.copy()
        result = calculate_de(
            adata_with_conditions,
            celltype_col="celltype",
            assay_oi="test",
        )
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# get_exprs_avg
# ---------------------------------------------------------------------------

class TestGetExprsAvg:
    def test_basic(self, adata_with_conditions):
        result = get_exprs_avg(
            adata_with_conditions,
            celltype_col="celltype",
        )
        assert isinstance(result, pd.DataFrame)
        assert {"gene", "cluster_id", "avg_expr"}.issubset(set(result.columns))
        assert len(result) > 0

    def test_with_condition(self, adata_with_conditions):
        result = get_exprs_avg(
            adata_with_conditions,
            celltype_col="celltype",
            condition_oi="treated",
            condition_col="condition",
        )
        assert len(result) > 0

    def test_mismatched_args(self, adata_with_conditions):
        with pytest.raises(ValueError, match="both condition_col and condition_oi"):
            get_exprs_avg(
                adata_with_conditions,
                celltype_col="celltype",
                condition_oi="treated",
            )

    def test_with_features(self, adata_with_conditions):
        result = get_exprs_avg(
            adata_with_conditions,
            celltype_col="celltype",
            features=["L1", "L2"],
        )
        assert set(result["gene"]).issubset({"L1", "L2"})

    def test_no_valid_features_returns_empty(self, adata_with_conditions):
        result = get_exprs_avg(
            adata_with_conditions,
            celltype_col="celltype",
            features=["NONEXISTENT"],
        )
        assert len(result) == 0

    def test_with_layer(self, adata_with_conditions):
        adata_with_conditions.layers["test"] = adata_with_conditions.X.copy()
        result = get_exprs_avg(
            adata_with_conditions,
            celltype_col="celltype",
            assay_oi="test",
        )
        assert len(result) > 0


# ---------------------------------------------------------------------------
# process_table_to_ic
# ---------------------------------------------------------------------------

class TestProcessTableToIc:
    @pytest.fixture
    def lr_net(self):
        return pd.DataFrame({
            "from": ["L1", "L2", "L3"],
            "to": ["R1", "R2", "R3"],
        })

    @pytest.fixture
    def expression_table(self):
        return pd.DataFrame({
            "gene": ["L1", "L2", "R1", "R2", "L1", "L2", "R1", "R2"],
            "cluster_id": ["TypeA", "TypeA", "TypeA", "TypeA",
                           "TypeB", "TypeB", "TypeB", "TypeB"],
            "avg_expr": [1.0, 0.5, 0.8, 0.3, 0.2, 0.9, 0.4, 0.7],
        })

    @pytest.fixture
    def de_table(self):
        return pd.DataFrame({
            "gene": ["L1", "L2", "R1", "R2", "L1", "L2", "R1", "R2"],
            "cluster_id": ["TypeA", "TypeA", "TypeA", "TypeA",
                           "TypeB", "TypeB", "TypeB", "TypeB"],
            "avg_log2FC": [1.5, 0.3, 0.8, -0.2, -0.1, 0.9, 0.4, 0.7],
            "p_val": [0.01, 0.05, 0.001, 0.5, 0.3, 0.02, 0.1, 0.001],
            "p_val_adj": [0.05, 0.1, 0.005, 0.8, 0.5, 0.05, 0.2, 0.005],
            "pct.1": [0.8, 0.5, 0.9, 0.3, 0.2, 0.7, 0.4, 0.8],
        })

    def test_expression_type(self, expression_table, lr_net):
        result = process_table_to_ic(
            expression_table, table_type="expression", lr_network=lr_net,
        )
        assert isinstance(result, pd.DataFrame)
        assert "ligand" in result.columns
        assert "receptor" in result.columns
        assert "ligand_receptor_prod" in result.columns

    def test_celltype_de_type(self, de_table, lr_net):
        result = process_table_to_ic(
            de_table, table_type="celltype_DE", lr_network=lr_net,
            senders_oi=["TypeA"], receivers_oi=["TypeB"],
        )
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "lfc_ligand" in result.columns
            assert "lfc_receptor" in result.columns

    def test_group_de_type(self, lr_net):
        de = pd.DataFrame({
            "gene": ["L1", "L2", "R1", "R2"],
            "avg_log2FC": [1.0, 0.5, 0.8, 0.3],
            "p_val": [0.01, 0.05, 0.001, 0.1],
            "p_val_adj": [0.05, 0.1, 0.005, 0.2],
        })
        result = process_table_to_ic(
            de, table_type="group_DE", lr_network=lr_net,
        )
        assert isinstance(result, pd.DataFrame)

    def test_no_lr_network_raises(self, expression_table):
        with pytest.raises(ValueError, match="lr_network must be provided"):
            process_table_to_ic(expression_table, lr_network=None)

    def test_invalid_type_raises(self, expression_table, lr_net):
        with pytest.raises(ValueError, match="table_type must be"):
            process_table_to_ic(expression_table, table_type="invalid", lr_network=lr_net)

    def test_group_de_with_senders_raises(self, lr_net):
        de = pd.DataFrame({
            "gene": ["L1"], "avg_log2FC": [1.0],
            "p_val": [0.01], "p_val_adj": [0.05],
        })
        with pytest.raises(ValueError, match="senders_oi is given"):
            process_table_to_ic(
                de, table_type="group_DE", lr_network=lr_net, senders_oi=["TypeA"],
            )


# ---------------------------------------------------------------------------
# generate_info_tables
# ---------------------------------------------------------------------------

class TestGenerateInfoTables:
    def test_basic(self, adata_with_conditions, lr_network_named):
        result = generate_info_tables(
            adata_with_conditions,
            celltype_col="celltype",
            senders_oi=["TypeA"],
            receivers_oi=["TypeB"],
            lr_network=lr_network_named,
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            scenario="case_control",
        )
        assert isinstance(result, dict)
        assert "sender_receiver_de" in result
        assert "sender_receiver_info" in result
        assert "lr_condition_de" in result

    def test_one_condition_scenario(self, adata_with_conditions, lr_network_named):
        result = generate_info_tables(
            adata_with_conditions,
            celltype_col="celltype",
            senders_oi=["TypeA"],
            receivers_oi=["TypeB"],
            lr_network=lr_network_named,
            scenario="one_condition",
        )
        assert result["lr_condition_de"] is None

    def test_missing_celltype_col_raises(self, adata_with_conditions, lr_network_named):
        with pytest.raises(KeyError, match="not found in adata.obs"):
            generate_info_tables(
                adata_with_conditions,
                celltype_col="nonexistent",
                senders_oi=["TypeA"],
                receivers_oi=["TypeB"],
                lr_network=lr_network_named,
                scenario="one_condition",
            )

    def test_mismatched_condition_raises(self, adata_with_conditions, lr_network_named):
        with pytest.raises(ValueError, match="all None or all provided"):
            generate_info_tables(
                adata_with_conditions,
                celltype_col="celltype",
                senders_oi=["TypeA"],
                receivers_oi=["TypeB"],
                lr_network=lr_network_named,
                condition_col="condition",
                scenario="case_control",
            )

    def test_case_control_without_conditions_raises(self, adata_with_conditions, lr_network_named):
        with pytest.raises(ValueError, match="condition_\\* arguments are not provided"):
            generate_info_tables(
                adata_with_conditions,
                celltype_col="celltype",
                senders_oi=["TypeA"],
                receivers_oi=["TypeB"],
                lr_network=lr_network_named,
                scenario="case_control",
            )

    def test_invalid_scenario_raises(self, adata_with_conditions, lr_network_named):
        with pytest.raises(ValueError, match="scenario must be"):
            generate_info_tables(
                adata_with_conditions,
                celltype_col="celltype",
                senders_oi=["TypeA"],
                receivers_oi=["TypeB"],
                lr_network=lr_network_named,
                scenario="invalid",
            )

    def test_missing_sender_raises(self, adata_with_conditions, lr_network_named):
        with pytest.raises(ValueError, match="Senders not in data"):
            generate_info_tables(
                adata_with_conditions,
                celltype_col="celltype",
                senders_oi=["NONEXISTENT"],
                receivers_oi=["TypeB"],
                lr_network=lr_network_named,
                scenario="one_condition",
            )


# ---------------------------------------------------------------------------
# generate_prioritization_tables
# ---------------------------------------------------------------------------

class TestGeneratePrioritizationTables:
    @pytest.fixture
    def prio_inputs(self):
        sender_receiver_info = pd.DataFrame({
            "sender": ["TypeA", "TypeA"],
            "receiver": ["TypeB", "TypeB"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "avg_ligand": [1.0, 0.5],
            "avg_receptor": [0.8, 0.6],
            "ligand_receptor_prod": [0.8, 0.3],
        })
        sender_receiver_de = pd.DataFrame({
            "sender": ["TypeA", "TypeA"],
            "receiver": ["TypeB", "TypeB"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "lfc_ligand": [1.5, 0.3],
            "lfc_receptor": [0.8, 0.2],
            "p_val_ligand": [0.001, 0.05],
            "p_val_receptor": [0.01, 0.1],
            "p_adj_ligand": [0.005, 0.1],
            "p_adj_receptor": [0.05, 0.2],
            "pct_expressed_sender": [0.8, 0.5],
            "pct_expressed_receiver": [0.9, 0.3],
            "avg_ligand": [1.0, 0.5],
            "avg_receptor": [0.8, 0.6],
            "ligand_receptor_prod": [0.8, 0.3],
            "ligand_receptor_lfc_avg": [1.15, 0.25],
        })
        ligand_activities = pd.DataFrame({
            "test_ligand": ["L1", "L2"],
            "aupr_corrected": [0.15, 0.05],
        })
        lr_condition_de = pd.DataFrame({
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "lfc_ligand": [2.0, 0.5],
            "lfc_receptor": [1.0, 0.3],
            "p_val_ligand": [0.001, 0.05],
            "p_val_receptor": [0.01, 0.1],
        })
        return sender_receiver_info, sender_receiver_de, ligand_activities, lr_condition_de

    def test_case_control(self, prio_inputs):
        info, de, la, cond = prio_inputs
        result = generate_prioritization_tables(
            sender_receiver_info=info,
            sender_receiver_de=de,
            ligand_activities=la,
            lr_condition_de=cond,
            scenario="case_control",
        )
        assert isinstance(result, pd.DataFrame)
        assert "prioritization_score" in result.columns
        assert "prioritization_rank" in result.columns
        assert len(result) > 0

    def test_one_condition(self, prio_inputs):
        info, de, la, _ = prio_inputs
        result = generate_prioritization_tables(
            sender_receiver_info=info,
            sender_receiver_de=de,
            ligand_activities=la,
            lr_condition_de=None,
            scenario="one_condition",
        )
        assert isinstance(result, pd.DataFrame)
        assert "prioritization_score" in result.columns

    def test_custom_weights(self, prio_inputs):
        info, de, la, _ = prio_inputs
        weights = {
            "de_ligand": 2.0,
            "de_receptor": 1.0,
            "activity_scaled": 3.0,
            "exprs_ligand": 1.0,
            "exprs_receptor": 1.0,
            "ligand_condition_specificity": 0.0,
            "receptor_condition_specificity": 0.0,
        }
        result = generate_prioritization_tables(
            sender_receiver_info=info,
            sender_receiver_de=de,
            ligand_activities=la,
            prioritizing_weights=weights,
        )
        assert len(result) > 0

    def test_missing_weights_raises(self, prio_inputs):
        info, de, la, _ = prio_inputs
        with pytest.raises(ValueError, match="missing keys"):
            generate_prioritization_tables(
                sender_receiver_info=info,
                sender_receiver_de=de,
                ligand_activities=la,
                prioritizing_weights={"de_ligand": 1.0},
            )

    def test_invalid_scenario_raises(self, prio_inputs):
        info, de, la, _ = prio_inputs
        with pytest.raises(ValueError, match="scenario must be"):
            generate_prioritization_tables(
                sender_receiver_info=info,
                sender_receiver_de=de,
                ligand_activities=la,
                scenario="invalid",
            )

    def test_case_control_without_cond_de_raises(self, prio_inputs):
        info, de, la, _ = prio_inputs
        with pytest.raises(ValueError, match="lr_condition_de is None"):
            generate_prioritization_tables(
                sender_receiver_info=info,
                sender_receiver_de=de,
                ligand_activities=la,
                lr_condition_de=None,
                scenario="case_control",
            )

    def test_nonzero_cond_weight_without_data_raises(self, prio_inputs):
        info, de, la, _ = prio_inputs
        weights = {
            "de_ligand": 1.0, "de_receptor": 1.0, "activity_scaled": 1.0,
            "exprs_ligand": 1.0, "exprs_receptor": 1.0,
            "ligand_condition_specificity": 1.0,
            "receptor_condition_specificity": 0.0,
        }
        with pytest.raises(ValueError, match="No lr_condition_de"):
            generate_prioritization_tables(
                sender_receiver_info=info,
                sender_receiver_de=de,
                ligand_activities=la,
                lr_condition_de=None,
                prioritizing_weights=weights,
            )

    def test_scores_in_range(self, prio_inputs):
        info, de, la, cond = prio_inputs
        result = generate_prioritization_tables(
            sender_receiver_info=info,
            sender_receiver_de=de,
            ligand_activities=la,
            lr_condition_de=cond,
        )
        assert (result["prioritization_score"] >= 0).all()
        assert (result["prioritization_score"] <= 1).all()

    def test_one_condition_with_cond_de_warns(self, prio_inputs):
        info, de, la, cond = prio_inputs
        with pytest.warns(UserWarning, match="will not be used"):
            result = generate_prioritization_tables(
                sender_receiver_info=info,
                sender_receiver_de=de,
                ligand_activities=la,
                lr_condition_de=cond,
                scenario="one_condition",
            )
        assert isinstance(result, pd.DataFrame)


class TestProcessTableToIcEdgeCases:
    def test_expression_with_senders_warns(self):
        lr = pd.DataFrame({"from": ["L1"], "to": ["R1"]})
        expr = pd.DataFrame({
            "gene": ["L1", "R1", "L1", "R1"],
            "cluster_id": ["A", "A", "B", "B"],
            "avg_expr": [1.0, 0.5, 0.3, 0.8],
        })
        with pytest.warns(UserWarning, match="senders_oi is given"):
            process_table_to_ic(expr, table_type="expression", lr_network=lr, senders_oi=["A"])

    def test_expression_with_receivers_warns(self):
        lr = pd.DataFrame({"from": ["L1"], "to": ["R1"]})
        expr = pd.DataFrame({
            "gene": ["L1", "R1", "L1", "R1"],
            "cluster_id": ["A", "A", "B", "B"],
            "avg_expr": [1.0, 0.5, 0.3, 0.8],
        })
        with pytest.warns(UserWarning, match="receivers_oi is given"):
            process_table_to_ic(expr, table_type="expression", lr_network=lr, receivers_oi=["B"])

    def test_celltype_de_without_senders_warns(self):
        lr = pd.DataFrame({"from": ["L1"], "to": ["R1"]})
        de = pd.DataFrame({
            "gene": ["L1", "R1"],
            "cluster_id": ["A", "A"],
            "avg_log2FC": [1.0, 0.5],
            "p_val": [0.01, 0.05],
            "p_val_adj": [0.05, 0.1],
            "pct.1": [0.8, 0.5],
        })
        with pytest.warns(UserWarning, match="senders_oi is None"):
            process_table_to_ic(de, table_type="celltype_DE", lr_network=lr)

    def test_group_de_with_receivers_raises(self):
        lr = pd.DataFrame({"from": ["L1"], "to": ["R1"]})
        de = pd.DataFrame({
            "gene": ["L1"], "avg_log2FC": [1.0],
            "p_val": [0.01], "p_val_adj": [0.05],
        })
        with pytest.raises(ValueError, match="receivers_oi is given"):
            process_table_to_ic(de, table_type="group_DE", lr_network=lr, receivers_oi=["B"])


class TestGenerateInfoTablesEdgeCases:
    def test_one_condition_with_conditions_warns(self, adata_with_conditions, lr_network_named):
        with pytest.warns(UserWarning, match="condition_\\* arguments are provided"):
            result = generate_info_tables(
                adata_with_conditions,
                celltype_col="celltype",
                senders_oi=["TypeA"],
                receivers_oi=["TypeB"],
                lr_network=lr_network_named,
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
                scenario="one_condition",
            )
        assert isinstance(result, dict)
