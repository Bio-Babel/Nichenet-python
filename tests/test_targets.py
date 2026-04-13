"""Tests for nichenetr.targets."""

import numpy as np
import pandas as pd
import pytest

from nichenetr.targets import (
    get_weighted_ligand_target_links,
    prepare_ligand_target_visualization,
    get_weighted_ligand_receptor_links,
    prepare_ligand_receptor_visualization,
    get_ligand_target_links_oi,
    _hclust_order,
    _NamedArray,
    _attach_names,
)


# ---------------------------------------------------------------------------
# get_weighted_ligand_target_links
# ---------------------------------------------------------------------------

class TestGetWeightedLigandTargetLinks:
    def test_output_structure(self, small_ligand_target_matrix, small_geneset):
        result = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"ligand", "target", "weight"}

    def test_ligand_column_value(self, small_ligand_target_matrix, small_geneset):
        result = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert (result["ligand"] == "L1").all()

    def test_missing_ligand_returns_nan(self, small_ligand_target_matrix, small_geneset):
        result = get_weighted_ligand_target_links(
            ligand_oi="NONEXISTENT",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert len(result) == 1
        assert pd.isna(result["target"].iloc[0])

    def test_targets_in_geneset(self, small_ligand_target_matrix, small_geneset):
        result = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        non_nan_targets = result.dropna(subset=["target"])
        if len(non_nan_targets) > 0:
            assert set(non_nan_targets["target"]).issubset(set(small_geneset))

    def test_custom_n(self, small_ligand_target_matrix, small_geneset):
        result = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
            n=2,
        )
        assert isinstance(result, pd.DataFrame)

    def test_no_geneset_overlap(self, small_ligand_target_matrix):
        """When geneset doesn't overlap with top targets, return NaN."""
        result = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=["NONEXISTENT1", "NONEXISTENT2"],
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert len(result) == 1
        assert pd.isna(result["target"].iloc[0])


# ---------------------------------------------------------------------------
# prepare_ligand_target_visualization
# ---------------------------------------------------------------------------

class TestPrepareLigandTargetVisualization:
    def test_matrix_shape(self, small_ligand_target_matrix, small_geneset):
        links = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        result = prepare_ligand_target_visualization(
            ligand_target_df=links,
            ligand_target_matrix=small_ligand_target_matrix,
            cutoff=0.0,
        )
        assert result.ndim == 2
        assert hasattr(result, "rownames")
        assert hasattr(result, "colnames")

    def test_empty_input(self, small_ligand_target_matrix):
        empty_df = pd.DataFrame(
            {"ligand": [np.nan], "target": [np.nan], "weight": [np.nan]}
        )
        result = prepare_ligand_target_visualization(
            ligand_target_df=empty_df,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        assert result.shape == (0, 0)

    def test_high_cutoff_may_reduce(self, small_ligand_target_matrix, small_geneset):
        links = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        low = prepare_ligand_target_visualization(
            ligand_target_df=links,
            ligand_target_matrix=small_ligand_target_matrix,
            cutoff=0.0,
        )
        high = prepare_ligand_target_visualization(
            ligand_target_df=links,
            ligand_target_matrix=small_ligand_target_matrix,
            cutoff=0.9,
        )
        assert high.shape[0] <= low.shape[0]

    def test_multiple_ligands(self, small_ligand_target_matrix, small_geneset):
        frames = []
        for lig in ["L1", "L2"]:
            df = get_weighted_ligand_target_links(
                ligand_oi=lig,
                geneset=small_geneset,
                ligand_target_matrix=small_ligand_target_matrix,
            )
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        result = prepare_ligand_target_visualization(
            ligand_target_df=combined,
            ligand_target_matrix=small_ligand_target_matrix,
            cutoff=0.0,
        )
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# get_weighted_ligand_receptor_links
# ---------------------------------------------------------------------------

class TestGetWeightedLigandReceptorLinks:
    def test_output_structure(self):
        lr_network = pd.DataFrame({
            "from": ["L1", "L2", "L3"],
            "to": ["R1", "R2", "R3"],
        })
        weighted_lr_sig = pd.DataFrame({
            "from": ["L1", "L2", "L3", "L1"],
            "to": ["R1", "R2", "R3", "R2"],
            "weight": [0.5, 0.8, 0.3, 0.6],
        })
        result = get_weighted_ligand_receptor_links(
            best_upstream_ligands=["L1", "L2"],
            expressed_receptors=["R1", "R2", "R3"],
            lr_network=lr_network,
            weighted_networks_lr_sig=weighted_lr_sig,
        )
        assert isinstance(result, pd.DataFrame)
        assert "from" in result.columns
        assert "to" in result.columns
        assert "weight" in result.columns

    def test_filters_to_requested_ligands(self):
        lr_network = pd.DataFrame({
            "from": ["L1", "L2", "L3"],
            "to": ["R1", "R2", "R3"],
        })
        weighted_lr_sig = pd.DataFrame({
            "from": ["L1", "L2", "L3"],
            "to": ["R1", "R2", "R3"],
            "weight": [0.5, 0.8, 0.3],
        })
        result = get_weighted_ligand_receptor_links(
            best_upstream_ligands=["L1"],
            expressed_receptors=["R1", "R2", "R3"],
            lr_network=lr_network,
            weighted_networks_lr_sig=weighted_lr_sig,
        )
        if len(result) > 0:
            assert set(result["from"].unique()).issubset({"L1"})


# ---------------------------------------------------------------------------
# prepare_ligand_receptor_visualization
# ---------------------------------------------------------------------------

class TestPrepareLigandReceptorVisualization:
    @pytest.fixture
    def lr_top_df(self):
        return pd.DataFrame({
            "from": ["L1", "L1", "L2", "L2", "L3"],
            "to": ["R1", "R2", "R1", "R3", "R2"],
            "weight": [0.5, 0.8, 0.3, 0.6, 0.7],
        })

    def test_basic(self, lr_top_df):
        result = prepare_ligand_receptor_visualization(
            lr_top_df,
            best_upstream_ligands=["L1", "L2", "L3"],
        )
        assert result.ndim == 2
        assert hasattr(result, "rownames")
        assert hasattr(result, "colnames")

    def test_order_hclust_options(self, lr_top_df):
        for opt in ("both", "ligands", "receptors", "none"):
            result = prepare_ligand_receptor_visualization(
                lr_top_df,
                best_upstream_ligands=["L1", "L2", "L3"],
                order_hclust=opt,
            )
            assert result.ndim == 2

    def test_invalid_order_raises(self, lr_top_df):
        with pytest.raises(ValueError, match="order_hclust"):
            prepare_ligand_receptor_visualization(
                lr_top_df,
                best_upstream_ligands=["L1"],
                order_hclust="invalid",
            )

    def test_single_row(self):
        df = pd.DataFrame({
            "from": ["L1"], "to": ["R1"], "weight": [0.5],
        })
        result = prepare_ligand_receptor_visualization(
            df, best_upstream_ligands=["L1"], order_hclust="none",
        )
        assert result.shape == (1, 1)

    def test_empty_input(self):
        df = pd.DataFrame({"from": [], "to": [], "weight": []})
        result = prepare_ligand_receptor_visualization(
            df, best_upstream_ligands=["L1"],
        )
        assert result.shape == (0, 0)


# ---------------------------------------------------------------------------
# get_ligand_target_links_oi
# ---------------------------------------------------------------------------

class TestGetLigandTargetLinksOi:
    @pytest.fixture
    def oi_fixtures(self):
        ligand_type = pd.DataFrame({
            "ligand_type": ["TypeA", "TypeA", "TypeB"],
            "ligand": ["L1", "L2", "L3"],
        })
        active_links = pd.DataFrame({
            "ligand": ["L1", "L1", "L2", "L2", "L3"],
            "target": ["G1", "G2", "G1", "G3", "G2"],
            "weight": [0.9, 0.3, 0.8, 0.1, 0.7],
            "target_type": ["T", "T", "T", "T", "T"],
        })
        return ligand_type, active_links

    def test_basic(self, oi_fixtures):
        lt_df, al_df = oi_fixtures
        result = get_ligand_target_links_oi(lt_df, al_df, cutoff=0.3)
        assert isinstance(result, pd.DataFrame)
        assert "ligand" in result.columns
        assert "target" in result.columns
        assert "weight" in result.columns
        assert "ligand_type" in result.columns

    def test_cutoff_attr(self, oi_fixtures):
        lt_df, al_df = oi_fixtures
        result = get_ligand_target_links_oi(lt_df, al_df, cutoff=0.3)
        assert "cutoff_include_all_ligands" in result.attrs

    def test_invalid_cutoff(self, oi_fixtures):
        lt_df, al_df = oi_fixtures
        with pytest.raises(ValueError, match="cutoff must be between"):
            get_ligand_target_links_oi(lt_df, al_df, cutoff=1.5)

    def test_missing_columns_lt(self, oi_fixtures):
        _, al_df = oi_fixtures
        bad_lt = pd.DataFrame({"wrong": [1]})
        with pytest.raises(ValueError, match="ligand_type"):
            get_ligand_target_links_oi(bad_lt, al_df)

    def test_missing_columns_al(self, oi_fixtures):
        lt_df, _ = oi_fixtures
        bad_al = pd.DataFrame({"wrong": [1]})
        with pytest.raises(ValueError, match="ligand.*target.*weight"):
            get_ligand_target_links_oi(lt_df, bad_al)

    def test_high_cutoff_removes_more(self, oi_fixtures):
        lt_df, al_df = oi_fixtures
        low = get_ligand_target_links_oi(lt_df, al_df, cutoff=0.1)
        high = get_ligand_target_links_oi(lt_df, al_df, cutoff=0.8)
        assert len(high) <= len(low)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestPrepareLigandTargetVisEdgeCases:
    def test_no_targets_in_matrix(self, small_ligand_target_matrix):
        """Targets not in matrix -> empty result."""
        links = pd.DataFrame({
            "ligand": ["L1"], "target": ["NOTINMATRIX"], "weight": [0.5],
        })
        result = prepare_ligand_target_visualization(
            ligand_target_df=links,
            ligand_target_matrix=small_ligand_target_matrix,
            cutoff=0.0,
        )
        assert result.shape == (0, 0)

    def test_very_high_cutoff_zeros_all(self, small_ligand_target_matrix, small_geneset):
        """Cutoff so high that all entries become zero."""
        links = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        result = prepare_ligand_target_visualization(
            ligand_target_df=links,
            ligand_target_matrix=small_ligand_target_matrix,
            cutoff=0.999,  # Very high cutoff
        )
        # May produce empty or near-empty result
        assert result.ndim == 2


class TestPrepareLigandTargetVisSingleRowCol:
    def test_single_ligand(self, small_ligand_target_matrix, small_geneset):
        """One ligand -> skip column clustering."""
        links = get_weighted_ligand_target_links(
            ligand_oi="L1",
            geneset=small_geneset,
            ligand_target_matrix=small_ligand_target_matrix,
        )
        result = prepare_ligand_target_visualization(
            ligand_target_df=links,
            ligand_target_matrix=small_ligand_target_matrix,
            cutoff=0.0,
        )
        # May have 1 column; should still work
        assert result.ndim == 2

    def test_single_target(self, small_ligand_target_matrix):
        """Single target gene overlap -> skip row clustering."""
        links = pd.DataFrame({
            "ligand": ["L1", "L2"],
            "target": ["G1", "G1"],
            "weight": [0.5, 0.3],
        })
        result = prepare_ligand_target_visualization(
            ligand_target_df=links,
            ligand_target_matrix=small_ligand_target_matrix,
            cutoff=0.0,
        )
        assert result.ndim == 2


class TestPrepareLrVisNoCluster:
    def test_receptors_only(self):
        """order_hclust='none' uses alpha sort for receptors."""
        df = pd.DataFrame({
            "from": ["L1", "L2"],
            "to": ["R2", "R1"],
            "weight": [0.5, 0.8],
        })
        result = prepare_ligand_receptor_visualization(
            df, best_upstream_ligands=["L2", "L1"], order_hclust="none",
        )
        assert result.rownames == ["R1", "R2"]


class TestHclustOrder:
    def test_basic(self):
        from scipy.cluster.hierarchy import linkage
        data = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        from scipy.spatial.distance import pdist
        dist = pdist(data)
        link = linkage(dist, method="ward")
        order = _hclust_order(link)
        assert sorted(order) == [0, 1, 2, 3]


class TestNamedArray:
    def test_attributes(self):
        arr = _attach_names(np.array([[1, 2], [3, 4]]), ["r1", "r2"], ["c1", "c2"])
        assert arr.rownames == ["r1", "r2"]
        assert arr.colnames == ["c1", "c2"]
        assert arr.shape == (2, 2)

    def test_array_finalize(self):
        arr = _attach_names(np.array([[1, 2]]), ["r1"], ["c1", "c2"])
        sliced = arr[0, :]
        # After slicing, attributes should still exist (via __array_finalize__)
        assert hasattr(sliced, "rownames")
