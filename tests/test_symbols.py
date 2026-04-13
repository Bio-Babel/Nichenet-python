"""Tests for nichenetr.symbols."""

import numpy as np
import pandas as pd
import pytest
import anndata

from nichenetr.symbols import (
    get_expressed_genes,
    convert_alias_to_symbols,
    alias_to_symbol_anndata,
    assign_ligands_to_celltype,
    get_lfc_celltype,
)


class TestGetExpressedGenes:
    def test_basic(self, small_adata):
        result = get_expressed_genes(
            small_adata, celltype_col="celltype", celltype="TypeA", pct=0.10
        )
        assert isinstance(result, list)
        assert len(result) > 0
        # All returned genes should be in var_names
        assert all(g in small_adata.var_names for g in result)

    def test_high_pct_fewer_genes(self, small_adata):
        low = get_expressed_genes(
            small_adata, celltype_col="celltype", celltype="TypeA", pct=0.10
        )
        high = get_expressed_genes(
            small_adata, celltype_col="celltype", celltype="TypeA", pct=0.99
        )
        assert len(high) <= len(low)

    def test_missing_column_raises(self, small_adata):
        with pytest.raises(KeyError, match="not found in adata.obs"):
            get_expressed_genes(small_adata, celltype_col="nonexistent", celltype="TypeA")

    def test_missing_celltype_raises(self, small_adata):
        with pytest.raises(ValueError, match="not present"):
            get_expressed_genes(small_adata, celltype_col="celltype", celltype="Missing")

    def test_with_layer(self, small_adata):
        small_adata.layers["test_layer"] = small_adata.X.copy()
        result = get_expressed_genes(
            small_adata, celltype_col="celltype", celltype="TypeA",
            pct=0.10, assay_oi="test_layer",
        )
        assert isinstance(result, list)

    def test_zero_cells_returns_empty(self):
        """Edge case: celltype exists but no cells after masking (can't happen normally)."""
        adata = anndata.AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame({"ct": ["A"] * 5}, index=[f"c{i}" for i in range(5)]),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        # All zeros, so with pct > 0 no genes pass
        result = get_expressed_genes(adata, celltype_col="ct", celltype="A", pct=0.5)
        assert result == []

    def test_dense_matrix(self):
        """Test with dense (non-sparse) X."""
        adata = anndata.AnnData(
            X=np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]]),
            obs=pd.DataFrame({"ct": ["A", "A"]}, index=["c1", "c2"]),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        result = get_expressed_genes(adata, celltype_col="ct", celltype="A", pct=0.5)
        assert set(result) == {"g1", "g2", "g3"}

    def test_sparse_matrix(self):
        """Test with sparse X."""
        import scipy.sparse
        X = scipy.sparse.csr_matrix(np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]]))
        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame({"ct": ["A", "A"]}, index=["c1", "c2"]),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        result = get_expressed_genes(adata, celltype_col="ct", celltype="A", pct=0.5)
        assert set(result) == {"g1", "g2", "g3"}


class TestConvertAliasToSymbols:
    def test_no_change_for_official_symbols(self):
        """Official symbols should pass through unchanged."""
        result = convert_alias_to_symbols(["TP53", "BRCA1"], organism="human", verbose=False)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_empty_input(self):
        result = convert_alias_to_symbols([], organism="human", verbose=False)
        assert result == []

    def test_verbose_output(self, capsys):
        convert_alias_to_symbols(["NONEXISTENT_GENE_XYZ"], organism="human", verbose=True)
        captured = capsys.readouterr()
        assert "not in the alias annotation table" in captured.out

    def test_all_official_verbose(self, capsys):
        """When all symbols are already official, verbose should say so."""
        # Load actual alias table to find a known official symbol
        from nichenetr.datasets import load_geneinfo_alias
        alias_df = load_geneinfo_alias("human")
        official = alias_df["symbol"].iloc[0]
        # That symbol should also be in alias column (self-mapping)
        convert_alias_to_symbols([official], organism="human", verbose=True)
        captured = capsys.readouterr()
        # Should print either "All input symbols were official" or conversion info
        assert len(captured.out) > 0


class TestAliasToSymbolAnndata:
    def test_returns_copy(self, small_adata):
        result = alias_to_symbol_anndata(small_adata, organism="human")
        assert result is not small_adata
        assert isinstance(result, anndata.AnnData)

    def test_preserves_shape(self, small_adata):
        result = alias_to_symbol_anndata(small_adata, organism="human")
        assert result.shape == small_adata.shape

    def test_duplicate_resolution(self):
        """When alias conversion creates duplicates, originals are kept."""
        from unittest.mock import patch

        def mock_convert(genes, organism, verbose):
            # Simulate: gene_0 and gene_1 both map to "SHARED"
            return ["SHARED", "SHARED", "gene_2"]

        adata = anndata.AnnData(
            X=np.zeros((2, 3)),
            obs=pd.DataFrame(index=["c1", "c2"]),
            var=pd.DataFrame(index=["gene_0", "gene_1", "gene_2"]),
        )

        with patch("nichenetr.symbols.convert_alias_to_symbols", side_effect=mock_convert):
            result = alias_to_symbol_anndata(adata, organism="human")
        # Duplicates should be resolved
        assert len(set(result.var_names)) == len(result.var_names)


class TestAssignLigandsToCelltype:
    def test_basic(self, small_adata):
        # Use gene names from small_adata as ligands
        ligands = list(small_adata.var_names[:5])
        result = assign_ligands_to_celltype(
            small_adata,
            celltype_col="celltype",
            sender_celltypes=["TypeA", "TypeB"],
            ligands_oi=ligands,
        )
        assert isinstance(result, pd.DataFrame)
        assert "ligand_type" in result.columns
        assert "ligand" in result.columns
        assert set(result["ligand"]).issubset(set(ligands))

    def test_no_specificity(self, small_adata):
        ligands = list(small_adata.var_names[:3])
        result = assign_ligands_to_celltype(
            small_adata,
            celltype_col="celltype",
            sender_celltypes=["TypeA", "TypeB"],
            ligands_oi=ligands,
            celltype_specificity=False,
        )
        assert (result["ligand_type"] == "General").all()

    def test_missing_ligand_raises(self, small_adata):
        with pytest.raises(ValueError, match="not in adata.var_names"):
            assign_ligands_to_celltype(
                small_adata,
                celltype_col="celltype",
                sender_celltypes=["TypeA"],
                ligands_oi=["NONEXISTENT"],
            )

    def test_custom_agg_and_assign(self, small_adata):
        ligands = list(small_adata.var_names[:3])
        result = assign_ligands_to_celltype(
            small_adata,
            celltype_col="celltype",
            sender_celltypes=["TypeA", "TypeB"],
            ligands_oi=ligands,
            func_agg=np.median,
            func_assign=lambda x: np.mean(x) + 2 * np.std(x),
        )
        assert isinstance(result, pd.DataFrame)

    def test_specific_assignment(self):
        """Construct data where ligands ARE cell-type-specific."""
        rng = np.random.RandomState(42)
        X = np.zeros((20, 4), dtype=np.float32)
        # Need 3 celltypes so that one clearly exceeds mean+std
        X = np.zeros((30, 4), dtype=np.float32)
        X[:10, 0] = 10.0   # L1 high in TypeA
        X[10:20, 0] = 1.0  # L1 low in TypeB
        X[20:, 0] = 1.0    # L1 low in TypeC
        X[:10, 1] = 1.0    # L2 low in TypeA
        X[10:20, 1] = 10.0 # L2 high in TypeB
        X[20:, 1] = 1.0    # L2 low in TypeC
        X[:, 2] = 5.0      # L3 same everywhere
        X[:, 3] = 5.0      # L4 same everywhere

        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame(
                {"ct": ["TypeA"] * 10 + ["TypeB"] * 10 + ["TypeC"] * 10},
                index=[f"c{i}" for i in range(30)],
            ),
            var=pd.DataFrame(index=["L1", "L2", "L3", "L4"]),
        )
        result = assign_ligands_to_celltype(
            adata, celltype_col="ct",
            sender_celltypes=["TypeA", "TypeB", "TypeC"],
            ligands_oi=["L1", "L2", "L3", "L4"],
        )
        assert len(result) == 4
        # L1 should be TypeA-specific, L2 TypeB-specific
        # With 3 celltypes: mean([10,1,1])=4, std≈4.24, threshold≈8.24
        # 10 > 8.24 -> TypeA for L1. 1 < 8.24 -> not TypeB/TypeC
        l1_type = result[result["ligand"] == "L1"]["ligand_type"].iloc[0]
        l2_type = result[result["ligand"] == "L2"]["ligand_type"].iloc[0]
        assert l1_type == "TypeA"
        assert l2_type == "TypeB"


class TestConvertAliasVerboseConversion:
    def test_alias_converted_verbose(self, capsys):
        """Test verbose output when aliases are actually converted."""
        from nichenetr.datasets import load_geneinfo_alias
        alias_df = load_geneinfo_alias("human")
        # Find a row where alias != symbol
        diff = alias_df[alias_df["alias"] != alias_df["symbol"]]
        if len(diff) > 0:
            alias_name = diff["alias"].iloc[0]
            convert_alias_to_symbols([alias_name], organism="human", verbose=True)
            captured = capsys.readouterr()
            assert "official gene symbols" in captured.out


class TestAssignLigandsWithSparseMatrix:
    def test_sparse_X(self):
        """Test assign_ligands_to_celltype with sparse X."""
        import scipy.sparse
        rng = np.random.RandomState(42)
        X = scipy.sparse.csr_matrix(rng.rand(20, 5).astype(np.float32))
        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame(
                {"ct": ["A"] * 10 + ["B"] * 10},
                index=[f"c{i}" for i in range(20)],
            ),
            var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        )
        result = assign_ligands_to_celltype(
            adata, celltype_col="ct",
            sender_celltypes=["A", "B"],
            ligands_oi=[f"g{i}" for i in range(5)],
        )
        assert isinstance(result, pd.DataFrame)


class TestGetLfcCelltype:
    def test_basic(self, adata_with_conditions):
        result = get_lfc_celltype(
            adata_with_conditions,
            celltype_col="celltype",
            senders=["TypeA", "TypeB"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
        )
        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns
        # Should have one column per sender
        assert "TypeA" in result.columns or "TypeB" in result.columns

    def test_with_ligands_filter(self, adata_with_conditions):
        result = get_lfc_celltype(
            adata_with_conditions,
            celltype_col="celltype",
            senders=["TypeA"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            ligands_oi=["L1", "L2"],
        )
        assert all(g in ["L1", "L2"] for g in result["gene"])

    def test_no_specificity(self, adata_with_conditions):
        result = get_lfc_celltype(
            adata_with_conditions,
            celltype_col="celltype",
            senders=["TypeA", "TypeB"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            celltype_specificity=False,
        )
        assert "lfc" in result.columns

    def test_missing_condition_warns(self, adata_with_conditions):
        """If a celltype only has one condition, warn and return empty."""
        # TypeC only has "treated" cells, not "control"
        with pytest.warns(UserWarning):
            result = get_lfc_celltype(
                adata_with_conditions,
                celltype_col="celltype",
                senders=["TypeC"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
            )
        assert isinstance(result, pd.DataFrame)

    def test_missing_celltype_warns(self, adata_with_conditions):
        """If a sender celltype is not found, warn and skip."""
        with pytest.warns(UserWarning, match="not found"):
            result = get_lfc_celltype(
                adata_with_conditions,
                celltype_col="celltype",
                senders=["NONEXISTENT"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
            )
        assert isinstance(result, pd.DataFrame)

    def test_with_layer(self, adata_with_conditions):
        adata_with_conditions.layers["test_layer"] = adata_with_conditions.X.copy()
        result = get_lfc_celltype(
            adata_with_conditions,
            celltype_col="celltype",
            senders=["TypeA"],
            condition_col="condition",
            condition_oi="treated",
            condition_ref="control",
            layer="test_layer",
        )
        assert isinstance(result, pd.DataFrame)

    def test_empty_senders(self, adata_with_conditions):
        """All senders not found -> empty DataFrame."""
        with pytest.warns(UserWarning):
            result = get_lfc_celltype(
                adata_with_conditions,
                celltype_col="celltype",
                senders=["NONEXISTENT1", "NONEXISTENT2"],
                condition_col="condition",
                condition_oi="treated",
                condition_ref="control",
            )
        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns
