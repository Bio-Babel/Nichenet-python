"""Data loaders for all nichenetr data assets.

Provides convenience functions to load remote and bundled data assets
used by the nichenetr package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import scipy.sparse

from ._download import resolve_data_path

__all__ = [
    "NamedMatrix",
    "load_lr_network",
    "load_ligand_target_matrix",
    "load_weighted_networks",
    "load_sig_network",
    "load_gr_network",
    "load_ligand_tf_matrix",
    "load_seurat_obj",
    "load_hnscc_expression",
    "load_pemt_signature",
    "load_source_weights_df",
    "load_hyperparameter_list",
    "load_geneinfo",
    "load_geneinfo_alias",
    "load_optimized_source_weights_df",
]

_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


class NamedMatrix(NamedTuple):
    """A sparse matrix with row and column names.

    Attributes
    ----------
    data : scipy.sparse.csr_matrix
        The sparse data matrix.
    rownames : list[str]
        Row names (one per row of *data*).
    colnames : list[str]
        Column names (one per column of *data*).
    """

    data: scipy.sparse.csr_matrix
    rownames: list[str]
    colnames: list[str]


def _load_named_matrix(stem: str) -> NamedMatrix:
    """Load an .npz + _rownames.json + _colnames.json triplet.

    Parameters
    ----------
    stem : str
        Filename stem shared by the three files (e.g.
        ``"ligand_target_matrix_mouse"``).

    Returns
    -------
    NamedMatrix
        Sparse matrix with associated row and column names.
    """
    npz_path = resolve_data_path(f"{stem}.npz")
    row_path = resolve_data_path(f"{stem}_rownames.json")
    col_path = resolve_data_path(f"{stem}_colnames.json")

    data = scipy.sparse.load_npz(npz_path)
    if not scipy.sparse.issparse(data) or not isinstance(data, scipy.sparse.csr_matrix):
        data = scipy.sparse.csr_matrix(data)

    with open(row_path) as f:
        rownames: list[str] = json.load(f)
    with open(col_path) as f:
        colnames: list[str] = json.load(f)

    return NamedMatrix(data=data, rownames=rownames, colnames=colnames)


# ---------------------------------------------------------------------------
# Remote assets
# ---------------------------------------------------------------------------


def load_lr_network(organism: str = "mouse") -> pd.DataFrame:
    """Load the ligand-receptor network.

    Parameters
    ----------
    organism : str, optional
        Species identifier (``"mouse"`` or ``"human"``). Default is
        ``"mouse"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``from``, ``to``, ``source``, ``database``.
    """
    path = resolve_data_path(f"lr_network_{organism}.parquet")
    return pd.read_parquet(path)


def load_ligand_target_matrix(organism: str = "mouse") -> NamedMatrix:
    """Load the ligand-target prior model as a sparse matrix.

    Parameters
    ----------
    organism : str, optional
        Species identifier (``"mouse"`` or ``"human"``). Default is
        ``"mouse"``.

    Returns
    -------
    NamedMatrix
        Sparse matrix where rows are target genes and columns are ligands.
    """
    return _load_named_matrix(f"ligand_target_matrix_{organism}")


def load_weighted_networks(organism: str = "mouse") -> dict[str, pd.DataFrame]:
    """Load the weighted signaling and gene-regulatory networks.

    Parameters
    ----------
    organism : str, optional
        Species identifier (``"mouse"`` or ``"human"``). Default is
        ``"mouse"``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys ``"lr_sig"`` and ``"gr"``, each mapping to a
        DataFrame.
    """
    lr_sig_path = resolve_data_path(f"weighted_networks_{organism}_lr_sig.parquet")
    gr_path = resolve_data_path(f"weighted_networks_{organism}_gr.parquet")
    return {
        "lr_sig": pd.read_parquet(lr_sig_path),
        "gr": pd.read_parquet(gr_path),
    }


def load_sig_network() -> pd.DataFrame:
    """Load the signaling network (human).

    Returns
    -------
    pd.DataFrame
        Signaling network DataFrame.
    """
    path = resolve_data_path("sig_network_human.parquet")
    return pd.read_parquet(path)


def load_gr_network() -> pd.DataFrame:
    """Load the gene-regulatory network (human).

    Returns
    -------
    pd.DataFrame
        Gene-regulatory network DataFrame.
    """
    path = resolve_data_path("gr_network_human.parquet")
    return pd.read_parquet(path)


def load_ligand_tf_matrix() -> NamedMatrix:
    """Load the ligand-transcription factor matrix.

    Returns
    -------
    NamedMatrix
        Sparse matrix with TFs as rows and ligands as columns.
    """
    return _load_named_matrix("ligand_tf_matrix")


def load_seurat_obj() -> "anndata.AnnData":  # noqa: F821
    """Load the example Seurat object converted to AnnData format.

    Returns
    -------
    anndata.AnnData
        Annotated data matrix.
    """
    import anndata

    path = resolve_data_path("seuratObj.h5ad")
    return anndata.read_h5ad(path)


def load_hnscc_expression() -> dict:
    """Load HNSCC expression data and sample information.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"expression"`` : `NamedMatrix` — sparse expression matrix with
          gene and cell names.
        - ``"sample_info"`` : `pd.DataFrame` — per-cell metadata.
    """
    expression = _load_named_matrix("hnscc_expression")
    sample_info_path = resolve_data_path("hnscc_sample_info.parquet")
    sample_info = pd.read_parquet(sample_info_path)
    return {
        "expression": expression,
        "sample_info": sample_info,
    }


def load_pemt_signature() -> list[str]:
    """Load the pEMT gene signature.

    Returns
    -------
    list[str]
        List of gene names in the pEMT signature.
    """
    path = resolve_data_path("pemt_signature.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Bundled assets (in nichenetr/resources/)
# ---------------------------------------------------------------------------


def load_source_weights_df() -> pd.DataFrame:
    """Load the source weights DataFrame.

    Returns
    -------
    pd.DataFrame
        Data-source reliability weights.
    """
    return pd.read_parquet(_RESOURCES_DIR / "source_weights_df.parquet")


def load_hyperparameter_list() -> dict:
    """Load the default hyperparameter list.

    Returns
    -------
    dict
        Hyperparameters for model construction.
    """
    with open(_RESOURCES_DIR / "hyperparameter_list.json") as f:
        return json.load(f)


def load_geneinfo(version: str = "2022") -> pd.DataFrame:
    """Load gene information table.

    Parameters
    ----------
    version : str, optional
        Version tag. Use ``"2022"`` (default) for the 2022 snapshot or
        ``"human"`` for the legacy human-only table.

    Returns
    -------
    pd.DataFrame
        Gene information table.
    """
    filename = f"geneinfo_{version}.parquet"
    path = _RESOURCES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No geneinfo resource for version={version!r}. "
            f"Expected file: {path}"
        )
    return pd.read_parquet(path)


def load_geneinfo_alias(organism: str = "human") -> pd.DataFrame:
    """Load gene alias mapping table.

    Parameters
    ----------
    organism : str, optional
        Species identifier (``"human"`` or ``"mouse"``). Default is
        ``"human"``.

    Returns
    -------
    pd.DataFrame
        Gene alias mapping table.
    """
    filename = f"geneinfo_alias_{organism}.parquet"
    path = _RESOURCES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No geneinfo_alias resource for organism={organism!r}. "
            f"Expected file: {path}"
        )
    return pd.read_parquet(path)


def load_optimized_source_weights_df() -> pd.DataFrame:
    """Load the optimized source weights DataFrame.

    Returns
    -------
    pd.DataFrame
        Optimized data-source reliability weights.
    """
    return pd.read_parquet(_RESOURCES_DIR / "optimized_source_weights_df.parquet")
