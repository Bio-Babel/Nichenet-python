"""Ligand-target signaling network extraction and visualization.

Infer signaling paths from ligands to target genes via transcription
factors (TFs), format the resulting subnetwork for visualization with
networkx, and map supporting data sources onto each edge.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .datasets import NamedMatrix

__all__ = [
    "get_ligand_signaling_path",
    "format_signaling_graph",
    "infer_supporting_datasources",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_network_df(
    ligand: str,
    target: str,
    top_n: int,
    weighted_networks: Dict[str, pd.DataFrame],
    ligand_tf_matrix: NamedMatrix,
) -> pd.DataFrame:
    """Find the top-*top_n* TFs linking *ligand* to *target*.

    For a given (ligand, target) pair, rank candidate TFs by the
    product of the ligand-TF regulatory potential and the GRN weight
    from TF to target, then return the top *top_n* rows.

    Parameters
    ----------
    ligand : str
        Ligand of interest (must be a column in *ligand_tf_matrix*).
    target : str
        Target gene of interest.
    top_n : int
        Number of top TFs to keep.
    weighted_networks : dict
        Must contain ``"gr"`` (DataFrame with columns ``from``, ``to``,
        ``weight``).
    ligand_tf_matrix : NamedMatrix
        Matrix with TFs as rows and ligands as columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``TF``, ``weight``, ``ligand``, ``weight_grn``,
        ``total_weight``, ``to``.
    """
    col_idx = ligand_tf_matrix.colnames.index(ligand)
    col_vec = np.asarray(
        ligand_tf_matrix.data[:, col_idx].todense()
    ).ravel()

    tf_df = pd.DataFrame({
        "TF": ligand_tf_matrix.rownames,
        "weight": col_vec,
    })
    tf_df = tf_df[tf_df["weight"] > 0].copy()
    tf_df["ligand"] = ligand

    gr = weighted_networks["gr"]
    reg = gr[gr["to"] == target][["from", "weight"]].rename(
        columns={"from": "TF", "weight": "weight_grn"}
    )

    combined = tf_df.merge(reg, on="TF", how="inner")
    combined["total_weight"] = combined["weight"] * combined["weight_grn"]
    combined["to"] = target
    combined = combined.sort_values("total_weight", ascending=False).head(top_n)
    return combined


def _construct_ligand_signaling_df(
    ligands_all: List[str],
    targets_all: List[str],
    top_n: int,
    weighted_networks: Dict[str, pd.DataFrame],
    ligand_tf_matrix: NamedMatrix,
) -> pd.DataFrame:
    """Build a DataFrame of top TF regulators for every (ligand, target) pair.

    Parameters
    ----------
    ligands_all : list of str
        Ligands of interest.
    targets_all : list of str
        Target genes of interest.
    top_n : int
        Number of top TFs per (ligand, target) pair.
    weighted_networks : dict
        Must contain ``"gr"`` DataFrame.
    ligand_tf_matrix : NamedMatrix
        Ligand-TF regulatory potential matrix.

    Returns
    -------
    pd.DataFrame
        Concatenated rows from :func:`_get_network_df` for all pairs.
    """
    frames: list[pd.DataFrame] = []
    for lig, tgt in product(ligands_all, targets_all):
        df = _get_network_df(lig, tgt, top_n, weighted_networks, ligand_tf_matrix)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["TF", "weight", "ligand", "weight_grn", "total_weight", "to"]
        )
    return pd.concat(frames, ignore_index=True)


def _get_shortest_path_signaling(
    ligand: str,
    signaling_df: pd.DataFrame,
    signaling_graph: nx.DiGraph,
) -> List[str]:
    """Find intermediate nodes on shortest paths from *ligand* to its TFs.

    Parameters
    ----------
    ligand : str
        Source ligand node.
    signaling_df : pd.DataFrame
        Must contain columns ``ligand`` and ``TF``.
    signaling_graph : nx.DiGraph
        Directed graph with ``weight`` edge attributes representing
        inverse original weights (i.e., distances).

    Returns
    -------
    list of str
        Unique intermediate node names (excluding the ligand itself).
    """
    tfs = signaling_df.loc[signaling_df["ligand"] == ligand, "TF"].unique().tolist()
    nodes: list[str] = []
    for tf in tfs:
        try:
            path = nx.dijkstra_path(signaling_graph, ligand, tf, weight="weight")
            nodes.extend(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    # Remove the ligand itself and deduplicate
    return list(dict.fromkeys(n for n in nodes if n != ligand))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_ligand_signaling_path(
    ligand_tf_matrix: NamedMatrix,
    ligands_all: List[str],
    targets_all: List[str],
    top_n_regulators: int = 4,
    weighted_networks: Optional[Dict[str, pd.DataFrame]] = None,
    ligands_position: str = "cols",
    minmax_scaling: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Extract signaling paths from ligands to target genes through TFs.

    For each (ligand, target) pair the top *top_n_regulators*
    transcription factors are selected by combined ligand-TF and
    TF-target weight.  Shortest weighted paths from each ligand to
    its TFs are then traced through the integrated signaling network,
    yielding a compact signaling sub-network.

    Parameters
    ----------
    ligand_tf_matrix : NamedMatrix
        Ligand-TF regulatory potential matrix.  Ligands in columns
        (default) or rows depending on *ligands_position*.
    ligands_all : list of str
        Ligands of interest.
    targets_all : list of str
        Target genes of interest.
    top_n_regulators : int, default 4
        Number of top TFs per (ligand, target) pair.
    weighted_networks : dict, optional
        Must contain ``"lr_sig"`` and ``"gr"`` DataFrames, each with
        columns ``from``, ``to``, ``weight``.
    ligands_position : {'cols', 'rows'}, default 'cols'
        Whether ligands correspond to columns or rows in
        *ligand_tf_matrix*.
    minmax_scaling : bool, default False
        If ``True``, apply min-max scaling (shifted to [0.75, 1.75])
        to edge weights before returning.

    Returns
    -------
    dict
        ``"sig"`` : pd.DataFrame -- signaling edges (from, to, weight).
        ``"gr"``  : pd.DataFrame -- gene-regulatory edges (from, to,
        weight).

    Raises
    ------
    ValueError
        If inputs fail validation checks.
    """
    # --- validation -----------------------------------------------------------
    if weighted_networks is None:
        raise ValueError("weighted_networks must be provided")
    if not isinstance(weighted_networks, dict):
        raise ValueError("weighted_networks must be a dict")
    for key in ("lr_sig", "gr"):
        if key not in weighted_networks or not isinstance(
            weighted_networks[key], pd.DataFrame
        ):
            raise ValueError(f"weighted_networks['{key}'] must be a DataFrame")
        if "weight" not in weighted_networks[key].columns:
            raise ValueError(
                f"weighted_networks['{key}'] must contain a 'weight' column"
            )

    if ligands_position not in ("cols", "rows"):
        raise ValueError("ligands_position must be 'cols' or 'rows'")

    if ligands_position == "cols":
        available = set(ligand_tf_matrix.colnames)
    else:
        available = set(ligand_tf_matrix.rownames)
    missing = [l for l in ligands_all if l not in available]
    if missing:
        raise ValueError(
            f"Ligands not found in ligand_tf_matrix: {missing}"
        )

    gr_targets = set(weighted_networks["gr"]["to"].unique())
    missing_targets = [t for t in targets_all if t not in gr_targets]
    if missing_targets:
        raise ValueError(
            f"Target genes not in gene regulatory network: {missing_targets}"
        )

    if not isinstance(top_n_regulators, (int, float)) or top_n_regulators <= 0:
        raise ValueError("top_n_regulators must be a number > 0")
    top_n_regulators = int(top_n_regulators)

    # --- transpose if ligands are in rows ------------------------------------
    if ligands_position == "rows":
        import scipy.sparse

        ligand_tf_matrix = NamedMatrix(
            data=ligand_tf_matrix.data.T.tocsr(),
            rownames=ligand_tf_matrix.colnames,
            colnames=ligand_tf_matrix.rownames,
        )

    # --- construct TF ranking ------------------------------------------------
    final_combined_df = _construct_ligand_signaling_df(
        ligands_all, targets_all, top_n_regulators,
        weighted_networks, ligand_tf_matrix,
    )

    # --- build signaling graph with inverse weights (distances) --------------
    lr_sig = weighted_networks["lr_sig"].copy()
    lr_sig_inv = lr_sig.assign(weight=1.0 / lr_sig["weight"])
    G = nx.DiGraph()
    for _, row in lr_sig_inv.iterrows():
        src, dst, w = row["from"], row["to"], row["weight"]
        # keep the minimum distance if there are parallel edges
        if G.has_edge(src, dst):
            G[src][dst]["weight"] = min(G[src][dst]["weight"], w)
        else:
            G.add_edge(src, dst, weight=w)

    # --- find intermediate nodes via shortest paths --------------------------
    tf_nodes: list[str] = []
    for lig in ligands_all:
        tf_nodes.extend(
            _get_shortest_path_signaling(lig, final_combined_df, G)
        )
    tf_nodes = list(dict.fromkeys(tf_nodes))  # unique, order-preserving

    # --- extract signaling sub-network edges ---------------------------------
    node_set = set(ligands_all) | set(tf_nodes)
    sig = lr_sig[
        lr_sig["from"].isin(node_set) & lr_sig["to"].isin(set(tf_nodes))
    ].copy()
    # Aggregate parallel edges
    sig = (
        sig.groupby(["from", "to"], as_index=False)
        .agg({"weight": "sum"})
    )

    # --- extract gene-regulatory sub-network edges ---------------------------
    tf_set = set(final_combined_df["TF"].unique())
    target_set = set(targets_all)
    gr_full = weighted_networks["gr"]
    gr = gr_full[
        gr_full["from"].isin(tf_set) & gr_full["to"].isin(target_set)
    ].drop_duplicates()

    # --- optional min-max scaling --------------------------------------------
    if minmax_scaling:
        for df in (sig, gr):
            if len(df) > 0:
                w = df["weight"]
                w_range = w.max() - w.min()
                if w_range > 0:
                    df["weight"] = ((w - w.min()) / w_range) + 0.75
                else:
                    df["weight"] = 0.75

    return {"sig": sig.reset_index(drop=True), "gr": gr.reset_index(drop=True)}


def format_signaling_graph(
    signaling_graph_list: Dict[str, pd.DataFrame],
    ligands_all: List[str],
    targets_all: List[str],
    sig_color: str = "steelblue",
    gr_color: str = "orange",
) -> Dict[str, pd.DataFrame]:
    """Format a signaling graph for networkx visualization.

    Converts the signaling and gene-regulatory edge DataFrames
    returned by :func:`get_ligand_signaling_path` into a pair of
    ``nodes`` and ``edges`` DataFrames suitable for building a
    :class:`networkx.DiGraph`.

    Parameters
    ----------
    signaling_graph_list : dict
        Must contain ``"sig"`` and ``"gr"`` DataFrames (each with
        columns ``from``, ``to``, ``weight``).
    ligands_all : list of str
        Ligand gene names (coloured with *sig_color*).
    targets_all : list of str
        Target gene names (coloured with *gr_color*).
    sig_color : str, default 'steelblue'
        Colour for signaling edges and ligand nodes.
    gr_color : str, default 'orange'
        Colour for gene-regulatory edges and target nodes.

    Returns
    -------
    dict
        ``"nodes"`` : pd.DataFrame with columns ``id``, ``label``,
        ``color``, ``type`` (one of ``"ligand"``, ``"target"``,
        ``"mediator"``).

        ``"edges"`` : pd.DataFrame with columns ``from``, ``to``,
        ``weight``, ``color``, ``layer`` (``"sig"`` or ``"gr"``).

    Raises
    ------
    ValueError
        If inputs fail validation checks.
    """
    if not isinstance(signaling_graph_list, dict):
        raise ValueError("signaling_graph_list must be a dict")
    for key in ("sig", "gr"):
        if key not in signaling_graph_list or not isinstance(
            signaling_graph_list[key], pd.DataFrame
        ):
            raise ValueError(
                f"signaling_graph_list['{key}'] must be a DataFrame"
            )

    tf_signaling = signaling_graph_list["sig"]
    tf_regulatory = signaling_graph_list["gr"]

    sig_sources = set(tf_signaling["from"].unique())
    gr_dests = set(tf_regulatory["to"].unique())
    ligand_set = set(ligands_all)
    target_set = set(targets_all)

    missing_lig = ligand_set - sig_sources
    if missing_lig:
        raise ValueError(
            f"Ligands not found in signaling_graph_list['sig']: {missing_lig}"
        )
    missing_tgt = target_set - gr_dests
    if missing_tgt:
        raise ValueError(
            f"Targets not found in signaling_graph_list['gr']: {missing_tgt}"
        )

    if not isinstance(sig_color, str):
        raise ValueError("sig_color must be a single colour string")
    if not isinstance(gr_color, str):
        raise ValueError("gr_color must be a single colour string")

    # --- build combined edge list --------------------------------------------
    sig_edges = tf_signaling[["from", "to", "weight"]].copy()
    sig_edges["color"] = sig_color
    sig_edges["layer"] = "sig"

    gr_edges = tf_regulatory[["from", "to", "weight"]].copy()
    gr_edges["color"] = gr_color
    gr_edges["layer"] = "gr"

    edges = pd.concat([sig_edges, gr_edges], ignore_index=True)

    # --- build node list -----------------------------------------------------
    all_node_names = list(
        dict.fromkeys(edges["from"].tolist() + edges["to"].tolist())
    )

    node_records: list[dict] = []
    for idx, name in enumerate(all_node_names, start=1):
        if name in ligand_set:
            color = sig_color
            ntype = "ligand"
        elif name in target_set:
            color = gr_color
            ntype = "target"
        else:
            color = "#7F7F7F"
            ntype = "mediator"
        node_records.append({
            "id": idx,
            "label": name,
            "color": color,
            "type": ntype,
        })

    nodes = pd.DataFrame(node_records)
    return {"nodes": nodes, "edges": edges}


def infer_supporting_datasources(
    signaling_graph_list: Dict[str, pd.DataFrame],
    lr_network: pd.DataFrame,
    sig_network: pd.DataFrame,
    gr_network: pd.DataFrame,
) -> pd.DataFrame:
    """Map data sources supporting each edge in the signaling graph.

    For every edge in the extracted signaling sub-network, look up
    which original source databases (from *lr_network*, *sig_network*,
    *gr_network*) contain that interaction.

    Parameters
    ----------
    signaling_graph_list : dict
        Must contain ``"sig"`` and ``"gr"`` DataFrames (each with
        columns ``from``, ``to``, ``weight``).
    lr_network : pd.DataFrame
        Ligand-receptor network with at least columns ``from``,
        ``to``, ``source``.
    sig_network : pd.DataFrame
        Signaling network with at least columns ``from``, ``to``,
        ``source``.
    gr_network : pd.DataFrame
        Gene-regulatory network with at least columns ``from``,
        ``to``, ``source``.

    Returns
    -------
    pd.DataFrame
        Columns: ``from``, ``to``, ``source``, ``database`` (if
        present in the input networks), ``layer`` (``"regulatory"``
        or ``"ligand_signaling"``).

    Raises
    ------
    ValueError
        If inputs fail validation checks.
    """
    for name, obj in [
        ("lr_network", lr_network),
        ("sig_network", sig_network),
        ("gr_network", gr_network),
    ]:
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"{name} must be a DataFrame")

    if not isinstance(signaling_graph_list, dict):
        raise ValueError("signaling_graph_list must be a dict")
    for key in ("sig", "gr"):
        if key not in signaling_graph_list or not isinstance(
            signaling_graph_list[key], pd.DataFrame
        ):
            raise ValueError(
                f"signaling_graph_list['{key}'] must be a DataFrame"
            )

    # Signaling edges (from, to) -- unique pairs
    sig_pairs = signaling_graph_list["sig"][["from", "to"]].drop_duplicates()
    # Regulatory edges (from, to) -- unique pairs
    reg_pairs = signaling_graph_list["gr"][["from", "to"]].drop_duplicates()

    # Merge regulatory pairs against gr_network
    reg_sources = reg_pairs.merge(gr_network, on=["from", "to"], how="inner")
    reg_sources["layer"] = "regulatory"

    # Merge signaling pairs against combined lr + sig network
    lr_sig_combined = pd.concat([lr_network, sig_network], ignore_index=True)
    sig_sources = sig_pairs.merge(lr_sig_combined, on=["from", "to"], how="inner")
    sig_sources["layer"] = "ligand_signaling"

    result = pd.concat([reg_sources, sig_sources], ignore_index=True)

    # Keep a consistent set of output columns
    base_cols = ["from", "to", "source"]
    extra_cols = [
        c for c in ["database"] if c in result.columns
    ]
    return result[base_cols + extra_cols + ["layer"]].reset_index(drop=True)
