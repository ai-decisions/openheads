"""Bridge between NetworkX graphs and PyTorch Geometric Data objects."""

from __future__ import annotations

from typing import Any

import networkx as nx
import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
    from torch_geometric.data import Data

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Data = None  # type: ignore[assignment,misc]


N_FEATURES = 5


def nx_to_pyg(graph: nx.Graph) -> Any:
    """Convert a NetworkX graph to a PyTorch Geometric Data object.

    Node features (structural, scale-invariant via per-graph normalization):
    - degree centrality
    - betweenness centrality (k-approximation for large graphs)
    - clustering coefficient
    - pagerank
    - katz centrality

    Returns:
        torch_geometric.data.Data with x, edge_index, and node_names mapping.
        Returns None if torch is not available.
    """
    if not HAS_TORCH:
        logger.warning("torch_not_available", msg="Skipping nx_to_pyg conversion")
        return None

    if graph.number_of_nodes() < 2:
        return Data(
            x=torch.zeros((graph.number_of_nodes(), N_FEATURES), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            node_names=list(graph.nodes()),
        )

    nodes = list(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Structural features matching training data
    degree_c = nx.degree_centrality(graph)
    k = min(100, graph.number_of_nodes())
    betweenness_c = nx.betweenness_centrality(graph, k=k)
    clustering_c = nx.clustering(graph)
    pagerank = nx.pagerank(graph)
    try:
        katz_c = nx.katz_centrality(graph, max_iter=1000, tol=1e-4)
    except Exception as exc:
        logger.debug("katz_centrality_failed", error=str(exc), n_nodes=graph.number_of_nodes())
        katz_c = {n: 0.0 for n in nodes}

    features = []
    for node in nodes:
        features.append(
            [
                degree_c.get(node, 0.0),
                betweenness_c.get(node, 0.0),
                clustering_c.get(node, 0.0),
                pagerank.get(node, 0.0),
                katz_c.get(node, 0.0),
            ]
        )

    x = torch.tensor(features, dtype=torch.float)

    # Per-graph min-max normalization — makes features scale-invariant
    # so a model trained on a 200K-node graph works on 100-node production graphs
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    x_range = x_max - x_min
    x_range[x_range < 1e-8] = 1.0
    x = (x - x_min) / x_range

    # Build edge_index
    edges_src = []
    edges_dst = []
    for u, v in graph.edges():
        edges_src.extend([node_to_idx[u], node_to_idx[v]])
        edges_dst.extend([node_to_idx[v], node_to_idx[u]])

    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    # Attach mapping for later use
    data.node_names = nodes  # type: ignore[attr-defined]
    data.node_to_idx = node_to_idx  # type: ignore[attr-defined]

    logger.info(
        "nx_to_pyg_converted",
        nodes=len(nodes),
        edges=edge_index.shape[1],
        features=x.shape[1],
    )

    return data
