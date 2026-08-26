"""Synthetic graph generators for GNN testing and development.

Generates graphs with known patterns (hubs, communities, anomalies, AI-agent structures)
so tests can run without the Elliptic dataset.
"""

from typing import Any

import networkx as nx
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class SyntheticGraphGenerator:
    """Generate synthetic graphs with known structural patterns."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_shadow_network(
        self,
        n_nodes: int = 500,
        n_communities: int = 5,
        hub_fraction: float = 0.05,
        illicit_fraction: float = 0.1,
    ) -> dict[str, Any]:
        """Generate a graph with hub/community/anomaly patterns.

        Returns dict compatible with PyG Data construction:
            - node_features: np.ndarray [n_nodes, n_features]
            - edge_index: np.ndarray [2, n_edges]
            - labels: np.ndarray [n_nodes] (0=licit, 1=illicit, 2=unknown)
            - timesteps: np.ndarray [n_nodes] (1-49 for temporal split)
        """
        # Build community structure
        sizes = self._split_sizes(n_nodes, n_communities)
        p_in = 0.15  # intra-community edge probability
        p_out = 0.01  # inter-community edge probability
        graph = nx.stochastic_block_model(sizes, self._block_matrix(n_communities, p_in, p_out))

        # Add hub nodes with extra connections
        n_hubs = max(1, int(n_nodes * hub_fraction))
        hub_nodes = self._rng.choice(n_nodes, size=n_hubs, replace=False).tolist()
        for hub in hub_nodes:
            targets = self._rng.choice(n_nodes, size=min(30, n_nodes // 5), replace=False)
            for t in targets:
                if t != hub:
                    graph.add_edge(hub, t)

        # Generate node features (simulate Elliptic-like 166 features + 3 structural)
        n_features = 169
        node_features = self._rng.standard_normal((n_nodes, n_features)).astype(np.float32)

        # Compute structural features and inject them
        degree_c = nx.degree_centrality(graph)
        betweenness_c = nx.betweenness_centrality(graph)
        clustering_c = nx.clustering(graph)
        for i in range(n_nodes):
            node_features[i, 166] = degree_c.get(i, 0)
            node_features[i, 167] = betweenness_c.get(i, 0)
            node_features[i, 168] = clustering_c.get(i, 0)

        # Assign labels
        labels = np.zeros(n_nodes, dtype=np.int64)  # 0 = licit
        n_illicit = max(1, int(n_nodes * illicit_fraction))
        illicit_nodes = self._rng.choice(n_nodes, size=n_illicit, replace=False)
        labels[illicit_nodes] = 1  # illicit

        # Mark some as unknown
        n_unknown = max(1, int(n_nodes * 0.3))
        unknown_nodes = self._rng.choice(n_nodes, size=n_unknown, replace=False)
        labels[unknown_nodes] = 2  # unknown

        # Assign timesteps (1-49)
        timesteps = self._rng.integers(1, 50, size=n_nodes)

        # Edge index
        edges = list(graph.edges())
        if edges:
            edge_index = np.array(edges, dtype=np.int64).T
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        logger.info(
            "synthetic_shadow_network",
            nodes=n_nodes,
            edges=len(edges),
            communities=n_communities,
            hubs=n_hubs,
            illicit=int(np.sum(labels == 1)),
            licit=int(np.sum(labels == 0)),
            unknown=int(np.sum(labels == 2)),
        )

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "labels": labels,
            "timesteps": timesteps,
            "hub_nodes": hub_nodes,
            "n_communities": n_communities,
        }

    def generate_agent_patterns(
        self,
        n_agents: int = 10,
        fan_size: int = 5,
    ) -> dict[str, Any]:
        """Generate fan-out → transit → fan-in AI-agent structures.

        Each agent creates: fan_size source wallets → transit node → fan_size sink wallets.
        Returns same dict format as generate_shadow_network.
        """
        nodes_per_agent = 1 + fan_size * 2  # transit + fan_out + fan_in
        n_nodes = n_agents * nodes_per_agent
        n_features = 169

        node_features = self._rng.standard_normal((n_nodes, n_features)).astype(np.float32)
        labels = np.full(n_nodes, 2, dtype=np.int64)  # all agent/unknown
        edges = []

        for agent_idx in range(n_agents):
            base = agent_idx * nodes_per_agent
            transit = base
            labels[transit] = 1  # transit nodes are illicit

            # Fan-out: sources → transit
            for i in range(fan_size):
                src = base + 1 + i
                edges.append((src, transit))
                labels[src] = 0  # source wallets look licit

            # Fan-in: transit → sinks
            for i in range(fan_size):
                sink = base + 1 + fan_size + i
                edges.append((transit, sink))
                labels[sink] = 0  # sink wallets look licit

            # Agent patterns have super-regular features
            for i in range(nodes_per_agent):
                node_features[base + i, :10] *= 0.1  # reduce variance (agent signature)

        if edges:
            edge_index = np.array(edges, dtype=np.int64).T
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        timesteps = self._rng.integers(40, 50, size=n_nodes)  # agents appear in later timesteps

        logger.info(
            "synthetic_agent_patterns",
            agents=n_agents,
            nodes=n_nodes,
            edges=len(edges),
        )

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "labels": labels,
            "timesteps": timesteps,
            "n_agents": n_agents,
        }

    def _split_sizes(self, total: int, k: int) -> list[int]:
        """Split total into k roughly equal parts."""
        base = total // k
        remainder = total % k
        return [base + (1 if i < remainder else 0) for i in range(k)]

    @staticmethod
    def _block_matrix(k: int, p_in: float, p_out: float) -> list[list[float]]:
        """Create block probability matrix for stochastic block model."""
        return [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
