"""GNN Community Analysis — HDBSCAN clustering + shadow pathway detection."""

from typing import Any

import networkx as nx
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class GNNCommunityAnalyzer:
    """Analyze GNN embeddings: cluster nodes, detect shadow pathways, compare with classical."""

    def __init__(
        self,
        high_risk_threshold: float = 0.6,
        low_risk_threshold: float = 0.3,
    ) -> None:
        self.high_risk_threshold = high_risk_threshold
        self.low_risk_threshold = low_risk_threshold

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
    ) -> dict[str, Any]:
        """Cluster GNN embeddings using HDBSCAN.

        Args:
            embeddings: array of shape [n_nodes, embedding_dim]
            min_cluster_size: minimum cluster size for HDBSCAN

        Returns:
            dict with cluster_labels, n_clusters, cluster_sizes, noise_count
        """
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int(np.sum(labels == -1))

        cluster_sizes = {}
        for c in range(n_clusters):
            cluster_sizes[c] = int(np.sum(labels == c))

        logger.info(
            "embeddings_clustered",
            n_clusters=n_clusters,
            noise_count=noise_count,
            total_nodes=len(labels),
        )

        return {
            "cluster_labels": labels,
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "noise_count": noise_count,
        }

    def detect_shadow_pathways(
        self,
        graph: nx.Graph,
        cluster_labels: np.ndarray,
        anomaly_scores: dict[int, float],
        top_k: int = 10,
        nodes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect shadow pathways between high-risk and low-risk clusters.

        Args:
            graph: NetworkX graph (nodes may be strings).
            cluster_labels: integer cluster labels per node index.
            anomaly_scores: {index: score} mapping.
            top_k: max pathways to return.
            nodes: ordered node names matching index positions.

        Returns:
            list of pathway dicts sorted by score (descending)
        """
        if len(graph.nodes) == 0:
            return []

        # Map indices to graph node names
        if nodes is None:
            nodes = list(graph.nodes())
        idx_to_node = {i: n for i, n in enumerate(nodes)}

        # Classify clusters by risk
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        high_risk_clusters: list[int] = []
        low_risk_clusters: list[int] = []

        for c in range(n_clusters):
            cluster_nodes = [i for i, lbl in enumerate(cluster_labels) if lbl == c]
            if not cluster_nodes:
                continue
            mean_anomaly = np.mean([anomaly_scores.get(n, 0) for n in cluster_nodes])
            if mean_anomaly > self.high_risk_threshold:
                high_risk_clusters.append(c)
            elif mean_anomaly < self.low_risk_threshold:
                low_risk_clusters.append(c)

        if not high_risk_clusters or not low_risk_clusters:
            logger.info("no_shadow_pathways", reason="missing high or low risk clusters")
            return []

        # Build node sets per cluster (integer indices)
        cluster_node_sets: dict[int, set[int]] = {}
        for c in high_risk_clusters + low_risk_clusters:
            cluster_node_sets[c] = {i for i, lbl in enumerate(cluster_labels) if lbl == c}

        # Build reverse mapping: graph node name → index
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Find paths between high-risk and low-risk representative nodes
        pathways: list[dict[str, Any]] = []
        for hr_cluster in high_risk_clusters:
            hr_nodes = list(cluster_node_sets[hr_cluster])
            hr_rep_idx = max(hr_nodes, key=lambda n: anomaly_scores.get(n, 0))
            hr_rep_name = idx_to_node.get(hr_rep_idx)

            for lr_cluster in low_risk_clusters:
                lr_nodes = list(cluster_node_sets[lr_cluster])
                lr_rep_idx = min(lr_nodes, key=lambda n: anomaly_scores.get(n, 0))
                lr_rep_name = idx_to_node.get(lr_rep_idx)

                if hr_rep_name not in graph or lr_rep_name not in graph:
                    continue

                try:
                    path_names = nx.shortest_path(graph, hr_rep_name, lr_rep_name)
                except nx.NetworkXNoPath:
                    continue

                # Convert path node names back to indices for anomaly lookup
                path_indices = [node_to_idx[n] for n in path_names if n in node_to_idx]

                # Identify bridge nodes (not in either cluster)
                combined = cluster_node_sets[hr_cluster] | cluster_node_sets[lr_cluster]
                bridge_indices = [i for i in path_indices if i not in combined]
                bridge_names = [idx_to_node[i] for i in bridge_indices]

                # Score: mean bridge anomaly * (1/path_length)
                if bridge_indices:
                    bridge_anomaly = np.mean([anomaly_scores.get(n, 0) for n in bridge_indices])
                else:
                    bridge_anomaly = 0.0

                path_length = len(path_names)
                score = float(bridge_anomaly * (1.0 / max(path_length, 1)))

                pathways.append(
                    {
                        "high_risk_cluster": hr_cluster,
                        "low_risk_cluster": lr_cluster,
                        "path": path_names,
                        "path_length": path_length,
                        "bridge_nodes": bridge_names,
                        "bridge_anomaly": round(float(bridge_anomaly), 4),
                        "score": round(score, 4),
                    }
                )

        # Sort by score descending, limit to top_k
        pathways.sort(key=lambda p: p["score"], reverse=True)
        pathways = pathways[:top_k]

        logger.info(
            "shadow_pathways_detected",
            count=len(pathways),
            high_risk_clusters=len(high_risk_clusters),
            low_risk_clusters=len(low_risk_clusters),
        )
        return pathways

    def compare_with_classical(
        self,
        graph: nx.Graph,
        gnn_labels: np.ndarray,
        nodes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare GNN clusters with Louvain community detection.

        Args:
            graph: NetworkX graph (nodes may be strings).
            gnn_labels: cluster labels per node index.
            nodes: ordered node names matching index positions.

        Returns:
            dict with overlap metrics (NMI, n_gnn_clusters, n_louvain_communities)
        """
        from sklearn.metrics import normalized_mutual_info_score

        if nodes is None:
            nodes = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Louvain communities
        louvain_communities = nx.community.louvain_communities(graph, seed=42)
        louvain_labels = np.full(len(gnn_labels), -1, dtype=int)
        for idx, community in enumerate(louvain_communities):
            for node in community:
                node_idx = node_to_idx.get(node)
                if node_idx is not None:
                    louvain_labels[node_idx] = idx

        # Filter out noise nodes (-1) for NMI calculation
        valid_mask = (gnn_labels >= 0) & (louvain_labels >= 0)
        if valid_mask.sum() > 0:
            nmi = float(
                normalized_mutual_info_score(gnn_labels[valid_mask], louvain_labels[valid_mask])
            )
        else:
            nmi = 0.0

        n_gnn = len(set(gnn_labels)) - (1 if -1 in gnn_labels else 0)
        n_louvain = len(louvain_communities)

        result = {
            "nmi": round(nmi, 4),
            "n_gnn_clusters": n_gnn,
            "n_louvain_communities": n_louvain,
        }

        logger.info("classical_comparison", **result)
        return result

    def generate_gnn_brief(
        self,
        bedrock_client: Any,
        pathways: list[dict[str, Any]],
        cluster_result: dict[str, Any],
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a structural brief with GNN insights via an LLM client.

        The client must expose an ``invoke(prompt) -> str`` method.

        Returns:
            dict with brief_type="gnn_analysis", title, content, structured_data
        """
        context = self._build_brief_context(pathways, cluster_result, comparison)

        prompt = (
            "You are a financial crime analyst. Based on the following GNN analysis "
            "of a transaction network, write a concise intelligence brief (3-5 paragraphs). "
            "Focus on: shadow pathways detected, bridge nodes that may facilitate "
            "money laundering, and how GNN communities differ from classical detection.\n\n"
            f"{context}"
        )

        response = bedrock_client.invoke(prompt)
        content = response if isinstance(response, str) else str(response)

        return {
            "brief_type": "gnn_analysis",
            "title": "GNN Shadow Pathway Analysis",
            "content": content,
            "structured_data": {
                "n_clusters": cluster_result.get("n_clusters", 0),
                "n_pathways": len(pathways),
                "top_pathway_score": pathways[0]["score"] if pathways else 0,
                "nmi_vs_louvain": comparison.get("nmi", 0),
            },
        }

    @staticmethod
    def _build_brief_context(
        pathways: list[dict[str, Any]],
        cluster_result: dict[str, Any],
        comparison: dict[str, Any],
    ) -> str:
        lines = [
            f"GNN Clusters: {cluster_result.get('n_clusters', 0)}",
            f"Noise nodes: {cluster_result.get('noise_count', 0)}",
            f"Louvain communities: {comparison.get('n_louvain_communities', 0)}",
            f"NMI (GNN vs Louvain): {comparison.get('nmi', 0)}",
            f"Shadow pathways found: {len(pathways)}",
        ]

        for i, pw in enumerate(pathways[:5]):
            lines.append(
                f"  Pathway {i + 1}: cluster {pw['high_risk_cluster']} → "
                f"cluster {pw['low_risk_cluster']}, "
                f"length={pw['path_length']}, "
                f"bridge_nodes={len(pw['bridge_nodes'])}, "
                f"score={pw['score']}"
            )

        return "\n".join(lines)
