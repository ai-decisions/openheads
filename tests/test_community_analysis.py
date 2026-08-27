"""Unit tests for GNN community analysis + shadow pathways."""

from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from openheads.community_analysis import GNNCommunityAnalyzer


@pytest.fixture
def analyzer():
    return GNNCommunityAnalyzer(high_risk_threshold=0.6, low_risk_threshold=0.3)


def _make_two_cluster_graph():
    """Create a graph with 2 clear clusters connected by a bridge node."""
    g = nx.Graph()
    # Cluster A: nodes 0-4 (high risk)
    for i in range(5):
        for j in range(i + 1, 5):
            g.add_edge(i, j)
    # Bridge node 5 — add BEFORE cluster B so graph.nodes() order matches labels
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    # Cluster B: nodes 6-10 (low risk)
    for i in range(6, 11):
        for j in range(i + 1, 11):
            g.add_edge(i, j)
    return g


class TestClusterEmbeddings:
    """Only THIS class calls cluster_embeddings, which needs hdbscan. The
    skip is scoped here: a module-level importorskip silently skipped the
    whole file — ten tests, seven of which never touch hdbscan."""

    @pytest.fixture(autouse=True)
    def _needs_hdbscan(self) -> None:
        pytest.importorskip("hdbscan", reason="hdbscan not installed")

    def test_clusters_found(self, analyzer) -> None:
        """HDBSCAN finds clusters in well-separated embeddings."""
        rng = np.random.default_rng(42)
        # Two well-separated clusters
        cluster_a = rng.normal(loc=0, scale=0.1, size=(30, 16))
        cluster_b = rng.normal(loc=5, scale=0.1, size=(30, 16))
        embeddings = np.vstack([cluster_a, cluster_b])

        result = analyzer.cluster_embeddings(embeddings, min_cluster_size=5)
        assert result["n_clusters"] >= 2
        assert len(result["cluster_labels"]) == 60

    def test_uniform_data(self, analyzer) -> None:
        """Very tight data may produce one cluster or all noise (HDBSCAN behavior)."""
        rng = np.random.default_rng(42)
        embeddings = rng.normal(loc=0, scale=0.01, size=(30, 16))
        result = analyzer.cluster_embeddings(embeddings, min_cluster_size=5)
        # HDBSCAN may classify all as noise when variance is too low
        assert result["n_clusters"] >= 0
        total = sum(result["cluster_sizes"].values()) + result["noise_count"]
        assert total == 30

    def test_cluster_sizes_sum(self, analyzer) -> None:
        """Cluster sizes + noise = total nodes."""
        rng = np.random.default_rng(42)
        cluster_a = rng.normal(loc=0, scale=0.1, size=(25, 8))
        cluster_b = rng.normal(loc=10, scale=0.1, size=(25, 8))
        embeddings = np.vstack([cluster_a, cluster_b])

        result = analyzer.cluster_embeddings(embeddings, min_cluster_size=5)
        total = sum(result["cluster_sizes"].values()) + result["noise_count"]
        assert total == 50


class TestDetectShadowPathways:
    def test_finds_pathway(self, analyzer) -> None:
        """Detects shadow pathway between high and low risk clusters."""
        g = _make_two_cluster_graph()
        # Cluster labels: 0-4=cluster 0, 5=noise(-1), 6-10=cluster 1
        labels = np.array([0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1])
        anomaly_scores = {
            **{i: 0.8 for i in range(5)},  # cluster 0: high risk
            5: 0.5,  # bridge
            **{i: 0.1 for i in range(6, 11)},  # cluster 1: low risk
        }

        pathways = analyzer.detect_shadow_pathways(g, labels, anomaly_scores)
        assert len(pathways) >= 1
        assert pathways[0]["bridge_nodes"] == [5]
        assert pathways[0]["high_risk_cluster"] == 0
        assert pathways[0]["low_risk_cluster"] == 1

    def test_empty_graph(self, analyzer) -> None:
        """Empty graph returns no pathways."""
        g = nx.Graph()
        pathways = analyzer.detect_shadow_pathways(g, np.array([]), {})
        assert pathways == []

    def test_no_high_risk(self, analyzer) -> None:
        """No pathways if no high risk clusters exist."""
        g = _make_two_cluster_graph()
        labels = np.array([0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1])
        anomaly_scores = {i: 0.1 for i in range(11)}  # all low risk

        pathways = analyzer.detect_shadow_pathways(g, labels, anomaly_scores)
        assert pathways == []

    def test_pathway_score(self, analyzer) -> None:
        """Pathway score = bridge_anomaly * (1/path_length)."""
        g = _make_two_cluster_graph()
        labels = np.array([0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1])
        anomaly_scores = {
            **{i: 0.8 for i in range(5)},
            5: 0.6,
            **{i: 0.1 for i in range(6, 11)},
        }

        pathways = analyzer.detect_shadow_pathways(g, labels, anomaly_scores)
        assert len(pathways) >= 1
        pw = pathways[0]
        expected_score = round(pw["bridge_anomaly"] * (1.0 / pw["path_length"]), 4)
        assert pw["score"] == expected_score


class TestCompareWithClassical:
    def test_returns_metrics(self, analyzer) -> None:
        """Comparison returns NMI and cluster counts."""
        g = _make_two_cluster_graph()
        gnn_labels = np.array([0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1])

        result = analyzer.compare_with_classical(g, gnn_labels)
        assert "nmi" in result
        assert "n_gnn_clusters" in result
        assert "n_louvain_communities" in result
        assert 0 <= result["nmi"] <= 1
        assert result["n_gnn_clusters"] == 2


class TestGenerateGNNBrief:
    def test_generates_brief(self, analyzer) -> None:
        """Brief generation calls the injected LLM client and returns structured result."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Shadow pathway analysis shows..."

        pathways = [
            {
                "high_risk_cluster": 0,
                "low_risk_cluster": 1,
                "path": [3, 5, 7],
                "path_length": 3,
                "bridge_nodes": [5],
                "bridge_anomaly": 0.6,
                "score": 0.2,
            }
        ]
        cluster_result = {"n_clusters": 2, "noise_count": 5}
        comparison = {"nmi": 0.75, "n_louvain_communities": 3}

        brief = analyzer.generate_gnn_brief(mock_llm, pathways, cluster_result, comparison)
        assert brief["brief_type"] == "gnn_analysis"
        assert brief["title"] == "GNN Shadow Pathway Analysis"
        assert "Shadow pathway" in brief["content"]
        assert brief["structured_data"]["n_pathways"] == 1
        mock_llm.invoke.assert_called_once()

    def test_brief_context_format(self, analyzer) -> None:
        """Brief context includes all key metrics."""
        context = analyzer._build_brief_context(
            pathways=[],
            cluster_result={"n_clusters": 3, "noise_count": 10},
            comparison={"nmi": 0.8, "n_louvain_communities": 4},
        )
        assert "GNN Clusters: 3" in context
        assert "Noise nodes: 10" in context
        assert "NMI (GNN vs Louvain): 0.8" in context
