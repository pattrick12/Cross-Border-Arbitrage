"""
Graph Embeddings — Node2Vec + KMeans Clustering
=================================================
Builds a correlation-based graph of FX instruments,
generates low-dimensional node embeddings via Node2Vec,
and clusters them with KMeans to identify co-moving groups
for pair selection.

Uses lightweight `node2vec` + `gensim` (not PyTorch Geometric).
"""

import os
import sys
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger("GraphEmbeddings")

try:
    from node2vec import Node2Vec
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False
    logger.warning("node2vec or sklearn not installed. Graph embeddings disabled.")


class CurrencyGraphEmbedder:
    """
    Builds a weighted graph from a correlation matrix,
    runs Node2Vec to generate embeddings, and clusters
    instruments with KMeans.

    Parameters
    ----------
    symbols : list[str]
        List of instrument symbols (e.g., ["EUR_USD", "GBP_USD", ...]).
    embedding_dim : int
        Dimensionality of the node embeddings (default: 16).
    n_clusters : int
        Number of KMeans clusters (default: 4).
    correlation_threshold : float
        Minimum |correlation| to create an edge (default: 0.2).
    """

    def __init__(
        self,
        symbols: List[str],
        embedding_dim: int = 16,
        n_clusters: int = 4,
        correlation_threshold: float = 0.2,
    ):
        self.symbols = symbols
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.correlation_threshold = correlation_threshold

        self.graph: Optional[nx.Graph] = None
        self.embeddings: Optional[Dict[str, np.ndarray]] = None
        self.clusters: Optional[Dict[str, int]] = None
        self.cluster_pairs: Optional[List[Tuple[str, str]]] = None

    def build_graph(self, correlation_matrix: np.ndarray) -> nx.Graph:
        """
        Build a weighted NetworkX graph from a correlation matrix.

        Edge weight = |correlation|, only edges above threshold.
        """
        n = len(self.symbols)
        assert correlation_matrix.shape == (n, n), \
            f"Correlation matrix shape {correlation_matrix.shape} != ({n}, {n})"

        G = nx.Graph()
        for i, sym in enumerate(self.symbols):
            G.add_node(sym)

        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(correlation_matrix[i, j])
                if corr >= self.correlation_threshold:
                    # Weight = correlation magnitude (higher = closer)
                    G.add_edge(self.symbols[i], self.symbols[j], weight=corr)

        self.graph = G
        logger.info(
            f"Built graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges "
            f"(threshold={self.correlation_threshold})"
        )
        return G

    def compute_embeddings(
        self,
        walk_length: int = 30,
        num_walks: int = 100,
        p: float = 1.0,
        q: float = 2.0,
        workers: int = 2,
    ) -> Dict[str, np.ndarray]:
        """
        Run Node2Vec on the graph to generate embeddings.

        Parameters
        ----------
        walk_length : Length of each random walk.
        num_walks : Number of random walks per node.
        p : Return parameter (1 = unbiased).
        q : In-out parameter (>1 = local, <1 = global).
        workers : Number of parallel workers.

        Returns
        -------
        Dict mapping symbol → embedding vector.
        """
        if not HAS_NODE2VEC:
            logger.error("node2vec not installed.")
            return {}

        if self.graph is None or self.graph.number_of_edges() == 0:
            logger.warning("Graph has no edges. Cannot compute embeddings.")
            # Return zero embeddings
            self.embeddings = {s: np.zeros(self.embedding_dim) for s in self.symbols}
            return self.embeddings

        logger.info(
            f"Running Node2Vec: walk_length={walk_length}, "
            f"num_walks={num_walks}, dim={self.embedding_dim}"
        )

        t0 = time.time()

        node2vec = Node2Vec(
            self.graph,
            dimensions=self.embedding_dim,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers,
            weight_key="weight",
            quiet=True,
        )

        model = node2vec.fit(
            window=10,
            min_count=1,
            batch_words=4,
            epochs=10,
        )

        self.embeddings = {}
        for sym in self.symbols:
            if sym in model.wv:
                self.embeddings[sym] = model.wv[sym]
            else:
                self.embeddings[sym] = np.zeros(self.embedding_dim)

        elapsed = time.time() - t0
        logger.info(f"Node2Vec completed in {elapsed:.2f}s")

        return self.embeddings

    def cluster(self) -> Dict[str, int]:
        """
        Run KMeans clustering on embeddings.

        Returns
        -------
        Dict mapping symbol → cluster_id.
        """
        if not self.embeddings:
            logger.error("No embeddings computed. Run compute_embeddings() first.")
            return {}

        # Stack embeddings into matrix
        symbols = list(self.embeddings.keys())
        X = np.vstack([self.embeddings[s] for s in symbols])

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal k if needed (use configured n_clusters)
        k = min(self.n_clusters, len(symbols))

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        self.clusters = {sym: int(label) for sym, label in zip(symbols, labels)}

        # Log cluster composition
        from collections import Counter
        cluster_counts = Counter(labels)
        for cid, count in sorted(cluster_counts.items()):
            members = [s for s, c in self.clusters.items() if c == cid]
            logger.info(f"  Cluster {cid} ({count} members): {members}")

        return self.clusters

    def get_candidate_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Generate candidate pairs for stat-arb from same-cluster instruments.

        Returns
        -------
        List of (symbol_a, symbol_b, cluster_id) tuples.
        """
        if not self.clusters:
            logger.error("No clusters computed. Run cluster() first.")
            return []

        # Group by cluster
        from collections import defaultdict
        cluster_members = defaultdict(list)
        for sym, cid in self.clusters.items():
            cluster_members[cid].append(sym)

        pairs = []
        for cid, members in cluster_members.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.append((members[i], members[j], cid))

        self.cluster_pairs = pairs
        logger.info(f"Generated {len(pairs)} candidate pairs from {len(cluster_members)} clusters")
        return pairs

    def get_embedding_vector(self, symbol: str) -> np.ndarray:
        """Get the embedding vector for a symbol (for ML features)."""
        if self.embeddings and symbol in self.embeddings:
            return self.embeddings[symbol]
        return np.zeros(self.embedding_dim)

    def run_pipeline(
        self,
        correlation_matrix: np.ndarray,
    ) -> List[Tuple[str, str, int]]:
        """
        Full pipeline: build graph → embeddings → cluster → pairs.

        Returns candidate pairs for stat-arb.
        """
        logger.info("Running full graph embedding pipeline...")
        self.build_graph(correlation_matrix)
        self.compute_embeddings()
        self.cluster()
        return self.get_candidate_pairs()
