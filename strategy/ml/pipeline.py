"""
ML Pipeline Orchestrator
=========================
Ties together the full prediction pipeline:

    Features → ARIMA/GARCH → LSTM → XGBoost → P(direction)

Two modes:
  1. Training: Batch-process historical data from QuestDB
  2. Inference: Real-time prediction from streaming features

Feature vector for XGBoost:
    [log_return, rolling_mean_5, rolling_std_5, rolling_mean_20, rolling_std_20,
     arima_mean, garch_vol, lstm_h0, lstm_h1, ..., lstm_h31,
     graph_e0, graph_e1, ..., graph_e15]
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import get_logger
from strategy.ml.econometrics import EconometricFeatureExtractor
from strategy.ml.lstm_model import LSTMFeatureExtractor
from strategy.ml.xgboost_classifier import DirectionClassifier

logger = get_logger("MLPipeline")

# Default dimensions
TABULAR_FEATURES = [
    "log_return", "rolling_mean_5", "rolling_std_5",
    "rolling_mean_20", "rolling_std_20",
    "arima_mean", "garch_vol", "ou_zscore",
]
LSTM_HIDDEN_DIM = 32
GRAPH_EMBEDDING_DIM = 16


class MLPipeline:
    """
    End-to-end ML prediction pipeline.

    Architecture:
        Raw features → Econometrics (ARIMA/GARCH) → augmented features
        → LSTM (sequence → hidden state) → XGBoost (tabular + hidden state)
        → P(price_up)

    Parameters
    ----------
    sequence_length : int
        LSTM input sequence length.
    lstm_hidden_dim : int
        LSTM hidden state dimensionality.
    graph_embedding_dim : int
        Graph embedding vector dimensionality.
    model_dir : str
        Directory for saved models.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        lstm_hidden_dim: int = LSTM_HIDDEN_DIM,
        graph_embedding_dim: int = GRAPH_EMBEDDING_DIM,
        model_dir: str = "./models",
    ):
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.graph_embedding_dim = graph_embedding_dim
        self.model_dir = model_dir

        # Total feature dim for XGBoost:
        # tabular (8) + lstm_hidden (32) + graph_embedding (16) = 56
        self.n_tabular = len(TABULAR_FEATURES)
        self.total_features = self.n_tabular + lstm_hidden_dim + graph_embedding_dim

        # LSTM input dim: tabular features (no graph embedding in sequence)
        self.lstm_input_dim = self.n_tabular

        # Component models
        self.econometrics = EconometricFeatureExtractor()
        self.lstm = LSTMFeatureExtractor(
            input_dim=self.lstm_input_dim,
            sequence_length=sequence_length,
            hidden_dim=lstm_hidden_dim,
            model_path=os.path.join(model_dir, "lstm_extractor.keras"),
        )
        self.classifier = DirectionClassifier(
            model_path=os.path.join(model_dir, "xgboost_direction.json"),
        )

        # Per-symbol feature buffers for real-time inference
        self.feature_buffers: Dict[str, deque] = {}

        # Feature names for XGBoost
        self.feature_names = (
            TABULAR_FEATURES
            + [f"lstm_h{i}" for i in range(lstm_hidden_dim)]
            + [f"graph_e{i}" for i in range(graph_embedding_dim)]
        )

        self._is_trained = False

    def train(
        self,
        returns_dict: Dict[str, np.ndarray],
        graph_embeddings: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 20,
    ) -> Dict:
        """
        Train the full pipeline on historical data.

        Parameters
        ----------
        returns_dict : dict mapping symbol → array of log-returns
        graph_embeddings : dict mapping symbol → embedding vector
        epochs : LSTM training epochs

        Returns
        -------
        Dict with training metrics.
        """
        logger.info("=== Training ML Pipeline ===")
        t0 = time.time()
        all_metrics = {}

        # Collect all features across symbols
        all_X_lstm = []
        all_y_lstm = []
        all_X_xgb = []
        all_y_xgb = []

        for symbol, returns in returns_dict.items():
            if len(returns) < self.sequence_length + 50:
                logger.debug(f"Skipping {symbol}: only {len(returns)} returns")
                continue

            # Step 1: Econometric features
            econ_features = []
            for i in range(50, len(returns)):
                window = returns[max(0, i - 200):i]
                feat = self.econometrics.extract(window)
                econ_features.append(feat)

            # Step 2: Build tabular feature matrix
            # Align: returns[50:] with econ_features
            aligned_returns = returns[50:]
            n = min(len(aligned_returns), len(econ_features))

            feature_matrix = np.zeros((n, self.lstm_input_dim))
            for i in range(n):
                feature_matrix[i, 0] = aligned_returns[i]
                # Rolling stats (recompute from aligned returns)
                if i >= 5:
                    window_5 = aligned_returns[i - 5:i]
                    feature_matrix[i, 1] = np.mean(window_5)
                    feature_matrix[i, 2] = np.std(window_5)
                if i >= 20:
                    window_20 = aligned_returns[i - 20:i]
                    feature_matrix[i, 3] = np.mean(window_20)
                    feature_matrix[i, 4] = np.std(window_20)
                feature_matrix[i, 5] = econ_features[i]["arima_mean"]
                feature_matrix[i, 6] = econ_features[i]["garch_vol"]
                # ou_zscore = 0 for training (will be filled in inference)
                feature_matrix[i, 7] = 0.0

            # Step 3: Prepare LSTM sequences
            X_seq, y_seq = LSTMFeatureExtractor.prepare_sequences(
                feature_matrix, self.sequence_length
            )
            if len(X_seq) > 0:
                all_X_lstm.append(X_seq)
                all_y_lstm.append(y_seq)

            # Step 4: Extract LSTM hidden states (after LSTM training below)
            # For now, collect what we have
            graph_emb = (
                graph_embeddings.get(symbol, np.zeros(self.graph_embedding_dim))
                if graph_embeddings else np.zeros(self.graph_embedding_dim)
            )

        # Train LSTM
        if all_X_lstm:
            X_lstm = np.concatenate(all_X_lstm, axis=0)
            y_lstm = np.concatenate(all_y_lstm, axis=0)
            logger.info(f"LSTM training data: {X_lstm.shape}")
            lstm_metrics = self.lstm.train(X_lstm, y_lstm, epochs=epochs)
            all_metrics["lstm"] = lstm_metrics

            # Now extract hidden states for XGBoost training
            for symbol, returns in returns_dict.items():
                if len(returns) < self.sequence_length + 50:
                    continue

                # Rebuild feature matrix
                econ_features = []
                for i in range(50, len(returns)):
                    window = returns[max(0, i - 200):i]
                    feat = self.econometrics.extract(window)
                    econ_features.append(feat)

                aligned_returns = returns[50:]
                n = min(len(aligned_returns), len(econ_features))

                feature_matrix = np.zeros((n, self.lstm_input_dim))
                for i in range(n):
                    feature_matrix[i, 0] = aligned_returns[i]
                    if i >= 5:
                        feature_matrix[i, 1] = np.mean(aligned_returns[i - 5:i])
                        feature_matrix[i, 2] = np.std(aligned_returns[i - 5:i])
                    if i >= 20:
                        feature_matrix[i, 3] = np.mean(aligned_returns[i - 20:i])
                        feature_matrix[i, 4] = np.std(aligned_returns[i - 20:i])
                    feature_matrix[i, 5] = econ_features[i]["arima_mean"]
                    feature_matrix[i, 6] = econ_features[i]["garch_vol"]

                X_seq, y_seq = LSTMFeatureExtractor.prepare_sequences(
                    feature_matrix, self.sequence_length
                )
                if len(X_seq) == 0:
                    continue

                # Extract hidden states
                hidden_states = self.lstm.extract_hidden_state(X_seq)

                # Graph embedding (constant per symbol)
                graph_emb = (
                    graph_embeddings.get(symbol, np.zeros(self.graph_embedding_dim))
                    if graph_embeddings else np.zeros(self.graph_embedding_dim)
                )
                graph_emb_tiled = np.tile(graph_emb, (len(hidden_states), 1))

                # Tabular features (from the end of each sequence)
                tabular = feature_matrix[self.sequence_length:][:len(hidden_states)]

                # Concatenate: tabular + lstm_hidden + graph_embedding
                X_combined = np.hstack([tabular, hidden_states, graph_emb_tiled])
                y_labels = DirectionClassifier.create_labels(y_seq.flatten())

                all_X_xgb.append(X_combined)
                all_y_xgb.append(y_labels)

        # Train XGBoost
        if all_X_xgb:
            X_xgb = np.concatenate(all_X_xgb, axis=0)
            y_xgb = np.concatenate(all_y_xgb, axis=0)
            logger.info(f"XGBoost training data: {X_xgb.shape}")
            xgb_metrics = self.classifier.train(
                X_xgb, y_xgb,
                feature_names=self.feature_names[:X_xgb.shape[1]],
            )
            all_metrics["xgboost"] = xgb_metrics
            self._is_trained = True

        elapsed = time.time() - t0
        logger.info(f"Pipeline training complete in {elapsed:.1f}s")
        return all_metrics

    def predict(
        self,
        symbol: str,
        features: Dict[str, float],
        graph_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Real-time inference: predict P(price_up).

        Parameters
        ----------
        symbol : instrument symbol
        features : dict with keys matching TABULAR_FEATURES
        graph_embedding : optional Node2Vec embedding

        Returns
        -------
        Probability of price going up (0.0 to 1.0).
        """
        # Initialize buffer for this symbol
        if symbol not in self.feature_buffers:
            self.feature_buffers[symbol] = deque(maxlen=self.sequence_length)

        # Build feature vector for this timestep
        feat_vec = np.zeros(self.lstm_input_dim)
        for i, name in enumerate(TABULAR_FEATURES):
            feat_vec[i] = features.get(name, 0.0)

        self.feature_buffers[symbol].append(feat_vec)

        # Need full sequence for LSTM
        if len(self.feature_buffers[symbol]) < self.sequence_length:
            return 0.5  # Neutral

        # LSTM hidden state
        sequence = np.array(list(self.feature_buffers[symbol]))
        hidden_state = self.lstm.extract_hidden_state(sequence)
        if hidden_state.ndim > 1:
            hidden_state = hidden_state[0]

        # Graph embedding
        if graph_embedding is None:
            graph_embedding = np.zeros(self.graph_embedding_dim)

        # Concatenate for XGBoost
        x = np.concatenate([feat_vec, hidden_state, graph_embedding])
        prob = self.classifier.predict_proba(x.reshape(1, -1))

        return float(prob[0])

    def load(self) -> bool:
        """Load all saved models."""
        lstm_ok = self.lstm.load()
        xgb_ok = self.classifier.load()
        self._is_trained = xgb_ok
        return lstm_ok and xgb_ok

    @property
    def is_trained(self) -> bool:
        return self._is_trained
