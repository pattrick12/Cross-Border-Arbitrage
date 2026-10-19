"""
LSTM Model — Sequence Feature Extractor
=========================================
Keras LSTM that takes multi-feature sequences and outputs
a fixed-size hidden-state vector (NOT a price prediction).

The hidden state captures temporal patterns and is passed
to XGBoost as an additional feature vector.

Input features per timestep:
    [log_return, volume, arima_mean, garch_vol, graph_embedding...]

Output:
    Hidden state vector of size `hidden_dim` (default 32).
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import get_logger

logger = get_logger("LSTM")

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
    logger.info(f"TensorFlow {tf.__version__} loaded (GPU: {len(tf.config.list_physical_devices('GPU'))})")
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not installed. LSTM disabled — using zero embeddings.")


class LSTMFeatureExtractor:
    """
    LSTM that extracts temporal hidden state vectors from feature sequences.

    Architecture:
        Input → LSTM(hidden_dim) → Dense(hidden_dim) → Output (hidden state)

    The model is trained as an autoencoder: the hidden state is decoded
    back to predict the next-step return (supervised signal), but we extract
    the hidden state as the feature — not the prediction.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep.
    sequence_length : int
        Number of timesteps in each input sequence.
    hidden_dim : int
        Dimensionality of the output hidden state.
    model_path : str
        Path to save/load trained model weights.
    """

    def __init__(
        self,
        input_dim: int = 6,
        sequence_length: int = 60,
        hidden_dim: int = 32,
        model_path: str = "./models/lstm_extractor.keras",
    ):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.model_path = model_path
        self.model = None
        self.encoder = None  # Extracts hidden state only

        if HAS_TF:
            self._build_model()

    def _build_model(self):
        """Build the LSTM autoencoder architecture."""
        # Input
        inputs = keras.Input(shape=(self.sequence_length, self.input_dim), name="sequence_input")

        # LSTM encoder
        lstm_out = layers.LSTM(
            self.hidden_dim,
            return_sequences=False,
            name="lstm_encoder",
        )(inputs)

        # Hidden state (this is what we extract)
        hidden = layers.Dense(
            self.hidden_dim,
            activation="relu",
            name="hidden_state",
        )(lstm_out)

        # Dropout for regularization
        hidden_drop = layers.Dropout(0.2)(hidden)

        # Decoder head: predict next return (training signal)
        output = layers.Dense(1, activation="linear", name="return_prediction")(hidden_drop)

        # Full model (for training)
        self.model = keras.Model(inputs=inputs, outputs=output, name="lstm_extractor")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )

        # Encoder model (for inference — outputs hidden state)
        self.encoder = keras.Model(
            inputs=inputs,
            outputs=self.model.get_layer("hidden_state").output,
            name="lstm_encoder_only",
        )

        logger.info(
            f"LSTM model built: input=({self.sequence_length}, {self.input_dim}) "
            f"→ hidden={self.hidden_dim} → output=1"
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> dict:
        """
        Train the LSTM on sequence data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, sequence_length, input_dim)
        y : np.ndarray of shape (n_samples, 1) — next-step return labels

        Returns
        -------
        Training history dict.
        """
        if not HAS_TF or self.model is None:
            logger.warning("TF not available, skipping training.")
            return {}

        logger.info(f"Training LSTM: X={X.shape}, y={y.shape}, epochs={epochs}")

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True,
                ),
            ],
        )

        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        logger.info(f"LSTM saved to {self.model_path}")

        final_loss = history.history["loss"][-1]
        val_loss = history.history.get("val_loss", [0])[-1]
        logger.info(f"Training complete: loss={final_loss:.6f}, val_loss={val_loss:.6f}")

        return history.history

    def extract_hidden_state(self, X: np.ndarray) -> np.ndarray:
        """
        Extract hidden state vectors from input sequences.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, sequence_length, input_dim)
            or (sequence_length, input_dim) for a single sample.

        Returns
        -------
        np.ndarray of shape (n_samples, hidden_dim)
        """
        if not HAS_TF or self.encoder is None:
            # Return zero embeddings
            if X.ndim == 2:
                return np.zeros(self.hidden_dim)
            return np.zeros((X.shape[0], self.hidden_dim))

        if X.ndim == 2:
            X = X[np.newaxis, ...]

        return self.encoder.predict(X, verbose=0)

    def load(self) -> bool:
        """Load a previously saved model."""
        if not HAS_TF:
            return False

        if not os.path.exists(self.model_path):
            logger.debug(f"No saved model found at {self.model_path}")
            return False

        try:
            self.model = keras.models.load_model(self.model_path)
            # Rebuild encoder from loaded model
            self.encoder = keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("hidden_state").output,
            )
            logger.info(f"LSTM loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LSTM: {e}")
            return False

    @staticmethod
    def prepare_sequences(
        features: np.ndarray,
        sequence_length: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a 2D feature array into sequences + labels for training.

        Parameters
        ----------
        features : np.ndarray of shape (n_timesteps, n_features)
            First column is assumed to be log_return (used as label).
        sequence_length : int
            Length of each input sequence.

        Returns
        -------
        (X, y) where X has shape (n_samples, seq_len, n_features)
        and y has shape (n_samples, 1).
        """
        n = len(features)
        if n <= sequence_length:
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(sequence_length, n):
            X.append(features[i - sequence_length:i])
            y.append(features[i, 0])  # Next return as label

        return np.array(X), np.array(y).reshape(-1, 1)
