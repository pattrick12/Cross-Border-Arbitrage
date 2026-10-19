"""
XGBoost Classifier — Final Prediction Model
=============================================
Binary classifier that combines:
  - LSTM hidden states (temporal patterns)
  - Tabular features (returns, volatility, graph embeddings)
  - Econometric features (ARIMA mean, GARCH vol)

Output: P(price_up) probability for trade direction.

Treated as CLASSIFICATION (up/down), not regression (price level).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import get_logger

logger = get_logger("XGBoostClassifier")

try:
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, classification_report,
    )
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost or sklearn not installed.")


class DirectionClassifier:
    """
    XGBoost binary classifier for price direction prediction.

    Features consumed:
        - LSTM hidden state vector (dim = hidden_dim)
        - log_return, rolling_mean_5, rolling_std_5
        - rolling_mean_20, rolling_std_20
        - arima_mean, garch_vol
        - ou_zscore
        - graph_embedding vector (dim = embedding_dim)

    Label: 1 if next-period return > 0, else 0.

    Parameters
    ----------
    model_path : str
        Path to save/load the model.
    n_estimators : int
        Number of boosting rounds (default: 200).
    max_depth : int
        Maximum tree depth (default: 5).
    learning_rate : float
        Boosting learning rate (default: 0.05).
    """

    def __init__(
        self,
        model_path: str = "./models/xgboost_direction.json",
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
    ):
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[Dict[str, float]] = None

        if HAS_XGB:
            self._build_model()

    def _build_model(self):
        """Initialize XGBoost classifier."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            tree_method="hist",           # Fast histogram-based
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,                # L1 regularization
            reg_lambda=1.0,               # L2 regularization
            min_child_weight=3,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cross_validate: bool = True,
    ) -> Dict:
        """
        Train the XGBoost classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,) — binary labels (0/1)
        feature_names : optional list of feature column names
        cross_validate : if True, run time-series cross-validation

        Returns
        -------
        Dict with training metrics.
        """
        if not HAS_XGB:
            return {"error": "xgboost not installed"}

        if feature_names:
            self.feature_names = feature_names

        logger.info(f"Training XGBoost: X={X.shape}, y={y.shape}, "
                     f"positive_rate={y.mean():.3f}")

        metrics = {}

        # Time-series cross-validation
        if cross_validate and len(X) > 200:
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                self.model, X, y,
                cv=tscv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            metrics["cv_auc_mean"] = float(cv_scores.mean())
            metrics["cv_auc_std"] = float(cv_scores.std())
            logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train-test split (last 20% as test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Fit
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            metrics["auc"] = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            metrics["auc"] = 0.5

        logger.info(
            f"Test metrics: acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}"
        )

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            if self.feature_names and len(self.feature_names) == len(importances):
                self.feature_importances_ = dict(
                    zip(self.feature_names, importances.tolist())
                )
                # Log top 10 features
                sorted_feats = sorted(
                    self.feature_importances_.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
                logger.info("Top features:")
                for fname, imp in sorted_feats:
                    logger.info(f"  {fname}: {imp:.4f}")

        # Save model
        self.save()

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict P(price_up) for each sample.

        Returns
        -------
        np.ndarray of shape (n_samples,) with probabilities.
        """
        if not HAS_XGB or self.model is None:
            return np.full(X.shape[0] if X.ndim > 1 else 1, 0.5)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return np.full(X.shape[0], 0.5)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary direction (0 or 1)."""
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

    def save(self):
        """Save model to disk."""
        if self.model is None:
            return

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)

        # Save feature names alongside
        meta_path = self.model_path.replace(".json", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "feature_importances": self.feature_importances_,
            }, f, indent=2)

        logger.info(f"XGBoost model saved to {self.model_path}")

    def load(self) -> bool:
        """Load a previously saved model."""
        if not HAS_XGB:
            return False

        if not os.path.exists(self.model_path):
            logger.debug(f"No saved model at {self.model_path}")
            return False

        try:
            self._build_model()
            self.model.load_model(self.model_path)

            meta_path = self.model_path.replace(".json", "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    self.feature_names = meta.get("feature_names", [])
                    self.feature_importances_ = meta.get("feature_importances")

            logger.info(f"XGBoost model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load XGBoost: {e}")
            return False

    @staticmethod
    def create_labels(returns: np.ndarray) -> np.ndarray:
        """
        Create binary labels from returns.

        1 if return > 0, else 0.
        """
        return (returns > 0).astype(int)
