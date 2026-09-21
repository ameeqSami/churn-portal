"""
ctf.py  –  Correlation Threshold Filter
========================================
Provides the `CorrelationThresholdFilter` sklearn-compatible transformer
that performs feature selection based on each feature's Pearson correlation
with the binary target variable.

Why this matters for churn prediction:
  Random forests can handle many features, but keeping only features with
  a meaningful correlation to the target (|r| >= threshold) reduces noise,
  speeds up training, and helps prevent the model from over-fitting on
  spurious patterns.

Typical usage inside a sklearn Pipeline:
    Pipeline([
        ('clean', clean_cls()),
        ('ctf',   CorrelationThresholdFilter(threshold=0.14)),
        ('model', RandomForestClassifier()),
    ])
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class CorrelationThresholdFilter(BaseEstimator, TransformerMixin):
    """
    Feature selector that keeps only columns whose absolute Pearson
    correlation with the binary target y is >= ``threshold``.

    Parameters
    ----------
    threshold : float, default=0.10
        Minimum |correlation| a feature must have with the target to be
        retained.  Raise this to be more aggressive; lower to keep more
        features.

    Attributes
    ----------
    selected_features : pd.Index or None
        Column names that passed the correlation threshold, populated
        after ``fit()``.
    """

    def __init__(self, threshold=0.10):
        self.threshold = threshold
        self.selected_features = None  # Set by fit(); None before fitting

    def fit(self, x, y):
        """
        Compute per-feature correlation with y and remember which features
        surpass the threshold.

        Parameters
        ----------
        x : pd.DataFrame
            Cleaned and encoded feature matrix.
        y : array-like of shape (n_samples,)
            Binary target (0 = no churn, 1 = churn).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If no feature survives the threshold (prevents silent downstream
            errors from an empty feature matrix).
        """
        # Reset index to guarantee row-wise alignment between X and y
        X = x.copy().reset_index(drop=True)
        Y = y.copy().reset_index(drop=True)

        # Convert y to a plain Series so corrwith works correctly
        y_ser = pd.Series(np.asarray(Y).ravel())

        # Compute Pearson correlation of each feature column with the target
        corrs = X.corrwith(y_ser)

        # Keep only features whose absolute correlation meets the threshold
        self.selected_features = corrs[corrs.abs() >= self.threshold].index

        # Guard: if all features are dropped the pipeline would fail silently
        if len(self.selected_features) == 0:
            max_corr = corrs.abs().max()
            raise ValueError(
                f"CorrelationThresholdFilter Error: All features were dropped!\n"
                f"   - Your threshold: {self.threshold}\n"
                f"   - Max correlation in your data: {max_corr:.4f}\n"
                f"   Fix: Lower your threshold below {max_corr:.4f}."
            )
        return self

    def transform(self, x):
        """
        Retain only the columns selected during ``fit()``.

        Parameters
        ----------
        x : pd.DataFrame
            Feature matrix (cleaned and encoded).

        Returns
        -------
        pd.DataFrame
            Subset of x containing only ``selected_features`` columns.
        """
        # Slice the dataframe to keep only correlated features
        return x[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        """
        Return names of the features that survived the filter (sklearn API).

        Returns
        -------
        pd.Index
            Feature names selected during fit.
        """
        return self.selected_features