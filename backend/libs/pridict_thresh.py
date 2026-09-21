"""
pridict_thresh.py  –  Custom Threshold Classifier Wrapper
==========================================================
Provides `pridict_thresh`, a thin sklearn-compatible wrapper around any
probabilistic classifier (e.g. RandomForestClassifier, XGBClassifier) that
applies a **custom decision threshold** instead of the default 0.5.

Why use a custom threshold for churn prediction?
  - The cost of missing a churner (False Negative) is typically much higher
    than the cost of a false alarm (False Positive).
  - By raising the threshold (e.g. to 0.64) we require higher confidence
    before predicting "Churn", which improves **Precision** at the expense
    of **Recall**.  Tune ``thresh`` on a validation set using F1, Precision-
    Recall curves, or business-cost analysis.

Typical usage inside a sklearn Pipeline:
    Pipeline([
        ('clean', clean_cls()),
        ('ctf',   CorrelationThresholdFilter(threshold=0.14)),
        ('pridict_thresh', pridict_thresh(
            RandomForestClassifier(n_estimators=200, ...),
            thresh=0.64
        )),
    ])
"""

from sklearn.base import BaseEstimator, ClassifierMixin


class pridict_thresh(BaseEstimator, ClassifierMixin):
    """
    Wrapper that replaces a classifier's default 0.5 decision boundary
    with a user-supplied threshold applied to the positive-class probability.

    Parameters
    ----------
    model : sklearn-compatible classifier
        Any estimator that implements ``fit``, ``predict_proba``,
        and optionally ``predict``.
    thresh : float, default=0.5
        Decision threshold in [0, 1].  A sample is predicted as positive
        (churn=1) only when P(churn) > ``thresh``.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels seen during fit (forwarded from the wrapped model).
    is_fitted_ : bool
        Set to True after fit() completes; used by sklearn's
        ``check_is_fitted`` utility.
    """

    def __init__(self, model, thresh=0.5):
        self.thresh = thresh          # Custom probability cutoff
        self.model = model            # Underlying sklearn-compatible classifier
        self._estimator_type = "classifier"  # Required for sklearn Pipeline compatibility

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y, **kwargs):
        """
        Train the wrapped model on (X, y).

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Training feature matrix (already cleaned and encoded).
        y : array-like of shape (n_samples,)
            Binary target labels (0 = no churn, 1 = churn).
        **kwargs : dict
            Extra keyword arguments forwarded to the underlying model's fit().

        Returns
        -------
        self
        """
        # Validate decision threshold
        if not (0.0 <= self.thresh <= 1.0):
            raise ValueError(f"thresh must be in [0.0, 1.0], got {self.thresh}")

        # Delegate training to the wrapped classifier
        self.model.fit(X, y, **kwargs)

        # Expose classes_ so sklearn utilities (e.g. cross_val_score) work correctly
        self.classes_ = self.model.classes_

        # Mark the estimator as fitted for check_is_fitted compatibility
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        """
        sklearn hook used by ``check_is_fitted``.
        Returns True only after ``fit()`` has been called.
        """
        return getattr(self, 'is_fitted_', False)

    def predict_proba(self, X):
        """
        Return class probabilities from the underlying model.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Columns: [P(no churn), P(churn)].
        """
        # Delegate directly to the inner model — threshold does NOT affect probabilities
        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predict binary class labels using the custom threshold.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Binary predictions: 1 if P(churn) > ``self.thresh`` else 0.
        """
        # Validate decision threshold
        if not (0.0 <= self.thresh <= 1.0):
            raise ValueError(f"thresh must be in [0.0, 1.0], got {self.thresh}")

        # Get churn probability from the positive-class column (index 1)
        y_prob = self.model.predict_proba(X)[:, 1]

        # Apply custom threshold instead of sklearn's default 0.5
        y_pred = (y_prob > self.thresh).astype(int)
        return y_pred