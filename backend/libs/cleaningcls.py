"""
cleaningcls.py
==============
Provides the `clean_cls` sklearn-compatible transformer that performs
all raw data cleaning steps needed before the churn prediction pipeline
can train or infer. Steps include:

  - Dropping the non-predictive `customerID` column
  - Coercing `TotalCharges` from string to numeric (some rows are blank strings)
  - Binary-encoding all Yes/No columns (1/0)
  - Binary-encoding the `gender` column (Male=1, Female=0)
  - One-hot encoding remaining multi-category categorical columns
  - Re-indexing the transformed frame to guarantee the same column order
    seen during `fit()` (avoids silent feature-mismatch bugs at inference)

Usage inside a sklearn Pipeline:
    Pipeline([('clean', clean_cls()), ...])
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class clean_cls(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible data cleaner for the Telco Customer Churn dataset.

    Attributes
    ----------
    dummy_columns : list or None
        Categorical columns (>2 unique object values) detected during fit
        that will receive one-hot encoding.
    feature_columns : list or None
        Ordered list of all output column names produced after a full clean
        on the training set. Used in `transform` to guarantee column alignment.
    """

    def __init__(self):
        # Will be populated during fit(); stores column names to one-hot encode
        self.dummy_columns = None
        # Will be populated during fit(); stores final column order after encoding
        self.feature_columns = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean(self, X):
        """
        Apply all cleaning rules to dataframe X.

        Parameters
        ----------
        X : pd.DataFrame
            Raw (or partially processed) customer data.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe, without customerID and with numeric/binary
            columns encoded. One-hot encoding is applied only for the
            columns stored in ``self.dummy_columns`` (set during fit).
        """
        X = X.copy()  # Never mutate the caller's dataframe

        # --- Drop non-predictive identifier ---
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'], errors='ignore')

        # --- Fix TotalCharges: blank strings become NaN, then fill with 0 as float ---
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0).astype(float)

        # --- Binary-encode known binary Yes/No columns to 1/0 ---
        bin_cols = getattr(self, 'binary_columns', None)
        if bin_cols is None:
            if hasattr(self, 'X') and self.X is not None:
                bin_cols = [
                    c for c in self.X.columns
                    if set(self.X[c].dropna().unique()).issubset({'Yes', 'No'})
                ]
            else:
                bin_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

        for col in bin_cols:
            if col in X.columns:
                X[col] = X[col].map({'Yes': 1, 'No': 0})

        # --- Binary-encode gender (Male=1, Female=0) ---
        if 'gender' in X.columns and X['gender'].dtype == 'object':
            X['gender'] = X['gender'].map({'Male': 1, 'Female': 0})

        # --- One-hot encode multi-category columns (only those from fit) ---
        if self.dummy_columns:
            cats = getattr(self, 'categories_', None)
            if cats is None:
                if hasattr(self, 'X') and self.X is not None:
                    cats = {
                        col: sorted(self.X[col].dropna().unique().tolist())
                        for col in self.dummy_columns
                        if col in self.X.columns
                    }
                else:
                    cats = {
                        'MultipleLines': ['No', 'No phone service', 'Yes'],
                        'InternetService': ['DSL', 'Fiber optic', 'No'],
                        'OnlineSecurity': ['No', 'No internet service', 'Yes'],
                        'OnlineBackup': ['No', 'No internet service', 'Yes'],
                        'DeviceProtection': ['No', 'No internet service', 'Yes'],
                        'TechSupport': ['No', 'No internet service', 'Yes'],
                        'StreamingTV': ['No', 'No internet service', 'Yes'],
                        'StreamingMovies': ['No', 'No internet service', 'Yes'],
                        'Contract': ['Month-to-month', 'One year', 'Two year'],
                        'PaymentMethod': [
                            'Bank transfer (automatic)',
                            'Credit card (automatic)',
                            'Electronic check',
                            'Mailed check'
                        ]
                    }
            for col in self.dummy_columns:
                if col in X.columns and col in cats:
                    X[col] = pd.Categorical(X[col], categories=cats[col])

            X = pd.get_dummies(X, columns=self.dummy_columns, drop_first=True, dtype=int)

        return X

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Learn which categorical columns need one-hot encoding and record
        the final column order produced on the training set.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix (before any cleaning).
        y : ignored

        Returns
        -------
        self
        """
        self.X = X.copy()

        # Strip customerID before inspecting dtypes
        if 'customerID' in self.X.columns:
            self.X = self.X.drop(columns=['customerID'], errors='ignore')

        # Coerce TotalCharges so it does not appear as an object column
        if 'TotalCharges' in self.X.columns:
            self.X['TotalCharges'] = pd.to_numeric(
                self.X['TotalCharges'], errors='coerce'
            ).fillna(0).astype(float)

        # Identify binary columns that map Yes/No -> 1/0
        self.binary_columns = [
            col for col in self.X.columns
            if set(self.X[col].dropna().unique()).issubset({'Yes', 'No'})
        ]

        # Identify columns that need one-hot encoding:
        # object dtype with more than 2 unique values (binary cols handled separately)
        self.dummy_columns = []
        self.categories_ = {}
        for col in self.X.columns:
            if self.X[col].nunique() > 2 and self.X[col].dtype == 'object':
                self.dummy_columns.append(col)
                self.categories_[col] = sorted(self.X[col].dropna().unique().tolist())

        # Apply the full cleaning pipeline once to capture the column layout
        cleaned_X = self._clean(self.X)
        self.feature_columns = cleaned_X.columns.tolist()
        return self

    def transform(self, X):
        """
        Apply the learned cleaning rules to X and align columns.

        Any new dummy columns not seen during fit are dropped; any missing
        ones are filled with 0 so downstream steps always receive the same
        feature matrix shape.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to clean (can be train or test/inference data).

        Returns
        -------
        pd.DataFrame
            Cleaned and column-aligned dataframe.
        """
        cleaned_X = self._clean(X)

        # Guarantee column alignment with training schema
        if self.feature_columns is not None:
            cleaned_X = cleaned_X.reindex(columns=self.feature_columns, fill_value=0)
        return cleaned_X

    def get_feature_names_out(self, input_features=None):
        """
        Return the list of output feature names (sklearn convention).

        Returns
        -------
        list
            Column names after cleaning and encoding.
        """
        return self.feature_columns
