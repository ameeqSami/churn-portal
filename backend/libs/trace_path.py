"""
trace_path.py
=============
Utilities for **tracing and explaining** the decision path a customer takes
through a trained decision tree (or one tree inside a RandomForest).

Public API
----------
trace_customer_path(pipeline_dt, X_filtered, customer_idx=0, tree_index=None)
    Prints a human-readable, step-by-step explanation of every split node
    a customer passes through, ending at the leaf with its class distribution.

get_customer_path_df(pipeline_dt, X_filtered, customer_idx=0, tree_index=None)
    Returns the same decision path as a structured ``pd.DataFrame`` for
    programmatic use or further analysis.

Internal helpers (not part of public API)
-----------------------------------------
_extract_predictor_and_model   – resolves the classifier from a pipeline
_get_feature_names             – retrieves column names from various sources
_select_tree_estimator         – for RandomForests, picks the individual tree
                                  whose prediction matches the ensemble vote

Supported model types
---------------------
- sklearn ``DecisionTreeClassifier``
- sklearn ``RandomForestClassifier``  (or any ensemble with ``estimators_``)
- Any wrapper around the above that stores the model in a ``.model`` attribute
  (e.g. ``pridict_thresh``)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# =========================================================================
# Private helpers
# =========================================================================

def _extract_predictor_and_model(pipeline_dt):
    """
    Resolve the *predictor step* and the *underlying classifier* from either
    a fitted sklearn Pipeline or a bare estimator.

    For pipelines the function checks, in order:
      1. A step named ``'pridict_thresh'`` (custom threshold wrapper)
      2. A step named ``'model'``
      3. The last step in the pipeline as a fallback

    For bare estimators the object itself is both predictor and model
    (unless it has a ``.model`` attribute, as with ``pridict_thresh``).

    Parameters
    ----------
    pipeline_dt : sklearn Pipeline or estimator
        Fitted pipeline or standalone classifier.

    Returns
    -------
    predictor : estimator
        The step used for predict / predict_proba calls.
    model : sklearn estimator
        The underlying decision tree or forest.
    """
    if hasattr(pipeline_dt, 'named_steps'):
        # It's a Pipeline — locate the final classifier step
        if 'pridict_thresh' in pipeline_dt.named_steps:
            predictor = pipeline_dt.named_steps['pridict_thresh']
            model = getattr(predictor, 'model', predictor)
        elif 'model' in pipeline_dt.named_steps:
            predictor = pipeline_dt.named_steps['model']
            model = getattr(predictor, 'model', predictor)
        else:
            # Fall back to the very last step
            predictor = pipeline_dt.steps[-1][1]
            model = getattr(predictor, 'model', predictor)
    else:
        # Bare estimator (no Pipeline wrapper)
        predictor = pipeline_dt
        model = getattr(predictor, 'model', predictor)

    return predictor, model


def _get_feature_names(pipeline_dt, X_filtered):
    """
    Retrieve feature names from the most reliable available source.

    Priority:
      1. ``X_filtered.columns`` – if X_filtered is a DataFrame (best source)
      2. Pipeline's ``ctf`` step ``get_feature_names_out()``
      3. Pipeline's ``feature_names_in_`` attribute
      4. Generic placeholder names ``feature_0, feature_1, ...``

    Parameters
    ----------
    pipeline_dt : sklearn Pipeline or estimator
        The fitted pipeline.
    X_filtered : pd.DataFrame or ndarray
        The feature matrix that was passed to the tree's decision_path.

    Returns
    -------
    list of str
        Feature name strings aligned with columns of X_filtered.
    """
    if isinstance(X_filtered, pd.DataFrame):
        # Most reliable: the DataFrame already carries correct column names
        return list(X_filtered.columns)

    if hasattr(pipeline_dt, 'named_steps'):
        # Try to read names from the correlation filter step
        if 'ctf' in pipeline_dt.named_steps and hasattr(
            pipeline_dt['ctf'], 'get_feature_names_out'
        ):
            return list(pipeline_dt['ctf'].get_feature_names_out())

    if hasattr(pipeline_dt, 'feature_names_in_'):
        return list(pipeline_dt.feature_names_in_)

    # Last resort: generate generic names
    return [f"feature_{i}" for i in range(X_filtered.shape[1])]


def _select_tree_estimator(predictor, model, sample_customer, tree_index=None):
    """
    For a RandomForest, find the individual tree whose prediction most
    closely agrees with the ensemble's final (post-voting / post-threshold)
    prediction for ``sample_customer``.

    If the model is a single DecisionTree the function returns it directly.

    Algorithm
    ---------
    1. Compute the ensemble prediction (using ``predictor.predict`` so that
       any custom threshold in ``pridict_thresh`` is honoured).
    2. Collect each tree's leaf-level prediction and churn probability.
    3. Among trees that agree with the ensemble vote, pick the one whose
       churn probability is closest to the ensemble's churn probability.
       This gives the most "representative" explanation.
    4. If ``tree_index`` is explicitly provided, that tree is used directly.

    Parameters
    ----------
    predictor : estimator
        The final pipeline step (may be ``pridict_thresh`` or a plain model).
    model : sklearn estimator
        Underlying RandomForest or DecisionTree extracted from ``predictor``.
    sample_customer : pd.DataFrame or pd.Series or ndarray
        A single customer row.
    tree_index : int or None
        If given, forces selection of this specific tree (0-indexed).

    Returns
    -------
    dict with keys:
        tree_model      – The chosen DecisionTree estimator
        tree_index      – Index within ``model.estimators_`` (None for DT)
        is_rf           – True if the model is a RandomForest
        ensemble_pred   – Ensemble's binary prediction (0 or 1)
        ensemble_prob   – Ensemble's churn probability (float or None)
        tree_pred       – Chosen tree's binary prediction
        matching_count  – Number of trees that agreed with ensemble vote
        total_trees     – Total number of trees in the forest
    """
    # --- Normalise input to both DataFrame and numpy array forms ---
    if isinstance(sample_customer, pd.DataFrame):
        sample_df = sample_customer
        sample_arr = sample_customer.values
    elif isinstance(sample_customer, pd.Series):
        sample_df = sample_customer.to_frame().T
        sample_arr = sample_df.values
    else:
        sample_arr = np.atleast_2d(sample_customer)
        sample_df = pd.DataFrame(sample_arr)

    # --- Step 1: Get the ensemble's final prediction (respects custom threshold) ---
    if hasattr(predictor, 'predict'):
        ensemble_pred = int(predictor.predict(sample_df)[0])
    else:
        ensemble_pred = int(model.predict(sample_arr)[0])

    # Also capture the ensemble's churn probability for comparison
    ensemble_prob = None
    if hasattr(predictor, 'predict_proba'):
        prob_arr = predictor.predict_proba(sample_df)[0]
        ensemble_prob = float(prob_arr[1]) if len(prob_arr) > 1 else float(prob_arr[0])
    elif hasattr(model, 'predict_proba'):
        prob_arr = model.predict_proba(sample_arr)[0]
        ensemble_prob = float(prob_arr[1]) if len(prob_arr) > 1 else float(prob_arr[0])

    # --- Step 2: Short-circuit if it's not a RandomForest ---
    if not isinstance(model, RandomForestClassifier) and not hasattr(model, 'estimators_'):
        return {
            'tree_model': model,
            'tree_index': None,
            'is_rf': False,
            'ensemble_pred': ensemble_pred,
            'ensemble_prob': ensemble_prob,
            'tree_pred': ensemble_pred,
            'matching_count': 1,
            'total_trees': 1,
        }

    # --- Step 3: Evaluate every tree in the forest for this customer ---
    estimators = model.estimators_
    total_trees = len(estimators)
    tree_preds = []
    tree_probs = []

    for est in estimators:
        # Leaf node reached by this customer
        leaf_idx = est.apply(sample_arr)[0]
        counts = est.tree_.value[leaf_idx][0]      # Sample counts per class at leaf
        total_count = counts.sum()
        # Churn probability = proportion of churn samples at the leaf
        p1 = counts[1] / total_count if (len(counts) > 1 and total_count > 0) else 0.0
        t_pred = int(np.argmax(counts))            # Majority-class prediction
        tree_preds.append(t_pred)
        tree_probs.append(p1)

    # --- Step 4: Filter to trees agreeing with the ensemble vote ---
    matching_indices = [idx for idx, p in enumerate(tree_preds) if p == ensemble_pred]
    matching_count = len(matching_indices)

    if tree_index is not None:
        # User forced a specific tree
        chosen_index = tree_index
    else:
        if matching_indices:
            # Among matching trees, pick the one closest in probability to the ensemble
            if ensemble_prob is not None:
                chosen_index = min(
                    matching_indices,
                    key=lambda idx: abs(tree_probs[idx] - ensemble_prob)
                )
            else:
                chosen_index = matching_indices[0]
        else:
            # No tree agreed — fall back to tree #0 (should be rare)
            chosen_index = 0

    chosen_tree = estimators[chosen_index]
    chosen_tree_pred = tree_preds[chosen_index]

    return {
        'tree_model': chosen_tree,
        'tree_index': chosen_index,
        'is_rf': True,
        'ensemble_pred': ensemble_pred,
        'ensemble_prob': ensemble_prob,
        'tree_pred': chosen_tree_pred,
        'matching_count': matching_count,
        'total_trees': total_trees,
    }


# =========================================================================
# Public API
# =========================================================================

def trace_customer_path(pipeline_dt, X_filtered, customer_idx=0, tree_index=None):
    """
    Print a human-readable trace of the decision path taken by a customer
    through a decision tree (or one tree in a RandomForest).

    For each **split node** on the path the function prints:
      - The feature that was tested
      - The tree's threshold rule
      - The customer's actual value and which branch they took

    For the final **leaf node** the function prints:
      - The number of training samples that reached this leaf
      - The class distribution (% No Churn vs % Churn)
      - The final prediction

    For RandomForest models the function first selects a representative
    tree (one that agrees with the ensemble vote) and shows a header with
    the ensemble statistics.

    Parameters
    ----------
    pipeline_dt : sklearn Pipeline or fitted estimator
        The complete fitted churn pipeline.
    X_filtered : pd.DataFrame or ndarray
        Feature matrix **after** all preprocessing steps (clean + ctf).
        Shape: (n_customers, n_features).
    customer_idx : int, default=0
        Row index of the customer to trace within X_filtered.
    tree_index : int or None, default=None
        Force a specific tree (0-indexed) to be used for the trace.
        If None, the most representative tree is selected automatically.

    Returns
    -------
    None
        Output is printed to stdout.
    """
    # Resolve the predictor step and underlying model from the pipeline
    predictor, model = _extract_predictor_and_model(pipeline_dt)

    # Prepare the single customer row and the full numpy array
    if isinstance(X_filtered, pd.DataFrame):
        sample_customer = X_filtered.iloc[[customer_idx]]
        X_arr = X_filtered.values
    else:
        X_arr = np.atleast_2d(X_filtered)
        sample_customer = X_arr[customer_idx:customer_idx + 1]

    # Select the tree to trace
    info = _select_tree_estimator(predictor, model, sample_customer, tree_index=tree_index)
    tree_model = info['tree_model']

    # --- Print header showing ensemble / model info ---
    if info['is_rf']:
        chosen_idx = info['tree_index']
        total = info['total_trees']
        matches = info['matching_count']
        ens_pred = info['ensemble_pred']
        tree_pred = info['tree_pred']
        ens_prob_str = (
            f" ({info['ensemble_prob']:.1%} churn prob)"
            if info['ensemble_prob'] is not None
            else ""
        )
        print(f"[INFO] RandomForest detected - Ensemble voted: {ens_pred}{ens_prob_str}")
        print(
            f"[INFO] {matches}/{total} trees agreed with ensemble vote. "
            f"Selected Tree #{chosen_idx} (tree pred: {tree_pred})"
        )
        if tree_pred != ens_pred:
            # Warn if the selected tree disagrees with the ensemble
            print(
                f"[WARNING] Tree #{chosen_idx} predicted {tree_pred}, "
                f"which differs from ensemble vote {ens_pred}!"
            )
        print("=" * 40)
    else:
        print(f"[INFO] Single DecisionTree detected - Prediction: {info['ensemble_pred']}")
        print("=" * 40)

    # --- Compute decision path through the selected tree ---
    node_indicator = tree_model.decision_path(X_arr)    # Sparse matrix of visited nodes
    leave_id = tree_model.apply(X_arr)                  # Leaf node ID for each sample
    # Extract just the node IDs for this specific customer
    node_index = node_indicator.indices[
        node_indicator.indptr[customer_idx]:node_indicator.indptr[customer_idx + 1]
    ]

    print(f"Customer's Decision Path (Node IDs): {node_index.tolist()}")
    print(f"Final Leaf Node: {leave_id[customer_idx]}")
    print("=" * 40)

    # --- Walk each node on the path and print its details ---
    tree = tree_model.tree_
    feature_names = _get_feature_names(pipeline_dt, X_filtered)
    sample_series = (
        X_filtered.iloc[customer_idx]
        if isinstance(X_filtered, pd.DataFrame)
        else pd.Series(X_arr[customer_idx], index=feature_names)
    )

    for node_id in node_index:
        # Split node: children differ → this node makes a binary decision
        if tree.children_left[node_id] != tree.children_right[node_id]:
            feat_idx = tree.feature[node_id]
            thresh = tree.threshold[node_id]
            feat_name = feature_names[feat_idx]
            customer_value = sample_series[feat_name]
            direction = "LEFT (<=)" if customer_value <= thresh else "RIGHT (>)"

            print(f"Node {node_id} (Split Node):")
            print(f"  - Feature checked: '{feat_name}'")
            print(f"  - Tree rule: <= {thresh:.4f}")
            print(f"  - Customer's value: {customer_value}  ->  Goes {direction}")
            print("-" * 40)
        else:
            # Leaf node: no further splitting, contains the final prediction
            class_dist = tree.value[node_id][0]
            total = class_dist.sum()
            p0 = class_dist[0] / total if total > 0 else 0.0
            p1 = class_dist[1] / total if (len(class_dist) > 1 and total > 0) else 0.0
            leaf_pred = int(np.argmax(class_dist))
            samples = int(tree.n_node_samples[node_id])

            print(f"Node {node_id} (Leaf - Final Destination):")
            print(f"  - Node samples: {samples}")
            print(f"  - Class distribution  ->  No Churn: {p0:.1%}, Churn: {p1:.1%}")
            print(f"  - Churn probability at this leaf: {p1:.1%}")
            print(f"  - Tree decision: {'Churn (1)' if leaf_pred == 1 else 'No Churn (0)'}")
            print("=" * 40)


def get_customer_path_df(pipeline_dt, X_filtered, customer_idx=0, tree_index=None):
    """
    Return the decision path of a customer as a structured ``pd.DataFrame``.

    Same logic as :func:`trace_customer_path` but returns data instead of
    printing it, making it easy to display in a notebook, export to CSV,
    or feed into a visualisation tool.

    Parameters
    ----------
    pipeline_dt : sklearn Pipeline or fitted estimator
        The complete fitted churn pipeline.
    X_filtered : pd.DataFrame or ndarray
        Feature matrix after all preprocessing steps.
    customer_idx : int, default=0
        Row index of the customer to trace.
    tree_index : int or None, default=None
        Force a specific tree; if None, the best-matching tree is chosen.

    Returns
    -------
    pd.DataFrame
        One row per node on the decision path.  Columns:
        ``Node ID``, ``Node Type``, ``Feature Checked``, ``Threshold Rule``,
        ``Customer Value``, ``Condition Met``, ``Direction``, ``Samples``,
        ``Churn Probability``, ``Class Distribution``, and (for leaf rows)
        ``Prediction``.
    """
    # Resolve model references from the pipeline
    predictor, model = _extract_predictor_and_model(pipeline_dt)

    # Prepare the sample and full array
    if isinstance(X_filtered, pd.DataFrame):
        sample_customer = X_filtered.iloc[[customer_idx]]
        X_arr = X_filtered.values
    else:
        X_arr = np.atleast_2d(X_filtered)
        sample_customer = X_arr[customer_idx:customer_idx + 1]

    # Select the representative tree
    info = _select_tree_estimator(predictor, model, sample_customer, tree_index=tree_index)
    tree_model = info['tree_model']

    # Compute decision path for all customers; extract indices for our target customer
    node_indicator = tree_model.decision_path(X_arr)
    leave_id = tree_model.apply(X_arr)
    node_index = node_indicator.indices[
        node_indicator.indptr[customer_idx]:node_indicator.indptr[customer_idx + 1]
    ]

    # Gather feature names and the customer's feature values as a Series
    tree = tree_model.tree_
    feature_names = _get_feature_names(pipeline_dt, X_filtered)
    sample_series = (
        X_filtered.iloc[customer_idx]
        if isinstance(X_filtered, pd.DataFrame)
        else pd.Series(X_arr[customer_idx], index=feature_names)
    )

    path_data = []  # Accumulate one dict per node

    for node_id in node_index:
        # Common columns for all node types
        class_dist = tree.value[node_id][0]
        total = class_dist.sum()
        p0 = round(class_dist[0] / total, 4) if total > 0 else 0.0
        p1 = round(class_dist[1] / total, 4) if (len(class_dist) > 1 and total > 0) else 0.0
        samples = int(tree.n_node_samples[node_id])

        if tree.children_left[node_id] != tree.children_right[node_id]:
            # --- Split node ---
            feat_idx = tree.feature[node_id]
            thresh = tree.threshold[node_id]
            feat_name = feature_names[feat_idx]
            customer_value = sample_series[feat_name]
            condition_met = customer_value <= thresh   # True → goes left branch

            path_data.append({
                "Node ID": node_id,
                "Node Type": "Split Node",
                "Feature Checked": feat_name,
                "Threshold Rule": f"<= {thresh:.4f}",
                "Customer Value": customer_value,
                "Condition Met": condition_met,       # True = left branch taken
                "Direction": "Left (<=)" if condition_met else "Right (>)",
                "Samples": samples,
                "Churn Probability": p1,
                "Class Distribution": f"No Churn: {p0:.1%}, Churn: {p1:.1%}"
            })
        else:
            # --- Leaf node (final destination) ---
            leaf_pred = int(np.argmax(class_dist))    # 0=No Churn, 1=Churn
            path_data.append({
                "Node ID": node_id,
                "Node Type": "Leaf Node",
                "Feature Checked": "N/A (Final Destination)",
                "Threshold Rule": "N/A",
                "Customer Value": "N/A",
                "Condition Met": "N/A",
                "Direction": "N/A",
                "Samples": samples,
                "Churn Probability": p1,
                "Class Distribution": f"No Churn: {p0:.1%}, Churn: {p1:.1%}",
                "Prediction": leaf_pred               # Final binary class prediction
            })

    return pd.DataFrame(path_data)