"""Supervised gradient-boosting recommender for debug action selection."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from src.config import get_settings
from src.data.action_inference import ACTION_LABELS
from src.data.paths import COMPILED_DATASET_PATH
from src.models.ml_features import FEATURE_NAMES, cases_feature_matrix

# Internal action strings (dataset labels)
INTERNAL_ACTIONS: Tuple[str, ...] = tuple(ACTION_LABELS.values())

_ACTION_TO_IDX = {a: i for i, a in enumerate(INTERNAL_ACTIONS)}
_IDX_TO_ACTION = {i: a for a, i in _ACTION_TO_IDX.items()}

_MODEL_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None


def _dataset_fingerprint() -> str:
    parts = ["v1-ml-gb"]
    if os.path.exists(COMPILED_DATASET_PATH):
        parts.append(f"{COMPILED_DATASET_PATH}:{os.path.getmtime(COMPILED_DATASET_PATH)}")
    return "|".join(parts)


def _valid_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in cases:
        action = c.get("true_action")
        if action in _ACTION_TO_IDX:
            out.append(c)
    return out


def train_action_model(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Train multiclass model on die features → debug action."""
    labeled = _valid_cases(cases)
    if len(labeled) < 5:
        raise ValueError(f"Need at least 5 labeled cases to train ML recommender, got {len(labeled)}")

    X = cases_feature_matrix(labeled)
    y = np.array([_ACTION_TO_IDX[c["true_action"]] for c in labeled], dtype=np.int64)

    model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.08,
        max_iter=200,
        random_state=42,
    )
    model.fit(X, y)

    cv_folds = min(5, len(labeled))
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
    train_acc = float(accuracy_score(y, model.predict(X)))

    fingerprint = _dataset_fingerprint()
    meta = {
        "fingerprint": fingerprint,
        "trained_at": time.time(),
        "n_cases": len(labeled),
        "train_accuracy": round(train_acc, 4),
        "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
        "cv_accuracy_std": round(float(np.std(cv_scores)), 4),
        "feature_names": list(FEATURE_NAMES),
        "classes": list(INTERNAL_ACTIONS),
        "model_type": "HistGradientBoostingClassifier",
    }
    bundle = {"model": model, "meta": meta}
    return bundle


def save_action_model(bundle: Dict[str, Any], path: Optional[str] = None) -> str:
    settings = get_settings()
    out = path or settings.ml_model_path
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    joblib.dump(bundle, out)
    global _MODEL_CACHE
    _MODEL_CACHE = (bundle["meta"]["fingerprint"], bundle)
    return out


def load_action_model(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    global _MODEL_CACHE
    settings = get_settings()
    p = path or settings.ml_model_path
    fingerprint = _dataset_fingerprint()
    if _MODEL_CACHE is not None and _MODEL_CACHE[0] == fingerprint:
        return _MODEL_CACHE[1]

    if not os.path.isfile(p):
        return None
    try:
        bundle = joblib.load(p)
        meta = bundle.get("meta") or {}
        if meta.get("fingerprint") != fingerprint:
            return None
        _MODEL_CACHE = (fingerprint, bundle)
        return bundle
    except (OSError, ValueError, KeyError):
        return None


def predict_action(
    case: Dict[str, Any],
    bundle: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Dict[str, float]]:
    """Return (internal_action, confidence, per-class probabilities)."""
    bundle = bundle or load_action_model()
    if bundle is None:
        action = str(case.get("true_action") or INTERNAL_ACTIONS[3])
        return action, 0.5, {action: 0.5}

    model = bundle["model"]
    X = cases_feature_vector(case).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    action = _IDX_TO_ACTION.get(idx, INTERNAL_ACTIONS[3])
    conf = float(probs[idx])
    prob_map = {
        _IDX_TO_ACTION[i]: float(probs[i])
        for i in range(len(probs))
        if i in _IDX_TO_ACTION
    }
    return action, conf, prob_map


def cases_feature_vector(case: Dict[str, Any]) -> np.ndarray:
    from src.models.ml_features import case_feature_vector

    return case_feature_vector(case)


def warm_ml_action_model(cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Train or load supervised model; returns status dict."""
    settings = get_settings()
    if cases is None:
        if not os.path.exists(COMPILED_DATASET_PATH):
            return {"status": "no_dataset", "trained": False}
        with open(COMPILED_DATASET_PATH, encoding="utf-8") as f:
            cases = json.load(f)

    existing = load_action_model()
    if existing and not settings.ml_auto_train:
        return {
            "status": "loaded",
            "trained": False,
            "meta": existing.get("meta"),
            "path": settings.ml_model_path,
        }

    if existing is None or settings.ml_auto_train:
        try:
            bundle = train_action_model(cases or [])
            path = save_action_model(bundle)
            return {
                "status": "trained",
                "trained": True,
                "meta": bundle["meta"],
                "path": path,
            }
        except ValueError as e:
            return {"status": "skipped", "trained": False, "reason": str(e)}

    return {
        "status": "loaded",
        "trained": False,
        "meta": existing.get("meta"),
        "path": settings.ml_model_path,
    }


def model_status() -> Dict[str, Any]:
    bundle = load_action_model()
    settings = get_settings()
    if not bundle:
        return {
            "loaded": False,
            "path": settings.ml_model_path,
            "recommendation_source": settings.recommendation_source,
        }
    return {
        "loaded": True,
        "path": settings.ml_model_path,
        "recommendation_source": settings.recommendation_source,
        "meta": bundle.get("meta"),
    }
