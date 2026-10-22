"""Tests for supervised ML recommendation pipeline."""

import json
import os

from src.data.dataset_builder import build_compiled_dataset
from src.data.ml_recommendations import build_ml_recommendations
from src.models.supervised_recommender import train_action_model, save_action_model, predict_action
from src.data.paths import COMPILED_DATASET_PATH


def test_ml_train_predict_and_recommendations():
    cases = build_compiled_dataset(write=False)
    assert len(cases) >= 5

    bundle = train_action_model(cases)
    assert bundle["meta"]["train_accuracy"] > 0.5

    path = save_action_model(bundle)
    assert os.path.isfile(path)

    action, conf, probs = predict_action(cases[0], bundle=bundle)
    assert action in probs
    assert 0 < conf <= 1

    rows = build_ml_recommendations(cases, limit=10)
    assert len(rows) <= 10
    assert rows[0]["id"] == "DBG-REC-001"
    assert "recommendation" in rows[0]
    assert rows[0].get("mlConfidence") is not None or rows[0].get("confidence") is not None
