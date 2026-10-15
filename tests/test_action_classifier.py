import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.action_classifier import (
    ACTION_LABELS,
    FEATURE_DIM,
    ActionClassifier,
    HeuristicClassifier,
)


@pytest.fixture
def classifier():
    return ActionClassifier(model_path=None)


def test_update_none_returns_idle(classifier):
    label, conf, _ = classifier.update(None)
    assert label == "idle"
    assert conf == 1.0


def test_update_returns_valid_label(classifier):
    fv = np.random.rand(FEATURE_DIM).astype(np.float32)
    label, conf, _ = classifier.update(fv)
    assert label in ACTION_LABELS
    assert 0.0 <= conf <= 1.0


def _open_hand() -> np.ndarray:
    """Wrist-relative landmarks for an open hand: all fingers extended, no pinch."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[3] = (-0.8, -0.5, 0.0)  # thumb IP
    lm[4] = (-1.2, -0.5, 0.0)  # thumb tip
    lm[9] = (-0.15, -0.6, 0.0)  # middle MCP
    for i, (tip_i, pip_i) in enumerate([(8, 6), (12, 10), (16, 14), (20, 18)]):
        x = (i - 1.5) * 0.3
        lm[pip_i] = (x, -1.0, 0.0)
        lm[tip_i] = (x, -1.6, 0.0)
    return lm


def test_heuristic_idle_detection():
    h = HeuristicClassifier()
    label, conf, _ = h.predict(_open_hand(), velocity=0.01, spread=0.2, finger_curl=0.2)
    assert label == "idle"
    assert conf > 0.5


def test_heuristic_reach_detection():
    h = HeuristicClassifier()
    label, _, _ = h.predict(_open_hand(), velocity=0.05, spread=0.3, finger_curl=0.15)
    assert label == "reach"


def test_heuristic_pick_detection():
    h = HeuristicClassifier()
    lm = _open_hand()
    lm[4] = lm[8]  # thumb tip touching index tip
    label, _, _ = h.predict(lm, velocity=0.01, spread=0.2, finger_curl=0.2)
    assert label == "pick"


def test_buffer_fills_without_crash(classifier):
    fv = np.random.rand(FEATURE_DIM).astype(np.float32)
    for _ in range(25):
        label, _, _ = classifier.update(fv)
    assert label in ACTION_LABELS


def test_smoothing_consistent_action(classifier):
    """Repeated same action should return stable smoothed label."""
    fv = np.zeros(FEATURE_DIM, dtype=np.float32)  # near-zero → idle-like
    labels = set()
    for _ in range(10):
        label, _, _ = classifier.update(fv)
        labels.add(label)
    # With temporal smoothing, should not flip wildly
    assert len(labels) <= 2
