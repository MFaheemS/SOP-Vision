import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.action_classifier import ActionClassifier, HeuristicClassifier, ACTION_LABELS, FEATURE_DIM


@pytest.fixture
def classifier():
    return ActionClassifier(model_path=None)


def test_update_none_returns_idle(classifier):
    label, conf = classifier.update(None)
    assert label == "idle"
    assert conf == 1.0


def test_update_returns_valid_label(classifier):
    fv = np.random.rand(FEATURE_DIM).astype(np.float32)
    label, conf = classifier.update(fv)
    assert label in ACTION_LABELS
    assert 0.0 <= conf <= 1.0


def test_heuristic_idle_detection():
    h = HeuristicClassifier()
    lm = np.zeros((21, 3), dtype=np.float32)
    label, conf = h.predict(lm, velocity=0.01, spread=0.2, finger_curl=0.2)
    assert label == "idle"
    assert conf > 0.5


def test_heuristic_reach_detection():
    h = HeuristicClassifier()
    lm = np.zeros((21, 3), dtype=np.float32)
    label, conf = h.predict(lm, velocity=0.05, spread=0.3, finger_curl=0.15)
    assert label == "reach"


def test_buffer_fills_without_crash(classifier):
    fv = np.random.rand(FEATURE_DIM).astype(np.float32)
    for _ in range(25):
        label, conf = classifier.update(fv)
    assert label in ACTION_LABELS


def test_smoothing_consistent_action(classifier):
    """Repeated same action should return stable smoothed label."""
    fv = np.zeros(FEATURE_DIM, dtype=np.float32)  # near-zero → idle-like
    labels = set()
    for _ in range(10):
        label, _ = classifier.update(fv)
        labels.add(label)
    # With temporal smoothing, should not flip wildly
    assert len(labels) <= 2
