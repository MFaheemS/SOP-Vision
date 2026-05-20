import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hand_detector import HandDetector


@pytest.fixture(scope="module")
def detector():
    d = HandDetector()
    yield d
    d.close()


def test_detect_returns_expected_keys(detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert "landmarks" in result
    assert "hand_detected" in result
    assert "frame" in result
    assert isinstance(result["landmarks"], list)


def test_no_hand_on_blank_frame(detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result["hand_detected"] is False
    assert len(result["landmarks"]) == 0


def test_feature_vector_none_without_landmarks(detector):
    fv = detector.get_feature_vector([])
    assert fv is None


def test_feature_vector_shape_with_fake_landmarks():
    fake = [np.random.rand(21, 3).astype(np.float32)]
    detector = HandDetector()
    fv = detector.get_feature_vector(fake)
    assert fv is not None
    assert fv.shape == (63,)
    detector.close()


def test_feature_vector_wrist_relative():
    wrist = np.array([0.5, 0.5, 0.0])
    lm = np.random.rand(21, 3).astype(np.float32)
    lm[0] = wrist
    detector = HandDetector()
    fv = detector.get_feature_vector([lm])
    reshaped = fv.reshape(21, 3)
    # After wrist-relative normalization, wrist should be near zero
    assert np.allclose(reshaped[0], 0.0, atol=1e-5)
    detector.close()
