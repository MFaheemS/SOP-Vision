import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sop_validator import SOPValidator

CONFIG = str(Path(__file__).parent.parent / "configs" / "sop_config.yaml")
DWELL = 3  # short dwell for fast tests


@pytest.fixture
def validator():
    return SOPValidator(CONFIG, min_dwell_frames=DWELL)


def _push(validator, action, n=None):
    """Push an action n (or dwell+1) times."""
    n = n or DWELL + 1
    state = {}
    for _ in range(n):
        state = validator.update(action, confidence=0.9)
    return state


def test_initial_state(validator):
    state = validator._state()
    assert state["current_step_idx"] == 0
    assert state["completed"] == []
    assert not state["procedure_done"]


def test_compliant_full_sequence(validator):
    for step in ["reach", "pick", "inspect", "place", "verify"]:
        _push(validator, step)
    state = validator._state()
    assert state["procedure_done"]
    assert state["skipped"] == []
    assert state["out_of_order"] == []
    report = validator.get_report()
    assert report.is_compliant
    assert report.completion_pct == 100.0


def test_step_skipped(validator):
    _push(validator, "reach")
    _push(validator, "inspect")  # skipped pick
    state = validator._state()
    assert "pick" in state["skipped"]
    assert "inspect" in state["out_of_order"]


def test_idle_does_not_advance(validator):
    _push(validator, "idle", n=20)
    state = validator._state()
    assert state["current_step_idx"] == 0


def test_low_confidence_ignored(validator):
    state = None
    for _ in range(10):
        state = validator.update("reach", confidence=0.3)
    assert state["current_step_idx"] == 0


def test_reset_clears_state(validator):
    _push(validator, "reach")
    validator.reset()
    state = validator._state()
    assert state["current_step_idx"] == 0
    assert state["completed"] == []


def test_dwell_progress_updates(validator):
    validator.update("reach", 0.9)
    state = validator.update("reach", 0.9)
    assert state["dwell_progress"] > 0
    assert state["dwell_action"] == "reach"
