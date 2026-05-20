from dataclasses import dataclass, field
from collections import deque
import yaml
from pathlib import Path


@dataclass
class StepResult:
    step_name: str
    completed: bool
    out_of_order: bool = False
    skipped: bool = False


@dataclass
class ComplianceReport:
    procedure_name: str
    total_steps: int
    completed_steps: int
    missed_steps: list[str]
    out_of_order_steps: list[str]
    is_compliant: bool
    completion_pct: float


class SOPValidator:
    """
    Tracks action recognition output against the expected SOP step sequence.

    The validator uses a dwell-based confirmation: an action must be sustained
    for `min_dwell_frames` consecutive frames before it is accepted as a step.
    This prevents noise from mis-triggering step completions.
    """

    def __init__(self, config_path: str, min_dwell_frames: int = 12):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.procedure_name: str = cfg["procedure_name"]
        self.expected_sequence: list[str] = cfg["expected_sequence"]
        self.min_dwell_frames = min_dwell_frames

        self.current_step_idx: int = 0
        self.completed_steps: list[str] = []
        self.out_of_order: list[str] = []
        self.skipped_steps: list[str] = []

        self._dwell_action: str | None = None
        self._dwell_count: int = 0

        # Sliding history for UI display
        self.event_log: deque = deque(maxlen=8)

    def update(self, action: str, confidence: float) -> dict:
        """
        Feed current predicted action. Returns current state dict for UI.
        """
        if confidence < 0.5 or action == "idle":
            self._dwell_action = None
            self._dwell_count = 0
            return self._state()

        # Dwell confirmation
        if action == self._dwell_action:
            self._dwell_count += 1
        else:
            self._dwell_action = action
            self._dwell_count = 1

        if self._dwell_count < self.min_dwell_frames:
            return self._state()

        # Action confirmed — check against SOP
        if self.current_step_idx >= len(self.expected_sequence):
            return self._state()  # All steps done

        expected = self.expected_sequence[self.current_step_idx]

        if action == expected:
            self.completed_steps.append(action)
            self.event_log.append({"step": action, "status": "completed"})
            self.current_step_idx += 1
            self._dwell_count = 0
        elif action in self.expected_sequence:
            future_idx = self.expected_sequence.index(action)
            if future_idx > self.current_step_idx:
                # Jumped ahead — mark skipped steps
                for skipped in self.expected_sequence[self.current_step_idx:future_idx]:
                    self.skipped_steps.append(skipped)
                    self.event_log.append({"step": skipped, "status": "skipped"})
                self.out_of_order.append(action)
                self.completed_steps.append(action)
                self.event_log.append({"step": action, "status": "out_of_order"})
                self.current_step_idx = future_idx + 1
                self._dwell_count = 0

        return self._state()

    def _state(self) -> dict:
        done = self.current_step_idx >= len(self.expected_sequence)
        next_step = (
            self.expected_sequence[self.current_step_idx]
            if not done else None
        )
        return {
            "procedure": self.procedure_name,
            "current_step_idx": self.current_step_idx,
            "total_steps": len(self.expected_sequence),
            "next_expected": next_step,
            "completed": self.completed_steps.copy(),
            "out_of_order": self.out_of_order.copy(),
            "skipped": self.skipped_steps.copy(),
            "procedure_done": done,
            "event_log": list(self.event_log),
            "dwell_progress": min(self._dwell_count / self.min_dwell_frames, 1.0),
            "dwell_action": self._dwell_action,
        }

    def reset(self):
        self.current_step_idx = 0
        self.completed_steps.clear()
        self.out_of_order.clear()
        self.skipped_steps.clear()
        self._dwell_action = None
        self._dwell_count = 0
        self.event_log.clear()

    def get_report(self) -> ComplianceReport:
        total = len(self.expected_sequence)
        done = len(self.completed_steps)
        missed = [s for s in self.expected_sequence if s not in self.completed_steps]
        return ComplianceReport(
            procedure_name=self.procedure_name,
            total_steps=total,
            completed_steps=done,
            missed_steps=missed,
            out_of_order_steps=self.out_of_order.copy(),
            is_compliant=len(missed) == 0 and len(self.out_of_order) == 0,
            completion_pct=round(done / total * 100, 1) if total > 0 else 0.0,
        )
