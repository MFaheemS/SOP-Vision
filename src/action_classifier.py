import numpy as np
from collections import deque
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


ACTION_LABELS = ["idle", "reach", "pick", "inspect", "place", "verify"]
NUM_CLASSES = len(ACTION_LABELS)
FEATURE_DIM = 63   # 21 landmarks × 3 coords
WINDOW_SIZE = 20   # frames per prediction


if TORCH_AVAILABLE:
    class ActionLSTM(nn.Module):
        """Bidirectional LSTM that classifies hand actions from landmark sequences."""

        def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 128,
                     num_layers: int = 2, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, bidirectional=True, dropout=dropout
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            out = self.classifier(out[:, -1, :])
            return out


def _finger_states(lm: np.ndarray) -> list[bool]:
    """
    Returns [thumb, index, middle, ring, pinky] extended (True) or curled (False).
    lm is (21, 3) wrist-relative, scale-normalized.
    """
    # Thumb: compare tip(4) vs ip(3) along x-axis, flipped for left hand
    thumb_extended = lm[4][0] > lm[3][0] if lm[9][0] > 0 else lm[4][0] < lm[3][0]

    fingers_extended = []
    for tip_i, pip_i in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        tip_dist = np.linalg.norm(lm[tip_i])
        pip_dist = np.linalg.norm(lm[pip_i])
        fingers_extended.append(tip_dist > pip_dist * 1.1)

    return [thumb_extended] + fingers_extended


def _is_pinch(lm: np.ndarray, threshold: float = 0.65) -> bool:
    """True when thumb tip (4) and index tip (8) are close — pinch/pick gesture."""
    dist = float(np.linalg.norm(lm[4] - lm[8]))
    return dist < threshold


class HeuristicClassifier:
    """
    Rule-based fallback classifier using hand landmark geometry.
    Used when no trained model is available.

    Intended gesture mappings (physically distinct):
      idle    — hand still, any relaxed pose
      reach   — ALL 4+ fingers open and spread, hand moving toward target
      pick    — thumb-index PINCH (tips touching/close), other fingers curled
      inspect — PEACE SIGN: only index + middle extended, hand relatively still
      place   — FIST (all fingers curled), hand moving
      verify  — INDEX POINT only (gun shape): index extended, rest curled
    """

    def __init__(self):
        self.last_debug: dict = {}

    def predict(self, landmarks: np.ndarray, velocity: float, spread: float,
                finger_curl: float) -> tuple[str, float, dict]:
        states = _finger_states(landmarks)
        thumb, index, middle, ring, pinky = states
        n_extended = sum(states)
        pinch_dist = float(np.linalg.norm(landmarks[4] - landmarks[8]))
        pinch = _is_pinch(landmarks)

        dbg = {
            "pinch_dist": round(pinch_dist, 3),
            "pinch": pinch,
            "fingers": f"T{int(thumb)}I{int(index)}M{int(middle)}R{int(ring)}P{int(pinky)}",
            "n_ext": n_extended,
            "velocity": round(velocity, 4),
            "curl": round(finger_curl, 3),
        }
        self.last_debug = dbg

        # ── pick: pinch gesture — thumb-index close, dominant signal
        if pinch:
            return "pick", 0.90, dbg

        # ── verify: index-only point (gun shape), no pinch
        if index and not middle and not ring and not pinky and not pinch:
            return "verify", 0.87, dbg

        # ── inspect: peace sign — index + middle only ────────────────────────
        if index and middle and not ring and not pinky:
            return "inspect", 0.85, dbg

        # ── idle: hand still ─────────────────────────────────────────────────
        if velocity < 0.018:
            return "idle", 0.85, dbg

        # ── reach: open hand moving (4-5 fingers extended) ──────────────────
        if n_extended >= 4 and velocity >= 0.018:
            return "reach", 0.83, dbg

        # ── place: fist moving (0-1 fingers extended) ────────────────────────
        if n_extended <= 1 and velocity >= 0.018:
            return "place", 0.80, dbg

        # fallback
        if velocity >= 0.018:
            return "reach", 0.60, dbg
        return "idle", 0.60, dbg


class ActionClassifier:
    """
    Wraps LSTM model with sliding-window buffering and temporal smoothing.
    Falls back to HeuristicClassifier if no model weights found.
    """

    def __init__(self, model_path: str | None = None, window_size: int = WINDOW_SIZE,
                 device: str = "cpu"):
        self.window_size = window_size
        self.device = device
        self.buffer: deque = deque(maxlen=window_size)
        self.prev_landmarks: np.ndarray | None = None

        self.model: ActionLSTM | None = None
        self.heuristic = HeuristicClassifier()
        # (label, confidence) pairs for weighted smoothing
        self.smoothing_window: deque = deque(maxlen=9)

        if model_path and Path(model_path).exists() and TORCH_AVAILABLE:
            self.model = ActionLSTM().to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()

    def _compute_motion_features(self, lm: np.ndarray) -> tuple[float, float, float]:
        """Returns velocity, spread, and finger curl from landmark array."""
        velocity = 0.0
        if self.prev_landmarks is not None:
            velocity = float(np.mean(np.linalg.norm(lm - self.prev_landmarks, axis=1)))

        fingertips = lm[[4, 8, 12, 16, 20]]
        palm = lm[0]
        spread = float(np.mean(np.linalg.norm(fingertips - palm, axis=1)))

        # curl: compare tip-to-wrist vs pip-to-wrist (extended tip is farther than pip)
        finger_pairs = [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]
        curls = []
        for tip_i, pip_i in finger_pairs:
            tip_dist = np.linalg.norm(lm[tip_i] - lm[0])
            pip_dist = np.linalg.norm(lm[pip_i] - lm[0])
            if pip_dist > 1e-6:
                # 0 = fully extended, 1 = fully curled
                curl = 1.0 - min(tip_dist / (pip_dist * 1.35), 1.0)
                curls.append(max(0.0, curl))
        finger_curl = float(np.mean(curls)) if curls else 0.0

        return velocity, spread, finger_curl

    def update(self, feature_vector: np.ndarray | None) -> tuple[str, float]:
        """
        Feed one frame's feature vector. Returns (action_label, confidence).
        Uses heuristic when buffer not full or model not loaded.
        """
        if feature_vector is None:
            self.prev_landmarks = None
            return "idle", 1.0, {
                "pinch_dist": "—",
                "pinch": "—",
                "fingers": "—",
                "velocity": "—",
                "curl": "—",
                "n_ext": "—",
            }

        lm = feature_vector.reshape(21, 3)
        velocity, spread, finger_curl = self._compute_motion_features(lm)
        self.prev_landmarks = lm.copy()
        self.buffer.append(feature_vector)

        heuristic_label, heuristic_conf, dbg = self.heuristic.predict(lm, velocity, spread, finger_curl)

        if self.model is not None and len(self.buffer) == self.window_size:
            seq = torch.tensor(np.array(self.buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(seq)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            idx = int(np.argmax(probs))
            label, conf = ACTION_LABELS[idx], float(probs[idx])
        else:
            label, conf = heuristic_label, heuristic_conf

        # Confidence-weighted majority vote over last N predictions
        self.smoothing_window.append((label, conf))
        scores: dict[str, float] = {}
        for l, c in self.smoothing_window:
            scores[l] = scores.get(l, 0.0) + c
        smoothed = max(scores, key=scores.get)
        return smoothed, conf, dbg
