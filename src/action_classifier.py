import numpy as np
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path


ACTION_LABELS = ["idle", "reach", "pick", "inspect", "place", "verify"]
NUM_CLASSES = len(ACTION_LABELS)
FEATURE_DIM = 63   # 21 landmarks × 3 coords
WINDOW_SIZE = 20   # frames per prediction


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
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = self.classifier(out[:, -1, :])
        return out


class HeuristicClassifier:
    """
    Rule-based fallback classifier using hand landmark geometry.
    Used when no trained model is available — lets the demo run immediately.
    """

    def predict(self, landmarks: np.ndarray, velocity: float, spread: float,
                finger_curl: float) -> tuple[str, float]:
        """
        landmarks : (21, 3) wrist-relative, normalized
        velocity  : mean landmark displacement from previous frame
        spread    : mean distance of fingertips from palm center
        finger_curl: mean curl ratio of fingers (0=open, 1=fist)
        """
        if velocity < 0.02 and finger_curl < 0.35:
            return "idle", 0.80
        if velocity < 0.02 and finger_curl >= 0.35:
            return "inspect", 0.75
        if velocity >= 0.02 and finger_curl < 0.3:
            return "reach", 0.72
        if velocity >= 0.02 and finger_curl >= 0.55:
            return "pick", 0.70
        # index finger pointing (verify): index extended, others curled
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        index_extended = np.linalg.norm(index_tip - index_mcp) > 0.25
        if index_extended and finger_curl >= 0.5:
            return "verify", 0.68
        return "place", 0.65


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
        self.smoothing_window: deque = deque(maxlen=5)

        self.model: ActionLSTM | None = None
        self.heuristic = HeuristicClassifier()

        if model_path and Path(model_path).exists():
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

        # curl: ratio of fingertip distance to MCP distance for each finger
        fingers = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]
        curls = []
        for tip_i, mcp_i in fingers:
            tip_dist = np.linalg.norm(lm[tip_i] - lm[0])
            mcp_dist = np.linalg.norm(lm[mcp_i] - lm[0])
            if mcp_dist > 1e-6:
                curl = 1.0 - min(tip_dist / (mcp_dist * 1.8), 1.0)
                curls.append(curl)
        finger_curl = float(np.mean(curls)) if curls else 0.0

        return velocity, spread, finger_curl

    def update(self, feature_vector: np.ndarray | None) -> tuple[str, float]:
        """
        Feed one frame's feature vector. Returns (action_label, confidence).
        Uses heuristic when buffer not full or model not loaded.
        """
        if feature_vector is None:
            self.prev_landmarks = None
            return "idle", 1.0

        lm = feature_vector.reshape(21, 3)
        velocity, spread, finger_curl = self._compute_motion_features(lm)
        self.prev_landmarks = lm.copy()
        self.buffer.append(feature_vector)

        if self.model is not None and len(self.buffer) == self.window_size:
            seq = torch.tensor(np.array(self.buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(seq)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            idx = int(np.argmax(probs))
            label, conf = ACTION_LABELS[idx], float(probs[idx])
        else:
            label, conf = self.heuristic.predict(lm, velocity, spread, finger_curl)

        # Temporal smoothing: majority vote over last N predictions
        self.smoothing_window.append(label)
        votes = {}
        for l in self.smoothing_window:
            votes[l] = votes.get(l, 0) + 1
        smoothed = max(votes, key=votes.get)
        return smoothed, conf
