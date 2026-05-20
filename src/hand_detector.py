import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = "models/hand_landmarker.task"

# MediaPipe hand connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def _ensure_model(model_path: str = DEFAULT_MODEL_PATH):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(model_path).exists():
        print(f"Downloading hand landmark model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Download complete.")


class HandDetector:
    """Extracts hand landmarks using MediaPipe Tasks HandLandmarker."""

    def __init__(self, max_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.6,
                 model_path: str = DEFAULT_MODEL_PATH):
        _ensure_model(model_path)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> dict:
        """
        Returns normalized landmarks, world landmarks, handedness, and annotated frame.
        landmark shape per hand: (21, 3) — x, y, z normalized [0,1]
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        landmarks = []
        world_landmarks = []
        handedness = []

        h, w = frame.shape[:2]

        for i, hand_lm in enumerate(result.hand_landmarks):
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)
            landmarks.append(lm_array)

            if result.hand_world_landmarks:
                wlm = result.hand_world_landmarks[i]
                world_landmarks.append(
                    np.array([[lm.x, lm.y, lm.z] for lm in wlm], dtype=np.float32)
                )

            if result.handedness:
                label = result.handedness[i][0].display_name
                handedness.append(label)

            # Draw skeleton manually
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, (0, 150, 80), 1, cv2.LINE_AA)

        return {
            "landmarks": landmarks,
            "world_landmarks": world_landmarks,
            "handedness": handedness,
            "frame": frame,
            "hand_detected": len(landmarks) > 0,
        }

    def get_feature_vector(self, landmarks: list) -> np.ndarray | None:
        """
        Flatten primary hand landmarks into a 63-dim wrist-relative,
        scale-normalized feature vector. Returns None if no hand detected.
        """
        if not landmarks:
            return None
        lm = landmarks[0].copy()
        wrist = lm[0].copy()
        lm -= wrist
        span = np.linalg.norm(lm[9])
        if span > 1e-6:
            lm /= span
        return lm.flatten().astype(np.float32)

    def close(self):
        self._landmarker.close()
