"""
Webcam-based training data collector.
Records landmark sequences for each action class and saves as .npz files.

Usage:
    python scripts/collect_data.py --action reach --samples 200 --out data/landmarks
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.hand_detector import HandDetector
from src.action_classifier import ACTION_LABELS, WINDOW_SIZE, FEATURE_DIM


def collect(action: str, num_samples: int, out_dir: str, window_size: int = WINDOW_SIZE):
    if action not in ACTION_LABELS:
        print(f"Unknown action '{action}'. Choose from: {ACTION_LABELS}")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    buffer: deque = deque(maxlen=window_size)
    collected: list[np.ndarray] = []
    recording = False

    print(f"\nCollecting '{action}' — press [SPACE] to start/stop recording, [Q] to quit.")
    print(f"Target: {num_samples} samples | Window: {window_size} frames\n")

    while len(collected) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detector.detect(frame)
        fv = detector.get_feature_vector(detection["landmarks"])
        if fv is not None:
            buffer.append(fv)

        status = f"[RECORDING]" if recording else "[PAUSED]"
        color = (0, 80, 255) if recording else (120, 120, 120)
        cv2.putText(frame, f"{status}  Action: {action.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Collected: {len(collected)} / {num_samples}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            recording = not recording

        if recording and len(buffer) == window_size:
            collected.append(np.array(buffer))
            buffer.clear()

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if collected:
        sequences = np.array(collected, dtype=np.float32)
        labels = np.full(len(sequences), ACTION_LABELS.index(action), dtype=np.int64)
        out_file = Path(out_dir) / f"{action}.npz"
        np.savez(out_file, sequences=sequences, labels=labels)
        print(f"Saved {len(sequences)} samples to {out_file}")
    else:
        print("No samples collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=ACTION_LABELS)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--out", default="data/landmarks")
    args = parser.parse_args()
    collect(args.action, args.samples, args.out)
