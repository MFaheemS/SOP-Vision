import cv2
import numpy as np
from pathlib import Path

from src.hand_detector import HandDetector
from src.action_classifier import ActionClassifier
from src.sop_validator import SOPValidator
from src.annotator import FrameAnnotator


class SOPVisionPipeline:
    """
    End-to-end inference pipeline:
      Frame → Hand Detection → Feature Extraction → Action Classification
             → SOP Validation → Annotated Output
    """

    def __init__(self, config_path: str = "configs/sop_config.yaml",
                 model_path: str | None = None, device: str = "cpu"):
        self.detector = HandDetector()
        self.classifier = ActionClassifier(model_path=model_path, device=device)
        self.validator = SOPValidator(config_path)
        self.annotator = FrameAnnotator()
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        self.frame_count += 1

        detection = self.detector.detect(frame)
        feature_vec = self.detector.get_feature_vector(detection["landmarks"])
        action, confidence = self.classifier.update(feature_vec)
        sop_state = self.validator.update(action, confidence)
        annotated = self.annotator.annotate(frame, detection, action, confidence, sop_state)

        return annotated, {
            "action": action,
            "confidence": confidence,
            "sop": sop_state,
            "hand_detected": detection["hand_detected"],
            "frame_count": self.frame_count,
        }

    def run_on_video(self, source: int | str = 0, output_path: str | None = None,
                     display: bool = True) -> dict:
        """
        Runs pipeline on webcam (source=0) or video file.
        Returns final compliance report as dict.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        writer = None
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, info = self.process_frame(frame)

                if writer:
                    writer.write(annotated)

                if display:
                    cv2.imshow("SOP-Vision", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    if key == ord("r"):
                        self.validator.reset()
                        self.annotator.trail.clear()
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        report = self.validator.get_report()
        return {
            "procedure": report.procedure_name,
            "completed_steps": report.completed_steps,
            "total_steps": report.total_steps,
            "missed_steps": report.missed_steps,
            "out_of_order_steps": report.out_of_order_steps,
            "is_compliant": report.is_compliant,
            "completion_pct": report.completion_pct,
        }

    def close(self):
        self.detector.close()
