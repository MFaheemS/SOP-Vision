import cv2
import numpy as np
from collections import deque


STATUS_COLORS = {
    "completed": (50, 205, 50),
    "out_of_order": (0, 165, 255),
    "skipped": (0, 0, 220),
    "pending": (180, 180, 180),
    "active": (255, 255, 255),
}

ACTION_COLORS = {
    "idle":    (120, 120, 120),
    "reach":   (255, 200, 0),
    "pick":    (0, 200, 255),
    "inspect": (180, 100, 255),
    "place":   (50, 220, 100),
    "verify":  (50, 255, 255),
}


class FrameAnnotator:
    """Draws SOP compliance overlay, action label, and trajectory trail on frames."""

    def __init__(self, window_size: int = 30):
        self.trail: deque = deque(maxlen=window_size)  # wrist positions for trail

    def annotate(self, frame: np.ndarray, detection: dict, action: str,
                 confidence: float, sop_state: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # --- Wrist trail ---
        if detection["hand_detected"] and detection["landmarks"]:
            wrist = detection["landmarks"][0][0]
            px, py = int(wrist[0] * w), int(wrist[1] * h)
            self.trail.append((px, py))
        else:
            self.trail.append(None)

        self._draw_trail(overlay)

        # --- Action label pill ---
        color = ACTION_COLORS.get(action, (200, 200, 200))
        self._draw_pill(overlay, action.upper(), confidence, (20, 20), color)

        # --- Dwell progress bar ---
        if sop_state.get("dwell_progress", 0) > 0 and sop_state.get("dwell_action"):
            self._draw_dwell_bar(overlay, sop_state["dwell_progress"],
                                 sop_state["dwell_action"], w)

        # --- SOP panel (right side) ---
        self._draw_sop_panel(overlay, sop_state, w, h)

        # --- Compliance badge ---
        if sop_state.get("procedure_done"):
            self._draw_completion_badge(overlay, sop_state, w, h)

        return overlay

    def _draw_trail(self, frame: np.ndarray):
        points = [p for p in self.trail if p is not None]
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(alpha * 4))
            color = (int(50 * alpha), int(200 * alpha), int(255 * alpha))
            cv2.line(frame, points[i - 1], points[i], color, thickness, cv2.LINE_AA)

    def _draw_pill(self, frame: np.ndarray, label: str, conf: float,
                   pos: tuple, color: tuple):
        x, y = pos
        text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        pad = 10
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                      (30, 30, 30), -1)
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                      color, 2)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    cv2.LINE_AA)

    def _draw_dwell_bar(self, frame: np.ndarray, progress: float, action: str, w: int):
        bar_w = int(w * 0.4)
        x0, y0 = (w - bar_w) // 2, 15
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 14), (40, 40, 40), -1)
        fill = int(bar_w * progress)
        color = ACTION_COLORS.get(action, (200, 200, 200))
        cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + 14), color, -1)
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 14), (180, 180, 180), 1)
        cv2.putText(frame, f"Confirming: {action}", (x0, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_sop_panel(self, frame: np.ndarray, state: dict, w: int, h: int):
        panel_w, panel_h = 240, min(h - 20, 300)
        x0, y0 = w - panel_w - 10, 10
        sub = frame[y0:y0 + panel_h, x0:x0 + panel_w]
        black = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.35, black, 0.65, 0, sub)
        frame[y0:y0 + panel_h, x0:x0 + panel_w] = sub

        cv2.putText(frame, "SOP STEPS", (x0 + 8, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        steps = state.get("completed", [])
        skipped = state.get("skipped", [])
        out_of_order = state.get("out_of_order", [])
        expected_all = [
            "reach", "pick", "inspect", "place", "verify"
        ]
        current_idx = state.get("current_step_idx", 0)

        for i, step in enumerate(expected_all):
            ty = y0 + 38 + i * 46
            if step in skipped:
                status, col = "✕ SKIPPED", STATUS_COLORS["skipped"]
            elif step in out_of_order:
                status, col = "⚠ OUT OF ORDER", STATUS_COLORS["out_of_order"]
            elif step in steps:
                status, col = "✓ DONE", STATUS_COLORS["completed"]
            elif i == current_idx:
                status, col = "→ NEXT", STATUS_COLORS["active"]
            else:
                status, col = "○ PENDING", STATUS_COLORS["pending"]

            cv2.putText(frame, f"{i + 1}. {step.upper()}", (x0 + 8, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)
            cv2.putText(frame, status, (x0 + 8, ty + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        # Progress bar at bottom of panel
        bar_y = y0 + panel_h - 16
        total = state.get("total_steps", 5)
        done = len(steps)
        pct = done / total if total > 0 else 0
        bw = panel_w - 16
        cv2.rectangle(frame, (x0 + 8, bar_y), (x0 + 8 + bw, bar_y + 8), (60, 60, 60), -1)
        cv2.rectangle(frame, (x0 + 8, bar_y), (x0 + 8 + int(bw * pct), bar_y + 8),
                      (50, 205, 50), -1)
        cv2.putText(frame, f"{pct:.0%} complete", (x0 + 8, bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    def _draw_completion_badge(self, frame: np.ndarray, state: dict, w: int, h: int):
        skipped = state.get("skipped", [])
        oor = state.get("out_of_order", [])
        compliant = len(skipped) == 0 and len(oor) == 0
        text = "PROCEDURE COMPLETE — COMPLIANT" if compliant else "PROCEDURE DONE — VIOLATIONS FOUND"
        color = (50, 220, 50) if compliant else (30, 30, 220)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cx = (w - tw) // 2
        cv2.rectangle(frame, (cx - 12, h - 45), (cx + tw + 12, h - 15), (20, 20, 20), -1)
        cv2.rectangle(frame, (cx - 12, h - 45), (cx + tw + 12, h - 15), color, 2)
        cv2.putText(frame, text, (cx, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                    cv2.LINE_AA)
