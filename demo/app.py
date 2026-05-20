"""
Streamlit demo for SOP-Vision.
Run:  streamlit run demo/app.py
"""
import sys
import time
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import SOPVisionPipeline


st.set_page_config(
    page_title="SOP-Vision",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-box { background:#111; border-radius:8px; padding:12px; text-align:center; }
  .step-done  { color:#32CD32; font-weight:bold; }
  .step-skip  { color:#DD2222; }
  .step-oor   { color:#FFA500; }
  .step-next  { color:#FFFFFF; font-weight:bold; }
  .step-pend  { color:#666; }
</style>
""", unsafe_allow_html=True)

st.title("SOP-Vision")
st.caption("Real-Time Egocentric Procedure Compliance from First-Person Video")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    source_type = st.radio("Input Source", ["Webcam", "Video File"])
    video_file = None
    if source_type == "Video File":
        video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    model_path = st.text_input("Model Path (optional)", value="",
                                help="Path to trained action_lstm.pt — leave blank for heuristic mode")
    dwell = st.slider("Step Confirmation Frames", 8, 30, 12,
                      help="Frames an action must persist before counted as a step")
    st.divider()
    st.markdown("**Controls**")
    st.markdown("- `Q` / `ESC` — quit  \n- `R` — reset SOP")
    st.divider()
    st.markdown("**Gesture Guide**")
    gesture_guide = {
        "reach":   ("🖐️ Open hand",        "Spread all 4+ fingers wide and move hand toward the object"),
        "pick":    ("🤌 Pinch",             "Touch thumb tip to index tip, curl remaining fingers"),
        "inspect": ("✌️ Peace sign",        "Extend only index + middle finger, hold still"),
        "place":   ("✊ Fist + move",       "Curl all fingers into a fist and move hand downward"),
        "verify":  ("☝️ Index point",       "Extend index finger only (gun shape), rest curled"),
    }
    for i, (step, (icon, desc)) in enumerate(gesture_guide.items(), 1):
        st.markdown(f"**{i}. {step.capitalize()}** {icon}")
        st.caption(desc)

# ── Main ──────────────────────────────────────────────────────────────────────
col_feed, col_status = st.columns([2, 1])

with col_feed:
    run_btn = st.button("▶  Start", type="primary", use_container_width=True)
    stop_btn = st.button("■  Stop", use_container_width=True)
    frame_placeholder = st.empty()

GESTURE_HINT = {
    "reach":   "🖐️ Open hand — spread all fingers and move toward object",
    "pick":    "🤌 Pinch — touch thumb tip to index tip, curl others",
    "inspect": "✌️ Peace sign — extend index + middle only, hold still",
    "place":   "✊ Fist + move — curl all fingers and move hand down",
    "verify":  "☝️ Point — extend index finger only (gun shape)",
}

with col_status:
    st.subheader("Live Status")
    action_ph = st.empty()
    conf_ph = st.empty()
    next_step_ph = st.empty()
    steps_ph = st.empty()
    report_ph = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False

if run_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

if st.session_state.running:
    mp = model_path.strip() if model_path.strip() else None
    pipeline = SOPVisionPipeline(
        config_path="configs/sop_config.yaml",
        model_path=mp,
    )
    pipeline.validator.min_dwell_frames = dwell

    if source_type == "Webcam":
        source = 0
    else:
        if video_file is None:
            st.warning("Please upload a video file.")
            st.stop()
        tmp = Path("data/tmp_upload.mp4")
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_bytes(video_file.read())
        source = str(tmp)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Cannot open video source.")
        st.stop()

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, info = pipeline.process_frame(frame)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            sop = info["sop"]
            action_ph.metric("Action", info["action"].upper())
            conf_ph.metric("Confidence", f"{info['confidence']:.0%}")

            next_step = sop.get("next_expected")
            if next_step and not sop.get("procedure_done"):
                hint = GESTURE_HINT.get(next_step, "")
                next_step_ph.info(f"**Next:** {next_step.upper()}\n\n{hint}")
            else:
                next_step_ph.empty()

            step_lines = []
            for i, s in enumerate(["reach", "pick", "inspect", "place", "verify"]):
                if s in sop.get("skipped", []):
                    step_lines.append(f'<span class="step-skip">✕ {s}</span>')
                elif s in sop.get("out_of_order", []):
                    step_lines.append(f'<span class="step-oor">⚠ {s}</span>')
                elif s in sop.get("completed", []):
                    step_lines.append(f'<span class="step-done">✓ {s}</span>')
                elif i == sop.get("current_step_idx", 0):
                    step_lines.append(f'<span class="step-next">→ {s}</span>')
                else:
                    step_lines.append(f'<span class="step-pend">○ {s}</span>')

            steps_ph.markdown("<br>".join(step_lines), unsafe_allow_html=True)

            if sop.get("procedure_done"):
                report = pipeline.validator.get_report()
                color = "green" if report.is_compliant else "red"
                status = "COMPLIANT ✓" if report.is_compliant else "VIOLATIONS ✗"
                report_ph.markdown(
                    f"<div style='color:{color};font-size:1.2em;font-weight:bold'>"
                    f"{status}<br>{report.completion_pct}% complete</div>",
                    unsafe_allow_html=True
                )
                break

            time.sleep(0.01)
    finally:
        cap.release()
        pipeline.close()
        st.session_state.running = False
