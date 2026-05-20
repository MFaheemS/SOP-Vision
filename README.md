# SOP-Vision

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sop-vision-ydakrxmrmfysepcy7kk4rx.streamlit.app/)

**Real-Time Egocentric Procedure Compliance Detection**

SOP-Vision analyzes first-person (egocentric) video streams to detect hand actions and verify that assembly or inspection procedures are followed in the correct sequence — flagging missed or out-of-order steps in real time.

Applicable to: manufacturing assembly, surgical procedures, lab protocols, aircraft maintenance, and any workflow where step-order compliance matters.

---

## How It Works

```
Webcam / Video
      │
      ▼
MediaPipe Hands ──► 21 Landmarks (x, y, z) per frame
      │
      ▼
Feature Extraction ──► Wrist-relative, scale-normalized 63-dim vector
      │
      ▼
ActionLSTM (or Heuristic fallback) ──► Action label + confidence
      │
      ▼
SOPValidator ──► Dwell-confirmed step tracking → Compliance state
      │
      ▼
FrameAnnotator ──► Annotated frame with overlay + trail + SOP panel
```

**Action Classes:** `idle` · `reach` · `pick` · `inspect` · `place` · `verify`

The validator uses **dwell-based confirmation** — an action must persist for N consecutive frames before it advances the SOP, preventing noise from triggering false step completions.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit demo

```bash
streamlit run demo/app.py
```

Open [http://localhost:8501](http://localhost:8501) — works out of the box with your webcam using the heuristic classifier. No training required.

### 3. Run on a video file

```python
from src.pipeline import SOPVisionPipeline

pipeline = SOPVisionPipeline()
report = pipeline.run_on_video(source="my_video.mp4", output_path="annotated.mp4")
print(report)
pipeline.close()
```

---

## Training a Custom Model

### Step 1 — Collect landmark data

Record ~200 samples per action class via webcam:

```bash
python scripts/collect_data.py --action reach   --samples 200
python scripts/collect_data.py --action pick     --samples 200
python scripts/collect_data.py --action inspect  --samples 200
python scripts/collect_data.py --action place    --samples 200
python scripts/collect_data.py --action verify   --samples 200
python scripts/collect_data.py --action idle     --samples 200
```

### Step 2 — Train

```bash
python -m training.train --data_dir data/landmarks --epochs 50 --model_out models/action_lstm.pt
```

### Step 3 — Run with trained model

```bash
streamlit run demo/app.py
```

Enter the model path in the sidebar.

---

## Define a Custom SOP

Edit `configs/sop_config.yaml` to change the procedure name, step sequence, or dwell settings:

```yaml
procedure_name: "Surgical Instrument Check"
expected_sequence:
  - reach
  - pick
  - inspect
  - verify
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501).

---

## Project Structure

```
SOP-Vision/
├── src/
│   ├── hand_detector.py       # MediaPipe landmark extraction
│   ├── action_classifier.py   # Bidirectional LSTM + heuristic fallback
│   ├── sop_validator.py       # Dwell-based SOP sequence compliance
│   ├── annotator.py           # OpenCV overlay: trail, SOP panel, badges
│   └── pipeline.py            # End-to-end inference orchestrator
├── training/
│   ├── dataset.py             # LandmarkDataset with augmentation
│   └── train.py               # Training loop with val metrics
├── scripts/
│   └── collect_data.py        # Webcam data collection tool
├── demo/
│   └── app.py                 # Streamlit UI
├── configs/
│   └── sop_config.yaml        # Procedure definition
├── tests/                     # pytest suite
├── docker/Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml   # CI: test + lint + Docker build
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Hand Tracking | MediaPipe Hands |
| Action Recognition | Bidirectional LSTM (PyTorch) |
| Feature Engineering | Wrist-relative landmark normalization |
| Compliance Logic | Dwell-based finite state machine |
| Visualization | OpenCV overlays + Streamlit |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
