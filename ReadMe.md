# 🦴 Pose Estimation Comparison: MediaPipe vs YOLO11

<p align="center">
  <img src="docs/preview.gif" alt="MediaPipe vs YOLO demo" width="800"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/MediaPipe-0.10+-green?style=flat-square" />
  <img src="https://img.shields.io/badge/YOLO11-ultralytics-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/OpenCV-4.8+-red?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

---

A single-file Python script that runs **Google MediaPipe Pose Landmarker** and **Ultralytics YOLO11 Pose** models simultaneously on a single video/webcam feed, presenting their pose estimation side-by-side. 

Real-time FPS tracking is rendered on each panel, with optional support to output the side-by-side comparison directly to a video file.

---

## ✨ Features

- **Resource-Optimized Single Feed** — Reads from a single camera/video stream to process both models on identical frames, avoiding resource conflicts and allowing seamless webcam sharing on Windows.
- **Real-Time FPS Tracking** — Independent moving-average FPS counters for both MediaPipe and YOLO11 panels.
- **Plug & Play** — Zero configuration required. The script automatically downloads all necessary YOLO weights and MediaPipe task files on the first run.
- **Recording Support** — Save your benchmark comparison in high-definition MP4 format using the `--save` flag.
- **Highly Configurable** — Easily switch between all YOLO11-pose variants (`n/s/m/l/x`) and MediaPipe complexity levels (`0`, `1`, `2`) directly from the command line.

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/pose-comparison.git
cd pose-comparison
pip install -r Requirements.TXT
```

> **Note:** Python **3.10+** is required.

---

## 💻 Usage

### 📷 Live Webcam (Default)

```bash
python Compare_pose.py
```

### 🎞️ Video File Source

```bash
python Compare_pose.py --source video.mp4
```

### 💾 Save the Output Comparison

```bash
python Compare_pose.py --source video.mp4 --save output.mp4
```

### ⚙️ Command-Line Arguments

```bash
python Compare_pose.py \
  --source          video.mp4       # Source: video file path or webcam index (default: 0) \
  --save            output.mp4      # Path to save output video (optional) \
  --mp-complexity   1               # MediaPipe complexity: 0 (fast) / 1 (balanced) / 2 (accurate) \
  --yolo-model      yolo11n-pose.pt # YOLO model weights \
  --width           1280            # Total combined canvas width (pixels) \
  --height          540             # Panel height (pixels)
```

---

## 🧠 Model Configuration Options

### MediaPipe — `--mp-complexity`

| Value | Description | Speed |
|-------|-------------|-------|
| `0` | Lite model | Fastest |
| `1` | Balanced (default) | Medium |
| `2` | Heavy model | Most Accurate |

### YOLO11 — `--yolo-model`

| Model Weights | Variant | Inference Speed |
|---------------|---------|-----------------|
| `yolo11n-pose.pt` | Nano (default) | Fastest |
| `yolo11s-pose.pt` | Small | Fast |
| `yolo11m-pose.pt` | Medium | Balanced |
| `yolo11l-pose.pt` | Large | Accurate |
| `yolo11x-pose.pt` | Extra-large | Most Accurate |

> **Note:** On the first execution, any missing model weights or task assets will be downloaded automatically.

---

## 📊 Output Layout

```
┌──────────────────────────────┬──────────────────────────────┐
│  MediaPipe  │  FPS:  42.3    │  YOLO11     │  FPS:  28.7   │
│                              │                              │
│     [Pose Skeleton Viz]      │     [Pose Skeleton Viz]      │
│                              │                              │
└──────────────────────────────┴──────────────────────────────┘
  Frame: 000123
```

To exit the visualization, simply press **`q`** or **`ESC`** while the OpenCV window is focused.

---

## 📝 Observations & Notes

- **Camera Sharing:** Unlike traditional dual-feed approaches, this script leverages a single shared `VideoCapture` instance. This prevents native lockups and permission issues when accessing webcams on Windows operating systems.
- **Model Behaviors:**
  - **YOLO11-pose** provides superior robustness in multi-person scenarios, complex occlusions, and diverse background settings.
  - **MediaPipe** is highly optimized for single-person scenarios and exhibits exceptionally low latency, making it ideal for localized and CPU-bound environments.

---

## 📦 Dependencies

```text
opencv-python>=4.8.0
mediapipe>=0.10.0
ultralytics>=8.3.0
numpy>=1.24.0
```

---

## 📄 License

MIT © 2024