"""
Pose Estimation Comparison: MediaPipe vs YOLO
=============================================
Kullanim:
    python compare_pose.py                          # webcam (kamera 0)
    python compare_pose.py --source video.mp4       # video dosyasi
    python compare_pose.py --source video.mp4 --save cikti.mp4
"""

import cv2
import time
import argparse
import numpy as np
from collections import deque

# ---------- Bagimlilik kontrolleri ------------------------------------------
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
    print("[UYARI] mediapipe yuklu degil  ->  pip install mediapipe")

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[UYARI] ultralytics yuklu degil ->  pip install ultralytics")

# ---------- Renk paleti ------------------------------------------------------
COLORS = {
    "mp_bone":    (0, 220, 120),
    "mp_joint":   (255, 255, 255),
    "yolo_bone":  (0, 160, 255),
    "yolo_joint": (255, 220,  50),
    "hud_bg":     (10,  10,  10),
    "fps_mp":     (80, 230, 140),
    "fps_yolo":   (80, 180, 255),
    "divider":    (55,  55,  55),
}

# YOLO COCO-17 iskelet baglantilari
YOLO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# ---------- FPS sayaci -------------------------------------------------------
class FPSCounter:
    def __init__(self, window=20):
        self._q = deque(maxlen=window)
        self._t = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self._q.append(now - self._t)
        self._t = now
        return 1.0 / (sum(self._q) / len(self._q)) if self._q else 0.0

# ---------- MediaPipe isleme ------------------------------------------------
class MPProcessor:
    """MediaPipe Pose Landmarker (yeni Task API)"""
    def __init__(self, complexity=1):
        if not MEDIAPIPE_OK:
            raise RuntimeError("mediapipe yuklu degil")

        model_map = {
            0: "pose_landmarker_lite.task",
            1: "pose_landmarker_full.task",
            2: "pose_landmarker_heavy.task",
        }
        model_asset_path = model_map.get(complexity, "pose_landmarker_full.task")

        try:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_asset_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=1, # Tek kisi algilama
                output_segmentation_masks=False,
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[HATA] MediaPipe modeli yuklenemedi: {model_asset_path}")
            print(f"Hata detayi: {e}")
            print("\nModel dosyasini manuel olarak indirip script ile ayni klasore koymayi deneyin:")
            print("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
            raise

        self.connections = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
        self.fps = FPSCounter()

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_image)
        out = frame.copy()

        if res.pose_landmarks:
            for person_landmarks in res.pose_landmarks:
                for connection in self.connections:
                    a, b = connection.start, connection.end
                    lm_a, lm_b = person_landmarks[a], person_landmarks[b]
                    if lm_a.visibility > 0.3 and lm_b.visibility > 0.3:
                        p1 = (int(lm_a.x * w), int(lm_a.y * h))
                        p2 = (int(lm_b.x * w), int(lm_b.y * h))
                        cv2.line(out, p1, p2, COLORS["mp_bone"], 2, cv2.LINE_AA)
                for pt in person_landmarks:
                    if pt.visibility > 0.3:
                        px, py = int(pt.x * w), int(pt.y * h)
                        cv2.circle(out, (px, py), 4, COLORS["mp_bone"],  -1, cv2.LINE_AA)
                        cv2.circle(out, (px, py), 2, COLORS["mp_joint"], -1, cv2.LINE_AA)

        return out, self.fps.tick()

    def release(self):
        self.landmarker.close()

# ---------- YOLO isleme -----------------------------------------------------
class YOLOProcessor:
    def __init__(self, model_name="yolo11n-pose.pt"):
        if not YOLO_OK:
            raise RuntimeError("ultralytics yuklu degil")
        self.model = YOLO(model_name)
        self.fps = FPSCounter()

    def process(self, frame):
        results = self.model(frame, verbose=False)
        out = frame.copy()

        for r in results:
            if r.keypoints is None:
                continue
            kps   = r.keypoints.xy.cpu().numpy()
            confs = r.keypoints.conf

            for i, person_kps in enumerate(kps):
                conf_arr = confs[i].cpu().numpy() if confs is not None else np.ones(17)

                for a, b in YOLO_SKELETON:
                    if conf_arr[a] > 0.3 and conf_arr[b] > 0.3:
                        p1 = tuple(person_kps[a].astype(int))
                        p2 = tuple(person_kps[b].astype(int))
                        if all(v > 0 for v in p1 + p2):
                            cv2.line(out, p1, p2, COLORS["yolo_bone"], 2, cv2.LINE_AA)

                for j, (x, y) in enumerate(person_kps):
                    if conf_arr[j] > 0.3 and x > 0 and y > 0:
                        cv2.circle(out, (int(x), int(y)), 4, COLORS["yolo_bone"],  -1, cv2.LINE_AA)
                        cv2.circle(out, (int(x), int(y)), 2, COLORS["yolo_joint"], -1, cv2.LINE_AA)

        return out, self.fps.tick()

# ---------- HUD -------------------------------------------------------------
def draw_hud(canvas, fps_mp, fps_yolo, frame_idx):
    h, w = canvas.shape[:2]
    half = w // 2

    # Orta cizgi
    cv2.line(canvas, (half, 0), (half, h), COLORS["divider"], 2)

    # Sol baslik
    cv2.rectangle(canvas, (0, 0), (half, 34), COLORS["hud_bg"], -1)
    cv2.putText(canvas, f"  MediaPipe  |  FPS: {fps_mp:5.1f}",
                (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS["fps_mp"], 1, cv2.LINE_AA)

    # Sag baslik
    cv2.rectangle(canvas, (half, 0), (w, 34), COLORS["hud_bg"], -1)
    cv2.putText(canvas, f"  YOLO11     |  FPS: {fps_yolo:5.1f}",
                (half + 6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS["fps_yolo"], 1, cv2.LINE_AA)

    # Kare sayaci
    cv2.putText(canvas, f"Frame: {frame_idx:06d}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

# ---------- Ana dongu -------------------------------------------------------
def run(source, save_path, complexity, yolo_model, canvas_w, panel_h):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[HATA] Kaynak acilamadi: {source}")
        return

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    panel_w = canvas_w // 2

    print(f"[BILGI] Kaynak : {source}  |  {orig_w}x{orig_h}  |  {total} kare")
    print(f"[BILGI] Panel  : {panel_w}x{panel_h}  |  Toplam genislik: {canvas_w}")
    print("[BILGI] Baslatiliyor — cikmak icin 'q' veya ESC\n")

    mp_proc   = MPProcessor(complexity=complexity)
    yolo_proc = YOLOProcessor(model_name=yolo_model)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, 25.0, (canvas_w, panel_h))
        print(f"[BILGI] Kayit: {save_path}")

    frame_idx = 0

    while True:
        ret, frm = cap.read()
        if not ret:
            print("[BILGI] Video bitti.")
            break

        out_mp,   fps_mp   = mp_proc.process(frm)
        out_yolo, fps_yolo = yolo_proc.process(frm)

        out_mp   = cv2.resize(out_mp,   (panel_w, panel_h))
        out_yolo = cv2.resize(out_yolo, (panel_w, panel_h))

        canvas = np.concatenate([out_mp, out_yolo], axis=1)
        draw_hud(canvas, fps_mp, fps_yolo, frame_idx)

        if writer:
            writer.write(canvas)

        cv2.imshow("Pose Comparison: MediaPipe vs YOLO", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("[BILGI] Kullanici tarafindan durduruldu.")
            break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  kare {frame_idx}/{total}  |  MP: {fps_mp:.1f} fps  |  YOLO: {fps_yolo:.1f} fps")

    cap.release()
    if writer:
        writer.release()
    mp_proc.release()
    cv2.destroyAllWindows()
    print("[BILGI] Tamamlandi.")

# ---------- CLI -------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MediaPipe vs YOLO Pose Karsilastirmasi")
    p.add_argument("--source",       default="0",              help="Video yolu veya webcam indeksi (varsayilan: 0)")
    p.add_argument("--save",         default=None,             help="Cikti video yolu (orn: output.mp4)")
    p.add_argument("--mp-complexity",type=int, default=1, choices=[0,1,2], help="MediaPipe model karmasikligi")
    p.add_argument("--yolo-model",   default="yolo11n-pose.pt",help="YOLO model agirligi")
    p.add_argument("--width",        type=int, default=1280,   help="Toplam canvas genisligi")
    p.add_argument("--height",       type=int, default=540,    help="Panel yuksekligi")
    args = p.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run(
        source      = src,
        save_path   = args.save,
        complexity  = args.mp_complexity,
        yolo_model  = args.yolo_model,
        canvas_w    = args.width,
        panel_h     = args.height,
    )