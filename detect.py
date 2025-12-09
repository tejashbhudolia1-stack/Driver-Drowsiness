#!/usr/bin/env python3
"""
detect.py — Real-time drowsiness, yawning, and distraction detection
Hybrid validation: Mediapipe FaceMesh + dlib EAR (Option A).

This is your working detect.py improved with:
 - [REWORK] Gaze detection for LEFT/RIGHT now uses robust Head Yaw (rotation)
 - [REWORK] Vertical (UP/DOWN) gaze now uses thresholds from calibration file.
 - All other performance tweaks and timers remain
 - [FIX] VideoWriter initialization moved to after camera capture
"""

import os
import time
import csv
from datetime import datetime
from collections import deque
import threading
import math # Added for head tilt angle calculation

import cv2
import mediapipe as mp
import numpy as np
import joblib

# Try to import dlib (optional). If unavailable, we gracefully fall back.
try:
    import dlib
    DLIB_AVAILABLE = True
except Exception:
    DLIB_AVAILABLE = False

# Try to import winsound for Windows beep
try:
    import winsound
    WINSOUND = True
except Exception:
    WINSOUND = False

# --- CONFIG ---
class Config:
    MODELS_DIR = "models"
    RECORDINGS_DIR = "recordings" # Logs will also be saved here
    
    DLIB_SHAPE_MODEL = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")

    # Mediapipe landmark indices
    LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 153, 154, 155]
    RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 380, 381, 382]
    LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]
    RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]
    CHIN_INDEX = 152
    MOUTH_INDICES = [13, 14, 87, 317]  # top & bottom lips for yawning

    # Head tilt (roll)
    LEFT_EYE_OUTER = 130
    RIGHT_EYE_OUTER = 359
    
    # [NEW] Head yaw (left/right turn)
    NOSE_TIP_INDEX = 1
    LEFT_FACE_EDGE_INDEX = 132
    RIGHT_FACE_EDGE_INDEX = 361

    # [MODIFIED] Timers adjusted for FRAME_SKIP = 1 (We process ~30 FPS)
    EYE_CLOSURE_WINDOW = 27          # ~0.9s @ 30 FPS
    EYE_CLOSURE_THRESHOLD = 0.99    

    NOD_FRAMES_THRESHOLD = 9        # ~0.3s @ 30 FPS
    DROP_FRAMES_THRESHOLD = 30      # ~1 sec @ 30 FPS
    TILT_ANGLE_THRESHOLD = 18       # degrees from baseline (for drowsiness)
    TILT_FRAMES_THRESHOLD = 45      # ~1.5 sec @ 30 FPS

    # Gaze tuning
    YAWN_THRESHOLD = 0.60
    YAWN_FRAMES_THRESHOLD = 90      # ~3 sec @ 30 FPS (Note: was 8 sec, 90 frames is 3s)

    # [NEW] Distraction thresholds (30 processed FPS assumption)
    DISTRACTION_LR_FRAMES_THRESHOLD = 150 # ~5 sec @ 30 FPS
    DISTRACTION_UP_FRAMES_THRESHOLD = 60  # ~2 sec @ 30 FPS
    # [NEW] Down distraction threshold
    DISTRACTION_DOWN_FRAMES_THRESHOLD = 90 # ~3 sec @ 30 FPS


    BASELINE_DURATION_S = 2.5      # This is in seconds, no change
    NOD_THRESHOLD_NORMALIZED = 0.025 # This is a ratio, no change

    # Dlib (optional) settings
    PROCESS_WIDTH = 640            # width to resize frames for processing
    FRAME_SKIP = 1                 # process every Nth frame
    
# === GAZE SMOOTHING & HYSTERESIS PARAMETERS ===
GAZE_SMOOTH_ALPHA = 0.6 # [CHANGED] Increased for faster reaction
GAZE_DEBOUNCE_FRAMES = 12 # [CHANGED] ~0.4s @ 30 FPS

# These are offsets from the calibrated baseline
GAZE_H_THRESH = 0.14 # This is a ratio - widened L/R threshold

# [REMOVED] IRIS_LOST_VALUE. We now check for 'None' directly.

# --- UI ---
class UI:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    COLOR_AWAKE = (0, 255, 0)
    COLOR_WARN = (0, 165, 255)
    COLOR_ALERT = (0, 0, 255)
    COLOR_INFO = (255, 255, 0)


# --- HELPERS ---
def ensure_log_file(log_path):
    """Creates the session-specific log file with a header if it doesn't exist."""
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["Timestamp", "User", "Event", "Detail"])

def log_event(log_path, user, event, detail=""):
    """Appends an event to the session-specific log file."""
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user, event, detail])


def get_eye_aspect_ratio_mp(landmarks, indices, frame_shape):
    """Compute vertical/horizontal ratio from Mediapipe eye landmarks (proxy EAR)."""
    h, w, _ = frame_shape
    if not landmarks:
        return 0.0
        
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    if pts.shape[0] < 9: 
        return 0.0
    top = np.mean([pts[1][1], pts[2][1], pts[3][1]])
    bottom = np.mean([pts[4][1], pts[5][1], pts[6][1]])
    vertical = abs(bottom - top)
    horizontal = abs(pts[0][0] - pts[8][0]) 
    return vertical / horizontal if horizontal > 0 else 0.0


def get_mouth_open_ratio(landmarks, frame_shape):
    """Estimate mouth opening to detect yawns (Mouth aspect ratio proxy)."""
    h, w, _ = frame_shape
    top = landmarks[Config.MOUTH_INDICES[0]].y * h
    bottom = landmarks[Config.MOUTH_INDICES[1]].y * h
    left = landmarks[Config.MOUTH_INDICES[2]].x * w
    right = landmarks[Config.MOUTH_INDICES[3]].x * w
    vertical = abs(bottom - top)
    horizontal = abs(right - left)
    return vertical / horizontal if horizontal > 0 else 0.0


def get_vertical_gaze_ratio(landmarks, frame_shape):
    """ Calculates VERTICAL (Y) gaze ratio from iris."""
    h, w, _ = frame_shape
    try:
        left_iris = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in Config.LEFT_IRIS_INDICES], axis=0)
        right_iris = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in Config.RIGHT_IRIS_INDICES], axis=0)

        left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in Config.LEFT_EYE_INDICES])
        right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in Config.RIGHT_EYE_INDICES])

        l_min_y = left_eye[:, 1].min()
        l_max_y = left_eye[:, 1].max()
        r_min_y = right_eye[:, 1].min()
        r_max_y = right_eye[:, 1].max()

        l_y = (left_iris[1] - l_min_y) / max(1, (l_max_y - l_min_y))
        r_y = (right_iris[1] - r_min_y) / max(1, (r_max_y - r_min_y))

        gy = (l_y + r_y) / 2.0
        return gy

    except Exception:
        # This is the "obvious" check. Failure to find iris landmarks
        # (e.g., eyes closed) will cause an exception.
        return None

def get_head_yaw_ratio(landmarks):
    """ [NEW] Calculates HORIZONTAL (X) head yaw ratio from face landmarks."""
    try:
        nose_x = landmarks[Config.NOSE_TIP_INDEX].x
        left_edge_x = landmarks[Config.LEFT_FACE_EDGE_INDEX].x
        right_edge_x = landmarks[Config.RIGHT_FACE_EDGE_INDEX].x
        
        yaw_ratio = (nose_x - left_edge_x) / max(1e-6, (right_edge_x - left_edge_x))
        return yaw_ratio
    except Exception:
        return None

# [REMOVED] track_iris_gaze function. Logic is now in the main loop.

def run_baseline_calibration(cap, face_mesh):
    """
    [MODIFIED] This function still runs to get DYNAMIC baselines for
    chin, head angle, head yaw, and center gaze.
    """
    print(f"[INFO] Calibrating dynamic baseline for {Config.BASELINE_DURATION_S} seconds...")
    t0 = time.time()
    chin_positions, head_angles, gaze_y_positions, yaw_ratios = [], [], [], []

    while time.time() - t0 < Config.BASELINE_DURATION_S:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        small = resize_for_processing(frame)
        h_proc, w_proc = small.shape[:2] 
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. Chin (Head Drop)
            chin_positions.append(landmarks[Config.CHIN_INDEX].y)

            # 2. Head Tilt (Roll)
            try:
                dx_norm = landmarks[Config.RIGHT_EYE_OUTER].x - landmarks[Config.LEFT_EYE_OUTER].x
                dy_norm = landmarks[Config.RIGHT_EYE_OUTER].y - landmarks[Config.LEFT_EYE_OUTER].y
                angle_deg = math.degrees(math.atan2(dy_norm * (h_proc / w_proc), dx_norm))
                head_angles.append(angle_deg)
            except Exception:
                pass 
            
            # 3. Vertical Gaze (Iris)
            gy = get_vertical_gaze_ratio(landmarks, small.shape)
            if gy is not None:
                gaze_y_positions.append(gy)
                
            # 4. Head Yaw (Left/Right)
            yaw_ratio = get_head_yaw_ratio(landmarks)
            if yaw_ratio is not None:
                yaw_ratios.append(yaw_ratio)

        cv2.putText(frame, "Calibrating... Look Center", (10, 30), UI.FONT, 0.8, UI.COLOR_INFO, 2)
        cv2.imshow("Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return None, None, None, None

    # Return all 4 dynamic values
    chin_med = np.median(chin_positions) if chin_positions else None
    angle_med = np.median(head_angles) if head_angles else None
    gaze_y_med = np.median(gaze_y_positions) if gaze_y_positions else None
    yaw_ratio_med = np.median(yaw_ratios) if yaw_ratios else None

    return chin_med, angle_med, gaze_y_med, yaw_ratio_med


def draw_ui(frame, status, gaze_str, user, elapsed_time, total_yawns, mp_ear, dlib_ear, mar, use_dlib_flag):
    color = UI.COLOR_AWAKE
    if status == "Drowsy" or status == "Yawning":
        color = UI.COLOR_ALERT
    elif status == "Distracted":
        color = UI.COLOR_WARN
    
    cv2.putText(frame, f"STATUS: {status}", (10, 40), UI.FONT, 1.2, color, 2)
    cv2.putText(frame, f"GAZE: {gaze_str}", (10, 80), UI.FONT, 0.9, UI.COLOR_INFO, 2)
    cv2.putText(frame, f"YAWNS: {total_yawns}", (10, 120), UI.FONT, 0.9, (255, 255, 255), 2)

    ear_color = (255, 255, 255) # White
    cv2.putText(frame, f"MP EAR: {mp_ear:.3f}", (10, 160), UI.FONT, 0.7, ear_color, 2)
    
    y_offset = 190
    if use_dlib_flag:
        cv2.putText(frame, f"DLIB EAR: {dlib_ear:.3f}", (10, y_offset), UI.FONT, 0.7, ear_color, 2)
        y_offset += 30 # Move MAR down
    
    # [NEW] Add MAR value
    cv2.putText(frame, f"MAR: {mar:.3f}", (10, y_offset), UI.FONT, 0.7, ear_color, 2)
    
    cv2.putText(frame, f"USER: {user}", (10, frame.shape[0] - 10), UI.FONT, 0.6, (200, 200, 200), 1)

    mins, secs = divmod(int(elapsed_time), 60)
    timer_text = f"REC {mins:02}:{secs:02}"
    (tw, th), _ = cv2.getTextSize(timer_text, UI.FONT, 0.8, 2)
    cv2.putText(frame, timer_text, (frame.shape[1] - tw - 20, 40), UI.FONT, 0.8, (0, 0, 255), 2)


# --- DLIB helper (optional) ---
def compute_dlib_ear_from_shape_np(shape_np):
    left_idx = list(range(36, 42))  
    right_idx = list(range(42, 48)) 

    left_eye = shape_np[left_idx]
    right_eye = shape_np[right_idx]

    def ear(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    leftEAR = ear(left_eye)
    rightEAR = ear(right_eye)
    return (leftEAR + rightEAR) / 2.0


# --- UTILS: speed helpers & beep ---
def resize_for_processing(frame):
    h, w = frame.shape[:2]
    if w <= Config.PROCESS_WIDTH:
        return frame.copy()
    new_w = Config.PROCESS_WIDTH
    new_h = int(h * (new_w / w))
    return cv2.resize(frame, (new_w, new_h))


def beep_alert():
    """Non-blocking beep (tries winsound on Windows, otherwise system bell)."""
    def _beep():
        if WINSOUND:
            try:
                winsound.Beep(1000, 450)
            except Exception:
                print("\a", end="")
        else:
            print("\a", end="", flush=True)
    threading.Thread(target=_beep, daemon=True).start()


# --- MAIN ---
def main():
    if not os.path.exists(Config.MODELS_DIR):
        print("[ERROR] Models directory missing. Run calibrate.py first.")
        return

    models = [m for m in os.listdir(Config.MODELS_DIR) if m.endswith(".pkl")]
    if not models:
        print("[ERROR] No models found.")
        return

    print("Available profiles:")
    for i, m in enumerate(models):
        print(f"  {i+1}. {m}")
    choice = input("Select user: ").strip()
    try:
        model_file = models[int(choice) - 1]
    except Exception:
        model_file = choice if choice in models else None
    if not model_file:
        print("Invalid choice.")
        return

    model_path = os.path.join(Config.MODELS_DIR, model_file)
    model = joblib.load(model_path)
    print(f"[INFO] Loaded calibration: {model_file}")

    # Load adaptive thresholds
    dlib_open_ear = model.get("eye_open_EAR") or model.get("eye_open_EAR_val")
    dlib_closed_ear = model.get("eye_closed_EAR") or model.get("eye_closed_EAR_val")
    mp_open_ratio = model.get("eye_open_ratio")
    mp_closed_ratio = model.get("eye_closed_ratio")

    # [NEW] Load gaze thresholds from file
    calibrated_gaze_center = model.get("baseline_gaze_y")
    calibrated_gaze_up = model.get("gaze_up_val")
    calibrated_gaze_down = model.get("gaze_down_val")
    
    if any(v is None for v in [calibrated_gaze_center, calibrated_gaze_up, calibrated_gaze_down]):
        print("\n[ERROR] Your calibration file is missing the new gaze values.")
        print("Please re-run 'calibrate.py' to generate a new model file.")
        return

    if dlib_open_ear is not None and dlib_closed_ear is not None:
        adaptive_dlib_thresh = float((dlib_open_ear + dlib_closed_ear) / 2.0)
        adaptive_dlib_thresh = max(0.12, adaptive_dlib_thresh - 0.01)
        print(f"[INFO] Adaptive Dlib EAR threshold computed: {adaptive_dlib_thresh:.3f}")
    else:
        adaptive_dlib_thresh = 0.25 
        print(f"[WARN] Calibration lacks Dlib EAR values; using default Dlib EAR threshold {adaptive_dlib_thresh:.3f}")

    if mp_open_ratio is not None and mp_closed_ratio is not None:
        adaptive_mp_thresh = (mp_open_ratio + mp_closed_ratio) / 2.0
        adaptive_mp_thresh = max(0.1, adaptive_mp_thresh - 0.01) 
        print(f"[INFO] Adaptive Mediapipe ratio threshold computed: {adaptive_mp_thresh:.3f}")
    else:
        adaptive_mp_thresh = None 
        mp_baseline = mp_open_ratio or dlib_open_ear or 0.28
        print(f"[WARN] Calibration lacks 'eye_closed_ratio'; using relative MP logic (baseline: {mp_baseline:.3f}).")


    os.makedirs(Config.RECORDINGS_DIR, exist_ok=True)
    
    session_time = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    video_filename = os.path.join(Config.RECORDINGS_DIR, session_time + ".avi")
    log_path = os.path.join(Config.RECORDINGS_DIR, session_time + ".csv")

    ensure_log_file(log_path)


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return
    
    # [FIX] REMOVED VideoWriter and start_time initialization from here
    # It is now initialized *inside* the loop after the first frame is read.

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    use_dlib = False
    dlib_detector = None
    dlib_predictor = None
    
    if DLIB_AVAILABLE and os.path.exists(Config.DLIB_SHAPE_MODEL):
        try:
            dlib_detector = dlib.get_frontal_face_detector()
            dlib_predictor = dlib.shape_predictor(Config.DLIB_SHAPE_MODEL)
            use_dlib = True
            print("[INFO] Dlib loaded and will be used for EAR.")
        except Exception as e:
            print(f"[WARN] Dlib present but failed to initialize: {e}. Falling back to Mediapipe-only.")
            use_dlib = False
    else:
        if not DLIB_AVAILABLE:
            print("[WARN] dlib not installed — running Mediapipe-only.")
        else:
            print(f"[WARN] Missing {Config.DLIB_SHAPE_MODEL} — run in Mediapipe-only mode.")

    # --- NEW BLOCK START ---
    # Show instructions for the *dynamic* baseline calibration
    print(f"""
    -----------------------------------------------------
        Welcome to the Drowsiness Detection System
    -----------------------------------------------------

    Profile '{model_file}' is loaded.

    Before detection starts, we must run a short {Config.BASELINE_DURATION_S}-second
    dynamic calibration.

    [ACTION] Please look *directly at the camera* and sit
             in your normal, alert driving/working position.

    This will establish your baseline "center" for gaze,
    head tilt, and head position.
    """)
    
    try:
        input("==> Press ENTER to begin dynamic calibration... ")
    except KeyboardInterrupt:
        print("\n\n[INFO] Detection cancelled by user.")
        # Clean up resources if user quits here
        cap.release()
        # out.release() # 'out' doesn't exist yet
        face_mesh.close()
        return
    
    print("\n[INFO] Starting dynamic calibration...")
    # --- NEW BLOCK END ---

    # [MODIFIED] Get all 4 DYNAMIC baseline values
    baseline_chin, baseline_head_angle, dynamic_gaze_center, baseline_yaw_ratio = run_baseline_calibration(cap, face_mesh)
    
    if any(v is None for v in [baseline_chin, baseline_head_angle, dynamic_gaze_center, baseline_yaw_ratio]):
        print("\n[ERROR] Dynamic calibration failed. Could not read all metrics.")
        print("Please ensure your face is well-lit and centered. Quitting.")
        cap.release()
        cv2.destroyAllWindows()
        return
    else:
        print("\n--- DYNAMIC CALIBRATION COMPLETE ---")
        print(f"[INFO] Baseline chin: {baseline_chin:.3f}, Baseline angle: {baseline_head_angle:.2f} deg")
        print(f"[INFO] Baseline Head Yaw: {baseline_yaw_ratio:.3f} (0.0=Left, 1.0=Right)")
        print(f"[INFO] Baseline Gaze Y: {dynamic_gaze_center:.3f} (0.0=UP, 1.0=DOWN)")
        print("------------------------------------")

    if 0.3 <= dynamic_gaze_center <= 0.7 and 0.3 <= baseline_yaw_ratio <= 0.7:
        print(f"[INFO] Dynamic Gaze Y ({dynamic_gaze_center:.3f}) and Yaw ({baseline_yaw_ratio:.3f}) are in the valid range (0.3-0.7).")
    else:
        print(f"\n[ERROR] Dynamic calibration values are outside the 0.3-0.7 range.")
        if not (0.3 <= dynamic_gaze_center <= 0.7):
            print(f"[ERROR] Gaze Y ({dynamic_gaze_center:.3f}) is bad. You may not be looking directly at the camera.")
        if not (0.3 <= baseline_yaw_ratio <= 0.7):
            print(f"[ERROR] Head Yaw ({baseline_yaw_ratio:.3f}) is bad. You may not be facing the camera directly.")
        print("Please quit (Ctrl+C or close window) and run the script again.")
        cap.release()
        cv2.destroyAllWindows()
        return 

    # [FIXED] Define dynamic gaze thresholds based on DYNAMIC center and LOADED (calibrated) ranges
    
    # 1. [NEW] Auto-correct for inverted calibration (UP > DOWN)
    #    Ensure 'target_up' is always the smaller value and 'target_down' is the larger.
    target_up = min(calibrated_gaze_up, calibrated_gaze_down)
    target_down = max(calibrated_gaze_up, calibrated_gaze_down)
    
    # 2. Calculate the gaze range from the calibration file
    #    This is the fixed range of motion the user demonstrated
    gaze_up_range = abs(calibrated_gaze_center - target_up)
    gaze_down_range = abs(target_down - calibrated_gaze_center)
    
    # 3. Apply this fixed range to the new *dynamic* center
    
    # [TUNED] Set 'UP' detection to be very sensitive (40% enter, 20% exit)
    GAZE_UP_ENTER = dynamic_gaze_center - (gaze_up_range * 0.4) 
    GAZE_UP_EXIT = dynamic_gaze_center - (gaze_up_range * 0.2)
    
    # [RE-ENABLED] 'DOWN' detection with 60% enter, 30% exit
    GAZE_DOWN_ENTER = dynamic_gaze_center + (gaze_down_range * 0.5) 
    GAZE_DOWN_EXIT = dynamic_gaze_center + (gaze_down_range * 0.2)
    
    GAZE_LEFT_ENTER = baseline_yaw_ratio - GAZE_H_THRESH      
    GAZE_RIGHT_ENTER = baseline_yaw_ratio + GAZE_H_THRESH      
    GAZE_LEFT_EXIT = baseline_yaw_ratio - (GAZE_H_THRESH * 0.7)      
    GAZE_RIGHT_EXIT = baseline_yaw_ratio + (GAZE_H_THRESH * 0.7)    

    print(f"[INFO] Gaze UP_ENTER:   < {GAZE_UP_ENTER:.3f} (Calibrated Target: {target_up:.3f})")
    print(f"[INFO] Gaze DOWN_ENTER: > {GAZE_DOWN_ENTER:.3f} (Calibrated Target: {target_down:.3f})")
    print(f"[INFO] Gaze LEFT_ENTER: < {GAZE_LEFT_ENTER:.3f}")
    print(f"[INFO] Gaze RIGHT_ENTER: > {GAZE_RIGHT_ENTER:.3f}")
    
    if target_up != calibrated_gaze_up:
         print(f"[WARN] Your calibrated UP/DOWN values were inverted. Auto-corrected.")
    
    input("\nPress ENTER to activate detection...")

    yaw_ema = baseline_yaw_ratio
    gaze_y_ema = dynamic_gaze_center
    
    gaze_candidate = "CENTER"
    gaze_debounce_count = 0
    gaze = "CENTER"

    eye_history = deque(maxlen=Config.EYE_CLOSURE_WINDOW)
    gaze_lr_counter = 0
    gaze_up_counter = 0
    gaze_down_counter = 0 # [NEW]
    nod_counter = 0
    tilt_counter = 0  
    drop_counter = 0  
    yawn_counter = 0
    total_yawns = 0
    last_log_status = ""
    current_status = "Awake"

    mp_ear_display = 0.0
    dlib_ear_display = 0.0
    mar_display = 0.0 # [NEW]

    frame_count = 0
    print("\n[INFO] Starting detection and recording. Press ESC to exit.")

    # [FIX] Initialize writer and timer *after* calibration,
    # using the *actual* frame dimensions.
    out = None
    start_time = time.time() # Timer starts now
    
    while True:
        ret, orig_frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame = cv2.flip(orig_frame, 1)
        
        # [FIX] One-time initialization of the VideoWriter
        if out is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_filename, fourcc, 20.0, (w, h))
            print(f"[INFO] Video recording started ({w}x{h} @ 20 FPS).")
        
        proc = None
        gray = None
        rgb = None 
        elapsed_time = time.time() - start_time

        face_detected = False
        is_drowsy = False 
        is_distracted = False 
        is_yawning = False
        is_head_drop = False # [NEW] Flag to track head drop
        is_closed = False    # [NEW] Flag for eye closure state

        if (frame_count % Config.FRAME_SKIP) == 0:
            proc = resize_for_processing(frame)
            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(rgb)
            face_detected = bool(results.multi_face_landmarks)

            mp_ear = None
            dlib_ear = None

            if face_detected:
                landmarks = results.multi_face_landmarks[0].landmark
                h_proc, w_proc = proc.shape[:2]  

                left_ear_mp = get_eye_aspect_ratio_mp(landmarks, Config.LEFT_EYE_INDICES, proc.shape)
                right_ear_mp = get_eye_aspect_ratio_mp(landmarks, Config.RIGHT_EYE_INDICES, proc.shape)
                mp_ear = (left_ear_mp + right_ear_mp) / 2.0
                mp_ear_display = mp_ear

                if use_dlib:
                    try:
                        rects = dlib_detector(gray, 0) 
                        if len(rects) > 0:
                            shape = dlib_predictor(gray, rects[0])
                            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                            dlib_ear = compute_dlib_ear_from_shape_np(shape_np)
                            dlib_ear_display = dlib_ear
                    except Exception:
                        dlib_ear = None
                
                if adaptive_mp_thresh is not None:
                    mp_closed = mp_ear < adaptive_mp_thresh
                else:
                    mp_ratio = mp_ear / mp_baseline if mp_baseline > 0 else 1.0
                    mp_closed = mp_ratio < 0.78 

                dlib_closed = None
                if dlib_ear is not None:
                    dlib_closed = dlib_ear < adaptive_dlib_thresh

                if dlib_closed is not None:
                    is_closed = mp_closed or dlib_closed # [MODIFIED] Set flag
                else:
                    is_closed = mp_closed # [MODIFIED] Set flag

                eye_history.append(0 if is_closed else 1)

                if len(eye_history) == Config.EYE_CLOSURE_WINDOW:
                    closed_ratio = 1 - (sum(eye_history) / len(eye_history))
                    if closed_ratio >= Config.EYE_CLOSURE_THRESHOLD:
                        is_drowsy = True

                if baseline_chin > 0:
                    current_chin_y = landmarks[Config.CHIN_INDEX].y
                    chin_diff = current_chin_y - baseline_chin 
                    
                    if chin_diff > Config.NOD_THRESHOLD_NORMALIZED: 
                        nod_counter += 1
                        if nod_counter >= Config.NOD_FRAMES_THRESHOLD:
                            is_drowsy = True 
                        
                        drop_counter += 1
                        if drop_counter >= Config.DROP_FRAMES_THRESHOLD:
                            is_drowsy = True 
                            is_head_drop = True # [NEW]
                    else:
                        nod_counter = 0 
                        drop_counter = 0 
                
                try:
                    dx_norm = landmarks[Config.RIGHT_EYE_OUTER].x - landmarks[Config.LEFT_EYE_OUTER].x
                    dy_norm = landmarks[Config.RIGHT_EYE_OUTER].y - landmarks[Config.LEFT_EYE_OUTER].y
                    current_angle_deg = math.degrees(math.atan2(dy_norm * (h_proc / w_proc), dx_norm))
                    angle_diff = abs(current_angle_deg - baseline_head_angle)

                    if angle_diff > Config.TILT_ANGLE_THRESHOLD:
                        tilt_counter += 1
                        if tilt_counter >= Config.TILT_FRAMES_THRESHOLD:
                            is_drowsy = True
                    else:
                        tilt_counter = 0
                except Exception:
                    tilt_counter = 0 

                mouth_ratio = get_mouth_open_ratio(landmarks, proc.shape)
                mar_display = mouth_ratio # [NEW]
                
                if mouth_ratio > Config.YAWN_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter == Config.YAWN_FRAMES_THRESHOLD:
                        total_yawns += 1
                        is_yawning = True
                        is_drowsy = True
                else:
                    yawn_counter = 0

                # [REWORKED] Separated Gaze Logic
                
                # 1. Get raw values
                raw_yaw = get_head_yaw_ratio(landmarks)
                raw_gaze_y = get_vertical_gaze_ratio(landmarks, proc.shape) # This will be None if iris lost

                # 2. Apply smoothing
                if raw_yaw is None:
                    yaw_ema = 0.9 * yaw_ema + 0.1 * baseline_yaw_ratio
                else:
                    yaw_ema = GAZE_SMOOTH_ALPHA * raw_yaw + (1.0 - GAZE_SMOOTH_ALPHA) * yaw_ema
                
                # Only smooth vertical gaze if we have a valid signal
                if raw_gaze_y is not None:
                    gaze_y_ema = GAZE_SMOOTH_ALPHA * raw_gaze_y + (1.0 - GAZE_SMOOTH_ALPHA) * gaze_y_ema
                # If signal is lost, just keep the last valid 'gaze_y_ema'
                # (it will be ignored by the logic below anyway)

                # 3. Apply State-Based Logic
                candidate = gaze_candidate 

                if gaze == "CENTER":
                    # --- Check for ENTER state ---
                    if yaw_ema <= GAZE_LEFT_ENTER:
                        candidate = "LEFT"
                    elif yaw_ema >= GAZE_RIGHT_ENTER:
                        candidate = "RIGHT"
                    # Only check UP/DOWN if iris is valid
                    elif raw_gaze_y is not None and gaze_y_ema <= GAZE_UP_ENTER: 
                        candidate = "UP"
                    elif raw_gaze_y is not None and gaze_y_ema >= GAZE_DOWN_ENTER:
                        candidate = "DOWN"
                
                elif gaze == "UP":
                    # --- Check for EXIT state ---
                    # If iris is lost OR we move back to center, exit.
                    if raw_gaze_y is None or gaze_y_ema >= GAZE_UP_EXIT:
                        candidate = "CENTER"
                
                elif gaze == "DOWN":
                    # --- Check for EXIT state ---
                    if raw_gaze_y is None or gaze_y_ema <= GAZE_DOWN_EXIT:
                        candidate = "CENTER"
                
                elif gaze == "LEFT":
                     # --- Check for EXIT state ---
                     # Note: This is independent of iris.
                    if yaw_ema >= GAZE_LEFT_EXIT:
                        candidate = "CENTER"

                elif gaze == "RIGHT":
                     # --- Check for EXIT state ---
                     # Note: This is independent of iris.
                    if yaw_ema <= GAZE_RIGHT_EXIT:
                        candidate = "CENTER"
                

                # --- Debounce logic (no change) ---
                if candidate == gaze_candidate:
                    gaze_debounce_count += 1
                else:
                    gaze_candidate = candidate
                    gaze_debounce_count = 1

                if gaze_debounce_count >= GAZE_DEBOUNCE_FRAMES:
                    gaze = gaze_candidate
                
                # [REWORKED] Distraction logic - Now includes DOWN
                if gaze in ("LEFT", "RIGHT"):
                    gaze_lr_counter += 1
                    if gaze_lr_counter >= Config.DISTRACTION_LR_FRAMES_THRESHOLD:
                        is_distracted = True
                else:
                    gaze_lr_counter = 0
                
                if gaze == "UP":
                    gaze_up_counter += 1
                    if gaze_up_counter >= Config.DISTRACTION_UP_FRAMES_THRESHOLD:
                        is_distracted = True
                else:
                    gaze_up_counter = 0
                
                if gaze == "DOWN": # [NEW]
                    gaze_down_counter += 1
                    if gaze_down_counter >= Config.DISTRACTION_DOWN_FRAMES_THRESHOLD:
                        is_distracted = True
                else:
                    gaze_down_counter = 0
            
            else: 
                nod_counter = 0
                drop_counter = 0
                tilt_counter = 0
                gaze_lr_counter = 0 
                gaze_up_counter = 0 
                gaze_down_counter = 0 # [NEW]
                eye_history.clear()
                mp_ear_display = 0.0
                dlib_ear_display = 0.0
                mar_display = 0.0 # [NEW]

            if is_yawning:
                current_status = "Yawning"
            elif is_drowsy:
                current_status = "Drowsy"
            elif is_distracted: 
                current_status = "Distracted"
            else:
                current_status = "Awake"

            if current_status != "Awake" and current_status != last_log_status:
                log_event(log_path, model_file, current_status, gaze)
                if current_status in ("Drowsy", "Yawning", "Distracted"):
                    beep_alert()
                last_log_status = current_status
            elif current_status == "Awake":
                last_log_status = "" 

        # [FINAL FIX] Set gaze string for display, with CORRECTED priority
        
        # 1. Check for eye closure first. This is the top priority.
        #    (If eyes are shut, it's "CLOSED", period.)
        if is_closed:
            gaze_to_display = "CLOSED"
        
        # 2. If eyes are open, check for a head drop.
        elif is_head_drop:
            gaze_to_display = "DOWN"
        
        # 3. If eyes are open and head is not dropped, show the real gaze.
        else:
            gaze_to_display = gaze


        # [CHANGED] Pass mar_display and gaze_to_display
        draw_ui(frame, current_status, gaze_to_display, model_file, elapsed_time, total_yawns, mp_ear_display, dlib_ear_display, mar_display, use_dlib)
        
        # Write the *full-size* frame to the video file
        if out:
            out.write(frame)
        
        cv2.imshow("Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    # [FIX] Check if 'out' was initialized before releasing
    if out:
        out.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print(f"[INFO] Session video saved to: {video_filename}")
    print(f"[INFO] Session log saved to: {log_path}")
    print("[INFO] Application terminated.")


if __name__ == "__main__":
    main()