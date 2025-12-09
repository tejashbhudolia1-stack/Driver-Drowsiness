#!/usr/bin/env python3

"""
calibrate.py — Captures and saves personalized calibration data.

This script captures:
1.  'Look Center' (Baseline Gaze Y, Open EAR)
2.  'Look Up' (Max Gaze Y)
3.  'Look Down' (Min Gaze Y)
4.  'Eyes Closed' (Closed EAR)
"""

import os
import time
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Try to import dlib (optional)
try:
    import dlib
    DLIB_AVAILABLE = True
except Exception:
    DLIB_AVAILABLE = False

# --- CONFIG (Mirrors detect.py) ---
class Config:
    MODELS_DIR = "models"
    DLIB_SHAPE_MODEL = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")

    # Mediapipe landmark indices
    LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 153, 154, 155]
    RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 380, 381, 382]
    LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]
    RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]
    
    PROCESS_WIDTH = 640
    CALIBRATION_TIME_S = 3.5 # Seconds per step

class UI:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    COLOR_INFO = (255, 255, 0)
    COLOR_WARN = (0, 165, 255)
    COLOR_ALERT = (0, 0, 255)

# --- HELPER FUNCTIONS (Copied from detect.py) ---

def resize_for_processing(frame):
    h, w = frame.shape[:2]
    if w <= Config.PROCESS_WIDTH:
        return frame.copy()
    new_w = Config.PROCESS_WIDTH
    new_h = int(h * (new_w / w))
    return cv2.resize(frame, (new_w, new_h))

def get_eye_aspect_ratio_mp(landmarks, indices, frame_shape):
    h, w, _ = frame_shape
    if not landmarks: return 0.0
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    if pts.shape[0] < 9: return 0.0
    top = np.mean([pts[1][1], pts[2][1], pts[3][1]])
    bottom = np.mean([pts[4][1], pts[5][1], pts[6][1]])
    vertical = abs(bottom - top)
    horizontal = abs(pts[0][0] - pts[8][0]) 
    return vertical / horizontal if horizontal > 0 else 0.0

def get_vertical_gaze_ratio(landmarks, frame_shape):
    h, w, _ = frame_shape
    try:
        left_iris = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in Config.LEFT_IRIS_INDICES], axis=0)
        right_iris = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in Config.RIGHT_IRIS_INDICES], axis=0)
        left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in Config.LEFT_EYE_INDICES])
        right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in Config.RIGHT_EYE_INDICES])
        l_min_y, l_max_y = left_eye[:, 1].min(), left_eye[:, 1].max()
        r_min_y, r_max_y = right_eye[:, 1].min(), right_eye[:, 1].max()
        l_y = (left_iris[1] - l_min_y) / max(1, (l_max_y - l_min_y))
        r_y = (right_iris[1] - r_min_y) / max(1, (r_max_y - r_min_y))
        gy = (l_y + r_y) / 2.0
        return gy
    except Exception:
        return None

def compute_dlib_ear_from_shape_np(shape_np):
    left_idx = list(range(36, 42))  
    right_idx = list(range(42, 48)) 
    left_eye, right_eye = shape_np[left_idx], shape_np[right_idx]
    def ear(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0
    return (ear(left_eye) + ear(right_eye)) / 2.0

def show_instructions(frame, text, color=UI.COLOR_INFO):
    """Utility to draw text on the frame for instructions."""
    (tw, th), _ = cv2.getTextSize(text, UI.FONT, 1.0, 2)
    x = (frame.shape[1] - tw) // 2
    y = 60
    cv2.putText(frame, text, (x, y), UI.FONT, 1.0, color, 2)
    return frame

# --- MAIN CALIBRATION ---

def main():
    # --- NEW BLOCK START ---
    # Show instructions and wait for the user to start
    print(f"""
    -----------------------------------------------------
       Welcome to the Drowsiness Detection Calibration
    -----------------------------------------------------

    This script will guide you through 4 steps to create
    a personalized profile for accurate detection.

    [INFO] Please ensure you are in a well-lit room and
           facing your camera directly.

    The steps are:
    1.  Look CENTER
    2.  Look UP
    3.  Look DOWN
    4.  CLOSE Your Eyes

    Each step will take {Config.CALIBRATION_TIME_S} seconds.
    """)
    
    try:
        input("==> Press ENTER to begin calibration... ")
    except KeyboardInterrupt:
        print("\n\n[INFO] Calibration cancelled by user.")
        return
    
    print("\n[INFO] Starting calibration...")
    # --- NEW BLOCK END ---
    
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    
    username = input("Enter a username for this profile (e.g., 'john_doe'): ").strip().lower()
    if not username:
        print("Invalid username.")
        return
    model_path = os.path.join(Config.MODELS_DIR, f"{username}.pkl")
    if os.path.exists(model_path):
        if input(f"[WARN] {model_path} already exists. Overwrite? (y/n): ").lower() != 'y':
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    use_dlib = False
    dlib_detector = None
    dlib_predictor = None
    if DLIB_AVAILABLE and os.path.exists(Config.DLIB_SHAPE_MODEL):
        dlib_detector = dlib.get_frontal_face_detector()
        dlib_predictor = dlib.shape_predictor(Config.DLIB_SHAPE_MODEL)
        use_dlib = True
        print("[INFO] Dlib loaded.")
    else:
        print("[WARN] Dlib not found or model missing. Calibrating Mediapipe-only.")

    model_data = {}
    
    # --- Step 1: Look Center (and Eyes Open EAR) ---
    print(f"\n[STEP 1/4] Calibrating 'Center'. Please look directly at the camera for {Config.CALIBRATION_TIME_S}s.")
    gaze_y_vals, mp_open_ears, dlib_open_ears = [], [], []
    t0 = time.time()
    while time.time() - t0 < Config.CALIBRATION_TIME_S:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        small = resize_for_processing(frame)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Gaze
            gy = get_vertical_gaze_ratio(landmarks, small.shape)
            if gy is not None: gaze_y_vals.append(gy)
            
            # MP EAR
            mp_ear = (get_eye_aspect_ratio_mp(landmarks, Config.LEFT_EYE_INDICES, small.shape) +
                      get_eye_aspect_ratio_mp(landmarks, Config.RIGHT_EYE_INDICES, small.shape)) / 2.0
            mp_open_ears.append(mp_ear)
            
            # Dlib EAR
            if use_dlib:
                rects = dlib_detector(gray, 0)
                if len(rects) > 0:
                    shape = dlib_predictor(gray, rects[0])
                    shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                    dlib_open_ears.append(compute_dlib_ear_from_shape_np(shape_np))
        
        frame = show_instructions(frame, "LOOK CENTER")
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27: cap.release(); cv2.destroyAllWindows(); return
    
    model_data["baseline_gaze_y"] = np.median(gaze_y_vals)
    model_data["eye_open_ratio"] = np.median(mp_open_ears) # MP Open
    if dlib_open_ears: model_data["eye_open_EAR"] = np.median(dlib_open_ears) # Dlib Open

    # --- Step 2: Look Up ---
    print(f"\n[STEP 2/4] Calibrating 'Up'. Please look UP for {Config.CALIBRATION_TIME_S}s.")
    gaze_y_vals = []
    t0 = time.time()
    while time.time() - t0 < Config.CALIBRATION_TIME_S:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        small = resize_for_processing(frame)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            gy = get_vertical_gaze_ratio(landmarks, small.shape)
            if gy is not None: gaze_y_vals.append(gy)
        
        frame = show_instructions(frame, "LOOK UP", UI.COLOR_WARN)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27: cap.release(); cv2.destroyAllWindows(); return
    
    model_data["gaze_up_val"] = np.median(gaze_y_vals)

    # --- Step 3: Look Down ---
    print(f"\n[STEP 3/4] Calibrating 'Down'. Please look DOWN for {Config.CALIBRATION_TIME_S}s.")
    gaze_y_vals = []
    t0 = time.time()
    while time.time() - t0 < Config.CALIBRATION_TIME_S:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        small = resize_for_processing(frame)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            gy = get_vertical_gaze_ratio(landmarks, small.shape)
            if gy is not None: gaze_y_vals.append(gy)
        
        frame = show_instructions(frame, "LOOK DOWN", UI.COLOR_WARN)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27: cap.release(); cv2.destroyAllWindows(); return
    
    model_data["gaze_down_val"] = np.median(gaze_y_vals)

    # --- Step 4: Eyes Closed ---
    print(f"\n[STEP 4/4] Calibrating 'Closed'. Please CLOSE YOUR EYES for {Config.CALIBRATION_TIME_S}s.")
    mp_closed_ears, dlib_closed_ears = [], []
    t0 = time.time()
    while time.time() - t0 < Config.CALIBRATION_TIME_S:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        small = resize_for_processing(frame)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # MP EAR
            mp_ear = (get_eye_aspect_ratio_mp(landmarks, Config.LEFT_EYE_INDICES, small.shape) +
                      get_eye_aspect_ratio_mp(landmarks, Config.RIGHT_EYE_INDICES, small.shape)) / 2.0
            mp_closed_ears.append(mp_ear)
            
            # Dlib EAR
            if use_dlib:
                rects = dlib_detector(gray, 0)
                if len(rects) > 0:
                    shape = dlib_predictor(gray, rects[0])
                    shape_np = np.array([[p.x, p.y] for p in shape.parts()])
                    dlib_closed_ears.append(compute_dlib_ear_from_shape_np(shape_np))
        
        frame = show_instructions(frame, "CLOSE YOUR EYES", UI.COLOR_ALERT)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == 27: cap.release(); cv2.destroyAllWindows(); return
    
    model_data["eye_closed_ratio"] = np.median(mp_closed_ears) # MP Closed
    if dlib_closed_ears: model_data["eye_closed_EAR"] = np.median(dlib_closed_ears) # Dlib Closed

    # --- Save Data ---
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

    if any(v is None for v in model_data.values()):
        print("\n[ERROR] Calibration failed. Some values could not be read.")
        print("Please ensure your face is well-lit and not obscured.")
        return

    joblib.dump(model_data, model_path)
    print(f"\n[SUCCESS] Calibration complete! Data saved to {model_path}")
    print("--- CALIBRATION RESULTS ---")
    print(f"  Gaze (Up/Center/Down): {model_data['gaze_up_val']:.3f} / {model_data['baseline_gaze_y']:.3f} / {model_data['gaze_down_val']:.3f}")
    print(f"  MP EAR (Open/Closed):  {model_data['eye_open_ratio']:.3f} / {model_data['eye_closed_ratio']:.3f}")
    if "eye_open_EAR" in model_data:
        print(f"  Dlib EAR (Open/Closed): {model_data['eye_open_EAR']:.3f} / {model_data['eye_closed_EAR']:.3f}")
    print("---------------------------")


if __name__ == "__main__":
    main()