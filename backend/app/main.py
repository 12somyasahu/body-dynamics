import asyncio
import time
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.processing.pose_estimator import PoseEstimator
from app.processing.stats import (
    calculate_angle, compute_com, KalmanCOM, 
    StabilityTracker, PhaseTracker, SupportTracker
)
from app.realtime.frame_buffer import FrameBuffer

# =========================================================
# 1. SETUP & CONFIGURATION
# =========================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=2)

# SENSITIVITY TUNING
# We loosened these so the engine "feels" the ground more easily.
GROUND_EPS = 0.08        # Distance from lowest point to be "on floor"
FOOT_CONTACT_TIME = 0.05 # How fast the engine trusts the foot is down
MAX_TRAIL = 30           # Number of points in the COM trail

def process_frame_sync(frame_bytes, pose_estimator, kalman):
    try:
        if not frame_bytes: return None, None, None
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0: return None, None, None

        keypoints = pose_estimator.process(frame)
        if not keypoints: return None, None, None

        raw_com = compute_com(keypoints) #
        filtered_com = kalman.update(raw_com) if raw_com is not None else None
        
        angle = None
        if len(keypoints) > 15 and keypoints[11] and keypoints[13] and keypoints[15]:
            angle = calculate_angle(keypoints[11], keypoints[13], keypoints[15]) #

        return {"type": "pose", "keypoints": {"person_0": keypoints}}, filtered_com, angle
    except Exception as e:
        logger.error(f"Sync Processing Error: {e}")
        return None, None, None

# =========================================================
# 2. WEBSOCKET ENGINE
# =========================================================
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    fb, pe, kl = FrameBuffer(), PoseEstimator(), KalmanCOM()
    st_tracker, su_tracker, ph_tracker = StabilityTracker(), SupportTracker(), PhaseTracker()

    com_history = [] 
    frames, start_time = 0, time.time()
    foot_state = {"left": {"stable_since": None}, "right": {"stable_since": None}}

    try:
        while True:
            try:
                frame_bytes = await asyncio.wait_for(websocket.receive_bytes(), timeout=2.0)
                await fb.push(frame_bytes)
            except asyncio.TimeoutError: continue
            
            latest = await fb.pop()
            if not latest:
                await asyncio.sleep(0.001)
                continue

            # Process with Lag Protection
            loop = asyncio.get_event_loop()
            try:
                future = loop.run_in_executor(executor, process_frame_sync, latest, pe, kl)
                payload, f_com, angle = await asyncio.wait_for(future, timeout=0.2)
            except asyncio.TimeoutError: continue
            
            if not payload:
                kl.reset() #
                continue
            
            frames += 1
            keypoints = payload["keypoints"]["person_0"]

            # --- DYNAMIC GROUND DETECTION ---
            # [cite_start]Finds the lowest point in the current frame [cite: 203]
            foot_indices = [27, 28, 29, 30, 31, 32]
            current_ys = [kp["y"] for i, kp in enumerate(keypoints) if i in foot_indices and kp]
            ground_y = max(current_ys) if current_ys else None

            def is_grounded(side, ankle_idx):
                kp = keypoints[ankle_idx]
                if not kp or ground_y is None: return False
                # Is foot near the dynamic ground level?
                if abs(kp["y"] - ground_y) < GROUND_EPS:
                    if foot_state[side]["stable_since"] is None: 
                        foot_state[side]["stable_since"] = time.time()
                    return (time.time() - foot_state[side]["stable_since"]) > FOOT_CONTACT_TIME
                foot_state[side]["stable_since"] = None
                return False

            l_g, r_g = is_grounded("left", 27), is_grounded("right", 28)
            support_type = "double_foot" if (l_g and r_g) else "single_foot" if (l_g or r_g) else "none"
            
            # --- BASE OF SUPPORT (BOS) ---
            # [cite_start]Build the area between grounded feet [cite: 176]
            bos_pts = []
            if l_g: bos_pts.extend([(kp["x"], kp["y"]) for i, kp in enumerate(keypoints) if i in [27, 29, 31] and kp])
            if r_g: bos_pts.extend([(kp["x"], kp["y"]) for i, kp in enumerate(keypoints) if i in [28, 30, 32] and kp])
            
            # Update trackers from stats.py
            support = su_tracker.update(support_type, bos_pts)
            # Stability now has a real BOS to compare against
            stab_state, margin = st_tracker.update(f_com[0] if f_com is not None else None, support["polygon"] if support else None)
            
            # [cite_start]Update COM Trail [cite: 201]
            if f_com is not None:
                com_history.append({"x": round(float(f_com[0]), 3), "y": round(float(f_com[1]), 3)})
                if len(com_history) > MAX_TRAIL: com_history.pop(0)

            # Send full biomechanical packet
            await websocket.send_json({
                "type": "stats",
                "fps": round(frames / (max(time.time() - start_time, 1)), 1),
                "keypoints": {"person_0": keypoints},
                "com_trail": com_history,
                "stability": {"state": stab_state, "margin": margin},
                "phase": ph_tracker.update(support_type, stab_state, 0),
                "support": support_type,
                "joint_angles": {"left_elbow": angle}
            })

    except WebSocketDisconnect: logger.info("Client disconnected")
    finally:
        pe.close() #
        kl.reset() #