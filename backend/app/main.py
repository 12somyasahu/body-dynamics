import asyncio
import time
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.processing.pose_estimator import PoseEstimator
from app.processing.stats import (
    calculate_angle, compute_com, KalmanCOM, 
    StabilityTracker, PhaseTracker, SupportTracker
)
from app.realtime.frame_buffer import FrameBuffer
from app.config import config  # Import centralized config

# =========================================================
# LOGGING SETUP
# =========================================================
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format=config.log_format
)
logger = logging.getLogger(__name__)

# =========================================================
# APP SETUP
# =========================================================
app = FastAPI(
    title=config.get('app', 'name', default="Body-Dynamics"),
    version=config.get('app', 'version', default="1.0.0")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

executor = ThreadPoolExecutor(max_workers=config.max_workers)

# =========================================================
# GLOBAL METRICS
# =========================================================
class Metrics:
    def __init__(self):
        self.total_frames = 0
        self.active_sessions = 0
        self.processing_times = deque(maxlen=config.get('metrics', 'performance_window', default=100))
        self.errors = 0
    
    def record_frame(self, duration_ms):
        self.total_frames += 1
        self.processing_times.append(duration_ms)
    
    def record_error(self):
        self.errors += 1
    
    def get_stats(self):
        if not self.processing_times:
            return {
                "total_frames": self.total_frames,
                "active_sessions": self.active_sessions,
                "avg_processing_ms": 0,
                "errors": self.errors
            }
        
        times = list(self.processing_times)
        return {
            "total_frames": self.total_frames,
            "active_sessions": self.active_sessions,
            "avg_processing_ms": round(sum(times) / len(times), 2),
            "p95_processing_ms": round(np.percentile(times, 95), 2),
            "p99_processing_ms": round(np.percentile(times, 99), 2),
            "errors": self.errors
        }

metrics = Metrics()

# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": config.get('app', 'name'),
        "version": config.get('app', 'version')
    }

@app.get("/metrics")
async def get_metrics():
    """Performance monitoring endpoint"""
    return {
        "app": metrics.get_stats(),
        "executor": {
            "queue_size": executor._work_queue.qsize(),
            "active_workers": len([t for t in executor._threads if t.is_alive()]),
            "max_workers": executor._max_workers
        },
        "config": {
            "ground_epsilon": config.ground_epsilon,
            "foot_contact_time": config.foot_contact_time,
            "max_trail_length": config.max_trail
        }
    }

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return config.to_dict()

# =========================================================
# FRAME PROCESSING
# =========================================================
def process_frame_sync(frame_bytes, pose_estimator, kalman):
    """
    CPU-bound frame processing (runs in thread pool).
    Returns: (pose_payload, filtered_com, angle) or (None, None, None)
    """
    start_time = time.time()
    
    try:
        if not frame_bytes:
            return None, None, None
        
        # Decode frame
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None or frame.size == 0:
            return None, None, None

        # Pose estimation
        keypoints = pose_estimator.process(frame)
        if not keypoints:
            return None, None, None

        # COM calculation
        raw_com = compute_com(keypoints)
        filtered_com = kalman.update(raw_com) if raw_com is not None else None
        
        # Joint angle
        angle = None
        if len(keypoints) > 15 and keypoints[11] and keypoints[13] and keypoints[15]:
            angle = calculate_angle(keypoints[11], keypoints[13], keypoints[15])

        # Track performance
        processing_time = (time.time() - start_time) * 1000
        metrics.record_frame(processing_time)
        
        if processing_time > 100:
            logger.warning(f"Slow frame: {processing_time:.1f}ms")

        return {
            "type": "pose",
            "keypoints": {"person_0": keypoints}
        }, filtered_com, angle
    
    except Exception as e:
        logger.error(f"Frame processing error: {e}", exc_info=True)
        metrics.record_error()
        return None, None, None

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def detect_ground_contact(keypoints, foot_state, ground_y):
    """Detect which feet are touching ground."""
    if ground_y is None:
        return False, False
    
    now = time.time()
    
    def is_grounded(side, ankle_idx):
        if ankle_idx >= len(keypoints) or not keypoints[ankle_idx]:
            foot_state[side]["stable_since"] = None
            return False
        
        kp = keypoints[ankle_idx]
        if abs(kp["y"] - ground_y) < config.ground_epsilon:
            if foot_state[side]["stable_since"] is None:
                foot_state[side]["stable_since"] = now
            
            time_stable = now - foot_state[side]["stable_since"]
            return time_stable > config.foot_contact_time
        
        foot_state[side]["stable_since"] = None
        return False
    
    return is_grounded("left", 27), is_grounded("right", 28)

def build_base_of_support(keypoints, left_grounded, right_grounded):
    """Build BOS polygon from grounded feet."""
    bos_pts = []
    
    if left_grounded:
        for idx in [27, 29, 31]:
            if idx < len(keypoints) and keypoints[idx]:
                bos_pts.append((keypoints[idx]["x"], keypoints[idx]["y"]))
    
    if right_grounded:
        for idx in [28, 30, 32]:
            if idx < len(keypoints) and keypoints[idx]:
                bos_pts.append((keypoints[idx]["x"], keypoints[idx]["y"]))
    
    if len(bos_pts) < 2:
        return "none", None
    
    support_type = "double_foot" if (left_grounded and right_grounded) else "single_foot"
    return support_type, bos_pts

def interpret_joint_angle(angle):
    """Convert angle to semantic state."""
    if angle is None:
        return None
    
    flexion_threshold = config.get('biomechanics', 'joints', 'elbow', 'flexion_threshold', default=140)
    extension_threshold = config.get('biomechanics', 'joints', 'elbow', 'extension_threshold', default=160)
    
    if angle < flexion_threshold:
        return "flexion"
    elif angle > extension_threshold:
        return "extension"
    else:
        return "transition"

# =========================================================
# WEBSOCKET
# =========================================================
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    metrics.active_sessions += 1
    
    # Initialize components
    fb = FrameBuffer()
    pe = PoseEstimator()
    kl = KalmanCOM()
    st_tracker = StabilityTracker()
    su_tracker = SupportTracker()
    ph_tracker = PhaseTracker()
    
    # State
    com_history = []
    frames = 0
    start_time = time.time()
    foot_state = {"left": {"stable_since": None}, "right": {"stable_since": None}}
    
    loop = asyncio.get_event_loop()
    
    async def receive_frames():
        try:
            while True:
                try:
                    frame_bytes = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=2.0
                    )
                    await fb.push(frame_bytes)
                except asyncio.TimeoutError:
                    continue
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Receive error: {e}")
    
    async def process_frames():
        nonlocal frames, com_history
        
        try:
            while True:
                latest = await fb.pop()
                if not latest:
                    await asyncio.sleep(0.001)
                    continue
                
                frames += 1
                
                # Process with timeout
                try:
                    future = loop.run_in_executor(executor, process_frame_sync, latest, pe, kl)
                    payload, f_com, angle = await asyncio.wait_for(
                        future,
                        timeout=config.processing_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Frame {frames} timeout")
                    continue
                
                if not payload:
                    kl.reset()
                    continue
                
                keypoints = payload["keypoints"]["person_0"]
                
                # Ground detection
                foot_indices = [27, 28, 29, 30, 31, 32]
                current_ys = [
                    keypoints[i]["y"] 
                    for i in foot_indices 
                    if i < len(keypoints) and keypoints[i]
                ]
                ground_y = max(current_ys) if current_ys else None
                
                # Foot contact
                l_g, r_g = detect_ground_contact(keypoints, foot_state, ground_y)
                
                # Base of support
                raw_support, raw_bos = build_base_of_support(keypoints, l_g, r_g)
                support = su_tracker.update(raw_support, raw_bos)
                support_type = support["support"] if support else None
                bos_polygon = support["polygon"] if support else None
                
                # Stability
                stab_state, margin = None, None
                if f_com is not None and bos_polygon:
                    stab_state, margin = st_tracker.update(float(f_com[0]), bos_polygon)
                
                # Phase
                phase = ph_tracker.update(support_type, stab_state, 0)
                
                # COM trail
                if f_com is not None:
                    com_history.append({
                        "x": round(float(f_com[0]), 3),
                        "y": round(float(f_com[1]), 3),
                        "t": round(time.time(), 3)
                    })
                    if len(com_history) > config.max_trail:
                        com_history.pop(0)
                
                # Joint interpretation
                elbow_state = interpret_joint_angle(angle)
                
                # Send data
                try:
                    await websocket.send_json(payload)
                    
                    if frames % config.stats_interval == 0:
                        elapsed = time.time() - start_time
                        fps = frames / elapsed if elapsed > 0 else 0.0
                        
                        await websocket.send_json({
                            "type": "stats",
                            "fps": round(fps, 1),
                            "frames": frames,
                            "keypoints": {"person_0": keypoints},
                            "com_trail": com_history,
                            "stability": {
                                "state": stab_state,
                                "margin": round(margin, 3) if margin else None
                            } if stab_state else None,
                            "phase": phase,
                            "support": support_type,
                            "joint_angles": {
                                "left_elbow": round(angle, 1) if angle else None,
                                "left_elbow_state": elbow_state
                            }
                        })
                except Exception as e:
                    logger.error(f"Send error: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Process error: {e}", exc_info=True)
    
    try:
        await asyncio.gather(receive_frames(), process_frames())
    except WebSocketDisconnect:
        logger.info("Disconnected")
    finally:
        metrics.active_sessions -= 1
        pe.close()
        kl.reset()
        logger.info(f"Session ended: {frames} frames in {time.time()-start_time:.1f}s")

# =========================================================
# LIFECYCLE
# =========================================================
@app.on_event("startup")
async def startup():
    logger.info("Body-Dynamics Engine Starting...")
    logger.info(f"Environment: {config.get('app', 'environment')}")
    logger.info(f"Max workers: {config.max_workers}")
    logger.info(f"Config: GROUND_EPS={config.ground_epsilon}, FOOT_TIME={config.foot_contact_time}")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down...")
    executor.shutdown(wait=True)
