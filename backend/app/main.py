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
from app.processing.stats import calculate_angle, compute_com, KalmanCOM
from app.realtime.frame_buffer import FrameBuffer
from app.engine.motion_engine import MotionEngine
from app.config import config


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
        self.processing_times = deque(
            maxlen=config.get('metrics', 'performance_window', default=100)
        )
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
    return {
        "app": metrics.get_stats(),
        "executor": {
            "queue_size": executor._work_queue.qsize(),
            "active_workers": len(
                [t for t in executor._threads if t.is_alive()]
            ),
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
    return config.to_dict()


# =========================================================
# FRAME PROCESSING (UNCHANGED)
# =========================================================
def process_frame_sync(frame_bytes, pose_estimator, kalman):
    start_time = time.time()

    try:
        if not frame_bytes:
            return None, None, None

        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return None, None, None

        keypoints = pose_estimator.process(frame)
        if not keypoints:
            return None, None, None

        raw_com = compute_com(keypoints)
        filtered_com = kalman.update(raw_com) if raw_com is not None else None

        angle = None
        if (
            len(keypoints) > 15 and
            keypoints[11] and keypoints[13] and keypoints[15]
        ):
            angle = calculate_angle(
                keypoints[11],
                keypoints[13],
                keypoints[15]
            )

        processing_time = (time.time() - start_time) * 1000
        metrics.record_frame(processing_time)

        return {
            "type": "pose",
            "keypoints": {"person_0": keypoints}
        }, filtered_com, angle

    except Exception as e:
        logger.error(f"Frame processing error: {e}", exc_info=True)
        metrics.record_error()
        return None, None, None


# =========================================================
# HELPER FUNCTIONS (GROUND ONLY)
# =========================================================
def detect_ground_contact(keypoints, foot_state, ground_y):
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

            return (now - foot_state[side]["stable_since"]) > config.foot_contact_time

        foot_state[side]["stable_since"] = None
        return False

    return is_grounded("left", 27), is_grounded("right", 28)


def build_base_of_support(keypoints, left_grounded, right_grounded):
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

    support_type = (
        "double_foot" if (left_grounded and right_grounded)
        else "single_foot"
    )

    return support_type, bos_pts


# =========================================================
# WEBSOCKET
# =========================================================
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    metrics.active_sessions += 1

    fb = FrameBuffer()
    pe = PoseEstimator()
    kl = KalmanCOM()
    engine = MotionEngine()

    frames = 0
    start_time = time.time()
    foot_state = {"left": {"stable_since": None}, "right": {"stable_since": None}}

    loop = asyncio.get_event_loop()

    async def receive_frames():
        try:
            while True:
                frame_bytes = await websocket.receive_bytes()
                await fb.push(frame_bytes)
        except WebSocketDisconnect:
            lo
