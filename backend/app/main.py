import asyncio
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.processing.pose_estimator import PoseEstimatorStub as PoseEstimator
from app.processing.stats import calculate_angle, compute_com, KalmanCOM
from app.realtime.frame_buffer import FrameBuffer


# -----------------------------
# App setup
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One worker for CPU-bound inference
executor = ThreadPoolExecutor(max_workers=1)


@app.get("/")
async def health():
    return {"status": "ok"}


# -----------------------------
# CPU-heavy work (SYNC)
# -----------------------------
def process_frame_sync(frame_bytes, pose_estimator, kalman):
    # Decode JPEG
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, None

    # Pose estimation
    keypoints = pose_estimator.process(frame)
    if not keypoints:
        return None, None, None

    # Left elbow angle
    left_elbow_angle = None
    if len(keypoints) > 15:
        left_elbow_angle = calculate_angle(
            keypoints[11],  # left shoulder
            keypoints[13],  # left elbow
            keypoints[15],  # left wrist
        )

    # COM + Kalman
    raw_com = compute_com(keypoints)
    filtered_com = kalman.update(raw_com) if raw_com else None

    pose_payload = {
        "type": "pose",
        "keypoints": {
            "person_0": keypoints
        }
    }

    return pose_payload, filtered_com, left_elbow_angle


# -----------------------------
# WebSocket
# -----------------------------
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()

    frame_buffer = FrameBuffer()
    pose_estimator = PoseEstimator()
    kalman = KalmanCOM()

    loop = asyncio.get_event_loop()

    # ---- STATE (belongs here) ----
    prev_com = None
    prev_time = None
    frames = 0
    start_time = time.time()

    async def receive_frames():
        while True:
            frame = await websocket.receive_bytes()
            await frame_buffer.push(frame)

    async def process_frames():
        nonlocal prev_com, prev_time, frames

        while True:
            frame = await frame_buffer.pop()
            if frame is None:
                await asyncio.sleep(0.001)
                continue

            frames += 1

            pose_payload, filtered_com, left_elbow_angle = await loop.run_in_executor(
                executor,
                process_frame_sync,
                frame,
                pose_estimator,
                kalman
            )

            if not pose_payload:
                continue

            # --------------------
            # Velocity (HERE, not in worker)
            # --------------------
            velocity = None
            now = time.time()

            if filtered_com and prev_com and prev_time:
                dt = now - prev_time
                if dt > 0:
                    vx = (filtered_com[0] - prev_com[0]) / dt
                    vy = (filtered_com[1] - prev_com[1]) / dt
                    velocity = (vx, vy)

            prev_com = filtered_com
            prev_time = now

            # --------------------
            # Send pose
            # --------------------
            await websocket.send_json(pose_payload)

            # --------------------
            # Send stats (every 10 frames)
            # --------------------
            if frames % 10 == 0:
                elapsed = time.time() - start_time
                fps = frames / elapsed if elapsed > 0 else 0.0

                await websocket.send_json({
                    "type": "stats",
                    "frames_received": frames,
                    "uptime_sec": round(elapsed, 2),
                    "input_fps": round(fps, 2),
                    "left_elbow_angle": round(left_elbow_angle, 1) if left_elbow_angle else None,
                    "com": [round(filtered_com[0], 3), round(filtered_com[1], 3)] if filtered_com else None,
                    "com_velocity": [round(velocity[0], 3), round(velocity[1], 3)] if velocity else None
                })

    try:
        await asyncio.gather(
            receive_frames(),
            process_frames()
        )
    except WebSocketDisconnect:
        print("WebSocket disconnected")
