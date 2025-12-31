import asyncio
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.processing.pose_estimator import PoseEstimator
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

executor = ThreadPoolExecutor(max_workers=1)


@app.get("/")
async def health():
    return {"status": "ok"}


# -----------------------------
# CPU-heavy inference (SYNC)
# -----------------------------
def process_frame_sync(frame_bytes, pose_estimator, kalman):
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, None

    keypoints = pose_estimator.process(frame)
    if not keypoints:
        return None, None, None

    left_elbow_angle = None
    if len(keypoints) > 15:
        left_elbow_angle = calculate_angle(
            keypoints[11],
            keypoints[13],
            keypoints[15],
        )

    raw_com = compute_com(keypoints)
    filtered_com = kalman.update(raw_com) if raw_com else None

    pose_payload = {
        "type": "pose",
        "keypoints": {"person_0": keypoints}
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

    # ---- session state ----
    prev_com = None
    prev_time = None
    frames = 0
    start_time = time.time()

    movement_active = False
    last_weight_shift = "centered"
    weight_hold_start = None

    # ---- thresholds ----
    WEIGHT_THRESHOLD = 0.03
    MAX_COM_SPEED = 0.8
    MOVE_START_SPEED = 0.15
    MOVE_STOP_SPEED = 0.05
    WEIGHT_HOLD_TIME = 0.4

    async def receive_frames():
        while True:
            frame = await websocket.receive_bytes()
            await frame_buffer.push(frame)

    async def process_frames():
        nonlocal prev_com, prev_time, frames
        nonlocal movement_active, last_weight_shift, weight_hold_start

        while True:
            event = None

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

            keypoints = pose_payload["keypoints"]["person_0"]

            # -----------------------------
            # Velocity
            # -----------------------------
            velocity = None
            speed = None
            now = time.time()

            if filtered_com and prev_com and prev_time:
                dt = now - prev_time
                if dt > 0:
                    vx = (filtered_com[0] - prev_com[0]) / dt
                    vy = (filtered_com[1] - prev_com[1]) / dt
                    velocity = (vx, vy)
                    speed = (vx ** 2 + vy ** 2) ** 0.5

            prev_com = filtered_com
            prev_time = now

            # -----------------------------
            # Weight transfer
            # -----------------------------
            weight_shift = "centered"

            if filtered_com and len(keypoints) > 24:
                left_hip = keypoints[23]
                right_hip = keypoints[24]
                hip_center_x = (left_hip[0] + right_hip[0]) / 2
                offset_x = filtered_com[0] - hip_center_x

                if offset_x < -WEIGHT_THRESHOLD:
                    weight_shift = "left"
                elif offset_x > WEIGHT_THRESHOLD:
                    weight_shift = "right"

                if velocity:
                    vx = velocity[0]
                    if weight_shift == "left" and vx > 0:
                        weight_shift = "centered"
                    if weight_shift == "right" and vx < 0:
                        weight_shift = "centered"

            # -----------------------------
            # Weight shift event
            # -----------------------------
            if weight_shift != "centered":
                if weight_shift != last_weight_shift:
                    weight_hold_start = now
                    last_weight_shift = weight_shift
                else:
                    if weight_hold_start and (now - weight_hold_start) > WEIGHT_HOLD_TIME:
                        event = f"weight_shift_{weight_shift}"
            else:
                last_weight_shift = "centered"
                weight_hold_start = None

            # -----------------------------
            # Stability
            # -----------------------------
            stability = None
            if speed is not None:
                stability = 100 - (speed / MAX_COM_SPEED) * 100
                stability = round(max(0, min(100, stability)), 1)

            # -----------------------------
            # Movement events
            # -----------------------------
            if speed is not None:
                if not movement_active and speed > MOVE_START_SPEED:
                    movement_active = True
                    event = "movement_started"

                elif movement_active and speed < MOVE_STOP_SPEED:
                    movement_active = False
                    event = "movement_stopped"

            # -----------------------------
            # Send pose
            # -----------------------------
            await websocket.send_json(pose_payload)

            # -----------------------------
            # Send stats
            # -----------------------------
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
                    "com_velocity": [round(velocity[0], 3), round(velocity[1], 3)] if velocity else None,
                    "weight_shift": weight_shift,
                    "stability": stability,
                    "event": event
                })

    try:
        await asyncio.gather(
            receive_frames(),
            process_frames()
        )
    except WebSocketDisconnect:
        print("WebSocket disconnected")
