import asyncio
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.processing.pose_estimator import PoseEstimator
from app.processing.stats import (
    calculate_angle,
    compute_com,
    KalmanCOM,
    StabilityTracker
)
from app.realtime.frame_buffer import FrameBuffer


# =========================================================
# App setup setup of app 
# =========================================================
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


# =========================================================
# CPU-heavy inference (SYNC)
# =========================================================
def process_frame_sync(frame_bytes, pose_estimator, kalman):
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, None

    keypoints = pose_estimator.process(frame)
    if not keypoints:
        return None, None, None

    # Left elbow angle (guarded)
    left_elbow_angle = None
    if (
        len(keypoints) > 15 and
        keypoints[11] and keypoints[13] and keypoints[15]
    ):
        left_elbow_angle = calculate_angle(
            keypoints[11],
            keypoints[13],
            keypoints[15],
        )

    raw_com = compute_com(keypoints)
    filtered_com = kalman.update(raw_com) if raw_com is not None else None

    return {
        "type": "pose",
        "keypoints": {"person_0": keypoints}
    }, filtered_com, left_elbow_angle


# =========================================================
# WebSocket
# =========================================================
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()

    frame_buffer = FrameBuffer()
    pose_estimator = PoseEstimator()
    kalman = KalmanCOM()
    stability_tracker = StabilityTracker()

    loop = asyncio.get_event_loop()

    prev_com = None
    prev_time = None
    frames = 0
    start_time = time.time()
    movement_active = False

    # Foot contact state
    foot_state = {
        "left":  {"y": None, "vy": None, "stable_since": None, "grounded": False},
        "right": {"y": None, "vy": None, "stable_since": None, "grounded": False},
    }

    MOVE_START_SPEED = 0.15
    MOVE_STOP_SPEED = 0.05

    FOOT_VY_EPS = 0.015
    FOOT_CONTACT_TIME = 0.25
    GROUND_EPS = 0.025

    async def receive_frames():
        while True:
            frame = await websocket.receive_bytes()
            await frame_buffer.push(frame)

    async def process_frames():
        nonlocal prev_com, prev_time, frames, movement_active

        while True:
            frame = await frame_buffer.pop()
            if frame is None:
                await asyncio.sleep(0.001)
                continue

            frames += 1
            now = time.time()

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

            # -------------------------------------------------
            # COM motion (for movement events only)
            # -------------------------------------------------
            speed = None
            if filtered_com is not None and prev_com is not None and prev_time is not None:
                dt = now - prev_time
                if dt > 0:
                    vx = float(filtered_com[0] - prev_com[0]) / dt
                    vy = float(filtered_com[1] - prev_com[1]) / dt
                    speed = (vx**2 + vy**2) ** 0.5

            prev_com = filtered_com
            prev_time = now

            motion_event = None
            if speed is not None:
                if not movement_active and speed > MOVE_START_SPEED:
                    movement_active = True
                    motion_event = "movement_started"
                elif movement_active and speed < MOVE_STOP_SPEED:
                    movement_active = False
                    motion_event = "movement_stopped"

            # -------------------------------------------------
            # GLOBAL GROUND ESTIMATION
            # -------------------------------------------------
            foot_ys = []
            for idx in [27, 28, 29, 30, 31, 32]:
                if idx < len(keypoints) and keypoints[idx]:
                    foot_ys.append(float(keypoints[idx]["y"]))

            global_ground_y = max(foot_ys) if foot_ys else None

            # -------------------------------------------------
            # FOOT CONTACT DETECTION
            # -------------------------------------------------
            def update_foot(side, ankle_idx):
                state = foot_state[side]
                if (
                    ankle_idx >= len(keypoints) or
                    not keypoints[ankle_idx] or
                    global_ground_y is None
                ):
                    state["grounded"] = False
                    state["stable_since"] = None
                    return

                y = float(keypoints[ankle_idx]["y"])
                vy = y - state["y"] if state["y"] is not None else 0.0
                state["y"] = y
                state["vy"] = vy

                near_ground = abs(y - global_ground_y) < GROUND_EPS

                if abs(vy) < FOOT_VY_EPS and near_ground:
                    if state["stable_since"] is None:
                        state["stable_since"] = now
                    elif now - state["stable_since"] > FOOT_CONTACT_TIME:
                        state["grounded"] = True
                else:
                    state["stable_since"] = None
                    state["grounded"] = False

            update_foot("left", 27)
            update_foot("right", 28)

            left_grounded = foot_state["left"]["grounded"]
            right_grounded = foot_state["right"]["grounded"]

            # -------------------------------------------------
            # BASE OF SUPPORT (polygon)
            # -------------------------------------------------
            bos_polygon = None
            support_type = None

            def kp(i):
                return (
                    float(keypoints[i]["x"]),
                    float(keypoints[i]["y"])
                )

            if left_grounded or right_grounded:
                pts = []

                if left_grounded and len(keypoints) > 32:
                    if keypoints[29] and keypoints[31]:
                        pts.extend([kp(29), kp(31)])

                if right_grounded and len(keypoints) > 32:
                    if keypoints[32] and keypoints[30]:
                        pts.extend([kp(32), kp(30)])

                if len(pts) >= 2:
                    bos_polygon = pts
                    support_type = (
                        "double_foot" if left_grounded and right_grounded
                        else "single_foot"
                    )

            # -------------------------------------------------
            # STABILITY (delegated to stats.py)
            # -------------------------------------------------
            stability_state = None
            margin = None
            balance_event = None

            if filtered_com is not None and bos_polygon:
                stability_state, margin = stability_tracker.update(
                    com_x=float(filtered_com[0]),
                    bos_polygon=bos_polygon
                )

                if stability_state == "unstable":
                    balance_event = "loss_of_balance_risk"

            # -------------------------------------------------
            # Elbow semantics
            # -------------------------------------------------
            elbow_state = None
            if left_elbow_angle is not None:
                if left_elbow_angle < 140:
                    elbow_state = "flexion"
                elif left_elbow_angle > 160:
                    elbow_state = "extension"
                else:
                    elbow_state = "transition"

            # -------------------------------------------------
            # SEND DATA
            # -------------------------------------------------
            await websocket.send_json(pose_payload)

            if frames % 10 == 0:
                elapsed = time.time() - start_time
                fps = frames / elapsed if elapsed > 0 else 0.0

                await websocket.send_json({
                    "type": "stats",
                    "frames_received": int(frames),
                    "uptime_sec": round(float(elapsed), 2),
                    "input_fps": round(float(fps), 2),

                    "joint_angles": {
                        "left_elbow": {
                            "angle_deg": round(float(left_elbow_angle), 1)
                            if left_elbow_angle is not None else None,
                            "state": elbow_state,
                            "valid": bool(left_elbow_angle is not None)
                        }
                    },

                    "center_of_mass": {
                        "x": round(float(filtered_com[0]), 3),
                        "y": round(float(filtered_com[1]), 3),
                        "reference": "image_space",
                        "in_frame": bool(
                            0.0 <= float(filtered_com[0]) <= 1.0 and
                            0.0 <= float(filtered_com[1]) <= 1.0
                        )
                    } if filtered_com is not None else None,

                    "base_of_support": {
                        "polygon": bos_polygon,
                        "support": support_type
                    } if bos_polygon else None,

                    "stability": {
                        "state": stability_state,
                        "margin": margin,
                        "basis": "com_vs_base_of_support"
                    } if stability_state else None,

                    "events": {
                        "motion": motion_event,
                        "balance": balance_event,
                        "phase": None
                    }
                })

    try:
        await asyncio.gather(receive_frames(), process_frames())
    except WebSocketDisconnect:
        print("WebSocket disconnected")
