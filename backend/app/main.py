import time
import cv2
import numpy as np
import mediapipe as mp

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect

from app.processing.stats import angle_between

# =========================
# App setup
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# =========================
# MediaPipe Pose
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =========================
# WebSocket endpoint
# =========================
@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()

    start_time = time.time()
    frames = 0

    try:
        while True:
            message = await websocket.receive()

            # Expect binary JPEG frames
            if "bytes" not in message:
                continue

            frame_bytes = message["bytes"]
            frames += 1

            # Decode JPEG -> image
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            # =========================
            # Extract keypoints
            # =========================
            keypoints = []
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    keypoints.append([lm.x, lm.y])

            # =========================
            # Compute left elbow angle
            # (11 shoulder, 13 elbow, 15 wrist)
            # =========================
            left_elbow_angle = None
            if len(keypoints) > 15:
                left_elbow_angle = angle_between(
                    keypoints[11],
                    keypoints[13],
                    keypoints[15]
                )

            # =========================
            # Send pose (every frame)
            # =========================
            await websocket.send_json({
                "type": "pose",
                "keypoints": {
                    "person_0": keypoints
                }
            })

            # =========================
            # Send stats (every 10 frames)
            # =========================
            if frames % 10 == 0:
                elapsed = time.time() - start_time
                await websocket.send_json({
                    "type": "stats",
                    "frames_received": frames,
                    "uptime_sec": round(elapsed, 2),
                    "input_fps": round(frames / elapsed, 2),
                    "left_elbow_angle": round(left_elbow_angle, 1)
                    if left_elbow_angle is not None else None
                })

    except WebSocketDisconnect:
        print("WebSocket disconnected cleanly LEss go")
