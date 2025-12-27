import cv2
import numpy as np
import mediapipe as mp
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()

    start_time = time.time()
    frames = 0

    try:
        while True:
            message = await websocket.receive()

            if "bytes" not in message:
                continue

            frame_bytes = message["bytes"]
            frames += 1

            # Decode JPEG
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            keypoints = []
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    keypoints.append([lm.x, lm.y])

            # Send pose
            await websocket.send_json({
                "type": "pose",
                "keypoints": {
                    "person_0": keypoints
                }
            })

            # Send stats occasionally
            if frames % 10 == 0:
                elapsed = time.time() - start_time
                await websocket.send_json({
                    "type": "stats",
                    "frames_received": frames,
                    "uptime_sec": round(elapsed, 2),
                    "input_fps": round(frames / elapsed, 2)
                })

    except WebSocketDisconnect:
        print("WebSocket disconnected cleanly")
