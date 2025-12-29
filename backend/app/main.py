import time
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.processing.pose_estimator import PoseEstimatorStub
from app.processing.stats import calculate_angle, compute_com

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pose_estimator = PoseEstimatorStub()


@app.get("/")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()

    frames = 0
    start_time = time.time()

    try:
        while True:
            message = await websocket.receive()

            # Expect binary frame
            if "bytes" not in message:
                continue

            frame_bytes = message["bytes"]
            frames += 1

            # Decode JPEG frame
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # --------------------
            # Pose Estimation
            # --------------------
            keypoints = pose_estimator.process(frame)

            # --------------------
            # Elbow Angle (Left)
            # --------------------
            left_elbow_angle = None
            if len(keypoints) > 15:
                left_elbow_angle = calculate_angle(
                    keypoints[11],  # left shoulder
                    keypoints[13],  # left elbow
                    keypoints[15],  # left wrist
                )

            # --------------------
            # Center of Mass
            # --------------------
            com = compute_com(keypoints)

            # --------------------
            # Send Pose
            # --------------------
            await websocket.send_json({
                "type": "pose",
                "keypoints": {
                    "person_0": keypoints
                }
            })

            # --------------------
            # Send Stats (every 10 frames)
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
                    "com": [round(com[0], 3), round(com[1], 3)] if com else None
                })

    except WebSocketDisconnect:
        print("WebSocket disconnected")
