from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

from app.realtime.session import LiveSession
from app.processing.pose_stub import PoseEstimatorStub

app = FastAPI(title="Body Dynamics")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()

    
    session = LiveSession(buffer_size=30)
    estimator = PoseEstimatorStub()

    async def processing_loop():
        while True:
            frame = await session.frame_buffer.pop()
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            # Fake pose estimation
            pose = estimator.estimate(frame)
            await session.on_processed()

            # SEND POSE DATA
            await websocket.send_json({
                "type": "pose",
                "keypoints": pose
            })

    processor_task = asyncio.create_task(processing_loop())

    try:
        while True:
            message = await websocket.receive()

            # TEXT messages (debug)
            if "text" in message:
                await websocket.send_json({
                    "type": "text_ack",
                    "message": message["text"]
                })

            # BINARY messages (video frames)
            elif "bytes" in message:
                await session.on_frame(message["bytes"])

                # SEND STATS every 10 frames
                if session.frames_received % 10 == 0:
                    await websocket.send_json({
                        "type": "stats",
                        **session.stats()
                    })

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    finally:
        processor_task.cancel()
