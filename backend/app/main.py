from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

from app.realtime.session import LiveSession

app = FastAPI(title="Body Dynamics")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()

    # One session per connection
    session = LiveSession(buffer_size=30)

    async def processing_loop():
        """
        This runs independently of the WebSocket receive loop.
        Later: ML, pose, tracking will live here.
        """
        while True:
            frame = await session.frame_buffer.pop()
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            # Placeholder for processing (ML later)
            await session.on_processed()

    # Start background processing task
    processor = asyncio.create_task(processing_loop())

    try:
        while True:
            message = await websocket.receive()

            # TEXT control messages
            if "text" in message:
                await websocket.send_json({
                    "type": "text_ack",
                    "message": message["text"]
                })

            # BINARY frame messages
            elif "bytes" in message:
                await session.on_frame(message["bytes"])

                # Send lightweight stats periodically
                if session.frames_received % 10 == 0:
                    await websocket.send_json({
                        "type": "stats",
                        **session.stats()
                    })

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    finally:
        processor.cancel()
