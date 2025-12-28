import time
from .frame_buffer import FrameBuffer


class LiveSession:
    
    def __init__(self, buffer_size: int = 30):
        self.frame_buffer = FrameBuffer(max_size=buffer_size)

        self.frames_received = 0
        self.frames_processed = 0

        self.started_at = time.time()
        self.last_frame_time = None

    async def on_frame(self, frame: bytes) -> None:
        
        self.frames_received += 1
        self.last_frame_time = time.time()
        await self.frame_buffer.push(frame)

    async def on_processed(self) -> None:
       
        self.frames_processed += 1

    def stats(self) -> dict:
        """
        Lightweight stats snapshot for UI/debugging.
        """
        elapsed = max(time.time() - self.started_at, 1e-6)
        return {
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "uptime_sec": round(elapsed, 2),
            "input_fps": round(self.frames_received / elapsed, 2),
        }
