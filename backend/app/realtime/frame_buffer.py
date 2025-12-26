import asyncio
from collections import deque
from typing import Optional


class FrameBuffer:
   
    def __init__(self, max_size: int = 30):
        self.buffer = deque(maxlen=max_size)
        self.lock = asyncio.Lock()

    async def push(self, frame: bytes) -> None:
       
        async with self.lock:
            self.buffer.append(frame)

    async def pop(self) -> Optional[bytes]:
       
        async with self.lock:
            if not self.buffer:
                return None
            return self.buffer.popleft()

    async def size(self) -> int:
        """Current buffer size."""
        async with self.lock:
            return len(self.buffer)
