import asyncio
from typing import Optional


class FrameBuffer:
    """
    Realtime frame buffer.
    Always stores ONLY the latest frame.
    Older frames are discarded automatically.
    """

    def __init__(self):
        self._frame: Optional[bytes] = None
        self._lock = asyncio.Lock()

    async def push(self, frame: bytes) -> None:
        """
        Store the latest frame.
        Overwrites any previous frame.
        """
        async with self._lock:
            self._frame = frame

    async def pop(self) -> Optional[bytes]:
        """
        Retrieve and clear the latest frame.
        Returns None if no frame is available.
        """
        async with self._lock:
            frame = self._frame
            self._frame = None
            return frame

    async def has_frame(self) -> bool:
        async with self._lock:
            return self._frame is not None
