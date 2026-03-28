import cv2
import threading
import queue
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """Async camera capture using a daemon thread and thread-safe queue."""

    def __init__(self, device_id: int = 0, fps: int = 30):
        self.device_id = device_id
        self.fps = fps
        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._thread: threading.Thread | None = None
        self._running = False
        self._frame_count = 0
        self._cap: cv2.VideoCapture | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self):
        """Start the camera capture thread."""
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            logger.error(f"Camera device {self.device_id} not found. "
                         "Check device_id in config.yaml or try 0, 1, 2.")
            raise RuntimeError(f"Cannot open camera device {self.device_id}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Camera started on device {self.device_id}")

    def stop(self):
        """Stop the capture thread and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("Camera stopped")

    def get_frame(self) -> np.ndarray | None:
        """Get the latest frame from the queue (non-blocking)."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def _capture_loop(self):
        fps_timer = time.time()
        fps_count = 0

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.01)
                continue

            self._frame_count += 1
            fps_count += 1

            # Drop oldest frame if queue is full to keep latency low
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

            # Log FPS every 5 seconds
            elapsed = time.time() - fps_timer
            if elapsed >= 5.0:
                actual_fps = fps_count / elapsed
                logger.info(f"Camera FPS: {actual_fps:.1f}")
                fps_timer = time.time()
                fps_count = 0
