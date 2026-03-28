import torch
from collections import deque


class AdaptiveBuffer:
    """
    Adaptive temporal buffer that adjusts window size based on motion complexity.
    Low complexity (static signs) -> smaller window.
    High complexity (dynamic signs) -> larger window.
    """

    def __init__(self, min_frames: int = 16, max_frames: int = 64,
                 low_thresh: float = 0.2, high_thresh: float = 0.6):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self._buffer: deque = deque(maxlen=max_frames)
        self._complexity_history: deque = deque(maxlen=max_frames)

    def add_frame(self, frame_features: torch.Tensor, complexity: float):
        """Add a feature tensor and its associated complexity score."""
        self._buffer.append(frame_features.detach().cpu())
        self._complexity_history.append(complexity)

    def get_window(self) -> torch.Tensor:
        """
        Return appropriately-sized temporal window based on recent complexity.
        Returns tensor of shape [T, feature_dim].
        """
        if len(self._buffer) == 0:
            return torch.zeros(1, 1024)

        avg_complexity = sum(self._complexity_history) / len(self._complexity_history)

        if avg_complexity < self.low_thresh:
            window_size = self.min_frames
        elif avg_complexity > self.high_thresh:
            window_size = self.max_frames
        else:
            # Linear interpolation
            ratio = (avg_complexity - self.low_thresh) / (self.high_thresh - self.low_thresh)
            window_size = int(self.min_frames + ratio * (self.max_frames - self.min_frames))

        frames = list(self._buffer)[-window_size:]
        return torch.stack(frames, dim=0)  # [T, feature_dim]

    def is_ready(self) -> bool:
        """True when buffer has minimum required frames."""
        return len(self._buffer) >= self.min_frames

    def clear(self):
        """Reset the buffer."""
        self._buffer.clear()
        self._complexity_history.clear()
