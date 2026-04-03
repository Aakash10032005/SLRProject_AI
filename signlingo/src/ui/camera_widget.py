import numpy as np
import cv2
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


class CameraWidget(QLabel):
    """Displays live camera feed with sign label and confidence overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setText("No Camera")
        self.setStyleSheet("color: #94A3B8; font-size: 18px;")
        self._last_label = ''
        self._last_confidence = 0.0

    def update_frame(self, frame: np.ndarray, label: str, confidence: float):
        """Overlay label and confidence bar, then display frame."""
        if frame is None:
            return

        self._last_label = label
        self._last_confidence = confidence

        display = frame.copy()
        h, w = display.shape[:2]

        # Draw label
        if label:
            cv2.putText(display, label, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Draw confidence bar
        bar_w = int(w * confidence)
        color = (34, 197, 94) if confidence > 0.75 else (234, 179, 8) if confidence > 0.5 else (239, 68, 68)
        cv2.rectangle(display, (0, h - 10), (bar_w, h), color, -1)

        # BGR → RGB; contiguous buffer + deep QImage copy so Qt does not reference freed ndarray
        rgb = np.ascontiguousarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        h_rgb, w_rgb, ch = rgb.shape
        bytes_per_line = ch * w_rgb
        qimg = QImage(
            rgb.data, w_rgb, h_rgb, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit widget maintaining aspect ratio
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)
