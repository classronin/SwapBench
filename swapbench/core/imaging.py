"""图像读写与几何工具，全部是纯函数。"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

RESAMPLE = Image.Resampling.LANCZOS


def load_bgr(path: str | Path) -> np.ndarray:
    """cv2.imread 读不了含中文的路径，统一走 PIL。"""
    with Image.open(str(path)) as handle:
        return cv2.cvtColor(np.array(handle.convert("RGB")), cv2.COLOR_RGB2BGR)


def save_bgr(path: str | Path, image: np.ndarray) -> None:
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(str(path))


def to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def fit_within(width: int, height: int, box_w: int, box_h: int) -> tuple[int, int]:
    """等比缩放到不超过 box_w x box_h，不放大。"""
    if width <= 0 or height <= 0 or box_w <= 0 or box_h <= 0:
        return max(1, width), max(1, height)
    scale = min(box_w / width, box_h / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def face_box(bbox, shape, margin: float = 0.25) -> tuple[int, int, int, int]:
    """按人脸 bbox 外扩 margin 计算裁剪框，并钳制到图片范围内。"""
    height, width = shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    body_w, body_h = x2 - x1, y2 - y1
    return (
        int(max(0, x1 - body_w * margin)),
        int(max(0, y1 - body_h * margin)),
        int(min(width, x2 + body_w * margin)),
        int(min(height, y2 + body_h * margin)),
    )
