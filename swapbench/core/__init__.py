"""纯图像处理与推理核心：不涉及界面，也不涉及进程通信。"""

from .imaging import face_box, load_bgr, save_bgr
from .masking import MaskOptions

__all__ = ["MaskOptions", "face_box", "load_bgr", "save_bgr"]
