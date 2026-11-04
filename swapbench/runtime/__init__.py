"""进程边界：父进程代理 + 常驻推理子进程。"""

from .client import FaceProxy, SwapBackend

__all__ = ["FaceProxy", "SwapBackend"]
