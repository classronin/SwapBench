"""SwapBackend：推理子进程的父进程代理。

推理（模型加载/检测/换脸/GFPGAN）全部在常驻子进程 swapbench.runtime.worker 中执行
（onnxruntime 创建会话的 C++ 调用全程持有 GIL，同进程会拖死 UI 帧循环，
实测单次 500-700ms；子进程隔离后父进程 GIL 全程空闲）。

- 本类方法与推理核心（core.engine.SwapBackend）签名一致，界面无需感知进程边界；
- 方法应在后台线程调用（ui.window._run_task 已保证）：阻塞读管道会释放 GIL；
- 检测结果反序列化为 FaceProxy（dict 子类 + 属性访问，替代 insightface.Face）；
- provider 来自子进程会话的 get_providers() 实况，不再看可用列表
  （available 里有 CUDA 不代表 cudnn 可用，缺失时会静默回退 CPU）。
"""

from __future__ import annotations

import pickle
import struct
import subprocess
import sys
import threading
from typing import Callable

from ..config import APP_DIR, DEFAULT_DET_SIZE

ProgressCallback = Callable[[float, str], None]


class FaceProxy(dict):
    """insightface.Face 的跨进程替身：dict + 属性访问。"""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class SwapBackend:
    """推理子进程代理，方法只在后台线程调用。"""

    def __init__(self, on_progress: ProgressCallback | None = None):
        self._on_progress: ProgressCallback = on_progress or (lambda value, label: None)
        self._proc: subprocess.Popen | None = None
        self._ready = False
        self.provider = "CPUExecutionProvider"
        self.model_names: list[str] = []
        self.det_size = DEFAULT_DET_SIZE
        # IPC 串行锁：模型加载线程 / 检测线程可能并发 _request，
        # 两个线程在同一管道上互写 pickle 帧会直接损坏协议
        self._ipc_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    # ---------------- 进程与协议 ----------------

    def _ensure_proc(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-m", "swapbench.runtime.worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                cwd=str(APP_DIR),
            )
            self._ready = False

    def _read_exact(self, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                return None                       # 子进程退出
            buf += chunk
        return buf

    def _send(self, msg, proc: subprocess.Popen | None = None) -> None:
        proc = proc or self._proc
        data = pickle.dumps(msg, protocol=4)
        try:
            proc.stdin.write(struct.pack("<Q", len(data)))
            proc.stdin.write(data)
            proc.stdin.flush()
        except OSError:
            raise RuntimeError("推理进程已退出") from None

    def _request(self, msg):
        """发一个请求，读响应；progress 帧就地转发回调，最终帧决定返回/抛错。

        持锁覆盖整个发送-接收周期：响应帧和进度帧只属于当前请求，
        并发调用者排队等待（子进程本身也是单循环逐请求处理）。
        """
        with self._ipc_lock:
            self._ensure_proc()
            self._send(msg)
            while True:
                head = self._read_exact(8)
                if head is None:
                    raise RuntimeError("推理进程意外退出")
                (n,) = struct.unpack("<Q", head)
                data = self._read_exact(n)
                if data is None:
                    raise RuntimeError("推理进程意外退出")
                frame = pickle.loads(data)
                kind = frame[0]
                if kind == "progress":
                    self._on_progress(frame[1], frame[2])
                elif kind == "result":
                    return frame[1]
                elif kind == "error":
                    raise RuntimeError(frame[1])

    # ---------------- 与 core.engine 相同的对外 API ----------------

    def load(self, det_size: tuple[int, int]) -> None:
        info = self._request(("load", det_size))
        self.provider = info["provider"]
        self.model_names = list(info["model_names"])
        self.det_size = det_size
        self._ready = True

    def set_det_size(self, det_size: tuple[int, int]) -> None:
        self._request(("set_det_size", det_size))
        self.det_size = det_size

    def detect(self, image):
        return [FaceProxy(d) for d in self._request(("detect", image))]

    def swap(self, target, target_face, source_face, paste_back: bool, use_restorer: bool,
             mask_options=None):
        # FaceProxy 是 dict 子类，pickle 后在子进程重建 insightface.Face；
        # mask_options 用普通 dict 传，父进程无需 import cv2/insightface 的遮罩模块。
        payload = None
        if mask_options is not None:
            payload = mask_options.to_dict() if hasattr(mask_options, "to_dict") else dict(mask_options)
        return self._request(("swap", target, dict(target_face), dict(source_face),
                              paste_back, use_restorer, payload))

    def close(self) -> None:
        """通知子进程退出。拿不到 IPC 锁（有请求在途）就跳过——父进程退出后
        管道 EOF 会兜底让子进程自行结束，无需强等。"""
        if self._ipc_lock.acquire(timeout=0.2):
            try:
                proc, self._proc = self._proc, None
                self._ready = False
                if proc is not None and proc.poll() is None:
                    try:
                        self._send(("close",), proc)
                    except (OSError, RuntimeError):
                        pass
            finally:
                self._ipc_lock.release()
