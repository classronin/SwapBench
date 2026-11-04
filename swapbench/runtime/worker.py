"""SwapBench 换脸的推理子进程。

架构（解决「模型加载/推理期间 UI 卡顿」）：
    onnxruntime 创建会话的 C++ 构造函数全程持有 GIL（实测单次 500-700ms），
    与 UI 线程同进程时无论怎么调 switchinterval 都无法避免帧卡顿。
    所以推理整体搬进这个常驻子进程，与父进程用 stdin/stdout 管道通信：

    请求（父 → 子）  ("load", det_size) / ("detect", img) /
                     ("swap", target, face_dict, source_dict, paste_back, use_restorer) /
                     ("set_det_size", det_size) / ("close",)
    响应（子 → 父）  ("progress", value, label) × n → ("result", payload) 或 ("error", tb)

    帧格式：8 字节小端长度 + pickle 数据。所有 GL/推理开销（GIL）都在子进程，
    父进程的阻塞读管道会释放 GIL，UI 全程流畅。

启动：python -m swapbench.runtime.worker（由 runtime/client.py 以子进程方式拉起，勿直接运行）。
"""

from __future__ import annotations

import pickle
import struct
import sys
import traceback
import warnings

# 与 main.py 相同的兜底：skimage 旧 API 的 FutureWarning 保持控制台干净
warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)

import numpy as np  # noqa: E402

_OUT = sys.stdout.buffer          # 协议专用句柄（任何库的 print 都不许碰它）
sys.stdout = sys.stderr           # 库的输出全部改道 stderr，防止污染帧流


class _ParentGone(Exception):
    """回复/进度写不出去（父进程已退出）——立即静默收尾，不打任何堆栈。"""


def _send(obj) -> None:
    data = pickle.dumps(obj, protocol=4)
    _OUT.write(struct.pack("<Q", len(data)))
    _OUT.write(data)
    _OUT.flush()


def _try_send(obj) -> bool:
    """发送一帧；返回 False 表示管道已断（父进程已退出）。"""
    try:
        _send(obj)
        return True
    except OSError:
        return False


def _recv():
    n = sys.stdin.buffer.read(8)
    if n is None or len(n) < 8:
        return None                               # 父进程关闭 → 退出
    (n,) = struct.unpack("<Q", n)
    data = sys.stdin.buffer.read(n)
    if data is None or len(data) < n:
        return None
    return pickle.loads(data)


def _actual_provider(analysis) -> str:
    """会话真实生效的 provider（available 列表不代表初始化成功，cudnn 缺失时静默回退 CPU）。"""
    try:
        return str(analysis.det_model.session.get_providers()[0])
    except Exception:
        return "unknown"


def main() -> None:
    from insightface.app.common import Face

    from ..core.engine import SwapBackend
    from ..core.masking import MaskOptions

    def progress(value: float, label: str) -> None:
        # 加载/换脸进行中父进程退出 → 管道断开 → 中止当前推理（结果已无人接收）
        if not _try_send(("progress", value, label)):
            raise _ParentGone()

    backend = SwapBackend(on_progress=progress)
    try:
        while True:
            msg = _recv()
            if msg is None:
                return                    # 父进程关闭 → 正常退出
            try:
                kind = msg[0]
                if kind == "load":
                    backend.load(msg[1])
                    if not _try_send(("result", {
                        "provider": _actual_provider(backend.analysis),
                        "model_names": sorted(backend.analysis.models),
                    })):
                        return
                elif kind == "detect":
                    faces = backend.detect(msg[1])
                    # Face 是 dict 子类，字段全是 ndarray/标量，可直接 pickle
                    if not _try_send(("result", [dict(f) for f in faces])):
                        return
                elif kind == "swap":
                    _, target, face, source, paste_back, use_restorer = msg[:6]
                    mask_payload = msg[6] if len(msg) > 6 else None
                    mask_options = MaskOptions.from_dict(mask_payload) if mask_payload else None
                    out = backend.swap(target, Face(face), Face(source), paste_back, use_restorer,
                                       mask_options)
                    if not _try_send(("result", out)):
                        return
                elif kind == "set_det_size":
                    backend.set_det_size(msg[1])
                    if not _try_send(("result", None)):
                        return
                elif kind == "close":
                    _try_send(("result", None))
                    return
                else:
                    if not _try_send(("error", f"未知请求：{kind!r}")):
                        return
            except _ParentGone:
                return                    # 静默收尾，不刷堆栈
            except Exception:
                if not _try_send(("error", traceback.format_exc())):
                    return                # 错误帧也发不出去 → 父进程已退出
    finally:
        try:
            _OUT.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
