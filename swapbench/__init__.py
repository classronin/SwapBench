"""SwapBench 换脸。

模块划分：
- config       ：模型路径、素材目录与界面常量
- core.engine  ：InsightFace 推理核心（只在子进程中运行）
- core.imaging ：图像读取 / 保存 / 缩放的纯函数
- core.masking ：XSeg 遮挡保护 + 106 点嘴部回贴
- runtime.client：推理子进程的父进程代理（UI 后台线程调用）
- runtime.worker：推理子进程入口（python -m swapbench.runtime.worker）
- ui.window    ：pyglet 主窗口
- ui.compare   ：双图对比查看器（滚轮缩放 / 拖动 / 分割线 / 圆形）
- args         ：命令行共用的遮罩参数
- cli          ：单图命令行（python -m swapbench.cli）
- video        ：视频命令行（python -m swapbench.video）
"""

from .runtime.client import SwapBackend

__all__ = ["SwapBackend"]
