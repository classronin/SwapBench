"""SwapBench 换脸：程序入口（pyglet GUI）。

运行：python main.py
模型全部从本目录 models\\ 读取，不会联网下载：
  models/buffalo_l          人脸检测 + 识别（SCRFD + ArcFace）
  models/inswapper_128.onnx 换脸模型
  models/xseg.onnx          遮挡保护（可选，右栏「遮挡保护」开关）
  models/GFPGANv1.4.onnx    画质修复（可选，默认开启）

交互速查：
  拖文件   左侧「源人脸」→ 设源人脸；中间「目标」→ 设目标图 / 视频 / 动画 GIF·WebP
  图片模式 1=目标 2=对比 3=保存；对比视图滚轮缩放 · 拖动平移 · 拖分割线 · Space 圆形对比 · +/- 调整圆
  视频模式 1=目标 2=成片 C=对比 3=另存 O=系统播放器；Space 播放/暂停（应用内预览无声）
  底部按钮：目标 / 对比 / 保存；切视图/换脸完成不重置缩放、平移与分割线
  保存命名：<原名>_<n>.<后缀>，n 从 0 起取第一个未占用编号，不会叠加重复后缀
  XSeg 遮挡保护用右栏「遮挡保护」按钮开关，对下一次换脸生效
  Home 复位视图；左右窗口的分隔条按住拖动可调整左栏宽度
"""

from __future__ import annotations

import warnings

# insightface 仍在用 skimage 旧 API（SimilarityTransform.estimate），skimage
# 0.26 起每次估计变换都会打 FutureWarning。face_align.py 已手动修补，但
# venv 重装会还原文件 —— 这里在 import 之前兜底过滤，保持控制台干净。
warnings.filterwarnings("ignore", message="`estimate` is deprecated", category=FutureWarning)

from swapbench.ui.window import FaceSwapPygletApp


def main() -> int:
    FaceSwapPygletApp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
