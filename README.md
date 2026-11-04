# SwapBench 换脸

本地离线的人脸替换工具：图形界面里拖入一张**源人脸**、一张**目标图**、一段**视频**或**动画 GIF / WebP**，一键换脸并对比/播放确认后保存；命令行同样支持单图、视频与动画图片（视频保留原音轨）。

## 安装运行

- NVIDIA 显卡 + `onnxruntime-gpu`, 需对应 CUDA 版本。
- 无 N 卡时可改用 CPU 版 `onnxruntime`（速度慢一些）
- 需要系统 PATH 中有 `ffmpeg`（只负责编码视频与封装音轨，推理仍是本地模型）。

```powershell
git clone https://github.com/ClassRonin/SwapBench.git
cd SwapBench

uv venv
uv pip install -r requirements.txt
uv run main.py
```

## 模型准备

下载后放入程序目录的 `models\`：

```
models/
├─ buffalo_l/             人脸检测 + 识别（det_10g.onnx、w600k_r50.onnx 等 5 个 onnx）
├─ inswapper_128.onnx     换脸模型
├─ xseg.onnx              遮挡遮罩（可选，缺失时「遮挡保护」自动禁用）
└─ GFPGANv1.4.onnx        画质修复（可选，缺失时自动跳过）
```

国外： [buffalo_l.zip](https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip)
— [inswapper_128.onnx](https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx)
— [GFPGANv1.4.onnx](https://huggingface.co/Neus/GFPGANv1.4/resolve/main/GFPGANv1.4.onnx)
— [xseg.onnx](https://github.com/yakhyo/face-segmentation/releases/download/weights/xseg.onnx)

国内： [buffalo_l.zip](https://gh-proxy.com/https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip)
— [inswapper_128.onnx](https://hf-mirror.com/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx)
— [GFPGANv1.4.onnx](https://hf-mirror.com/Neus/GFPGANv1.4/resolve/main/GFPGANv1.4.onnx)
— [xseg.onnx](https://gh-proxy.com/https://github.com/yakhyo/face-segmentation/releases/download/weights/xseg.onnx)

### 快捷键

| 按键 | 作用 |
|---|---|
| `1` / `2` | 图片模式：切换 目标 / 对比 视图；视频模式：切换 目标 / 成片 |
| `C` | 视频模式：目标 / 成片对比（可拖分割线，同步播放） |
| `3` | 图片模式：保存结果；视频模式：另存成片 |
| `O` | 视频模式：用系统播放器打开成片（有声音） |
| `Space` | 图片模式：圆形放大镜对比；视频模式：播放 / 暂停 |
| `+` / `-` | 图片模式：调整圆形放大镜大小 |
| `Esc` | 退出程序 |
| `Home` | 复位视图（缩放 + 平移） |
| 滚轮 / 拖动 | 视图内缩放 / 平移 |
| 左中分隔条 | 悬停出现 ↔ 光标，按住拖动调整左栏宽度 |

---

![ ](https://classronin.github.io/picx-images-hosting/1.5trubb4zme.webp)
![ ](https://classronin.github.io/picx-images-hosting/0.1zj2scn1p1.webp)

---

## 致谢

- [InsightFace](https://github.com/deepinsight/insightface) —— 人脸检测、识别与换脸模型
- [GFPGAN](https://github.com/TencentARC/GFPGAN) —— 人像画质修复
- [pyglet](https://github.com/pyglet/pyglet) —— 窗口与图形

## 免责声明

本工具仅供学习研究与图像处理技术交流。请仅对你拥有权利或已获授权的照片使用，不得用于伪造他人身份、侵犯肖像权或隐私等违法行为，由此产生的后果由使用者自行承担。
