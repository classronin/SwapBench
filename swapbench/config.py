"""路径与界面常量，全部集中在这里维护。"""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]

# 模型根目录：固定为程序目录下的 models
MODEL_ROOT = APP_DIR / "models"
# 示例图片与文档截图
ASSETS_DIR = APP_DIR / "assets"
PACK_NAME = "buffalo_l"
PACK_DIR = MODEL_ROOT / PACK_NAME
SWAPPER_FILE = MODEL_ROOT / "inswapper_128.onnx"
GFPGAN_FILE = MODEL_ROOT / "GFPGANv1.4.onnx"
XSEG_FILE = MODEL_ROOT / "xseg.onnx"

DETECTION_SIZES: dict[str, tuple[int, int]] = {
    "小 480": (480, 480),
    "中 640": (640, 640),
    "大 960": (960, 960),
    "特大 1280": (1280, 1280),
}
DEFAULT_DET_SIZE = DETECTION_SIZES["中 640"]
