"""遮罩预览：不启动 GUI，直接走推理子进程跑一次换脸。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe -m swapbench.cli --mode both
    .venv\\Scripts\\python.exe -m swapbench.cli --mode mouth --mouth-size 45
    .venv\\Scripts\\python.exe -m swapbench.cli --mode both --inner 0.6 --occ-dilate 5

默认读取 assets/人脸.jpg、assets/目标.jpg，输出 mask_preview.png。
视频换脸见 swapbench/video.py。
"""

from __future__ import annotations

import argparse
import sys

from .args import add_mask_arguments, mask_options_from, restorer_enabled
from .config import APP_DIR, ASSETS_DIR
from .core.imaging import load_bgr, save_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SwapBench 遮罩预览（单张图片）")
    parser.add_argument("--source", default=str(ASSETS_DIR / "人脸.jpg"))
    parser.add_argument("--target", default=str(ASSETS_DIR / "目标.jpg"))
    parser.add_argument("--out", default=str(APP_DIR / "mask_preview.png"))
    add_mask_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    options = mask_options_from(args)
    use_restorer = restorer_enabled(args)

    from .runtime.client import SwapBackend

    backend = SwapBackend(on_progress=lambda v, s: None)
    try:
        backend.load((args.det_size, args.det_size))
        print(f"provider={backend.provider} models={','.join(backend.model_names)}")

        source_img = load_bgr(args.source)
        target_img = load_bgr(args.target)
        source_faces = backend.detect(source_img)
        target_faces = backend.detect(target_img)
        if not source_faces or not target_faces:
            print("源图或目标图没有检测到人脸", file=sys.stderr)
            return 2

        # detect() 已按人脸面积从大到小排序
        source_face = source_faces[0]
        target_face = target_faces[0]
        result = backend.swap(
            target_img, target_face, source_face,
            paste_back=True, use_restorer=use_restorer, mask_options=options,
        )
        save_bgr(args.out, result)
        print(f"已保存：{args.out}  ({result.shape[1]}x{result.shape[0]})")
        print(f"遮罩选项：{options.to_dict()}")
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
