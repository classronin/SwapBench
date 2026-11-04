"""命令行共用的遮罩参数（`swapbench.cli` 与 `swapbench.video` 共用）。"""

from __future__ import annotations

import argparse

from .config import GFPGAN_FILE
from .core.masking import MaskOptions


def add_mask_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["off", "occluder", "mouth", "both"], default="both",
                        help="遮罩模式：遮挡保护 / 嘴部回贴 / 两者（默认）")
    parser.add_argument("--mouth-size", type=float, default=30.0,
                        help="0-100，外嘴扩张（盖掉换脸的唇形）")
    parser.add_argument("--mouth-feather", type=float, default=6.0, help="嘴部遮罩羽化 sigma")
    parser.add_argument("--inner", type=float, default=0.0, help="0-1，内嘴原图混合")
    parser.add_argument("--occ-dilate", type=int, default=2,
                        help="遮挡保护边界调整像素：正=向脸内扩张，负=腐蚀（接缝往头发/物体里推）")
    parser.add_argument("--occ-feather", type=float, default=12.0, help="遮挡保护羽化 sigma")
    parser.add_argument("--occ-no-match", action="store_true",
                        help="关闭遮挡回贴前的肤色对齐（默认开启，避免脸上留下暗线）")
    parser.add_argument("--no-restorer", action="store_true", help="关闭 GFPGAN（更快）")
    parser.add_argument("--det-size", type=int, default=640)


def mask_options_from(args: argparse.Namespace) -> MaskOptions:
    return MaskOptions(
        occluder=args.mode in ("occluder", "both"),
        occluder_dilate=args.occ_dilate,
        occluder_feather=args.occ_feather,
        occluder_match=not args.occ_no_match,
        mouth=args.mode in ("mouth", "both"),
        mouth_size=args.mouth_size,
        mouth_feather=args.mouth_feather,
        inner_mouth=args.inner,
    )


def restorer_enabled(args: argparse.Namespace) -> bool:
    """GFPGAN 开关：命令行关闭或模型缺失时不修复。"""
    return not args.no_restorer and GFPGAN_FILE.is_file()
