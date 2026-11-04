"""嘴部 / 遮挡物遮罩：纯函数 + 可选的推理会话封装。

两类用途：

1. **遮挡保护（occluder）** —— 用 ``xseg.onnx`` 对目标脸做前向，得到
   “人脸 vs 非人脸”软遮罩；把非人脸区域（嘴里的瓶子、手、头发等）从原图
   贴回换脸结果，避免 inswapper 自带的大核羽化把遮挡物边缘糊掉。回贴前会把
   原图的 LAB 均值对齐到换脸结果、并用较大的羽化，否则 XSeg 边界切过脸颊
   皮肤时会留下一条暗线。
2. **原嘴回贴（mouth）** —— 用 InsightFace 106 点的外唇轮廓做多边形，
   把目标图原来的嘴唇/嘴内物体贴回换脸结果；可选内嘴（牙齿/舌头）混合。

顺序很重要（见 ``SwapBackend.swap``）：换脸 → **嘴部回贴** → 画质修复 →
**遮挡保护**。嘴部先贴、让 GFPGAN 把接缝一起锐化；遮挡物后贴，才不会被
GFPGAN 再糊一次。

所有函数都只依赖 cv2 / numpy；XSeg 会话由调用方传入，方便在子进程里懒加载。
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np
from insightface.utils import face_align

# InsightFace 106 点里嘴部点的索引约定（与 Deep-Live-Cam / roop-unleashed 一致）：
# 52-71 是嘴部 20 点；52-70 的凸包是外唇轮廓，下面 8 点是内唇开口。
OUTER_MOUTH_IDX = list(range(52, 71))
INNER_MOUTH_IDX = [54, 57, 60, 62, 65, 66, 69, 70]


@dataclass
class MaskOptions:
    """换脸后的遮罩选项。

    两个开关默认关闭（保持旧行为）；下面的尺寸/羽化默认值是按“嘴里叼着杯子”
    这类遮挡场景调出来的，打开开关后基本不用再动。
    """

    occluder: bool = False          # 用 XSeg 保护非人脸遮挡物
    occluder_dilate: int = 2        # 保护区域向人脸内扩张的像素（负数=腐蚀，接缝往头发/物体里推）
    occluder_feather: float = 12.0  # 保护区域边缘羽化（高斯 sigma）
    occluder_match: bool = True     # 回贴前把原图的 LAB 均值对齐到换脸结果（消除肤色台阶）
    occluder_match_band: int = 12   # 统计色彩偏移用的“脸内边界带”宽度（像素）

    mouth: bool = False             # 把目标图原嘴贴回换脸结果
    mouth_size: float = 30.0        # 0-100，外嘴多边形向外扩张（覆盖换脸后的唇形）
    mouth_feather: float = 6.0      # 嘴部遮罩羽化（高斯 sigma）
    inner_mouth: float = 0.0        # 0-1，内嘴（牙齿/舌头）原图混合比例
    color_transfer: bool = True     # 嘴部回贴前做 LAB 色彩迁移
    color_contrast: float = 0.0     # 0=只对齐均值（保住嘴唇红色），1=同时对齐方差

    def enabled(self) -> bool:
        return self.occluder or self.mouth_enabled()

    def mouth_enabled(self) -> bool:
        return self.mouth or self.inner_mouth > 0.0

    def to_dict(self) -> dict:
        return {
            "occluder": bool(self.occluder),
            "occluder_dilate": int(self.occluder_dilate),
            "occluder_feather": float(self.occluder_feather),
            "occluder_match": bool(self.occluder_match),
            "occluder_match_band": int(self.occluder_match_band),
            "mouth": bool(self.mouth),
            "mouth_size": float(self.mouth_size),
            "mouth_feather": float(self.mouth_feather),
            "inner_mouth": float(self.inner_mouth),
            "color_transfer": bool(self.color_transfer),
            "color_contrast": float(self.color_contrast),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "MaskOptions":
        """从普通 dict 还原；未知键忽略、缺失键用默认值（兼容旧版本传参）。"""
        if not data:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def core_mouth(self) -> "MaskOptions":
        """“芯部”嘴部参数：比羽化区小一圈、几乎不羽化。

        画质修复前后的两遍贴嘴用：先按用户参数（大范围 + 大羽化）贴一遍，
        让 GFPGAN 把接缝一起锐化；修复后再用这个小芯部贴一遍，把嘴唇本身
        换回原图的清晰像素（边缘落在上一遍的纯原图区域里，看不出接缝）。
        """
        return MaskOptions(
            occluder=False,
            mouth=self.mouth,
            mouth_size=max(0.0, float(self.mouth_size) * 0.65),
            mouth_feather=max(1.0, float(self.mouth_feather) * 0.25),
            inner_mouth=float(self.inner_mouth),
            color_transfer=bool(self.color_transfer),
            color_contrast=float(self.color_contrast),
        )


# ---------------- XSeg ----------------

def xseg_face_mask(session, image: np.ndarray, face, size: int = 256,
                   pad: int = 48) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """返回 ``(软遮罩, 有效区域, ROI)``，两者都只覆盖 ROI（1=人脸，0=非人脸/遮挡物）。

    ``session`` 是 ``models/xseg.onnx`` 的 InferenceSession。输入按 inswapper
    相同的 5 点模板对齐到 ``size``，输出再反算回原图坐标。

    ``有效区域`` 是对齐方框的覆盖图：XSeg 只“看见”对齐方框内的像素，方框外
    的 0 并不代表遮挡物。遮挡保护必须乘上它，否则方框外的整片背景/头发都会被
    当成遮挡物，在 ROI 矩形边界处留下一条硬边。

    只反算到人脸所在的矩形 ROI，而不是整张图：1080p 视频里整图做 warp /
    羽化 / LAB 转换要几百毫秒，而 ROI 之外全是 0。``pad`` 给后续的膨胀、
    羽化和边界带统计留余量，必须大于它们的最大作用半径。
    """
    aimg, matrix = face_align.norm_crop2(image, face.kps, size)
    blob = (aimg.astype(np.float32) / 255.0)[None, ...]
    out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    mask = np.clip(np.asarray(out).reshape(size, size).astype(np.float32), 0.0, 1.0)

    inverse = cv2.invertAffineTransform(matrix)
    # 对齐图 4 个角映射回原图 → 这块 warp 的精确外接矩形
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    pts = cv2.transform(corners[None, ...], inverse)[0]
    height, width = image.shape[:2]
    x0 = max(0, int(np.floor(pts[:, 0].min())) - int(pad))
    y0 = max(0, int(np.floor(pts[:, 1].min())) - int(pad))
    x1 = min(width, int(np.ceil(pts[:, 0].max())) + int(pad))
    y1 = min(height, int(np.ceil(pts[:, 1].max())) + int(pad))
    if x1 <= x0 or y1 <= y0:
        return (np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32),
                (0, 0, 0, 0))

    local = inverse.copy()
    local[0, 2] -= x0
    local[1, 2] -= y0
    roi = cv2.warpAffine(
        mask, local, (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR, borderValue=0.0,
    )
    coverage = cv2.warpAffine(
        np.ones((size, size), dtype=np.float32), local, (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR, borderValue=0.0,
    )
    return (np.clip(roi, 0.0, 1.0), np.clip(coverage, 0.0, 1.0), (x0, y0, x1, y1))


def soften_mask(mask: np.ndarray, dilate_px: int = 0, feather: float = 0.0) -> np.ndarray:
    """形态学调整 + 高斯羽化，返回 [0,1] float32。

    ``dilate_px`` 正数=膨胀，**负数=腐蚀**（把边界往遮罩内部收）。腐蚀用在
    遮挡保护上可以把接缝从皮肤推进头发/物体内部，避免在人脸上留下一条边。
    """
    out = np.asarray(mask, dtype=np.float32)
    px = int(dilate_px)
    if px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
        out = cv2.dilate(out, k)
    elif px < 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-px * 2 + 1, -px * 2 + 1))
        out = cv2.erode(out, k)
    sigma = float(feather)
    if sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    return np.clip(out, 0.0, 1.0)


# ---------------- 多边形 / 嘴部 ----------------

def outer_mouth_polygon(landmarks: np.ndarray, size: float = 0.0, chin_bias: bool = True) -> np.ndarray:
    """外唇凸包；``size`` 0-100，越大越向外/向下（下巴方向）扩张。"""
    pts = np.asarray(landmarks, dtype=np.float32)[OUTER_MOUTH_IDX]
    s = float(np.clip(size, 0.0, 100.0)) / 100.0
    expansion = 1.0 + s * 2.0
    center = pts.mean(axis=0)
    offsets = pts - center
    scale_y = np.full(len(pts), expansion, dtype=np.float32)
    if chin_bias and s > 0.0:
        chin = 1.0 + s * 2.0
        scale_y = np.where(offsets[:, 1] > 0.0, expansion * chin, expansion)
    expanded = pts.copy()
    expanded[:, 0] = center[0] + offsets[:, 0] * expansion
    expanded[:, 1] = center[1] + offsets[:, 1] * scale_y
    return cv2.convexHull(expanded.astype(np.int32))


def inner_mouth_polygon(landmarks: np.ndarray) -> np.ndarray:
    """内唇开口凸包（牙齿/舌头区域）。"""
    pts = np.asarray(landmarks, dtype=np.int32)[INNER_MOUTH_IDX]
    return cv2.convexHull(pts)


def _clip_roi(x0: int, y0: int, x1: int, y1: int, pad: int,
              shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """把矩形扩展 ``pad`` 像素并裁剪到图像范围内。"""
    height, width = shape[:2]
    return (
        max(0, int(x0) - int(pad)),
        max(0, int(y0) - int(pad)),
        min(width, int(x1) + int(pad)),
        min(height, int(y1) + int(pad)),
    )


def polygon_mask(shape: tuple[int, int], polygon: np.ndarray, feather: float = 5.0) -> np.ndarray:
    """在 ROI 内填充多边形并羽化，返回原图尺寸的 [0,1] float32 遮罩。"""
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    if polygon is None or len(polygon) < 3:
        return mask

    pad = max(2, int(feather * 3))
    x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return mask

    roi = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local = polygon.astype(np.int32) - np.array([x0, y0], dtype=np.int32)
    cv2.fillPoly(roi, [local], 255)

    roi_f = roi.astype(np.float32) / 255.0
    if feather > 0:
        roi_f = cv2.GaussianBlur(roi_f, (0, 0), float(feather))
    peak = float(roi_f.max())
    if peak > 1e-6:
        roi_f /= peak
    mask[y0:y1, x0:x1] = roi_f
    return np.clip(mask, 0.0, 1.0)


# ---------------- 色彩 / 合成 ----------------

def color_transfer(source: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None,
                   contrast: float = 0.0) -> np.ndarray:
    """把 source 的 LAB 均值对齐到 target；``mask`` 限定统计区域。

    ``contrast`` 是方差比重的指数：0=只对齐均值（保留 original 的对比度/饱和度），
    1=同时对齐方差（roop-unleashed 的完整做法）。

    嘴部默认用 0：整片嘴的统计里皮肤占绝大多数，对齐方差会把嘴唇的红色和
    明暗差一起压平，贴回去就是“没有嘴唇的嘴”。只对齐均值既能修掉整体偏色，
    又能保住嘴唇本身的颜色对比。
    """
    src = source.astype(np.float32) / 255.0
    dst = target.astype(np.float32) / 255.0
    # 灰度/黑白素材：色度没有意义，跳过，否则会出现蓝色伪影
    if (np.mean(np.abs(src[..., 0] - src[..., 1])) < 0.02
            and np.mean(np.abs(src[..., 1] - src[..., 2])) < 0.02):
        return source

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    dst_lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
    roi_mask = None
    if mask is not None:
        roi_mask = (np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        if int((roi_mask > 0).sum()) < 16:
            return source

    src_mean, src_std = cv2.meanStdDev(src_lab, mask=roi_mask)
    dst_mean, dst_std = cv2.meanStdDev(dst_lab, mask=roi_mask)
    src_mean = src_mean.reshape(1, 1, 3).astype(np.float32)
    src_std = np.maximum(src_std.reshape(1, 1, 3), 1e-3).astype(np.float32)
    dst_mean = dst_mean.reshape(1, 1, 3).astype(np.float32)
    dst_std = dst_std.reshape(1, 1, 3).astype(np.float32)

    if contrast > 0.0:
        ratio = np.power(dst_std / src_std, float(contrast)).astype(np.float32)
    else:
        ratio = np.ones((1, 1, 3), dtype=np.float32)
    result = (src_lab - src_mean) * ratio + dst_mean
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return np.clip(result * 255.0, 0.0, 255.0).round().astype(np.uint8)


def lab_mean_offset(original: np.ndarray, swapped: np.ndarray, face_mask: np.ndarray,
                    band_px: int = 12) -> np.ndarray | None:
    """估计 ``original`` 往 ``swapped`` 靠的 LAB 均值偏移（只在“脸内边界带”统计）。

    遮挡保护把 XSeg 判定为非人脸的原图像素贴回换脸结果，接缝就落在 XSeg 边界
    上。换脸结果的肤色来自源人脸，和目标图的肤色往往不同，于是这条接缝在
    脸颊/下颌的皮肤上表现为一条暗线（或亮线）。这里只取脸内部、紧贴边界的
    一圈像素估计两者的均值差，把原图整体平移过去，台阶就消失了；遮挡物本身
    （瓶子等）的结构和纹理完全不受影响，只是整体色偏了几个色阶。

    返回 ``None`` 表示统计区域太小，调用方应当跳过对齐。
    """
    height, width = original.shape[:2]
    band_px = max(1, int(band_px))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1))
    inner = cv2.erode((np.asarray(face_mask) > 0.5).astype(np.float32), k)
    band = np.clip(np.asarray(face_mask, dtype=np.float32) - inner, 0.0, 1.0)
    roi = (band * 255.0).round().astype(np.uint8)
    if int((roi > 0).sum()) < 64:
        return None

    src_lab = cv2.cvtColor(original.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    dst_lab = cv2.cvtColor(swapped.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    src_mean, _ = cv2.meanStdDev(src_lab, mask=roi)
    dst_mean, _ = cv2.meanStdDev(dst_lab, mask=roi)
    offset = (dst_mean.reshape(1, 1, 3) - src_mean.reshape(1, 1, 3)).astype(np.float32)
    # 限制幅度：统计偏了也不要把遮挡物整体染色
    limit = np.array([6.0, 8.0, 8.0], dtype=np.float32).reshape(1, 1, 3)
    return np.clip(offset, -limit, limit)


def shift_lab(image: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """整幅图在 LAB 空间平移 ``offset``（float32 LAB：L 0-100，a/b -127-127）。"""
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab = lab + np.asarray(offset, dtype=np.float32).reshape(1, 1, 3)
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
    lab[..., 1:] = np.clip(lab[..., 1:], -127.0, 127.0)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(out * 255.0, 0.0, 255.0).round().astype(np.uint8)


def composite(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``mask=1`` 处取 source，``mask=0`` 处取 target。"""
    m = np.clip(mask, 0.0, 1.0)[..., None].astype(np.float32)
    out = source.astype(np.float32) * m + target.astype(np.float32) * (1.0 - m)
    return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


# ---------------- 组合入口 ----------------

def apply_occluder(
    original: np.ndarray,
    swapped: np.ndarray,
    face,
    options: MaskOptions,
    xseg_session,
) -> np.ndarray:
    """把目标图里的非人脸遮挡物（瓶子、手、头发…）贴回 ``swapped``。

    这一步必须放在画质修复**之后**：GFPGAN 处理的是整张脸的区域，会把遮挡物
    一起重新生成/糊掉；反过来先做遮挡再修复，保护就白做了。

    回贴前会先把原图的 LAB 均值对齐到换脸结果（``occluder_match``）：XSeg 的
    边界并不总是落在头发/物体上，常常切过脸颊、下颌的皮肤；如果两侧肤色有
    差异，直接贴原像素就会在脸上留下一条暗线（/亮线）。只平移均值不做方差
    对齐，遮挡物自身的纹理和对比度保持原样。
    """
    if not options.occluder:
        return swapped
    if xseg_session is None:
        raise RuntimeError("缺少 XSeg 会话，无法做遮挡保护")
    pad = (abs(int(options.occluder_dilate))
           + int(float(options.occluder_feather) * 3.0)
           + int(options.occluder_match_band) + 16)
    face_mask, coverage, (x0, y0, x1, y1) = xseg_face_mask(xseg_session, original, face, pad=pad)
    if face_mask.size == 0:
        return swapped
    protect = soften_mask(
        1.0 - face_mask,
        dilate_px=int(options.occluder_dilate),
        feather=float(options.occluder_feather),
    )
    # XSeg 只在对齐方框内有效，方框外它输出 0（=非人脸）。直接取 1-mask 会把
    # 方框外的头发/背景整片当成遮挡物，一路保护到 ROI 矩形边界后突然中断，
    # 在画面上留下一条横/竖硬边（人脸很大、人脸被画面边缘裁掉时最明显）。
    protect *= soften_mask(coverage, feather=float(options.occluder_feather))
    # 只在 ROI 内做 LAB 对齐与合成：ROI 之外一定取 swapped 原像素
    output = swapped.copy()
    dst = output[y0:y1, x0:x1]
    source = original[y0:y1, x0:x1]
    # 只回贴“换脸确实改动过”的非人脸像素（遮挡物/头发/接缝），
    # 背景里 swapped 本来就等于原图，不需要贴，也就不会被 LAB 偏移染色
    # （否则 ROI 边界会在纯色背景上留下一个可见的矩形）。
    diff = cv2.absdiff(source, dst).max(axis=2).astype(np.float32)
    changed = np.clip(diff / 4.0, 0.0, 1.0)
    restore = protect * changed
    if options.occluder_match:
        offset = lab_mean_offset(source, dst, face_mask, int(options.occluder_match_band))
        if offset is not None:
            source = shift_lab(source, offset)
    dst[:] = composite(source, dst, restore)
    return output


def _blend_polygon(out: np.ndarray, original: np.ndarray, polygon: np.ndarray, feather: float,
                   *, color_match: bool = False, contrast: float = 0.0,
                   strength: float = 1.0) -> np.ndarray:
    """在多边形外接 ROI 内做一次回贴，ROI 之外保持 ``out`` 原样。"""
    x, y, w, h = cv2.boundingRect(np.asarray(polygon, dtype=np.int32))
    x0, y0, x1, y1 = _clip_roi(x, y, x + w, y + h, int(feather * 3) + 8, original.shape)
    if x1 <= x0 or y1 <= y0:
        return out

    local = np.asarray(polygon, dtype=np.int32) - np.array([x0, y0], dtype=np.int32)
    mask = polygon_mask((y1 - y0, x1 - x0), local, feather)
    if strength < 1.0:
        mask *= float(strength)

    result = out.copy()
    dst = result[y0:y1, x0:x1]
    src = original[y0:y1, x0:x1]
    if color_match:
        src = color_transfer(src, dst, mask, contrast)
    dst[:] = composite(src, dst, mask)
    return result


def apply_mouth(original: np.ndarray, swapped: np.ndarray, face, options: MaskOptions) -> np.ndarray:
    """把目标图原来的嘴（含嘴里的物体）贴回 ``swapped``。

    外嘴多边形默认按 ``mouth_size`` 向外扩一圈，因为换脸结果里的唇形来自源
    人脸、通常比目标图的嘴更宽：只贴原嘴而不扩，会露出半圈源人脸的嘴唇，
    看起来就是“双层嘴唇”。外扩后整片嘴唇都来自原图，只剩一条落在皮肤上的
    羽化边，不会再有重唇。

    这一步放在画质修复**之前**：贴着皮肤缝的羽化边会被 GFPGAN 一并锐化，
    接缝看不出来；而瓶子这类遮挡物在修复之后再补（见 ``apply_occluder``）。
    """
    if not options.mouth_enabled():
        return swapped

    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None:
        return swapped

    out = swapped
    if options.mouth:
        out = _blend_polygon(
            out, original,
            outer_mouth_polygon(landmarks, options.mouth_size),
            options.mouth_feather,
            color_match=options.color_transfer,
            contrast=options.color_contrast,
        )

    if options.inner_mouth > 0.0:
        # 牙齿/舌头直接用原图，避免色彩迁移改变口腔颜色
        out = _blend_polygon(
            out, original,
            inner_mouth_polygon(landmarks),
            options.mouth_feather,
            strength=float(np.clip(options.inner_mouth, 0.0, 1.0)),
        )

    return out


def apply_masks(
    original: np.ndarray,
    swapped: np.ndarray,
    face,
    options: MaskOptions,
    xseg_session=None,
) -> np.ndarray:
    """一次性把遮挡物 + 嘴部贴回（等价于“两步都在修复之后”的旧顺序）。

    正常换脸流程不要直接调它，见 ``SwapBackend.swap``：嘴部在修复前、遮挡物
    在修复后。这个函数保留给离线脚本/对比用。
    """
    out = apply_occluder(original, swapped, face, options, xseg_session)
    out = apply_mouth(original, out, face, options)
    return out
