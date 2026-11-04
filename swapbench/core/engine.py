"""InsightFace 推理核心：模型加载、人脸检测、换脸、画质修复。

只在子进程（swapbench.runtime.worker）中使用；父进程通过 swapbench.runtime.client 代理访问。
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import onnxruntime
import insightface
from insightface.app import FaceAnalysis

from ..config import DEFAULT_DET_SIZE, GFPGAN_FILE, PACK_DIR, SWAPPER_FILE, XSEG_FILE
from .imaging import face_box
from .masking import MaskOptions, apply_mouth, apply_occluder

ProgressCallback = Callable[[float, str], None]


class SwapBackend:
    """InsightFace 推理封装，方法只在后台线程调用。"""

    def __init__(self, on_progress: ProgressCallback | None = None):
        self._on_progress: ProgressCallback = on_progress or (lambda value, label: None)
        self.analysis: FaceAnalysis | None = None
        self.swapper = None
        self.restorer = None
        self.xseg = None
        self.provider = "CPUExecutionProvider"
        self.det_size = DEFAULT_DET_SIZE

    @property
    def ready(self) -> bool:
        return self.analysis is not None and self.swapper is not None

    # ---------------- 模型加载 ----------------

    def providers(self) -> list[str]:
        available = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def load(self, det_size: tuple[int, int]) -> None:
        if not PACK_DIR.is_dir():
            raise FileNotFoundError(f"缺少人脸分析包：{PACK_DIR}")
        if not SWAPPER_FILE.is_file():
            raise FileNotFoundError(f"缺少换脸模型：{SWAPPER_FILE}")

        providers = self.providers()
        self.provider = providers[0]
        self._step(0.05, "加载人脸分析包…")
        self.analysis = FaceAnalysis(name=str(PACK_DIR), providers=providers)
        self.analysis.prepare(ctx_id=self._ctx_id(), det_size=det_size)
        self.det_size = det_size

        self._step(0.6, "加载换脸模型…")
        self.swapper = insightface.model_zoo.get_model(str(SWAPPER_FILE), providers=providers)
        self._step(1.0, "模型就绪")

    def set_det_size(self, det_size: tuple[int, int]) -> None:
        self.det_size = det_size
        if self.analysis is not None:
            self.analysis.prepare(ctx_id=self._ctx_id(), det_size=det_size)

    def _ctx_id(self) -> int:
        return 0 if self.provider.startswith("CUDA") else -1

    def _step(self, value: float, label: str) -> None:
        self._on_progress(value, label)

    # ---------------- 检测与换脸 ----------------

    def detect(self, image: np.ndarray):
        faces = self.analysis.get(image)
        return sorted(faces, key=lambda f: float(f.bbox[2] - f.bbox[0]) * float(f.bbox[3] - f.bbox[1]), reverse=True)

    def swap(self, target: np.ndarray, target_face, source_face, paste_back: bool, use_restorer: bool,
             mask_options: MaskOptions | None = None) -> np.ndarray:
        self._step(0.35, "换脸推理…")
        output = self.swapper.get(target, target_face, source_face, paste_back=paste_back)
        if not paste_back:
            output = output[0] if isinstance(output, tuple) else output
        # 遮罩坐标是原目标图坐标系，只有 paste_back=True（输出=原图大小）时可用
        options = mask_options if (paste_back and mask_options is not None) else None

        # 1) 嘴部回贴放在画质修复之前：接缝会被 GFPGAN 一起锐化，看不出贴片；
        #    也避免“双层嘴唇”——换脸结果的唇形比原图宽，必须整片盖掉。
        mouth_ready = options is not None and options.mouth_enabled()
        if mouth_ready:
            self._step(0.55, "嘴部回贴…")
            output = apply_mouth(target, output, target_face, options)

        if use_restorer:
            self._step(0.72, "画质修复…")
            output = self.restore_faces(output, [target_face]) if paste_back else self.restore(output)

        # 2) 修复后再贴一遍嘴唇“芯部”（小范围 / 几乎不羽化）：上面那遍的大羽化
        #    边已经被 GFPGAN 抹平，这一遍只把嘴唇本身的像素换回原图，避免修复
        #    模型把嘴唇洗淡、看起来像没有嘴唇。
        if mouth_ready and options.mouth and options.mouth_size >= 10.0:
            output = apply_mouth(target, output, target_face, options.core_mouth())

        # 3) 遮挡物（瓶子/手/头发）必须在修复之后贴回，否则会被 GFPGAN 重新糊掉
        if options is not None and options.occluder:
            self._step(0.88, "遮挡保护…")
            output = apply_occluder(target, output, target_face, options, self.ensure_xseg())

        self._step(1.0, "完成")
        return output

    # ---------------- 画质修复 ----------------

    def ensure_xseg(self):
        """XSeg 遮挡遮罩会话（懒加载，缺失时给出明确错误）。"""
        if self.xseg is not None:
            return self.xseg
        if not XSEG_FILE.is_file():
            raise FileNotFoundError(f"缺少遮挡遮罩模型：{XSEG_FILE}")
        onnxruntime.set_default_logger_severity(3)
        self.xseg = onnxruntime.InferenceSession(str(XSEG_FILE), providers=self.providers())
        return self.xseg

    def ensure_restorer(self):
        if self.restorer is not None:
            return self.restorer
        if not GFPGAN_FILE.is_file():
            raise FileNotFoundError(f"缺少画质修复模型：{GFPGAN_FILE}")
        onnxruntime.set_default_logger_severity(3)
        self.restorer = onnxruntime.InferenceSession(str(GFPGAN_FILE), providers=self.providers())
        return self.restorer

    def restore(self, image: np.ndarray) -> np.ndarray:
        """GFPGAN 固定 512x512 输入，输出尺寸还原到输入图大小。"""
        session = self.ensure_restorer()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        height, width = image.shape[:2]
        resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
        out = np.asarray(session.run([output_name], {input_name: blob})[0])
        if out.ndim == 4:
            out = out[0]
        if out.shape[0] == 3:
            out = out.transpose(1, 2, 0)
        out = out.astype(np.float32)
        if float(out.min()) < 0.0:
            out = (out + 1.0) * 0.5
        bgr = cv2.cvtColor((np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)

    def restore_faces(self, image: np.ndarray, faces) -> np.ndarray:
        """GFPGAN 只吃 512x512，所以逐张脸裁区域修复；官方 GUI 是整图缩放，大图会糊。
        区域边缘做渐变混合，否则修复块会留下矩形硬边。"""
        output = image.copy()
        for face in faces:
            x1, y1, x2, y2 = face_box(face.bbox, output.shape, 0.6)
            crop = output[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fixed = self.restore(crop)
            height, width = crop.shape[:2]
            edge = max(4, min(height, width) // 8)
            mask = np.ones((height, width), dtype=np.float32)
            mask[:, :edge] = 0.0
            mask[:, -edge:] = 0.0
            mask[:edge, :] = 0.0
            mask[-edge:, :] = 0.0
            mask = cv2.GaussianBlur(mask, (0, 0), edge / 2.0)
            blended = (crop * (1.0 - mask[..., None]) + fixed * mask[..., None]).round().astype(np.uint8)
            output[y1:y2, x1:x2] = blended
        return output
