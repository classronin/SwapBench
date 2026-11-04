"""SwapBench 换脸的 pyglet 图形界面。

- 全部控件自绘（pyglet 没有控件库），retained 模式 + 脏标记重建；
- 推理走 swapbench.runtime.client.SwapBackend 代理：实际加载/检测/换脸在
  swapbench.runtime.worker 子进程执行（会话创建的 C++ 调用握 GIL 数百 ms，
  同进程会冻结 UI），结果经管道回传、回调队列回主线程；
- 中央对比视图内嵌 swapbench.ui.compare.DiffViewer（滚轮缩放 / 拖动平移 /
  拖分割线 / 圆形对比）。

拖放：图片拖到左侧「源人脸」面板 → 加载源人脸；拖到中间「目标」区域 → 加载目标图、
视频或动画图片（GIF / WebP）。视频与动画图片会切到视频模式：中间可播放目标 / 成片 /
同步对比，右侧选人脸后「开始换脸」逐帧处理，处理中可取消。
按键：图片模式 1/2 切视图、3=保存；视频模式 1/2 切目标/成片、C=对比、空格播放/暂停、3=另存、O=系统播放器。
ESC 退出，Home/Space/+/- 作用于对比视图（视频模式下 Space=播放/暂停）。
"""

from __future__ import annotations

import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

import cv2
import pyglet
from pyglet import gl, shapes
from pyglet.window import key, mouse

from ..core.imaging import face_box, load_bgr, save_bgr
from ..naming import next_output_path
from ..runtime.client import SwapBackend
from ..config import DETECTION_SIZES, GFPGAN_FILE, MODEL_ROOT, PACK_NAME, XSEG_FILE
from ..video import (
    animation_to_mp4,
    face_embedding,
    is_animated_image,
    process_video,
    probe_video,
)
from .compare import DiffViewer, bgr_to_texture

# ---------------- 外观常量 ----------------

FONT = "Microsoft YaHei UI"
FONT_MONO = "Consolas"

PANEL = (40, 41, 44)
PANEL_DARK = (33, 34, 37)
VIEW_BG = (22, 23, 25)
BTN = (60, 62, 66)
BTN_HOVER = (76, 79, 85)
BTN_DISABLED = (47, 48, 51)
ACCENT = (240, 152, 48)            # 主强调色：橙色（按钮 / 进度条 / 滑条 / 选中态）
ACCENT_HOVER = (212, 130, 36)      # 悬停时略深
CHIP = (58, 60, 66)
BORDER = (58, 61, 66)
TXT = (212, 216, 221)
TXT_DIM = (146, 152, 159)
TXT_FAINT = (112, 118, 124)
TXT_DISABLED = (118, 122, 127)
TXT_ON_ACCENT = (20, 20, 20)       # 橙色底上用黑字
TRACK = (48, 50, 54)
PROGRESS_TEXT = (255, 170, 56)     # 画面顶部进度文字：橙色
PROGRESS_BG = (18, 18, 20, 205)    # 半透明深色底，保证压在视频上也清晰

MODE_TARGET = "目标"
MODE_COMPARE = "对比"
MODE_RESULT = "结果"          # 不再单独提供"结果"入口，仅供内部使用
MODES = [MODE_TARGET, MODE_COMPARE]

LEFT_W = 290
RIGHT_W = 200
PAD = 10
GAP = 8

FACE_THUMB = 44            # 人脸卡片缩略图边长
FACE_CARD_H = 52

# 右栏遮罩细调滑条：(标题, 属性名, 最小, 最大, 步长, 归属开关)
# 归属开关决定滑条是否可动（「遮挡保护」关着时遮挡两条变灰）。
SLIDER_ROW_H = 22
MASK_SLIDERS = (
    ("嘴部范围", "mask_mouth_size", 0.0, 60.0, 1.0, "mouth"),
    ("嘴部羽化", "mask_mouth_feather", 0.0, 12.0, 0.5, "mouth"),
    ("内嘴混合", "mask_inner", 0.0, 1.0, 0.05, "mouth"),
    ("遮挡内缩", "mask_occ_dilate", -6.0, 10.0, 1.0, "occluder"),
    ("遮挡羽化", "mask_occ_feather", 0.0, 16.0, 0.5, "occluder"),
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}


# ---------------- 文件对话框（tkinter，独立于界面） ----------------

_TK_ROOT = None


def _tk_root():
    global _TK_ROOT
    if _TK_ROOT is None:
        import tkinter as tk

        _TK_ROOT = tk.Tk()
        _TK_ROOT.withdraw()
    try:
        _TK_ROOT.attributes("-topmost", True)
        _TK_ROOT.update()
        _TK_ROOT.attributes("-topmost", False)
    except Exception:
        pass
    return _TK_ROOT


def _ask_save_image(default_name: str) -> str:
    from tkinter import filedialog

    return filedialog.asksaveasfilename(
        parent=_tk_root(), title="保存结果",
        defaultextension=".png", initialfile=default_name,
        filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")],
    )


def _ask_save_video(default_name: str) -> str:
    from tkinter import filedialog

    return filedialog.asksaveasfilename(
        parent=_tk_root(), title="另存视频",
        defaultextension=".mp4", initialfile=default_name,
        filetypes=[("MP4", "*.mp4")],
    )


def _fmt_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ---------------- 命中区域 ----------------


class Zone:
    """可交互控件：矩形 + 动作。"""

    __slots__ = ("rect", "kind", "cb", "enabled", "hover", "tag")

    def __init__(self, rect, kind: str, cb, enabled: bool = True, tag: str = ""):
        self.rect = rect          # (x, y_bottom, w, h)，窗口坐标、y 从底部
        self.kind = kind          # button / segment / switch / card / slider
        self.cb = cb
        self.enabled = enabled
        self.hover = False
        self.tag = tag            # 滑条等需要回查自身状态的控件用

    def hit(self, x: float, y: float) -> bool:
        rx, ry, rw, rh = self.rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh


# ---------------- 主窗口 ----------------


class FaceSwapPygletApp(pyglet.window.Window):
    def __init__(self):
        # 事件回调（on_resize / on_deactivate / on_draw…）可能在 super().__init__()
        # 内部就被触发（set_visible → _on_internal_resize，见 pyglet window/__init__.py
        # 598 行），所以这些回调依赖的实例状态必须先于 super() 初始化。否则首次
        # resize/deactivate 会 AttributeError：单窗口时异常恰好发生在 WndProc 的
        # ctypes 回调里被静默吞掉（看似正常实则 _relayout 失败），同进程创建
        # 第二个窗口时则直接崩溃。
        # 数据状态
        self.source_img = None
        self.source_path: Path | None = None
        self.source_face = None
        self.target_img = None
        self.target_path: Path | None = None
        self.result_img = None
        self.target_faces: list = []
        self.face_thumbs: list = []          # [texture | None]，与 target_faces 对齐
        self.sel_face_idx = 0
        self.det_idx = list(DETECTION_SIZES).index("中 640")
        self.mode = MODE_TARGET
        self.paste_on = True
        self.gfp_on = True
        self.mask_occluder = XSEG_FILE.is_file()
        self.mask_mouth = True
        # 遮罩细调（右栏滑条）——默认值按“嘴里的杯子”这类遮挡场景调过：
        # 外嘴扩 30 盖掉换脸结果带来的唇形差异，羽化 6 交给 GFPGAN 收边；
        # 遮挡保护羽化加大到 8，让接缝不落在皮肤上留下硬边。
        self.mask_mouth_size = 30.0
        self.mask_mouth_feather = 6.0
        self.mask_inner = 0.0
        self.mask_occ_dilate = 2.0
        self.mask_occ_feather = 12.0
        self._restorer_warned = False

        # 视频模式：中间区域拖入视频后，target_img 只保留首帧用于选脸，
        # 播放 / 换脸走 VideoCapture + process_video（不经过图片流程）。
        self.video_mode = False
        self.video_path: Path | None = None      # 实际解码用的文件（动画图片为临时 mp4）
        self.video_origin: Path | None = None    # 用户拖入的原始文件（决定成片名字）
        self._video_tempdir: Path | None = None  # 动画图片转换出的临时目录
        self.video_meta = None
        self.video_first = None
        self.video_result: Path | None = None
        self._video_view = "orig"                 # "orig" 原视频 / "swap" 成片 / "compare" 对比
        self._video_cap = None
        self._video_cap2 = None                   # 对比视图的右路（成片）
        self._video_playing = False
        self._video_clock = False
        self._video_cancel: threading.Event | None = None
        self._video_stats = None
        self._video_job = False                   # 视频任务进行中：进度只由总帧数驱动

        # 页脚状态
        self.progress = 0.0
        self.progress_label = "待命"
        self.chip_model = "模型：未加载"
        self.chip_device = "设备：—"
        self.chip_swap = "换脸：未加载"
        self.log_lines: list[str] = []
        self.face_scroll = 0.0
        self._src_gen = 0                        # 源图代际：换源图 +1，丢弃迟到回调
        self._tgt_gen = 0                        # 目标图代际：换目标图 +1，丢弃迟到回调
        self._pending: tuple[str, callable] | None = None   # busy 期间排队的任务（后者覆盖）
        self._job_label = ""                                # 当前后台任务名（区分检测 / 换脸）

        # 对比查看器（内嵌 ImageDiff 功能）：中间 = 目标图/对比，左侧 = 源人脸
        self.viewer = DiffViewer(on_hover=self._on_viewer_hover)
        self.source_viewer = DiffViewer(on_hover=self._on_viewer_hover)

        # 界面
        self._mouse = (0.0, 0.0)
        self._hover_zone: Zone | None = None
        self._cursor_name = "none"
        self.left_w: float = LEFT_W                # 左栏宽度（左中分隔条可拖动调整）
        self._divider_drag = False
        self._slider_drag: str | None = None       # 正在拖动的滑条 tag
        self._sliders: dict = {}                   # tag → 滑条绘制引用（原地更新用）
        self._list_rect: tuple = (0, 0, 0, 0)      # 人脸列表区域（滚轮用）
        self._geo: dict = {}                       # 分隔条轻量重排所需的原地更新引用
        self._progress_fill = None                 # 进度条填充（原地改宽度）
        self._progress_text = None                 # 视频处理进度文字（原地改 text）
        self._progress_bg = None                   # 进度文字背景框（2px 描边）
        self._progress_fill_track_w = 0.0
        self._zones: list[Zone] = []
        self._keep: list = []
        self._overlay_keep: list = []
        self._overlay_valid = False               # 人脸框 overlay 是否与数据同步
        self._ui_dirty = True

        # 后端 + 主线程回调队列
        self._queue: queue.Queue = queue.Queue()
        self.backend = SwapBackend(
            on_progress=self._enqueue_backend_progress
        )
        self.busy = False

        super().__init__(
            1024, 700, "SwapBench 换脸",
            resizable=True, file_drops=True, vsync=True,
        )
        self.set_minimum_size(1024, 700)
        gl.glClearColor(24 / 255, 24 / 255, 26 / 255, 1.0)

        self._layout = self._compute_layout()
        self._back = pyglet.graphics.Batch()
        self._front = pyglet.graphics.Batch()
        self._cards = pyglet.graphics.Batch()

        self._log("模型目录：" + str(MODEL_ROOT))
        self._log("拖文件：左侧=源人脸，中间=目标图/视频/动画GIF·WebP · 1/2 切视图/目标·成片 · C 视频对比，ESC 退出")
        self._log("对比视图：滚轮缩放 · 拖动平移 · 拖分割线 · Home 复位 · Space 圆形对比 · +/- 调整圆")
        self._log("遮挡保护（XSeg）：右栏开关，对下一次换脸生效")

        pyglet.clock.schedule_interval(self._poll, 0.04)
        pyglet.clock.schedule_once(lambda dt: self._start_model_load(), 0.4)

    @staticmethod
    def run() -> None:
        app = FaceSwapPygletApp()
        app._relayout()
        pyglet.app.run()

    # ---------------- 布局 ----------------

    def _compute_layout(self) -> dict:
        w, h = self.width, self.height
        panel_top = h - PAD
        panel_bottom = PAD
        lp_x, rp_x = PAD, w - RIGHT_W - PAD
        left_w = self.left_w                       # 可拖动调整（左中分隔条）
        center_x0 = lp_x + left_w + GAP
        center_x1 = rp_x - GAP
        return {
            "w": w, "h": h,
            "left": (lp_x, panel_bottom, left_w, panel_top - panel_bottom),
            "right": (rp_x, panel_bottom, RIGHT_W, panel_top - panel_bottom),
            "cx0": center_x0, "rp_x": rp_x,
            # 左侧源人脸交互画布（占满面板，随窗口自动缩放）
            "left_view": (lp_x + 8, panel_bottom + 8, left_w - 16, panel_top - 4 - (panel_bottom + 8)),
            # 底部留 46px 给「目标图 / 对比 / 保存」居中按钮条
            "viewer": (center_x0, panel_bottom + 46, center_x1 - center_x0,
                       panel_top - 6 - (panel_bottom + 46)),
        }

    def _relayout(self) -> None:
        self._layout = self._compute_layout()
        self.viewer.layout(*self._layout["viewer"])
        self.source_viewer.layout(*self._layout["left_view"])
        self._ui_dirty = True

    # ---------------- 轻量重排（分隔条拖动专用，GPU 高效路径） ----------------

    def _apply_left_w(self) -> None:
        """分隔条拖动期间的原地重排：不重建 Batch/Label。

        pyglet 效率逻辑（app/base.py + shapes.py）：
        - 事件循环每帧都会跑 window.draw()，`_ui_dirty` 触发的 _rebuild_ui
          （重建全部 VertexList + Label 文本排版）每帧执行就是 GPU 高占用的根源；
        - shapes 的 position/width/height/color setter 只改已有 vertex 数据，
          TextLayout 的 x/y 也只做平移，代价可忽略；
        - 所以拖动期间只更新必要几何 + 两个 DiffViewer 的布局（refit=False）。
        """
        if not self._geo:
            self._relayout()
            return
        lo = self._compute_layout()
        self._layout = lo
        self.viewer.layout(*lo["viewer"], refit=False)
        self.source_viewer.layout(*lo["left_view"], refit=False)
        g = self._geo

        def set_rect(shape, r):
            shape.position = (r[0], r[1])
            shape.width = r[2]
            shape.height = r[3]

        set_rect(g["left"], lo["left"])
        set_rect(g["vrect"], lo["viewer"])
        set_rect(g["lv"], lo["left_view"])
        set_rect(g["div"], (lo["left"][0] + self.left_w + GAP / 2 - 1,
                            lo["left"][1], 2, lo["h"] - 2 * lo["left"][1]))
        # 中部底部按钮条随 viewer 水平移动（宽度按当前模式实际按钮组计算）
        vx, vy, vw, vh = lo["viewer"]
        bar_h, bar_y = 30, 20
        widths = [item[3] for item in g["bar"]]
        bar_x = vx + (vw - (sum(widths) + 6 * (len(widths) - 1))) / 2
        for i, (zone, shape, lb, w) in enumerate(g["bar"]):
            x = bar_x + sum(widths[:i]) + 6 * i
            zone.rect = (x, bar_y, w, bar_h)
            set_rect(shape, zone.rect)
            lb.x = x + w / 2
            lb.y = bar_y + bar_h / 2
        if self._progress_text is not None:
            self._progress_text.x = vx + vw / 2
            self._progress_text.y = vy + vh - 10
            self._place_progress_text()
        # 空态提示跟随
        for lb, dy in g["vhint"]:
            lb.x = vx + vw / 2
            lb.y = vy + vh / 2 + dy
        if g.get("phint") is not None:
            lx0, ly0, lw0, lh0 = lo["left_view"]
            g["phint"].x = lx0 + lw0 / 2
            g["phint"].y = ly0 + lh0 / 2

    # ---------------- 日志 / 进度 / 队列 ----------------

    def _log(self, text: str) -> None:
        self.log_lines.append(f"[{time.strftime('%H:%M:%S')}] {text}")
        if len(self.log_lines) > 300:
            del self.log_lines[:-300]
        self._ui_dirty = True

    def _place_progress_text(self) -> None:
        """让深色描边底框跟随进度文字（长度变化时原地改宽高）。"""
        text, bg = self._progress_text, self._progress_bg
        if text is None or bg is None:
            return
        has_text = bool(text.text)
        bg.visible = has_text
        if not has_text:
            return
        pad_x, pad_y = 7.0, 4.0
        bg.width = text.content_width + pad_x * 2
        bg.height = text.content_height + pad_y * 2
        bg.x = text.x - bg.width / 2
        bg.y = text.y - text.content_height - pad_y

    def _set_progress(self, value: float, label_: str) -> None:
        self.progress = max(0.0, min(1.0, value))
        self.progress_label = label_
        # 视频逐帧回调会高频调用：只原地改进度条/文字，不触发整界面重建
        if self._progress_fill is not None:
            self._progress_fill.width = self._progress_fill_track_w * self.progress
        if self._progress_text is not None and self.video_mode:
            self._progress_text.text = label_
            self._place_progress_text()

    def _call(self, fn) -> None:
        self._queue.put(fn)

    def _enqueue_backend_progress(self, value: float, label_: str) -> None:
        """推理子进程的细粒度进度（换脸内部阶段）。

        视频换脸时这些回调来自每一帧的 swap（换脸 35% → 修复 → 遮罩…），
        并不是整段视频的进度；如果让它更新进度条，就会「每帧跳一次」并
        覆盖掉总帧数百分比。视频任务期间直接丢弃，只保留 process_video
        的 done/total 进度。
        """
        if self._video_job:
            return
        self._call(lambda: self._set_progress(value, label_))

    def _poll(self, _dt) -> None:
        while True:
            try:
                fn = self._queue.get_nowait()
            except queue.Empty:
                return
            fn()

    # ---------------- 任务调度 ----------------

    def _run_task(self, label_: str, job) -> None:
        if self.busy:
            # 不丢弃：排队等待（后者覆盖前者）。快速连续拖入源图+目标图时，
            # 两个检测都能跑完；旧实现直接拒绝会漏掉目标图的人脸。
            self._pending = (label_, job)
            return
        self.busy = True
        self._job_label = label_
        self._ui_dirty = True                     # 按钮切到「取消」/禁用态
        self._set_progress(0.0, label_)

        def runner():
            try:
                job()
            except Exception as exc:
                # 注意：except 块结束时 Python 会隐式 del 掉 exc（cell 被清空），
                # 而 lambda 要等主线程 _poll 时才执行 —— 必须先把值绑定到普通变量
                msg = str(exc)
                tb = traceback.format_exc(limit=3).strip()
                self._call(lambda: self._task_failed(msg, tb))

        threading.Thread(target=runner, daemon=True).start()

    def _task_failed(self, msg: str, tb: str) -> None:
        self._video_cancel = None
        self._video_job = False
        self._log(f"失败：{msg.splitlines()[0] if msg else '未知错误'}")
        if tb:
            tail = tb.splitlines()[-1]
            self._log(f"    {tail}")
        self._finish("失败")

    def _finish(self, label_: str = "待命") -> None:
        self.busy = False
        self._job_label = ""
        self.progress_label = label_
        self._ui_dirty = True
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._run_task(*pending)

    def _restorer_enabled(self) -> bool:
        if not self.gfp_on:
            return False
        if not GFPGAN_FILE.is_file():
            if not self._restorer_warned:
                self._restorer_warned = True
                self._log(f"未找到画质修复模型 {GFPGAN_FILE.name}，已跳过 GFPGAN 修复")
            return False
        return True

    # GFPGAN 开关隐藏：gfp_on 恒为 True，画质修复自动启用（缺模型时自动跳过）

    # ---------------- 模型加载 ----------------

    def _start_model_load(self) -> None:
        def runner():
            try:
                self.backend.load(self._det_size())
                self._call(self._on_model_ready)
            except Exception as exc:
                self._call(lambda: self._log(f"模型加载失败：{exc}"))

        threading.Thread(target=runner, daemon=True).start()

    def _on_model_ready(self) -> None:
        names = ", ".join(self.backend.model_names)
        device = "GPU / CUDA" if self.backend.provider.startswith("CUDA") else "CPU"
        self.chip_model = f"模型：{PACK_NAME}"
        self.chip_device = f"设备：{device}"
        self.chip_swap = "换脸：inswapper_128"
        self._log(f"模型就绪：{names}；推理设备 {device}")
        # 模型就绪前拖入的图在此补检测；两个检测经由 _pending 排队串行执行
        if self.source_img is not None and self.source_face is None:
            self._detect_source()
        if self.target_img is not None and not self.target_faces:
            self._detect_target()
        self._ui_dirty = True

    def _det_size(self) -> tuple[int, int]:
        return list(DETECTION_SIZES.values())[self.det_idx]

    def _change_det_size(self, idx: int) -> None:
        if idx == self.det_idx:
            return
        self.det_idx = idx
        name = list(DETECTION_SIZES)[idx]
        if self.backend.ready:
            # 不能在主线程做这个 IPC：worker 忙时（如检测进行中）会阻塞 UI
            # 整整一个任务周期。fire-and-forget 到后台线程，管道自身保证顺序。
            def _apply_det_size():
                try:
                    self.backend.set_det_size(self._det_size())
                except Exception as exc:
                    self._call(lambda: self._log(f"检测尺寸设置失败：{exc}"))

            threading.Thread(target=_apply_det_size, daemon=True).start()
        self._log(f"检测尺寸改为 {name}，重新选择图片后生效")
        self._ui_dirty = True

    # ---------------- 图片装载（拖放） ----------------

    # 打开文件按钮已移除：源人脸 / 目标图只通过拖放加载

    def _apply_source(self, path) -> None:
        try:
            image = load_bgr(path)
        except Exception as exc:
            self._log(f"读取源图失败：{exc}")
            return
        self.source_img = image
        self.source_path = Path(path)
        self.source_face = None
        self._src_gen += 1
        # 左侧画布装载为交互式查看器（滚轮缩放 / 拖动平移，与中间一致）
        self.source_viewer.set_images(image)
        w_, h_ = image.shape[1], image.shape[0]
        self._log(f"已选择源人脸：{self.source_path.name}  {w_}x{h_}")
        if self.backend.ready:
            self._detect_source()
        else:
            self._log("模型加载中，稍后自动识别")
        self._ui_dirty = True

    def _apply_target(self, path) -> None:
        self._clear_video()
        try:
            image = load_bgr(path)
        except Exception as exc:
            self._log(f"读取目标图失败：{exc}")
            return
        self.target_img = image
        self.target_path = Path(path)
        self.result_img = None
        self.target_faces = []
        self.sel_face_idx = 0
        self.face_thumbs = []
        self._tgt_gen += 1
        self.mode = MODE_TARGET
        self._invalidate_overlay()
        self._log(f"已选择目标图：{Path(path).name}  {image.shape[1]}x{image.shape[0]}")
        self.viewer.set_images(image)
        if self.backend.ready:
            self._detect_target()
        else:
            self._log("模型加载中，稍后自动检测人脸")
        self._ui_dirty = True

    # ---------------- 视频装载 / 播放 ----------------

    def _apply_video(self, path) -> None:
        # 检测任务在跑时允许装载视频（其结果会因代际 +1 作废）；
        # 但换脸/视频处理中不允许换素材，避免输出与界面状态错配。
        if self._job_label in ("换脸中…", "视频换脸中…"):
            self._log("当前换脸任务未结束，请等它完成后再拖入视频")
            return
        origin = Path(path)
        tempdir: Path | None = None
        media = origin
        if is_animated_image(origin):
            # 动画 GIF / WebP：Pillow 解码后先写成临时无声 mp4，预览 / 换脸 /
            # 对比全部复用视频流程（OpenCV 不能直接读动画 WebP）。
            try:
                tempdir = Path(tempfile.mkdtemp(prefix="swapbench_anim_"))
                media = tempdir / "frames.mp4"
                self._log(f"动画图片：{origin.name} → 逐帧视频…")
                animation_to_mp4(origin, media)
            except Exception as exc:
                if tempdir is not None:
                    shutil.rmtree(tempdir, ignore_errors=True)
                self._log(f"读取动画图片失败：{exc}")
                return
        try:
            meta, first = probe_video(media)
        except Exception as exc:
            if tempdir is not None:
                shutil.rmtree(tempdir, ignore_errors=True)
            self._log(f"读取视频失败：{exc}")
            return
        self._clear_video()
        self.video_mode = True
        self.video_path = media
        self.video_origin = origin
        self._video_tempdir = tempdir
        self.video_meta = meta
        self.video_first = first
        # target_img 只保留首帧：右侧人脸列表、缩略图和「第几张脸」都以它为准
        self.target_img = first
        self.target_path = origin
        self.result_img = None
        self.target_faces = []
        self.sel_face_idx = 0
        self.face_thumbs = []
        self._tgt_gen += 1
        self.mode = MODE_TARGET
        self._invalidate_overlay()
        self.viewer.set_images(first)
        name = origin.name
        kind = "动画图片" if tempdir is not None else "目标视频"
        self._log(
            f"已选择{kind}：{name}  {meta.width}x{meta.height}  {meta.fps:.2f} fps"
            + (f"  {meta.duration:.2f}s" if meta.duration else "")
        )
        if tempdir is not None:
            self._log("动画图片按视频逐帧处理，成片为无声 mp4")
        self._log("视频模式：右侧选人脸 →「开始换脸」；空格播放/暂停（应用内预览无声，成片保留音轨）")
        self._log("视图：1 目标 · 2 成片 · C 对比（可拖分割线）；换脸完成后视图/分割线不会重置")
        if self.backend.ready:
            self._detect_target()
        else:
            self._log("模型加载中，稍后自动检测人脸")
        self._ui_dirty = True

    def _clear_video(self) -> None:
        """退出视频模式：停止播放、释放 capture、清空视频状态。"""
        self._stop_playback()
        self._release_video_caps()
        if self._video_tempdir is not None:
            shutil.rmtree(self._video_tempdir, ignore_errors=True)
            self._video_tempdir = None
        self.video_mode = False
        self.video_path = None
        self.video_origin = None
        self.video_meta = None
        self.video_first = None
        self.video_result = None
        self._video_view = "orig"
        self._video_stats = None

    def _target_source_image(self):
        """目标检测/缩略图用的图像：视频模式固定用首帧，避免播放中换帧。"""
        if self.video_mode and self.video_first is not None:
            return self.video_first
        return self.target_img

    def _release_video_caps(self) -> None:
        for attr in ("_video_cap", "_video_cap2"):
            cap = getattr(self, attr)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _open_video_caps(self) -> bool:
        """按当前视图打开 1～2 路解码器；失败时保持关闭并返回 False。"""
        self._release_video_caps()
        if self.video_path is None:
            return False
        if self._video_view == "compare":
            if self.video_result is None:
                return False
            self._video_cap = cv2.VideoCapture(str(self.video_path))
            self._video_cap2 = cv2.VideoCapture(str(self.video_result))
            if self._video_cap.isOpened() and self._video_cap2.isOpened():
                return True
            self._release_video_caps()
            return False
        path = self.video_result if self._video_view == "swap" else self.video_path
        if path is None:
            return False
        self._video_cap = cv2.VideoCapture(str(path))
        if self._video_cap.isOpened():
            return True
        self._release_video_caps()
        return False

    def _video_caps_ready(self) -> bool:
        if self._video_view == "compare":
            return (self._video_cap is not None and self._video_cap2 is not None
                    and self._video_cap.isOpened() and self._video_cap2.isOpened())
        return self._video_cap is not None and self._video_cap.isOpened()

    def _read_display(self, loop: bool = False) -> bool:
        """读下一帧并显示；``loop`` 时播完回到开头（对比视图两路一起回）。

        所有更新都走查看器的原地接口，播放、切视图、换脸完成都不会
        重置缩放 / 平移 / 分割线位置。
        """
        if self._video_view == "compare":
            if self._video_cap is None or self._video_cap2 is None:
                return False
            ok1, left = self._video_cap.read()
            ok2, right = self._video_cap2.read()
            if not (ok1 and ok2) and loop:
                # 两路长度可能差几帧（-shortest）：任一结束就一起回到开头，
                # 避免右路单独回到第 0 帧造成错位。
                self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._video_cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok1, left = self._video_cap.read()
                ok2, right = self._video_cap2.read()
            if ok1 and ok2:
                self.viewer.set_pair_frames(left, right)
                return True
            if ok1 or ok2:
                # 只有一路能读（成片比原视频短）：先按单画面显示，下一帧
                # 播放循环会把两路一起拉回开头。
                self.viewer.set_single_frame(left if ok1 else right)
                return True
            return False
        if self._video_cap is None:
            return False
        ok, frame = self._video_cap.read()
        if not ok and loop:
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._video_cap.read()
        if ok:
            self.viewer.set_single_frame(frame)
        return ok

    def _set_video_view(self, view: str) -> None:
        if not self.video_mode or view == self._video_view:
            return
        if self.busy:
            self._log("视频处理中，先等它完成再切换视图")
            return
        if view in ("swap", "compare") and self.video_result is None:
            self._log("还没有换脸成片，先执行一次视频换脸")
            return
        was_playing = self._video_playing
        self._stop_playback()
        previous = self._video_view
        self._video_view = view
        if not self._open_video_caps():
            self._video_view = previous
            self._log("视频打不开了（文件可能已被移动或删除）")
            self._ui_dirty = True
            return
        self._read_display()              # 先回到当前视图的第一帧
        # 不调用 reset_view()：切视图保留用户调好的缩放 / 平移 / 分割线
        if was_playing:
            self._toggle_playback()
        self._ui_dirty = True

    def _toggle_playback(self) -> None:
        if not self.video_mode:
            return
        if self.busy:
            self._log("视频处理中，先等它完成再播放")
            return
        if self._video_playing:
            self._stop_playback()
            self._ui_dirty = True
            return
        if not self._video_caps_ready():
            if not self._open_video_caps():
                self._log("视频打不开了")
                return
        self._video_playing = True
        fps = self.video_meta.fps if self.video_meta else 25.0
        interval = max(1.0 / max(1.0, fps), 1.0 / 60.0)
        if not self._video_clock:
            pyglet.clock.schedule_interval(self._video_tick, interval)
            self._video_clock = True
        self._ui_dirty = True

    def _stop_playback(self) -> None:
        if self._video_clock:
            try:
                pyglet.clock.unschedule(self._video_tick)
            except Exception:
                pass
            self._video_clock = False
        self._video_playing = False
        self._ui_dirty = True

    def _video_tick(self, _dt) -> None:
        if not self._video_playing or not self._video_caps_ready():
            return
        # 播完回到开头循环，方便反复检查（对比视图两路同步回开头）
        if not self._read_display(loop=True):
            self._stop_playback()

    def _video_progress_text(self, done: int, total: int | None, speed: float, eta: float) -> str:
        if total:
            pct = done / total * 100
            return (f"视频换脸 {pct:.0f}% · {done}/{total} 帧 · "
                    f"{speed:.2f} fps · 剩余 {_fmt_clock(eta)}")
        return f"视频换脸 {done} 帧 · {speed:.2f} fps"

    def _on_video_progress(self, done: int, total: int | None, speed: float, eta: float) -> None:
        fraction = done / total if total else 0.0
        self._set_progress(fraction, self._video_progress_text(done, total, speed, eta))

    def _start_video_swap(self) -> None:
        if self.busy:
            return
        if not (self.backend.ready and self.source_face is not None
                and self.video_path is not None and self.target_faces):
            self._log("需要先准备源人脸和视频里检测到的人脸")
            return
        self._stop_playback()
        # 释放文件句柄：Windows 上播放器还开着成片时无法覆盖同名文件
        self._release_video_caps()
        face = self.target_faces[min(self.sel_face_idx, len(self.target_faces) - 1)]
        embedding = face_embedding(face)
        video = self.video_path
        origin = self.video_origin or video
        source_path = self.source_path
        out = next_output_path(origin, ".mp4")
        if out.resolve() == origin.resolve():
            self._log("输出路径不能与源视频相同")
            return
        keep_audio = self._video_tempdir is None
        options = self._mask_options()
        use_restorer = self._restorer_enabled()
        source_face = self.source_face
        if options:
            self._log(
                "遮罩：" + ("遮挡保护 " if options["occluder"] else "")
                + ("嘴部回贴 " if options["mouth"] else "")
                + f"(嘴 {int(options['mouth_size'])}/羽化 {options['mouth_feather']:g}"
                + f" · 遮挡 {int(options['occluder_dilate'])}/{options['occluder_feather']:g})"
            )
        self._log(f"开始视频换脸：{origin.name} → {out.name}"
                  + (f"（跟踪 #{self.sel_face_idx} 号人脸）" if embedding is not None else ""))
        cancel = threading.Event()
        self._video_cancel = cancel
        self._video_job = True

        def job():
            stats = process_video(
                video, out, source_face, self.backend,
                options=options, use_restorer=use_restorer,
                face_index=self.sel_face_idx,
                track_embedding=embedding,
                det_every=1, crf=18, keep_audio=keep_audio,
                on_progress=lambda d, t, s, e: self._call(
                    lambda: self._on_video_progress(d, t, s, e)),
                should_cancel=cancel.is_set,
            )
            self._call(lambda: self._on_video_done(stats, out, video, source_path))

        self._run_task("视频换脸中…", job)

    def _cancel_video(self) -> None:
        if self._video_cancel is not None:
            self._video_cancel.set()
            self._log("正在取消视频换脸…")

    def _on_video_done(self, stats, out: Path, video: Path, source_path: Path | None = None) -> None:
        self._video_cancel = None
        self._video_job = False
        if stats.cancelled:
            self._finish("已取消")
            self._set_progress(0.0, "已取消")
            self._log("视频换脸已取消，半成品已删除")
            if self.video_mode and self._open_video_caps():
                self._read_display()          # 回到原视频 / 上一次成片的首帧
            return
        if not self.video_mode or self.video_path != video:
            self._finish("完成")
            self._log(f"视频已更换，这次成片仍保存在：{out}")
            return
        self.video_result = out
        self._video_stats = stats
        if self._video_view != "compare":
            # 原视频视图完成后自动切到成片；本来就在对比就继续对比新成片
            self._video_view = "swap"
        self._finish("完成")
        self._set_progress(1.0, f"完成 {stats.done} 帧")
        speed = stats.done / stats.elapsed if stats.elapsed > 0 else 0.0
        self._log(f"视频换脸完成：{stats.done} 帧 / {stats.elapsed:.1f}s（{speed:.2f} fps）"
                  f" · 换脸 {stats.swapped} · 原样 {stats.passthrough}"
                  f" · 失败 {stats.failed} · 检测缺失 {stats.missed}")
        if source_path is not None and source_path != self.source_path:
            self._log("提示：处理期间源人脸换过，这次成片用的是之前的源脸")
        if self._video_tempdir is None:
            self._log(f"成片：{out}（带原音轨；应用内预览无声，可用系统播放器打开）")
        else:
            self._log(f"成片：{out}（动画图片无音轨；应用内预览无声）")
        if self._open_video_caps():
            self._read_display()
            self._toggle_playback()       # 处理完自动播放成片
        self._ui_dirty = True

    def _open_video_external(self) -> None:
        """用系统播放器打开成片（应用内预览无声，这里能听到原音轨）。"""
        path = self.video_result
        if path is None or not path.is_file():
            self._log("还没有换脸成片")
            return
        try:
            os.startfile(str(path))       # noqa: S606 - Windows 默认播放器
            self._log(f"已用系统播放器打开：{path.name}")
        except Exception as exc:
            self._log(f"打开成片失败：{exc}")

    def _save_video(self) -> None:
        if self.video_result is None or not self.video_result.is_file():
            return
        origin = self.video_origin or self.video_path
        default = next_output_path(origin, ".mp4").name if origin else "result_0.mp4"
        path = _ask_save_video(default)
        if not path:
            return
        try:
            shutil.copy2(self.video_result, path)
            self._log(f"已另存视频：{path}")
        except Exception as exc:
            self._log(f"另存视频失败：{exc}")

    # ---------------- 检测 ----------------

    def _detect_source(self) -> None:
        img, gen = self.source_img, self._src_gen

        def job():
            faces = self.backend.detect(img)
            self._call(lambda: self._on_source_detected(faces, gen))

        self._run_task("识别源人脸…", job)

    def _on_source_detected(self, faces, gen: int = -1) -> None:
        if 0 <= gen != self._src_gen:
            return                                 # 检测期间换过源图，结果作废
        self._finish()
        if not faces:
            self.source_face = None
            self._log("源图中未检测到人脸")
        else:
            self.source_face = faces[0]
            score = float(faces[0].det_score)
            face_w = int(faces[0].bbox[2] - faces[0].bbox[0])
            name = self.source_path.name if self.source_path else "源图"
            self._log(
                f"源脸识别成功：{name}｜取最大的一张：置信度 {score:.2f}，脸宽 {face_w}px（共 {len(faces)} 张脸）"
            )
        self._ui_dirty = True

    def _detect_target(self) -> None:
        img, gen = self._target_source_image(), self._tgt_gen

        def job():
            faces = self.backend.detect(img)
            self._call(lambda: self._on_target_detected(faces, gen))

        self._run_task("检测目标图中的人脸…", job)

    def _on_target_detected(self, faces, gen: int = -1) -> None:
        if 0 <= gen != self._tgt_gen:
            return                                 # 检测期间换过目标图，结果作废
        self._finish()
        self.target_faces = faces
        self.sel_face_idx = 0
        self.face_thumbs = []
        self._invalidate_overlay()
        image = self._target_source_image()
        if faces and image is not None:
            for f in faces:
                x1, y1, x2, y2 = face_box(f.bbox, image.shape)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    self.face_thumbs.append(None)
                    continue
                side = max(16, min(FACE_THUMB, crop.shape[0], crop.shape[1]))
                thumb = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
                self.face_thumbs.append(bgr_to_texture(thumb))
            target = "视频首帧" if self.video_mode else "目标图"
            self._log(f"{target}检出 {len(faces)} 张人脸，默认选中最大的一张")
        else:
            self._log("目标图中未检测到人脸")
        self._ui_dirty = True

    # ---------------- 换脸 ----------------

    def _toggle_occluder(self) -> None:
        if not self.mask_occluder and not XSEG_FILE.is_file():
            self._log("缺少 models/xseg.onnx，无法开启遮挡保护")
            return
        self.mask_occluder = not self.mask_occluder
        self._ui_dirty = True
        self._log(f"遮挡保护（XSeg）：{'开启' if self.mask_occluder else '关闭'}，对下一次换脸生效")

    def _toggle_mouth(self) -> None:
        self.mask_mouth = not self.mask_mouth
        self._ui_dirty = True
        self._log(f"嘴部回贴：{'开' if self.mask_mouth else '关'}")

    def _mask_options(self) -> dict | None:
        """按当前开关/滑条组装遮罩参数；全关时返回 None（走原来的纯换脸）。"""
        occluder = self.mask_occluder and XSEG_FILE.is_file()
        if self.mask_occluder and not occluder:
            self.mask_occluder = False
            self._ui_dirty = True
            self._log("缺少 xseg.onnx，已关闭遮挡保护")
        mouth = bool(self.mask_mouth)
        inner = float(self.mask_inner) if mouth else 0.0
        if not (occluder or mouth or inner > 0.0):
            return None
        return {
            "occluder": occluder,
            "occluder_dilate": int(round(self.mask_occ_dilate)),
            "occluder_feather": float(self.mask_occ_feather),
            "occluder_match": True,
            "occluder_match_band": 12,
            "mouth": mouth,
            "mouth_size": float(self.mask_mouth_size),
            "mouth_feather": float(self.mask_mouth_feather),
            "inner_mouth": inner,
            "color_transfer": True,
            "color_contrast": 0.0,
        }

    def _start_swap(self) -> None:
        if self.video_mode:
            self._start_video_swap()
            return
        if self.target_img is None or self.source_face is None or not self.target_faces:
            self._log("需要先同时具备源人脸和目标图中的人脸")
            return
        paste_back = self.paste_on
        use_restorer = self._restorer_enabled()
        mask_options = self._mask_options()
        if mask_options:
            self._log(
                "遮罩：" + ("遮挡保护 " if mask_options["occluder"] else "")
                + ("嘴部回贴 " if mask_options["mouth"] else "")
                + f"(嘴 {int(mask_options['mouth_size'])}/羽化 {mask_options['mouth_feather']:g}"
                + f" · 遮挡 {int(mask_options['occluder_dilate'])}/{mask_options['occluder_feather']:g})"
            )
        face = self.target_faces[min(self.sel_face_idx, len(self.target_faces) - 1)]
        target_img = self.target_img
        source_face = self.source_face
        target_path = self.target_path
        source_path = self.source_path

        def job():
            result = self.backend.swap(target_img, face, source_face, paste_back, use_restorer,
                                       mask_options=mask_options)
            self._call(lambda: self._on_swapped(result, target_path, source_path))

        self._run_task("换脸中…", job)

    def _on_swapped(self, result, target_path, source_path) -> None:
        if target_path != self.target_path or source_path != self.source_path:
            # 换脸期间用户拖入了新图：这个结果属于旧图组合，留下只会
            # 造成「新目标图 × 旧结果」的错误配对（保存也是错的内容）
            self._finish("完成")
            self._log("目标图或源脸已更换，丢弃过期的换脸结果")
            return
        self.result_img = result
        self.mode = MODE_COMPARE
        # 原地换图：保留用户已有的缩放 / 平移 / 分割线位置
        self.viewer.set_pair_frames(self.target_img, result)
        self._log(f"换脸完成 #{self.sel_face_idx}，输出 {result.shape[1]}x{result.shape[0]}，已切换到对比视图")
        self._finish("完成")

    def _save_result(self) -> None:
        if self.result_img is None:
            return
        default = (next_output_path(self.target_path, ".png").name
                   if self.target_path else "result_0.png")
        path = _ask_save_image(default)
        if not path:
            return
        try:
            save_bgr(path, self.result_img)
            self._log(f"已保存：{path}")
        except Exception as exc:
            self._log(f"保存失败：{exc}")

    def _select_face(self, index: int) -> None:
        self.sel_face_idx = index
        self._invalidate_overlay()
        self._ui_dirty = True

    def _switch_mode(self, mode: str) -> None:
        if self.video_mode:
            # 视频模式由底栏「目标 / 对比 / 保存」和 1/2/C 键控制
            return
        if mode == self.mode:
            return
        if mode == MODE_COMPARE and self.result_img is None:
            self._log("还没有换脸结果，先执行一次换脸再进入对比")
            return
        self.mode = mode
        self._invalidate_overlay()
        # 原地切图：保留缩放 / 平移 / 分割线（换脸开始后视图不再被重置）
        if mode == MODE_TARGET:
            self.viewer.set_single_frame(self.target_img)
        elif mode == MODE_RESULT:
            self.viewer.set_single_frame(
                self.result_img if self.result_img is not None else self.target_img)
        else:
            self.viewer.set_pair_frames(self.target_img, self.result_img)
        self._on_viewer_hover("none")
        self._ui_dirty = True

    # ---------------- 光标 ----------------

    def _on_viewer_hover(self, hover: str) -> None:
        self._set_cursor(hover)

    def _set_cursor(self, name: str) -> None:
        if name == self._cursor_name:
            return
        self._cursor_name = name
        try:
            if name == "hand":
                self.set_mouse_cursor(self.get_system_mouse_cursor(self.CURSOR_HAND))
            elif name == "split":
                self.set_mouse_cursor(self.get_system_mouse_cursor(self.CURSOR_SIZE_LEFT_RIGHT))
            else:
                self.set_mouse_cursor(self.get_system_mouse_cursor(self.CURSOR_DEFAULT))
        except Exception:
            pass

    # ---------------- 事件 ----------------

    def _scissor(self, r) -> None:
        """面板裁剪。glScissor 工作在 framebuffer 物理像素，而 pyglet 的
        glViewport 也用物理尺寸（window/__init__.py 872-874 行），两者必须
        一致。默认 platform DPI 模式下 framebuffer 与逻辑尺寸相同（比例=1），
        切到 scaled/stretch 模式后没有这个换算，裁剪框会整体偏小。
        """
        ms = self.get_framebuffer_size()[0] / max(self.width, 1)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(int(r[0] * ms), int(r[1] * ms), int(r[2] * ms), int(r[3] * ms))

    def on_draw(self):
        self.clear()
        if self._ui_dirty:
            self._rebuild_ui()
        self._back.draw()

        # 右侧人脸卡片（裁剪在面板区域内）
        self._scissor(self._layout["right"])
        self._cards.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

        # 左侧源人脸查看器（裁剪在左面板画布区域内）
        self._scissor(self._layout["left_view"])
        self.source_viewer.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

        # 对比查看器 + 目标图模式的人脸框
        if self.viewer.active:
            self._scissor(self._layout["viewer"])
            self.viewer.draw()
            if self.mode == MODE_TARGET and self.target_faces and not self.video_mode:
                if not self._overlay_valid:
                    self._rebuild_face_overlay()   # 仅内容变化时重建对象
                self._update_face_overlay()        # 几何每帧原地跟随视图
                self._overlay_batch.draw()
            gl.glDisable(gl.GL_SCISSOR_TEST)

        self._front.draw()

    def on_resize(self, width, height):
        super().on_resize(width, height)   # pyglet 2：必须交给父类更新视口/投影
        self._relayout()

    def on_deactivate(self):
        self.viewer.on_deactivate()
        self.source_viewer.on_deactivate()
        self._set_cursor("none")
        if self._video_playing:
            self._stop_playback()

    def on_close(self):
        # X 按钮 / ESC 都会走到这里：显式通知推理子进程退出（管道协议的
        # close 帧），随后 super() 恢复 pyglet 默认关窗行为（has_exit + close）
        self._clear_video()
        self.backend.close()
        super().on_close()

    def on_mouse_leave(self, x, y):
        # 鼠标离开窗口后 motion 不再触发，悬停态与光标就地复位，
        # 防止 ↔ / 手型图标残留在标题栏或其他窗口上
        self.viewer.state["hover"] = "none"
        self.source_viewer.state["hover"] = "none"
        self._set_cursor("none")

    # ---- 鼠标 ----

    def _in_left_view(self, x, y) -> bool:
        lx, ly, lw, lh = self._layout["left_view"]
        return lx <= x <= lx + lw and ly <= y <= ly + lh

    # —— 左中分隔条：按住拖动调整左栏宽度 ——

    def _divider_x(self) -> float:
        return self._layout["left"][0] + self.left_w + GAP / 2

    def _hit_divider(self, x: float, y: float) -> bool:
        """抓取区 = 整个面板间缝隙并向两侧各延伸 6px（含拖动中的补救判定）。"""
        lo = self._layout
        dx = self._divider_x()
        return lo["left"][0] + self.left_w - 6 <= x <= lo["cx0"] - 1 \
            and lo["left"][1] <= y <= lo["h"] - PAD

    def _max_left_w(self) -> float:
        # 保证中部查看器能放下底部按钮条（视频模式 目标/对比/保存 约 220px）
        return max(160.0, self.width - RIGHT_W - 2 * PAD - GAP - 236)

    def on_mouse_press(self, x, y, button, modifiers):
        if button != mouse.LEFT:
            return
        # 分隔条拖动优先（比 viewer / zones 先判定）
        if self._hit_divider(x, y):
            self._divider_drag = True
            self._set_cursor("split")
            return
        vx, vy, vw, vh = self._layout["viewer"]
        if vx <= x <= vx + vw and vy <= y <= vy + vh and self.viewer.active:
            self.viewer.on_press(x - vx, y - vy)
            return
        if self._in_left_view(x, y) and self.source_viewer.active:
            lx, ly, _, _ = self._layout["left_view"]
            self.source_viewer.on_press(x - lx, y - ly)
            return
        for zone in self._zones:
            if zone.hit(x, y):
                if zone.enabled:
                    if zone.kind == "slider":
                        self._slider_drag = zone.tag
                        self._slider_set(zone.tag, x)
                    else:
                        zone.cb()
                    self._ui_dirty = True
                return

    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            was_dragging = self._divider_drag
            self._divider_drag = False
            if self._slider_drag is not None:
                self._slider_drag = None
                self._ui_dirty = True
            self.viewer.on_release()
            self.source_viewer.on_release()
            if was_dragging:
                # 松手：光标按当前位置复位（停在分隔条上仍是 ↔，移开则
                # 恢复普通箭头），修复拖完后光标残留问题
                self._set_cursor("split" if self._hit_divider(x, y) else "none")

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._divider_drag:
            lp_x = self._layout["left"][0]
            new_w = max(160.0, min(self._max_left_w(), x - lp_x - GAP / 2))
            if abs(new_w - self.left_w) >= 0.25:      # 有实际变化才重排
                self.left_w = new_w
                self._apply_left_w()                   # 原地更新，不走全量重建
            return
        if self._slider_drag is not None:
            self._slider_set(self._slider_drag, x)
            return
        vx, vy, vw, vh = self._layout["viewer"]
        if self.viewer.state["mode"] is not None:
            self.viewer.on_drag(x - vx, y - vy, dx, dy)
            return
        lx, ly, _, _ = self._layout["left_view"]
        if self.source_viewer.state["mode"] is not None:
            self.source_viewer.on_drag(x - lx, y - ly, dx, dy)

    def on_mouse_motion(self, x, y, dx, dy):
        # 光标策略是「电平触发」：每个 motion 事件都重算当前应有的光标并
        # 显式设置（_set_cursor 内部按名字去重，重复设置零成本）。
        # 之前按「状态变化才设置」的边沿触发 + 中途强制复位，会让 viewer
        # 的悬停回调被掐掉 —— 分割线的 ↔ 只在按住时出现、悬停不显示。
        self._mouse = (x, y)
        # 1) 左中分隔条（最前，抓取区覆盖整个面板缝隙）
        if not self._divider_drag and self._hit_divider(x, y):
            self._set_cursor("split")
            return
        vx, vy, vw, vh = self._layout["viewer"]
        lx, ly, lw, lh = self._layout["left_view"]
        # 2) 中部对比查看器（分割线 ↔ / 图上手 / 无 → 都会回调 _set_cursor）
        if vx <= x <= vx + vw and vy <= y <= vy + vh and self.viewer.active:
            self.viewer.on_motion(x - vx, y - vy)
            return
        # 3) 左侧源人脸画布
        if self._in_left_view(x, y) and self.source_viewer.active:
            self.source_viewer.on_motion(x - lx, y - ly)
            return
        # 4) 其余区域（面板边缘/死区）：查看器悬停态与光标全部复位
        if self.viewer.state["hover"] != "none":
            self.viewer.state["hover"] = "none"
        if self.source_viewer.state["hover"] != "none":
            self.source_viewer.state["hover"] = "none"
        self._set_cursor("none")

        hovered = None
        for zone in self._zones:
            zone.hover = zone.enabled and zone.hit(x, y)
            if zone.hover:
                hovered = zone
        if hovered is not self._hover_zone:
            self._hover_zone = hovered
            self._ui_dirty = True
        self._set_cursor("hand" if hovered else "none")

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y == 0 and scroll_x == 0:
            return
        # pyglet win32 已知问题：WM_MOUSEWHEEL 上报的坐标用的是缓存的
        # 物理 _mouse_x/_mouse_y（没有像其他鼠标事件那样除以 _mouse_scale）。
        # Windows 显示缩放 ≠100% 时，滚轮命中区域会错位 —— 表现为
        # 「左/中图像滚轮缩放无反应」（其实事件落到了错误的区域）。
        # 这里统一归一成与布局一致的坐标后再做命中判定。
        ms = float(getattr(self, "_mouse_scale", 1.0) or 1.0)
        if abs(ms - 1.0) > 1e-6:
            x, y = x / ms, y / ms
        vx, vy, vw, vh = self._layout["viewer"]
        if vx <= x <= vx + vw and vy <= y <= vy + vh:
            self.viewer.on_scroll(x - vx, y - vy, scroll_y)
            return
        if self._in_left_view(x, y):
            lx, ly, _, _ = self._layout["left_view"]
            self.source_viewer.on_scroll(x - lx, y - ly, scroll_y)
            return
        rx, ry, rw, rh = self._layout["right"]
        if x >= rx and self.target_faces:
            card_h = FACE_CARD_H + 6
            list_h = float(self._list_rect[3]) or max(0.0, rh - 200)
            max_scroll = max(0.0, len(self.target_faces) * card_h - list_h)
            self.face_scroll = max(0.0, min(max_scroll, self.face_scroll - scroll_y * 44))
            self._ui_dirty = True

    # ---- 键盘 ----

    def on_key_press(self, symbol, modifiers):
        k = key
        if symbol == k.ESCAPE and not modifiers:
            # pyglet Window.on_key_press 的默认行为是 ESC → on_close 退出，
            # 但这里覆盖了默认实现，必须在显式恢复（docstring 承诺的「ESC 退出」）
            self.dispatch_event("on_close")
        elif symbol == k.HOME:
            self.viewer.on_home()
            self.source_viewer.on_home()
        elif symbol == k.SPACE:
            if self.video_mode:
                self._toggle_playback()
            else:
                self.viewer.on_space()
        elif symbol in (k.EQUAL, k.PLUS, k.NUM_ADD):
            if not self.video_mode:
                self.viewer.on_circle_zoom(1)
        elif symbol in (k.MINUS, k.NUM_SUBTRACT):
            if not self.video_mode:
                self.viewer.on_circle_zoom(-1)
        elif symbol in (k._1, k.NUM_1):
            if self.video_mode:
                self._set_video_view("orig")
            else:
                self._switch_mode(MODE_TARGET)
        elif symbol in (k._2, k.NUM_2):
            if self.video_mode:
                self._set_video_view("swap")
            else:
                self._switch_mode(MODE_COMPARE)
        elif symbol in (k._3, k.NUM_3):
            if self.video_mode:
                self._save_video()
            else:
                self._save_result()
        elif symbol == k.C:
            if self.video_mode:
                self._set_video_view("compare")
        elif symbol == k.O:
            if self.video_mode:
                self._open_video_external()

    def on_file_drop(self, x, y, paths):
        # 与 on_mouse_scroll 同样的坐标防御：win32 派发 on_file_drop 时
        # 用 DragQueryPoint 的原始坐标（_event_drop_files 1380 行只做了
        # y 翻转，没除 _mouse_scale）。默认 platform DPI 模式下 scale=1
        # 无影响，换 scaled/stretch 模式后没有这行左右判定就会错位。
        ms = float(getattr(self, "_mouse_scale", 1.0) or 1.0)
        if abs(ms - 1.0) > 1e-6:
            x, y = x / ms, y / ms
        valid = [p for p in paths if os.path.isfile(p)]
        if not valid:
            self._log("拖入的不是文件")
            return
        path = valid[0]
        ext = os.path.splitext(path)[1].lower()
        is_image = ext in IMAGE_EXTS
        is_video = ext in VIDEO_EXTS
        if not (is_image or is_video):
            self._log("不支持的文件格式（图片：jpg/png/bmp/webp/gif；视频：mp4/mov/mkv/avi/webm）")
            return
        # GIF / 动画 WebP 拖到中间按视频逐帧处理；单帧的仍按普通图片
        animated = (not is_video) and is_animated_image(path)
        if x < self._layout["cx0"]:
            if is_video:
                self._log("左侧只放源人脸图片；视频请拖到中间「目标」区域")
                return
            self._log(f"拖入源人脸：{os.path.basename(path)}")
            self._apply_source(path)
        elif x < self._layout["rp_x"]:
            if is_video or animated:
                kind = "动画图片" if animated else "目标视频"
                self._log(f"拖入{kind}：{os.path.basename(path)}")
                self._apply_video(path)
            else:
                self._log(f"拖入目标图：{os.path.basename(path)}")
                self._apply_target(path)
        else:
            self._log("请把图片/视频拖到左侧「源人脸」或中间「目标」区域")

    # ---------------- 界面重建 ----------------

    def _rebuild_ui(self) -> None:
        lo = self._layout
        w, h = lo["w"], lo["h"]
        self._back = pyglet.graphics.Batch()
        self._front = pyglet.graphics.Batch()
        self._cards = pyglet.graphics.Batch()
        self._zones = []
        # pyglet 特性：Batch/TextLayout 不持有控件对象引用，Shape/Label/Sprite
        # 被 GC 时会在 __del__ 里把顶点从 batch 删掉（pyglet/shapes.py __del__）。
        # 所以这里创建的每个对象都必须持有（_keep / _overlay_keep）。
        self._keep: list = []

        def hold(obj):
            if obj is not None:
                self._keep.append(obj)
            return obj

        def rect(batch, x, y, w_, h_, color):
            return hold(shapes.Rectangle(x, y, w_, h_, color=color, batch=batch))

        def label(batch, text, x, y, size=12, color=TXT, bold=False, anchor="left",
                  mono=False, width=None, anchor_y="bottom"):
            kw = {} if width is None else {"width": int(width), "multiline": True}
            if bold:
                kw["weight"] = "bold"
            item = pyglet.text.Label(
                text=text, font_name=FONT_MONO if mono else FONT, font_size=size,
                x=x, y=y, anchor_x=anchor, anchor_y=anchor_y, color=(*color, 255),
                batch=batch, **kw,
            )
            return hold(item)

        B, F = self._back, self._front
        # 轻量重排（分隔条拖动）所需的原地更新引用；其余仍走全量重建
        self._geo: dict = {}

        # ---- 背景 & 面板（顶部条已移除） ----
        self._geo["left"] = rect(B, *lo["left"], PANEL)
        rect(B, *lo["right"], PANEL)
        vx, vy, vw, vh = lo["viewer"]
        self._geo["vrect"] = rect(B, vx, vy, vw, vh, VIEW_BG)
        lx0, ly0, lw0, lh0 = lo["left_view"]
        self._geo["lv"] = rect(B, lx0, ly0, lw0, lh0, VIEW_BG)

        # 左中分隔条（悬停显示 ↔ 光标，可拖动调整左栏宽度）
        dxm = lo["left"][0] + self.left_w + GAP / 2
        self._geo["div"] = rect(B, dxm - 1, lo["left"][1], 2, lo["h"] - 2 * lo["left"][1], BORDER)

        # ---- 左面板：源人脸交互画布占满面板（滚轮缩放 / 拖动平移） ----
        self._geo["phint"] = None
        if self.source_img is None:
            self._geo["phint"] = label(F, "拖入源图像", lx0 + lw0 / 2, ly0 + lh0 / 2, size=14,
                                       color=TXT_FAINT, anchor="center", anchor_y="center")

        # ---- 右面板：人脸列表（顶部） + 功能组与加载状态（底部） ----
        rpx, rpy, rpw, rph = lo["right"]
        content_top = lo["h"] - PAD - 4     # 内容顶端（窗口坐标）
        rc_x = rpx + 12
        rc_w = RIGHT_W - 24

        # 遮罩开关（右栏顶部，两张小卡片）：遮挡保护 / 嘴部回贴
        row_h = 24
        row_y = content_top - row_h
        half = (rc_w - 4) // 2
        self._zone_segment(
            F, (rc_x, row_y, half, row_h), "遮挡保护",
            selected=self.mask_occluder, cb=self._toggle_occluder,
            enabled=XSEG_FILE.is_file(), size=11,
        )
        self._zone_segment(
            F, (rc_x + half + 4, row_y, half, row_h), "嘴部回贴",
            selected=self.mask_mouth, cb=self._toggle_mouth,
            size=11,
        )

        # 遮罩细调滑条（紧贴开关下方，随开关状态变灰）
        self._sliders = {}
        slider_top = row_y - 4
        for i, (text, attr, vmin, vmax, step, group) in enumerate(MASK_SLIDERS):
            enabled = self.mask_mouth if group == "mouth" else self.mask_occluder
            ry = slider_top - (i + 1) * SLIDER_ROW_H
            self._zone_slider(
                F, (rc_x, ry, rc_w, SLIDER_ROW_H), text=text, attr=attr,
                vmin=vmin, vmax=vmax, step=step, enabled=enabled,
            )
        mask_bottom = slider_top - len(MASK_SLIDERS) * SLIDER_ROW_H
        list_rect = (rpx + 8, 200, RIGHT_W - 16, (mask_bottom - 6) - 200)
        self._list_rect = list_rect
        self._build_face_cards(self._cards, list_rect)

        # —— 右侧底部功能组（贴面板底边，自下而上：开始换脸/尺寸2x2/进度/状态）——
        if self.video_mode and self.busy and self._video_cancel is not None:
            self._zone_button(F, (rc_x, 14, rc_w, 44), "取消",
                              15, self._cancel_video, accent=False, enabled=True)
        else:
            self._zone_button(F, (rc_x, 14, rc_w, 44), "开始换脸",
                              15, self._start_swap, accent=True, enabled=self._swap_ready())
        # GFPGAN 开关已隐藏，保持默认启用（gfp_on=True；缺模型自动跳过修复）

        # 检测尺寸 2x2：第一行 480/640，第二行 960/1280
        # （整组距开始换脸按钮 4px 间隙——旧布局 84/54 与按钮顶 58 重叠 4px）
        det_names = [n.split()[1] for n in DETECTION_SIZES]        # 480/640/960/1280
        bw = (rc_w - 4) // 2
        for i, name in enumerate(det_names):
            row, col = divmod(i, 2)
            rect_i = (rc_x + col * (bw + 4), 92 - row * 30, bw, 24)
            self._zone_segment(F, rect_i, name, selected=(i == self.det_idx),
                               cb=lambda i=i: self._change_det_size(i),
                               enabled=self.backend.ready)

        # 进度条（视频处理时额外在画面顶部居中显示帧数 / fps / 剩余时间；
        # 两者都在 _set_progress 里原地更新，不触发整界面重建）
        track = (rc_x, 126, rc_w, 5)
        rect(B, *track, TRACK)
        self._progress_fill_track_w = float(track[2])
        self._progress_fill = rect(B, track[0], track[1],
                                   track[2] * self.progress, track[3], ACCENT)
        text = self.progress_label if (
            self.video_mode and self.progress_label not in ("", "待命")) else ""
        self._progress_bg = hold(shapes.BorderedRectangle(
            0, 0, 1, 1, border=2, color=PROGRESS_BG,
            border_color=(*PROGRESS_TEXT, PROGRESS_BG[3]), batch=F,
        ))
        self._progress_text = label(F, text, vx + vw / 2, vy + vh - 10, size=12,
                                    color=PROGRESS_TEXT, anchor="center", anchor_y="top")
        self._place_progress_text()
        # 加载状态（自下而上：换脸 / 模型 / 设备）
        label(B, self.chip_swap, rc_x, 138, size=12, color=TXT_DIM)
        label(B, self.chip_model, rc_x, 156, size=12, color=TXT_DIM)
        label(B, self.chip_device, rc_x, 174, size=12, color=TXT_DIM)

        # ---- 中部底部：模式切换 + 播放/保存（水平居中） ----
        bar_h, bar_y = 30, 20
        bar_items: list = []                 # (zone, shape, label, width)
        if self.video_mode:
            video_ready = self.video_path is not None and not self.busy
            specs = [
                ("seg", "目标", self._video_view == "orig",
                 lambda: self._set_video_view("orig"), video_ready),
                ("seg", "对比", self._video_view == "compare",
                 lambda: self._set_video_view("compare"),
                 self.video_result is not None and not self.busy),
                ("btn", "保存", False, self._save_video,
                 self.video_result is not None and not self.busy),
            ]
        else:
            specs = []
            for m in MODES:
                specs.append(("seg", m, self.mode == m, lambda m=m: self._switch_mode(m),
                              m != MODE_COMPARE or self.result_img is not None))
            specs.append(("btn", "保存", False, self._save_result,
                          self.result_img is not None and not self.busy))

        widths = [72 if kind == "seg" else 64 for kind, *_ in specs]
        bar_w = sum(widths) + 6 * (len(widths) - 1)
        bar_x = vx + (vw - bar_w) / 2
        for i, ((kind, text, selected, cb, enabled), w) in enumerate(zip(specs, widths)):
            x = bar_x + sum(widths[:i]) + 6 * i
            rect_i = (x, bar_y, w, bar_h)
            if kind == "seg":
                z, sh, lb = self._zone_segment(
                    F, rect_i, text, selected, cb, enabled=enabled, size=13)
            else:
                z, sh, lb = self._zone_button(
                    F, rect_i, text, 12, cb, enabled=enabled)
            bar_items.append((z, sh, lb, w))
        self._geo["bar"] = bar_items

        # 对比视图底部提示（绘制在最上层）：相对视口中心的 y 偏移，轻量重排时跟随
        vhint = []
        if self.target_img is None:
            vhint.append((label(F, "拖目标图像或视频", vx + vw / 2, vy + vh / 2, size=15,
                                color=TXT_DIM, anchor="center", anchor_y="center"), 0))
        elif self.mode == MODE_COMPARE and self.result_img is None:
            vhint.append((label(F, "还没有换脸结果", vx + vw / 2, vy + vh / 2, size=14,
                                color=TXT_DIM, anchor="center", anchor_y="center"), 0))
        self._geo["vhint"] = vhint

        # 命中区域悬停态应用（重建发生在鼠标事件后，用最近鼠标位置回填 hover）
        for zone in self._zones:
            zone.hover = zone.enabled and zone.hit(*self._mouse)
        self._hover_zone = next((z for z in self._zones if z.hover), None)
        self._ui_dirty = False

    # ---- 控件构建 / 绘制 ----

    def _hold(self, obj):
        """持有 Shape/Label/Sprite，避免被 GC 时从 batch 中删除顶点。"""
        if obj is not None:
            self._keep.append(obj)
        return obj

    def _add_zone(self, rect, kind: str, cb, enabled: bool = True) -> Zone:
        zone = Zone(rect, kind, cb, enabled)
        self._zones.append(zone)
        return zone

    def _zone_button(self, batch, rect, text: str, size: int, cb,
                     accent: bool = False, enabled: bool = True, invisible: bool = False):
        zone = self._add_zone(rect, "button", cb, enabled)
        if not invisible:
            shape, label = self._draw_button(batch, zone, text, size=size, accent=accent)
            return zone, shape, label
        return zone

    def _zone_segment(self, batch, rect, text: str, selected: bool, cb,
                      enabled: bool = True, size=12):
        zone = self._add_zone(rect, "segment", cb, enabled)
        x, y, w_, h_ = rect
        if not zone.enabled:
            color, txt = BTN_DISABLED, TXT_DISABLED
        elif selected:
            color, txt = ACCENT, TXT_ON_ACCENT
        elif zone.hover:
            color, txt = BTN_HOVER, TXT
        else:
            color, txt = BTN, TXT_DIM
        shape = self._hold(shapes.Rectangle(x, y, w_, h_, color=color, batch=batch))
        label = self._hold(pyglet.text.Label(
            text, font_name=FONT, font_size=size,
            x=x + w_ / 2, y=y + h_ / 2, anchor_x="center", anchor_y="center",
            color=(*txt, 255), batch=batch,
        ))
        # 返回 (zone, shape, label)：轻量重排时原地更新顶点，不重建 UI
        return zone, shape, label

    # ---- 滑条 ----

    @staticmethod
    def _slider_t(value: float, vmin: float, vmax: float) -> float:
        if vmax <= vmin:
            return 0.0
        return min(1.0, max(0.0, (float(value) - vmin) / (vmax - vmin)))

    @staticmethod
    def _fmt_slider(value: float, step: float) -> str:
        if step >= 1.0:
            return str(int(round(value)))
        if step >= 0.1:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def _zone_slider(self, batch, rect, text: str, attr: str,
                     vmin: float, vmax: float, step: float, enabled: bool = True):
        zone = self._add_zone(rect, "slider", lambda: None, enabled)
        zone.tag = text
        x, y, w_, h_ = rect
        cy = y + h_ / 2.0
        label_w, value_w = 62, 30
        tx = x + label_w + 4
        tw = max(12.0, w_ - label_w - value_w - 4)
        value = float(getattr(self, attr))
        t = self._slider_t(value, vmin, vmax)
        color = TXT_DIM if enabled else TXT_DISABLED
        self._hold(pyglet.text.Label(
            text, font_name=FONT, font_size=10, x=x, y=cy,
            anchor_x="left", anchor_y="center", color=(*color, 255), batch=batch,
        ))
        self._hold(shapes.Rectangle(tx, cy - 2, tw, 4,
                                    color=TRACK if enabled else BTN_DISABLED, batch=batch))
        fill = self._hold(shapes.Rectangle(
            tx, cy - 2, tw * t, 4,
            color=ACCENT if enabled else BTN_DISABLED, batch=batch))
        knob = self._hold(shapes.Circle(
            tx + tw * t, cy, 4.5,
            color=TXT if enabled else TXT_DISABLED, batch=batch))
        vtxt = self._hold(pyglet.text.Label(
            self._fmt_slider(value, step), font_name=FONT, font_size=10,
            x=x + w_, y=cy, anchor_x="right", anchor_y="center",
            color=(*color, 255), batch=batch,
        ))
        self._sliders[text] = {
            "attr": attr, "fill": fill, "knob": knob, "vtxt": vtxt,
            "tx": tx, "tw": tw, "vmin": vmin, "vmax": vmax, "step": step,
        }
        return zone

    def _slider_set(self, tag: str, x: float) -> None:
        """按鼠标 x 更新滑条（原地改顶点/文本，不触发整界面重建）。"""
        entry = self._sliders.get(tag)
        if entry is None:
            return
        t = self._slider_t(x, entry["tx"], entry["tx"] + entry["tw"])
        raw = entry["vmin"] + t * (entry["vmax"] - entry["vmin"])
        step = entry["step"]
        value = round(raw / step) * step if step > 0 else raw
        value = max(entry["vmin"], min(entry["vmax"], value))
        if abs(float(getattr(self, entry["attr"])) - value) < 1e-9:
            return
        setattr(self, entry["attr"], value)
        t = self._slider_t(value, entry["vmin"], entry["vmax"])
        entry["fill"].width = max(0.0, entry["tw"] * t)
        entry["knob"].x = entry["tx"] + entry["tw"] * t
        entry["vtxt"].text = self._fmt_slider(value, step)

    def _draw_button(self, batch, zone: Zone, text: str, size=13, accent=False):
        x, y, w_, h_ = zone.rect
        if not zone.enabled:
            color, txt = BTN_DISABLED, TXT_DISABLED
        elif accent:
            color = ACCENT_HOVER if zone.hover else ACCENT
            txt = TXT_ON_ACCENT
        else:
            color = BTN_HOVER if zone.hover else BTN
            txt = TXT
        shape = self._hold(shapes.Rectangle(x, y, w_, h_, color=color, batch=batch))
        label = self._hold(pyglet.text.Label(
            text, font_name=FONT, font_size=size,
            x=x + w_ / 2, y=y + h_ / 2, anchor_x="center", anchor_y="center",
            color=(*txt, 255), batch=batch,
        ))
        return shape, label

    def _build_face_cards(self, batch, list_rect) -> None:
        fx, fy, fw, fh = list_rect
        card_h = FACE_CARD_H + 6
        count = len(self.target_faces)
        max_scroll = max(0.0, count * card_h - fh)
        self.face_scroll = max(0.0, min(max_scroll, self.face_scroll))

        for i in range(count):
            content_top = (i + 1) * card_h          # 卡片 i 在内容里的底边
            cy = fy + fh - content_top + self.face_scroll   # 窗口底部坐标
            if cy + FACE_CARD_H < fy or cy > fy + fh:
                continue
            tex = self.face_thumbs[i] if i < len(self.face_thumbs) else None
            zone = self._add_zone((fx, cy, fw, FACE_CARD_H), "card",
                                  lambda i=i: self._select_face(i))
            selected = i == self.sel_face_idx
            color = ACCENT if selected else (BTN_HOVER if zone.hover else BTN)
            self._hold(shapes.Rectangle(fx, cy, fw, FACE_CARD_H, color=color, batch=batch))
            if tex is not None:
                self._hold(pyglet.sprite.Sprite(
                    tex, x=fx + 6, y=cy + (FACE_CARD_H - tex.height) / 2, batch=batch))
            score = float(self.target_faces[i].det_score)
            self._hold(pyglet.text.Label(
                f"#{i}  {score:.2f}", font_name=FONT, font_size=12,
                x=fx + FACE_THUMB + 14, y=cy + FACE_CARD_H / 2,
                anchor_x="left", anchor_y="center",
                color=(*(TXT_ON_ACCENT if selected else TXT), 255), batch=batch,
            ))

    def _invalidate_overlay(self) -> None:
        """人脸框依赖的内容（faces 集合 / 选中项 / 模式）变化 → 下帧重建。"""
        self._overlay_valid = False

    def _rebuild_face_overlay(self) -> None:
        """目标图模式的人脸框 overlay：对象驻留 + 每帧原地更新几何。

        不能每帧重建 —— 目标视图 CPU 占用比对比视图高（6% vs 2%）的根源
        就在此：60fps 下反复 new Batch/BorderedRectangle/Label（Label 还带
        文本排版），与当年 GPU 高占用同源（pyglet 对象分配不免费）。
        """
        self._overlay_batch = pyglet.graphics.Batch()
        self._overlay_keep: list = []
        for index, face in enumerate(self.target_faces):
            color = ACCENT if index == self.sel_face_idx else (170, 170, 170)
            shape = shapes.BorderedRectangle(
                0, 0, 0, 0, border=1,
                color=(*color, 0), border_color=color, batch=self._overlay_batch,
            )
            label = pyglet.text.Label(
                str(index), font_name=FONT, font_size=11,
                x=0, y=0, anchor_x="left", anchor_y="top",
                color=(*color, 255), batch=self._overlay_batch,
            )
            self._overlay_keep.append(shape)
            self._overlay_keep.append(label)
        self._overlay_valid = True

    def _update_face_overlay(self) -> None:
        """每帧：人脸框位置/尺寸跟随当前视图（缩放平移），原地更新顶点。"""
        zoom = self.viewer.state["zoom"]
        border = min(4, max(1, int(round(1.2 * zoom))))
        for i, face in enumerate(self.target_faces):
            shape, label = self._overlay_keep[2 * i], self._overlay_keep[2 * i + 1]
            x1, y1, x2, y2 = (float(v) for v in face.bbox[:4])
            wx, wy_bottom = self.viewer.image_to_window(x1, y2)
            wx2, wy_top = self.viewer.image_to_window(x2, y1)
            shape.position = (wx, wy_bottom)
            shape.width = wx2 - wx
            shape.height = wy_top - wy_bottom
            shape.border = border
            label.x = wx + 3
            label.y = wy_bottom

    # ---- 状态小工具 ----

    def _swap_ready(self) -> bool:
        return self.backend.ready and self.source_face is not None and bool(self.target_faces) and not self.busy
