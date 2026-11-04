"""视频换脸：CLI 与 GUI 共用的逐帧处理流程，以及命令行入口。

命令行用法（项目根目录）：
    .venv\\Scripts\\python.exe -m swapbench.video --video assets\\2.mp4
    .venv\\Scripts\\python.exe -m swapbench.video --video in.mp4 --start 5 --duration 3
    .venv\\Scripts\\python.exe -m swapbench.video --video in.mp4 --limit 30 --mode off --no-restorer

流程：
    cv2.VideoCapture 逐帧读取 → 推理子进程检测 / 换脸 → 原始 BGR 帧经管道送 ffmpeg
    （libx264 编码，原视频音轨直接 copy）→ mp4。
    动画 GIF / WebP 先用 Pillow 解码并转成临时无声 mp4，再走同一流程。
    检测不到人脸或单帧失败时原样输出该帧；Ctrl+C 中断会删除半成品文件。

GUI 直接复用 ``process_video``：模型与源脸都已在常驻子进程里，不必重新加载。
需要系统 PATH 中有 ffmpeg（本模块只负责调用，不内置编码器）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

from .args import add_mask_arguments, mask_options_from, restorer_enabled
from .config import ASSETS_DIR
from .core.imaging import load_bgr
from .naming import next_output_path

# 动画图片：Pillow 解码全部帧后先转成临时 mp4，后续预览 / 换脸 / 对比
# 都复用视频流程（OpenCV 能直接读 GIF，但读不了动画 WebP）。
ANIMATED_IMAGE_EXTS = {".gif", ".webp"}
# 这些容器里的音轨通常是 AAC/MP3，可直接 copy 进 mp4；webm/mkv 等常见
# Opus/Vorbis/AC3，统一转成 AAC，保证 Windows 播放器兼容。
AUDIO_COPY_EXTS = {".mp4", ".m4v", ".mov"}


# ---------------- ffmpeg 输出 ----------------


class FFmpegWriter:
    """把原始 BGR 帧写进 ffmpeg 的 stdin，由 ffmpeg 编码成 H.264 mp4。"""

    def __init__(self, path: Path, size: tuple[int, int], fps: float,
                 audio_source: Path | None = None, start: float = 0.0,
                 duration: float | None = None, crf: int = 18,
                 audio_codec: str = "copy"):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("找不到 ffmpeg，请先安装并加入 PATH")

        width, height = size
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        if audio_source is not None:
            # 输入 0 = 原视频（只取音轨）；按相同的起点/时长裁剪，保持音画同步
            if start > 0:
                cmd += ["-ss", f"{start:.3f}"]
            if duration is not None:
                cmd += ["-t", f"{duration:.3f}"]
            cmd += ["-i", str(audio_source)]
            cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-"]
            cmd += ["-map", "0:a:0?", "-map", "1:v:0", "-c:a", audio_codec]
            if audio_codec != "copy":
                cmd += ["-b:a", "192k"]
        else:
            cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-"]

        if width % 2 or height % 2:
            # yuv420p 要求宽高为偶数，奇数尺寸补 1 像素黑边
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]

        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-shortest", str(path),
        ]

        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr,
        )

    def write(self, frame: np.ndarray) -> None:
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            self._raise_error(exc)

    def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            proc.stdin.close()
        except OSError:
            pass
        code = proc.wait()
        if code != 0:
            self._raise_error(None, code)

    def abort(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    def _raise_error(self, exc: Exception | None, code: int | None = None) -> None:
        self._stderr.seek(0)
        detail = self._stderr.read().decode("utf-8", "replace").strip()
        self.abort()
        message = f"ffmpeg 编码失败（退出码 {code}）" if code is not None else "ffmpeg 编码中断"
        if detail:
            message += "：\n" + "\n".join(detail.splitlines()[-6:])
        raise RuntimeError(message) from exc


# ---------------- 小工具 ----------------


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _unlink_retry(path: Path, attempts: int = 20, delay: float = 0.1) -> bool:
    """删除半成品；Windows 上 ffmpeg 退出后文件可能仍被短暂占用。"""
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            time.sleep(delay)
    return False


def _replace_retry(src: Path, dst: Path, attempts: int = 20, delay: float = 0.1) -> bool:
    """原子替换输出文件；被播放器 / 杀毒软件短暂占用时重试。"""
    for _ in range(attempts):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            time.sleep(delay)
    return False


# ---------------- 动画图片（GIF / WebP） ----------------

def is_animated_image(path: Path | str) -> bool:
    """扩展名属于动画图片且实际有多帧。单帧 GIF/WebP 仍按普通图片处理。"""
    path = Path(path)
    if path.suffix.lower() not in ANIMATED_IMAGE_EXTS:
        return False
    try:
        with Image.open(str(path)) as handle:
            return int(getattr(handle, "n_frames", 1) or 1) > 1
    except Exception:
        return False


def _animation_info(path: Path) -> tuple[int, int, list[int]]:
    """读动画尺寸与每帧时长；不保留像素，避免长动画占用大量内存。"""
    durations: list[int] = []
    size: tuple[int, int] | None = None
    with Image.open(str(path)) as handle:
        for frame in ImageSequence.Iterator(handle):
            durations.append(int(frame.info.get("duration", 0) or 0))
            if size is None:
                size = frame.size
    if size is None:
        raise RuntimeError(f"动画图片读不到任何帧：{path}")
    return size[0], size[1], durations


def _iter_animation_frames(path: Path):
    """逐帧解码为 BGR；透明区域合成到白底，避免出现黑边。"""
    with Image.open(str(path)) as handle:
        for frame in ImageSequence.Iterator(handle):
            rgba = frame.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(background, rgba).convert("RGB")
            yield cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _animation_fps(durations: list[int], frame_count: int) -> float:
    """按总时长折算平均帧率（GIF/WebP 每帧时长可能不同）。"""
    previous = 100
    total_ms = 0
    for value in durations:
        value = int(value or 0)
        if value <= 0:
            value = previous
        previous = value
        total_ms += value
    if total_ms <= 0 or frame_count <= 0:
        return 10.0
    return max(1.0, min(60.0, frame_count * 1000.0 / total_ms))


def animation_to_mp4(path: Path | str, out: Path | str) -> VideoMeta:
    """把动画图片写成无声 mp4（Pillow 解码 → ffmpeg 编码）。

    输出让 GUI / CLI 后续完全复用视频流程；返回与 ``probe_video`` 相同
    形式的元数据，供调用方直接使用。两遍扫描：第一遍只取尺寸/时长，
    第二遍逐帧写入，长动画也不会把全部帧堆在内存里。
    """
    path, out = Path(path), Path(out)
    width, height, durations = _animation_info(path)
    fps = _animation_fps(durations, len(durations))
    writer = FFmpegWriter(out, (width, height), fps, audio_source=None)
    count = 0
    try:
        for frame in _iter_animation_frames(path):
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            count += 1
        if count == 0:
            raise RuntimeError(f"动画图片读不到任何帧：{path}")
        writer.close()
    except BaseException:
        writer.abort()
        _unlink_retry(out)
        raise
    return VideoMeta(width, height, fps, count, count / fps)


def _clear_progress() -> None:
    sys.stdout.write("\r" + " " * 79 + "\r")
    sys.stdout.flush()


def _print_progress(done: int, total: int | None, speed: float, eta: float) -> None:
    if total:
        line = f"[{done:>5}/{total}] {done / total * 100:5.1f}%  {speed:5.2f} fps  剩余 {_fmt_time(eta)}"
    else:
        line = f"[{done:>5}/?]  {speed:5.2f} fps  已用 {_fmt_time(done / speed if speed > 0 else 0)}"
    sys.stdout.write("\r" + line.ljust(79))
    sys.stdout.flush()


# ---------------- 视频信息 / 选脸 ----------------


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frames: int                 # 0 = 未知
    duration: float             # 0 = 未知

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass
class VideoStats:
    done: int = 0
    swapped: int = 0
    passthrough: int = 0
    failed: int = 0
    missed: int = 0
    elapsed: float = 0.0
    cancelled: bool = False


def probe_video(path: Path | str) -> tuple[VideoMeta, np.ndarray]:
    """读取尺寸 / 帧率 / 帧数，并返回第一帧（GUI 装载预览用）。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("视频读不到任何帧（文件损坏，或起点超出时长？）")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if frames > 0 and fps > 0 else 0.0
        return VideoMeta(frame.shape[1], frame.shape[0], fps, frames, duration), frame
    finally:
        cap.release()


def face_embedding(face):
    """取 512 维归一化特征。

    insightface.Face 的 normed_embedding 是 property，不在 dict 里；
    跨进程回传的 FaceProxy 是 ``dict(f)``，只带原始 ``embedding``。
    所以这里优先取现成的归一化特征，取不到就自己归一化。
    """
    if isinstance(face, dict):
        emb = face.get("normed_embedding")
        if emb is None:
            emb = face.get("embedding")
    else:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else None


def _pick_face(faces, index: int, track_embedding, threshold: float, fallback: bool = False):
    """按特征相似度挑同一个人；没有特征或匹配不上时返回 None（沿用上一帧）。

    ``fallback=True``（还没有任何可用人脸时）退回按面积序号选，避免跟踪阈值
    偏严导致整段视频一张脸都没换。
    """
    if not faces:
        return None
    if track_embedding is None:
        return faces[min(int(index), len(faces) - 1)]
    best, best_sim = None, -1.0
    for face in faces:
        emb = face_embedding(face)
        if emb is None:
            continue
        sim = float(np.dot(emb, track_embedding))
        if sim > best_sim:
            best, best_sim = face, sim
    if best is not None and best_sim >= threshold:
        return best
    return faces[min(int(index), len(faces) - 1)] if fallback else None


def _bbox_iou(a, b) -> float:
    """两个人脸框的交并比；坐标是 [x1, y1, x2, y2]。"""
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in a)
        bx1, by1, bx2, by2 = (float(v) for v in b)
    except Exception:
        return 0.0
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
             + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


# ---------------- 逐帧处理（CLI / GUI 共用） ----------------


def process_video(
    video: Path | str,
    out: Path | str,
    source_face,
    backend,
    *,
    options=None,
    use_restorer: bool = True,
    face_index: int = 0,
    track_embedding=None,
    track_threshold: float = 0.30,
    det_every: int = 1,
    start: float = 0.0,
    duration: float | None = None,
    limit: int | None = None,
    crf: int = 18,
    keep_audio: bool = True,
    audio_codec: str | None = None,
    on_progress=None,
    on_error=None,
    should_cancel=None,
) -> VideoStats:
    """逐帧检测 / 换脸并写成 mp4。

    ``backend`` 只需提供 ``detect()`` 与 ``swap()``：命令行传自己的
    ``SwapBackend``，GUI 直接传已加载模型的常驻代理，不重复加载。

    ``track_embedding`` 非空时按特征相似度跟踪同一张脸（GUI 选人用），
    否则按 ``face_index`` 的面积序号选脸（命令行 ``--face-index`` 行为）。
    ``should_cancel`` 每帧轮询；取消、异常都会删除半成品。
    """
    video, out = Path(video), Path(out)
    # 先写同目录的 .part 临时文件，成功后再原子替换：取消 / 崩溃不会毁掉
    # 上一次的成片，也不会留下半截 mp4 冒充完整输出。
    part = out.with_name(f"{out.stem}.part{out.suffix}")
    stats = VideoStats()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video}")

    writer: FFmpegWriter | None = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(start) * 1000.0)
            if total > 0:
                total = max(0, total - int(round(start * fps)))

        limit_frames = int(limit) if limit and limit > 0 else (total if total > 0 else None)
        if duration is not None:
            dur_frames = max(1, int(round(float(duration) * fps)))
            limit_frames = dur_frames if limit_frames is None else min(limit_frames, dur_frames)

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("视频读不到任何帧（起点是否超出时长？）")
        height, width = frame.shape[:2]

        writer = FFmpegWriter(
            part, (width, height), fps,
            audio_source=video if keep_audio else None,
            start=float(start), duration=duration, crf=crf,
            audio_codec=audio_codec or (
                "copy" if video.suffix.lower() in AUDIO_COPY_EXTS else "aac"),
        )

        track = None if track_embedding is None else np.asarray(track_embedding, dtype=np.float32)
        last_face = None
        started = time.perf_counter()
        while limit_frames is None or stats.done < limit_frames:
            if stats.done > 0:
                ok, frame = cap.read()
                if not ok:
                    break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            if should_cancel is not None and should_cancel():
                stats.cancelled = True
                break

            stats.done += 1
            face = last_face
            # 第 1 帧必检测；之后每 det_every 帧检测一次；还没有可用脸时逐帧尝试
            if last_face is None or (stats.done - 1) % max(1, int(det_every)) == 0:
                faces = backend.detect(frame)
                chosen = _pick_face(faces, face_index, track, track_threshold,
                                    fallback=last_face is None)
                if chosen is not None:
                    face = chosen
                    last_face = chosen
                elif not faces:
                    # 检测器这一帧没结果：沿用上一帧的位置，避免整段闪断
                    stats.missed += 1
                    face = last_face
                elif track is not None and last_face is not None:
                    # 有脸但特征对不上：看是否还在原来的位置（转头/低分辨率时
                    # 特征会退化），位置也对不上就认为跟踪目标已经离开画面。
                    near = max(faces, key=lambda f: _bbox_iou(f.bbox, last_face.bbox))
                    stats.missed += 1
                    if _bbox_iou(near.bbox, last_face.bbox) >= 0.30:
                        face = last_face = near
                    else:
                        face = None
                        last_face = None
                else:
                    stats.missed += 1
                    face = last_face

            if face is None:
                result = frame
                stats.passthrough += 1
            else:
                try:
                    result = backend.swap(
                        frame, face, source_face,
                        paste_back=True, use_restorer=use_restorer, mask_options=options,
                    )
                    stats.swapped += 1
                except Exception as exc:            # 单帧失败不影响整段输出
                    stats.failed += 1
                    if on_error is not None:
                        on_error(stats.done, exc)
                    result = frame

            writer.write(result)
            if on_progress is not None:
                elapsed = time.perf_counter() - started
                speed = stats.done / elapsed if elapsed > 0 else 0.0
                eta = ((limit_frames - stats.done) / speed
                       if limit_frames is not None and speed > 0 else 0.0)
                on_progress(stats.done, limit_frames, speed, eta)

        if stats.cancelled:
            writer.abort()
            writer = None
            _unlink_retry(part)
        else:
            writer.close()
            writer = None
            if not _replace_retry(part, out):
                raise RuntimeError(f"无法写入输出文件（可能被占用）：{out}")
        stats.elapsed = time.perf_counter() - started
        return stats
    except BaseException:
        if writer is not None:
            writer.abort()
            writer = None
        _unlink_retry(part)
        raise
    finally:
        cap.release()


# ---------------- 命令行 ----------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SwapBench 视频换脸（命令行）")
    parser.add_argument("--video", required=True,
                        help="目标视频或动画图片（mp4 / webm / mov / gif / webp 等）")
    parser.add_argument("--source", default=str(ASSETS_DIR / "人脸.jpg"),
                        help="源人脸图片（效果最好的是清晰正脸）")
    parser.add_argument("--out", default=None,
                        help="输出 mp4；默认与原视频同目录，命名为 <原名>_<n>.mp4（n 从 0 起取第一个未占用编号）")
    parser.add_argument("--face-index", type=int, default=0,
                        help="目标视频有多张脸时替换第几张（按面积从大到小，0=最大）")
    parser.add_argument("--det-every", type=int, default=1,
                        help="每 N 帧重新检测一次人脸（默认 1=每帧；调大可提速，快速移动时可能抖动）")
    parser.add_argument("--start", type=float, default=0.0, help="从第几秒开始（默认 0）")
    parser.add_argument("--duration", type=float, default=None, help="只处理多少秒（默认整段）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少帧，0=不限（快速测试用）")
    parser.add_argument("--crf", type=int, default=18, help="x264 质量，越小越清晰（默认 18）")
    parser.add_argument("--no-audio", action="store_true", help="不保留原视频音轨")
    add_mask_arguments(parser)
    return parser.parse_args()


def main() -> int:
    if hasattr(signal, "SIGBREAK"):
        # Windows 的 Ctrl+Break 默认直接结束进程；转成 KeyboardInterrupt，
        # 才能走 finally 删除半成品（Ctrl+C 本身已经是 KeyboardInterrupt）。
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    args = parse_args()
    video = Path(args.video)
    if not video.is_file():
        print(f"找不到视频：{video}", file=sys.stderr)
        return 2
    source_path = Path(args.source)
    if not source_path.is_file():
        print(f"找不到源人脸图片：{source_path}", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("找不到 ffmpeg，请先安装并加入 PATH（视频编码用）", file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else next_output_path(video, ".mp4")
    if out.resolve() == video.resolve():
        print("输出路径不能与输入视频相同（ffmpeg 会先清空输出）", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)

    options = mask_options_from(args)
    use_restorer = restorer_enabled(args)
    tempdir: Path | None = None
    media = video
    try:
        if is_animated_image(video):
            # 动画图片先转成临时 mp4（无声），后续完全复用视频流程
            tempdir = Path(tempfile.mkdtemp(prefix="swapbench_anim_"))
            media = tempdir / "frames.mp4"
            print(f"动画图片：{video.name} → 逐帧视频…")
            animation_to_mp4(video, media)

        try:
            meta, _ = probe_video(media)
        except Exception as exc:
            print(f"失败：{exc}", file=sys.stderr)
            return 2

        print(f"输入：{video.name}  {meta.width}x{meta.height}  {meta.fps:.3f} fps  "
              + (f"共 {meta.frames} 帧 / {meta.duration:.2f}s" if meta.frames else "帧数未知"))
        print(f"输出：{out}")
        print(f"遮罩选项：{options.to_dict()}")

        from .runtime.client import SwapBackend

        backend = SwapBackend()
        errors = 0

        def on_error(done: int, exc: Exception) -> None:
            nonlocal errors
            errors += 1
            if errors <= 3:
                _clear_progress()
                print(f"第 {done} 帧换脸失败，原样输出：{exc}", file=sys.stderr)

        try:
            backend.load((args.det_size, args.det_size))
            print(f"provider={backend.provider}")

            source_faces = backend.detect(load_bgr(str(source_path)))
            if not source_faces:
                print("源人脸图片里没有检测到人脸", file=sys.stderr)
                return 2
            stats = process_video(
                media, out, source_faces[0], backend,
                options=options, use_restorer=use_restorer,
                face_index=args.face_index, det_every=args.det_every,
                start=args.start, duration=args.duration, limit=args.limit,
                crf=args.crf, keep_audio=not args.no_audio and media == video,
                on_progress=_print_progress, on_error=on_error,
            )
        except KeyboardInterrupt:
            _clear_progress()
            print("已中断。", file=sys.stderr)
            return 130
        except Exception as exc:
            _clear_progress()
            print(f"失败：{exc}", file=sys.stderr)
            return 1
        finally:
            backend.close()

        _clear_progress()
        speed = stats.done / stats.elapsed if stats.elapsed > 0 else 0.0
        print("─" * 60)
        if stats.cancelled:
            print("已取消，半成品已删除。", file=sys.stderr)
            return 130
        print(f"完成：{stats.done} 帧 / {stats.elapsed:.1f}s  （{speed:.2f} fps）")
        print(f"换脸 {stats.swapped} · 原样输出 {stats.passthrough} · "
              f"换脸失败 {stats.failed} · 检测缺失 {stats.missed}")
        print(f"输出：{out}")
        return 0
    finally:
        if tempdir is not None:
            shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
