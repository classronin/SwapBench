"""双图对比查看器（pyglet 2，内嵌组件版）。

- 分割线对比：左图原样显示，右图从分割线处叠加显示，分割线可用鼠标左右拖动。
- 圆形对比（Space 切换）：右图完整叠加，圆外部分由 shader discard，
  圆心跟随鼠标，+/- 调整圆半径。
- 鼠标滚轮缩放（以光标为锚点）、按住图像拖动平移、Home 复位视图。

与独立看图工具的差异：
- 不负责加载图片；图片通过 set_images() 由换脸工作流程提供。
- 所有渲染被裁剪在宿主给定的视口矩形内（glScissor），事件由宿主转发。
- ESC 退出窗口由宿主窗口统一处理。
"""

from __future__ import annotations

import cv2
import numpy as np
import pyglet
from pyglet import gl

ZOOM_MIN = 0.02
ZOOM_MAX = 20.0

SPLIT_HIT_RADIUS = 12         # 分割线命中判定（视口像素，宽些好抓）
SCROLL_ZOOM_FACTOR = 1.15     # 每格滚轮的缩放倍率
CIRCLE_SCALE_MIN = 0.05
CIRCLE_SCALE_MAX = 1.5


# ==================== 自定义 Shader（圆形裁剪） ====================
# vertex 部分与 pyglet Sprite 默认 shader 相同，只增加一个 v_local_pos 传出
CIRCLE_VERTEX_SOURCE = """#version 150 core
    in vec3 translate;
    in vec4 colors;
    in vec3 tex_coords;
    in vec2 scale;
    in vec3 position;
    in float rotation;

    out vec4 vertex_colors;
    out vec3 texture_coords;
    out vec2 v_local_pos;

    uniform WindowBlock
    {
        mat4 projection;
        mat4 view;
    } window;

    mat4 m_scale = mat4(1.0);
    mat4 m_rotation = mat4(1.0);
    mat4 m_translate = mat4(1.0);

    void main()
    {
        m_scale[0][0] = scale.x;
        m_scale[1][1] = scale.y;
        m_translate[3][0] = translate.x;
        m_translate[3][1] = translate.y;
        m_translate[3][2] = translate.z;
        m_rotation[0][0] =  cos(-radians(rotation));
        m_rotation[0][1] =  sin(-radians(rotation));
        m_rotation[1][0] = -sin(-radians(rotation));
        m_rotation[1][1] =  cos(-radians(rotation));

        gl_Position = window.projection * window.view * m_translate * m_rotation * m_scale * vec4(position, 1.0);

        vertex_colors = colors;
        texture_coords = tex_coords;
        v_local_pos = position.xy;
    }
"""

# fragment：use_circle==1 时按圆形 discard，否则原样贴图
CIRCLE_FRAGMENT_SOURCE = """#version 150 core
    in vec4 vertex_colors;
    in vec3 texture_coords;
    in vec2 v_local_pos;
    out vec4 final_colors;

    uniform sampler2D sprite_texture;
    uniform vec2  circle_center;
    uniform float circle_radius;
    uniform float circle_line_width;
    uniform int   use_circle;

    void main()
    {
        if (use_circle == 1) {
            float d = distance(v_local_pos, circle_center);
            if (d > circle_radius + circle_line_width) {
                discard;
            }
            if (d >= circle_radius) {
                final_colors = vec4(1.0, 1.0, 1.0, 1.0);
                return;
            }
        }
        final_colors = texture(sprite_texture, texture_coords.xy) * vertex_colors;
    }
"""


def _bgr_to_image_data(bgr: np.ndarray):
    """BGR ndarray -> 可直接上传的 ImageData（RGB、上下翻转到 GL 坐标）。

    视频逐帧播放会高频调用：这里不用 PIL，纯 numpy/cv2 转换，1080p
    大约省掉一半时间（PIL 往返 + tobytes 是主要开销）。
    """
    arr = np.asarray(bgr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb[::-1])
    return pyglet.image.ImageData(rgb.shape[1], rgb.shape[0], "RGB",
                                  rgb.tobytes(), pitch=rgb.shape[1] * 3)


def bgr_to_texture(bgr: np.ndarray):
    """ndarray(BGR uint8) -> pyglet Texture。GL 操作，只能在 pyglet 主线程调用。"""
    return _bgr_to_image_data(bgr).get_texture()


class DiffViewer:
    """对比查看器。

    视口局部坐标约定（与 ImageDiff.py 的 pan 空间保持一致）：
    - lx：距视口左边的像素（x 向右）。
    - lb：距视口底部的像素（y 向上），窗口事件的原生空间。
    - 内部平移状态 pan_y 沿用 ImageDiff 的「从顶部往下」定义，
      事件进入时通过 vh - lb 换算。
    """

    def __init__(self, on_hover=None):
        self.on_hover = on_hover or (lambda hover: None)   # "none" / "hand" / "split"
        self.vx = 0.0
        self.vy = 0.0
        self.vw = 0.0
        self.vh = 0.0

        self.img_w = 0
        self.img_h = 0
        self.has_pair = False
        self._tex_left = None
        self._tex_right = None

        self.state = {
            "zoom": 1.0,
            "fit_zoom": 1.0,
            "pan_x": 0.0,
            "pan_y": 0.0,
            "split": 0.5,
            "mode": None,           # None / "drag_view" / "drag_split"
            "hover": "none",
            "shape_mode": "split",  # "split" 左右分割 / "circle" 圆形对比
            "mouse_x": 0.0,         # 视口局部，底部为 0（圆形对比圆心用）
            "mouse_y": 0.0,
            "circle_scale": 0.25,
        }

        self.batch = pyglet.graphics.Batch()
        self.sprite_left = None
        self.sprite_right = None
        self.shape_line = None
        # 圆形裁剪 shader 惰性创建：DiffViewer 可能在宿主窗口的 GL context
        # 建立之前被创建（事件回调安全要求），而 program 需要 current context。
        self.circle_program = None

    def _ensure_program(self) -> None:
        if self.circle_program is None:
            self.circle_program = gl.current_context.create_program(
                (CIRCLE_VERTEX_SOURCE, "vertex"),
                (CIRCLE_FRAGMENT_SOURCE, "fragment"),
            )

    # ---------------- 数据装载 ----------------

    @property
    def active(self) -> bool:
        return self.img_w > 0 and self.img_h > 0

    def layout(self, vx: float, vy: float, vw: float, vh: float, refit: bool = True) -> None:
        """设置视口矩形（窗口坐标、y 从底部算起）。

        refit=True（窗口改变尺寸）时：重新适配图像并复位视图。
        refit=False（分隔条拖动等轻量重排）时：只平移/缩放已有精灵到新视口，
        保留用户的缩放与平移（pyglet 效率路径：只改顶点，不重建）。
        """
        old = (self.vx, self.vy, self.vw, self.vh)
        moved = old != (vx, vy, vw, vh)
        if not moved:
            return
        resized = (self.vw, self.vh) != (vw, vh)
        self.vx, self.vy, self.vw, self.vh = vx, vy, vw, vh
        if moved and not self.active:
            return
        if resized and refit:
            self.state["fit_zoom"] = max(
                ZOOM_MIN, min(vw / self.img_w, vh / self.img_h, 1.0)
            )
            self.reset_view()
        else:
            self._update_sprites()

    def set_images(self, left_bgr, right_bgr=None) -> None:
        """装载对比图（GL 操作，必须在 pyglet 主线程调用）。

        右图缺省时只显示左图；右图尺寸与左图不一致时缩放到左图大小。
        """
        if left_bgr is None:
            self.clear()
            return
        self.has_pair = right_bgr is not None
        if right_bgr is None:
            right_bgr = left_bgr
        elif right_bgr.shape[:2] != left_bgr.shape[:2]:
            height, width = left_bgr.shape[:2]
            right_bgr = cv2.resize(right_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

        self._tex_left = bgr_to_texture(left_bgr)
        self._tex_right = bgr_to_texture(right_bgr)
        self.img_w = int(left_bgr.shape[1])
        self.img_h = int(left_bgr.shape[0])

        # 新图：重建 batch、复位视图，避免残留旧的 sprite 几何
        self.batch = pyglet.graphics.Batch()
        self.sprite_left = self.sprite_right = self.shape_line = None
        self.state["shape_mode"] = "split"
        if self.vw > 0 and self.vh > 0:
            self.state["fit_zoom"] = max(
                ZOOM_MIN, min(self.vw / self.img_w, self.vh / self.img_h, 1.0)
            )
        self.reset_view()

    def _same_size(self, bgr) -> bool:
        height, width = bgr.shape[:2]
        return (self.active and self._tex_left is not None
                and width == self.img_w and height == self.img_h)

    def set_frame(self, bgr) -> None:
        """只替换左图纹理（视频逐帧播放用），保留缩放 / 平移 / 分割状态。

        尺寸不变时原地 ``blit_into``，不重建 batch 也不复位视图；尺寸变化
        （例如换了视频）退回 ``set_images``。
        """
        if bgr is None:
            return
        if not self._same_size(bgr):
            self.set_images(bgr)
            return
        self._tex_left.blit_into(_bgr_to_image_data(bgr), 0, 0, 0)

    def set_single_frame(self, bgr) -> None:
        """视频单画面更新：替换左图并隐藏右图，保留缩放 / 平移 / 分割线。

        与 ``set_frame`` 的唯一区别是会把 ``has_pair`` 置回 False；用于在
        「原视频 / 成片 / 对比」之间切换时不重置用户已经调好的视图。
        """
        if bgr is None:
            return
        if not self._same_size(bgr):
            self.set_images(bgr)
            return
        self._tex_left.blit_into(_bgr_to_image_data(bgr), 0, 0, 0)
        if self.has_pair:
            self.has_pair = False
            self._update_sprites()

    def set_pair_frames(self, left, right) -> None:
        """视频 / 图片对比原地更新：左右纹理都替换，保留缩放 / 平移 / 分割线。

        尺寸不变时只做两次 ``blit_into``，不重建 batch、不复位视图；任一侧
        尺寸变化（换了素材）才退回 ``set_images``。
        """
        if left is None or right is None:
            if left is not None:
                self.set_single_frame(left)
            return
        if not self._same_size(left) or self._tex_right is None:
            self.set_images(left, right)
            return
        height, width = left.shape[:2]
        if right.shape[:2] != (height, width):
            right = cv2.resize(right, (width, height), interpolation=cv2.INTER_LINEAR)
        self._tex_left.blit_into(_bgr_to_image_data(left), 0, 0, 0)
        self._tex_right.blit_into(_bgr_to_image_data(right), 0, 0, 0)
        if not self.has_pair:
            self.has_pair = True
            self._update_sprites()

    def clear(self) -> None:
        self.img_w = self.img_h = 0
        self.has_pair = False
        self._tex_left = self._tex_right = None
        self.batch = pyglet.graphics.Batch()
        self.sprite_left = self.sprite_right = self.shape_line = None

    # ---------------- 视图状态 ----------------

    def reset_view(self) -> None:
        """Home：适配视口，分割线归中，圆半径复位（与 ImageDiff 一致）。"""
        if not self.active:
            return
        s = self.state
        draw_w = self.img_w * s["fit_zoom"]
        draw_h = self.img_h * s["fit_zoom"]
        s["pan_x"] = (self.vw - draw_w) / 2
        s["pan_y"] = (self.vh - draw_h) / 2
        s["zoom"] = s["fit_zoom"]
        s["split"] = 0.5
        s["circle_scale"] = 0.25
        s["mode"] = None
        s["hover"] = "none"
        self.on_hover("none")
        self._update_sprites()

    def _geometry(self):
        s = self.state
        zoom = s["zoom"]
        draw_w = self.img_w * zoom
        draw_h = self.img_h * zoom
        px, py = s["pan_x"], s["pan_y"]
        split_x = px + draw_w * s["split"]
        draw_y = self.vh - py - draw_h  # 底部为 0 的 y
        return zoom, draw_w, draw_h, px, py, split_x, draw_y

    def _update_sprites(self) -> None:
        """把当前变换刷到 sprite 几何上（ImageDiff.py 的 redraw()）。"""
        if not self.active:
            return
        zoom, draw_w, draw_h, px, py, split_x, draw_y = self._geometry()
        s = self.state

        if self.sprite_left is None:
            self.sprite_left = pyglet.sprite.Sprite(
                self._tex_left, x=self.vx + px, y=self.vy + draw_y,
                subpixel=True, batch=self.batch,
            )
        self.sprite_left.x = self.vx + px
        self.sprite_left.y = self.vy + draw_y
        self.sprite_left.scale = zoom

        if not self.has_pair:
            if self.sprite_right is not None:
                self.sprite_right.visible = False
            if self.shape_line is not None:
                self.shape_line.visible = False
            return

        if self.sprite_right is None:
            self.sprite_right = pyglet.sprite.Sprite(
                self._tex_right, x=self.vx + split_x, y=self.vy + draw_y,
                subpixel=True, batch=self.batch, program=self.circle_program,
            )

        vl = self.sprite_right._vertex_list
        if s["shape_mode"] == "circle":
            # 圆形模式：右图完整叠在左图上，圆外丢弃；分割线隐藏
            self.sprite_right.visible = True
            self.sprite_right.x = self.vx + px
            self.sprite_right.y = self.vy + draw_y
            self.sprite_right.scale = zoom
            vl.position[:] = (
                0, 0, 0,
                self.img_w, 0, 0,
                self.img_w, self.img_h, 0,
                0, self.img_h, 0,
            )
            vl.tex_coords[:] = (
                0.0, 0.0, 0,
                1.0, 0.0, 0,
                1.0, 1.0, 0,
                0.0, 1.0, 0,
            )
            if self.shape_line is not None:
                self.shape_line.visible = False
        else:
            # 分割模式：右图只画分割线右侧，绘制白色分割线
            self.sprite_right.visible = True
            self.sprite_right.x = self.vx + split_x
            self.sprite_right.y = self.vy + draw_y
            self.sprite_right.scale = zoom
            w_right = self.img_w * (1.0 - s["split"])
            vl.position[:] = (
                0, 0, 0,
                w_right, 0, 0,
                w_right, self.img_h, 0,
                0, self.img_h, 0,
            )
            u_min = s["split"]
            vl.tex_coords[:] = (
                u_min, 0, 0,
                1.0, 0, 0,
                1.0, 1.0, 0,
                u_min, 1.0, 0,
            )
            if self.shape_line is None:
                self.shape_line = pyglet.shapes.Rectangle(
                    self.vx + split_x, self.vy + draw_y, 1, draw_h,
                    color=(255, 255, 255), batch=self.batch,
                )
                self.shape_line.opacity = 220
            else:
                self.shape_line.visible = True
                self.shape_line.x = self.vx + split_x
                self.shape_line.y = self.vy + draw_y
                self.shape_line.width = 1
                self.shape_line.height = draw_h

    # ---------------- 事件（宿主转发，全部为视口局部坐标 lb=距底部） ----------------

    def on_scroll(self, lx: float, lb: float, scroll_y: float) -> bool:
        """滚轮缩放（以光标为锚点）。返回是否处理。"""
        s = self.state
        if not self.active or scroll_y == 0:
            return False
        lt = self.vh - lb
        old_zoom = s["zoom"]
        factor = SCROLL_ZOOM_FACTOR if scroll_y > 0 else 1 / SCROLL_ZOOM_FACTOR
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return False
        img_x = (lx - s["pan_x"]) / old_zoom
        img_y = (lt - s["pan_y"]) / old_zoom
        s["pan_x"] = lx - img_x * new_zoom
        s["pan_y"] = lt - img_y * new_zoom
        s["zoom"] = new_zoom
        self._update_sprites()
        return True

    def on_press(self, lx: float, lb: float) -> bool:
        """左键按下：命中分割线 → drag_split，否则 drag_view。返回是否处理。"""
        s = self.state
        if not self.active:
            return False
        s["mouse_x"], s["mouse_y"] = lx, lb
        zoom, draw_w, draw_h, px, py, split_x, _ = self._geometry()
        lt = self.vh - lb
        in_img = (px <= lx <= px + draw_w) and (py <= lt <= py + draw_h)
        if not in_img:
            return False
        if s["shape_mode"] == "split" and self.has_pair and abs(lx - split_x) < SPLIT_HIT_RADIUS:
            s["mode"] = "drag_split"
            self.on_hover("split")
        else:
            s["mode"] = "drag_view"
            self.on_hover("hand")
        return True

    def on_drag(self, lx: float, lb: float, dx: float, dy: float) -> bool:
        """拖动。dx/dy 为窗口事件原生增量（dx 向右、dy 向上为正）。"""
        s = self.state
        s["mouse_x"], s["mouse_y"] = lx, lb
        if not self.active or s["mode"] is None:
            return False
        if s["mode"] == "drag_view":
            s["pan_x"] += dx
            s["pan_y"] -= dy   # pan_y 以顶部为 0，鼠标上移（dy>0）时图应上移
            self._update_sprites()
            return True
        if s["mode"] == "drag_split":
            _, draw_w, *_ = self._geometry()
            if draw_w <= 0:
                return False
            split = max(0.0, min(1.0, (lx - s["pan_x"]) / draw_w))
            if abs(split - s["split"]) > 1e-9:
                s["split"] = split
                self._update_sprites()
            return True
        return False

    def on_release(self) -> None:
        if self.state["mode"] is not None:
            self.state["mode"] = None
        if self.state["hover"] != "none":
            self.state["hover"] = "none"
            self.on_hover("none")

    def on_motion(self, lx: float, lb: float) -> bool:
        """未拖动时更新悬停光标。返回本事件是否属于查看器区域。"""
        s = self.state
        if not self.active:
            return False
        s["mouse_x"], s["mouse_y"] = lx, lb
        if s["mode"] is not None:
            return True
        zoom, draw_w, draw_h, px, py, split_x, _ = self._geometry()
        lt = self.vh - lb
        in_img = (px <= lx <= px + draw_w) and (py <= lt <= py + draw_h)
        # 电平触发：每个 motion 事件都无条件上报悬停态。回调最终落到
        # _set_cursor（按名字去重，重复上报零成本）。不能按「状态变化才
        # 回调」的边沿触发 —— App 层的光标复位会与之竞争，表现为分割线
        # 的 ↔ 只在按住时出现、悬停不显示。
        if s["shape_mode"] == "split" and self.has_pair and in_img and abs(lx - split_x) < SPLIT_HIT_RADIUS:
            s["hover"] = "split"
            self.on_hover("split")
        elif in_img:
            s["hover"] = "hand"
            self.on_hover("hand")
        else:
            s["hover"] = "none"
            self.on_hover("none")
        return True

    def on_deactivate(self) -> None:
        """窗口失焦时取消拖动（与 ImageDiff.py 行为一致）。"""
        if self.state["mode"] is not None:
            self.state["mode"] = None
            self.state["hover"] = "none"
            self.on_hover("none")

    # ---------------- 键盘 ----------------

    def on_home(self) -> None:
        self.reset_view()

    def on_space(self) -> None:
        """切换 分割线 / 圆形对比。"""
        if not self.active or not self.has_pair:
            return
        self.state["shape_mode"] = "circle" if self.state["shape_mode"] == "split" else "split"
        self._update_sprites()

    def on_circle_zoom(self, direction: float) -> None:
        """+ / - 调整圆形对比半径。"""
        if not self.active or not self.has_pair:
            return
        factor = 1.1 if direction > 0 else 1 / 1.1
        self.state["circle_scale"] = max(
            CIRCLE_SCALE_MIN, min(CIRCLE_SCALE_MAX, self.state["circle_scale"] * factor)
        )

    def image_to_window(self, x_img: float, y_img: float) -> tuple[float, float]:
        """图像坐标（y 从顶部算起）→ 窗口坐标（y 从底部算起）。供宿主绘制覆盖层。"""
        s = self.state
        return (
            self.vx + s["pan_x"] + x_img * s["zoom"],
            self.vy + self.vh - s["pan_y"] - y_img * s["zoom"],
        )

    # ---------------- 绘制 ----------------

    def draw(self) -> None:
        """绘制当前帧。宿主需自行开启/关闭 Scissor 裁剪。"""
        s = self.state
        if not self.active:
            return
        self._ensure_program()
        if s["shape_mode"] == "circle" and self.has_pair:
            zoom, _, _, px, _, _, draw_y = self._geometry()
            lx = (s["mouse_x"] - px) / zoom
            ly = (s["mouse_y"] - draw_y) / zoom
            lx = max(0.0, min(float(self.img_w), lx))
            ly = max(0.0, min(float(self.img_h), ly))
            r = min(self.vw, self.vh) * s["circle_scale"] / zoom
            self.circle_program.use()
            self.circle_program["circle_center"] = (lx, ly)
            self.circle_program["circle_radius"] = r
            self.circle_program["circle_line_width"] = 1.0 / zoom
            self.circle_program["use_circle"] = 1
        else:
            self.circle_program.use()
            self.circle_program["use_circle"] = 0
        self.batch.draw()
