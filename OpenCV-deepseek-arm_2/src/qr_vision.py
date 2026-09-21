#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_vision.py —— 二维码视觉基元（共用模块）

被 tools/test_camera_qr.py（机位自检）和 step3_hand_eye_calib.py（手眼标定）共用，
避免检测逻辑在两处各写一份、然后慢慢跑偏。

这里沉淀的是实测踩坑得来的结论，改动前请先读注释：
  · detect()      兜住 OpenCV 4.6.0 QRCodeDetector 的 kmeans 断言崩溃
  · wait_for_focus() 对焦要等 3~7 秒；判据用「能否解出码」而非 blur 阈值
  · 角点含义      检测器返回的是「数据区」角点(不含白边)，
                  顺序 [左上,右上,右下,左下]，corner[0] 恒为该码自己的
                  数据区左上角，与图像旋转无关(实测 0.00px 偏差)。
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────── 实测标定常数 ───────────────────────────
# ★ 以**数据区**（黑色图案，21 模块）为准 —— 卷尺只能量黑色图案，白边是纯白、
#   量不出边界。含白边的 29 模块由它反推。
# ★ 换纸 / 重印 / 重新量过之后，这几个数必须一起更新。曾经留着上一版的值
#   (39.285/28.45)，让 px_per_mm() 把画面覆盖宽度报大了 9% —— 它下游正是
#   「摄像头架够近了没」那个判断，报大了会让人以为纸装得下。
# ★ 这几个数和 data/ 那份 json 里的码尺寸是**两次独立测量**，差零点几毫米是正常的
#   （本纸: 这里量到 26.5，json 里是 26.0）。坐标映射不受影响 —— 四个锚点整体挪
#   0.35mm 只等效成一次 0.31mm 的平移；只有 px_per_mm() 这种要说"画面有多少毫米"
#   的地方才在乎绝对值，所以那里必须用真实的物理尺寸。
# ★ 别硬套一个"应该"的值: 数据区实测边长就是下面这个数，量到多少写多少。
MODULES_TOTAL = 29
DATA_MODULES = MODULES_TOTAL - 2 * 4     # 21 = 数据区(检测器返回的就是它)
DATA_SIDE_MM = 26.5                      # 数据区实测边长（卷尺量黑色图案）
MODULE_MM = DATA_SIDE_MM / DATA_MODULES  # 1.2619 mm 每模块(白边区与数据区模块同宽)
QR_SIDE_MM = MODULES_TOTAL * MODULE_MM   # 36.595 mm 含白边(29 模块)

# px/模块: 实测 4.07~4.21 → 150帧连续 4/4 全解；<3.5 开始丢码
GOOD_PX_PER_MODULE = 4.0
OK_PX_PER_MODULE = 3.0

FOCUS_TIMEOUT = 8.0          # 等自动对焦的最长秒数

# ★ 手动焦距锁定值（UVC V4L2_CID_FOCUS_ABSOLUTE 口径，量程 0~1023）。
#   值小 = 对**远**（纸面），值大 = 对**近**（吸盘码）。来历和实测表见 lock_focus。
#   ★ 只跟「相机架设 + 机械臂工作高度」有关，换机位要重量 ——
#     量法: python3 tools/measure_focus.py 210 340 10 12（它会把建议值打出来）。
FOCUS_LOCK = 250

# 「纸→机械臂」的等比缩放和 1.000 差多少才值得提一句（见 similarity_fit / robot_scale_note）。
# ★ 只是**提一句**的下限，不是合格线: 实测机械臂本来就常不成 1:1（一台 Magician 是
#   0.833），依此报警会把正常情况说成故障。真正的判据是归一化后的残差 RMS。
SCALE_TOL = 0.02

# 参考点取法 —— 对刀时吸盘要瞄的那个点。
#
# ★ 名字里的"角"一律指**数据区的角**（不含白边）。检测器返回的 quad 顺序恒为
#   [左上, 右上, 右下, 左下]，与图像怎么旋转无关（实测 0.00px 偏差），
#   所以 CORNER_ORDER 与 quad 的下标是一一对应的。
# ★ 别把这里的角和白边外框的角搞混: 两者相差 **4 个模块**（不是固定的毫米数 ——
#   模块多大取决于码印成多大: 本纸 26.5/21 是 5.05mm，曾用过的 39.285/29 那张是 5.42mm）。
#   而四点拟合残差恒为 0，这种整体偏移**不会以任何形式报出来**。
#   json 里两套坐标都有，见下面 load_paper_layout()：我们只取 symbol_corners_mm（数据区）。
CORNER_ORDER = ("corner_tl", "corner_tr", "corner_br", "corner_bl")   # ↔ quad[0..3]
REFERENCE_MODES = CORNER_ORDER + ("center",)

REFERENCE_LABEL = {
    "corner_tl": "数据区左上角",
    "corner_tr": "数据区右上角",
    "corner_br": "数据区右下角",
    "corner_bl": "数据区左下角",
    "center":    "数据区中心",
}

# 默认用右上角。
# ★ 四个角和中心在数学上完全等价，换的只是"吸盘瞄哪儿"，不存在哪个更准；
#   所以对刀和标定必须用同一个 —— 换的时候两边一起换（见下面的守卫）。
DEFAULT_REFERENCE = "corner_tr"


def contrast_mode(mode: str) -> str:
    """
    挑一套和 mode **不同**的参考点，专供自检做独立检验。

    ★ 四点拟合的残差恒为 0，拿同一套点去检验等于自证，必然通过（假绿灯）。
      必须换成另一套点，才有了识别力。
    ★ 角点 ↔ 中心的力臂约 13.3mm（一个码 26.5mm 的一半），足够把"角点取错
      一个模块"这类错误放大出来 —— 实测能把偏差放大到 7.63mm。
    """
    return "center" if mode != "center" else "corner_tl"


# ─────────────────────────── 纸布局 ───────────────────────────
@dataclass
class PaperLayout:
    """标定纸上四个码的已知几何（来自 step1_gen_paper.py 输出的 json）。"""
    codes: list[str]
    modules: int
    qr_side_mm: float
    paper_mm: tuple[float, float]
    reference: str                                       # 见 REFERENCE_MODES
    # {模式名: {码: (x_mm, y_mm)}} —— 五种参考点全存着，由 reference 挑一套用。
    #   全存的好处: 自检能拿"没参与拟合"的另一套点做独立检验（见 contrast_mode）。
    points_mm: dict = field(default_factory=dict)

    @property
    def reference_mm(self) -> dict:
        """按 reference 选中的那一套纸面坐标(mm)。"""
        return self.mm_for(self.reference)

    def mm_for(self, mode: str) -> dict:
        """取任意一套参考点的纸面坐标(mm)。"""
        if mode not in self.points_mm:
            raise ValueError(f"标定纸 json 里没有 {mode!r} 这套参考点"
                             f"（可用: {sorted(self.points_mm)}）")
        return self.points_mm[mode]


def load_paper_layout(json_path: str | Path,
                      reference: str = DEFAULT_REFERENCE) -> PaperLayout:
    """
    读 step1_gen_paper.py 产出的 calib_A4_qr.json，拿到四个码的已知物理坐标。

    reference 决定「参考点」用哪个，取值见 REFERENCE_MODES（默认右上角）。
    ★ 你用什么点去对刀，标定就用什么点 —— 两边必须一致，否则会有固定偏移。
    """
    if reference not in REFERENCE_MODES:
        raise ValueError(f"reference 只能是 {REFERENCE_MODES} 之一，收到 {reference!r}")
    d = json.loads(Path(json_path).read_text(encoding="utf-8"))
    qr = d["qr"]
    codes = list(qr.keys())
    first = qr[codes[0]]
    if "symbol_corners_mm" not in first:
        raise ValueError(f"{json_path} 里没有 symbol_corners_mm —— "
                         f"这是旧版 step1_gen_paper.py 生成的，请重新运行 step1_gen_paper.py")
    lay = PaperLayout(
        codes=codes,
        modules=int(first["modules"]),
        qr_side_mm=float(d["meta"]["qr_size_mm_actual"]),
        paper_mm=tuple(d["meta"]["paper_mm"]),
        reference=reference,
        points_mm={m: {} for m in REFERENCE_MODES},
    )
    # json 里的角点用 top_left/top_right/... 命名，这里映射回和 quad 同一套顺序的名字
    json_key = {"corner_tl": "top_left", "corner_tr": "top_right",
                "corner_br": "bottom_right", "corner_bl": "bottom_left"}
    for c in codes:
        rec = qr[c]
        # ★ 只取 symbol_corners_mm（数据区，不含白边）。取 outer_rect_mm 就会
        #   整体偏 4 个模块（本纸 5.05mm），而且残差查不出来。
        for mode in CORNER_ORDER:
            lay.points_mm[mode][c] = tuple(rec["symbol_corners_mm"][json_key[mode]])
        lay.points_mm["center"][c] = tuple(rec["center_mm"])
    return lay


def load_suction_spec() -> dict | None:
    """
    data/suction_qr.json 的全文（tools/gen_suction_qr.py 生成）。读不到返回 None。

    ★ 为什么要有这个函数而不是各处自己 read_text: 这份 json 的**字段名**
      （content / variants / data_mm_actual …）只该有一个地方知道。
    """
    from paths import SUCTION_QR_JSON
    try:
        d = json.loads(SUCTION_QR_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return d if isinstance(d, dict) else None


def load_suction_code() -> str | None:
    """
    吸盘上第 5 个码的内容。读不到（没生成过 / json 坏了）返回 None ——
    调用方据此认为「还没贴」。

    ★ 为什么从 json 读而不是写死常量: 内容和尺寸都在 json 里，写死两处必然跑偏。
    ★ 为什么放在 qr_vision: 它是**吸盘码这件事**的唯一真相来源，
      tools/test_camera_qr.py（机位自检）和 src/color_vision.py（生产识别）
      都要用同一个 —— 抄两份的第一个后果就是改了一处忘了另一处。
    """
    spec = load_suction_spec()
    try:
        return spec["content"]
    except (KeyError, TypeError):
        return None


def suction_size_candidates() -> list[tuple[str, float]]:
    """
    吸盘码**印出来可能是哪几张**：[(标签, 数据区mm)]，由大到小。

    ★ 为什么需要: gen_suction_qr.py 一次出 4 个尺寸（15/20/25/30mm 整块），
      用户挑一张贴上去，但**贴的是哪张没记在 json 里**。
      tools/measure_suction_map.py 靠「画面里量出的每模块像素数 ÷ 桌面比例」
      反推出这张码的真实数据区 mm，再用本函数给出的候选去认领是哪一张 ——
      认不出来（差得远）就说明那次测量本身有问题，比默默算下去强。
    """
    spec = load_suction_spec()
    out = []
    for tag, v in (spec or {}).get("variants", {}).items():
        try:
            out.append((tag, float(v["data_mm_actual"])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out, key=lambda kv: -kv[1])


def reference_point(quad: np.ndarray, mode: str = DEFAULT_REFERENCE) -> np.ndarray:
    """
    从检测器返回的四边形里取出「参考点」像素坐标。
    quad 顺序为 [左上,右上,右下,左下]（数据区，不含白边）。
    mode 取值见 REFERENCE_MODES；下标由 CORNER_ORDER 决定，别自己硬写数字。
    """
    if mode not in REFERENCE_MODES:
        raise ValueError(f"mode 只能是 {REFERENCE_MODES} 之一，收到 {mode!r}")
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    if mode == "center":
        return q.mean(axis=0)
    return q[CORNER_ORDER.index(mode)].copy()


# ─────────────────────────── 检测 ───────────────────────────
def detect(frame: np.ndarray, detector) -> tuple[dict, list]:
    """
    跑一次多码检测。返回 (decoded, located)：
      decoded: {内容: 四边形}  —— 解出内容
      located: [四边形, ...]   —— 只定位到、没解出内容
    """
    try:
        _ok, infos, pts, _ = detector.detectAndDecodeMulti(frame)
    except cv2.error:
        # OpenCV 4.6.0 的 QRCodeDetector 在退化输入(极小图/无码)上会内部
        # kmeans 断言崩溃，这是上游 bug。当作“没检测到”，不能让程序挂掉。
        return {}, []
    decoded, located = {}, []
    if pts is None or len(pts) == 0:
        return decoded, located
    infos = infos if infos is not None else ()
    for i, quad in enumerate(pts):
        text = infos[i] if i < len(infos) else ""
        q = np.asarray(quad, dtype=np.float32)
        if text:
            decoded[text] = q
        else:
            located.append(q)
    return decoded, located


# ── 滑窗补扫（detect_sweep 用）──
# ★ 两个窗口尺寸都要，**不能只留一个**（实测，见 detect_sweep 的说明）:
#   窗口太小 → 桌面大码（P2 一类）被切坏，实测 300px 窗让 P2 掉到 3/20；
#   窗口太大 → 吸盘码反而漏，实测 3x3 整幅切块 0/30。
SWEEP_WINDOWS = (400, 520)
SWEEP_STRIDE_RATIO = 0.75          # 步长 = 窗口 x 0.75（相邻窗重叠 25%）


def _window_offsets(shape: tuple, win: int, ratio: float):
    """滑窗的左上角坐标序列（含贴右边/下边的最后一窗）。"""
    h, w = shape[:2]
    win = min(win, w, h)
    if win <= 0:
        return
    step = max(1, int(win * ratio))
    xs = list(range(0, max(1, w - win + 1), step))
    ys = list(range(0, max(1, h - win + 1), step))
    if xs[-1] != w - win:
        xs.append(max(0, w - win))
    if ys[-1] != h - win:
        ys.append(max(0, h - win))
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, x0 + win, y0 + win


def detect_sweep(frame: np.ndarray, detector, want=None,
                 windows=SWEEP_WINDOWS,
                 stride_ratio: float = SWEEP_STRIDE_RATIO) -> tuple[dict, list]:
    """
    先整帧检一遍；还缺码就再开滑窗补扫，把结果并回整帧坐标系。
    返回 (decoded, located)，与 detect() 同构、可直接替换。

    want: 期望解出的内容（可选）。给了就能「够了就停」——
          整帧已经全解出时**一次滑窗都不跑**，耗时与 detect() 相同。

    ★★ 为什么必须有这一步（实测，别删）:
      OpenCV 4.6.0 的 QRCodeDetector 在**整帧**上会漏掉吸盘上那个码，
      而且行为很「跳」——同一张图，裁剪窗口只差一点点，检出结果就在
      「解得出 / 解不出」之间翻来覆去，不是平滑变化:

        以吸盘码为中心的裁剪，半边长 150px→解出, 175→漏, 200→解出,
        225→漏, 250→漏, 275→解出, 300~380→漏, 整帧→漏

      而把它单独裁成 300x300 连测 8 次，**8/8 稳定解出**（和线程数无关）。
      → 码本身没问题，是整帧检测定位不到它: 画面中央被机械臂那一大团深色
        轮廓占着，码的定位图案被淹在里面。给它一个范围合适的局部视野就能解开。

    ★ 为什么窗口尺寸要有两档:
      · 窗口**太小**会把桌面上的大码切坏。实测 300px 窗 66 个窗位，
        吸盘码 20/20，但 P2 掉到 3/20。
      · 窗口**太大**（等于整幅切 3x3 块）吸盘码又回到 0/30。
      实测 400 与 520 两档配合「整帧」一起取并集，5 个码才能都稳住。

    ★ 整帧那条路先跑、结果优先保留: 整帧给出的角点最可靠（滑窗在边缘处
      可能把定位图案截断）。滑窗只负责“补漏”，绝不覆盖已有结果。

    ★ 代价: 实测 1080p 约 120~200ms（跑满两档窗口时）。所以:
      · 交互预览别每帧都跑 —— 见 test_camera_qr.py 里的隔帧用法。
      · 传了 want 且整帧就全解出时，它退化成一次 detect()，几乎不要钱。
    """
    decoded, located = detect(frame, detector)

    def satisfied() -> bool:
        return want is not None and all(c in decoded for c in want)

    if satisfied():
        return decoded, located

    for win in windows:
        for x0, y0, x1, y1 in _window_offsets(frame.shape, win, stride_ratio):
            sub_dec, sub_loc = detect(frame[y0:y1, x0:x1], detector)
            for text, quad in list(sub_dec.items()) + [(None, q) for q in sub_loc]:
                q = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
                q[:, 0] += x0
                q[:, 1] += y0
                if text is None:
                    located.append(q)
                elif text not in decoded:
                    decoded[text] = q
            if satisfied():
                break
        if satisfied():
            break

    # 滑窗重叠区会把同一个码重复定位，按中心去重（也去掉已解码码的残影）
    anchors = [quad_center(q) for q in decoded.values()]
    kept = []
    for q in located:
        c = quad_center(q)
        if any(np.hypot(*(c - a)) < 30.0 for a in anchors):
            continue
        if any(np.hypot(*(c - quad_center(k))) < 30.0 for k in kept):
            continue
        kept.append(q)
    return decoded, kept


def quad_side_px(quad: np.ndarray) -> float:
    """四边形四边平均像素长度。"""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    sides = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    return float(np.mean(sides))


def px_per_module(quad: np.ndarray) -> float:
    """
    估算「每模块几像素」，全项目统一用这一个定义，别再各写一份。

    ★ 口径说明（重要，别想当然改）:
      检测器给的是**数据区**角点(21 模块)，而这里除以 MODULES_TOTAL(29)。
      所以返回值 = 数据区边长px / 29，比「真正的」每模块像素数小 21/29≈0.72 倍。
      这是历史口径，GOOD/OK_PX_PER_MODULE 两个阈值就是在这个口径下实测出来的
      （实测 4.07~4.21 时 150 帧 4/4 全解，<3.5 开始丢码）。
      换算成真值要乘 29/21：即约 5.5 px/模块 才是解码下限。
      不要单独改这个函数的口径 —— 一改，所有实测阈值全部失效。
    """
    return quad_side_px(quad) / MODULES_TOTAL


def px_per_mm(quad: np.ndarray) -> float:
    """
    真实的「像素/毫米」: 数据区边长px ÷ 数据区实测mm(DATA_SIDE_MM，本纸 26.5)。

    ★ 这是给「画面能覆盖多少毫米」这类诊断用的，跟 px_per_module 的历史口径
      是两套东西，别混着用。混用会差 29/21≈1.38 倍 —— 曾经就把覆盖范围
      多算了 38%，让人误以为纸装得下。
      （前提: 标定纸是按 100% 实际大小打印的。）
    """
    return quad_side_px(quad) / DATA_SIDE_MM


def quad_center(quad: np.ndarray) -> np.ndarray:
    return np.asarray(quad, dtype=np.float64).reshape(4, 2).mean(axis=0)


def touches_edge(quad: np.ndarray, shape: tuple, margin: int = 8) -> bool:
    """角点贴到画面边缘 → 码被裁切，永远解不出，必须重新构图。"""
    h, w = shape[0], shape[1]
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return bool((q[:, 0] <= margin).any() or (q[:, 0] >= w - 1 - margin).any()
                or (q[:, 1] <= margin).any() or (q[:, 1] >= h - 1 - margin).any())


def blur_score(gray: np.ndarray) -> float:
    """拉普拉斯方差：越大越清晰（1080p 清晰时 ~300，糊掉时 ~2）。"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def paper_fit(decoded: dict, layout: PaperLayout, shape: tuple,
              margin: int = 2) -> tuple[np.ndarray, list[str], float] | None:
    """
    整张纸到底有没有完整落在画面里？返回 (纸四角像素, 出界的边, 出界多少mm)，判断不了则 None。

    做法: 四个码的中心像素坐标 ↔ 它们已知的纸面坐标(mm) 求单应，
          再把纸的四个角(0,0)(W,0)(W,H)(0,H) 投回图像上看有没有出画。
    这比「用 px/mm 估覆盖范围」准 —— 后者默认纸正好居中，而实际常常是偏的；
    这里是投影计算，只要纸是平的、相机是透视的，就成立。

    ★ 为什么值得单独查: 四个码离纸边还有 15mm，所以「纸被切掉一条边」时
      码照样能解出，看着一切正常 —— 但机位已经到极限了，稍微再挪一下就
      开始丢码。实测 P3/P4 就是这么挂的。宁可现在就说清楚。
    """
    if not all(c in decoded for c in layout.codes):
        return None
    # 这里用「中心」而不是对刀参考点: 纸的四个角是从码的中心外推的，
    # 中心对四个角的影响最均衡。换参考点不影响本函数的结论。
    mm = np.float32([layout.mm_for("center")[c] for c in layout.codes])
    px = np.float32([quad_center(decoded[c]) for c in layout.codes])
    H = cv2.getPerspectiveTransform(mm, px)          # 纸面mm → 像素
    pw, ph = layout.paper_mm
    corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(corners, H).reshape(4, 2)

    h, w = shape[:2]
    outside = {"左": float(np.maximum(margin - pts[:, 0], 0).max()),
               "右": float(np.maximum(pts[:, 0] - (w - 1 - margin), 0).max()),
               "上": float(np.maximum(margin - pts[:, 1], 0).max()),
               "下": float(np.maximum(pts[:, 1] - (h - 1 - margin), 0).max())}
    cut = [k for k, v in outside.items() if v > 0]
    if not cut:
        return pts, [], 0.0
    # 把出界像素换算成 mm（用码的实测 px/mm）
    ppm = min(px_per_mm(decoded[c]) for c in layout.codes)
    return pts, cut, max(outside[k] for k in cut) / ppm


# ─────────────────────── 纸↔机械臂 的几何关系 ───────────────────────
def robot_xy_of(v) -> tuple[float, float]:
    """机械臂坐标有 [x, y] 和 {"x":..., "y":...} 两种写法，这里统一取出来。"""
    if isinstance(v, dict):
        return float(v["x"]), float(v["y"])
    return float(v[0]), float(v[1])


def similarity_fit(paper_mm: dict, robot_xy: dict,
                   codes) -> tuple[float, float, float, int] | None:
    """
    把「纸上坐标 → 机械臂 XY」按**相似变换**（等比缩放 + 旋转 + 平移，不含镜像）
    最小二乘拟合，返回 (缩放, 旋转角(度), 残差RMS(mm), 用到的点数)；点不足返回 None。

    ★★ 这个量是为了把三类毛病干净地分开:
        · 缩放正常，RMS 大      → **某一个点**教歪了（个别点的问题）
        · 缩放整体偏 1（0.83）  → 机械臂报的 XY 与真实毫米**不成 1:1**
        · 缩放正常、RMS 也正常  → 这套坐标是自洽的

      ★ 判据是**归一化之后**的残差 RMS，不是缩放本身。原因见下面这条 ——
        实测机械臂的缩放本来就不是 1，拿缩放当判据会把正常情况误报成故障。

    ★★ 缩放 ≠ 1 是**实测常态，不是故障**（详见 robot_scale_note）:
      机械臂的 GetPose 是按它**自己的运动学模型**算的。模型杆长与真机不符时，
      实际走的距离就和报告的不一样（实测一台 Magician: 步长 1 报 10mm、实走
      约 12mm，缩放 5/6 = 0.833）。手眼标定**不受影响** —— 对刀读报告值、
      执行也下报告值，同一个映射两边都用，比例自己抵消掉了。
      但纸面尺寸、按 mm 算的 Z 下降量都不再是真实距离，别直接当毫米用。

    这个量比「逐对距离」多抓到的东西: **小而系统**的形变。整体缩水 1% 时，
    250mm 的对角线才差 2.5mm，落在 3mm 容差里就溜过去了；而 6 条一起做最小
    二乘拟合，1% 会稳稳浮出来。

    只有 ≥2 个点才成立；2 个点是精确解（残差恒为 0），所以 RMS 只对 ≥3 个点有意义。
    """
    cs = [c for c in codes if c in paper_mm and c in robot_xy]
    if len(cs) < 2:
        return None
    x = np.array([paper_mm[c] for c in cs], float)
    y = np.array([robot_xy_of(robot_xy[c]) for c in cs], float)
    xc, yc = x - x.mean(0), y - y.mean(0)
    sxx = float((xc ** 2).sum())
    if sxx <= 0:
        return None
    # 复数表示下 c = Σ conj(x)·y / Σ|x|² = 缩放·e^{iθ}，实部虚部展开就是下面两行
    a = float((xc * yc).sum())                                    # Σ(x·X + y·Y)
    b = float((xc[:, 0] * yc[:, 1] - xc[:, 1] * yc[:, 0]).sum())   # Σ(x·Y − y·X)
    scale = float(np.hypot(a, b)) / sxx
    if scale <= 0:
        return None
    rot = float(np.degrees(np.arctan2(b, a)))
    ct, st = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    R = np.array([[ct, -st], [st, ct]])
    pred = scale * (xc @ R.T) + y.mean(0)
    rms = float(np.sqrt(((y - pred) ** 2).sum(1).mean()))
    return scale, rot, rms, len(cs)


def robot_scale_note(scale: float, n: int, short: bool = False,
                     axes: tuple[float, float] | None = None) -> str:
    """
    「机械臂报的 XY 不是毫米」的说明；正常时返回空串。

    ★ 措辞很关键: 这里**不能**说成故障、更不能叫人去回零重来。
      它是机械臂运动学模型与真机的固有差异，对刀/标定/执行全程一致使用
      报告值，比例自己抵消。实测踩过的坑: 第一版把它当成「丢步，先回零」，
      让人以为继续对刀毫无意义 —— 方向完全错了。

    ★ axes=(kx, ky): 纸上 X / Y 两个方向各自的比例（由 affine_fit 给的线性部分
      算列范数）。★★ 两个轴的比例可以**不一样**（实测一台 Magician: X 0.81、
      Y 0.93）—— 这时"一个缩放"根本描述不了这个映射，必须先说清这一点，
      否则用户会带着"等比缩放 0.85"的错误印象去推 Z 下降量之类的数。

    ★ short=True 给一行版，用在"每记一个点就打印"的地方 —— 那儿一屏里要塞下
      别的东西，而且下面还有 l 能看全文。

    ★ 调用方注意: 这是**说明**，不是错误。别把它塞进「⚠ 还有问题」那种列表里，
      否则又把用户吓回去（实测就是这么错的）。它只说明"报的数不是毫米"。
    """
    if axes is not None:
        kx, ky = float(axes[0]), float(axes[1])
        if min(kx, ky) > 0 and abs(kx - ky) > SCALE_TOL:
            if short:
                return (f"★ 机械臂报的 XY 不是毫米，而且 X/Y 两个轴比例还不一样"
                        f"（{kx:.3f} / {ky:.3f}；正常，见按 l 的说明）")
            return (f"机械臂报的 XY 与真实毫米不成 1:1，**两个轴的比例还不一样**:\n"
                    f"  纸上 X 方向 ×{kx:.4f}、Y 方向 ×{ky:.4f}"
                    f"（走 10mm 分别只报 {kx * 10:.1f} / {ky * 10:.1f} mm）。\n"
                    f"  ★ 同样不影响手眼标定 —— 对刀读报告值、执行也下报告值，\n"
                    f"    两边用同一个映射，比例自己抵消。一致性检查已改成按各轴分别\n"
                    f"    缩放（仿射）后再比，所以逐对距离看着都对得上。\n"
                    f"  ★ 但**纸面尺寸、按 mm 算的 Z 下降量都不是真实距离**，别当毫米用。\n"
                    f"  ★ 仍要确认它可重复（否则是丢步/打滑）: X、Y 各朝同一方向连走\n"
                    f"    三次 100mm，钢尺量实际走了多少。三段一样长 → 模型差异，放心继续；\n"
                    f"    忽长忽短 → 停下，回零并查皮带/联轴器。")
    if abs(scale - 1.0) <= SCALE_TOL:
        return ""
    if short:
        return (f"★ 机械臂报的 XY 缩放 {scale:.4f}（正常，见按 l 的说明；"
                f"它报的数不是真实毫米）")
    return (f"机械臂报的 XY 与真实毫米不成 1:1: 缩放 {scale:.4f}"
            f"（它走 10mm 只报 {scale * 10:.1f}mm；真实距离 ≈ 报告值 ÷ {scale:.3f}）。\n"
            f"  ★ 这不影响手眼标定 —— 对刀读报告值、执行也下报告值，两边用同一个映射，\n"
            f"    比例自己抵消。一致性检查已按这个缩放归一化后再比"
            f"（{n} 个点估的，越多越准）。\n"
            f"  ★ 但纸面尺寸、按 mm 算的 Z 下降量都不再是真实距离，别直接当毫米用。\n"
            f"  ★ 唯一要确认的是它**可重复**（否则就不是模型问题、而是丢步/打滑）:\n"
            f"    把 X 朝同一方向连走三次 100mm（步长 1 按十下），每次用钢尺量实际走了多少。\n"
            f"    三段一样长 → 模型差异，放心继续；忽长忽短 → 停下，回零并查皮带/联轴器。")


def affine_fit(paper_mm: dict, robot_xy: dict, codes) -> tuple[np.ndarray, float] | None:
    """
    最小二乘仿射拟合 纸上mm → 机械臂XY，返回 (2x3 矩阵, 各点最大残差mm)。

    比 similarity_fit 多两个自由度（允许 x/y 方向**分别**缩放、允许剪切）。
    ★ 这个区分很要紧: 机械臂的 XY 不成 1:1 时，**两个轴的比例可以不一样**
      （实测这台 X 方向约 0.81、Y 方向约 0.93）。用相似拟合去看就会把这种
      各向异性算成"某个点教歪了"，指错方向。判断"某个点到底有没有教歪"要用
      仿射残差，不能用相似残差。
    """
    cs = [c for c in codes if c in paper_mm and c in robot_xy]
    if len(cs) < 3:
        return None
    x = np.array([paper_mm[c] for c in cs], float)
    y = np.array([robot_xy_of(robot_xy[c]) for c in cs], float)
    A = np.hstack([x, np.ones((len(cs), 1))])
    if np.linalg.matrix_rank(A) < 3:          # 三点共线 → 仿射定不出来
        return None
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    worst = float(max(np.linalg.norm(r) for r in (y - A @ coef)))
    return coef.T, worst


def _normalize_pts(pts: np.ndarray) -> np.ndarray:
    """Hartley 归一化: 把点平移到质心、缩放到平均距离 sqrt(2)。只为数值稳定，不改结果。"""
    c = pts.mean(axis=0)
    d = float(np.mean(np.linalg.norm(pts - c, axis=1)))
    s = (2.0 ** 0.5) / d if d > 0 else 1.0
    return np.array([[s, 0.0, -s * float(c[0])],
                     [0.0, s, -s * float(c[1])],
                     [0.0, 0.0, 1.0]])


def homography_fit(paper_mm: dict, robot_xy: dict, codes) -> np.ndarray | None:
    """
    四点透视拟合 纸上mm → 机械臂XY，返回 3x3 单应矩阵（p_h = H @ [x,y,1]，再除以第三个分量）。

    ★★ 用之前必须记住它和 affine_fit 的**本质区别** —— 这一步最容易误用:

      · 仿射: 6 个自由度，四个点给 8 个方程 → **有 2 个冗余**，所以有残差。
        残差能告诉你「四点不自洽」，是**判据**。
      · 单应: 8 个自由度，四个点给 8 个方程 → **恰好定死**，残差恒为 0。
        任何四个点（哪怕标签全错、哪怕量歪了）都能精确穿过 ——
        所以单应残差 **恒等于 0，不能当判据用、更不能当"标定准"的证据**。
      · 单应的用处只在**插值**: 四点围成的四边形是梯形时，仿射只能取折中
        （本项目的实测四点就是，残差 6.7mm），单应能精确穿过四个角。
        这也正是 calib 方案里 getPerspectiveTransform() 干的事。

    点多于 4 个时按最小二乘解（SVD），此时残差才有意义。
    """
    cs = [c for c in codes if c in paper_mm and c in robot_xy]
    if len(cs) < 4:
        return None
    P = np.array([paper_mm[c] for c in cs], float)
    Q = np.array([robot_xy_of(robot_xy[c]) for c in cs], float)
    Tp, Tq = _normalize_pts(P), _normalize_pts(Q)

    def to_h(T, pts):
        ones = np.ones((len(pts), 1))
        h = np.hstack([pts, ones]) @ T.T
        return h[:, :2] / h[:, 2:3]

    p, q = to_h(Tp, P), to_h(Tq, Q)
    rows = []
    for (x, y), (u, v) in zip(p, q):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows, float))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tq) @ Hn @ Tp
    if abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def homography_apply(H, paper_xy) -> tuple[float, float]:
    """单应映射一个纸面点 → 机械臂 XY。"""
    p = np.asarray(paper_xy, float)
    h = np.asarray(H, float) @ np.array([p[0], p[1], 1.0])
    if abs(h[2]) < 1e-12:
        raise ValueError("单应分母为 0（该点在无穷远线上）")
    return float(h[0] / h[2]), float(h[1] / h[2])


def affine_winding_bad(aff) -> bool:
    """
    仿射拟合的线性部分**行列式为正** → 四点绕向反了（标签顺序错），返回 True。

    ★★ 为什么"正确"的行列式是**负**的 —— 这一步最容易搞反，写清楚:
      · 纸上坐标用的是**图像习惯（y 向下）**: calib_A4_qr.json 里 P1「左上」的
        y=39.50、P3「左下」的 y=170.50 —— 第二个分量变大 = 纸面往下。
      · 纸是**正面朝上**摊在桌上的，机械臂的 XY 又是**右手系**（Z 朝上）。
        从上往下看这张桌子: 机械臂的 X→Y 是**逆时针**，而纸面的 u→v（右→下）
        是**顺时针**。旋转改不了顺逆，所以「纸 → 机械臂」必然带一个镜像，
        行列式**一定 < 0**（= −kx·ky）。
      · 实测印证: 用实测里自洽的三个点（P1/P2/P3）精确定出的映射，
        行列式 = −0.759。

    于是行列式为正**不可能是**这张纸配这台机器的正常结果，只能是**标签错了**。

    ★ 这一条专治一个**仿射残差查不出来**的错: 把矩形对角线上相邻的两个角
      标反（P2↔P3 或 P1↔P4），得到的四点正好是原矩形的**对角镜像**，而镜像
      是仿射变换（残差≈0，能顺利骗过仿射残差检查），却不是相似变换。
      看行列式符号一眼就分开: 正确 −0.759（负）/ 对角镜像 +0.759（正）。
    """
    return float(np.linalg.det(aff[0][:, :2])) > 0.0


def shape_diagnosis(layout: "PaperLayout", robot_xy: dict, codes) -> list[str]:
    """
    四点整体不自洽时，回答「到底是哪个点错了、错了多少」。正常时返回空表。

    做法用的是**仿射不变量**: 四个点里，"第 4 点在前 3 点构成的三角形中的位置"
    是仿射不变的 —— 纸上是这样就该是这样。于是:
      · 用其余 3 点定出仿射映射（3 点必能精确定出），
      · 再看第 4 点差了多少，那个差**就等于这一点要挪多少才自洽**（纸面 mm）。

    ★ 谁是真凶？光看"要挪的量最小"会指错: 对边那个角往往也只需要挪差不多的量
      （实测 P3 和 P4 都是 26.00mm，一模一样，排序只能瞎猜）。真正的判据是
      **挪过去之后落在哪**: 如果正好落在**同一个二维码**的另一个已知参考点上
      （比如 P4 修正后正好落在 P4 的"数据区左上角"，差 2.5mm），那就不是巧合 ——
      是那个角瞄错了。所以先看"落点命中另一个参考点"，再看量的大小。

    ★ 只在整体不自洽时才说话 —— 自洽时这些数没有意义，说了只会让人乱改。
    """
    paper_mm = layout.reference_mm
    cs = [c for c in codes if c in paper_mm and c in robot_xy]
    if len(cs) != 4:
        return []
    x = {c: np.array(paper_mm[c], float) for c in cs}
    y = {c: np.array(robot_xy_of(robot_xy[c]), float) for c in cs}
    data_side = layout.qr_side_mm * DATA_MODULES / MODULES_TOTAL   # 数据区边长 mm
    hit_tol = data_side * 0.25

    cands = []
    for k in cs:
        rest = [c for c in cs if c != k]
        A = np.hstack([np.array([x[c] for c in rest]), np.ones((3, 1))])
        B = np.array([y[c] for c in rest])
        try:
            coef, *_ = np.linalg.lstsq(A, B, rcond=None)
        except np.linalg.LinAlgError:
            continue
        L = coef[:2].T                        # 2x2 线性部分
        if abs(np.linalg.det(L)) < 1e-9:
            continue
        # 第 4 点按这 3 点推出的仿射应该落在哪、实际差多少 → 反推该点要挪多少
        #   （除以 L → 得到的是**纸面** mm，好和纸上的角点直接比）
        d = np.linalg.solve(L, y[k] - (x[k] @ L.T + coef[2]))

        # ★ 挪过去之后落在哪个已知参考点上？（同码的另一个角 = 铁证）
        hit, err = None, float("inf")
        for mode, pts in layout.points_mm.items():
            if k not in pts:
                continue
            e = float(np.linalg.norm(x[k] + d - np.array(pts[k], float)))
            if e < err:
                hit, err = mode, e
        if err > hit_tol:
            hit = None
        cands.append((hit is None, float(np.linalg.norm(d)), k, d, hit, err))
    if not cands:
        return []
    # 命中另一个已知参考点的排前面；同样命中/同样没命中时，要挪得少的优先
    cands.sort(key=lambda t: (t[0], t[1]))

    out = ["四点形状**不是**纸上形状的等比/仿射放大缩小 —— 这不是缩放问题，是点错了。",
           "  只挪一个点就能让四点自洽，各点需要的修正量（纸面 mm）:",
           f"   · 数据区一边宽 {data_side:.2f}mm（修正量凑到这个数 = 瞄到了另一个角）"]
    for _no_hit, mag, k, d, hit, err in cands:
        tags = []
        if hit:
            tags.append(f"← 修正后正好落在 {k} 的「{REFERENCE_LABEL.get(hit, hit)}」"
                        f"（差 {err:.1f}mm）: 你瞄的是那个角吧")
        elif abs(abs(d[1]) - data_side) < hit_tol and abs(d[0]) < hit_tol:
            tags.append("← 正好一个数据区高度，多半是这个角瞄差了")
        elif abs(abs(d[0]) - data_side) < hit_tol and abs(d[1]) < hit_tol:
            tags.append("← 正好一个数据区宽度，多半是这个角瞄差了")
        out.append(f"   · {k} 挪 ({d[0]:+.1f}, {d[1]:+.1f}) mm  |{mag:5.1f}|"
                   + ("  " + " ".join(tags) if tags else ""))
    _no_hit, mag, k, d, hit, err = cands[0]
    out.append(f"  ★ 嫌疑最大: {k}（{mag:.1f}mm"
               + (f"，落点对上「{REFERENCE_LABEL.get(hit, hit)}」）" if hit
                  else "，要挪的向量最短）"))
    if hit:
        out.append(f"    → 回去看 {k}: 参考点是「{REFERENCE_LABEL[layout.reference]}」，"
                   f"你是不是量到了「{REFERENCE_LABEL.get(hit, hit)}」？")
    else:
        out.append(f"    先回去看 {k} 是不是瞄错了角: 参考点是"
                   f"「{REFERENCE_LABEL[layout.reference]}」，"
                   f"同码的另几个角只差一个数据区宽/高（{data_side:.0f}mm）。")
    return out


# ─────────────────────────── 摄像头 ───────────────────────────
# 支持任意标准 UVC 摄像头（内核 uvcvideo 接管，完全免驱）。实测过的两颗:
#   · 一声一视 YSYS X6L   eba4:1303   最高 3840x2160 MJPG
#   · DCX-5MAF-V1 (5MP)  0bda:5842   最高 2592x1944 MJPG @30fps
#
# ★ 为什么不写死 /dev/videoN: 插拔、换 USB 口、换摄像头都会让索引变。
#   实测同一台机器上先插 X6L(设备名 "YSYS X6L Camera")、后插 DCX(设备名
#   "USB Camera")，两次都恰好落在 /dev/video2 只是巧合；写死索引的脚本
#   换一次设备就会在第一步挂掉。→ index=None 时按「设备名」自动挑。
#
# ★ UVC 摄像头会额外注册一个 metadata 节点（/sys/.../index != 0）。
#   它不能用 cv2.VideoCapture 打开，打开会报 "can't open camera by index"，
#   看着像“摄像头坏了”，其实就是挑错了节点 —— 必须过滤掉。

# 设备名里含这些词 → 不要（笔记本内置 / 红外 / 元数据节点）
_NAME_EXCLUDE = ("integrated", "metadata", "ir camera", "infrared")
# 想优先挑哪个：按顺序匹配，命中不了就退回「第一个外接的」
_NAME_PREFER = ("usb camera", "ysys", "dcx", "x6l")


def _sysfs(video_node: str, attr: str) -> str:
    """读 /sys/class/video4linux/<node>/<attr>，读不到就返回空串。"""
    try:
        with open(f"/sys/class/video4linux/{video_node}/{attr}") as f:
            return f.read().strip()
    except OSError:
        return ""


def list_cameras() -> list[dict]:
    """列出系统里所有 V4L2 视频节点（含 metadata 节点，靠 metadata 字段区分）。"""
    cams = []
    for path in glob.glob("/dev/video*"):
        node = os.path.basename(path)
        m = re.search(r"(\d+)$", node)
        if not m:
            continue
        cams.append({
            "index":    int(m.group(1)),
            "path":     path,
            "name":     _sysfs(node, "name") or "(无名称)",
            "metadata": _sysfs(node, "index") not in ("", "0"),
        })
    return sorted(cams, key=lambda c: c["index"])


def find_camera(hint: str | None = None) -> dict | None:
    """
    自动挑一个能拍二维码的摄像头。
    优先外接（排除笔记本内置/红外）；一个外接都没有时退回内置。
    """
    usable = [c for c in list_cameras() if not c["metadata"]]

    if hint:
        for c in usable:
            if hint.lower() in c["name"].lower():
                return c
        return None

    external = [c for c in usable
                if not any(x in c["name"].lower() for x in _NAME_EXCLUDE)]
    pool = external or usable
    for pref in _NAME_PREFER:
        for c in pool:
            if pref in c["name"].lower():
                return c
    return pool[0] if pool else None


def describe_cameras() -> str:
    """给「打不开摄像头」的报错用：把当前所有节点列出来。"""
    lines = []
    for c in list_cameras():
        tag = "   [metadata，不能当摄像头打开]" if c["metadata"] else ""
        lines.append(f"       {c['path']}  {c['name']}{tag}")
    return "\n".join(lines) if lines else "       (一个 /dev/video* 都没有 — 摄像头没插好?)"


def lock_focus(cap, value, verbose: bool = True) -> bool:
    """
    把相机切到**手动焦距**并锁死这个值。返回 True = 真的锁上了。

    ★ 为什么必须锁（2026-09-20 实测，别改回去「光等自动对焦」）:
      吸盘码和纸面**不在同一个焦面** —— 吸盘码离相机 ~150mm、纸面 ~340mm。
      这颗相机（GP_Flip_Mirror 8M USB camera）的自动对焦停在一个**随它高兴**的
      位置上，实测吸盘码「时而解得出、时而解不出」；30 帧一帧都撞不上时整个
      流程就卡死在「没解出吸盘码」。等着自动对焦收敛解决不了这件事 ——
      它的判据（纸面码能不能解）太宽松，纸面码在 220 和 255 都解得出来。

      手扫一遍焦距（1080p 1920x1080，悬停高度，每个值 12 帧）:

          FOCUS  吸盘码   纸面4码全中
           210   0/12      11/12         焦点偏**远**（对着纸面）
           220   0/12      12/12
           230   0/12      12/12
           240  11/12      12/12   ← 稳的窗口从这儿起
           250  12/12      12/12   ★ FOCUS_LOCK
           260  12/12      12/12   ← 到这儿为止
           270  12/12       7/12         焦点偏**近**（对着吸盘码）
           280   7/12       3/12
           290  12/12       0/12
           320  12/12       0/12
           340  11/12       0/12         再大就糊成一片

      → 240~260 是唯一「吸盘码和纸面 4 码**同时**稳」的窗口。
        ★ 换一跑重扫，230/240 那几个边缘值的命中数会在几个帧之间跳
          （同一个 240，两跑分别量到 9/12 和 11/12）—— 所以取**窗口中间**，
          别贪某个单跑的峰值: 边缘上锁过去，一抖就掉出去。

      量法就是 tools/measure_focus.py（它会把建议值打出来）。
      锁死之后 wait_for_focus 第一帧就满足判据，等于不再需要等自动对焦 ——
      那 30 帧取并集的统计补丁（measure_suction_map.locate 的说明）也就不必再靠运气。

    ★ 副作用（知道的）: 锁焦距是**改相机自己的状态**，脚本退出后不会自动还原，
      而且 V4L2 的控制跨进程不一定留得住（实测重开可能回到自动对焦）。
      所以每个开相机的入口都显式锁一次，别指望「上次锁过」。
    """
    if value is None or value < 0:
        return False
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, int(value))
    got = float(cap.get(cv2.CAP_PROP_FOCUS))
    ok = abs(got - float(value)) <= 1.0
    if verbose:
        if ok:
            print(f"  焦距锁定: FOCUS={int(value)}（自动对焦已关）")
        else:
            print(f"  ⚠️ 焦距锁不上: 设 {int(value)} 读回 {got:g}"
                  f" —— 这颗相机可能不吃 UVC 焦点控制，只能退回等自动对焦")
    return ok


def open_camera(index: int | None = None, width: int = 1920, height: int = 1080,
                fourcc: str = "MJPG", warmup: int = 5,
                hint: str | None = None, verbose: bool = True,
                focus: int | None = None) -> cv2.VideoCapture:
    """
    打开摄像头；MJPG 才能跑满 1080p/4K 帧率。

      index=None → 自动挑外接摄像头（推荐，换设备/换 USB 口都不怕）
      index=N    → 用 /dev/videoN（向后兼容 --cam N）
      focus=V    → 开完机**顺手锁死手动焦距**（见 lock_focus）。
                   None = 不动它，继续靠自动对焦。

    ★ 分辨率只是「请求值」。不同型号上限不同（X6L=3840x2160，
      DCX-5MAF=2592x1944），超了就静默回落。不把实际值打印出来，
      很容易以为在跑 4K、其实只有 1080p —— 那样 px/模块 的账会全算错。
    """
    if index is None:
        cam = find_camera(hint)
        if cam is None:
            if verbose:
                print("  ⚠️ 没找到可用的摄像头，当前节点:")
                print(describe_cameras())
            return cv2.VideoCapture()
        index = cam["index"]
        if verbose:
            print(f"  自动选择摄像头: /dev/video{index}  ({cam['name']})")

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)          # 退回默认后端再试
    if not cap.isOpened():
        return cap

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(max(0, warmup)):            # 丢掉前几帧，等曝光/白平衡稳下来
        cap.read()

    if verbose:
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
        print(f"  实际分辨率: {aw}x{ah}  {fcc}")
        if (aw, ah) != (width, height):
            print(f"  ⚠️ 摄像头不支持 {width}x{height}，已回落到 {aw}x{ah}"
                  f"（X6L 上限 3840x2160，DCX-5MAF 上限 2592x1944）")
    # ★ 放在 warmup 之后: 先让曝光/白平衡稳，再动镜头 —— 而且锁完的第一帧
    #   就是清晰帧，调用方那边的 wait_for_focus 会立刻满足。
    if focus is not None:
        lock_focus(cap, focus, verbose=verbose)
    return cap


def wait_for_focus(cap, detector, expected, timeout: float = FOCUS_TIMEOUT,
                   quiet: bool = False):
    """
    等自动对焦收敛。返回 (最好的帧, 该帧blur, 每帧解出码数的列表)。

    ★ 实测这颗 YSYS X6L 的自动对焦要 3~7 秒才收住。预热不足时 blur≈28、
      4码全解不出，看着像“摄像头不行”，其实只是没等对焦。

    ★ 判据用「能否解出码」而不是「blur 够不够」: blur 阈值跟分辨率绑定，
      实测 4K 下 blur 只到 65 却能 100% 解出、1080p 要 ~178，
      固定阈值必然在某个分辨率上失灵，而“解出码”正是最终目的本身。

    ★ 计数只数 **expected 里的码**，不是「这一帧解出几个码」。
      画面里可能还有别的码（比如吸盘码），全算进去就会打印出「5/4 码」这种
      看着像 bug 的数。expected 给谁，这里就只对谁负责。

    ★ 现在**推荐先锁焦距**（open_camera(focus=...) → lock_focus）再进来:
      锁死了这颗函数第一帧就满足判据、立刻返回，那 3~7 秒的等待就省了。
      它留着当**退路** —— 相机不吃 UVC 焦点控制时，还是只能等自动对焦。
    """
    t0 = time.time()
    best_n, best_frame, best_blur = -1, None, -1.0
    blur_frame, blur_val = None, -1.0
    counts = []
    while time.time() - t0 < timeout:
        ok, f = cap.read()
        if not ok or f is None:
            break
        b = blur_score(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        decoded, _ = detect(f, detector)
        n = sum(1 for c in expected if c in decoded)
        counts.append(n)
        if n > best_n:
            best_n, best_frame, best_blur = n, f, b
        if b > blur_val:
            blur_val, blur_frame = b, f
        if n >= len(expected):
            if not quiet:
                print(f"  对焦收敛: 已解出 {n}/{len(expected)} 码, blur={b:.0f}"
                      f"  (等了 {time.time() - t0:.1f}s)")
            return f, b, counts
    if best_frame is None:
        return None, 0.0, counts
    if best_n <= 0 and blur_val > best_blur:
        best_frame, best_blur = blur_frame, blur_val   # 一个都没解出时取最清晰的
    if not quiet:
        print(f"  ⚠️ 对焦等待超时({timeout:.0f}s)，最多解出 {best_n}/{len(expected)} 个, "
              f"blur={best_blur:.0f}")
    return best_frame, best_blur, counts


def acquire_codes(cap, detector, expected, duration: float = 3.0,
                  max_frames: int = 120, quiet: bool = False):
    """
    在 duration 秒内反复抓帧，累积解出的码，直到 config 里的码全齐。

    为什么要多帧累积而不是抓一帧: 单帧会因对焦/曝光/噪声偶发丢码，
    而标定只需要四个点的位置，多等一两秒就能把成功率从"看运气"变成"稳"。
    返回 (decoded: {内容: 四边形}, frames_used, 每帧解出数列表)
    """
    t0 = time.time()
    union: dict[str, np.ndarray] = {}
    counts = []
    frames = 0
    while frames < max_frames and time.time() - t0 < duration:
        ok, f = cap.read()
        if not ok or f is None:
            break
        frames += 1
        decoded, _ = detect(f, detector)
        counts.append(len(decoded))
        for k, v in decoded.items():
            union.setdefault(k, v)
        if len(union) >= len(expected):
            break
    if not quiet:
        missed = [c for c in expected if c not in union]
        print(f"  多帧采样 {frames} 帧 / {time.time() - t0:.1f}s，"
              f"解出 {len(union)}/{len(expected)}" + (f"，缺 {missed}" if missed else ""))
    return union, frames, counts
