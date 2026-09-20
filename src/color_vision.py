#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_vision.py —— 认出「哪个方块在哪」，产出 world_state
=============================================================================
这是整条链路的**最上游**：DeepSeek 和机械臂都靠它给坐标。

    color_vision（本文件）  →  world_state  →  deepseek_brain ⑤  →  main.py ⑥  →  机械臂

────────────────────── 为什么不用「训练」 ──────────────────────
四个方块是**纯色**的（红黄蓝绿），方案第 61 行定的做法就是 HSV 阈值 + findContours。
这不是偷懒：训练一个模型要标数据、要显卡、要几百张图，而且**错了没法查** ——
阈值法错了一眼就能看出「是红色的 H 范围没罩住反光那一块」。
纯色物体上，阈值法又快又准又白盒。什么时候才需要训练: 方块表面有花纹、
颜色不固定、或者要在一堆杂物里找 —— 本项目的场景都不是。

───────────────────── 现在没有相机/标定也能用 ─────────────────────
    --hsv        **只实测 HSV、不算坐标** —— 二维码、标定、机械臂、底座全都不要。
                 只要方块 + 一个摄像头就能跑，用来调 HSV_RANGES。
                 ★ 这是「标定纸还没做好」时唯一能干的识别活儿: 阈值该定多少
                   取决于你的方块和你的灯，跟二维码一点关系都没有。

    --paper-mm   输出每个方块在**纸面**的毫米坐标，只用同一帧里的四个二维码换算。
                 ★ 不需要 step3 的矩阵、不需要机械臂、不需要底座。
                 你拿手机拍一张放好的方块，就能跑，能验「认不认得出、认得准不准」。

    默认         输出 world_state（**机械臂**毫米坐标），需要 output/hand_eye_matrix.json
                 （step3 标定出来的）。

────────────────────────── z_level 从哪来 ──────────────────────────
★★ 相机**看不出**摞了几层 —— 俯拍图里「一个方块在地上」和「一个方块摞在另一个上」
   长得一模一样。所以 z_level 不归视觉管，它归「记忆库」管:
   方案第五阶段第 3 条 —— 每执行完一条指令，把桌面状态刷新一遍（见 main.py 的
   refresh_world_state）。本文件只填 x/y，z_level 一律**从上一份 world_state 抄**，
   抄不到就是 0（地上）。
   这也是为什么这份状态必须落盘: 断了就再也补不回来。

用法:
  python3 src/color_vision.py --selftest              # 离线自检（合成图，不用相机）
  python3 src/color_vision.py --camera --hsv          # ★ 调阈值: 只见方块，不要二维码
  python3 src/color_vision.py --image 照片.jpg --hsv --debug  # 同上，用现成的照片
  python3 src/color_vision.py --image 照片.jpg --paper-mm    # 只用纸面坐标，不碰机械臂
  python3 src/color_vision.py --camera --paper-mm            # 现场看，不碰机械臂
  python3 src/color_vision.py --camera                       # 出 world_state（要标定过）
  python3 src/color_vision.py --image 照片.jpg --debug       # 出调试图，看掩膜干不干净
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (COLOR_DEBUG_PNG, PAPER_JSON, WORLD_HALF_KEY,    # noqa: E402
                   WORLD_HALF_LEFT, WORLD_HALF_RIGHT, WORLD_SRC_CAMERA,
                   WORLD_SRC_KEY, WORLD_STATE_JSON, ensure_output_dir)
import qr_vision as qv                                            # noqa: E402
from qr_vision import (DATA_MODULES, MODULES_TOTAL,               # noqa: E402
                       detect as qr_detect, load_paper_layout,
                       px_per_mm, quad_center)


# ═══════════════════════════ 一、HSV 阈值 ═══════════════════════════
# OpenCV 的 HSV 口径（别拿别处的数字套）: H 0~179, S 0~255, V 0~255。
#   H 把 360° 压缩成 180 格 → 红色在**两头**（0 附近和 180 附近）。
#
# ★★ 红色为什么是两条区间: 红色的色相正好跨在 0/180 的接缝上。只写 (0..10, ...)
#    会把偏紫的那半边红判成「不是红」；只写 (170..180, ...) 则漏掉偏橙的那半边。
#    这是纯色识别里最经典的一个坑，本项目四个方块里只有红色会踩到。
#
# ★ 这些数**是要按你的方块和灯光调的**，不是真理。调法:
#    1) 先跑 --image 照片.jpg --debug，看 output/color_debug.png 里的框有没有套准
#    2) 掩膜缺一块（反光/阴影）→ 放宽 S 或 V 的下限
#    3) 掩膜连成一片（两个方块粘住）→ 收紧 S/V 下限，或看下面的「贴太近」检查
#   灯光一变（白天/晚上/开台灯）就要重调一次 —— 这是阈值法唯一的代价。
HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red":    [((0, 110, 80), (10, 255, 255)),
               ((170, 110, 80), (180, 255, 255))],     # ★ 两头都要，见上
    "yellow": [((20, 110, 80), (35, 255, 255))],
    "green":  [((40, 90, 60), (85, 255, 255))],
    # ★ blue 的 V 下限从 60 降到 45: 实测这个蓝方块的**顶面** V 中位数只有 54
    #   （暗面朝上），卡在 60 时顶面整片被砍掉、只剩朝光的那条侧面亮边，
    #   于是掩膜是一条月牙、fill 0.34。降到 45 后顶面连上，成一块 172x155px
    #   （≈35mm，fill 0.62）。再往下（40）面积几乎不涨 —— 说明 45 已到边界，
    #   没有把背景吃进来，所以停在这里，别再降。
    "blue":   [((100, 110, 45), (130, 255, 255))],
}
CUBE_COLORS = tuple(HSV_RANGES)          # 顺序固定，输出 world_state 时按它排

# 方块尺寸的合理性检查（像素→毫米用二维码给的 px/mm 换算）。
# ★ 为什么要按 mm 而不是按像素面积筛: 相机高低一变，同样的方块像素面积差好几倍，
#   写死像素阈值等于把代码绑死在一个机位上。走 mm 就跟机位无关了。
MIN_SIDE_MM = 18.0        # 方块比这小 → 不是方块（噪点、反光碎块）
MAX_SIDE_MM = 45.0        # 比这大 → 两块粘一起了，或者根本不是方块
MIN_FILL = 0.55           # 轮廓面积 / 最小外接矩形面积。太小说明形状不成块

# 「桌面全平放」判据的门槛（见 reset_levels_if_flat）。
# ★ 为什么是 25 而不是方块边长 30: 方块紧挨着放（DeepSeek 的「放到右边」就是这么摆，
#   中心距正好 30mm）是**合法**的平放摆法，门槛取 30 会把正常平放判成叠放。
#   而两块真摞在一起时中心距≈0（俯视图里几乎重合），离 25 也很远 —— 两边都不沾。
FLAT_MIN_GAP_MM = 25.0

# 没拿到 px/mm（二维码没解出来）时的退路: 只按像素面积粗筛。
# ★ 这个退路**不可靠**，会顺着机位漂 —— 所以只在真拿不到 px/mm 时用，并且会打印警告。
MIN_AREA_PX_FALLBACK = 150

# ── --hsv 的「放宽探针」参数（只用来**量**，不参与识别，见 loose_color_mask）──
H_SLACK = 10              # 量 H 时把配置区间往两边各放宽这么多
LOOSE_SV = 40             # 量 S/V 时把下限降到这么低去探底


class VisionError(Exception):
    """认得出来但结果不能用（缺方块、缺二维码…）—— 报错要说人话。"""


# ═══════════════════════════ 二、认颜色 ═══════════════════════════
def color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """某个颜色的二值掩膜（多条 H 区间取并集，红色就靠这个）。"""
    if color not in HSV_RANGES:
        raise VisionError(f"不认识的颜色 {color!r}，只有 {CUBE_COLORS}")
    mask = None
    for lo, hi in HSV_RANGES[color]:
        part = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    开运算去噪点 → 闭运算补内部小洞（反光会让方块中间出现小孔）。

    ★ 顺序不能反: 先闭后开会把噪声和方块粘在一起再一起去不掉。
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)


def detect_cubes(frame: np.ndarray, ppm: float | None = None) -> dict[str, dict]:
    """
    认出每种颜色最大的那一块。返回 {颜色: {px, py, side_mm, area_mm2}}。

    ppm = 每毫米多少像素（来自同一帧的二维码）。给不出来就退回像素粗筛。

    ★ 一种颜色找到多块时**取最大的那块并报警**，不是直接报错: 桌上除了方块
      还有别的东西（红色笔、红色胶带）是常态，直接罢工太脆。但会明确打出来，
      免得你以为认到的是方块。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found: dict[str, dict] = {}
    for color in CUBE_COLORS:
        mask = clean_mask(color_mask(hsv, color))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        good = []
        for c in cnts:
            (_, _), (w, h), _ = cv2.minAreaRect(c)
            if w <= 0 or h <= 0:
                continue
            area = float(cv2.contourArea(c))
            if ppm:
                sw, sh = w / ppm, h / ppm
                if not (MIN_SIDE_MM <= min(sw, sh) and max(sw, sh) <= MAX_SIDE_MM):
                    continue
                if area / (w * h) < MIN_FILL:
                    continue
            else:
                if area < MIN_AREA_PX_FALLBACK:
                    continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            good.append({"px": M["m10"] / M["m00"], "py": M["m01"] / M["m00"],
                         "side_mm": (max(w, h) / ppm) if ppm else None,
                         "area_mm2": (area / ppm / ppm) if ppm else None,
                         "contour_px": area})

        if not good:
            continue
        # ★ 没有 px/mm 时按**像素面积**（contour_px）排，不是按轮廓个数/发现顺序。
        #   这里曾经写成 len(good)（= 第几个被 findContours 找到的），于是"取最大
        #   的那块"实际取到的是**最后找到**的那块 —— 小块在前、大块在后时会挑错，
        #   还照样打印「取最大的那块」。自检 _selftest_hsv_probe 钉住了这一条。
        good.sort(key=lambda d: -d["area_mm2"] if ppm else -d["contour_px"])
        if len(good) > 1:
            # 没拿到 px/mm（--hsv 那条通路就是）时只有像素面积 —— 报 0mm² 会让人
            # 以为"这块是空的"，所以按手里有的单位报。
            amm, npx = good[0]["area_mm2"], good[0]["contour_px"]
            size = f"{amm:.0f}mm²" if amm else f"{npx:.0f}px（没拿到 px/mm）"
            print(f"  ⚠ {color}: 找到 {len(good)} 块，取最大的那块 (面积 {size})。"
                  f"桌上还有同色的东西？或者掩膜把两处粘一起了？")
        found[color] = good[0]
    return found


# ═══════════════════════ 三、像素 → 纸面毫米 ═══════════════════════
def px_to_paper_map(decoded: dict, layout) -> np.ndarray:
    """
    拟合 像素→纸面mm 的 3x3 矩阵（用同一帧里四个码的**中心**）。

    ★ 为什么要反过来拟合（码中心px → 已知纸面mm），而不是像 qr_vision.paper_fit
      那样拟合 纸面mm→像素 再求逆: 4 点单应虽然可逆，但代码上直接拟合目标方向
      更不容易写错符号，而且点数不足时的报错更直白。
    """
    missing = [c for c in layout.codes if c not in decoded]
    if missing:
        raise VisionError(
            f"这一帧没解出全部二维码，缺 {missing} —— 没有它们就算不出像素↔毫米。\n"
            f"  · 把摄像头抬高/拉远，让四个码都进画面（连白边一起）\n"
            f"  · 或者先跑 tools/test_camera_qr.py 调机位")
    mm = np.float32([layout.mm_for("center")[c] for c in layout.codes])
    px = np.float32([quad_center(decoded[c]) for c in layout.codes])
    return cv2.getPerspectiveTransform(px, mm)          # px → mm


def paper_xy(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """把像素点按 3x3 透视矩阵投到纸面 mm。"""
    v = np.array([px, py, 1.0]) @ np.asarray(H, np.float64).T
    if abs(v[2]) < 1e-12:
        raise VisionError("透视变换退化（分母为 0）—— 点数是不是共线了？")
    return float(v[0] / v[2]), float(v[1] / v[2])


# ════════════════════ 四、纸面毫米 → 机械臂毫米 ════════════════════
def load_robot_matrix():
    """
    读 step3 标定出的 像素→机械臂 矩阵。没标定过就报错，**不兜底**。

    ★ 为什么不兜底: 手眼矩阵绑死「这台机械臂 + 这个机位」。拿一个猜的矩阵
      继续跑，坐标会错得**不报错** —— 全项目最危险的失败方式。
    """
    from step3_hand_eye_calib import load_matrix, pixel_to_robot
    from paths import MATRIX_JSON
    if not MATRIX_JSON.exists():
        raise VisionError(
            f"没有 {MATRIX_JSON.name} —— 还没做过手眼标定，换算不出机械臂坐标。\n"
            f"  想先不碰机械臂地验证识别:\n"
            f"      python3 src/color_vision.py --image 照片.jpg --paper-mm\n"
            f"  要出机械臂坐标得先:\n"
            f"      python3 src/step2_teach_coords.py --reference corner_tr\n"
            f"      python3 src/step3_hand_eye_calib.py")
    return load_matrix(), pixel_to_robot


# ═══════════════════════════ 五、总装 ═══════════════════════════
def prior_z_levels() -> dict[str, int]:
    """
    从上一份 world_state 里抄 z_level（相机看不出层数，只能靠记忆库）。

    ★ 抄不到就当 0（在地上）。这是**唯一合理**的默认值: 开机时方块都是平放的。
      真漏了（比如你把方块摞好了再开机）就手工改 output/world_state.json。
    """
    if not WORLD_STATE_JSON.exists():
        return {}
    try:
        d = json.loads(WORLD_STATE_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: int(v.get("z_level", 0)) for k, v in d.items()
            if isinstance(v, dict) and "z_level" in v}


def half_in_paper(H: np.ndarray, layout, px: float, py: float) -> str | None:
    """
    ★ 「这个像素点落在纸的哪半边」—— 返回 "left"/"right"，判不出来返回 None。

    ★ 为什么拿**纸自己的坐标系**判，而不是拿机械臂坐标: 用户说的「P1/P3 这左半侧」
      是**纸面上**的左右。纸面的 x 钉死在标定纸定义里（P1 x≈29.5、P2 x≈242.5），
      纸被挪了、机位被撞了，它都不变；而机械臂坐标（Y 轴）是跟着机位一起变的，
      拿它划线，下次重架摄像头就得重新量一个阈值。判据跟纸走，永远不用重标。
    ★ 判不出来（透视退化）返回 None —— 调用方据此**不写** half 键，也就不会补偿。
      宁可少补 5mm，也不赌错方向白偏。
    """
    try:
        x_paper, _ = paper_xy(H, px, py)
    except VisionError:
        return None
    return (WORLD_HALF_LEFT if x_paper < float(layout.paper_mm[0]) / 2.0
            else WORLD_HALF_RIGHT)


def build_world_state(cubes: dict[str, dict], mode: str, decoded, layout,
                      robot=None, prior=None) -> dict:
    """
    mode="paper"  → 输出纸面 mm（不用标定，验证识别用）
    mode="robot"  → 输出机械臂 mm（要标定矩阵）

    返回 {颜色: {"x":.., "y":.., "z_level":.., "src":"camera"[, "half":"left"/"right"]}}；
    缺方块会抛 VisionError。

    ★ half 是「这块在纸的哪半边」，只用来决定「第一次抓取要不要补偿 5mm」
      （见 main.FIRST_GRASP_OFFSET_MM 与 paths.WORLD_HALF_KEY 的说明）。
      四个码没解齐时**不写这个键** —— 判不出来就不补偿，不猜。
    """
    prior = prior if prior is not None else prior_z_levels()
    missing = [c for c in CUBE_COLORS if c not in cubes]
    if missing:
        # ★ 顺手报「认到了什么」: 一个都没认到（镜头盖/选错相机）和只缺一个
        #   （HSV 阈值偏了/被挡住）是两种完全不同的毛病，光看缺的名单分不出来。
        got = list(cubes) or ["（一个都没有）"]
        raise VisionError(
            f"没认出来的颜色: {missing}；本帧只认到: {got}\n"
            f"  · 方块摆出来了没？被别的方块挡住了没？\n"
            f"  · 跑 --debug 看 output/color_debug.png 里哪个颜色的框没套上\n"
            f"  · 是的话调 HSV_RANGES（本文件开头有调法）")

    # ★ 「像素→纸面」矩阵: 两个模式都要用它判半侧，所以在这儿算**一次**。
    #   paper 模式没有它整体就没意义 —— 原样抛出（下面分支里 re-raise 同一个异常）。
    #   robot 模式没有它只是判不出半侧 —— 容忍，不写 half 键。
    H_paper, paper_err = None, None
    try:
        H_paper = px_to_paper_map(decoded, layout)
    except VisionError as e:
        paper_err = e

    def _entry(color: str, x: float, y: float) -> dict:
        """
        ★ 每条记录都盖两个戳（都是主程序内部用的，不是给模型的字段）:
          · WORLD_SRC_KEY = "camera" —— 「这条 x/y 是相机给的，机械臂还没碰过」。
          · WORLD_HALF_KEY —— 「在纸的左半边还是右半边」，判不出来就不写这个键。
        主程序靠它们判断「第一次抓取要不要补偿 5mm」（见 main.FIRST_GRASP_OFFSET_MM）；
        执行完 refresh_world_state 会把这条**整条重写**（新记录两个键都没有），
        于是下一次抓它就是真实值了。两个戳都只说"这条坐标的来历"，
        **不参与坐标计算** —— print_table / z_level / 各类判据都只认 x/y/z_level。
        """
        e = {"x": round(x, 2), "y": round(y, 2),
             "z_level": int(prior.get(color, 0)),
             WORLD_SRC_KEY: WORLD_SRC_CAMERA}
        if H_paper is not None:
            half = half_in_paper(H_paper, layout, cubes[color]["px"], cubes[color]["py"])
            if half is not None:
                e[WORLD_HALF_KEY] = half
        return e

    state: dict[str, dict] = {}
    if mode == "paper":
        if H_paper is None:
            raise paper_err                 # 上面已经试过了，原样抛出，别重复拟合
        for color in CUBE_COLORS:
            c = cubes[color]
            x, y = paper_xy(H_paper, c["px"], c["py"])
            state[color] = _entry(color, x, y)
    else:
        if robot is None:
            raise VisionError("robot 模式要传 (矩阵, pixel_to_robot)")
        M, px_to_robot = robot
        for color in CUBE_COLORS:
            c = cubes[color]
            x, y = px_to_robot(M, c["px"], c["py"])
            state[color] = _entry(color, x, y)
    return state


def reset_levels_if_flat(state: dict) -> str | None:
    """
    ★ 「桌面全平放」判据 —— 就地把所有 z_level 归 0，返回人话说明；不该归就返回 None。

    判据: 四个方块**全认出来了**，而且两两之间的中心距都 >= FLAT_MIN_GAP_MM。
          俯视图里两块摞在一起 = 中心几乎重合（下面那块被挡住，多半根本认不出来），
          所以「四块都看得见 + 占地互不重叠」就**证明**桌上没有摞。

    ★ 为什么必须有这一条（2026-09-18 真事）: 相机重新识别会把 x/y 刷成新的，
      而 z_level 是 build_world_state 从记忆库照抄的 —— 手工把方块拆开重摆之后，
      位置变了、层数还是老的（绿块明明平放着，记忆里写着第 3 层）。
      吸盘于是按第 3 层去抓: 多抬 75mm、吸空 —— 这条指令就废了。
      层数估**低**才是危险方向（吸盘多压 25mm 会撞），所以判据必须是**证明**，
      不能是"猜个大概"，这也是为什么只做全局判据、不做单块容差比对。

    不归 0 的情况（返回 None，记忆库保持原样）:
      · 有任意两块靠得比门槛近 —— 可能真摞着（也可能只是摆得挤），宁可不动；
      · 少了任何一块 —— 证据不全（缺的那块多半正被压在下面），不能下结论；
      · 层数本来全是 0 —— 没什么可清的，不吭声。

    ★ 覆盖不到的情况（相机原理上做不到，只能手工改 output/world_state.json）:
      你把方块**手工**摞起来、而且对准了摞 —— 下面那块被完全挡住、四块凑不齐，
      build_world_state 会直接报错，根本走不到这里。
    """
    if any(c not in state for c in CUBE_COLORS):
        return None
    names = list(state)
    worst = min(float(np.hypot(state[a]["x"] - state[b]["x"],
                               state[a]["y"] - state[b]["y"]))
                for i, a in enumerate(names) for b in names[i + 1:])
    if worst < FLAT_MIN_GAP_MM:
        return None
    dropped = {c: int(v["z_level"]) for c, v in state.items() if v["z_level"]}
    if not dropped:
        return None
    for v in state.values():
        v["z_level"] = 0
    who = "、".join(f"{c}(原记第 {lv} 层)" for c, lv in dropped.items())
    return (f"桌上这四块都看得见、两两最少也隔 {worst:.0f}mm —— 不可能有摞，"
            f"层数一律按 0 算（{who}）")


def sanity_check(state: dict, mode: str, layout) -> list[str]:
    """一眼能看出来的毛病。返回问题列表（空 = 没看出问题）。不拦路，只提醒。"""
    probs: list[str] = []
    # 两个方块落得太近 → 要么认重了，要么方块真的叠在一起（叠着时相机只能看到上面那块）
    names = list(state)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = np.hypot(state[a]["x"] - state[b]["x"], state[a]["y"] - state[b]["y"])
            if d < MIN_SIDE_MM * 0.8:
                probs.append(f"{a} 和 {b} 只差 {d:.1f}mm —— 靠太近了。"
                             f"是不是有一块被挡住了？相机只能看到最上面那块")
    if mode == "paper":
        pw, ph = layout.paper_mm
        for c, v in state.items():
            if not (-10 <= v["x"] <= pw + 10 and -10 <= v["y"] <= ph + 10):
                probs.append(f"{c} 算到纸面 ({v['x']:.0f}, {v['y']:.0f})，"
                             f"跑到纸外了（纸是 {pw:.0f}x{ph:.0f}）—— "
                             f"要么方块真在纸外，要么二维码认歪了")
    return probs


def draw_debug(frame: np.ndarray, cubes: dict, state: dict | None,
               layout) -> np.ndarray:
    """把认到的框、名字、坐标画在图上。★ 调 HSV 就靠这张图。"""
    out = frame.copy()
    for color, c in cubes.items():
        cx, cy = int(round(c["px"])), int(round(c["py"]))
        # 用该颜色的 HSV 中值画框，这样「框的颜色」本身也提示掩膜质量
        lo, hi = HSV_RANGES[color][0]
        hsv = np.uint8([[[(lo[0] + hi[0]) // 2, 255, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), 4, tuple(int(v) for v in rgb), 2)
        label = color
        if state and color in state:
            label = f"{color} ({state[color]['x']:.0f},{state[color]['y']:.0f})"
        cv2.putText(out, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)          # 描边，深浅底都看得清
        cv2.putText(out, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def analyze(frame: np.ndarray, mode: str, layout, robot=None,
            prior=None) -> tuple[dict, dict, float | None, dict]:
    """
    一帧图 → (world_state, 认到的方块, px/mm, 二维码)。所有模式的公共路径。

    px/mm 顺便从二维码量一下: 它既是尺寸筛选的依据，也是一个**机位自检**
    —— 这个数比标定时小很多，就说明摄像头被挪近了/画面糊了。

    ★ robot=(矩阵, pixel_to_robot)、prior={颜色: 层数} 两个口子是给
      setup_camera.py 留的: 它手里那个矩阵是**刚算出来、还没落盘**的，
      不传进来就会走去读磁盘上的**旧**矩阵 —— 而旧矩阵正是要换掉的那个，
      用它会算出一整套偏掉的坐标，还不报错。默认 None = 老行为（读磁盘）。
    """
    decoded, located = qr_detect(frame, cv2.QRCodeDetector())
    ppm = None
    if all(c in decoded for c in layout.codes):
        ppm = min(px_per_mm(decoded[c]) for c in layout.codes)
    elif decoded:
        print(f"  ⚠ 只解出 {sorted(decoded)}，缺 "
              f"{[c for c in layout.codes if c not in decoded]} —— "
              f"尺寸筛选退回像素口径，结果的可靠性下降")
        ppm = min(px_per_mm(decoded[c]) for c in decoded)
    else:
        print("  ⚠ 一个二维码都没解出 —— 尺寸筛选退回像素口径")
    if located:
        print(f"  ℹ 还有 {len(located)} 个方块形的东西只定位到、没解出内容（不是本项目的码）")

    cubes = detect_cubes(frame, ppm)
    if robot is None and mode == "robot":
        robot = load_robot_matrix()
    state = build_world_state(cubes, mode, decoded, layout, robot=robot, prior=prior)
    # ★ 记忆库会把上次的层数照抄进来，但这一帧要是**证明**了桌面是平放的，
    #   那份层数就是过期的（你手工拆开重摆过），当场清掉。见 reset_levels_if_flat。
    flat_note = reset_levels_if_flat(state)
    if flat_note:
        print(f"  ★ {flat_note}")
    return state, cubes, ppm, decoded


# ═══════════════════════ 六、输入（相机 / 图片） ═══════════════════════
def frame_from_camera(which: int | None) -> np.ndarray:
    """抓一帧。★ 用完立刻放掉 —— 别占着摄像头，后面 step3 还要用。"""
    cap = qv.open_camera(which)
    try:
        for _ in range(10):                       # 丢掉曝光/白平衡还没稳的头几帧
            cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VisionError("摄像头读不出画面（被别的程序占着？）")
        return frame
    finally:
        cap.release()


def frame_from_file(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise VisionError(f"找不到图片 {p}")
    frame = cv2.imread(str(p))
    if frame is None:
        raise VisionError(f"{p.name} 读不了 —— 是不是 HEIC/RAW 这种 cv2 不认的格式？"
                          f"转成 JPG/PNG 再来")
    return frame


# ═══════════════════════════ 七、输出 ═══════════════════════════
def print_table(state: dict, mode: str) -> None:
    unit = "纸面 mm" if mode == "paper" else "机械臂 mm"
    print(f"\n  {'颜色':<8}{'X':>10}{'Y':>10}{'层':>5}   ({unit})")
    for c in CUBE_COLORS:
        if c not in state:
            continue
        v = state[c]
        print(f"  {c:<8}{v['x']:>10.1f}{v['y']:>10.1f}{v['z_level']:>5}")


def save_world_state(state: dict, path: Path | None = None) -> Path:
    """
    ★ 写盘这件事**只有这一处实现** —— 别的脚本一律调这里，别自己 write_text。
      记忆库的 .bak 备份、编码、缩进都靠这个函数统一保证。

    path 只在自检里用（往临时目录写，不碰真的 output/）；生产路径不传。
    """
    p = Path(path) if path else WORLD_STATE_JSON
    p.parent.mkdir(parents=True, exist_ok=True)
    # ★ 带上一份就备份: 记忆库是「一次执行一次刷新」累出来的，被一次误跑覆盖掉
    #   就没法还原了（相机只能给 x/y，z_level 补不回来）。
    if p.exists():
        bak = p.with_suffix(".json.bak")
        bak.write_bytes(p.read_bytes())
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ════════ 八、HSV 实测：没有二维码、没有标定、没有机械臂时的入口 ════════
# ★ 这一段是**唯一**完全不需要标定纸的识别通路，为什么单独做一条:
#   阈值表 HSV_RANGES 是拿合成图定不出来的 —— 它取决于你的方块材质、
#   你的灯光、你的摄像头。合成图里是纯色，真机上有反光、阴影、白平衡偏移。
#   而「调阈值」本来只需要**方块 + 摄像头**，和二维码一点关系都没有。
#   以前要走 analyze()，它必须先解出四个码才肯干活，于是「纸还没做好」
#   就被卡住了 —— 那是把两件不相干的事绑在了一起。
#
#   这里报的是**实测到的 HSV 区间**，不是帮你自动算一套阈值:
#   自动算出来的数在换一张桌子/换一盏灯之后就是错的，而你不知道它错在哪。
#   看到数、和 HSV_RANGES 比一眼，改哪个数是你的事。

def _hue_wraps(color: str) -> bool:
    """
    这个颜色的 H 区间是不是跨在 0/180 接缝上（四个色里只有红色）。

    ★ 从 HSV_RANGES **推**出来，不是写死 "red" —— 以后加了橙色、紫色之类的
      跨界色，这里自动跟着走，不用再想起来改一处。
    """
    rngs = HSV_RANGES[color]
    if len(rngs) < 2:
        return False
    lows = [lo[0] for lo, _ in rngs]
    highs = [hi[0] for _, hi in rngs]
    return min(lows) <= 10 and max(highs) >= 170


def _h_bounds(color: str) -> tuple[int, int]:
    """
    配置的 H 区间，**按折算后的口径**给（红 = -10 ~ +10，不是 0~10 / 170~180）。

    ★ 为什么要折算: 报出来的 H 是折过的，拿没折的边界去比就成了拿 170 跟 -9 比。
    """
    rngs = HSV_RANGES[color]
    if _hue_wraps(color):
        lows = [lo[0] - 180 if lo[0] > 90 else lo[0] for lo, _ in rngs]
        highs = [hi[0] - 180 if hi[0] > 90 else hi[0] for _, hi in rngs]
    else:
        lows = [lo[0] for lo, _ in rngs]
        highs = [hi[0] for _, hi in rngs]
    return min(lows), max(highs)


def loose_color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """
    「放宽掩膜」: H 往两边各放宽 H_SLACK、S/V 下限降到 LOOSE_SV。**只给 --hsv 量数用**，
    不参与任何识别 —— 识别走的是 color_mask()。

    ★ 为什么非要多做这一张: 直接拿配置掩膜里的像素去量配置的边界，是**循环论证** ——
      被区间挡在外面的像素压根不在掩膜里，所以那样量出来的 p5 永远 ≥ 配置下限、
      p95 永远 ≤ 配置上限，**永远说不出「你的区间写窄了」**。
      实测过: 一块 H 铺满 14~35 的黄方块，配置写 H 20-35，它会报「p5 = +20」
      正好压在下界上、显示「✅ 采用」，而方块左边那一大截其实一直在丢。
      放宽之后再量，p5/p95 才可能跑到配置区间**外面**去 —— 跑出去了就是该放宽。
    """
    mask = None
    for lo, hi in HSV_RANGES[color]:
        l = (max(0, lo[0] - H_SLACK), LOOSE_SV, LOOSE_SV)
        h = (min(179, hi[0] + H_SLACK), 255, 255)
        if l[0] > h[0]:                      # 放宽后两头错开了（极端配置），跳过
            continue
        part = cv2.inRange(hsv, np.array(l, np.uint8), np.array(h, np.uint8))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    if mask is None:
        return np.zeros(hsv.shape[:2], np.uint8)
    return mask


def _loose_component(hsv: np.ndarray, mask: np.ndarray, color: str) -> np.ndarray:
    """
    放宽掩膜里、**和现配置那块连长在一起**的那一块 —— 只有它才是同一个物体。

    ★ 不筛的话: 放宽 H 会把邻色的东西也圈进来（green 放宽到 30-95 就吃掉了
      H≈32 的橙黄），量出来的 H 和像素数全是**别人**的。实测过: 会报
      「放宽后那块多 13500px」这种假警报。
    """
    lm = clean_mask(loose_color_mask(hsv, color))
    n_lab, lab = cv2.connectedComponents(lm)
    if n_lab > 1 and mask.any():
        hit = np.bincount(lab[mask > 0].ravel(), minlength=n_lab)
        hit[0] = 0                                       # 0 是背景，不算
        if hit.max() > 0:
            lm = np.where(lab == int(hit.argmax()), 255, 0).astype(np.uint8)
    return lm


def _blob_stats(hsv: np.ndarray, mask: np.ndarray, color: str) -> dict | None:
    """在掩膜的最大一块里量形状 + H/S/V 的 p5/p50/p95。没轮廓就返回 None。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    area = float(cv2.contourArea(c))
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [c], -1, 255, -1)
    px = hsv[filled > 0]
    hue = px[:, 0].astype(np.int32)
    if _hue_wraps(color):
        # 把 170~180 那半边折成负数，否则百分位数会落在 0 和 180 中间，
        # 看着像「色相乱七八糟」，其实是一条连续的窄带。
        hue = np.where(hue > 90, hue - 180, hue)
    return {"blk": {"area": area, "long": max(w, h), "short": min(w, h),
                    "fill": area / (w * h) if w * h else 0.0,
                    "n": len(cnts)},
            "hsv": {"H": np.percentile(hue, [5, 50, 95]),
                    "S": np.percentile(px[:, 1], [5, 50, 95]),
                    "V": np.percentile(px[:, 2], [5, 50, 95])}}


def hsv_probe(frame: np.ndarray) -> list[dict]:
    """
    每种颜色的掩膜实测: 多少像素、最大那块什么样、那块里的 HSV 落在哪。

    ★ 「采用与否」调的是 detect_cubes() 本身（传 ppm=None），**不另写一套判据** ——
      另写一套的话，这里显示「合格」而真正跑的时候被筛掉，就成了骗人的工具。
    ★ 同时给一份「放宽后」的数（见 loose_color_mask）: 现配置那份量不出区间写窄了。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    total = float(frame.shape[0] * frame.shape[1])
    adopted = detect_cubes(frame, None)        # 真正那条筛选路径

    rows: list[dict] = []
    for color in CUBE_COLORS:
        mask = clean_mask(color_mask(hsv, color))
        n = int(cv2.countNonZero(mask))
        row = {"color": color, "px": n, "pct": 100.0 * n / total,
               "adopted": color in adopted, "blk": None, "hsv": None, "loose": None}
        st = _blob_stats(hsv, mask, color)
        if st:
            row["blk"], row["hsv"] = st["blk"], st["hsv"]
        # 放宽后再量一份: 这一份才可能跑到配置区间外面，见 loose_color_mask
        lm = _loose_component(hsv, mask, color)
        lst = _blob_stats(hsv, lm, color)
        if lst:
            lst["px"] = int(cv2.countNonZero(lm))
            row["loose"] = lst
        rows.append(row)
    return rows


def _verdict(row: dict) -> str:
    """
    这一色到底会不会被用上；没被用上是**卡在哪一步** —— 必须说清。

    ★★ 「采用」在这里是**打了折**的结论: --hsv 没有 px/mm，所以 detect_cubes
       的尺寸门槛（MIN_SIDE_MM~MAX_SIDE_MM）整个用不上，只剩一个 150px 的兜底。
       实测过: 笔记本摄像头对着房间，默认阈值能报出 17 万像素的「红」
       （8.5% 画面、24 块散着），而它会显示「✅ 采用」。
       所以这里把**不依赖 px/mm 的两个信号**补上: 形状（fill）和块数。
    """
    n, blk = row["px"], row["blk"]
    if not row["adopted"]:
        if n == 0:
            return "❌ 阈值完全没罩住（H 不在配置区间里，或 S/V 下限太高）"
        if blk is None:
            return "❌ 有掩膜像素但不构成轮廓（只剩噪点）"
        if blk["fill"] < MIN_FILL:
            return (f"⚠ 形状不成块（fill {blk['fill']:.2f} < {MIN_FILL}）——"
                    f"掩膜连成一片或碎成条: 收/放 S、V 的下限")
        if blk["area"] < MIN_AREA_PX_FALLBACK:
            return f"⚠ 最大那块只有 {blk['area']:.0f}px，小于 {MIN_AREA_PX_FALLBACK} 的兜底门槛"
        return "⚠ 有像样的掩膜却仍被筛掉 —— 看上面的告警行"

    if blk is None:
        return "✅ 采用"
    if blk["fill"] < MIN_FILL:
        return (f"⚠ 会被采用，但形状根本不成块（fill {blk['fill']:.2f} < {MIN_FILL}）——"
                f"没有 px/mm 时 detect_cubes 查不了 fill，这八成是一坨背景")
    if blk["n"] > 4:
        return (f"⚠ 会被采用，但同色有 {blk['n']} 块散在画面各处 ——"
                f"像是桌面/皮肤/木纹落进了阈值，不是方块")
    if blk["n"] > 1:
        return f"✅ 采用（同色 {blk['n']} 块，取最大那块）"
    return "✅ 采用"


def _loose_contaminated(row: dict) -> bool:
    """
    放宽后那块还**算不算同一个物体** —— 涨得太离谱就不是了。

    ★ 放宽 S/V 下限会把挨着的低饱和背景（桌面、木纹）一起连进来。实测过: 一块黄
      方块压在米色桌面上（桌面 S≈55），放宽后那块从 12100px 变成 945700px ——
      整个桌面都成了「同一块」。这时「多 Npx」和 S/V 的「放宽后」p5 **全是在量
      桌面**，照着它去降阈值正好把背景全收进来。
    """
    b, L = row["blk"], row["loose"]
    if not b or not L:
        return False
    return (L["blk"]["area"] > 3 * max(b["area"], 1)
            or (L["blk"]["long"] * L["blk"]["short"]
                > 3 * max(b["long"] * b["short"], 1)))


def _tri(L: dict | None, key: str, signed: bool = False) -> str:
    """把放宽后的 p5/p50/p95 打成一行。量不出来就一个破折号。"""
    if not L:
        return "—"
    f = "{:+.0f}" if signed else "{:.0f}"
    return " / ".join(f.format(x) for x in L["hsv"][key])


def _clip_notes(row: dict) -> list[str]:
    """
    拿**放宽后**的数去看配置切掉了什么 —— 这是「现配置」那一列永远看不出来的
    （掩膜按配置切出来，区间外的像素压根不在里面）。

    只在真切到东西时才出声，没切到就闭嘴。
    """
    L = row["loose"]
    if not L or not row["hsv"]:
        return []
    color = row["color"]
    hlo, hhi = _h_bounds(color)
    s_lo = min(lo[1] for lo, _ in HSV_RANGES[color])
    v_lo = min(lo[2] for lo, _ in HSV_RANGES[color])
    pH, pS, pV = L["hsv"]["H"], L["hsv"]["S"], L["hsv"]["V"]
    out = []
    if pH[0] < hlo - 0.5 or pH[2] > hhi + 0.5:
        out.append(f"H 在切边: 放宽后实测 {pH[0]:+.0f} ~ {pH[2]:+.0f}，"
                   f"配置只有 {hlo:+d} ~ {hhi:+d} —— 落在区间外的那截一直在丢，"
                   f"把配置往那边放宽")
    # ★ S/V 这条**不能只凭数字下结论**: 放宽后多进来的像素，可能是方块自己的
    #   阴影/侧面（那就该降下限），也可能是挨着的桌面/背景（那降了就把背景也收进来）。
    #   单看百分比分不出来，得去看掩膜图里灰的那层长什么样 —— 所以话说一半留着。
    cut = []
    if pS[0] < s_lo - 0.5:
        cut.append(f"S 最低 {pS[0]:.0f}（配置 {s_lo}）")
    if pV[0] < v_lo - 0.5:
        cut.append(f"V 最低 {pV[0]:.0f}（配置 {v_lo}）")
    if cut:
        if _loose_contaminated(row):
            out.append("下限那几条**不可信**: 放宽后连进来一大片别的东西"
                       "（不是方块的边），S/V 的「放宽后」是在量那一大片 ——"
                       " 去看掩膜图里灰的那层，灰是一整片桌面就**别降**")
        else:
            out.append(f"下限可能切边: {' / '.join(cut)} —— ★ 别只看这个数，去"
                       f"掩膜图里看**灰的那层**: 灰是贴着方块的一圈（方块的阴影/侧面）"
                       f"→ 该降; 灰是一大片连到桌面/背景 → 那是背景进来了，**别降**，"
                       f"反过来收 H 区间")
    if out and min(pS[0], pV[0]) <= LOOSE_SV + 0.5:
        out.append(f"放宽后 S/V 已经顶在探针底 {LOOSE_SV} 上 —— 方块暗到探针都探不到头。"
                   f"先解决打光/曝光，别一味降阈值（降下去背景也会跟着进来）")
    return out


def print_hsv_probe(rows: list[dict]) -> None:
    print("\n  ── 实测到的 HSV（H 0~179 / S,V 0~255，都是 OpenCV 口径）──")
    for r in rows:
        print(f"\n  {r['color']:<7} 掩膜 {r['px']:>7d}px ({r['pct']:.2f}%)   "
              f"{_verdict(r)}")
        cfg = " / ".join(f"H {lo[0]}-{hi[0]}, S≥{lo[1]}, V≥{lo[2]}"
                         for lo, hi in HSV_RANGES[r["color"]])
        print(f"          现配置: {cfg}")
        if r["blk"] and r["hsv"]:
            b, s, L = r["blk"], r["hsv"], r["loose"]
            print(f"          最大块 {b['area']:.0f}px  外接 {b['long']:.0f}x"
                  f"{b['short']:.0f}px  fill {b['fill']:.2f}  共 {b['n']} 块")
            print(f"          H   p5/p50/p95  现配置 {s['H'][0]:+.0f} / {s['H'][1]:+.0f} / "
                  f"{s['H'][2]:+.0f}      放宽后 {_tri(L, 'H', True)}")
            if _hue_wraps(r["color"]):
                print("               （已把 170~180 折成负数，所以是负的才对）")
            print(f"          S               现配置 {s['S'][0]:.0f} / {s['S'][1]:.0f} / "
                  f"{s['S'][2]:.0f}        放宽后 {_tri(L, 'S')}")
            print(f"          V               现配置 {s['V'][0]:.0f} / {s['V'][1]:.0f} / "
                  f"{s['V'][2]:.0f}        放宽后 {_tri(L, 'V')}")
            if L:
                # ★ 比的是**最大那块**的面积（不是整个掩膜的像素数）: 放宽后本来就
                #   可能多圈进几块别的东西，用总像素数会报出「丢了一大半」的假警报。
                gain = L["blk"]["area"] - b["area"]
                if _loose_contaminated(r):
                    print(f"          放宽后那块涨到 {L['blk']['area']:.0f}px"
                          f"（×{L['blk']['area'] / max(b['area'], 1):.1f}）——"
                          f" **这不是方块的边**，是放宽后连进来的一大片别的东西")
                    print("           → 下面 S/V 那两个「放宽后」的数**是在量那一大片**，"
                          "别照着它调; 去看掩膜图里灰的那层")
                else:
                    print(f"          放宽后那块多 {gain:.0f}px"
                          f"   （放宽 = H 两边各 ±{H_SLACK}、S/V 下限降到 {LOOSE_SV}）")
            for note in _clip_notes(r):
                print(f"          ★ {note}")
    print("\n  ★ 怎么看这几个数（每行都有「现配置」和「放宽后」两列）:")
    print("    · **现配置**那一列不能单独看**: 掩膜本来就是按配置切出来的，区间外的"
          "像素压根不在里面 —— 所以它的 H 永远落在配置区间**内**、S/V 的 p5 永远"
          "≥ 配置下限。光看它，**永远看不出「区间写窄了」**。")
    print(f"    · **放宽后**那一列才是判据: H 往两边各放宽 {H_SLACK}、S/V 下限降到"
          f" {LOOSE_SV} 重新切一遍再量。它才可能跑到配置区间外面去 ——"
          " 跑出去了就是**配置在切边**，该往那边放宽（下面会直接点出来）。")
    print("    · 「放宽后那块多 Npx」: 几乎不涨 = 配置没切到实体（那现配置掩膜里的杂块"
          "就是**真的同色背景**，该反过来**收** H 区间）; 涨一大截 = 在丢东西 ——"
          " **丢的是方块还是背景，去调试图看灰的那层**。")
    print(f"    · 探针也有底: 若放宽后的 p5 又正好压在放宽后的边界上（H 的 ±{H_SLACK}"
          f" 外、S/V 的 {LOOSE_SV}），说明外面还有更极端的 —— 那就不是阈值的事，"
          "先把光打亮，别让方块欠曝或过曝。")
    print("    · 四个颜色都「✅ 采用」才算调好；有一个 ❌ 就先解决它。")
    print("\n  ★★ 但「✅ 采用」在这里是**打了折**的: 画面里没有二维码，就没有"
          " px/mm，于是 detect_cubes 最管用的那道**尺寸门槛（18~45mm）用不上**。")
    print("     所以拍的时候**只把方块放进画面**（别连一大片桌面一起拍）——"
          "「✅」的意思只是「有像样的掩膜」，不等于「认出来的那个就是方块」。")


def draw_mask_montage(frame: np.ndarray) -> np.ndarray:
    """
    上：原图叠轮廓；下：四个掩膜 2x2，**白 = 现配置认到的，灰 = 放宽后才圈进来的**。
    ★ 调阈值就盯这张图。

    ★★ 轮廓必须画在**原分辨率**的图上、最后才缩放。掩膜和轮廓都是 1920x1080 的
       坐标系，直接往缩到 960 宽的画布上画，等于把所有坐标都放大一倍 ——
       1080p 的帧上轮廓会整个挪到右下角、多半直接跑出画面，看着就像「压根没标
       出来」（实测: 方块中心在 (410,510)，线画到了 (410,510) 的 2 倍位置，
       偏 326px）。

    ★★ 掩膜格子必须**保持长宽比**。原先是一行四格、每格 960/4=240 宽 —— 等于把
       1920 宽的整帧压进 240（x 压 8 倍）而 y 只压 2 倍，方块被拉成 14x55 的细长
       条，掩膜到底是什么形状根本看不出来 —— 而「形状」恰恰是这张图唯一要给人看
       的东西。改成 2x2、每格 480x270（16:9 原样），方块在格子里还是个方块。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    W = 960
    PAN_W = W // 2                           # 2x2 拼，每格 480 宽
    fh_native, fw_native = frame.shape[0], frame.shape[1]
    PAN_H = max(1, int(round(PAN_W * fh_native / fw_native)))   # 跟原帧同比例
    fh = max(1, int(round(fh_native * W / fw_native)))
    vis_full = frame.copy()                  # 轮廓先画在这上面（原分辨率坐标系）

    panels = []
    for color in CUBE_COLORS:
        mask = clean_mask(color_mask(hsv, color))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 原图上用该颜色的 HSV 中值描轮廓 —— 描出来的颜色不对，说明区间整体偏了
        lo, hi = HSV_RANGES[color][0]
        rgb = cv2.cvtColor(np.uint8([[[(lo[0] + hi[0]) // 2, 255, 255]]]),
                           cv2.COLOR_HSV2BGR)[0, 0]
        cv2.drawContours(vis_full, cnts, -1, tuple(int(v) for v in rgb), 2)

        # 灰 = 放宽后才圈进来的。★ 这一层是给人**看**的: 那个「放宽后那块多 Npx」
        #   到底是方块自己的边（阴影/侧面，说明该放宽）还是挨着的桌面（说明别动
        #   阈值），数字分不出来，一眼图就分出来了。
        lm = _loose_component(hsv, mask, color)
        extra = cv2.bitwise_and(lm, cv2.bitwise_not(mask))
        # ★ 缩放用 INTER_AREA +「沾到就算」，灰画在白的**上面**。这层灰是唯一的判据
        #   （README: 去灰的那层看是方块的边还是桌面），所以宁可白胖 1px 也要让它清楚。
        #   真帧上量过（1920x1080 → 480x270，缩 4 倍）: 那圈灰在原始分辨率下才一两个
        #   像素宽、还不规则，INTER_NEAREST 是「每 4 个像素挑一个」，会把细边漏掉一截;
        #   INTER_AREA 把格子里的覆盖平均一下，沾到就 >0，留住的灰多 1.3~1.7 倍
        #   （红 298→382px、绿 114→197px）。
        #   ★ 顺序也不能反: 白缩下来自己会往外糊 1px，灰若先画就被白盖掉
        #   （实测红 382→296px，白先画才对）。
        p = np.zeros((PAN_H, PAN_W, 3), np.uint8)
        p[cv2.resize(mask, (PAN_W, PAN_H), interpolation=cv2.INTER_AREA) > 0] = (255, 255, 255)
        p[cv2.resize(extra, (PAN_W, PAN_H), interpolation=cv2.INTER_AREA) > 0] = (128, 128, 128)
        label = (f"{color} {int(cv2.countNonZero(mask))}px"
                 f" +{int(cv2.countNonZero(extra))}")
        cv2.putText(p, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(p, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(p)
    vis = cv2.resize(vis_full, (W, fh), interpolation=cv2.INTER_AREA)
    grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    return np.vstack([vis, grid])


# ═══════════════════════════ 九、离线自检 ═══════════════════════════
def synth_sheet(layout, cubes_mm: dict, ppm: float = 8.0,
                warp: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    造一张「纸面几何严格按 layout」的合成图: 四个二维码 + 四个色块。

    ★ 公开（不再是 _synth_sheet）给 setup_camera.py --selftest 用: 那条链路要造一张
      「真值已知」的图来验证「矩阵 + 方块位置」一次算对。抄一份只会两边走样。

    ★ 为什么不直接拿 data/calib_A4_qr.png 来测: 那张 PNG 是 step1 按模板排的版，
      和 calib_A4_qr.json 的**实测**布局不是同一张纸（码的位置能差 20mm，
      符号尺寸 28.4 vs 26.5mm）。step3 的注释里说得很清楚: 它只是「一张有四个码
      的图」。拿它当基准，测出来的偏差是**数据**的偏差，不是**代码**的偏差 ——
      这种测试只会误导人。所以这里自己按 layout 画一张，真值就来自 layout。

    返回 (图, 期望的像素坐标{颜色: (px,py)}, 期望的像素坐标{码: (px,py)})。
    期望值用**同一个** 像素↔毫米 关系算出来，两边一致 —— 「拟合和检验用同一套，
    绝对尺寸自然抵消」（step3._warp_paper 里那句话）。
    """
    import step1_gen_paper as s1              # 复用它的二维码编码，避免两份实现

    pw, ph = layout.paper_mm
    W, H = int(round(pw * ppm)), int(round(ph * ppm))
    canvas = np.full((H, W, 3), 255, np.uint8)

    # 按 layout 的 corner_tr（数据区右上角）对准放码
    #
    # ★ 每模块像素数**按数据区尺寸反推**，不能拿符号整边长去除。
    #   数据区 21 模块 = layout 量出来的 DATA_SIDE_MM(26.5) 那么大；
    #   要是误按 29 模块去除，码会被画小 ~28%，模块只剩 5px —— 平视图还能
    #   勉强解出来，一加透视就全军覆没（这条自检第一版就是这么挂的）。
    data_mm = layout.qr_side_mm * DATA_MODULES / MODULES_TOTAL     # ≈26.5mm
    mpx_want = max(2, int(round(data_mm * ppm / DATA_MODULES)))    # 每模块像素
    for code in layout.codes:
        mods = s1.encode_qr_modules(code)
        n = mods.shape[0]
        sym, mpx = s1.render_symbol(mods, target_px=mpx_want * n)
        data = n - 2 * s1.QUIET
        sym_px = n * mpx
        # 数据区右上角在符号内部的像素偏移
        off = (s1.QUIET * mpx + data * mpx, s1.QUIET * mpx)
        tx, ty = layout.points_mm["corner_tr"][code]
        ox = int(round(tx * ppm - off[0]))
        oy = int(round(ty * ppm - off[1]))
        if oy < 0 or ox < 0 or oy + sym_px > H or ox + sym_px > W:
            raise AssertionError(f"合成图放不下 {code}（纸 {pw}x{ph}mm）")
        # render_symbol 出的是单通道灰度，画布是 3 通道 —— 转一次就够。
        canvas[oy:oy + sym_px, ox:ox + sym_px] = cv2.cvtColor(
            sym, cv2.COLOR_GRAY2BGR)

    # 色块: 用纯色填充，尺寸按方块边长
    side_mm = 30.0
    BGR = {"red": (0, 0, 255), "yellow": (0, 255, 255),
           "green": (0, 255, 0), "blue": (255, 0, 0)}
    for color, (x, y) in cubes_mm.items():
        h = side_mm / 2 * ppm
        x0, y0 = int(round(x * ppm - h)), int(round(y * ppm - h))
        x1, y1 = int(round(x * ppm + h)), int(round(y * ppm + h))
        canvas[y0:y1, x0:x1] = BGR[color]

    want_px = {c: (x * ppm, y * ppm) for c, (x, y) in cubes_mm.items()}
    want_code_px = {c: (layout.points_mm["corner_tr"][c][0] * ppm,
                        layout.points_mm["corner_tr"][c][1] * ppm)
                    for c in layout.codes}

    if not warp:
        return canvas, want_px, want_code_px

    # ★ 加一层透视: 纯正视图（没有旋转/倾斜）测不出映射写错方向这类问题。
    h, w = canvas.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[70, 55], [w + 55, 20], [w + 20, h + 40], [40, h - 10]])
    W2, H2 = w + 200, h + 150
    Wp = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(canvas, Wp, (W2, H2),
                                 borderValue=(64, 64, 64))

    def proj(p):
        v = np.array([p[0], p[1], 1.0]) @ Wp.T
        return float(v[0] / v[2]), float(v[1] / v[2])

    return (warped, {c: proj(p) for c, p in want_px.items()},
            {c: proj(p) for c, p in want_code_px.items()})


# 合成图上四个色块的位置（纸面 mm）。挑纸中央的空地，避开四个码。
SYNTH_CUBES_MM = {"red": (85.0, 95.0), "yellow": (205.0, 95.0),
                  "green": (205.0, 145.0), "blue": (85.0, 145.0)}


def _selftest_synth(layout, check) -> dict:
    """用合成图测「认得出 + 认得准」。不需要相机、不需要标定。"""
    frame, want_px, want_code_px = synth_sheet(layout, SYNTH_CUBES_MM)

    # 1. 二维码认得出吗（认不出后面全免谈）
    decoded, _ = qr_detect(frame, cv2.QRCodeDetector())
    check("合成图上的四个二维码都能解出",
          all(c in decoded for c in layout.codes),
          f"实解出 {sorted(decoded)}")
    if not all(c in decoded for c in layout.codes):
        return {}

    # 2. 参考点像素位置和期望值对得上（验「检测器的角点约定」没被理解错）
    errs = []
    for c in layout.codes:
        got = qv.reference_point(decoded[c], layout.reference)
        ex, ey = want_code_px[c]
        errs.append(np.hypot(got[0] - ex, got[1] - ey))
    check(f"四码参考点像素位置与期望一致（最大 {max(errs):.2f}px）",
          max(errs) < 3.0)

    # 3. 四个颜色都认出来
    ppm = min(px_per_mm(decoded[c]) for c in layout.codes)
    cubes = detect_cubes(frame, ppm)
    check("四个颜色都认出来了", sorted(cubes) == sorted(CUBE_COLORS),
          f"认到 {sorted(cubes)}；ppm={ppm:.2f}")
    if sorted(cubes) != sorted(CUBE_COLORS):
        return {}

    # 4. 每个色块的中心像素位置对不对
    errs = {c: float(np.hypot(cubes[c]["px"] - want_px[c][0],
                              cubes[c]["py"] - want_px[c][1]))
            for c in CUBE_COLORS}
    check(f"色块中心像素位置准确（最大 {max(errs.values()):.2f}px）",
          max(errs.values()) < 3.0, str({k: round(v, 2) for k, v in errs.items()}))

    # 5. 量出来的边长应该就是 30mm（验 px/mm 换算没搞反）
    sides = {c: cubes[c]["side_mm"] for c in CUBE_COLORS}
    check(f"量出的方块边长≈30mm（{min(sides.values()):.1f}~{max(sides.values()):.1f}）",
          all(27.0 <= v <= 33.0 for v in sides.values()))
    return {"frame": frame, "decoded": decoded, "cubes": cubes, "ppm": ppm}


def _selftest_paper_mm(layout, syn, check) -> None:
    """验证「像素 → 纸面 mm」的换算：把色块还原回纸面，应该回到当初画的位置。"""
    H = px_to_paper_map(syn["decoded"], layout)
    worst, detail = 0.0, {}
    for c in CUBE_COLORS:
        x, y = paper_xy(H, syn["cubes"][c]["px"], syn["cubes"][c]["py"])
        ex, ey = SYNTH_CUBES_MM[c]
        d = float(np.hypot(x - ex, y - ey))
        detail[c] = round(d, 2)
        worst = max(worst, d)
    check(f"色块还原回纸面 mm 准确（最大误差 {worst:.2f}mm）", worst < 1.0,
          str(detail))

    st = build_world_state(syn["cubes"], "paper", syn["decoded"], layout, prior={})
    # ★ 每条都得带「相机给的」戳（WORLD_SRC_KEY）—— 主程序靠它决定第一次抓取要不要
    #   补偿 5mm（main.FIRST_GRASP_OFFSET_MM）。戳丢了不会报错，只会让补偿**永远不生效**
    #   （用户看到的现象是"改了跟没改一样"），所以形状自检必须把它一起钉住。
    check("paper 模式产出的 world_state 形状对"
          "（四个颜色 + x/y/z_level + 来源戳 + 半侧戳）",
          sorted(st) == sorted(CUBE_COLORS)
          and all(set(v) == {"x", "y", "z_level", WORLD_SRC_KEY, WORLD_HALF_KEY}
                  for v in st.values())
          and all(v[WORLD_SRC_KEY] == WORLD_SRC_CAMERA for v in st.values()))

    # ★ 半侧戳必须按**纸面 x** 判，左右别搞反: 合成图里 red/blue 画在 x=85（<148.5，
    #   左），yellow/green 在 x=205（右）。判反了不会报错，只会让 5mm 补错半边。
    halfs = {c: st[c].get(WORLD_HALF_KEY) for c in CUBE_COLORS}
    check("半侧戳按纸面 x 判对（红/蓝在左，黄/绿在右）",
          halfs == {"red": WORLD_HALF_LEFT, "blue": WORLD_HALF_LEFT,
                    "yellow": WORLD_HALF_RIGHT, "green": WORLD_HALF_RIGHT},
          str(halfs))

    # ★ 判不出来时**不许写 half**（少一个二维码 → 拟合不出像素↔纸面）。
    #   ★ 这条是「宁可少补 5mm，也不赌错方向」的守门测试: robot 模式少了码，
    #     整帧照常出坐标（源戳在），但半侧戳必须缺席。
    partial = {c: v for c, v in syn["decoded"].items() if c != layout.codes[0]}
    st2 = build_world_state(syn["cubes"], "robot", partial, layout,
                            robot=(np.eye(3, dtype=np.float64),
                                   lambda m, px, py: (px, py)), prior={})
    check("少一个二维码 → 照样出坐标，但半侧戳缺席（判不出来就不猜）",
          all(WORLD_SRC_KEY in v for v in st2.values())
          and all(WORLD_HALF_KEY not in v for v in st2.values()),
          str({c: sorted(v) for c, v in st2.items()}))


def _selftest_perspective(layout, check) -> None:
    """把纸摆成斜的/转个角度，结果不该变 —— 不然就是映射里混进了「正对着拍」的假设。"""
    frame, want_px, _ = synth_sheet(layout, SYNTH_CUBES_MM, warp=True)
    decoded, _ = qr_detect(frame, cv2.QRCodeDetector())
    if not all(c in decoded for c in layout.codes):
        check("斜视图下四个码仍能解出", False)
        return
    ppm = min(px_per_mm(decoded[c]) for c in layout.codes)
    cubes = detect_cubes(frame, ppm)
    if sorted(cubes) != sorted(CUBE_COLORS):
        check("斜视图下四个颜色仍认得出", False, f"认到 {sorted(cubes)}")
        return
    H = px_to_paper_map(decoded, layout)
    worst = 0.0
    for c in CUBE_COLORS:
        x, y = paper_xy(H, cubes[c]["px"], cubes[c]["py"])
        ex, ey = SYNTH_CUBES_MM[c]
        worst = max(worst, float(np.hypot(x - ex, y - ey)))
    check(f"斜视图（透视+旋转）下纸面坐标依然准（最大误差 {worst:.2f}mm）",
          worst < 2.0)


def _selftest_zlevel(layout, syn, check) -> None:
    """
    z_level 必须沿用记忆库 —— 相机看不见层数，不能因为"看不见"就清零。
    ★ 注意和 reset_levels_if_flat 的区别: 那条是**这一帧证明了平放**才清，
      不是"看不见所以清"。这里的 build_world_state 本身照抄不误。
    """
    prior = {"red": 2, "blue": 1}
    st = build_world_state(syn["cubes"], "paper", syn["decoded"], layout, prior=prior)
    check("z_level 从记忆库沿用（red=2 / blue=1，其余 0）",
          st["red"]["z_level"] == 2 and st["blue"]["z_level"] == 1
          and st["green"]["z_level"] == 0 and st["yellow"]["z_level"] == 0)

    no_prior = build_world_state(syn["cubes"], "paper", syn["decoded"], layout, prior={})
    check("记忆库空时 z_level 一律为 0（假设都在地上）",
          all(v["z_level"] == 0 for v in no_prior.values()))


def _selftest_flat_reset(check) -> None:
    """
    ★ 手工拆开重摆之后，记忆库里过期的层数必须被清掉（2026-09-18 真事）。
    同时钉住「不许误伤」: 靠得近就不许清、缺一块就不许下结论。
    """
    def mk(blue_xy, prior=None):
        prior = prior or {}
        pts = {"red": (0.0, 0.0), "yellow": (200.0, 0.0),
               "green": (0.0, 200.0), "blue": blue_xy}
        return {c: {"x": float(x), "y": float(y), "z_level": int(prior.get(c, 0))}
                for c, (x, y) in pts.items()}

    st = mk((200.0, 60.0), {"green": 3, "blue": 1})
    note = reset_levels_if_flat(st)
    check("手工重摆后过期的层数被清掉（四块都看得见且隔得开）",
          note is not None and all(v["z_level"] == 0 for v in st.values()),
          note or "（没清）")
    check("说明里点名了原来记的是第几层",
          note is not None and "green" in note and "3" in note)

    st = mk((0.0, 30.0), {"green": 2})
    note = reset_levels_if_flat(st)
    check("方块紧挨着放（中心距正好 30mm = 一个边长）仍算平放，照样清",
          note is not None and st["green"]["z_level"] == 0)

    st = mk((0.0, 20.0), {"green": 3})
    note = reset_levels_if_flat(st)
    check("两块靠得比门槛近 → 一个层数都不动（可能真摞着）",
          note is None and st["green"]["z_level"] == 3)

    st = mk((200.0, 60.0))
    check("层数本来就是 0 → 不吭声（不刷无意义的提示）",
          reset_levels_if_flat(st) is None)

    st = mk((200.0, 60.0), {"green": 3})
    del st["yellow"]
    check("少一块（多半正被压在下面）→ 不下结论，层数保持原样",
          reset_levels_if_flat(st) is None and st["green"]["z_level"] == 3)


def _selftest_red_wrap(check) -> None:
    """
    ★ 红色跨 0/180 接缝 —— 这是纯色识别最经典的坑，必须有专门一条钉住。

    构造两个只差色相的红: H=5（偏橙）和 H=175（偏紫）。只写单边区间的实现
    必然漏掉其中一个，而漏掉的那个在实物上就是「一块红方块认不出来」。
    """
    hsv = np.zeros((1, 2, 3), np.uint8)
    hsv[0, 0] = (5, 255, 255)          # 偏橙的红
    hsv[0, 1] = (175, 255, 255)        # 偏紫的红
    m = color_mask(hsv, "red")
    check("红色两条 H 区间都生效（偏橙 H=5 和偏紫 H=175 都算红）",
          m[0, 0] == 255 and m[0, 1] == 255,
          f"H=5→{m[0, 0]}, H=175→{m[0, 1]}")
    # 反过来: 不能把别的颜色也算成红，否则掩膜会糊成一片
    hsv2 = np.zeros((1, 4, 3), np.uint8)
    for i, h in enumerate((30, 60, 120, 90)):       # 黄 绿 蓝 + 青
        hsv2[0, i] = (h, 255, 255)
    m2 = color_mask(hsv2, "red")
    check("黄/绿/蓝/青色相都不算红（区间没有张得太大）",
          int(m2.sum()) == 0, f"误判像素 {int((m2 > 0).sum())}")


def _selftest_hsv_probe(layout, check) -> None:
    """
    --hsv 这条通路: 只用颜色，**不用二维码**。

    ★ 为什么专门钉「不用二维码」: 这条命令的价值全在这一点上 —— 标定纸还没
      做好时它是唯一能干的识别活儿。哪天有人顺手在里面加一句「先解四个码」，
      功能就悄悄没了，而在**合成图上根本看不出来**（合成图里码是齐的）。
      所以这里喂一张明确**没有码**的图。
    """
    BGR = {"red": (0, 0, 255), "yellow": (0, 255, 255),
           "green": (0, 255, 0), "blue": (255, 0, 0)}
    img = np.full((360, 480, 3), 255, np.uint8)          # 白底，四色方块
    for i, (color, bgr) in enumerate(BGR.items()):
        y0, x0 = 20 + (i // 2) * 170, 30 + (i % 2) * 240
        img[y0:y0 + 90, x0:x0 + 150] = bgr

    dec, _ = qr_detect(img, cv2.QRCodeDetector())
    check("测试图里一个二维码都没有（这条通路本来就不该要码）", not dec, f"解出 {dec}")

    rows = {r["color"]: r for r in hsv_probe(img)}
    check("四色全部被采用（--hsv 的结论和真正筛选走的是同一套判据，不是另写一套）",
          all(rows[c]["adopted"] for c in CUBE_COLORS),
          " ".join(f"{c}:{'✅' if rows[c]['adopted'] else '❌'}" for c in CUBE_COLORS))

    # ★ 报出来的必须是**实测**到的窄带，而不是把配置区间原样抄一遍 ——
    #   抄一遍的话数字永远「看起来对」，一点诊断价值都没有。
    narrow = {}
    for c in ("yellow", "green", "blue"):
        p5, p95 = rows[c]["hsv"]["H"][0], rows[c]["hsv"]["H"][2]
        narrow[c] = float(p95 - p5)
        cfg = HSV_RANGES[c][0]
        narrow[c] = (narrow[c], float(cfg[1][0] - cfg[0][0]))
    check("报的 H 是实测的窄带，不是把配置区间抄回来",
          all(got < cfg / 2 for got, cfg in narrow.values()),
          " ".join(f"{c}: 实测带宽 {g:.0f} vs 配置 {w:.0f}" for c, (g, w) in narrow.items()))

    # 红色的折算: 造一块「偏橙 + 偏紫」连成一体的红 —— 只用一个纯色测不出接缝
    hsv = np.zeros((120, 240, 3), np.uint8)
    hsv[:, :120], hsv[:, 120:] = (5, 255, 255), (175, 255, 255)
    red_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    m = clean_mask(color_mask(cv2.cvtColor(red_img, cv2.COLOR_BGR2HSV), "red"))
    hh = cv2.cvtColor(red_img, cv2.COLOR_BGR2HSV)[:, :, 0][m > 0].astype(np.int32)
    raw = float(np.percentile(hh, 95) - np.percentile(hh, 5))
    got = {r["color"]: r for r in hsv_probe(red_img)}["red"]["hsv"]["H"]
    check("红色: 这块红确实横跨 0/180 两个瓣（原始 H 的 p5≈5、p95≈175）",
          raw > 100, f"原始跨度 {raw:.0f}")
    check("红色: 折算后是一条窄带（不折的话百分位数会落在两头之间，看着像乱码）",
          float(got[2] - got[0]) < 40, f"折算后 p5/p95 = {got[0]:+.0f}/{got[2]:+.0f}")

    # ★★ 这三条钉的是**最容易骗人**的地方: 「现配置」那一列是循环论证 —— 掩膜按
    #    配置切出来，区间外的像素不在里面，所以它的 p5 永远 ≥ 配置下限，永远说不出
    #    「区间写窄了」。造一块 H 铺满 14~35 的黄（配置写 20-35）: 现配置那列必然
    #    报「p5 = +20」压在下界上、还显示「✅ 采用」，方块左边那一截一直在丢。
    edge_hsv = np.zeros((120, 220, 3), np.uint8)
    for i, H in enumerate(range(14, 36)):
        edge_hsv[:, i * 10:(i + 1) * 10] = (H, 255, 255)
    er = {r["color"]: r for r in hsv_probe(cv2.cvtColor(edge_hsv, cv2.COLOR_HSV2BGR))}
    er = er["yellow"]
    lo_cfg = HSV_RANGES["yellow"][0][0][0]                  # = 20
    check("★ 现配置那列**量不出**「区间写窄了」（它的 p5 被自己的下界钉住）",
          abs(er["hsv"]["H"][0] - lo_cfg) < 1.5 and er["adopted"],
          f"现配置 p5 = {er['hsv']['H'][0]:+.0f}（下界 {lo_cfg}），采用={er['adopted']}")
    check("★ 放宽后那列看得出来: 方块其实从 14 就开始，配置 20 在切边",
          er["loose"]["hsv"]["H"][0] < lo_cfg - 1.5 and er["loose"]["hsv"]["H"][2] > 30,
          f"放宽后 H = {er['loose']['hsv']['H'][0]:+.0f}"
          f" ~ {er['loose']['hsv']['H'][2]:+.0f}")
    check("★ 切边会被**直接点出来**（不是让人自己盯着两列数字找）",
          any("在切边" in n for n in _clip_notes(er)),
          " / ".join(_clip_notes(er)) or "一条提示都没有")

    # ★ 放宽 H 会把邻色的东西也圈进来（green 放宽到 30-95 就吃掉了 H≈32 的橙黄）。
    #   造一块**比绿方块还大**的橙黄放在远处: 不按连通域筛的话，「放宽后那块」
    #   就变成别人了 —— 面积和 H 全是错的，会报出「多了一大块」的假警报。
    n_img = img.copy()
    n_img[290:355, 20:400] = cv2.cvtColor(np.uint8([[[32, 255, 255]]]),
                                          cv2.COLOR_HSV2BGR)[0, 0]   # 橙黄, 比绿方块大
    g = {r["color"]: r for r in hsv_probe(n_img)}["green"]
    check("★ 放宽后只认跟现配置那块连长在一起的那一块（别块再大也不吃进来）",
          abs(g["loose"]["blk"]["area"] - g["blk"]["area"]) < 1
          and abs(g["loose"]["hsv"]["H"][1] - g["hsv"]["H"][1]) < 2,
          f"green 放宽后那块 {g['loose']['blk']['area']:.0f}px / H中值 "
          f"{g['loose']['hsv']['H'][1]:+.0f}  vs  现配置 {g['blk']['area']:.0f}px / "
          f"H中值 {g['hsv']['H'][1]:+.0f}")

    # ★ 打印也是代码: 上面那个 loose 字典的嵌套就写错过一次（KeyError），而
    #   --selftest 走的是 hsv_probe、**不走 print** —— 不在这里跑一遍，
    #   就会出现「自检全绿、--hsv 一敲就崩」。
    import contextlib
    import io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            print_hsv_probe(hsv_probe(n_img))
        ok, msg = True, f"输出 {len(buf.getvalue().splitlines())} 行"
    except Exception as e:                               # noqa: BLE001
        ok, msg = False, f"{type(e).__name__}: {e}"
    check("整段打印不炸（打印代码也算在这条自检里）", ok, msg)

    # ★★ 没有 px/mm 时也必须是「取面积最大的那块」。这条专钉一个曾经的真 bug:
    #    sort key 写成了 len(good)（第几个找到的），大块在后时会被小块顶掉，
    #    却仍打印「取最大的那块」。下面故意让 findContours 先返回小块。
    two = np.full((300, 300, 3), 255, np.uint8)
    two[20:60, 20:60] = (0, 0, 255)          # 小红 ≈40x40
    two[150:290, 150:290] = (0, 0, 255)      # 大红 ≈140x140
    red2 = detect_cubes(two, None)["red"]
    check("★ 没有 px/mm 时也取面积最大的那块（不被 findContours 顺序左右）",
          red2["px"] > 150 and red2["contour_px"] > 10000,
          f"挑中中心 ({red2['px']:.0f}, {red2['py']:.0f})，"
          f"面积 {red2['contour_px']:.0f}px")

    # ★ 放宽 S/V 会把**挨着的低饱和背景**一起连进来 —— 实战里方块就压在桌面上，
    #   不拦住的话会报「放宽后那块多 945700px」这种荒唐数（实测: 整个米色桌面都成了
    #   同一块），而且 S/V 的「放宽后」p5 全是在量桌面，照着调正好把背景收进来。
    dsk = np.full((1080, 1920, 3), 255, np.uint8)
    dsk[500:1000, :] = cv2.cvtColor(np.uint8([[[22, 55, 150]]]),
                                    cv2.COLOR_HSV2BGR)[0, 0]        # 桌面: H≈22 S≈55
    dsk[560:670, 300:410] = (0, 255, 255)                       # 黄方块压在桌面上
    dy = {r["color"]: r for r in hsv_probe(dsk)}["yellow"]
    check("★ 放宽后整个桌面连成一坨时会被拦下（不当成「方块的边」去报大数）",
          _loose_contaminated(dy)
          and any("不可信" in n for n in _clip_notes(dy)),
          f"放宽后那块 {dy['loose']['blk']['area']:.0f}px vs 现配置 {dy['blk']['area']:.0f}px")

    mont = draw_mask_montage(img)
    vh = 360 * 960 // 480                      # 上半（原图）高
    ph = 480 * 360 // 480                      # 下半每格高 —— 2x2 就是两格
    check("掩膜图拼得出来（上半原图 + 下半 2x2 四个掩膜，宽度对齐）",
          mont.shape == (vh + 2 * ph, 960, 3), f"{mont.shape}")

    # ★★ 轮廓的**坐标系**必须对得上。掩膜是原分辨率的，画布是缩到 960 宽的 ——
    #    直接往缩过的画布上画，坐标就等于被放大了一倍：1080p 的帧上轮廓会整个挪到
    #    右下角、多半直接出画面，看着就像「压根没标出来」（用户就是这么撞上的）。
    #    ★ 为什么非要用 1080p 来测: 合成图是 480 宽（要**放大**到 960），放大时
    #      画错坐标反而缩在方块内部、看着还挺像；只有缩小的方向才会甩出画面。
    big = np.full((1080, 1920, 3), 255, np.uint8)
    big[300:520, 400:620] = (0, 0, 255)                  # 红方块，中心 (410, 510)
    mb = draw_mask_montage(big)
    fhb = int(round(1080 * 960 / 1920))
    ref = cv2.resize(big, (960, fhb), interpolation=cv2.INTER_AREA)
    drew = np.abs(mb[:fhb].astype(int) - ref.astype(int)).sum(axis=2) > 30
    ys, xs = np.nonzero(drew)
    off = float("nan")
    if len(ys):
        off = float(np.hypot(ys.mean() - 410 * fhb / 1080, xs.mean() - 510 * 0.5))
    check("★ 1080p 下轮廓画在方块上（不是把原图坐标直接画到缩过的画布上）",
          len(ys) > 0 and off < 30,
          f"描出来的线中心离方块 {off:.0f}px（画错坐标系会差 300px 以上）" if len(ys)
          else "一条线都没画出来")

    # ★★ 掩膜格子必须**没被拉变形**。以前是一行四格（每格 240 宽）: 1920x1080 的帧
    #    被压进 240x540 —— x 压 8 倍、y 压 2 倍，上面那个 220x220 的方块在格子里成了
    #    27x110 的细条，掩膜是什么形状根本看不出来。改成 2x2（每格 480x270）之后，
    #    方块在格子里还是 55x55 的方块 —— 这条就是钉这个的。
    pan_h = 480 * 1080 // 1920                                   # 270
    red_pan = mb[fhb:min(fhb + pan_h, mb.shape[0]), :480].copy()
    red_pan[:60] = 0                     # 顶上那行标签也是白的，别算进来
    wys, wxs = np.nonzero(np.all(red_pan > 200, axis=2))
    blobs = f"{wxs.max() - wxs.min() + 1}x{wys.max() - wys.min() + 1}px" if len(wys) else ""
    ratio = ((wxs.max() - wxs.min() + 1) / (wys.max() - wys.min() + 1)) if len(wys) else 0.0
    check("★ 掩膜格子没把方块拉变形（压扁的老版会变成细条）",
          len(wys) > 0 and 0.85 < ratio < 1.18,
          f"红方块的格子里那块 {blobs}，长宽比 {ratio:.2f}"
          f"（一行四格的老版是 0.25）" if len(wys) else "红的格子里没有白块")


def _selftest_robot(layout, syn, check) -> None:
    """
    robot 模式的接线对不对。★ 没有真矩阵，就造一个**已知的**矩阵:
    平移 + 2 倍缩放。这样「代码接错线」会立刻暴露，而它不需要真标定。

    ★ 关键: 要比的是 **build_world_state 吐出来的 x/y**，不是直接调
      pixel_to_robot —— 直接调只能证明 step3 那个函数没错，证明不了本文件
      把它接对了（比如误传了纸面 mm 进去）。所以下面还专门拿一个「故意接错」
      的算法做对照: 错的那个必须算出**不一样**的值，否则这条自检就是空转。
    """
    from step3_hand_eye_calib import pixel_to_robot
    M = np.array([[2.0, 0.0, 100.0],
                  [0.0, 2.0, -50.0],
                  [0.0, 0.0, 1.0]])
    st = build_world_state(syn["cubes"], "robot", syn["decoded"], layout,
                           robot=(M, pixel_to_robot), prior={"green": 1})

    bad_detail, ok, wrong_would_differ = {}, True, True
    for c in CUBE_COLORS:
        px, py = syn["cubes"][c]["px"], syn["cubes"][c]["py"]
        ex, ey = 2 * px + 100, 2 * py - 50
        dx = st[c]["x"] - ex
        dy = st[c]["y"] - ey
        bad_detail[c] = (round(dx, 3), round(dy, 3))     # 单位 mm
        # round(...,2) 是 build_world_state 干的，容差取 0.006 而不是 1e-6
        ok &= abs(dx) < 0.006 and abs(dy) < 0.006
        # 纸面 mm 大约是几十；喂给这个矩阵会得到几百 —— 差得远，能区分
        fx, fy = paper_xy(px_to_paper_map(syn["decoded"], layout), px, py)
        wrong_would_differ &= np.hypot(st[c]["x"] - (2 * fx + 100),
                                       st[c]["y"] - (2 * fy - 50)) > 1.0
    check("robot 模式下 world_state 的 x/y 就是矩阵算出来的（误差 mm 级）",
          ok, f"偏差(mm) {bad_detail}")
    check("（对照）喂纸面坐标会算出完全不同的值 —— 说明上一条不是空转",
          wrong_would_differ)
    check("robot 模式下 z_level 照样沿用记忆库", st["green"]["z_level"] == 1)


def _selftest_failures(layout, check) -> None:
    """凑不齐就该拒绝，不许硬着头皮给个错答案。"""
    frame, _, _ = synth_sheet(layout, SYNTH_CUBES_MM)
    decoded, _ = qr_detect(frame, cv2.QRCodeDetector())
    ppm = min(px_per_mm(decoded[c]) for c in layout.codes)
    cubes = detect_cubes(frame, ppm)

    # 少一个方块 → 必须报错
    few = {k: v for k, v in cubes.items() if k != "green"}
    try:
        build_world_state(few, "paper", decoded, layout, prior={})
        check("少一个方块时拒绝出结果", False, "它居然给了答案")
    except VisionError as e:
        check("少一个方块时拒绝出结果，并说清怎么办",
              "green" in str(e) and "HSV" in str(e))

    # 二维码不全 → 纸面换算必须报错
    half = {c: decoded[c] for c in list(decoded)[:2]}
    try:
        px_to_paper_map(half, layout)
        check("二维码不全时拒绝算像素↔毫米", False, "它居然算了")
    except VisionError as e:
        check("二维码不全时拒绝算像素↔毫米，并指出缺哪个码",
              "缺" in str(e))

    # 没有标定矩阵时的报错要指路，不能只抛 FileNotFoundError
    from paths import MATRIX_JSON
    if not MATRIX_JSON.exists():
        try:
            load_robot_matrix()
            check("没标定时...", False)
        except VisionError as e:
            check("没标定时明确拒绝，并给出 --paper-mm 这条退路",
                  "--paper-mm" in str(e) and "step3" in str(e))
    else:
        check("（本机已有标定矩阵，跳过「没标定」这一条）", True)

    # 靠太近要提醒
    close = {c: dict(v) for c, v in cubes.items()}
    close["green"] = dict(close["green"])
    close["green"]["px"] = close["blue"]["px"] + 3
    close["green"]["py"] = close["blue"]["py"] + 3
    st = build_world_state(close, "paper", decoded, layout, prior={})
    probs = sanity_check(st, "paper", layout)
    check("两个方块挨太近会提醒（可能是有一块被挡住了）",
          any("靠太近" in p for p in probs), str(probs[:1]))


def selftest() -> int:
    print("=" * 72)
    print("  颜色识别离线自检（合成图，不用相机、不用标定、不连机械臂）")
    print("=" * 72)
    layout = load_paper_layout(PAPER_JSON)
    ok_all = True

    def check(name, cond, extra=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  {'✅' if cond else '❌'} {name}" + (f"   {extra}" if extra else ""))

    print(f"\n[0] 标定纸: {layout.paper_mm[0]:.0f}x{layout.paper_mm[1]:.0f}mm，"
          f"{len(layout.codes)} 个码，参考点={layout.reference}")

    print("\n[1] 合成图: 认得出 + 认得准")
    syn = _selftest_synth(layout, check)
    if not syn:
        print("\n  ❌ 合成图这一步没过，后面的都不用跑了")
        return 1

    print("\n[2] 像素 → 纸面毫米")
    _selftest_paper_mm(layout, syn, check)

    print("\n[3] 斜视图（透视 + 旋转）")
    _selftest_perspective(layout, check)

    print("\n[4] z_level 归记忆库管")
    _selftest_zlevel(layout, syn, check)

    print("\n[5] 红色的 0/180 接缝")
    _selftest_red_wrap(check)

    print("\n[6] robot 模式接线（用造出来的已知矩阵）")
    _selftest_robot(layout, syn, check)

    print("\n[7] 拿不到数据时要拒绝，不许猜")
    _selftest_failures(layout, check)

    # 调试图能出（顺带验证画图代码没写崩）
    print("\n[8] 调试图")
    dbg = draw_debug(syn["frame"], syn["cubes"], None, layout)
    check("调试图画得出来（尺寸不变）", dbg.shape == syn["frame"].shape)

    print("\n[9] --hsv 实测（没有二维码那条通路）")
    _selftest_hsv_probe(layout, check)

    print("\n[10] 手工重摆过 → 过期的层数要清掉")
    _selftest_flat_reset(check)

    print()
    print("=" * 72)
    if ok_all:
        print("  ✅ 自检全部通过")
        print("  下一步（不用相机）: python3 src/color_vision.py --image 照片.jpg --paper-mm")
        print("  没有标定纸也能干:   python3 src/color_vision.py --camera --hsv")
    else:
        print("  ❌ 有项目没通过")
    print("=" * 72)
    return 0 if ok_all else 1


# ═══════════════════════════ 十、命令行 ═══════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(
        description="认颜色 → 方块坐标（world_state）",
        epilog="完全不需要标定纸的用法: --hsv（只看颜色，调阈值）")
    ap.add_argument("--selftest", action="store_true", help="离线自检（合成图）")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--image", default=None, help="从图片读（手机拍的也行）")
    src.add_argument("--camera", action="store_true", help="从摄像头抓一帧")
    src.add_argument("--cam", type=int, default=None, help="指定 /dev/videoN")
    ap.add_argument("--hsv", action="store_true",
                    help="★ 只实测 HSV、不换算坐标 —— 不用二维码/标定/机械臂（调阈值用）")
    ap.add_argument("--paper-mm", action="store_true",
                    help="输出纸面 mm，不需要标定矩阵，但**要画面里有四个码**")
    ap.add_argument("--out", default=None, help=f"world_state 存哪（默认 {WORLD_STATE_JSON.name}）")
    ap.add_argument("--debug", action="store_true", help=f"出调试图 {COLOR_DEBUG_PNG.name}")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    # ★ --hsv 走最前面: 它连 data/calib_A4_qr.json 都不读。
    #   调阈值这件事和「纸做好没有」本来就没关系，不该被它挡住。
    if args.hsv:
        try:
            frame = (frame_from_file(args.image) if args.image
                     else frame_from_camera(args.cam))
        except VisionError as e:
            print(f"\n✗ {e}")
            return 1
        print(f"[输入] {'图片 ' + args.image if args.image else '摄像头'}  "
              f"{frame.shape[1]}x{frame.shape[0]}")
        print_hsv_probe(hsv_probe(frame))
        ensure_output_dir()
        cv2.imwrite(str(COLOR_DEBUG_PNG), draw_mask_montage(frame))
        print(f"\n[调试图] {COLOR_DEBUG_PNG}")
        print(f"  上半 = 原图 + 四色轮廓 —— 框**该正好套在方块上**；框跑偏了就是阈值"
              f"不对（{frame.shape[1]}x{frame.shape[0]} 已缩到 960 宽）。")
        print("  下半 = 2x2 四个掩膜: 白 = 现配置认到的，灰 = 放宽后才圈进来的。"
              "缺一块就放宽下限；糊成一片就收紧。")
        print("        ★ 灰要是**一整片连到桌面**，那说明放宽收进来的是背景 ——"
              " 别跟着降阈值，反过来收 H 区间。")
        print("  ★ 这一步**没有**动 output/world_state.json（没有二维码就算不出坐标）")
        return 0

    layout = load_paper_layout(PAPER_JSON)
    mode = "paper" if args.paper_mm else "robot"

    try:
        if args.image:
            frame = frame_from_file(args.image)
            print(f"[输入] 图片 {args.image}  {frame.shape[1]}x{frame.shape[0]}")
        else:
            which = args.cam
            frame = frame_from_camera(which)
            print(f"[输入] 摄像头  {frame.shape[1]}x{frame.shape[0]}")
        state, cubes, ppm, decoded = analyze(frame, mode, layout)
    except VisionError as e:
        print(f"\n✗ {e}")
        return 1

    print_table(state, mode)
    if ppm:
        print(f"\n  px/mm = {ppm:.2f}（标定时应该是同一个量级；小很多说明机位挪近了）")

    probs = sanity_check(state, mode, layout)
    if probs:
        print("\n  ⚠ 看出这些毛病:")
        for p in probs:
            print(f"    · {p}")

    if mode == "robot":
        # ★ 走 save_world_state，别在这里自己 write_text —— 那样会**绕过 .bak 备份**，
        #   一次误跑就把记忆库覆盖掉了（相机只能给 x/y，z_level 补不回来）。
        out = save_world_state(state, path=Path(args.out) if args.out else None)
        print(f"\n[落盘] {out}")
        print(f"  下一步: python3 src/main.py \"把绿色方块放到红色方块右侧\" --world-json {out}")
        print("  ★ z_level 相机看不出来，是从记忆库继承的 —— "
              "这一帧要是证明了桌面平放，上面会有一行「层数一律按 0 算」")
    else:
        print("\n  （--paper-mm 只报纸面坐标，没动 output/world_state.json）")
        print("  ★ 这一步不需要标定矩阵；要出机械臂坐标去掉 --paper-mm")

    if args.debug:
        ensure_output_dir()
        dbg = draw_debug(frame, cubes, state, layout)
        cv2.imwrite(str(COLOR_DEBUG_PNG), dbg)
        print(f"[调试图] {COLOR_DEBUG_PNG}  ← 框没套准就调 HSV_RANGES")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断] 用户按了 Ctrl-C。")
        sys.exit(130)
