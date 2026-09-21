#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_suction_map.py —— 量出「机械臂走 1mm」↔「画面动几像素」（① 方向 + 比例）
=============================================================================
★ 先读这段，别把它当成标定脚本 —— 本项目现在有**两条路**，方向完全相反:

  · 手眼标定那条（step2 对刀 → step3 出 hand_eye_matrix.json → main.py）
    靠「纸面 mm ↔ 机械臂绝对坐标」的映射，前提是**工作台和机械臂的相对位置
    固定**。桌子一挪、机位一变，整条链子静默作废。

  · 相对运动那条（本脚本 + 吸盘上贴的第 5 个码）
    不要求任何绝对位置。每次只看「吸盘码此刻在画面哪儿」「方块在画面哪儿」，
    算出该走多少，走完再拍一张看还差多少。机械臂的绝对精度和桌子摆在哪都
    不再重要 —— 这正是当初加吸盘码的目的。

  本脚本是第二条路的**第一块地基**: 把「机械臂的 mm」和「画面的像素」对上。
  它**不读也不写** hand_eye_matrix.json，不需要 step2 对刀过。

★ 为什么只拍一张算不出来:
  「吸盘码在画面 (1061, 549)」本身毫无用处 —— 那只是个像素坐标。要知道
  「机械臂 +X 走 30mm 时吸盘码在画面里往哪边、动多少像素」只能**真的走一下**。
  所以 `--go` 会空走几步: 只有 XY、不下降、不吸、不碰任何东西，走完回原位。

★ 为什么还要解出 k（视差放大倍数）——这是本脚本真正值钱的地方:
  吸盘码比吸嘴**高 h=120mm**（用户实测）。相机俯视时高处的东西会被**朝外推**，
  离光心越远推得越多。于是「把吸盘码中心压在方块上」**不等于**「吸嘴在方块上」，
  吸嘴会朝画面中心偏；在纸边能偏到几十毫米 —— 这比标定里那些几毫米的残差
  大一个量级，是这条路上**最大的误差源**。

  这个放大倍数是 k = Z / (Z − h)，Z = 相机离纸面的高度。
  ★ k 不用卷尺量: 桌面 4 码都贴着纸面（h≈0），拿它们的 px/mm 当尺子；
    吸盘码在画面里「动得比机械臂快多少倍」就是 k。所以走一步就解出来了:
        |A| = s_table × k        （A = 实测的 px/机械臂mm）
    一旦知道 k，就知道吸嘴的真实落点，也顺手知道相机架多高（Z = h·k/(k−1)）。

★ 输出的数怎么用（② 也做完就能闭环）:
    δ_机械臂 = A⁻¹ · [ c·(方块像素 − 光心) − (吸盘码像素 − 光心) ]
    走 → 再拍 → 再看差多少 → 再走。**永远不需要绝对坐标。**
  ★★ 系数是 c = k(吸盘码那一层) ÷ k(方块顶面那一层) —— **不是**吸嘴处的 k
    （2026-09-20 重推，旧版写的是 k_tip，差一半）。理由: A 量的是**码在画面里
    动多快**，所以 A⁻¹ 把像素换成 mm 的因子天生带 k(z_qr)。详细推导见
    print_control 的注释。
  ★ ① 只能定 A 和 k；式子里的**光心 C** 还是个「画面中心」的假设 ——
    所以再来一步 ②（`--go-center`）把它实测出来。那一步**只动 Z、不横移**，
    让吸盘码在几个高度上各拍一次: 码在画面里会沿一条**过光心的射线**滑动，
    滑多远由 k 决定 → 一条直线拟合就把 C 解出来了（见 fit_optical_center）。
  ★ ① 解出来的 k 里混着**机械臂 XY 的刻度**（这台机器报的 XY 不是真毫米），
    实测把 k 顶高了 10%。拿卷尺量一次相机高度、加 `--cam-height` 就以真毫米
    为准，那个比值顺手变成「机械臂刻度偏差」的读数。
  ★ ② 做完**也还不能**一路用到底: c 里的 k 要按**当时的 z** 现算
    （抓的时候吸嘴降到方块顶面，和量的时候不是一个高度）。本脚本只**报数**，
    不驱动任何东西。

用法:
  python3 tools/measure_suction_map.py                    # 只拍一张，报数（不动臂）
  python3 tools/measure_suction_map.py --image a.jpg      # 用已有图片
  python3 tools/measure_suction_map.py --go               # ★ ① 真动: 空走几步量映射
  python3 tools/measure_suction_map.py --go --cam-height 323   # 卷尺量了就用真毫米当基准
  python3 tools/measure_suction_map.py --go-center        # ★ ② 真动: 只升降，实测光心
  python3 tools/measure_suction_map.py --selftest         # 离线自检（不连相机不连臂）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))          # 公共库在 src/

import qr_vision as qv                                                      # noqa: E402
from dobot_sdk import find_port, load_sdk                                   # noqa: E402
from paths import (PAPER_JSON, PICK_RESULT_JSON, ROBOT_POINTS_JSON,         # noqa: E402
                   SUCTION_MAP_JSON, ensure_output_dir)
from qr_vision import (detect_sweep, load_paper_layout, load_suction_code,   # noqa: E402
                       quad_center, quad_side_px, suction_size_candidates,
                       wait_for_focus)

# ═════════════════════════════ 可调参数 ═════════════════════════════
DEFAULT_STEP_MM = 30.0        # 每个方向空走多远。太小 → 像素动得少、噪声占比大
DEFAULT_QR_HEIGHT_MM = 120.0  # 吸盘码中心到吸嘴尖的高度（用户卷尺量的 12cm）
DEFAULT_HOVER_MM = 30.0       # --hover 时吸嘴停在「方块顶面 + 这么多」
# ★ 空走的安全余量: 吸嘴至少要比**方块顶面**高这么多才允许平移。
#   为什么用「高于方块顶面」而不是「高于纸面」: 桌面上的方块有 obj_h=26mm 高，
#   吸嘴只比纸面高 20mm 时平移过去是**直接撞方块**。按纸面算就漏掉了这一档
#   （2026-09-20 自查时发现的），按方块顶面算才对。
DEFAULT_OBJ_H_MM = 26.0       # 方块高（和 step4/main.py 默认一致）
MIN_CLEAR_ABOVE_OBJ_MM = 10.0  # 吸嘴要高出方块顶面这么多（默认悬停 30mm 轻松通过）

# ── ② 光心（--go-center）────────────────────────────────────────────
# 吸嘴离纸面停这几个高度，每个高度拍一次吸盘码。**XY 一步不横移。**
# ★ 为什么铺开到 6~52mm: 视差要靠**高度差**才看得出来 —— 停得越高，码离相机越近、
#   k 越大、画面里离光心越远，而"离光心多远"正是解 C 的方程。ρ=k/k₀ 的跨度
#   只有 ~30%（1080p 下），跨度越窄解出来的 C 越飘，所以尽量铺开。
# ★ 为什么最低敢停 6mm: 再低吸嘴就要碰纸了。**桌上必须先清空**（--obj-h 0）——
#   这条是纯升降，降下去正好压在方块上的话是真撞。
DEFAULT_CENTER_TIPS = (6.0, 12.0, 20.0, 30.0, 40.0, 52.0)
MIN_TIP_ABOVE_OBJ_MM = 3.0    # 最低那一停至少要高出障碍顶面这么多
# 光心的标准误超过这么多像素就**不收**（本次量测作废，让人去调机位/高度）。
# ★ 为什么闸门卡在「标准误」而不是「滑了多少像素」:
#   决定 C 准不准的是**杠杆**（ρ 铺开多少）+ 噪声，不是滑动距离。吸盘码要是正好
#   停在光心附近，它升降时几乎不滑 —— 但这时数据点本来就压在 C 上、**不用外推**，
#   C 反而是最准的；拿「滑得太少」把它毙掉就是误杀。反过来，ρ 铺得不够时杠杆差、
#   se 自然变大 —— 用 se 当闸门，这两种情况自然被分开，不用另外猜是哪个原因。
# ★ 10px 怎么来的: 纸面尺度下 1px ≈ 0.22mm → 10px ≈ 2.2mm；而闭环那条式子里
#   C 只以 (k−1)·C 的形式出现（抓的那个高度 k_tip≈1.06），实际影响 ≪1mm。
#   真正吃 C 的是**接近段**（吸盘码在 120mm 高处，k≈1.9）—— 那一步的影响
#   被 k 放大，所以这里留 10px 而不是更松。
MAX_CENTER_SE_PX = 10.0
FLUSH_FRAMES = 5              # 每次拍照前丢几帧 —— MJPG 缓冲里会留旧画面
# ★ 积累多少帧取并集（和 tools/test_camera_qr.py --shot 的 --frames 同一个数）。
#   为什么非得积累而不是"拍一张解一张": 吸盘码贴在抬起的末端上，**和纸面不在
#   同一个焦面**。相机自动对焦只能停在其中一个面上 —— wait_for_focus 的判据
#   只看纸面 4 码，于是对焦停在纸面，吸盘码就处在"时而解得出、时而解不出"的
#   状态（项目文档实测：同一机位能 0/30 也能 25/25）。所以吸盘码靠的是
#   **时间上的多样性**: AF 在纸面附近微调的那几帧里总有一两帧它对上了。
#   实测对比: test_camera_qr.py --shot 一次就 5/5（它积累 30 帧），
#   而本脚本原来只试 4 帧 → 老是差吸盘码这一张。8 倍的差距就在这儿。
ACCUM_FRAMES = 30
DEFAULT_SETTLE_S = 0.6        # 停稳再拍
SPEED_MM_S = 40.0             # 空走速度（和 step4 默认一致，不图快）
ACC = 40.0
RATIO = 30.0
# ════════════════════════════════════════════════════════════════════


# ─────────────────────────── 纯计算（可离线自检） ───────────────────────────
def fit_similarity(px_pts: np.ndarray, mm_pts: np.ndarray):
    """
    纸面 mm → 画面像素 的**相似变换**（等比缩放 + 旋转 + 平移，不允许镜像）。

    返回 (J, 每点残差)。J 是 2x2，J[:,0] = 纸面 +x 走 1mm 画面动多少像素。

    ★ 为什么用「相似」而不是一般仿射: 4 个点配 6 个自由度的仿射，残差会被
      各向异性/错切**吃掉** —— 纸不平、镜头畸变、某码认错，仿射都能凑上去，
      残差还是很小，等于把该报的错藏了。相似只有 4 个自由度（8 个方程），
      凑不动 → 残差才真的能当质量指标看。
    ★ 为什么不允许镜像: 本机位实测是纯旋转（det>0）。真出现镜像要先查原因
      （纸拿反了？），不该被拟合悄悄吸收掉。
    """
    mx, my = mm_pts[:, 0], mm_pts[:, 1]
    n = len(mm_pts)
    # px_x = a·mx − b·my + tx ;  px_y = b·mx + a·my + ty   （a,b 就是 s·cosθ, s·sinθ）
    A = np.vstack([np.column_stack([mx, -my, np.ones(n), np.zeros(n)]),
                   np.column_stack([my, mx, np.zeros(n), np.ones(n)])])
    B = np.concatenate([px_pts[:, 0], px_pts[:, 1]])
    sol, *_ = np.linalg.lstsq(A, B, rcond=None)
    a, b, tx, ty = (float(v) for v in sol)
    J = np.array([[a, -b], [b, a]])
    pred = np.column_stack([a * mx - b * my + tx, b * mx + a * my + ty])
    return J, np.linalg.norm(pred - px_pts, axis=1)


def fit_motion(samples) -> tuple[np.ndarray, np.ndarray]:
    """
    samples: [(臂X, 臂Y, 像素x, 像素y), …]  用**实测位姿**而不是指令增量做自变量。

    → (A, 每点残差)。A = d(像素)/d(机械臂 mm)，2x2。

    ★ 为什么自变量用实测位姿: 机械臂「指令走到 +30mm」和「真的停在 +30mm」是
      两回事（背隙、丢步、软着陆）。用回读的位姿当自变量，这些误差直接落进
      残差里被看见，而不是伪装成「映射不准」。
    """
    S = np.asarray(samples, dtype=np.float64)
    D = np.column_stack([S[:, 0], S[:, 1], np.ones(len(S))])
    M, *_ = np.linalg.lstsq(D, S[:, 2:4], rcond=None)
    return M[:2, :].T, np.linalg.norm(D @ M - S[:, 2:4], axis=1)


def fit_optical_center(stops):
    """
    ② 的核心算式。stops: [(像素x, 像素y, 边长px), …] —— 同一个 XY、不同高度上拍的吸盘码。
    → (光心 C, 滑动向量 v, C 的标准误px, 每点残差px, ρ序列)；点不够返回 None。

    模型:  proj_i = C + v · ρ_i        ρ_i = 边长_i / 边长_0

    ★ 为什么自变量用「边长的比」而不是「回读的高度」:
      边长 ∝ k = Z/(Z−z)，所以 ρ 就是 k 的比值 —— **纯光学量**。
      换成机械臂回读的高度，就等于把「机械臂报的 Z 不一定是毫米」这笔账
      （见 qr_vision.robot_scale_note: 实测一台 Magician 走 10mm 只报 8.3mm）
      又拖进来一次。而边长比把它整个绕开了 —— 这也是为什么 ② 不需要知道
      吸盘码印的是哪一张、也不需要 s_table。
    ★ 为什么 proj = C + v·ρ 是**一条直线**:
      吸嘴只升降、不横移 → 吸盘码在真实空间里只沿一条竖直线动 → 它在画面里的
      投影永远落在「光心 C」与「码在纸面的影子 L」连成的射线上；高度一变，
      码就沿这条射线滑，滑的位置由 k 定。射线上的点写成 C + k·(L−C)，
      除以 k₀ 归一就是 C + ρ·[k₀(L−C)]，即 C + v·ρ。ρ→0 处就是 C。
    ★ 所以 C 是**外推**出来的（ρ 只量到 1~1.3，C 在 ρ=0），不是直接看到的点。
      本函数把 C 的**标准误**一起返回，就是不让这个外推冒充精确值 ——
      报数时必须带上它，别只报一个光溜溜的 C。
    ★ 已知的**留白**: 这里把 ρ 当成精确值（普通最小二乘），可 ρ 是「两个带噪
      边长相除」，本身有 ~1% 的噪声。这一项会**沿着射线方向**把点推来推去，
      而沿射线方向的扰动几乎不产生残差 —— 于是它不会被下面的残差统计抓到，
      解出来的 ±se 因此是个**下界**（宁可说小也不说大）。实际影响是让 v 略微
      被**压缩**（经典的变量含误差衰减），C 跟着偏 ~1px 量级。
      · 影响随 |v| 走（扰动 = |v|·σ_ρ），而滑过的距离也随 |v| 走，所以**铺得开**
        的时候这一项大、**码停在光心附近**的时候这一项小；但前者是二阶地影响 C，
        后者是一阶 —— 两头都不至于失控，量级都在 1px 上下。
      · 一句话: ±se 当**下界**用（真实误差可能略大），所以闸门 MAX_CENTER_SE_PX
        留了余量，并且残差和 ρ 跨度都要一起看。
    """
    S = np.asarray(stops, dtype=np.float64)
    if len(S) < 3:
        return None
    rho = S[:, 2] / S[0, 2]
    if float(rho.max() - rho.min()) < 1e-6:        # 高度没变/码没解对
        return None
    D = np.column_stack([np.ones_like(rho), rho])  # [1, ρ]
    sol, *_ = np.linalg.lstsq(D, S[:, :2], rcond=None)
    C, v = sol[0], sol[1]
    resid = np.linalg.norm(D @ sol - S[:, :2], axis=1)
    dof = max(1, 2 * len(rho) - 4)                 # 两个坐标各拟合一次，共 4 个参数
    s2 = float((resid ** 2).sum()) / dof
    try:
        se = float(np.sqrt(s2 * np.linalg.inv(D.T @ D)[0, 0]))
    except np.linalg.LinAlgError:
        se = float("nan")
    return C, v, se, resid, rho


def cam_height_from_sizes(rho, zs: list[float], z_offset: float = 0.0) -> float | None:
    """
    用「边长比怎么随高度变」反解相机高度: ρ_i = k_i/k₀ = (Z−z₀)/(Z−z_i)。

    ★ z_offset 是**吸盘码比吸嘴尖高多少**（args.qr_height），必须加：
      zs 传进来的是吸嘴尖的高度（机械臂回读），而 ρ 说的是**码**的高度。
      少加这一项，解出来的就是「相机离吸嘴尖平面」而不是「离纸面」的高度，
      拿去跟 ① 的 cam_height 一比就差一个 120mm 左右 —— 会误报成「两条路对不上」。
    ★ 和 ① 那条 k_qr 反解出来的 cam_height 是**同一个量的两条独立来路**:
      ① 走的是「码在画面里动多快」（混进了机械臂 XY 的缩放），
      ② 走的是「码在画面里变多大」（纯光学）。两个数对得上才说明 ① 没跑偏。
    ★ 但这条仍然带「机械臂 Z 回读的缩放」这个未知因子 —— z 全是回读来的，
      而 z_offset 是拿尺子量的真实毫米。两者不同纲，所以这个校验只有 ~15% 的
      分辨力，别拿它当精测。
    """
    if len(rho) != len(zs) or len(rho) < 2:
        return None
    z0 = zs[0]
    out = []
    for r, z in zip(rho, zs):
        if abs(1.0 - float(r)) < 1e-9:
            continue
        Z = (z0 - float(r) * z) / (1.0 - float(r))       # 相机离**吸嘴尖平面**
        if Z > 0:
            out.append(float(Z) + float(z_offset))       # → 离**纸面**
    return float(np.median(out)) if out else None


def describe_axes(J: np.ndarray) -> dict:
    """把一个 2x2 映射翻译成人看的数: 每轴几 px/mm、朝哪个方向、有没有镜像、正交不。"""
    ex, ey = J[:, 0], J[:, 1]
    ang = lambda v: float(np.degrees(np.arctan2(v[1], v[0])))       # noqa: E731
    ax, ay = ang(ex), ang(ey)
    between = (ay - ax) % 180.0
    return {
        "px_per_mm_x": float(np.hypot(*ex)),
        "px_per_mm_y": float(np.hypot(*ey)),
        "angle_x_deg": ax,
        "angle_y_deg": ay,
        "skew_deg": float(between - 90.0),      # 两轴在画面里的夹角，应为 90°
        "det": float(np.linalg.det(J)),
    }


def cam_height(z_qr: float, k: float) -> float | None:
    """k = Z/(Z−z_qr) 反解相机离纸面的高度 Z = z_qr·k/(k−1)。k≤1 无解。"""
    if not (k > 1.0 + 1e-9):
        return None
    return z_qr * k / (k - 1.0)


def k_at_height(z: float, cam_z: float) -> float:
    """高度 z 处的视差放大倍数。z=0（纸面）时为 1。"""
    return cam_z / (cam_z - z)


def identify_sticker(data_mm_measured: float, candidates) -> tuple[str | None, float | None]:
    """量出来的「吸盘码数据区实际 mm」去认领是哪张打印件 → (标签, 该标签的 mm)。"""
    if not candidates:
        return None, None
    return min(candidates, key=lambda kv: abs(kv[1] - data_mm_measured))


# ─────────────────────────── 视线: 找码、量比例 ───────────────────────────
def grab(cap, flush: int = FLUSH_FRAMES):
    """拍一帧。先丢 flush 帧 —— 否则可能拿到运动**之前**的缓冲帧，数据全错。"""
    frame = None
    for _ in range(max(1, flush)):
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


def locate(cap, detector, want, focus_goal=None, frames: int = ACCUM_FRAMES):
    """
    等对焦 → 丢缓冲 → 积累若干帧**取并集**，直到 want 里的码全解出来。

    返回 (decoded, frame)；没凑齐也把手里最好的交出去（调用方自己判断缺谁）。

    ★ 为什么要「积累并集」，不是「连拍几张、哪张齐用哪张」:
      吸盘码和纸面**不在同一个焦面**。wait_for_focus 的判据只看纸面码（必须这样，
      见它自己的说明），对焦于是停在纸面，吸盘码便处在时而解得出、时而解不出的
      状态。单个帧凑齐 5 个码的概率因此不高，但**几十帧里总有一两帧吸盘码是对上
      的** —— 并集要的正是这个。tools/test_camera_qr.py --shot 走的就是这条路
      （积累 30 帧），实测一次就 5/5。

    ★★ 2026-09-20 之后: 根因已经用**锁焦距**解决了（qr_vision.lock_focus），
      那一段「总有一两帧对上」的运气不再是前提 —— 锁对了值，第一帧就 5 个码齐。
      并集留着当**余量**: 不锁焦距（--focus 负数）时它还是唯一的指望，
      而且悬停高度上曾经出现过「30 帧一帧都没对上」把流程卡死的情况。
    ★ 为什么并集**不会**污染几何: 取并集的前提是机械臂这一停**没动**，所以纸面码
      和吸盘码各自的像素位置在所有帧里是同一个值，谁在哪帧解出来都一样。
      一旦开始走位，就必须重新 locate —— sample() 每次停稳后都重来一遍，正是为此。
    ★ 为什么先丢 FLUSH_FRAMES 帧: MJPG 缓冲里留着**运动之前**的画面。不丢的话
      第一张就是旧图，吸盘码的像素位置会记成上一停的值 —— 而且完全看不出来，
      只是拟合残差莫名其妙地大。
    """
    # ① 等自动对焦。判据只看纸面码 —— 把吸盘码算进去会对焦等不到"5个全解出"。
    if focus_goal:
        wait_for_focus(cap, detector, focus_goal, quiet=True)
    # ② 丢缓冲，免得把运动前的旧画面当成本停的。
    if grab(cap) is None:
        return {}, None
    # ③ 积累并集，凑齐就提前收工。
    dec: dict = {}
    frame = None
    for _ in range(max(1, frames)):
        ok, f = cap.read()
        if not ok or f is None:
            continue
        frame = f
        got, _ = detect_sweep(f, detector, want=list(want))
        for k, v in got.items():
            dec.setdefault(k, v)          # 先解出来的那个位置算数
        if all(c in dec for c in want):
            break
    return dec, frame


def sample_suction(cap, detector, suction, focus_goal=None, frames: int = ACCUM_FRAMES):
    """
    在一个停点上盯着吸盘码拍 frames 帧，返回 (中心, 边长px, 命中帧数)；一帧都没解出返回 None。

    ★ 和 locate 的区别是**要不要中位数**:
      ① 只关心「码在哪儿」，第一帧解开够用了，于是 locate 够了就 break。
      ② 还要关心「码多大了」—— ρ = 边长/边长₀ 的跨度只有 ~30%，而边长本身有
      ~1% 的检测噪声（实测同一停点 146.0~147.3px）。1% 的单帧噪声外推到 ρ=0
      处的截距上就是十几像素，光心直接废掉。多帧取中位数把这一项压下去，
      而且**不花额外时间** —— 帧本来就要拍，只是不再提前收工。
    """
    if focus_goal:
        wait_for_focus(cap, detector, focus_goal, quiet=True)
    if grab(cap) is None:
        return None
    centers, sides = [], []
    for _ in range(max(1, frames)):
        ok, f = cap.read()
        if not ok or f is None:
            continue
        got, _ = detect_sweep(f, detector, want=[suction])
        q = got.get(suction)
        if q is None:
            continue
        centers.append(quad_center(q))
        sides.append(quad_side_px(q))
    if not centers:
        return None
    return (np.median(np.asarray(centers, dtype=np.float64), axis=0),
            float(np.median(sides)), len(centers))


def analyse_frame(dec: dict, layout, suction: str, suction_meas=None) -> dict:
    """
    把一次检测的结果翻译成数字。码不齐也不抛错 —— 缺什么就报什么。

    ★ suction_meas = (中心 (x,y), 边长px, 命中帧数) 是**多帧中位数**（sample_suction），
      给了就**压过** dec 里的单帧四边形。为什么非要有这个覆盖口:
        吸盘码在**另一个焦面**上，解码是"时而解得出、时而解不出"，而 dec 里存的只是
        **某一帧**的四边形。纸面 4 码不怕这个（4 个点做最小二乘，歪的那个会顶高残差、
        看得见）；吸盘码只有**一个**点、没有任何冗余来平掉单帧的抖动。
        2026-09-20 实测: 同一场景 9 帧的真实边长是 146.5~150.1px，而 ① 落盘的是
        188.7px（偏 26%）—— 下游那条「边长 → 认领是哪张打印件 → 反证 k」的校验
        整个建在这个坏数上，于是"k 被独立验证过"是个假象。所以相机这条路必须走中位数。
    """
    # ── 吸盘码: 优先「多帧中位数」，退回 dec 里的单帧四边形（--image 那条路）──
    s_center = s_side = s_hits = None
    if suction_meas is not None:
        s_center, s_side, s_hits = suction_meas
        s_center = np.asarray(s_center, dtype=np.float64)
    elif suction in dec:
        q = dec[suction]
        s_center, s_side = quad_center(q), quad_side_px(q)
    have_suction = s_center is not None

    out = {
        "missing": [c for c in list(layout.codes) + [suction]
                    if c not in dec and not (c == suction and have_suction)],
        "s_table": None, "paper": None, "paper_resid_px": None,
        "suction": None, "rot_deg": None,
    }
    have = [c for c in layout.codes if c in dec]
    if len(have) < 2:
        return out
    px = np.array([quad_center(dec[c]) for c in have])
    mm = np.array([layout.points_mm["center"][c] for c in have])
    J, resid = fit_similarity(px, mm)
    d = describe_axes(J)
    out["paper"] = d
    out["s_table"] = (d["px_per_mm_x"] + d["px_per_mm_y"]) / 2.0
    out["rot_deg"] = d["angle_x_deg"]
    out["paper_resid_px"] = float(resid.max())
    out["paper_n"] = len(have)

    if have_suction:
        # 两个 px/模块 口径都从边长推 —— 别再回头用四边形，那样覆盖就没意义了
        out["suction"] = {
            "px_center": [float(s_center[0]), float(s_center[1])],
            "side_px": float(s_side),
            "px_per_module_hist": float(s_side / qv.MODULES_TOTAL),  # 历史口径(/29)
            "px_per_module": float(s_side / qv.DATA_MODULES),        # 真值口径(/21)
        }
        if s_hits is not None:
            out["suction"]["n_frames"] = int(s_hits)
    return out


def print_vision(vis: dict, layout, suction: str) -> None:
    """把「桌面比例」和「吸盘码在画面哪儿」摆出来。"""
    print("\n── 桌面比例（拿纸面 4 码当尺子）──")
    if vis["s_table"] is None:
        n_have = len(layout.codes) - len([c for c in layout.codes if c in vis["missing"]])
        print(f"  ✗ 纸面码只认出 {n_have}/{len(layout.codes)} —— 少于 2 个就量不出比例。")
    else:
        d = vis["paper"]
        print(f"  s_table = {vis['s_table']:.4f} px/mm"
              f"     （纸面 +x 在画面里的方向 {d['angle_x_deg']:.2f}°）")
        # ★ 别把下面这行当"测出来了": fit_similarity 是 4 自由度的等比拟合，
        #   两轴必然相等、必然正交 —— 报出来只会给人"镜头很正"的错觉。
        #   各向异性和错切只有自由 2x2 的 fit_motion（--go 那条）才测得出来。
        print(f"  （纸面拟合按**等比**做，下面这行是按构造成立的，不是实测:"
              f" 两轴 {d['px_per_mm_x']:.4f} / {d['px_per_mm_y']:.4f} px/mm"
              f"  正交偏差 {d['skew_deg']:+.2f}°）")
        print(f"  拟合残差 最大 {vis['paper_resid_px']:.2f} px（{vis['paper_n']} 个码）"
              f"  ← 残差才是质量指标: 镜头歪/纸不平/畸变都会顶大它")
        if vis["paper_resid_px"] > 3.0:
            print("  ⚠ 残差偏大。纸不平/镜头畸变/某个码认错了都可能 —— 先别信这个数。")

    print("\n── 吸盘码 ──")
    s = vis["suction"]
    if s is None:
        print(f"  ✗ 没解出来（{suction!r}）。先用 tools/test_camera_qr.py --shot 调机位。")
        return
    print(f"  像素中心 ({s['px_center'][0]:.1f}, {s['px_center'][1]:.1f})"
          f"   数据区边长 {s['side_px']:.1f} px = {s['px_per_module']:.2f} px/模块")
    if vis["s_table"]:
        # 纸码的「每模块几像素」= s_table × 每模块多少 mm。两者一比就是放大倍数。
        paper_ppm = vis["s_table"] * qv.MODULE_MM
        rel = s["px_per_module"] / paper_ppm
        print(f"  纸码在画面里 {paper_ppm:.3f} px/模块"
              f"（= s_table {vis['s_table']:.4f} × 每模块 {qv.MODULE_MM:.4f}mm）")
        print(f"  ★ 相对桌面放大 ×{rel:.4f} —— 但这一张**还分不出 k**:")
        print(f"     放大倍数 = k × (吸盘码印多大 ÷ 纸码印多大)，两个未知数。")
        print(f"     加 --go 走一步，|A|/s_table 就把 k 单独解出来了。")


def print_motion(A: np.ndarray, resid: np.ndarray, samples, step_mm: float,
                 s_table: float | None) -> dict:
    """报「机械臂 mm → 画面像素」的实测映射，并从中解出 k。"""
    d = describe_axes(A)
    # 「原位」测了几次 —— 顺带当一次重复性检查（每次回原位应读到同一个数）
    n_home = sum(1 for s in samples if abs(s[0] - samples[0][0]) < 0.05
                 and abs(s[1] - samples[0][1]) < 0.05)
    print(f"\n── 空走实测（机械臂 mm → 画面像素）──")
    print(f"  每方向走 {step_mm:.0f}mm 再回原位，共 {len(samples)} 个测点"
          f"（其中「原位」测了 {n_home} 次）")
    print(f"  A = [[{A[0, 0]:8.4f}, {A[0, 1]:8.4f}]      px per 机械臂mm"
          f"\n       [{A[1, 0]:8.4f}, {A[1, 1]:8.4f}]]")
    print(f"  ∂/∂X: {d['px_per_mm_x']:.4f} px/mm  方向 {d['angle_x_deg']:+.2f}°")
    print(f"  ∂/∂Y: {d['px_per_mm_y']:.4f} px/mm  方向 {d['angle_y_deg']:+.2f}°")
    print(f"  两轴夹角偏差 {d['skew_deg']:+.2f}°（应为 0）   "
          f"det = {d['det']:+.3f}（正 = 无镜像）")
    print(f"  拟合残差 最大 {resid.max():.2f} px，均值 {resid.mean():.2f} px")
    if resid.max() > 5.0:
        print("  ⚠ 残差偏大 —— 可能没停稳就拍了，或某一步没走到位。重跑一次看看。")

    if not s_table:
        print("\n  ✗ 没有桌面比例，解不出 k。")
        return {}
    k_qr = (d["px_per_mm_x"] + d["px_per_mm_y"]) / 2.0 / s_table
    print(f"\n── 视差放大倍数 ──")
    print(f"  k_qr = |A| / s_table = {d['px_per_mm_x']:.4f} / {s_table:.4f} "
          f"= {k_qr:.4f}")
    if k_qr <= 1.005:
        print("  ⚠ k≈1 → 量出来「吸盘码就在纸面上」。可它明明比吸嘴高 ——")
        print("    要么吸盘码这次没解对（认到别的码/角点歪了），要么吸嘴已经贴到纸面。")
        return {"k_qr": k_qr}
    return {"k_qr": k_qr, **{f"A_{k}": v for k, v in d.items()}}


def use_ruler_cam_height(info: dict, cam_h: float, z_qr: float,
                         s_table: float | None) -> dict:
    """
    拿卷尺量的相机高度当 k 的基准，覆盖 |A|/s_table 解出来的那个。

    ★ 为什么必须留个入口给卷尺 —— k 有两条互相独立的来路:
      · |A|/s_table: A 是 px per **机械臂报的** mm，而机械臂报的 XY 不是真毫米
        （本项目多处实测: 两轴比例还不一样，见 qr_vision.robot_scale_note）。
        这个刻度误差**原封不动**乘进 k 里 —— 2026-09-20 实测把它顶高了 10%
        （k=1.89 vs 卷尺给的 1.72），并且顺手把「贴纸是哪一张」也带偏了。
      · 卷尺量的是真毫米，不含任何机械臂刻度。
      两条一比，比值就是机械臂 XY 的刻度偏差。**不是故障**（闭环里 A 和指令
      同纲、自己抵消），但它不是 k 的一部分，所以 k 以卷尺为准。
    """
    k_a = float(info.get("k_qr") or 0.0)
    k_ruler = k_at_height(z_qr, cam_h)
    print("\n── 卷尺校核（--cam-height）──")
    print(f"  你量的 Z_cam = {cam_h:.1f} mm → k_qr = Z/(Z−z_qr) = "
          f"{cam_h:.1f}/({cam_h:.1f}−{z_qr:.1f}) = {k_ruler:.4f}")
    if k_a > 1.0:
        ratio = k_a / k_ruler
        print(f"  ① 从 |A|/s_table 解出的 k_qr = {k_a:.4f}"
              f"  → 两者差 {abs(ratio - 1) * 100:.1f}%（比值 {ratio:.4f}）")
        print(f"  ★ 这个比值就是**机械臂 XY 的刻度偏差**: 机械臂报 1mm，真走 "
              f"{ratio:.4f}mm。")
        print(f"    正常现象（qr_vision.robot_scale_note）—— A 和指令同纲、闭环里"
              f"自己抵消;")
        print(f"    但它会原封不动乘进 |A|/s_table，所以 k 取卷尺的 "
              f"{k_ruler:.4f}，不取 {k_a:.4f}。")
        if abs(ratio - 1) > 0.25:
            print(f"    ⚠ 差得比已知的刻度误差大。先确认卷尺量的真是「纸面→镜头」"
                  f"（不是量到相机外壳顶），再看那次空走有没有走到位。")
    info["k_qr"] = k_ruler
    info["k_qr_from_A"] = round(k_a, 4) if k_a > 1.0 else None
    info["arm_xy_scale_from_A"] = round(k_a / k_ruler, 4) if k_a > 1.0 else None
    info["cam_height_source"] = "ruler"
    return info


def print_heights(k_qr: float, z_tip: float, z_qr: float, obj_h: float, s_table: float):
    """
    从 k 反解相机高度，再报视差到底有多大 —— 这是本脚本最该记住的数。

    ★ 为什么要把「量的时候」和「抓的时候」两个 k 都报出来: A 是在**悬停高度**
      量到的，所以解出来的 k_qr 属于那个高度。可真正抓的时候吸嘴要降到方块顶面
      （z=obj_h），高度变了、k 跟着变。这两个数不一样大 —— 差多少正是 ② 要补的。
    """
    cam_z = cam_height(z_qr, k_qr)
    print(f"\n── 高度与视差（最值钱的一段）──")
    print(f"  量的时候: 吸嘴离纸面 z_tip = {z_tip:.1f} mm，吸盘码在它上面 "
          f"{z_qr - z_tip:.0f} mm → z_qr = {z_qr:.1f} mm")
    if cam_z is None:
        print("  ✗ k≤1，反解不出相机高度。")
        return {}
    print(f"  → 相机离纸面 Z_cam = z_qr·k/(k−1) = {cam_z:.0f} mm"
          f"    （画面 1mm 在纸面上占 {s_table:.3f}px）")
    k_tip = k_at_height(z_tip, cam_z)
    print(f"  吸嘴处 k_tip = Z/(Z−{z_tip:.0f}) = {k_tip:.4f}"
          f"     吸盘码处 k_qr = Z/(Z−{z_qr:.0f}) = {k_qr:.4f}")
    ratio = 1.0 - k_tip / k_qr
    print(f"\n  ★ 若「把吸盘码中心压在方块上」就开始下降，吸嘴会**朝画面中心偏**：")
    print(f"      偏的比例 = 1 − k_tip/k_qr = {ratio * 100:.1f}%")
    print(f"      距光心 400px（≈{400 / s_table:.0f}mm）处 ≈ {ratio * 400 / s_table:.0f}mm"
          f" —— 比标定里那些几毫米的残差大一个量级。")

    # 真到抓取时吸嘴停在方块顶面，这两个数会变 —— ② 的入口
    k_grasp = k_at_height(obj_h, cam_z)
    print(f"\n  ★ 真抓时吸嘴降到方块顶面 z={obj_h:.0f}mm，k 会变成 "
          f"{k_grasp:.4f}（不是上面那个 {k_tip:.4f}）——")
    print(f"     所以闭环要按「当时的 z」现算 k_tip，不能拿这一次的数一路用下去。")
    return {"cam_z": cam_z, "k_tip": k_tip, "k_grasp": k_grasp, "pull_in_frac": ratio}


def print_control(A: np.ndarray, h: dict, z_qr: float, obj_h: float,
                  center, s_table: float) -> dict:
    """
    把闭环要用的数写清楚，并注明现在还差什么。

    ★★ 系数为什么是「吸盘码的 k ÷ 方块顶面的 k」，**不是吸嘴处的 k**（2026-09-20 重推）:
      A 量的是**码在画面里动多快** → A = s_table·k(z_qr)·(机械臂刻度)，
      所以 A⁻¹ 把像素换成 mm 时用的那个因子天生就带 k(z_qr)。
      推一遍: 要让吸嘴落在方块上，真实位移
          Δu = (方块−C)/(s·k_方块顶面) − (码−C)/(s·k_qr)
      而 δ_机械臂 = Δu ÷ 机械臂刻度，代进 A 就是下面这条式子。
      ★ 用错系数的后果**不是收敛慢，而是停在一个固定的错位上**: 这条式子是
        「一步解」—— 系数偏 factor 倍，机械臂就永远停在 factor·u_c 处不再动
        （u_c = 方块离光心的距离）。所以**离画面中心越远的方块错得越多**，
        正巧在光心附近的反而看着挺准 —— 这个特征不容易被发现，正是自检要用的一条。
      ★ 旧版这里写的是 k_tip = Z/(Z−z_tip)（悬停 15mm 时 1.05），而正确值
        ≈1.58 —— 差一半。别再照抄旧式子。
    """
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("\n  ✗ A 不可逆（两轴映射退化 —— 机械臂是不是没动？）")
        return {}
    cam_z = (h or {}).get("cam_z")
    c = None
    if cam_z:
        c = k_at_height(z_qr, cam_z) / k_at_height(obj_h, cam_z)
    print("\n── 闭环控制律（③ 做完才能一路用）──")
    print("  δ_机械臂 = A⁻¹ · [ c·(方块像素 − 光心) − (吸盘码像素 − 光心) ]")
    print(f"  A⁻¹ = [[{Ainv[0, 0]:9.5f}, {Ainv[0, 1]:9.5f}]"
          f"      mm per px"
          f"\n         [{Ainv[1, 0]:9.5f}, {Ainv[1, 1]:9.5f}]]")
    if c is not None:
        print(f"  c = k(码 z={z_qr:.0f}) / k(方块顶 z={obj_h:.0f}) = "
              f"{k_at_height(z_qr, cam_z):.4f} / {k_at_height(obj_h, cam_z):.4f}"
              f" = {c:.4f}")
        print(f"      ★ 是**吸盘码**那一层的 k，不是吸嘴处的 k"
              f"（吸嘴处 k={h.get('k_tip', float('nan')):.4f}，别拿它当系数）")
    else:
        print("  c = ✗ 解不出来（没有相机高度）")
    print(f"  光心 C = ({center[0]:.0f}, {center[1]:.0f}) px")
    print(f"  换算感: 画面里 1px ≈ {1.0 / s_table:.3f} mm（纸面尺度）")
    print("\n  ★ 现在**还不能**拿它驱机械臂，还差:")
    print("    · 「方块此刻在画面哪儿」还没接进来（color_vision 报的）—— ③ 没做。")
    print("    · 每走一步都要**重拍重算**（c 里的 k 按当时的 z 现算），")
    print("      别拿这次的数一路用下去。")
    return {"A_inv": Ainv.tolist(), "c_coef": round(c, 4) if c else None}


# ─────────────────────────── 纸面高度 ───────────────────────────
def load_table_z() -> float | None:
    """纸面 Z: 优先 --probe 记下的，其次 step2 落盘的。和 step4 同一套来源。"""
    for path, key in ((PICK_RESULT_JSON, "table_z"), (ROBOT_POINTS_JSON, "table_z")):
        if path.exists():
            try:
                v = json.loads(path.read_text(encoding="utf-8")).get(key)
            except (OSError, json.JSONDecodeError):
                continue
            if v is not None:
                return float(v)
    return None


# ─────────────────────────── 机械臂那一段 ───────────────────────────
def run_arm(args, layout, suction, vis: dict, detector, cap) -> dict:
    """
    连臂 → 空走几步量 A → 报数。**只动 XY**，不下降、不吸、不碰东西。
    返回要落盘的段（失败返回 {}）。
    """
    import step2_teach_coords as tc

    want = list(layout.codes) + [suction]
    # 等对焦时**只拿纸面 4 码当判据**（吸盘码在另一个焦面，算进去就等不到收敛）。
    focus_goal = list(layout.codes)
    api, dType = load_sdk()
    port = find_port(dType, args.port)
    if not port:
        print("✗ 没找到机械臂串口。电源开了吗？USB 插好了吗？")
        return {}
    print(f"\n[连接] {port} @ 115200 …")
    ret = dType.ConnectDobot(api, port, 115200)
    if ret[0] != 0:
        print(f"✗ 连接失败 state={ret[0]} (1=未找到设备 2=端口被占用)")
        return {}

    home = None
    moved = False
    try:
        dType.SetCmdTimeout(api, 5000)
        ares, alist = tc.read_alarms(api, dType)
        print(f"[报警] result={ares}  {tc.describe_alarms(alist)}")
        if tc.has_alarms(alist):
            tc.report_alarm_detail(alist)
            if tc.needs_homing(alist):
                print("\n✗ 有丢步报警: 零点已不可信，先回零再跑（tools/home_arm.py）。")
                return {}
            ok, _ = tc.resolve_alarms(api, dType, alist, args.clear_alarms)
            if not ok:
                return {}

        dType.SetQueuedCmdClear(api)
        dType.SetQueuedCmdStartExec(api)
        dType.SetPTPJointParams(api, SPEED_MM_S, SPEED_MM_S, SPEED_MM_S, SPEED_MM_S,
                                ACC, ACC, ACC, ACC, isQueued=0)
        tc._set_coord_params(api, dType, SPEED_MM_S, ACC)
        dType.SetPTPCommonParams(api, RATIO, RATIO, isQueued=0)

        limits = {a: tuple(v) for a, v in tc.DEF_LIMITS.items()}
        if args.limits:
            for spec in args.limits.split(","):
                ax, lo, hi = spec.split(":")
                limits[ax.strip()] = (float(lo), float(hi))

        table_z = args.table_z if args.table_z is not None else load_table_z()
        if table_z is None:
            print("✗ 不知道纸面 Z，就没法确认吸嘴的高度够不够安全。")
            print("  先跑 python3 src/step4_pick_test.py --probe，或给 --table-z。")
            return {}
        # Z 锁在方块顶面之上 —— 本脚本根本不该下降，这道闸是防手滑。
        limits["z"] = (table_z + args.obj_h, limits["z"][1])

        cur = tc.read_pose_stable(api, dType, 5)
        home = dict(cur)
        z_tip = cur["z"] - table_z

        print(f"\n[当前] {tc.fmt(cur)}")
        print(f"  纸面 Z={table_z:.2f} → 吸嘴离纸面 z_tip={z_tip:.1f}mm")
        print(f"  吸盘码 z_qr = z_tip + {args.qr_height:.0f} = {z_tip + args.qr_height:.1f}mm")

        # ── 高度: 要么先升到 --hover，要么确认现在够高 ──
        if args.hover is not None:
            target_z = min(table_z + args.obj_h + args.hover, limits["z"][1])
            if abs(target_z - cur["z"]) > 0.5:
                # ★ 这个参数能升也能降（--obj-h 0 --hover 15 就是从悬停处降下来），
                #   所以别写死"升高" —— 2026-09-20 就被这句带偏过。
                verb = "升到" if target_z > cur["z"] else "降到"
                print(f"\n[{verb}量测高度] Z={target_z:.2f}"
                      f"（纸面 {table_z:.2f} + 障碍高 {args.obj_h:.0f} "
                      f"+ {args.hover:.0f}）——")
                print("       高度一变 k 就变，量的高度要和以后用的一致；")
                print("       但吸嘴抬太高会看不清吸盘码 —— 方块不在场时贴纸面量更稳。")
                if not args.yes:
                    input("       回车开始（Ctrl-C 中止）… ")
                tgt = {"x": cur["x"], "y": cur["y"], "z": target_z, "r": cur["r"]}
                ok, why = tc.check_target(tc.read_pose_stable(api, dType, 5), tgt, limits)
                if ok:
                    ok, why = tc.move_to(api, dType, tgt, limits, args.mode)
                if not ok:
                    print(f"✗ 升高失败: {why}")
                    return {}
                tc.verify_arrival(api, dType, tgt)
                home = tc.read_pose_stable(api, dType, 5)
                z_tip = home["z"] - table_z
                moved = True
        need_z = table_z + args.obj_h + MIN_CLEAR_ABOVE_OBJ_MM
        if home["z"] < need_z:
            print(f"\n✗ 吸嘴现在 Z={home['z']:.2f}（离纸面 {z_tip:.1f}mm），"
                  f"比方块顶面只高 {z_tip - args.obj_h:.1f}mm ——")
            print(f"  平移会撞到方块。要求 Z ≥ {need_z:.2f}"
                  f"（纸面 {table_z:.2f} + 方块高 {args.obj_h:.0f} "
                  f"+ 余量 {MIN_CLEAR_ABOVE_OBJ_MM:.0f}）。")
            print(f"  两条路，看桌上有没有方块:\n"
                  f"   · 有方块 → 用 --hover {DEFAULT_HOVER_MM:.0f} 升起来再量。\n"
                  f"   · 没方块 → 把方块拿走，用 --obj-h 0 --hover 15 贴着纸面量 ——\n"
                  f"     ★ 这是**首选**: 吸盘码和纸面不在同一焦面，吸嘴抬得越高码越糊。\n"
                  f"       实测吸嘴离纸面 10mm 时连中 5 次、抬到 56mm 时连丢 5 次。\n"
                  f"       ① 只需要纸 + 吸盘码，方块本来就不必在场。")
            return {}

        # ── 确认 ──
        print(f"\n⚠ 接下来只动 XY: 机械臂会在原位附近走 {args.step:.0f}mm 的方步"
              f"（每向走一次、回原位），共 {4 * 2} 次移动。")
        print(f"  不下降、不吸、不碰任何东西。Z 锁在 {limits['z'][0]:.2f} 以上。")
        print("  清空机械臂周围；Ctrl-C 随时停；本脚本不回零、不写任何末端参数。")
        if not args.yes:
            if input("  确认开始？(y/N) ").strip().lower() != "y":
                print("已取消，未发任何运动指令（可能刚升高过）。")
                return {}

        def goto(x, y, z) -> bool:
            tgt = {"x": float(x), "y": float(y), "z": float(z), "r": home["r"]}
            ok, why = tc.check_target(tc.read_pose_stable(api, dType, 5), tgt, limits)
            if not ok:
                print(f"  ✗ {why}")
                return False
            ok, why = tc.move_to(api, dType, tgt, limits, args.mode)
            if not ok:
                print(f"  ✗ 走位失败: {why}")
                return False
            ok, why = tc.verify_arrival(api, dType, tgt)
            if not ok:
                # 没到位不算致命（背隙/负载），但要说出来 —— 残差本来就该看见，
                # 藏着就变成"映射不准"了。
                print(f"  ⚠ 没走到位: {why}")
            return True

        def sample(tag: str):
            """停在原地: 拍一组 + 回读位姿。返回 (臂X, 臂Y, 像素x, 像素y) 或 None。"""
            time.sleep(DEFAULT_SETTLE_S)
            dec, _f = locate(cap, detector, want, focus_goal=focus_goal,
                             frames=args.frames)
            pose = tc.read_pose_stable(api, dType, 5)
            if suction not in dec:
                print(f"  ✗ [{tag}] 这一停没解出吸盘码，跳过")
                return None
            c = quad_center(dec[suction])
            print(f"  · [{tag}] 臂({pose['x']:8.2f},{pose['y']:8.2f})"
                  f" → 码像素({c[0]:7.1f},{c[1]:7.1f})")
            return (pose["x"], pose["y"], float(c[0]), float(c[1]))

        print("\n[量测] 依次停在: 原位 → X+ → 原位 → X− → 原位 → Y+ → 原位 → Y− → 原位")
        samples = []
        s = sample("原位")
        if s is None:
            print("✗ 原位就看不清吸盘码 —— 先调机位，别动臂。")
            # ★ 高度是最容易被漏掉的那个原因（2026-09-20 实测踩过）: 吸盘码和纸面
            #   不在同一个焦面，对焦停在纸面时，吸嘴抬得越高、码离相机越近就越糊。
            #   所以"看不见码"未必是机位问题 —— 抬高了就先把高度降回来试。
            if z_tip > 25.0:
                print(f"   ★ 先怀疑高度: 吸嘴现在离纸面 {z_tip:.0f}mm。"
                      f"实测离纸面 10mm 时连中 5 次、\n"
                      f"     抬到 56mm 时连丢 5 次 —— 相机对焦在纸面上，码越高越糊。\n"
                      f"     方块不在桌上就用 --obj-h 0 --hover 15 贴纸面再量；\n"
                      f"     桌上确实有方块，那就只能调机位（或临时把方块挪开）。")
            return {}
        samples.append(s)
        origin = (home["x"], home["y"], home["z"])

        for tag, axis, sign in (("X+", "x", +1), ("X−", "x", -1),
                                ("Y+", "y", +1), ("Y−", "y", -1)):
            off = dict(home)
            off[axis] = home[axis] + sign * args.step
            moved = True
            if not goto(off["x"], off["y"], off["z"]):
                return {}
            if (s := sample(tag)) is not None:
                samples.append(s)
            if not goto(*origin):
                return {}
            if (s := sample("原位")) is not None:
                samples.append(s)

        if len(samples) < 5:
            print(f"\n✗ 只拿到 {len(samples)} 个有效测点，拟合不可靠（至少 5 个）。")
            return {}

        A, resid = fit_motion(samples)
        info = print_motion(A, resid, samples, args.step, vis["s_table"])
        if not info:
            return {}
        z_qr = z_tip + args.qr_height
        # ★ 先让卷尺（--cam-height）把 k_qr 顶掉，再往后走 —— 后面「认领贴纸」
        #   和 print_heights 都用这个 k，用错的 k 会一路错到底。
        if args.cam_height is not None:
            info = use_ruler_cam_height(info, args.cam_height, z_qr, vis["s_table"])
        h = print_heights(info["k_qr"], z_tip, z_qr, args.obj_h, vis["s_table"])

        # 反推「吸盘码数据区实际多大」→ 认领是哪张打印件
        sticker = None
        if vis["suction"]:
            side = vis["suction"]["side_px"]
            data_mm = side / (vis["s_table"] * info["k_qr"])
            cands = suction_size_candidates()
            tag, near = identify_sticker(data_mm, cands)
            print(f"\n── 吸盘码到底是哪一张 ──")
            print(f"  数据区实际 = 边长px ÷ (s_table × k_qr) = "
                  f"{side:.1f} ÷ ({vis['s_table']:.3f} × {info['k_qr']:.3f})"
                  f" = {data_mm:.3f} mm")
            if tag:
                err = abs(near - data_mm)
                print(f"  最接近印出来的「{tag}」那张（数据区 {near:.3f}mm，"
                      f"差 {err:.3f}mm）")
                if err > 0.8:
                    print(f"  ⚠ 差得有点多。k 或 s_table 里有一个不对劲 —— 上面那些数"
                          f"先别当准的用。")
                sticker = {"measured_data_mm": round(data_mm, 3),
                           "best_tag": tag, "best_mm": near, "err_mm": round(err, 3)}
            else:
                print("  （没有候选尺寸可比 —— data/suction_qr.json 里没有 variants？）")
                sticker = {"measured_data_mm": round(data_mm, 3)}

        ctrl = print_control(A, h, z_qr, args.obj_h, args.center, vis["s_table"])
        return {
            "A_row_major": A.tolist(),
            "A_inv_row_major": ctrl.get("A_inv"),
            "px_per_mm_x": info.get("A_px_per_mm_x"),
            "px_per_mm_y": info.get("A_px_per_mm_y"),
            "angle_x_deg": info.get("A_angle_x_deg"),
            "angle_y_deg": info.get("A_angle_y_deg"),
            "skew_deg": info.get("A_skew_deg"),
            "det": info.get("A_det"),
            "resid_max_px": round(float(resid.max()), 3),
            "resid_mean_px": round(float(resid.mean()), 3),
            "step_mm": args.step,
            "samples": [[round(v, 3) for v in s] for s in samples],
            "robot_home": {k: round(home[k], 3) for k in ("x", "y", "z", "r")},
            "table_z": round(table_z, 3),
            "z_tip_mm": round(z_tip, 2),
            "z_qr_mm": round(z_qr, 2),
            "k_qr": round(info["k_qr"], 4),
            # ★ 闭环控制律里的系数（③ 要用）: c = k(码)/k(方块顶面)。
            #   注意两件事，缺一个这数就是错的:
            #   · **不是**下面那个 k_tip —— k_tip 只说明视差有多大，不是系数;
            #   · c 随「当时的 z_tip」和「--obj-h」变，所以这里存的只是**本次那一套
            #     参数下的样例**（比如这次 --obj-h 0 就退化成 k(码)/1）。③ 必须
            #     按抓取那一刻现算，别直接拿这个数。
            "c_coef_example": ctrl.get("c_coef"),
            "c_coef_formula": "c = k(Z_cam, z_tip+qr_height) / k(Z_cam, obj_h)",
            "c_coef_basis": {"obj_h_mm": args.obj_h, "z_tip_mm": z_tip,
                             "z_qr_mm": z_qr, "cam_height_mm": h.get("cam_z")},
            "k_tip": round(h.get("k_tip", 0.0), 4) or None,
            "cam_height_mm": round(h["cam_z"], 1) if h.get("cam_z") else None,
            # ★ k 到底信谁、以及「机械臂 XY 刻度」这个附带产物：
            #   cam_height_source 有值（"ruler"）说明 k 是从卷尺来的；
            #   arm_xy_scale_from_A 是 ① 的 |A|/s_table 比卷尺高出的倍数
            #   —— 正常现象（不是故障），别拿它去调机械臂。
            "cam_height_source": info.get("cam_height_source", "from_1_k"),
            "cam_height_ruler_mm": args.cam_height,
            "k_qr_from_A": info.get("k_qr_from_A"),
            "arm_xy_scale_from_A": info.get("arm_xy_scale_from_A"),
            "pull_in_frac": round(h["pull_in_frac"], 4) if h else None,
            "sticker": sticker,
            "optical_center_px": list(args.center),
            "mode": args.mode,
        }
    finally:
        # ★ 不管量成没量成、是不是被 Ctrl-C 打断，都把臂放回量之前那个位姿。
        #   放不回原位就等于把机械臂丢在一个「和记录对不上」的位置上，后面
        #   任何脚本读到的当前位姿都会和操作者的印象不符。
        if home is not None and moved:
            try:
                tc.move_to(api, dType, home,
                           {"x": (-1e9, 1e9), "y": (-1e9, 1e9),
                            "z": (-1e9, 1e9), "r": (-1e9, 1e9)}, args.mode, timeout=20)
                print(f"\n[回位] 已回到原位 ({home['x']:.1f}, {home['y']:.1f}, {home['z']:.1f})")
            except Exception:
                pass
        try:
            dType.SetQueuedCmdStopExec(api)
        except Exception:
            pass
        try:
            dType.DisconnectDobot(api)
            print("[断开] 完成")
        except Exception:
            pass


# ─────────────────────────── ② 光心那一段 ───────────────────────────
def print_center(C, v, se, resid, rho, s_table: float | None,
                 cam_1: float | None, assumed) -> None:
    """把 ② 量到的东西报成人话 —— 包括**它有多不确定**。"""
    slide = float(np.hypot(*v) * (float(rho.max()) - float(rho.min())))
    print(f"\n── ② 实测光心 ──")
    print(f"  吸盘码在画面里沿一条过光心的射线滑了 {slide:.0f}px"
          f"（ρ = 边长/边长₀ 从 {rho.min():.3f} 铺到 {rho.max():.3f}）")
    print(f"  拟合残差 最大 {resid.max():.2f}px，均值 {resid.mean():.2f}px"
          + ("   ⚠ 偏大 —— 可能有某一停码解歪了（半个码也会解出个小边长）"
             if resid.max() > 4.0 else ""))
    print(f"\n  ★ 光心 C = ({C[0]:.0f}, {C[1]:.0f}) px   ± {se:.1f}px（标准误）")
    dx, dy = C[0] - assumed[0], C[1] - assumed[1]
    print(f"     和原来假设的 ({assumed[0]:.0f}, {assumed[1]:.0f}) 差 "
          f"({dx:+.0f}, {dy:+.0f})px"
          + (f" ≈ ({dx / s_table:+.1f}, {dy / s_table:+.1f})mm（纸面尺度）"
             if s_table else ""))
    if abs(se) > 0.0:
        print(f"     ★ C 是**外推**出来的（ρ 只量到 {rho.min():.2f}~{rho.max():.2f}，"
              f"C 在 ρ=0）—— 所以上面那个 ±{se:.1f}px 必须跟着一起用。")
        print(f"       它是个**下界**（ρ 自己的噪声没算进去，见 fit_optical_center），")
        print(f"       已经过了 {MAX_CENTER_SE_PX:.0f}px 那道闸门，但别再往下四舍五入掉。")
    if cam_1:
        print(f"\n  顺带一条**独立**校验: ② 还能用「边长怎么随高度变」反解相机高度"
              f"（cam_height_from_sizes）——")
        print(f"     ① 那条路（靠码在画面里动多快）给的是 {cam_1:.0f}mm，"
              f"两个数对得上才说明 ① 没跑偏。")


def run_center(args, layout, suction, vis: dict, detector, cap) -> dict:
    """
    ② 实测光心: **不横移**，只把吸嘴沿 Z 停在几个高度上，每个高度拍吸盘码。
    返回要落盘的 center 段（失败返回 {}）。

    ★ 为什么非得实测: 闭环那条式子里 (方块像素−光心) 和 (吸盘码像素−光心) 都带 C。
      拿"画面中心"当 C 是纯假设 —— 真光心偏多少，两项就一起偏多少。
    ★ 为什么敢只动 Z: 横移要靠「机械臂 XY 是毫米」这个前提（见 robot_scale_note，
      这台机器不是），而升降**只用来改变码的高度**，高度对不对不要紧 ——
      解 C 用的是边长的**比**。所以 ② 绕开了机械臂模型不准这件事。
    """
    import step2_teach_coords as tc

    api, dType = load_sdk()
    port = find_port(dType, args.port)
    if not port:
        print("✗ 没找到机械臂串口。电源开了吗？USB 插好了吗？")
        return {}
    print(f"\n[连接] {port} @ 115200 …")
    ret = dType.ConnectDobot(api, port, 115200)
    if ret[0] != 0:
        print(f"✗ 连接失败 state={ret[0]} (1=未找到设备 2=端口被占用)")
        return {}

    home = None
    moved = False
    try:
        dType.SetCmdTimeout(api, 5000)
        ares, alist = tc.read_alarms(api, dType)
        print(f"[报警] result={ares}  {tc.describe_alarms(alist)}")
        if tc.has_alarms(alist):
            tc.report_alarm_detail(alist)
            if tc.needs_homing(alist):
                print("\n✗ 有丢步报警: 零点已不可信，先回零再跑（tools/home_arm.py）。")
                return {}
            ok, _ = tc.resolve_alarms(api, dType, alist, args.clear_alarms)
            if not ok:
                return {}

        dType.SetQueuedCmdClear(api)
        dType.SetQueuedCmdStartExec(api)
        dType.SetPTPJointParams(api, SPEED_MM_S, SPEED_MM_S, SPEED_MM_S, SPEED_MM_S,
                                ACC, ACC, ACC, ACC, isQueued=0)
        tc._set_coord_params(api, dType, SPEED_MM_S, ACC)
        dType.SetPTPCommonParams(api, RATIO, RATIO, isQueued=0)

        limits = {a: tuple(v) for a, v in tc.DEF_LIMITS.items()}
        if args.limits:
            for spec in args.limits.split(","):
                ax, lo, hi = spec.split(":")
                limits[ax.strip()] = (float(lo), float(hi))

        table_z = args.table_z if args.table_z is not None else load_table_z()
        if table_z is None:
            print("✗ 不知道纸面 Z，就没法确认吸嘴的高度够不够安全。")
            print("  先跑 python3 src/step4_pick_test.py --probe，或给 --table-z。")
            return {}
        limits["z"] = (table_z + args.obj_h, limits["z"][1])

        cur = tc.read_pose_stable(api, dType, 5)
        home = dict(cur)
        print(f"\n[当前] {tc.fmt(cur)}")

        # ── 这一条是纯升降: 降到哪儿就压在哪儿，桌面**必须**是空的 ──
        # ★ --obj-h 是**用户声明**的，本脚本自己看不出桌上有没有东西 ——
        #   相机能看见台面，但「台面上那个是方块还是吸盘自己的影子」判不了，
        #   而这里要的是**最坏情况**的高度，判错就是撞。所以宁可让用户显式声明。
        tips = sorted(float(t) for t in args.center_tips)
        low = tips[0]
        if low < args.obj_h + MIN_TIP_ABOVE_OBJ_MM:
            print(f"\n✗ 最低那一停是吸嘴离纸面 {low:.1f}mm，而**你声明的**方块高是 "
                  f"{args.obj_h:.0f}mm ——")
            print(f"  本步是**纯升降**，降下去正好压在方块顶上。要求最低一停 ≥ "
                  f"{args.obj_h + MIN_TIP_ABOVE_OBJ_MM:.1f}mm。")
            print(f"  ★ 这个 {args.obj_h:.0f}mm 是默认值（和 step4/main.py 一致），"
                  f"**不是**它看见了方块。")
            print(f"  · 桌上确实没东西（方块已清）→ 加 --obj-h 0，它就不再拦 "
                  f"（★ 首选: 高度能铺得更开，C 更准）")
            print(f"  · 桌上有方块 → 只能把最低一停抬到 --obj-h 之上，"
                  f"高度铺不开、C 会飘。")
            return {}

        print(f"\n⚠ 接下来只动 Z: 吸嘴在**同一个 XY**上依次停在 "
              f"{'、'.join(f'{t:.0f}' for t in tips)} mm（离纸面），每个高度拍一次吸盘码。")
        print(f"  一步都不横移，不会碰到旁边的任何东西。")
        print(f"  高度下限 = 纸面 {table_z:.2f} + 你声明的障碍高 {args.obj_h:.0f} = "
              f"{limits['z'][0]:.2f}。")
        if args.obj_h <= 0.0:
            print(f"  ★ 你声明的是「桌上什么都没有」（--obj-h 0）—— 本脚本**看不出**"
                  f"桌上有没有东西，")
            print(f"     这个下限完全是照你这句话算的。降下去之前**再看一眼台面**。")
        print("  Ctrl-C 随时停；本脚本不回零、不写任何末端参数。")
        if not args.yes:
            if input("  确认开始？(y/N) ").strip().lower() != "y":
                print("已取消，未发任何运动指令。")
                return {}

        focus_goal = list(layout.codes)

        def goto_z(z) -> bool:
            tgt = {"x": home["x"], "y": home["y"], "z": float(z), "r": home["r"]}
            ok, why = tc.check_target(tc.read_pose_stable(api, dType, 5), tgt, limits)
            if not ok:
                print(f"  ✗ {why}")
                return False
            ok, why = tc.move_to(api, dType, tgt, limits, args.mode)
            if not ok:
                print(f"  ✗ 走位失败: {why}")
                return False
            ok, why = tc.verify_arrival(api, dType, tgt)
            if not ok:
                print(f"  ⚠ 没走到位: {why}")
            return True

        stops, zs = [], []
        print(f"\n[量测] 只在 {home['x']:.1f}, {home['y']:.1f} 上升降:")
        for tip in tips:
            z = table_z + tip
            moved = True
            if not goto_z(z):
                return {}
            time.sleep(DEFAULT_SETTLE_S)
            got = sample_suction(cap, detector, suction,
                                 focus_goal=focus_goal, frames=args.frames)
            if got is None:
                print(f"  ✗ [离纸面 {tip:4.1f}mm] 这一停没解出吸盘码，跳过")
                continue
            c, side, n = got
            pose = tc.read_pose_stable(api, dType, 5)
            print(f"  · [离纸面 {tip:4.1f}mm] 码({c[0]:7.1f},{c[1]:7.1f})  "
                  f"边长 {side:6.1f}px  （{n} 帧命中，回读 z={pose['z']:.1f}）")
            stops.append((float(c[0]), float(c[1]), float(side)))
            zs.append(float(pose["z"]))

        if len(stops) < 3:
            print(f"\n✗ 只拿到 {len(stops)} 个有效高度，解不出光心（至少 3 个）。")
            print("  吸盘码在低处最清楚（相机对焦在纸面）—— 试试把高度整体压低，"
                  "或把 --frames 调大。")
            return {}

        fit = fit_optical_center(stops)
        if fit is None:
            print("\n✗ 边长没随高度变 —— 拟合退化，解不出光心。")
            return {}
        C, v, se, resid, rho = fit
        slide = float(np.hypot(*v) * (float(rho.max()) - float(rho.min())))
        span = float(rho.max() - rho.min())
        if not np.isfinite(se) or se > MAX_CENTER_SE_PX:
            # ★ 闸门卡在 **se** 上（为什么不用 slide 当闸门见 MAX_CENTER_SE_PX 的
            #   注释）。se 由「杠杆 + 噪声」定，所以只要看它就能判，不必去猜是
            #   ρ 没铺开还是码解歪了 —— 下面把两个因子都摆出来给人自己看。
            print(f"\n✗ 这次光心解不出来: 标准误 ±{se:.1f}px，超过 {MAX_CENTER_SE_PX:.0f}px 就不收。")
            print(f"  （滑动 {slide:.0f}px = |v| {np.hypot(*v):.0f}px × ρ跨度 {span:.3f}；"
                  f"残差均值 {resid.mean():.2f}px）")
            if span < 0.3:
                print(f"  主因像是 **ρ 跨度只有 {span:.3f}**（铺开高度时约 0.4）——"
                      f" 高度没拉开:")
                print(f"    把 --center-tips 铺得更开（比如 "
                      f"{tips[0]:.0f},…,{tips[-1] * 2:.0f}）再跑。")
            if resid.mean() > 1.5:
                print(f"  残差均值 {resid.mean():.2f}px 偏大 —— 可能某一停的码**解歪了**"
                      f"（半个码也会解出个小边长）:")
                print(f"    看上面每停的「边长px」那列，有没有哪一停明显离群；"
                      f"把 --frames 调大再跑。")
            print(f"  ★ 本步**没有**落盘 —— 光心保持原值不动（宁可没有，也不要一个错的）。")
            return {}

        cam_1 = vis.get("cam_height_mm") if isinstance(vis, dict) else None
        print_center(C, v, se, resid, rho, vis.get("s_table"), cam_1, args.center)
        cam_2 = cam_height_from_sizes(rho, zs, z_offset=args.qr_height)
        if cam_2:
            print(f"     ② 这条路给的是 {cam_2:.0f}mm"
                  + (f" —— 和 ① 差 {abs(cam_2 - cam_1) / cam_1 * 100:.0f}%，"
                     f"对得上。" if cam_1 and abs(cam_2 - cam_1) / cam_1 < 0.15
                     else " —— 差得多，先别信 ① 的 k。" if cam_1
                     else f"（① 还没量过，没得比 —— 先跑 --go）"))
        print(f"\n  ★ 下一步: 把 ({args.center[0]:.0f},{args.center[1]:.0f}) 换成 "
              f"({C[0]:.0f},{C[1]:.0f})，闭环那条式子就只剩「按当时高度现算 k_tip」"
              f"这一件事了。")

        return {
            "optical_center_px": [round(float(C[0]), 1), round(float(C[1]), 1)],
            "se_px": round(float(se), 2),
            "slide_px": round(slide, 1),
            "assumed_before": [float(args.center[0]), float(args.center[1])],
            "rho": [round(float(r), 4) for r in rho],
            "stops_tip_mm": [round(t, 1) for t in tips],
            "stops": [[round(a, 1), round(b, 1), round(c, 1)] for a, b, c in stops],
            "resid_max_px": round(float(resid.max()), 3),
            "resid_mean_px": round(float(resid.mean()), 3),
            "cam_height_from_sizes_mm": round(cam_2, 1) if cam_2 else None,
            "slide_vec_px": [round(float(v[0]), 2), round(float(v[1]), 2)],
        }
    finally:
        if home is not None and moved:
            try:
                tc.move_to(api, dType, home,
                           {"x": (-1e9, 1e9), "y": (-1e9, 1e9),
                            "z": (-1e9, 1e9), "r": (-1e9, 1e9)}, args.mode, timeout=20)
                print(f"\n[回位] 已回到原位 ({home['x']:.1f}, {home['y']:.1f}, {home['z']:.1f})")
            except Exception:
                pass
        try:
            dType.SetQueuedCmdStopExec(api)
        except Exception:
            pass
        try:
            dType.DisconnectDobot(api)
            print("[断开] 完成")
        except Exception:
            pass


# ─────────────────────────── 自检 ───────────────────────────
def selftest() -> int:
    """离线自检: 不连相机、不连机械臂，用合成数据把每条算式钉一遍。"""
    bad = 0

    def check(name, cond, extra=""):
        nonlocal bad
        print(f"  {'✅' if cond else '❌'} {name}" + (f"   {extra}" if extra else ""))
        if not cond:
            bad += 1

    print("measure_suction_map.py 自检")
    rng = np.random.default_rng(7)

    # 1) 相似拟合: 已知 s/θ 能不能还原
    s_true, th = 4.617, np.radians(-178.46)
    R = s_true * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    mm = np.array([[29.55, 52.5], [242.55, 52.5], [28.45, 183.5], [242.45, 183.5]])
    px = mm @ R.T + np.array([7.0, -3.0])
    J, resid = fit_similarity(px, mm)
    check("相似拟合 还原 s", abs(np.hypot(*J[:, 0]) - s_true) < 1e-6,
          f"{np.hypot(*J[:, 0]):.6f} vs {s_true}")
    check("相似拟合 残差为 0", resid.max() < 1e-6)

    # 2) 相似拟合 抗噪: 加 1px 噪声后 s 仍应准到 1% 内
    J2, _ = fit_similarity(px + rng.normal(0, 1.0, px.shape), mm)
    check("相似拟合 抗 1px 噪声", abs(np.hypot(*J2[:, 0]) - s_true) / s_true < 0.01,
          f"{np.hypot(*J2[:, 0]):.4f}")

    # 3) 运动拟合: 能不能从 9 个测点还原 A
    A_true = np.array([[-4.61, 0.11], [-0.12, -4.62]]) * 1.5
    home = (200.0, 40.0)
    samples = []
    for dx, dy in ((0, 0), (30, 0), (0, 0), (-30, 0), (0, 0),
                   (0, 30), (0, 0), (0, -30), (0, 0)):
        p = np.array([home[0] + dx, home[1] + dy])
        q = A_true @ (p - np.array(home)) + np.array([1061.5, 549.0])
        samples.append((p[0], p[1], q[0], q[1]))
    A_hat, r = fit_motion(samples)
    check("运动拟合 还原 A", np.allclose(A_hat, A_true, atol=1e-6),
          f"max|Δ|={np.abs(A_hat - A_true).max():.2e}")
    check("运动拟合 残差为 0", r.max() < 1e-6)
    check("运动拟合 样本不足时也返回 2x2", fit_motion(samples[:5])[0].shape == (2, 2))

    # 4) 位姿自变量: 指令 30mm 但实际只走 29mm 时，A 应按**实测**比例给出
    s2 = []
    for dx, dy in ((0, 0), (29.0, 0), (0, 0), (0, 29.0)):
        p = np.array([home[0] + dx, home[1] + dy])
        q = A_true @ (p - np.array(home)) + np.array([1061.5, 549.0])
        s2.append((p[0], p[1], q[0], q[1]))
    A2, _ = fit_motion(s2)
    check("自变量用实测位姿（不走样地还原 A）", np.allclose(A2, A_true, atol=1e-6))

    # 5) describe_axes 的方向/行列式
    d = describe_axes(np.array([[-4.61, 0.0], [0.0, 4.62]]))
    check("describe_axes 侦测镜像", d["det"] < 0, f"det={d['det']:.3f}")
    d2 = describe_axes(np.array([[0.0, -4.6], [4.6, 0.0]]))
    check("describe_axes 正交偏差 0", abs(d2["skew_deg"]) < 1e-9)

    # 6) k ↔ 相机高度 往返
    Z, z = 534.0, 176.0
    k = k_at_height(z, Z)
    check("k = Z/(Z−z)", abs(k - Z / (Z - z)) < 1e-12, f"k={k:.4f}")
    check("相机高度反解往返", abs(cam_height(z, k) - Z) < 1e-9)
    check("k≤1 时无解", cam_height(z, 0.8) is None)
    check("纸面处 k=1", abs(k_at_height(0.0, Z) - 1.0) < 1e-12)

    # 7) 「把吸盘码压在方块上就抓」的偏差比例
    k_tip = k_at_height(56.0, 534.0)
    frac = 1.0 - k_tip / k_at_height(176.0, 534.0)
    check("视差内偏比例 > 20%（说明 ② 非做不可）", 0.2 < frac < 0.4,
          f"{frac * 100:.1f}%")

    # 8) 认领打印件
    cands = suction_size_candidates()
    check("读到候选打印尺寸", len(cands) == 4,
          "  ".join(f"{t}={m:.3f}" for t, m in cands))
    tag, near = identify_sticker(21.5, cands)
    check("认领 30mm 那张", tag == "30mm", f"{tag} {near:.3f}")
    tag2, _ = identify_sticker(18.0, cands)
    check("认领 25mm 那张", tag2 == "25mm", str(tag2))
    check("没有候选时不炸", identify_sticker(20.0, []) == (None, None))

    # 9) k 与打印件尺寸的联合关系（本脚本的核心恒等式）
    #    边长px = s_table × k_qr × 数据区mm  →  反推数据区mm 必须回到原值
    for tag3, data_mm in cands:
        s_room, kq = 4.617, 1.4967
        side_px = s_room * kq * data_mm
        back = side_px / (s_room * kq)
        if abs(back - data_mm) > 1e-9:
            check(f"恒定式往返 {tag3}", False, f"{back} vs {data_mm}")
            break
    else:
        check("恒定式 边长px = s_table × k × 数据区mm 往返一致", True)

    # 10) ② 光心: 合成一条「过光心的射线」，看能不能把 C 还原出来
    #     真值: 光心 C_t、滑动向量 v_t；第 i 停的 ρ 由高度按 k=Z/(Z−z) 定。
    Z_c, qr_off = 287.0, 120.0
    tips_t = [6.0, 12.0, 20.0, 30.0, 40.0, 52.0]
    C_t = np.array([968.0, 521.0])
    v_t = np.array([150.0, -55.0])
    z_qr_t = [t + qr_off for t in tips_t]
    kk_t = [k_at_height(z, Z_c) for z in z_qr_t]
    rho_t = np.array(kk_t) / kk_t[0]
    stops_t = [(*(C_t + v_t * r), 188.0 * float(r)) for r in rho_t]
    st = fit_optical_center(stops_t)
    check("② 至少要 3 个高度", fit_optical_center(stops_t[:2]) is None)
    check("② 高度没变时不硬解", fit_optical_center([(1.0, 2.0, 5.0)] * 4) is None)
    check("② ρ 的跨度够不够（合成数据本身要合法）",
          float(rho_t.max() - rho_t.min()) > 0.3,
          f"ρ {rho_t.min():.3f}~{rho_t.max():.3f}")
    if st is None:
        check("② 合成数据能解出光心", False, "fit 返回 None")
    else:
        C_h, v_h, se_h, res_h, rho_h = st
        check("② 还原光心 C", np.allclose(C_h, C_t, atol=1e-6),
              f"({C_h[0]:.2f},{C_h[1]:.2f}) vs ({C_t[0]:.0f},{C_t[1]:.0f})")
        check("② 还原滑动向量 v", np.allclose(v_h, v_t, atol=1e-6))
        check("② 无噪时残差为 0", res_h.max() < 1e-6)
        check("② 无噪时标准误近 0", se_h < 1e-6, f"{se_h:.2e}")

        # ★ 噪声: C 是**外推**出来的（ρ 只到 1.4，C 在 ρ=0），所以真正要验的不是
        #   「某一次准不准」，而是「报出来的 ±se 是不是这个外推的真实散布」。
        #   一次抽样的误差是随机的，比阈值没意义；跑一批看**散布**才对得上。
        errs, ses = [], []
        for _ in range(80):
            noisy = [(*(C_t + v_t * r + rng.normal(0, 0.8, 2)), 188.0 * r)
                     for r in rho_t]
            sn = fit_optical_center(noisy)
            if sn is None:
                continue
            errs.append(sn[0] - C_t)
            ses.append(sn[2])
        errs = np.asarray(errs)
        emp = float(np.hypot(errs[:, 0].std(ddof=1), errs[:, 1].std(ddof=1)) / np.sqrt(2))
        mean_se = float(np.mean(ses))
        check("② 外推的 C 没有系统性偏移（无偏）",
              np.linalg.norm(errs.mean(axis=0)) < 1.0,
              f"平均偏差 {np.linalg.norm(errs.mean(axis=0)):.2f}px")
        check("② 报的 ±se 和实测散布对得上（0.5~2×，否则 se 是假的）",
              0.5 < mean_se / emp < 2.0,
              f"报 {mean_se:.2f}px / 实测散布 {emp:.2f}px")

        # ★ 闸门为什么卡在 se 而不是「滑了多少像素」—— 两头都要验:
        #   (a) 码停在光心附近 → 几乎不滑，但数据点本来就压在 C 上，C 反而最准。
        #       这种不该被毙掉（拿 slide 当闸门就会误杀）。
        v_near = np.array([8.0, -5.0])
        st_near = fit_optical_center(
            [(*(C_t + v_near * r), 188.0 * float(r)) for r in rho_t])
        check("② 码停在光心附近（几乎不滑）也能解出 C —— 所以不能拿 slide 当闸门",
              st_near is not None and np.hypot(*(st_near[0] - C_t)) < 1e-6,
              "" if st_near is None else
              f"|v|={np.hypot(*st_near[1]):.1f}px，C 仍精确")

        #   (b) ρ 没铺开 → 杠杆差 → se **必须**顶上去，闸门才拦得住。
        #       同一组噪声，只改 ρ 的跨度，两个 se 要拉开量级。
        noise_t = rng.normal(0, 0.8, (len(rho_t), 2))

        def _mk(rhos):
            return [(*(C_t + v_t * r + n), 188.0 * float(r))
                    for r, n in zip(rhos, noise_t)]

        se_wide = fit_optical_center(_mk(rho_t))[2]
        se_narrow = fit_optical_center(_mk(np.linspace(1.0, 1.05, len(rho_t))))[2]
        check("② ρ 铺不开时 se 顶上去（所以 se 能当闸门）",
              se_narrow > 3.0 * se_wide,
              f"窄 {se_narrow:.1f}px vs 宽 {se_wide:.1f}px")
        check("② 铺不开的那次会被 MAX_CENTER_SE_PX 拦下",
              se_narrow > MAX_CENTER_SE_PX > se_wide,
              f"{se_wide:.1f} < {MAX_CENTER_SE_PX:.0f} < {se_narrow:.1f}")

        # 11) ② 的独立校验: 用边长比反解相机高度，必须回到 Z_c（含 qr_offset！）
        cam2 = cam_height_from_sizes(rho_t, tips_t, z_offset=qr_off)
        check("② 边长比反解相机高度（含码高偏移）",
              cam2 is not None and abs(cam2 - Z_c) < 0.5,
              f"{cam2:.1f} vs {Z_c:.1f}")
        cam_bad = cam_height_from_sizes(rho_t, tips_t)          # 忘了加偏移
        check("② 漏掉码高偏移就会差一个 qr_height（所以必须加）",
              cam_bad is not None and abs(cam_bad - Z_c) > qr_off * 0.9,
              f"{cam_bad:.1f}")

    # 12) ③ 控制律的系数: 用对了一步到位；用成 k_tip 就停在 factor·u_c（固定错位）
    #     ★ 2026-09-20 重推控制律时新加。钉住「系数错 ≠ 收敛慢」这个结论 ——
    #       它决定了 ③ 的自检必须拿「离光心多远」当自变量，而不是迭代次数。
    Zc3, obj3, qroff, tip3, s3, scale3 = 323.3, 26.0, 120.0, 15.0, 4.6, 1.15
    zqr3 = tip3 + qroff
    k_qr3, k_cube3 = k_at_height(zqr3, Zc3), k_at_height(obj3, Zc3)
    c_ok = k_qr3 / k_cube3
    A3 = s3 * k_qr3 * scale3 * np.eye(2)            # A = s·k(码)·机械臂刻度
    C3 = np.array([960.0, 540.0])
    u_c3 = np.array([70.0, -40.0])                  # 方块离光心的真实距离 mm
    cube3 = C3 + s3 * k_cube3 * u_c3                # 方块在画面里（静态）

    def _place(c_used, n=3):
        """从光心正下方开始，按 δ = A⁻¹·[c·(方块−C) − (码−C)] 走 n 步。"""
        u = np.zeros(2)
        for _ in range(n):
            qr = C3 + s3 * k_qr3 * u
            u = u + scale3 * (np.linalg.inv(A3) @ (c_used * (cube3 - C3) - (qr - C3)))
        return u

    u_ok = _place(c_ok)
    check("③ 系数用对 → 一步就落到方块上", np.allclose(u_ok, u_c3, atol=1e-9),
          f"残差 {np.linalg.norm(u_ok - u_c3):.1e} mm")
    c_bad = k_at_height(tip3, Zc3)                  # 旧版写的 k_tip
    u_bad = _place(c_bad)
    check("③ 系数用成 k_tip → 停在固定错位（不是收敛慢）",
          np.allclose(u_bad, (c_bad / c_ok) * u_c3, atol=1e-9),
          f"落在 {c_bad / c_ok:.3f}·u_c，偏 {np.linalg.norm(u_bad - u_c3):.1f}mm")
    check("③ 错位跟「离光心多远」成正比（近光心反而看着准）",
          np.linalg.norm(_place(c_bad, n=1) - u_c3) / np.linalg.norm(u_c3) > 0.3,
          f"{(1 - c_bad / c_ok) * 100:.0f}% × 离光心距离")

    # 13) 卷尺入口（--cam-height）: 顶掉 |A|/s_table，并把机械臂刻度偏差报出来
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        info_r = use_ruler_cam_height({"k_qr": 1.8906}, 323.3, 135.0, 4.5788)
        ctl = print_control(A3, {"cam_z": Zc3, "k_tip": k_at_height(tip3, Zc3)},
                            zqr3, obj3, C3, s3)
    check("卷尺顶掉 |A|/s_table 解出来的 k",
          abs(info_r["k_qr"] - k_at_height(135.0, 323.3)) < 1e-9,
          f"{info_r['k_qr']:.4f}")
    # （落盘/上报的数都 round 到 4 位，所以比对容差按 1e-3 —— 别用 1e-9）
    check("卷尺入口顺手报出机械臂 XY 刻度偏差",
          abs(info_r["arm_xy_scale_from_A"]
              - 1.8906 / k_at_height(135.0, 323.3)) < 1e-3,
          f"报 1mm 实走 ×{info_r['arm_xy_scale_from_A']:.4f}")
    check("控制律报的 c 是 k(码)/k(方块顶)，不是吸嘴处的 k",
          ctl.get("c_coef") is not None and abs(ctl["c_coef"] - c_ok) < 1e-3
          and abs(ctl["c_coef"] - k_at_height(tip3, Zc3)) > 0.4,
          f"c={ctl.get('c_coef')}  而 k_tip={k_at_height(tip3, Zc3):.4f}")

    print(f"\n{'✅ 全部通过' if bad == 0 else f'❌ {bad} 项失败'}")
    return 1 if bad else 0


# ─────────────────────────── CLI ───────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(
        description="量「机械臂 mm ↔ 画面像素」的映射（相对运动的地基），并报视差放大倍数 k")
    ap.add_argument("--go", action="store_true",
                    help="★ ① 真动机械臂: 空走几步量映射（只动 XY，不下降不吸）")
    ap.add_argument("--go-center", action="store_true",
                    help="★ ② 真动机械臂: 实测光心（**只升降、不横移**；要求桌面清空）")
    ap.add_argument("--center-tips", default=None,
                    help="② 停哪几个高度（吸嘴离纸面 mm，逗号分隔），默认 "
                         + ",".join(f"{v:.0f}" for v in DEFAULT_CENTER_TIPS))
    ap.add_argument("--image", default=None, help="用一张已有图片代替相机（不动臂）")
    ap.add_argument("--cam", type=int, default=None, help="摄像头编号，默认自动挑外接的")
    ap.add_argument("--focus", type=int, default=qv.FOCUS_LOCK,
                    help=f"**锁死手动焦距**（UVC 值，小=对远、大=对近）。默认 "
                         f"{qv.FOCUS_LOCK}，和 ③/④ 同一个值 —— 它在「吸盘码 + 纸面 "
                         f"4 码一帧全中」那段窗口里（见 qr_vision.lock_focus 的实测表）。"
                         f"给负数 = 不锁（等自动对焦）。"
                         f"★ 那张表是在**悬停高度**量的；① 在 z_tip=15（码离相机远 "
                         f"41mm），若重跑 ① 时吸盘码解不出，先把 --focus 调小些试。")
    ap.add_argument("--frames", type=int, default=ACCUM_FRAMES,
                    help=f"每停一次积累多少帧取并集（默认 {ACCUM_FRAMES}）。"
                         f"★ 锁了焦距之后这是**余量**不是主力（原来靠它堆出吸盘码那一张，"
                         f"见 locate 的说明）；不锁焦距（--focus 负数）时才回到堆帧数那条路")
    ap.add_argument("--step", type=float, default=DEFAULT_STEP_MM,
                    help=f"每个方向空走多远 mm（默认 {DEFAULT_STEP_MM:.0f}）")
    ap.add_argument("--qr-height", type=float, default=DEFAULT_QR_HEIGHT_MM,
                    help=f"吸盘码中心到吸嘴尖的高度 mm（默认 {DEFAULT_QR_HEIGHT_MM:.0f}）")
    ap.add_argument("--hover", type=float, default=None, nargs="?", const=DEFAULT_HOVER_MM,
                    help=f"先升到「方块顶面 + 这么多 mm」再量（默认不带; 不带 --hover 就用"
                         f"当前高度）。带 --hover 不带数字 = {DEFAULT_HOVER_MM:.0f}mm")
    ap.add_argument("--obj-h", type=float, default=DEFAULT_OBJ_H_MM,
                    help=f"桌面上最高的障碍物 mm —— **要你自己声明**，脚本看不出"
                         f"（默认 {DEFAULT_OBJ_H_MM:.0f}，和 step4/main.py 一致）。"
                         f"桌面已清空就写 --obj-h 0")
    ap.add_argument("--table-z", type=float, default=None,
                    help="纸面 Z，默认读 output/pick_test_result.json 或 robot_points.json")
    ap.add_argument("--cam-height", type=float, default=None, metavar="MM",
                    help="★ 拿卷尺量的「纸面→相机镜头」高度。给了就以它为 k 的基准"
                         "（k = Z/(Z−z_qr)），① 从 |A|/s_table 解出的那个降级成"
                         "**校核**（两者之比正好是机械臂 XY 的刻度误差）。"
                         "不给就还用 |A|/s_table（混着机械臂 XY 缩放，实测偏 10%）")
    ap.add_argument("--center", default=None,
                    help="光心像素，格式 'x,y'。默认用画面中心（**没实测过**）")
    ap.add_argument("--limits", default=None,
                    help='覆盖软限位，如 "x:50:380,y:-310:310"')
    ap.add_argument("--mode", default="movl", choices=("movl", "movj"),
                    help="movl 直线（默认，XY 微调时 Z 不会跑）")
    ap.add_argument("--port", default=None, help="串口，默认自动查找")
    ap.add_argument("--yes", action="store_true", help="跳过确认（升高/空走的回车）")
    ap.add_argument("--clear-alarms", action="store_true",
                    help="报警清不掉时强制清（**确认机械臂已脱离卡住状态**再用；"
                         "丢步报警一律不自动清，见 step2）")
    ap.add_argument("--out", default=None, help=f"落盘路径，默认 {SUCTION_MAP_JSON}")
    ap.add_argument("--selftest", action="store_true", help="离线自检，不连相机不连臂")
    return ap


def load_prev_out(out_path: str | None) -> dict:
    """读上一份落盘（①/② 的产物）。没有 / 坏了都当"没有"，不抛错。"""
    p = Path(out_path) if out_path else SUCTION_MAP_JSON
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    args = build_argparser().parse_args()
    if args.selftest:
        return selftest()

    if args.go and args.go_center:
        print("✗ --go（① 横走）和 --go-center（② 升降）是两次不同的运动，分开跑。")
        return 2

    print("=" * 72)
    print("  吸盘码 → 相对运动映射（① 方向 + 比例，② 光心）")
    print("=" * 72)

    # ★ 动臂的两道**静态**闸门放在碰相机之前 —— 目的是「在花 30 帧解码、更别说
    #   动任何一步之前就拒掉」。原来放在相机后面，相机没插好会把真正的错因
    #   （在管道里跑 / 拿固定图片量）盖掉，报出来的话就指错方向了。
    #   （「看不看得见吸盘码」那道闸门要等视觉结果，只能留在后面。）
    if args.go or args.go_center:
        flag = "--go" if args.go else "--go-center"
        if args.image:
            print(f"\n✗ --image 和 {flag} 是矛盾的: 用固定图片就看不到机械臂动过之后的画面。")
            print("  这一步量的就是**动完之后的实时画面**，得开相机。")
            return 2
        if args.go_center:
            if args.center_tips is None:
                args.center_tips = list(DEFAULT_CENTER_TIPS)
            else:
                try:
                    args.center_tips = [float(t) for t in str(args.center_tips).split(",")
                                        if t.strip()]
                except ValueError:
                    print(f"\n✗ --center-tips 格式应是 '6,12,20'，收到 {args.center_tips!r}")
                    return 2
            if len(args.center_tips) < 3:
                print(f"\n✗ --center-tips 至少要 3 个高度（收到 {len(args.center_tips)} 个）"
                      f" —— 解光心靠高度的**伸展**，两个点判不出残差。")
                return 2
        if not sys.stdin.isatty():
            print(f"\n✗ {flag} 会动机械臂，必须在交互式终端里跑。")
            print("  （现在 stdin 不是终端 —— 被管道/重定向/后台了，一步都不动。）")
            return 1

    layout = load_paper_layout(PAPER_JSON)
    suction = load_suction_code()
    if suction is None:
        print("✗ 读不到吸盘码内容 —— 先生成:\n    python3 tools/gen_suction_qr.py")
        return 2
    want = list(layout.codes) + [suction]
    print(f"要认的码: {' '.join(layout.codes)} + {suction}")

    # ── 光心 ──
    # ★ 三档优先级，越靠前越可信: ① --center 明写 → ② 上次 ② 实测过（落盘里
    #   optical_center_measured=true）→ ③ 画面中心（**纯假设**）。
    #   为什么要读上一份: ① 和 ② 是分开跑的，先跑 ② 量出 C、再跑 ① 量 A，
    #   要是 ① 每次都把 C 打回 (960,540)，那 ② 的结果就白量了。
    prev = load_prev_out(args.out)
    prev_center = (prev.get("optical_center_px")
                   if prev.get("optical_center_measured") else None)
    center_given = args.center is not None
    if center_given:
        try:
            center = tuple(float(v) for v in args.center.split(","))
        except ValueError:
            print(f"✗ --center 格式应是 'x,y'，收到 {args.center!r}")
            return 2
        if len(center) != 2:
            print(f"✗ --center 要两个数，收到 {args.center!r}")
            return 2
        center_src = "你在 --center 里给的"
    elif prev_center:
        center = tuple(float(v) for v in prev_center)
        center_src = "上次 ② 实测的（落盘里带回来的）"
    else:
        center = (960.0, 540.0)
        center_src = "**画面中心，只是一种假设，没实测**"
    args.center = center

    detector = cv2.QRCodeDetector()
    cap = None
    dec = None
    # ★ 必须在分支**之前**先定义成 None: --image 那条路不走 sample_suction，
    #   下面 try 里的 `if suction_meas is not None` 会直接 NameError。
    suction_meas = None
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f"✗ 读不到图片 {args.image}")
            return 2
        print(f"[图片] {args.image}  {frame.shape[1]}x{frame.shape[0]}")
    else:
        cap = qv.open_camera(args.cam, focus=args.focus)
        if not cap.isOpened():
            print("✗ 打不开摄像头。别的程序占着？/dev/video* 在不在？")
            return 1
        # 先拍一帧只为**判全黑** —— 相机开了却什么都没拍到，和"机位不对"的下一步
        # 动作完全相反，所以要在花 30 帧去解码**之前**就分开。
        frame = grab(cap, flush=8)
        if frame is None:
            print("✗ 相机读不到帧。")
            return 1
        print(f"[相机] 帧 {frame.shape[1]}x{frame.shape[0]}  亮度 {frame.mean():.1f}/255")
        if float(frame.mean()) < 5.0:
            print(f"\n✗ 画面几乎全黑（平均亮度 {frame.mean():.1f}/255）—— 相机开着，"
                  f"但什么都没拍到。")
            print("  常见原因: 镜头被盖着 / 隐私挡片还开着 / 笔记本合着盖 / 外接相机没插好。")
            print("  当前系统里可见的摄像头:")
            print(qv.describe_cameras())
            print("  「俯视台面」用的是外接相机 —— 上面要是只剩 Integrated_Webcam，"
                  "就是它没插好。")
            return 1
        # ★ 相机这条路**分两趟**量，因为两拨码要的东西根本不一样:
        #   · 纸面 4 码贴在**对焦面**上，帧帧都解得出 → locate（凑齐就收工，快）
        #   · 吸盘码在**另一个焦面**上，时解时不解 → sample_suction（收满帧取中位数）
        #   以前是把吸盘码一起塞进 locate 的 want，两头都坏: locate 得白等它 30 帧，
        #   而它交出来的只是**某一帧**的四边形（实测同场景真值 146.5~150.1px，
        #   它给出 188.7px）。所以现在 locate 只找纸面码，吸盘码单独走中位数那条。
        paper_want = list(layout.codes)
        print(f"  等对焦 + 积累最多 {args.frames} 帧（纸面 4 码）…")
        dec, frame = locate(cap, detector, paper_want,
                            focus_goal=paper_want, frames=args.frames)
        sm = sample_suction(cap, detector, suction,
                            focus_goal=paper_want, frames=args.frames)
        if sm is not None:
            suction_meas = (sm[0], sm[1], sm[2])

    try:
        if dec is None:                      # --image 那条路: 单帧就好了
            dec, _ = detect_sweep(frame, detector, want=want)
        if suction_meas is not None:
            print(f"\n  吸盘码按**多帧中位数**取: 边长 {suction_meas[1]:.1f}px"
                  f"（{suction_meas[2]} 帧命中）—— 不是某一帧的四边形")
        vis = analyse_frame(dec, layout, suction, suction_meas)
        print(f"\n[二维码] 解出 {len(want) - len(vis['missing'])}/{len(want)}"
              + (f"   缺: {' '.join(vis['missing'])}" if vis["missing"] else ""))
        print_vision(vis, layout, suction)
        print(f"\n  光心用 ({center[0]:.0f}, {center[1]:.0f}) —— {center_src}")

        motion = {}
        center_info = {}
        if args.go:
            if vis["s_table"] is None or vis["suction"] is None:
                print("\n✗ 桌面比例或吸盘码缺一个 —— 先调机位，别动臂。")
                return 2
            motion = run_arm(args, layout, suction, vis, detector, cap)
        elif args.go_center:
            if vis["suction"] is None:
                print("\n✗ 没看到吸盘码 —— ② 量的就是它，先调机位，别动臂。")
                return 2
            # ★ 把 ① 的 cam_height_mm 递给 ② 当独立校验 —— 它在落盘的 motion 段里，
            #   不在 vis 里（vis 只有视觉那一半）。不接这一步的话 cam_1 永远是 None，
            #   那条「两条路对得上」的交叉验证就静默失效了。
            ctx = dict(vis)
            ctx["cam_height_mm"] = (prev.get("motion") or {}).get("cam_height_mm")
            center_info = run_center(args, layout, suction, ctx, detector, cap)
        else:
            print("\n（没给 --go / --go-center，机械臂一步没动。）")
            print("  · 要量「mm ↔ 像素」→ 加 --go")
            print("  · 要实测光心 C     → 加 --go-center（只升降，要求桌面清空）")
            print("  只拍一张**分不出** k: 放大倍数 = k × (吸盘码印多大 ÷ 纸码印多大)。")

        # ── 落盘: ① 和 ② 分开跑，所以这里必须**合**而不是**盖** —— 跑 ②
        #    不能把 ① 的 motion 抹掉，反之亦然。 ──
        if center_info:
            C_out, C_measured = center_info["optical_center_px"], True
        elif prev_center:
            C_out, C_measured = list(prev_center), True
        else:
            C_out, C_measured = list(center), False

        out = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "source": "tools/measure_suction_map.py",
            "note": ("相对运动那条路的地基: A = d(像素)/d(机械臂mm)。**不是**手眼标定，"
                     "不读 hand_eye_matrix.json。k_qr 是吸盘码高度的视差放大倍数，"
                     "用来把「吸盘码的影子」换成「吸嘴的真实落点」。"),
            "suction_content": suction,
            "vision": vis,
            "motion": motion or prev.get("motion"),
            "center": center_info or prev.get("center"),
            "optical_center_px": C_out,
            "optical_center_measured": C_measured,
            "qr_height_mm": args.qr_height,
            # ★ 存下量这份地图时相机的焦距锁在哪儿 —— 和 C/k 同性: 换机位就得重量。
            #   负数 = 当时没锁（靠自动对焦）。③/④ 只把它当**记录**用，
            #   真正的默认值是 qr_vision.FOCUS_LOCK。
            "focus_lock": args.focus,
        }
        ensure_output_dir()
        path = Path(args.out) if args.out else SUCTION_MAP_JSON
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ 已落盘: {path}")
        if out["motion"] is None:
            print("   （这一份只有视觉部分 —— motion 是 null，还没量 mm↔像素）")
        if not C_measured:
            print("   （光心还是 (960,540) 那个假设 —— 跑 --go-center 把它实测出来）")
    finally:
        if cap is not None:
            cap.release()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[已中止]")
        sys.exit(130)
