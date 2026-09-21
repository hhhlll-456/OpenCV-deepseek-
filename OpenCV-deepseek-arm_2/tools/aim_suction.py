#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aim_suction.py —— ③ 闭环: 把吸嘴**瞄到方块正上方**（相对运动那条路的最后一步）
==============================================================================
前两步都是「量一次、存下来」:
  ① tools/measure_suction_map.py --go          → A = d(像素)/d(机械臂mm)，以及 k
  ② tools/measure_suction_map.py --go-center   → 光心 C
本脚本把它们**用起来**: 每走一步都重拍重算，直到吸嘴压在方块上。

控制律（和 ①/② 报的是同一条）:
    δ_机械臂 = A⁻¹ · [ c·(方块像素 − C) − (吸盘码像素 − C) ]
    c = k(Z_cam, 吸盘码那一层) ÷ k(Z_cam, 方块顶面那一层)

★ 为什么这条律**只吃像素**（不读 hand_eye_matrix.json、也不算绝对坐标）:
  式子里的每一项要么是常数（A⁻¹、C、c），要么是**当帧现拍**的像素。δ 是个
  **增量** —— 指令是「现在的位姿 + δ」，所以机械臂的绝对精度、零点漂没漂、
  桌子摆在哪，全都进了那两项像素里、自己抵消掉了。这正是当初加吸盘码的目的
  （见 measure_suction_map 开头那段）。

★ 为什么**不需要纸面 4 码**（少一个失败模式）:
  律本身只用「方块像素」「吸盘码像素」「光心」。纸面 4 码在这里只干两件事:
  把像素翻译成毫米给人看、以及顺带给出 ppm 让方块识别有个尺度。纸被挡住/
  挪走，照样能瞄 —— 这是 ③ 和 step2/step3 那条路**根本不同**的地方。

★★ 这条律是「一步解」，不是「慢慢逼近」——所以系数 c 不能错:
  在纸面 mm 里展开一遍（w_c=方块位置, u=吸嘴当前位置, σ=机械臂报数刻度）:
      δ = (w_c − u)/σ      →  u_new = u + σ·δ = **w_c**，一步就落在方块上。
  系数用错 factor 倍（比如错用 k_tip）时，不动点变成 u* = factor·w_c ——
  机械臂**停在那个固定错位上不再动**，不是收敛慢。而且错位 = (1−factor)·|w_c|
  **跟「方块离光心多远」成正比**: 正巧在光心附近的方块看着挺准，越靠边越离谱。
  这就是 selftest 里那三条用「离 C 的距离」当自变量的原因。

★ 为什么每步都要**重新量**，不能一次算完就不管了:
  · A 有 ~2px 残差、机械臂有背隙、C 有 ±3.6px 的标准误 —— 一步到不了位。
  · 吸嘴一旦升降，k(z_qr) 就变，qr_px 跟着挪（纯 Z 升降也会挪！下面
    predict_after_descend 就是这个: 它算出「只降不横移」之后 δ 变成多少）。
  · 相机/纸被碰过就更不用说。重量的代价只是一张照片。

用法:
  python3 tools/aim_suction.py --color red              # 只拍一张报数（不动臂）
  python3 tools/aim_suction.py --color red --image a.jpg # 用已有图片报数
  python3 tools/aim_suction.py --color red --go         # ★ 真动: 闭环瞄过去
  python3 tools/aim_suction.py --selftest               # 离线自检（不连相机不连臂）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # 同目录的 measure_suction_map
sys.path.insert(0, str(HERE.parent / "src"))

import color_vision as cvis                                                 # noqa: E402
import measure_suction_map as msm                                           # noqa: E402
import qr_vision as qv                                                      # noqa: E402
from dobot_sdk import find_port, load_sdk                                   # noqa: E402
from paths import SUCTION_MAP_JSON                                          # noqa: E402
from qr_vision import (detect_sweep, load_paper_layout, load_suction_code,   # noqa: E402
                       open_camera, quad_center, wait_for_focus)

# ═════════════════════════════ 可调参数 ═════════════════════════════
DEFAULT_TOL_MM = 0.3          # 律说「再走不到这么多」就算到位
# ★ 单步钳幅: 地图要是坏的，别让它一步甩出去。
#   ★ 2026-09-20 从 60 放宽到 80: 实测首步要 62.11mm，被 60 钳掉一点 → 逼出**第二次**
#     迭代，而第二次的测量已经被吸嘴压上来的遮挡弄脏了（方块像素凭空漂 11mm）。
#     悬停高度上没有任何障碍（z 下限已抬到方块顶面之上），一步直走是安全的 ——
#     宁可一次走完那次**干净**的测量，也别为了钳幅去讨第二次脏测量。
MAX_STEP_MM = 80.0
# ★★ 「走完一整步还是看不见 = 够准了」的闸值（mm）。吸嘴压到方块上时，白色贴纸+
#   黑色支架必然把目标色挡掉 —— 这是**几何必然**，越迭代越看不见，最后那几毫米
#   永远量不准。所以: 只要上一次是**真量出来的**、并且已经朝它走完了律要的距离，
#   之后目标消失就判为「够准了、可以收工」。
#   ★ 闸的**是「走完之后预计还差多少」= 上次读数 − 真走了多少**，不是上次读数本身。
#     ★ 2026-09-20 实测踩到的坑: 原判据写的是 `上次读数 ≤ 10`，于是
#       「量到 14.67mm → 朝它走满 14.67mm → 目标被挡掉」被判成**失败** ——
#       可那正是**到了**的样子: 走完了才被自己挡住。量到的距离越大，越说明当时
#       还没到，可它同时也说明**这一步走得越远**，两个效应正好抵消。
#       拿 `14.67 ≤ 10?` 去卡，等于把「走得越远」读成「离得越远」—— 反了。
#   ★ 它**不是**「看不见 ⇒ 我到了」: 那是个不可证伪的判据（灯变了、方块被挪走、
#     地图过期、颜色配错，全都会「看不见」），拿它当到达凭据等于盲扎。
#     这里的逻辑是**有界的**: 触发前必须先有一次**真的量出来**的读数；
#     而且被**钳幅**打断的那一步照样保守（量到 120、只敢走 80 → 预计还差 40 → 照样中止）。
#   给负数 = 关掉这条，退回「看不见就中止」（原来那条最保守的行为）。
LOST_OK_MM = 10.0
MAX_ITER = 8                  # 迭代上限（正常 2~3 步就收敛）
DEFAULT_SETTLE_S = 0.6        # 停稳再拍（和 measure_suction_map 一致）
SPEED_MM_S = 40.0
ACC = 40.0
RATIO = 30.0
# ★ 平移的安全余量: 和 ① 一样，按**方块顶面**算，不是按纸面。
#   ③ 只横移不下降，所以这条闸门是它的**唯一**防撞线。
# ════════════════════════════════════════════════════════════════════


# ─────────────────────── 纯计算（可离线自检） ───────────────────────
def load_map(path=None) -> tuple[dict, str | None]:
    """
    读 ①/② 的落盘，交出闭环需要的三样: A⁻¹、光心 C、相机高度 Z_cam。

    缺任何一样就返回 ( {}, 人话原因 ) —— **不兜底、不猜**。
    ★ 为什么这里不兜底（和 color_vision.load_robot_matrix 同一个理由）:
      拿一个猜的地图继续跑，机械臂会**走错但一路不报错** —— 这是全项目最危险
      的失败方式。缺东西就说缺东西，让人回去补 ①/②。
    """
    p = Path(path) if path else SUCTION_MAP_JSON
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, f"读不到地图 {p} —— 先跑 ①: measure_suction_map.py --go"
    mo = d.get("motion") or {}
    A_inv = mo.get("A_inv_row_major")
    if not A_inv:
        return {}, "地图里没有 A_inv —— 先跑 ①: measure_suction_map.py --go"
    cam_z = mo.get("cam_height_mm")
    if not cam_z:
        return {}, ("地图里没有相机高度 —— 加 --cam-height 323（卷尺量的"
                    "纸面→镜头）重跑 ①")
    if not d.get("optical_center_measured"):
        return {}, ("光心还是 (960,540) 那个**假设**，没实测过 —— 先跑 ②: "
                    "measure_suction_map.py --go-center（只升降，要求桌面清空）")
    C = d.get("optical_center_px")
    if not C:
        return {}, "地图里没有光心 —— 先跑 ②: measure_suction_map.py --go-center"
    return {"A_inv": np.asarray(A_inv, dtype=float), "C": np.asarray(C, dtype=float),
            "cam_z": float(cam_z), "raw": d,
            # ★ ① 量 A 时码挂在哪一层（老地图没这个键 → None）。给 rescale_A_inv 用。
            "map_z_qr": (float(mo["z_qr_mm"]) if mo.get("z_qr_mm") is not None
                         else None)}, None


def rescale_A_inv(A_inv, map_z_qr, z_qr_now: float, cam_z: float):
    """
    把地图里的 A⁻¹ 从「① 量它时那个高度」折算到「这次真正用的高度」。

    ★ 为什么必须折: A = d(像素)/d(机械臂mm) 里含 k(z_qr)=Z_cam/(Z_cam−z_qr) ——
      码挂得越高，画面里动得越快。① 在 z_tip=15（z_qr=135, k=1.718）量的，
      而 ③/④ 是在**悬停高度**上用的（z_tip≈56 → z_qr≈176, k=2.197）:
        g = k(176)/k(135) = 1.279  → A 偏小 28%，每步超调 28%。
      实测后果: 起步差 50mm 时，不折要 **5 步**才进 0.3mm，折了 **1 步**。

    ★ 但折不折都**落得对**: 律里 (码像素−C) 和 c·(方块像素−C) 带的是同一个
      k(z_qr_now)，δ = A⁻¹·s·k(z_qr_now)·(w_c−u)，δ=0 仍在 u=w_c。
      所以这是**速度**问题，不是对错问题 —— 折它只是别白多拍四张照片、
      少让背隙积一点。
    ★ 老地图没记 z_qr_mm → 原样返回（**不猜高度**）: 猜错了同样是只影响步长，
      但没必要在一个能读出来的数上引入猜测。调用方会把这件事说出来。
    """
    if map_z_qr is None or not cam_z:
        return np.asarray(A_inv, dtype=float), None
    g = msm.k_at_height(z_qr_now, cam_z) / msm.k_at_height(map_z_qr, cam_z)
    return np.asarray(A_inv, dtype=float) / g, g


def c_coef(cam_z: float, z_qr: float, obj_h: float) -> float:
    """
    闭环那条律的系数 c = k(吸盘码那一层) ÷ k(方块顶面那一层)。

    ★ 为什么是**这两个**高度（推导见 measure_suction_map.print_control）:
      A 量的是「码在画面里动多快」，所以 A⁻¹ 的 px→mm 因子天生带 k(z_qr)；
      而方块是**顶面**朝着相机，它的视差按 k(obj_h) 走。两个一除就是 c。
    ★ 为什么不能拿吸嘴处的 k（k_tip）顶替: 悬停 15mm 时 k_tip≈1.05，而正确值
      ≈1.58 —— 差一半，后果是固定的错位（见本文件开头那段）。
    """
    return msm.k_at_height(z_qr, cam_z) / msm.k_at_height(obj_h, cam_z)


def aim_delta(A_inv, C, qr_px, cube_px, c) -> np.ndarray:
    """一条式子: 该走多少（机械臂报的 mm）。输入全是当帧现拍的像素。"""
    A_inv = np.asarray(A_inv, dtype=float)
    C = np.asarray(C, dtype=float)
    return A_inv @ (c * (np.asarray(cube_px, dtype=float) - C)
                    - (np.asarray(qr_px, dtype=float) - C))


def clamp_step(delta: np.ndarray, max_step: float = MAX_STEP_MM):
    """单步钳幅 → (钳过的 δ, 缩放系数)。返回 1.0 表示没钳。"""
    n = float(np.linalg.norm(delta))
    if n <= max_step or n <= 0.0:
        return delta, 1.0
    return delta * (max_step / n), max_step / n


def predict_after_descend(delta_now, z_qr: float, dz: float, cam_z: float):
    """
    纯算: 吸嘴**只降 dz**（XY 一个不动）之后，律会要求的新 δ（机械臂 mm）。

    ★ 结果干净得意外 —— 只是被 k 比缩放了一下:
        δ = A⁻¹·s·k(z_qr)·(w_c − u)      w_c=方块的地面位置, u=吸嘴的地面位置
        下降只改 k(z_qr)；w_c、u 是**地面**位置，纯 Z 平移一个都不动
        → δ_aft = (k_aft/k_now) · δ_now
    ★ 于是最该说出口的是这条（也正是 ④ 敢「瞄完就盲降」的根据）:
        **收敛了（δ≈0）→ 降下去 δ 还是 ≈0 —— 下降不会把瞄准弄坏。**
      没收敛时下降要补的也只是 δ_now 的一个零头（k_aft<k_now，所以更小）。
    ★ 旧版这里写的是 (k_aft/k_now−1)·(qr_px−C): 那是个**像素**量（码在画面里
      滑了多远），却贴着「mm」的标签报出去，还被说成「下降后会偏的毫米数」。
      两处都对不上 —— 它跟码**自己**在哪(u)有关，跟「离方块还差多少(w_c−u)」
      无关，所以收敛时它**不为零**，而真正要补的恰恰是零。现在一律按 mm 报。
    """
    z_after = z_qr - dz
    # ★ 两头都要拦: 现在这个高度就在相机平面上/以上（地图坏了），
    #   或**升**到相机平面上/以上（k 发散，下降预测无从谈起）。
    if not cam_z or cam_z <= z_qr or z_after >= cam_z:
        return None
    k_now, k_aft = msm.k_at_height(z_qr, cam_z), msm.k_at_height(z_after, cam_z)
    return (k_aft / k_now) * np.asarray(delta_now, dtype=float)


def solve_loop(sample, move, A_inv, C, c, tol_mm: float = DEFAULT_TOL_MM,
               max_step: float = MAX_STEP_MM, max_iter: int = MAX_ITER,
               log=print, lost_ok_mm: float | None = LOST_OK_MM):
    """
    闭环本体。**故意做成两个回调**，这样离线自检能塞一台假机械臂进去跑。

      sample() -> (qr_px, cube_px) | None   每次调用都**重新量**（这才是闭环）
      move(delta, step_no) -> bool          走一步；False = 走位失败
    → (每步记录, 是否收敛, 原因)

    ★ 判据用「这一步律要求走多少」而不是「像素差多少」: δ 本身就已经是
      像素差除以 A 换成 mm 的结果 —— 它就是「还差几毫米」，单位正是机械臂
      要动的那个单位。拿像素当判据还得再挑一个阈值，多一层猜。

    ★★ lost_ok_mm —— 「走完一步还是看不见 = 够准了」。返回值三态，调用方要分清:
         ok=True,  why=None   → 真的收敛（raw ≤ tol）
         ok=True,  why="…"    → **靠这条收的**: 目标被吸嘴挡掉了。调用方该把它当
                                **带警告的成功**说出来，别悄悄当成正常收敛。
         ok=False, why="…"    → 真失败（从头就没量到 / 走位失败 / 步数用光）

    ★ 为什么必须有这条: 吸嘴压到方块上方时，白色贴纸+黑色支架**必然**挡掉目标色，
      越迭代越看不见 —— 最后那几毫米是**几何上量不到**的，不是调参数能救的。
      没有这条，闭环的结局永远是「量不到了 → 中止」，④ 因此一步都降不下去。
    ★ 闸的是**「走完之后预计还差多少」= 上次读数 − 上一步真走的距离**，不是上次读数
      本身（★ 后者是 2026-09-20 实测踩的坑: 量到 14.67mm、走满 14.67mm、目标被挡掉，
      反被判失败 —— 那恰恰是走到了。详见 LOST_OK_MM）。
    ★ 为什么**不能**简化成「目标看不见 ⇒ 到位了」: 那不可证伪 —— 地图过期、
      方块被挪走、灯变了、颜色配错，全都会「看不见」，那样就成了闭着眼扎下去。
      这里要求**先有一次真的量出来**的读数、且这一步是**朝它走完**的，是**有界**的推断；
      被钳幅截断的那一步（量到 120 只走 80）照样按「预计还差 40」中止。
    """
    hist: list[dict] = []
    for i in range(max_iter + 1):
        got = sample()
        if got is None:
            # ★★ 要闸的是「**走完上一步之后**预计还差多少」= 上次读数 − 上一步真走的距离。
            #   上一步就是朝那次测量走的，所以量得越远、这一步也走得越远，两者抵消。
            #   ★ 别闸「上次读数」本身: 那样「量到 14.67 → 走满 14.67 → 被自己挡住」
            #     会被判成失败，而那恰恰是**走到了**的证据（走完了才挡得住）。见 LOST_OK_MM。
            if lost_ok_mm is not None and lost_ok_mm >= 0 and hist:
                r = hist[-1]
                exp = r["raw_norm"] - r["norm"]        # 钳幅没动过 norm 时 exp 正好是 0
                if exp <= lost_ok_mm:
                    return hist, True, (f"目标被吸嘴挡掉了 —— 上一次真量到 "
                                        f"{r['raw_norm']:.2f}mm、已经朝它走了 "
                                        f"{r['norm']:.2f}mm，预计还差 {exp:.2f}mm"
                                        f"（≤{lost_ok_mm:.0f}mm 就算够准）")
            return hist, False, "这一次没认出方块或吸盘码"
        qr_px, cube_px = got
        d = np.asarray(aim_delta(A_inv, C, qr_px, cube_px, c), dtype=float)
        raw = float(np.linalg.norm(d))
        d, scale = clamp_step(d, max_step)
        rec = {"i": i, "qr_px": np.asarray(qr_px, dtype=float),
               "cube_px": np.asarray(cube_px, dtype=float),
               "delta": d, "norm": float(np.linalg.norm(d)), "raw_norm": raw,
               "clamped": scale < 1.0,
               # ★ 这次量完之后**有没有真的走**。别用 len(hist)-1 数步数:
               #   靠「近距丢失」收工时最后一次量没能变成记录（sample 返回 None），
               #   于是「走了 2 步」会被报成「1 步」—— 正好是调试时最需要信的数。
               "moved": False}
        hist.append(rec)
        if raw <= tol_mm:            # ★ 判据用**钳幅前**的：钳幅只是安全网
            return hist, True, None
        if i == max_iter:
            break
        if not move(d, i + 1):
            return hist, False, "走位失败"
        rec["moved"] = True
    return hist, False, f"{max_iter} 步还没进去 {tol_mm:.1f}mm"


def steps_taken(hist) -> int:
    """
    真走了几步 = 有多少次量**后面真的跟了一步**。

    ★ 不要再用 len(hist)-1: 靠「近距丢失」收工时，最后那次量是 None、进不了 hist，
      于是「走了 2 步」会被报成「1 步」。步数正是调试时最需要信的那个数。
    """
    return sum(1 for r in hist if r.get("moved"))


def resid_mm(hist) -> float | None:
    """
    收工时**还差多少 mm** = 最后一次量到的读数 − 那之后真走的距离。没量到过 → None。

    ★ 为什么不能直接报 hist[-1]["raw_norm"]（那是**走之前**的账，不是残差）:
      · 真收敛收工: 最后一次量 raw ≤ tol 就直接返回、一步没走 → raw_norm 就是残差，
        报它没错。这也是唯一一种「两者相等」的情形。
      · 近距丢失收工: 最后一次量是在**走完上一步之后**拍的，raw_norm 是「当时还差
        多少」，而之后我们又朝它走了 norm —— 真残差是 raw_norm − norm（通常≈0）。
        报 raw_norm 等于把那一步**重复算了一遍**，看着像没走到，其实早到了。
      · 钳幅那一步同理: 量到 120 只走了 80 → 残差 40，不是 120。
    ★ 和 steps_taken 同一个理由挨着放: 这两个数都从 hist 里数出来，语义只该在这儿定，
      别让调用方各写一份（④ 的落盘原来就是这么写错的）。
    """
    if not hist:
        return None
    r = hist[-1]
    return float(r["raw_norm"] - (r["norm"] if r.get("moved") else 0.0))


# ─────────────────────────── 看一眼: 码 + 方块 ───────────────────────────
def read_scene(frame, detector, layout, suction, color: str, cam_z: float,
               obj_h: float, qr_px=None) -> dict:
    """
    从一帧里量出闭环要的东西: 吸盘码像素、方块像素、纸面比例（人看用）。

    返回 dict，缺什么就写进 "why"。

    ★ 顺序是钉死的「先纸面 → 再 ppm → 再方块」: detect_cubes 拿到 ppm 才会按
      **真毫米**筛大小（否则退回像素阈值，桌上别的同色小东西就混进来了）。
      而纸面尺度只有解出 ≥3 个码才有。
    ★ qr_px 给了就压过这一帧解出来的（相机那条路用多帧中位数递进来 —— 吸盘码
      只有一个点、没有冗余，单帧的抖动没人平得掉，见 measure_suction_map
      .analyse_frame 的说明）。
    """
    dec, _ = detect_sweep(frame, detector,
                          want=list(layout.codes) + ([suction] if suction else []))
    out = {"qr_px": None, "cube_px": None, "s_table": None, "paper_n": 0,
           "why": None, "cubes": {}}
    if qr_px is not None:
        out["qr_px"] = np.asarray(qr_px, dtype=float)
    elif suction and suction in dec:
        out["qr_px"] = quad_center(dec[suction])
    n_paper = len([c for c in layout.codes if c in dec])
    out["paper_n"] = n_paper
    if n_paper >= 3:
        vis = msm.analyse_frame(dec, layout, suction or "")
        out["s_table"] = vis["s_table"]
    ppm = (out["s_table"] * msm.k_at_height(obj_h, cam_z)
           if out["s_table"] else None)
    out["cubes"] = cvis.detect_cubes(frame, ppm)
    if color in out["cubes"]:
        out["cube_px"] = np.array([out["cubes"][color]["px"],
                                   out["cubes"][color]["py"]], dtype=float)
    if out["qr_px"] is None:
        out["why"] = "没解出吸盘码"
    elif out["cube_px"] is None:
        have = "、".join(out["cubes"]) or "什么都没有"
        out["why"] = f"没认出 {color} 方块（这一帧认到的: {have}）"
    return out


# ─────────────────────────── 机械臂那一段 ───────────────────────────
def make_arm_io(api, dType, tc, cap, detector, layout, suction, limits,
                home, mode, color, cam_z, obj_h, frames):
    """
    造出闭环要的那两个回调 —— ③ 和 ④ 共用，免得两处各写一份读数/走位。

      sample() -> (qr_px, cube_px) | None    每次调用都**重新量**（这才是闭环）
      move(delta, step_no) -> bool           只动 XY；Z、R 锁在 home 上

    ★ move 只写 XY，Z 直接用 home["z"] —— 这里**故意不接 Z 参数**，所以
      「③ 横移时 Z 一点都不变」是结构上保证的，不靠调用方自觉。
      要下降的（④）自己另写升降，别从这里开口子。
    ★ δ 是**增量**（「还差几毫米」，见 aim_delta/solve_loop），所以加在**当下
      实测位姿**上，不是加在 home 上 —— 详见 move 里那段。
    """
    def sample():
        time.sleep(DEFAULT_SETTLE_S)
        sm = msm.sample_suction(cap, detector, suction,
                                focus_goal=list(layout.codes), frames=frames)
        ok, frame = cap.read()
        if not ok or frame is None:
            print("  ✗ 相机读不到帧")
            return None
        sc = read_scene(frame, detector, layout, suction, color, cam_z,
                        obj_h, qr_px=(sm[0] if sm else None))
        if sc["qr_px"] is None or sc["cube_px"] is None:
            print(f"  ✗ 这一停没量到: {sc['why']}")
            return None
        return sc["qr_px"], sc["cube_px"]

    def move(delta, step_no) -> bool:
        # ★★ δ 是**增量**（「还差几毫米」），必须加在**当下实测位姿**上。
        #   2026-09-20 实测踩到的坑: 这里原写成 home+δ（当成「从家起步的绝对目标」），
        #   于是臂在「目标」和「家」之间来回弹 ——
        #     第 1 步量出差 54.9mm → 走到目标 (home+42,+35)   ← 眼睛看着**是对准的**
        #     第 2 步在目标处量，只差 1.1mm → 却发回 home-1.1 = **退回 43mm**
        #     第 3 步在家处量，又差 43mm → 再走到目标 … 往复到 max_iter，永远进不了 tol
        #   屏幕上"第 2 步走 (-1.10,-4.93)mm → 目标 (198.25,-4.93)"就是这个:
        #   打印的 δ 是 1mm，臂却退了 43mm。
        #   ★ 自检照不出来: 自检里的假臂写的是 st["u"] += σ·d（增量），
        #     真臂这里写的是 home+δ（绝对）—— 契约在假臂那边，真臂违了约。
        cur = tc.read_pose_stable(api, dType, 5)
        tgt = {"x": cur["x"] + float(delta[0]), "y": cur["y"] + float(delta[1]),
               "z": home["z"], "r": home["r"]}
        print(f"  → 第 {step_no} 步 走 ({delta[0]:+6.2f},{delta[1]:+6.2f})mm"
              f" → 目标 ({tgt['x']:.2f},{tgt['y']:.2f})")
        ok, why2 = tc.check_target(cur, tgt, limits)
        if not ok:
            print(f"     ✗ {why2}")
            return False
        ok, why2 = tc.move_to(api, dType, tgt, limits, mode)
        if not ok:
            print(f"     ✗ 走位失败: {why2}")
            return False
        ok, why2 = tc.verify_arrival(api, dType, tgt)
        if not ok:
            print(f"     ⚠ 没走到位: {why2}")
        return True

    return sample, move


def run_aim(args) -> int:
    """真动: 连臂 → 闭环横移 → 报数。**只动 XY，不下降、不吸。**"""
    mp, why = load_map(args.map)
    if why:
        print(f"✗ {why}")
        return 1
    A_inv, C = mp["A_inv"], mp["C"]
    cam_z = mp["cam_z"]
    mo = mp["raw"].get("motion") or {}

    table_z = args.table_z if args.table_z is not None else msm.load_table_z()
    if table_z is None:
        print("✗ 不知道纸面 Z，就没法确认横移安不安全。")
        print("  先跑 python3 src/step4_pick_test.py --probe，或给 --table-z。")
        return 1
    z_qr = None    # 连接后按实测位姿现算

    import step2_teach_coords as tc
    api, dType = load_sdk()
    port = find_port(dType, args.port)
    if not port:
        print("✗ 没找到机械臂串口。电源开了吗？USB 插好了吗？")
        return 1
    print(f"\n[连接] {port} @ 115200 …")
    ret = dType.ConnectDobot(api, port, 115200)
    if ret[0] != 0:
        print(f"✗ 连接失败 state={ret[0]} (1=未找到设备 2=端口被占用)")
        return 1

    cap = None
    try:
        dType.SetCmdTimeout(api, 5000)
        ares, alist = tc.read_alarms(api, dType)
        print(f"[报警] result={ares}  {tc.describe_alarms(alist)}")
        if tc.has_alarms(alist):
            tc.report_alarm_detail(alist)
            if tc.needs_homing(alist):
                print("\n✗ 有丢步报警: 零点已不可信，先回零再跑（tools/home_arm.py）。")
                return 1
            ok, _ = tc.resolve_alarms(api, dType, alist, args.clear_alarms)
            if not ok:
                return 1
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
        limits["z"] = (table_z + args.obj_h, limits["z"][1])

        cur = tc.read_pose_stable(api, dType, 5)
        home = dict(cur)
        print(f"\n[当前] {tc.fmt(cur)}")
        z_tip = home["z"] - table_z
        print(f"  纸面 Z={table_z:.2f} → 吸嘴离纸面 z_tip={z_tip:.1f}mm")

        # ── 先按 --hover 把吸嘴摆到该有的高度（纯升降，XY 不动），再查防撞 ──
        # ★ 顺序不能反: 闸门要是摆在 raise 前面，吸嘴只要一开始就低就永远被拒，
        #   而错误提示让人加的那个 --hover 恰恰走不到（实测就是这样卡住的）。
        #   ① 收工时吸嘴停在 z_tip=15 —— 比 26mm 的方块顶面还低 11mm，
        #   所以「一开始就低」是常态，不是意外。
        if args.hover is not None:
            target_z = min(table_z + args.obj_h + args.hover, limits["z"][1])
            if abs(target_z - cur["z"]) > 0.5:
                verb = "升到" if target_z > cur["z"] else "降到"
                print(f"\n[{verb}瞄准高度] Z={target_z:.2f}"
                      f"（纸面 {table_z:.2f} + 障碍高 {args.obj_h:.0f} + "
                      f"{args.hover:.0f}）")
                if not args.yes:
                    input("       回车开始（Ctrl-C 中止）… ")
                tgt = {"x": cur["x"], "y": cur["y"], "z": target_z, "r": cur["r"]}
                ok, why2 = tc.check_target(tc.read_pose_stable(api, dType, 5),
                                           tgt, limits)
                if ok:
                    ok, why2 = tc.move_to(api, dType, tgt, limits, args.mode)
                if not ok:
                    print(f"✗ 升降失败: {why2}")
                    return 1
                tc.verify_arrival(api, dType, tgt)
                home = tc.read_pose_stable(api, dType, 5)
                z_tip = home["z"] - table_z
                print(f"  现在 Z={home['z']:.2f} → 吸嘴离纸面 z_tip={z_tip:.1f}mm")

        # ── 只横移，所以高度是唯一防撞线（摆到 --hover 之后，这里是最后的裁决）──
        need_z = table_z + args.obj_h + msm.MIN_CLEAR_ABOVE_OBJ_MM
        if home["z"] < need_z:
            print(f"\n✗ 吸嘴现在 Z={home['z']:.2f}（离纸面 {z_tip:.1f}mm），"
                  f"比方块顶面只高 {z_tip - args.obj_h:.1f}mm ——")
            print(f"  横移会撞到方块。要求 Z ≥ {need_z:.2f}"
                  f"（纸面 {table_z:.2f} + 方块高 {args.obj_h:.0f} + "
                  f"余量 {msm.MIN_CLEAR_ABOVE_OBJ_MM:.0f}）。")
            if args.hover is None:
                print(f"  用 --hover {msm.DEFAULT_HOVER_MM:.0f} 先升到方块顶面之上再瞄。")
            else:
                print(f"  ★ 你给的 --hover {args.hover:.0f} 不够: 得 ≥ "
                      f"{msm.MIN_CLEAR_ABOVE_OBJ_MM:.0f}（吸嘴至少要高出方块顶面这么多）。")
            return 1

        # ★ c 要按**码当时的真实高度**现算 —— 不是用 ① 量的时候那个高度。
        z_qr = z_tip + args.qr_height
        c = c_coef(cam_z, z_qr, args.obj_h)
        print(f"\n[闭环系数] Z_cam={cam_z:.1f}  Z 锁在 {home['z']:.2f}"
              f" → z_tip={z_tip:.1f} → z_qr={z_qr:.1f}")
        print(f"  c = k({z_qr:.0f}) / k({args.obj_h:.0f}) = "
              f"{msm.k_at_height(z_qr, cam_z):.4f} / "
              f"{msm.k_at_height(args.obj_h, cam_z):.4f} = {c:.4f}")
        A_inv, g = rescale_A_inv(A_inv, mp["map_z_qr"], z_qr, cam_z)
        if g is None:
            print("  ⚠ 地图里没记 ① 量 A 时的码高度 → A 不折算。"
                  "落点不受影响，但可能要**多走三四步**（见 rescale_A_inv）。")
        elif abs(g - 1.0) > 0.02:
            print(f"  A 折算到本高度: k 比 {g:.4f}（① 在 "
                  f"z_qr={mp['map_z_qr']:.0f}，现在 {z_qr:.0f}）"
                  f"→ A⁻¹ ÷{g:.3f}，不然每步超调 {abs(1 - g) * 100:.0f}%")

        layout = load_paper_layout(msm.PAPER_JSON)
        suction = load_suction_code()
        if not suction:
            print("✗ 读不到吸盘码内容（data/suction_qr.json）—— 先跑 tools/gen_suction_qr.py")
            return 1
        cap = open_camera(args.cam, focus=args.focus)
        if not cap.isOpened():
            print("✗ 打不开摄像头")
            return 1
        detector = cv2.QRCodeDetector()

        sample, move = make_arm_io(api, dType, tc, cap, detector, layout, suction,
                                   limits, home, args.mode, args.color, cam_z,
                                   args.obj_h, args.frames)

        print(f"\n⚠ 接下来**只动 XY**（瞄准 {args.color} 方块）: 每次横移后重拍重算，"
              f"最多 {args.max_iter} 步。")
        print(f"  不下降、不吸、不碰东西。Z 锁在 {home['z']:.2f}（离纸面 "
              f"{z_tip:.1f}mm）。")
        print("  Ctrl-C 随时停；本脚本不回零、不写任何末端参数。")
        if not args.yes:
            if input("  确认开始？(y/N) ").strip().lower() != "y":
                print("已取消，未发任何运动指令。")
                return 1

        print("\n[闭环] 每步: 拍 → 算 δ → 走 → 再拍")
        hist, ok, why2 = solve_loop(sample, move, A_inv, C, c,
                                    tol_mm=args.tol, max_step=args.max_step,
                                    max_iter=args.max_iter,
                                    lost_ok_mm=args.lost_ok)
        for r in hist:
            print(f"  · 第 {r['i']} 次量: 码({r['qr_px'][0]:7.1f},{r['qr_px'][1]:7.1f})"
                  f"  方块({r['cube_px'][0]:7.1f},{r['cube_px'][1]:7.1f})"
                  f"  → 还差 {r['raw_norm']:6.2f}mm"
                  + ("  （钳到 %.0fmm）" % r["norm"] if r["clamped"] else ""))

        print("\n" + "─" * 68)
        n_steps = steps_taken(hist)
        if hist:
            r0, r1 = hist[0], hist[-1]
            print(f"首次要求走 {r0['raw_norm']:.2f}mm → 最后 {r1['raw_norm']:.2f}mm"
                  f"（{n_steps} 步）")
            if n_steps <= 1:
                print("  ★ 一步就收 —— 正是「一步解」该有的样子（见开头那段推导）。")
        if ok and why2:
            # ★ 三态里的中间那条: 靠「走完一步还是看不见」收的。**别说成收敛** ——
            #   落点精度是「预计还差多少」，不是 tol，而且没人再量得到它。
            r1l = hist[-1]
            exp_l = r1l["raw_norm"] - r1l["norm"]
            print(f"⚠ 收工（**不是**收敛）: {why2}")
            print(f"  ★ 吸嘴压到 {args.color} 上方时，白色贴纸+黑色支架必然把目标色"
                  f"挡掉 —— 这是几何必然，越迭代越看不见，最后那几毫米量不到。")
            print(f"  ★ 所以这次落点精度 = **预计还差 {exp_l:.2f}mm**"
                  f"（上次真量到 {r1l['raw_norm']:.2f}mm、朝它走了 {r1l['norm']:.2f}mm），"
                  f"不是 {args.tol:.1f}mm。要更准只能换个不挡视线的机位。")
        elif ok:
            print(f"✅ 到位: 律说「再走不到 {args.tol:.1f}mm」。")
            # ★ 这一步是最值钱的读数: 降下去之后 δ 会被 k 比缩放 —— 收敛了就还是 ≈0，
            #   即「下降不动落点」。这正是 ④ 敢瞄完盲降的根据，见 predict_after_descend。
            if args.obj_h > 0 and hist:
                dz = z_qr - (args.qr_height + args.obj_h)     # 悬停 → 方块顶面，降多少
                d_now = aim_delta(A_inv, C, r1["qr_px"], r1["cube_px"], c)
                d_after = predict_after_descend(d_now, z_qr, dz, cam_z)
                if d_after is not None:
                    k_before = msm.k_at_height(z_qr, cam_z)
                    k_after = msm.k_at_height(z_qr - dz, cam_z)
                    print(f"   ★ 若现在**只降不横移** dz={dz:.0f} 到方块顶面（z_qr "
                          f"{z_qr:.0f}→{z_qr - dz:.0f}mm，k {k_before:.3f}→{k_after:.3f}）:")
                    print(f"     律要的 δ 变成 ({d_after[0]:+.2f},{d_after[1]:+.2f})mm "
                          f"= δ_now × {k_after / k_before:.3f}。")
                    print(f"     ★ 已经收敛（|δ_now|={np.linalg.norm(d_now):.2f}mm ≈ 0），"
                          f"所以降下去还是 ≈0 —— **下降不会把瞄准弄坏**，"
                          f"④ 就是这样瞄完就降的。")
        else:
            print(f"❌ 没到位: {why2}")
            print("  常见原因: ①/② 的地图过期（相机或纸动过就重跑）、"
                  "方块被挡住、吸盘码没解出来。")
            print("  桌面坐标没动，可以直接重跑一次，或先不加 --go 单独看一张。")
            # ★ ③ 失败 ≠ 抓不了: ④ 会**从头自己瞄一遍**（它有自己的 solve_loop），
            #   不复用 ③ 的结果 —— ③ 只是「不下降、不吸、不碰」的演练。
            #   不给这句话，屏幕上只剩一个 ❌，很容易以为整条路都断了。
            hv = "" if args.hover is None else f" --hover {args.hover:g}"
            print(f"  ★ 但 ③ 只是**演练**，④ 会自己从头瞄一遍、不依赖这次结果。"
                  f"要直接抓:"
                  f"\n    python3 tools/pick_suction.py --color {args.color}{hv} --hold --go")
        if ok:
            # ★ ③ **只瞄准，一个东西都不抓**（不下降、不吸、不碰）。成功之后不
            #   给这句话，屏幕上就只剩一个 ⚠/✅，很容易被读成「失败了」或者
            #   「是不是没抓」—— 抓取是 ④ 的事，把该跑的命令递到手上。
            hv = "" if args.hover is None else f" --hover {args.hover:g}"
            print(f"\n▶ ③ 到这儿就结束了（只横移、不下降、不吸）。要真抓:"
                  f"\n    python3 tools/pick_suction.py --color {args.color}{hv} --hold --go")
        return 0 if ok else 1
    finally:
        if cap is not None:
            cap.release()
        try:
            dType.DisconnectDobot(api)
        except Exception:
            pass


def report_only(args) -> int:
    """只拍一张报数（不动臂）。--image 或直接开相机。"""
    mp, why = load_map(args.map)
    if why:
        print(f"✗ {why}")
        return 1
    A_inv, C, cam_z = mp["A_inv"], mp["C"], mp["cam_z"]
    layout = load_paper_layout(msm.PAPER_JSON)
    suction = load_suction_code()
    detector = cv2.QRCodeDetector()

    qr_med, frame = None, None
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f"✗ 读不到图片 {args.image}")
            return 1
    else:
        cap = open_camera(args.cam, focus=args.focus)
        if not cap.isOpened():
            print("✗ 打不开摄像头")
            return 1
        try:
            # 吸盘码在另一个焦面 → 和 ① 一样走「多帧中位数」，别用单帧
            sm = msm.sample_suction(cap, detector, suction or "QR_SUCTION",
                                    focus_goal=list(layout.codes), frames=args.frames)
            qr_med = sm[0] if sm else None
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            print("✗ 相机读不到帧")
            return 1

    sc = read_scene(frame, detector, layout, suction, args.color, cam_z,
                    args.obj_h, qr_px=qr_med)
    print("=" * 68)
    print("  ③ 只报数（机械臂一步没动）")
    print("=" * 68)
    print(f"  地图: {args.map or SUCTION_MAP_JSON}")
    print(f"  光心 C = ({C[0]:.1f}, {C[1]:.1f})px   Z_cam = {cam_z:.1f}mm")
    if sc["s_table"]:
        print(f"  纸面比例 {sc['s_table']:.3f}px/mm（{sc['paper_n']} 个码拟合）")
    else:
        print("  纸面比例 ✗ 没拟合出来（纸被挡住/挪走？不影响瞄准）")
    print(f"  认到的方块: {'、'.join(sc['cubes']) or '（无）'}")
    if sc["qr_px"] is None or sc["cube_px"] is None:
        print(f"\n✗ 量不齐: {sc['why']}")
        print(f"  ★ 吸盘码在**另一个焦面**上（离相机 ~150mm，纸面 ~340mm）。"
              f"现在焦距锁在 FOCUS={args.focus if args.focus is not None else '—'}。"
              f"自动对焦会停在随它高兴的位置上，吸盘码就时解得出时解不出 ——")
        print(f"    重扫一次焦距找那段「5 个码一帧全中」的窗口，"
              f"再用 --focus <值> 覆盖: v4l-utils 的 v4l2-ctl 或扫焦距的小脚本都行。")
        return 1

    # ★ 这个 z_tip 是**声明**的，不是量出来的 —— 不连臂就看不出吸嘴现在多高。
    #   律对 z_tip 不敏感（c 里是两个 k 相除），但报数时要说清这是假设。
    #   --hover 的口径跟 run_aim **同一条**: 「方块顶面 + 这么多 mm」，不是绝对高度。
    #   不连臂时只好假设吸嘴就在 run_aim 会摆到的那个高度上。
    hover = args.hover if args.hover is not None else msm.DEFAULT_HOVER_MM
    z_tip = args.obj_h + hover
    z_qr = z_tip + args.qr_height
    c = c_coef(cam_z, z_qr, args.obj_h)
    # ★ 和 run_aim 用**同一本账**: ① 是在 z_qr=135 量 A 的，这里按当前位置折算。
    #   不折也能落对，只是报出来的「要走多少」会是真距离的 g 倍 —— 而 --go 里折过，
    #   两边对不上会让人以为报数和实走是两个东西。折了以后 d 就是**真还差多少 mm**。
    A_inv, g = rescale_A_inv(A_inv, mp["map_z_qr"], z_qr, cam_z)
    d = aim_delta(A_inv, C, sc["qr_px"], sc["cube_px"], c)
    cube_side = sc["cubes"][args.color].get("side_mm")
    print(f"\n  吸盘码像素 ({sc['qr_px'][0]:7.1f},{sc['qr_px'][1]:7.1f})"
          + ("" if qr_med is not None else "   （单帧，没去抖）"))
    print(f"  方块像素   ({sc['cube_px'][0]:7.1f},{sc['cube_px'][1]:7.1f})"
          + (f"   顶面边长 {cube_side:.1f}mm" if cube_side else ""))
    print(f"  ★ 假设吸嘴在「方块顶面 +{hover:.0f}」= 离纸面 z_tip={z_tip:.1f}"
          f"（**声明**，不连臂量不出来）→ z_qr={z_qr:.1f}")
    print(f"  c = k({z_qr:.0f})/k({args.obj_h:.0f}) = {c:.4f}")
    if g is not None and abs(g - 1.0) > 0.02:
        print(f"  A 已从 ① 的高度（z_qr={mp['map_z_qr']:.0f}）折到 {z_qr:.0f}mm"
              f"（÷{g:.3f}）—— 和 --go 同一本账。")
    print(f"\n  ★ 律要求走: ({d[0]:+.2f}, {d[1]:+.2f})mm   合 "
          f"{np.linalg.norm(d):.2f}mm")
    dz = hover                                   # 悬停 → 方块顶面，正好降一个 hover
    d2 = predict_after_descend(d, z_qr, dz, cam_z)
    if d2 is not None:
        print(f"  ★ 若吸嘴落到方块顶面（z_tip→{args.obj_h:.0f}，z_qr→"
              f"{args.qr_height + args.obj_h:.0f}mm，降 dz={dz:.0f}），"
              f"律要的 δ 变成 ({d2[0]:+.2f},{d2[1]:+.2f})mm")
        print(f"     ★ δ 只被 k 比缩放（×{msm.k_at_height(z_qr - dz, cam_z) / msm.k_at_height(z_qr, cam_z):.3f}）"
              f"—— **收敛时 δ≈0，降下去还是 ≈0**: 下降不会把瞄准弄坏。")
    print("\n  要真走就加 --go（只横移、不下降、不吸）。")
    return 0


# ─────────────────────────── 自检 ───────────────────────────
def selftest() -> int:
    """离线自检: 用一台**假机械臂**把闭环的每条性质钉一遍。"""
    bad = 0

    def check(name, cond, extra=""):
        nonlocal bad
        print(f"  {'✅' if cond else '❌'} {name}" + (f"   {extra}" if extra else ""))
        if not cond:
            bad += 1

    print("aim_suction.py 自检")
    np.random.default_rng(7)

    # ── 假机械臂 ──
    # 纸面 mm 里: 码在吸嘴正上方，所以码的地面位置 = 吸嘴的**真实** XY（记为 u）。
    #   码像素 − C = s·k(z_qr)·u        方块像素 − C = s·k(obj_h)·w_c
    #   机械臂报 1mm 真走 σ mm → A = s·k_qr·σ，这正是地图里 A 的定义。
    Zc, s_tab = 323.0, 4.575
    z_qr, obj_h = 135.0, 26.0
    k_qr, k_cube = msm.k_at_height(z_qr, Zc), msm.k_at_height(obj_h, Zc)
    C = np.array([1004.6, 589.1])
    c_ok = k_qr / k_cube

    def make_arm(sigma, w_c, u0=None, A_noise=0.0, rng=None):
        """返回 (A_inv_controller, sample, move, state)"""
        A_true = s_tab * k_qr * sigma * np.eye(2)
        A_ctrl = A_true.copy()
        if A_noise:
            A_ctrl = A_ctrl + np.eye(2) * A_noise
        st = {"u": np.zeros(2) if u0 is None else np.asarray(u0, float)}

        def sample():
            return (C + s_tab * k_qr * st["u"],
                    C + s_tab * k_cube * np.asarray(w_c, float))

        def move(d, _n):
            st["u"] = st["u"] + sigma * np.asarray(d, float)
            return True

        return np.linalg.inv(A_ctrl), sample, move, st

    # 1) 系数用对 → 一步就落在方块上
    #    ★ w_c 要挑得比 MAX_STEP 短，不然先被钳幅，测的就不是「一步解」了。
    A_inv, smp, mv, st = make_arm(1.0, (45.0, -25.0))
    hist, ok, why = solve_loop(smp, mv, A_inv, C, c_ok, log=lambda *a, **k: None)
    check("③ 系数用对 → 一步到位", ok and len(hist) == 2,
          f"{len(hist)} 次量，残差 {hist[-1]['raw_norm']:.2e}mm")
    check("③ 第二步的 δ 就是机器精度", hist[-1]["raw_norm"] < 1e-9,
          f"{hist[-1]['raw_norm']:.2e}mm")

    # 2) 系数错用 k_tip → 停在固定错位。★ 要害是**律自己会以为到位了**（δ=0 正好是
    #    错系数的那个不动点），所以这条必须用**机械臂的真实落点**判，不能信 ok。
    #    换句话说: 闭环**没法自证系数对不对** —— 这就是 ③ 的自检必须拿假臂的原因。
    #    ★ w_c 仍要短于 MAX_STEP: 这是「**一步**走到错地方」，钳幅会把一步拆成两步，
    #      固定错位还是那个错位，但就看不出「一步」了。
    c_bad = msm.k_at_height(15.0, Zc) / msm.k_at_height(obj_h, Zc)
    W_BAD = np.array([40.0, -20.0])
    A_inv2, smp2, mv2, st2 = make_arm(1.0, W_BAD)
    hist2, ok2, _ = solve_loop(smp2, mv2, A_inv2, C, c_bad, log=lambda *a, **k: None)
    factor = c_bad / c_ok
    err_mm = float(np.linalg.norm(st2["u"] - W_BAD))
    want_err = float(np.linalg.norm((1 - factor) * W_BAD))
    check("③ 系数错用 k_tip → 律以为到位了，其实停在固定错位",
          ok2 and len(hist2) == 2 and abs(err_mm - want_err) < 1e-6,
          f"落在 {factor:.3f}·u_c，真偏 {err_mm:.2f}mm（该 {want_err:.2f}）")

    # 3) 错位 ∝ 「离 C 多远」——近光心的方块反而看着准
    errs = []
    for wc in ((30.0, 0.0), (60.0, 0.0), (120.0, 0.0)):
        _, s3, m3, st3 = make_arm(1.0, wc)
        solve_loop(s3, m3, A_inv, C, c_bad, log=lambda *a, **k: None)
        errs.append(float(np.linalg.norm(st3["u"] - np.array(wc))))
    check("③ 错位跟「离光心多远」成正比", abs(errs[0] - errs[2] / 4) < 1e-9
          and abs(errs[1] - errs[2] / 2) < 1e-9,
          f"30/60/120mm → 偏 {errs[0]:.2f}/{errs[1]:.2f}/{errs[2]:.2f}mm")

    # 4) 机械臂刻度不是 1（报数≠真毫米）时，闭环照样收敛
    #    δ = (w_c−u)/σ，u_new = u + σδ = w_c —— σ **自己约掉**，还是一步。
    #    ★ 于是 w_c 也必须比 MAX_STEP 短，否则先被钳幅，测不到「σ 被约掉」这件事。
    A_inv4, smp4, mv4, st4 = make_arm(1.0927, (40.0, 30.0))
    hist4, ok4, _ = solve_loop(smp4, mv4, A_inv4, C, c_ok, log=lambda *a, **k: None)
    check("③ 机械臂刻度 1.093 也一步到位", ok4 and len(hist4) == 2,
          f"{len(hist4) - 1} 步，残差 {hist4[-1]['raw_norm']:.2e}mm")

    # 5) A 有 2% 偏差 → 仍然收敛（只是多走几步: 每步只剩 1/1.02 的余量）
    A_inv5, smp5, mv5, st5 = make_arm(1.0, (40.0, 30.0), A_noise=0.02 * s_tab * k_qr)
    hist5, ok5, _ = solve_loop(smp5, mv5, A_inv5, C, c_ok, log=lambda *a, **k: None)
    e5 = float(np.linalg.norm(st5["u"] - np.array([40.0, 30.0])))
    check("③ A 偏 2% 仍收敛", ok5 and e5 < 1.0,
          f"{len(hist5) - 1} 步，落点差 {e5:.3f}mm")

    # 6) 单步钳幅: 起步很远时先夹到 MAX_STEP_MM，但最终仍到位
    A_inv6, smp6, mv6, st6 = make_arm(1.0, (400.0, 0.0))
    hist6, ok6, _ = solve_loop(smp6, mv6, A_inv6, C, c_ok, log=lambda *a, **k: None)
    check("③ 单步钳幅后仍到位", ok6 and hist6[0]["clamped"],
          f"首步律要 {hist6[0]['raw_norm']:.0f}mm，钳到 {hist6[0]['norm']:.0f}mm"
          f"，共 {len(hist6) - 1} 步")

    # 6b) ★ 高度折算 —— ① 是在 z_qr=135 量 A 的，③/④ 却在 176 上瞄（k 差 1.279）。
    #     ★★ 先分清两个完全不同的东西（这里最容易混）:
    #       · A 只在**地图**里、只在 135 那个高度量过 → 到了 176 就「偏小 g−1=28%」
    #       · c 是**每次现算**的（c_coef(cam_z, z_qr, obj_h)，z_qr 取当下值）
    #     本组测的就是「A 偏了、c 没偏」这一种。此时:
    #       c·(方块−C) = s·k(176)·w_c，码−C = s·k(176)·u → δ = (g/σ)(w_c − u)
    #       u ← u + σδ = u + g(w_c − u) → **不动点仍是 w_c**，误差每步乘 (g−1)
    #     ★ 所以不折也**压在方块上**，只是从 40mm 起步要爬 4 步；折了 1 步。
    #     ★ 故意**不能**把 135 那个 c 拿来用 —— 那是「c 错」不是「A 错」，落点会歪成
    #       w_c/g（见下面 6d 那条对照，两件事必须分开钉）。
    Z_MAP, Z_RUN = 135.0, 176.0
    g_want = msm.k_at_height(Z_RUN, Zc) / msm.k_at_height(Z_MAP, Zc)
    c_run = c_coef(Zc, Z_RUN, obj_h)          # ★ 当下高度的 c —— ③/④ 就是这么算的

    def arm_at(z_true, sigma, w_c):
        """真码挂在 z_true 上（地图里却以为在 Z_MAP）—— 专门用来测折算。"""
        kt = msm.k_at_height(z_true, Zc)
        st = {"u": np.zeros(2)}

        def sample():
            return (C + s_tab * kt * st["u"],
                    C + s_tab * k_cube * np.asarray(w_c, float))

        def move(d, _n):
            st["u"] = st["u"] + sigma * np.asarray(d, float)
            return True

        return sample, move, st

    A_inv_map = np.linalg.inv(s_tab * msm.k_at_height(Z_MAP, Zc) * np.eye(2))
    W_H = np.array([35.0, 20.0])     # |W_H|=40.3 → 不折时首步 1.279× = 51.5 < MAX_STEP，不触发钳幅
    smpB, mvB, stB = arm_at(Z_RUN, 1.0, W_H)
    histB, okB, _ = solve_loop(smpB, mvB, A_inv_map, C, c_run, log=lambda *a, **k: None)
    eB = float(np.linalg.norm(stB["u"] - W_H))
    check("③ 高度不折算 → 仍压在方块上（<tol），但要多爬几步",
          okB and eB <= DEFAULT_TOL_MM and len(histB) - 1 >= 3,
          f"{len(histB) - 1} 步，落点差 {eB:.3f}mm（每步超调 {(g_want - 1) * 100:.0f}%）")

    A_inv_r, g_got = rescale_A_inv(A_inv_map, Z_MAP, Z_RUN, Zc)
    check("③ 折算系数就是 k(现在)/k(①量A时)，没有别的魔数",
          g_got is not None and abs(g_got - g_want) < 1e-12,
          f"g={g_got:.4f} = k({Z_RUN:.0f})/k({Z_MAP:.0f})")
    smpC, mvC, stC = arm_at(Z_RUN, 1.0, W_H)
    histC, okC, _ = solve_loop(smpC, mvC, A_inv_r, C, c_run, log=lambda *a, **k: None)
    eC = float(np.linalg.norm(stC["u"] - W_H))
    check("③ 折算后 → 回到一步到位", okC and len(histC) == 2,
          f"{len(histC) - 1} 步（不折是 {len(histB) - 1} 步）")
    check("③ 折算买到的是**步数**，不是精度（两者都落在方块上）",
          eC < 1e-9 and eB <= DEFAULT_TOL_MM,
          f"折 {eC:.2e}mm / 不折 {eB:.3f}mm，都 ≤ tol {DEFAULT_TOL_MM}mm")

    # 6c) 折算的两条边界: 高度一样 → g=1 原样；老地图没记高度 → **不猜**，原样
    same, g_same = rescale_A_inv(A_inv_map, Z_MAP, Z_MAP, Zc)
    check("③ 高度相同 → g=1、A 原样",
          g_same is not None and abs(g_same - 1.0) < 1e-12 and np.allclose(same, A_inv_map))
    nz, g_none = rescale_A_inv(A_inv_map, None, Z_RUN, Zc)
    check("③ 老地图没记码高度 → 不折算、不猜高度",
          g_none is None and np.allclose(nz, A_inv_map))

    # 6d) ★ 对照: 把 c 也一起用错（拿 135 那个 c 去 176 上跑）—— 这次**落点会歪**，
    #     歪成 w_c/g。这条和 6b 并排看，就是把「A 的标量误差」和「c 的误差」
    #     彻底分开: 前者只费步数，后者费的是精度。③ 的全部底气就在这条分界上。
    #     ★ 比 wantD 时留 tol 的余量: 律是「raw≤tol 就停」，不是「走到不动点才停」，
    #       所以停下时还差一点。raw = g·e（判据带 g），于是**停时的位置残差 ≤ tol/g**。
    smpD, mvD, stD = arm_at(Z_RUN, 1.0, W_H)
    c_stale = c_coef(Zc, Z_MAP, obj_h)
    histD, okD, _ = solve_loop(smpD, mvD, A_inv_map, C, c_stale, log=lambda *a, **k: None)
    eD = float(np.linalg.norm(stD["u"] - W_H))
    wantD = float(np.linalg.norm(W_H / g_want - W_H))
    check("③ 对照: c 用错（陈的）→ 落点真歪，律还以为到位了",
          okD and abs(eD - wantD) <= DEFAULT_TOL_MM and eD > 5.0,
          f"停在 w_c/g、偏 {eD:.2f}mm（不动点该 {wantD:.2f}，"
          f"差 {abs(eD - wantD):.2f} ≤ tol/g）—— 这就是 6b 没发生的那种")

    # 7) sample 返回 None → 干净退出，不硬撑
    _, _, mv7, _ = make_arm(1.0, (50.0, 0.0))
    h7, ok7, why7 = solve_loop(lambda: None, mv7, A_inv, C, c_ok,
                               log=lambda *a, **k: None)
    check("③ 量不到就干净退出", not ok7 and h7 == [] and "没认出" in (why7 or ""), why7 or "")

    # 7b~7f) ★★ 「走完一步还是看不见 = 够准了」（2026-09-20 实测加的、又改过的那条）——
    #        吸嘴压上方块时目标色**必然**被挡掉，最后那几毫米量不到。
    #        ★ 这几条一起钉住「它是有界的推断，不是『看不见就当到了』」，也钉住
    #          **闸的是「走完之后预计还差多少」，不是「上次量到多少」**。
    def arm_that_goes_blind(w_c, blind_after):
        """量 blind_after 次之后就「被吸嘴挡住了」（sample 返回 None）。"""
        A_i, smp, mv, st = make_arm(1.0, w_c)
        seen = {"n": 0}

        def sample():
            seen["n"] += 1
            return None if seen["n"] > blind_after else smp()

        return A_i, sample, mv, st

    # 7b) 真量到 6mm、朝它走满 6mm、随后丢失 → 认账（why 有字面、ok 仍为真）
    A_inv7b, smp7b, mv7b, _ = arm_that_goes_blind((6.0, 0.0), 1)
    h7b, ok7b, why7b = solve_loop(smp7b, mv7b, A_inv7b, C, c_ok,
                                  lost_ok_mm=LOST_OK_MM, log=lambda *a, **k: None)
    check("③ ★ 近距走满后丢失 → 认账（ok=True、why 有字面）",
          ok7b and why7b and len(h7b) == 1
          and h7b[-1]["raw_norm"] - h7b[-1]["norm"] <= LOST_OK_MM
          and "挡掉" in why7b,
          f"量到 {h7b[-1]['raw_norm']:.2f}mm、走 {h7b[-1]['norm']:.2f}mm → "
          f"预计还差 {h7b[-1]['raw_norm'] - h7b[-1]['norm']:.2f}mm: {why7b}")

    # 7c) ★ 被**钳幅**截断、走完还差得远（量到 120、只敢走 MAX_STEP=80 → 预计还差 40）
    #     → **不认账**，照样中止。这条才是「有界」那半边的证明: 钳幅那一步不豁免。
    A_inv7c, smp7c, mv7c, _ = arm_that_goes_blind((120.0, 0.0), 1)
    h7c, ok7c, why7c = solve_loop(smp7c, mv7c, A_inv7c, C, c_ok,
                                  lost_ok_mm=LOST_OK_MM, log=lambda *a, **k: None)
    check("③ ★ 钳幅后仍差得远就丢 → 不认账（钳幅那一步不豁免）",
          not ok7c and "没认出" in (why7c or ""),
          f"量到 {h7c[-1]['raw_norm']:.2f}mm（钳到 {h7c[-1]['norm']:.0f}mm）→ "
          f"预计还差 {h7c[-1]['raw_norm'] - h7c[-1]['norm']:.2f}mm > "
          f"{LOST_OK_MM:.0f}mm → 中止")

    # 7f) ★★ 实测那一跑: 量到 14.67mm → 朝它走满 14.67mm → 目标被挡掉。
    #     ★ 旧判据（raw_norm ≤ 10）在这里判**失败**，可它恰恰是**走到了**的证据 ——
    #       走完了才挡得住。这条把它永久钉住。
    A_inv7f, smp7f, mv7f, _ = arm_that_goes_blind((2.31, 14.48), 1)
    h7f, ok7f, why7f = solve_loop(smp7f, mv7f, A_inv7f, C, c_ok,
                                  lost_ok_mm=LOST_OK_MM, log=lambda *a, **k: None)
    r7f = h7f[-1]
    check("③ ★★ 量到 14.67mm 又走满 14.67mm 才丢 → 认账（旧判据在这里判错）",
          ok7f and why7f and abs(r7f["raw_norm"] - r7f["norm"]) < 1e-9
          and r7f["raw_norm"] > LOST_OK_MM and "挡掉" in why7f,
          f"量到 {r7f['raw_norm']:.2f}mm、走 {r7f['norm']:.2f}mm（都没钳）→ "
          f"预计还差 {r7f['raw_norm'] - r7f['norm']:.2f}mm —— 旧判据会拿 "
          f"{r7f['raw_norm']:.2f} > {LOST_OK_MM:.0f} 判失败")

    # 7d) 把闸值关掉（负数）→ 即使量近过也退回老行为
    A_inv7d, smp7d, mv7d, _ = arm_that_goes_blind((6.0, 0.0), 1)
    h7d, ok7d, why7d = solve_loop(smp7d, mv7d, A_inv7d, C, c_ok,
                                  lost_ok_mm=-1, log=lambda *a, **k: None)
    check("③ ★ --lost-ok 给负数 → 关掉这条，退回「看不见就中止」",
          not ok7d and "没认出" in (why7d or ""),
          why7d or "")

    # 7e) 步数 = **真走了几步**。★ 别退化成 len(hist)-1: 近距丢失收工时最后那次量
    #     是 None、进不了 hist，len(hist)-1 会少数一步（实测把 2 步报成 1 步）。
    check("③ ★ 步数用 steps_taken，不用 len(hist)-1（丢失收工时不少数）",
          steps_taken(h7b) == 1 and steps_taken(hist) == len(hist) - 1,
          f"丢失收工: {steps_taken(h7b)} 步 / hist {len(h7b)} 条；"
          f"真收敛: {steps_taken(hist)} 步 / hist {len(hist)} 条")

    # 7g) ★★ 残差 = **走完最后一步之后**还差多少，不是最后那次量的 raw_norm。
    #     ★ 近距丢失收工时两者差**一整步**: 实测记的是「差 14.67mm」，可那 14.67 已经
    #       走掉了、人就压在方块上 —— 落盘报 14.67 会让人以为差得离谱，回头白调参数。
    #       三种收工方式都要对: 真收敛（没走，相等）/ 丢失（走完了，≈0）/ 钳幅（只走了一截）。
    check("③ ★ 残差要报「走完之后还差多少」，不是走之前那次读数",
          abs(resid_mm(h7b) - 0.0) < 1e-9
          and abs(resid_mm(hist) - hist[-1]["raw_norm"]) < 1e-9
          and resid_mm(h7c) > LOST_OK_MM,
          f"丢失收工: {resid_mm(h7b):.2f}mm（raw 却是 {h7b[-1]['raw_norm']:.2f}）；"
          f"真收敛: {resid_mm(hist):.2e}mm；钳幅那步: {resid_mm(h7c):.1f}mm")
    check("③ 从来没量到 → 残差说 None，不编一个 0", resid_mm([]) is None)

    # 8) move 失败 → 干净退出
    A_inv8, smp8, _, _ = make_arm(1.0, (50.0, 0.0))
    h8, ok8, why8 = solve_loop(smp8, lambda d, n: False, A_inv8, C, c_ok,
                               log=lambda *a, **k: None)
    check("③ 走位失败就干净退出", not ok8 and "走位失败" in (why8 or ""), why8 or "")

    # 9) 下降预测: δ 只被 k 比缩放 → **收敛了降下去还是 ≈0**（④ 敢盲降的根据）
    d_now = np.array([6.0, -4.0])                # 假装还没瞄上，律还要走 7.2mm
    dz_top = z_qr - obj_h                        # 悬停 → 方块顶面
    d_after = predict_after_descend(d_now, z_qr, dz_top, Zc)
    ratio = msm.k_at_height(obj_h, Zc) / msm.k_at_height(z_qr, Zc)
    check("③ 下降预测: δ_aft = δ_now × (k_aft/k_now)",
          d_after is not None and np.allclose(d_after, ratio * d_now, atol=1e-12),
          f"降 {dz_top:.0f}mm → ×{ratio:.4f}（{np.linalg.norm(d_now):.2f}→"
          f"{np.linalg.norm(d_after):.2f}mm）")
    check("③ 下降只会**缩小** δ（k_aft < k_now）", ratio < 1.0)
    check("③ ★ 已收敛时下降不用补（δ=0 → 0）—— 这就是 ④ 盲降的根据",
          np.allclose(predict_after_descend(np.zeros(2), z_qr, dz_top, Zc),
                      np.zeros(2), atol=1e-12))
    check("③ 高度不合法（升到相机平面以上 / 现在就超了）→ 说 None，不硬算",
          predict_after_descend(d_now, z_qr, -(Zc - z_qr), Zc) is None
          and predict_after_descend(d_now, Zc + 1.0, 10.0, Zc) is None)

    # 10) clamp_step 的边界
    d, sc = clamp_step(np.array([3.0, 4.0]), 60.0)
    check("③ 钳幅 不超限就不动", np.allclose(d, [3.0, 4.0]) and sc == 1.0)
    d, sc = clamp_step(np.array([60.0, 80.0]), 60.0)
    check("③ 钳幅 超了就按比例夹", abs(np.linalg.norm(d) - 60.0) < 1e-9 and sc < 1.0,
          f"|δ|={np.linalg.norm(d):.3f}mm")

    # 11) c 的两条性质: z_qr 越高 c 越大；obj_h=0 时 c 退化成 k_qr
    check("③ c 随码越高越大", c_coef(Zc, 135.0, 26.0) > c_coef(Zc, 50.0, 26.0))
    check("③ obj_h=0 时 c 退化成 k_qr",
          abs(c_coef(Zc, 135.0, 0.0) - msm.k_at_height(135.0, Zc)) < 1e-12)

    # 11b) ★ 实测踩出来的那条: 工具叫用户加 --hover，那这个悬停高度就**必须自己
    #      过得了防撞闸门**，否则按提示加了照样被拒（raise 摆上去仍 < need_z）。
    #      同时钉住「先 raise 后查闸门」这个顺序依赖的两半是自洽的。
    z_t = -18.42
    check("③ 默认悬停高度本身就过防撞闸门（--hover 才真的有用）",
          z_t + obj_h + msm.DEFAULT_HOVER_MM >= z_t + obj_h + msm.MIN_CLEAR_ABOVE_OBJ_MM,
          f"悬停余量 {msm.DEFAULT_HOVER_MM:.0f} ≥ 闸门要求 "
          f"{msm.MIN_CLEAR_ABOVE_OBJ_MM:.0f}mm")

    # 12) 系数用成 k_tip 的后果 —— 用真地图的数报一次，给人一个具体量级
    z_tip_now = 15.0
    c_right = c_coef(Zc, z_tip_now + 120.0, 26.0)
    c_wrong = msm.k_at_height(z_tip_now, Zc) / msm.k_at_height(26.0, Zc)
    ratio = c_wrong / c_right
    check("③ 旧系数偏差有多大（报个数给人看）", 0.5 < ratio < 0.8,
          f"k_tip 方案 {c_wrong:.3f} vs 正确 {c_right:.3f} → 只走到 "
          f"{ratio * 100:.0f}% 处")

    # 13) 地图缺失时要说人话（不兜底）
    #     用一个不存在的路径 —— 必须返回原因而不是抛异常
    mp, why = load_map("/tmp/definitely_not_here_suction_map.json")
    check("③ 地图读不到时报人话", mp == {} and bool(why), (why or "")[:40])

    print(f"\n{'✅ 全部通过' if not bad else f'❌ {bad} 项没过'}")
    return 1 if bad else 0


# ─────────────────────────── CLI ───────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(
        description="③ 闭环: 把吸嘴瞄到方块正上方（每步重拍重算，只用相对运动）")
    ap.add_argument("--color", default=None,
                    help=f"瞄哪个颜色的方块（{'/'.join(cvis.CUBE_COLORS)}）")
    ap.add_argument("--go", action="store_true",
                    help="★ 真动机械臂: 闭环横移（只动 XY，不下降不吸）")
    ap.add_argument("--image", default=None,
                    help="用一张已有图片只报数（不连臂，也不开相机）")
    ap.add_argument("--map", default=None,
                    help=f"①/② 的地图，默认 {SUCTION_MAP_JSON}")
    ap.add_argument("--obj-h", type=float, default=msm.DEFAULT_OBJ_H_MM,
                    help=f"方块高 mm。**要你自己声明**（默认 "
                         f"{msm.DEFAULT_OBJ_H_MM:.0f}）—— c 和防撞闸门都用它")
    ap.add_argument("--qr-height", type=float, default=msm.DEFAULT_QR_HEIGHT_MM,
                    help=f"吸盘码中心到吸嘴尖 mm（默认 {msm.DEFAULT_QR_HEIGHT_MM:.0f}）")
    ap.add_argument("--hover", type=float, default=None, nargs="?",
                    const=msm.DEFAULT_HOVER_MM,
                    help="瞄准前先**只升降**（XY 不动）到「方块顶面 + 这么多 mm」。"
                         "带上时裸写 --hover 就等于 "
                         f"{msm.DEFAULT_HOVER_MM:.0f}；不带时吸嘴停在原地，"
                         f"只有它正好高出方块顶面 {msm.MIN_CLEAR_ABOVE_OBJ_MM:.0f}mm "
                         "以上才让横移。★ ① 收工时吸嘴停在 z_tip=15（比方块还低），"
                         "所以第一次瞄基本都要带这个。")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL_MM,
                    help=f"律说「再走不到这么多 mm」就算到位（默认 {DEFAULT_TOL_MM}）")
    ap.add_argument("--max-step", type=float, default=MAX_STEP_MM,
                    help=f"单步钳幅 mm（默认 {MAX_STEP_MM:.0f}）—— 地图坏了时的保险")
    ap.add_argument("--lost-ok", type=float, default=LOST_OK_MM, metavar="MM",
                    help=f"「近距丢失 = 够准了」的闸值（默认 {LOST_OK_MM:.0f}mm）: "
                         f"曾经量到过 ≤MM 之后目标被吸嘴挡住，就算收工。"
                         f"给负数 = 关掉，退回「看不见就中止」")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER,
                    help=f"迭代上限（默认 {MAX_ITER}）")
    ap.add_argument("--table-z", type=float, default=None,
                    help="纸面 Z，默认读 output/pick_test_result.json")
    ap.add_argument("--frames", type=int, default=msm.ACCUM_FRAMES,
                    help=f"每次量积累多少帧（默认 {msm.ACCUM_FRAMES}）")
    ap.add_argument("--cam", type=int, default=None, help="摄像头编号，默认自动挑")
    ap.add_argument("--focus", type=int, default=qv.FOCUS_LOCK,
                    help=f"**锁死手动焦距**（UVC 值，量程 0~1023；小=对远、大=对近）。"
                         f"默认 {qv.FOCUS_LOCK} —— 那个值上「吸盘码 + 纸面 4 码」"
                         f"一帧全中，是实测扫出来的（见 qr_vision.lock_focus 的表）。"
                         f"给负数 = 不锁，退回等自动对焦。"
                         f"换机位/改工作高度后要重扫。")
    ap.add_argument("--port", default=None, help="机械臂串口，默认自动查找")
    ap.add_argument("--mode", default="movl", choices=("movl", "movj"),
                    help="movl 直线（默认，XY 微调时 Z 不会跑）")
    ap.add_argument("--limits", default=None,
                    help='覆盖软限位，如 "x:50:380,y:-310:310"')
    ap.add_argument("--clear-alarms", action="store_true",
                    help="报警清不掉时强制清（确认已脱离卡住状态再用）")
    ap.add_argument("--yes", action="store_true", help="跳过确认回车")
    ap.add_argument("--selftest", action="store_true", help="离线自检，不连相机不连臂")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.selftest:
        return selftest()
    if not args.color:
        print(f"✗ 要指定瞄哪个颜色的方块: --color {'|'.join(cvis.CUBE_COLORS)}")
        return 2
    if args.color not in cvis.CUBE_COLORS:
        print(f"✗ 不认识的颜色 {args.color!r}，只有 {'/'.join(cvis.CUBE_COLORS)}")
        return 2
    if args.go and args.image:
        print("✗ --go（真动）和 --image（只看图）是两条路，分开跑。")
        return 2
    return run_aim(args) if args.go else report_only(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[已中止]")
        sys.exit(130)
