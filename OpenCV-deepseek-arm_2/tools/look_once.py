#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
look_once.py —— **拍一张照片**，把四个方块的机械臂坐标算出来写进
output/world_state.json，然后关掉相机。之后整条抓放流程交给 src/main.py，
摄像头再也用不着。

★ 为什么「一张就够」—— ③ 那个闭环本来就是多余的
  ③ 每一步算的是 δ = A⁻¹·[ c·(方块像素−C) − (码像素−C) ]，而 δ 是**增量**。
  把臂当时的坐标 u 加回去:

      u* = u + δ = [ u − A⁻¹·(码像素−C) ] + A⁻¹·c·(方块像素−C)
                   └───────── K: 这**一张照片**就定死 ─────────┘

  K 对整张照片是一个常数（只取决于「拍的时候臂报的坐标」和「它的码落在哪个
  像素」）。K 一旦定下来，画面里**任何一个像素**都能直接换成机械臂坐标:

      机械臂坐标(p) = K + A⁻¹·c·(p − C)

  四个方块的像素一起代进去 → 四个机械臂坐标同时拿到。一次拍摄，全部算完。

★ 为什么这条比闭环严格更好
  · 遮挡问题**根本不存在**: 拍这一张时臂停在 home（停在一边），什么都不挡。
    ③ 是**边瞄边拍**，所以越靠近越看不见 —— 那是流程造出来的问题，不是几何必然。
    为了绕开它加的那套闸门（LOST_OK_MM / 近距丢失收工）在这里全都用不上了。
  · 臂停哪儿都行: 换个停点，码的像素跟着变，K 重新一减就抵消了，算出来的方块
    坐标不变（方块自己**不在动** —— 臂横移不会挪动桌上的方块）—— 自检里钉了这条。
    ★ 前提是那几个码都在**画面里**，而这一条对吸盘码最紧: 它挂在 z_qr 那一层
      （比桌面近 z_qr 毫米），视场小得多 —— 臂停远了它先出画，纸面码还在。
      home 位天然满足: ①（measure_suction_map --go）就是在那个位子上同时量到
      吸盘码和四个纸面码的，量不到它当时就报错不干了。
  · 光心正下方是哪个机械臂坐标（P_C）**不用知道**，也不必量: 它整体被 K 吃掉。
  · 精度 = A 的精度，一发定。没有第二次测量来修正，也没有第二次测量来添乱。
  · ★ **但 A 的「高度」必须先折算** —— 这是全脚本唯一一处「③ 的结论不能照搬」的
    地方。③ 的 A⁻¹ 出现在**增量**里，且 (码像素−C) 与 c·(方块像素−C) 是同一瞬
    拍的，它的尺度两边同时出现、自己抵消（所以 rescale_A_inv 的注释说「折不折
    都落得对」）。本脚本是绝对式子，A⁻¹ 只出现一次、还被冻进 K 里，**尺度不再
    抵消** —— 它变成一个以拍照位为中心的**整体放大**:
        λ = k(这张的 z_qr) / k(① 量 A 时的 z_qr)
    默认拍照高度（悬停 30 + 方块 26 → z_qr=176）对 ① 的 z_qr=135 是 λ=1.279:
    不折就是四个方块一起**朝外放大 28%**，离拍照位越远偏得越多，四个象限全都
    朝外 —— 真机上看到的那个 2cm。折算统一在 photo_coeffs 里做，调用处漏不掉。
    这件事由两条自检夹住: 第 11 条**算出**「折了 0.2mm、不折偏 21.6mm」，
    第 13 条**查调用处**有没有把 ① 的高度真传进来（11 够不着 run_hardware，
    那要真机；13 是在原地钉住，不必等真机）。

★ K 到底是什么、为什么每次现算
  K = P_C − eps（P_C = 光轴正下方那个机械臂坐标，eps = 码相对「臂报的坐标」的结构
  偏移）。所以 K 其实是**这套架子（机位 + 码的安装）的常数**，同一个停点摆着不动
  它就不变 —— 换句话说，它**本来是可以存盘的**。
  我们不存，是因为存了就变成手眼矩阵那条路上那个坑: 相机被撞一下，存下来的 K 就
  过期了，而它照样给你算出一整套偏掉的坐标、一路不报错。**从同一张照片里现算**，
  量的和用的是同一瞬的架子，自洽这件事是结构上保证的。

★ 和 src/main.py 怎么接
  main.py 要的 output/world_state.json 就是 {颜色: {x, y, z_level}}（机械臂 mm）。
  它的生成口子是 color_vision.analyze(..., robot=(M, pixel_to_robot))，
  **默认那个 M 来自 step3 的手眼标定矩阵** —— 就是被否掉的那条「对刀」路。
  本脚本把那个 callable 换成上面的 K 式子；其余（z_level、near 戳、平放复位、
  落盘、.bak 备份）全部复用 color_vision 现成的实现，main.py 一行都不改。

★ 这个脚本**只拍一张**。多帧只是同一个停点上连拍取中位数压抖动
  （吸盘码只有一个点、没有冗余，单帧抖动没人平得掉，见 measure_suction_map
  .analyse_frame），**臂一步都不动**。和 ③ 那种「走一步拍一张」是两件事。

★ 一次拍摄做不到的事（别指望它）
  · **层数**（z_level）相机看不出来 —— 那还得靠记忆库 output/world_state.json
    里上次的层数（prior_z_levels），或者你把方块摆平了再拍（reset_levels_if_flat
    会自己认出来并归零）。
  · **吸盘码→吸嘴**的结构偏移: K 把「像素世界」对齐到**臂报的坐标**，而吸嘴和
    码都在这个参考点上各有一个固定偏移。剩下的那个偏移是个**常数**（机器人 mm），
    量一次就永久有效 —— 它不会像手眼矩阵那样随「方块摆在纸的哪一角」变向。
  · 码的像素抖动会**整体**平移四个坐标: 1px ≈ 0.2mm（自检里印了这个数），
    所以吸盘码必须走多帧中位数，不能拿单帧。

用法:
    python3 tools/look_once.py                       # 停到 home → 拍一张 → 算 → 写盘
    python3 tools/look_once.py --dry                 # 同上，但不写 world_state.json
    python3 tools/look_once.py --hover 40            # 拍照高度 = 方块顶面 + 40
    python3 tools/look_once.py --image 照片.jpg --pose 199.3 0.0 -3.4
                                                     # 离线: 用现成照片（pose = 拍那
                                                     # 张时臂报的 x y z）
    python3 tools/look_once.py --selftest            # 纯算自检，不碰硬件
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import cv2                                                            # noqa: E402
import numpy as np                                                    # noqa: E402

import aim_suction as aim                                             # noqa: E402
import color_vision as cvis                                           # noqa: E402
import measure_suction_map as msm                                     # noqa: E402
import qr_vision as qv                                                # noqa: E402
from dobot_sdk import find_port, load_sdk                             # noqa: E402
from paths import ensure_output_dir                                   # noqa: E402
from qr_vision import (detect_sweep, load_paper_layout,               # noqa: E402
                       load_suction_code, open_camera, quad_center)

POSE_TOL_MM = 0.5     # 拍这张的**前后**臂动了超过这么多 → K 不可信，中止
RULER_WARN = 0.03     # 纸面标尺自检相对误差超过这个就报警


# ───────────────────────── 纯计算（可离线自检） ─────────────────────────
def offset_K(qr_px, arm_xy, A_inv, C) -> np.ndarray:
    """
    这一张照片的常数 K = 臂当时报的坐标 − A⁻¹·(码像素 − C)。

    ★ 它就是「像素世界」和「机械臂世界」之间的那个平移量，等于 P_C − eps
      （光轴正下方那个机械臂坐标 − 码相对「臂报的坐标」的结构偏移）。所以它是
      **这套架子的常数**，不是「停点的函数」—— 臂挪到别处，码的像素跟着挪同样
      多，两下一减就消掉了（自检里钉了这条）。
    ★ 尽管算出来是常数，还是**每次现算、绝不落盘复用**: 相机被撞一下 P_C 就变了，
      存下来的 K 会给你算出一整套偏掉的坐标还不报错 —— 那正是手眼矩阵的坑。
    """
    return (np.asarray(arm_xy, dtype=float)
            - np.asarray(A_inv, dtype=float) @ (np.asarray(qr_px, dtype=float)
                                                - np.asarray(C, dtype=float)))


def make_px2robot(A_inv, C, c, K):
    """
    造出 color_vision.build_world_state 要的那个 pixel_to_robot 函数。

    ★ 高度差全在 c 里: c = k(码那一层)/k(方块顶面)。方块是**顶面**朝着相机，
      视差按 k(obj_h) 走；而 A 是码那一层量出来的，天生带 k(z_qr)。这里用的
      是 aim_suction.c_coef 算出来的**同一个** c，不是另抄一份公式。
    ★ 签名 (M, px, py) 是 build_world_state 定死的: 它调 px_to_robot(M, px, py)。
      M 是携带参数的那个 dict（这里只用它记账，真实参数已经从闭包进来了）。
    """
    A_inv = np.asarray(A_inv, dtype=float)
    C = np.asarray(C, dtype=float)
    K = np.asarray(K, dtype=float)

    def px2robot(_M, px, py):
        xy = K + A_inv @ (float(c) * (np.array([float(px), float(py)]) - C))
        return float(xy[0]), float(xy[1])

    return px2robot


def photo_coeffs(A_inv_map, map_z_qr, cam_z, z_tip, qr_height, obj_h):
    """
    这一张照片真正要用的系数: (A⁻¹_这张, c, z_qr, g)。

    ★★ 为什么 A⁻¹ 必须跟着高度折 —— 真机上那 2cm 就是漏了这一下
      ③ 那条**增量**律里，(码像素−C) 和 c·(方块像素−C) 是**同一瞬**拍的，A⁻¹ 的
      尺度在两边同时出现，于是 δ=0 的条件里它自己抵消了 —— 所以
      aim_suction.rescale_A_inv 的注释说「折不折都落得对，只是步数问题」。
      **那句话只对增量律成立。** 本脚本是**一发定**的绝对式子:

          r(p) = K + A⁻¹·c·(p − C),     K = u − A⁻¹·(码像素 − C)

      这里 A⁻¹ 只出现一次、还被冻进 K 里，**尺度不再抵消** —— 它等价于「以拍照
      位 u 为中心，把整个桌面缩放 λ 倍」:

          λ = k(这张的 z_qr) / k(① 量 A 时的 z_qr)

      默认参数下 λ = k(176)/k(135) = 1.279: 四个方块一起**朝远离 u 的方向放大
      28%**，离 u 越远偏越多（60~110mm → 偏 1.2~2cm），而且四个象限**全都朝外**。
      真机上看到的就是「左上角的方块吸盘偏左上、右下角偏右下，约 2cm」。

    ★ 为什么把这件事塞进一个函数、而不是在调用处各写一遍:
      正因为它是**漏一行也不报错**的那类错（老代码就是算出了 g、还把它印在
      「折到这儿（÷1.279）」里，然后原样把没折的 A⁻¹ 传下去）。放进一个函数、
      让 compute_and_report 自己调，调用方**没有机会忘**。

    ★ map_z_qr 是 None（老地图没记 ① 量 A 时的高度）→ 不折、g=None，并在报告里
      说清楚: 猜高度同样是错的，宁可把「没折」标出来。
    """
    z_qr = float(z_tip) + float(qr_height)
    A_inv, g = aim.rescale_A_inv(A_inv_map, map_z_qr, z_qr, cam_z)
    return np.asarray(A_inv, dtype=float), aim.c_coef(cam_z, z_qr, obj_h), z_qr, g


def paper_ruler(decoded, layout, A_inv, z_qr, cam_z) -> list[dict]:
    """
    自检: 拿**纸**当尺子，把这张照片的变换量一遍。

    ★ 为什么这是最值钱的一条自检: 四个纸面码之间的距离是**印出来的**，不依赖
      任何标定。把它们按两种路子从同一张照片换成毫米 ——
        · 走 A（机械臂动出来的尺度）: |A⁻¹·k(z_qr)·(pᵢ−pⱼ)|
        · 走纸面单应（纸自己的尺度）: |paper_xy(pᵢ) − paper_xy(pⱼ)|
      —— 应该相等。不等就说明 A 的**尺度**和纸对不上，而 A 错了会让四个方块
      **整体**偏，且一路上不报错。
    ★ 这里用的是 k(z_qr) 而不是方块的 c: 纸面在高度 0，k(0)=1，所以系数是
      k(z_qr)/k(0) = k(z_qr)。
    ★ K 不参与 —— 两两距离跟整体平移无关。所以这条只查尺度，不查那个偏移。
    """
    H = cvis.px_to_paper_map(decoded, layout)
    k = msm.k_at_height(z_qr, cam_z)
    A_inv = np.asarray(A_inv, dtype=float)
    codes = [c for c in layout.codes if c in decoded]
    rows = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            pa = np.asarray(quad_center(decoded[a]), dtype=float)
            pb = np.asarray(quad_center(decoded[b]), dtype=float)
            d_robot = float(np.linalg.norm(A_inv @ (k * (pa - pb))))
            xa, ya = cvis.paper_xy(H, pa[0], pa[1])
            xb, yb = cvis.paper_xy(H, pb[0], pb[1])
            d_paper = float(np.hypot(xa - xb, ya - yb))
            rows.append({"pair": f"{a}–{b}", "robot_mm": d_robot,
                         "paper_mm": d_paper,
                         "rel": (abs(d_robot - d_paper)
                                 / max(1e-9, abs(d_paper)))})
    return rows


def px_jitter_mm(A_inv, c) -> float:
    """
    吸盘码像素抖动 **1px** → 四个方块坐标一起平移多少 mm。

    ★ 为什么这个数非印不可: 它把「K 是从一个单点来的」这件事量化了。吸盘码只有
      一个点、没有冗余，单帧检测噪声直接变成**所有**方块坐标的共模偏移 —— 所以
      它必须走多帧中位数（msm.sample_suction 干的就是这件事）。
    ★ 用**谱范数**（ord=2，最大奇异值），不是 numpy 默认的 Frobenius 范数 ——
      后者会多乘一个 √2，报出来的数白白大 41%（自检里钉了这条）。
      物理含义是「1px 误差落在最坏的那个方向上」。
    """
    return float(np.linalg.norm(np.asarray(A_inv, dtype=float) * float(c), 2))


# ─────────────────────────── 拍那一张 ───────────────────────────
def shoot(cap, detector, layout, suction, frames: int):
    """
    在**当前停点**拍。返回 (frame, qr_px, qr_frames)。

    ★ 多帧只为了压吸盘码的抖动（见 px_jitter_mm），臂一步都不动。
    ★ 方块和纸面码用最后那一帧整帧解一次 —— 它们有冗余（每个码 4 个角、
      共 4 个码），单帧就够，不需要中位数。
    """
    sm = msm.sample_suction(cap, detector, suction,
                            focus_goal=list(layout.codes), frames=frames)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None, 0
    return frame, (sm[0] if sm else None), (sm[2] if sm else 0)


def compute_and_report(frame, qr_px, arm_xy, A_inv_map, C, cam_z, map_z_qr,
                       z_tip, qr_height, obj_h,
                       layout, suction, detector, dry: bool, src: str) -> int:
    """
    一张图 + 臂的坐标 → 四个方块的机械臂坐标 → (可选) 写盘。

    ★ 顺序钉死「先纸面 → 再 ppm → 再方块」: detect_cubes 拿到 ppm 才会按
      **真毫米**筛大小；ppm 又只有解出纸面码才有。和 ③ 的 read_scene 同一套顺序。

    ★ 收进来的是**拍照高度**（z_tip / qr_height），不是算好的 (A⁻¹, c, z_qr):
      A⁻¹ 的高度折算放在本函数里做，调用处就没法漏（详见 photo_coeffs）。
    """
    A_inv, c, z_qr, g = photo_coeffs(A_inv_map, map_z_qr, cam_z,
                                     z_tip, qr_height, obj_h)
    if g is None:
        print("\n✗ 地图里没记 ① 量 A 时吸盘码挂在哪一层（motion.z_qr_mm）——")
        print("  这张照片的 A⁻¹ **折不了**。而本脚本是**一发定**的绝对式子: A⁻¹")
        print("  的尺度不减（它被冻进 K 里），不折就等于「以拍照位为中心整体放大")
        print("  λ = k(这张的 z_qr)/k(① 的 z_qr)」—— 默认拍照高度下 λ≈1.28，")
        print("  四个方块会**一起朝外偏 ~2cm**，而且一路上不报错。")
        print("  这正是本脚本绝不做的事。宁可空跑一次，也不留半套偏掉的坐标给 "
              "main.py。")
        print("  → 重跑 ① 把那个高度补进地图: "
              "python3 tools/measure_suction_map.py --go")
        return 1

    print(f"\n[这张照片的账] 吸嘴离纸面 z_tip={z_tip:.1f}"
          f" + 码高 {qr_height:.0f} → 吸盘码 z_qr={z_qr:.1f}mm")
    print(f"  c = k({z_qr:.0f})/k({obj_h:.0f}) = {c:.4f}"
          f"   A⁻¹ 从 ① 的高度（z_qr={map_z_qr:.0f}）折到这儿（÷{g:.4f}）")

    dec, _ = detect_sweep(frame, detector,
                          want=list(layout.codes) + ([suction] if suction else []))
    hits = [x for x in layout.codes if x in dec]
    marks = "  ".join(("✅ " if x in dec else "❌ ") + x
                      for x in list(layout.codes) + ([suction] if suction else []))
    print(f"\n  ── 二维码 {len(hits) + (1 if suction in dec else 0)}/"
          f"{len(layout.codes) + 1} ──\n   {marks}")

    missing = [x for x in layout.codes if x not in dec]
    if missing:
        print(f"\n✗ 纸面码缺 {missing} —— 一个都不能缺。")
        print("  四码是用来定「纸面尺度」和「每块方块离哪个角码最近」的，缺一个就"
              "没有准确的 ppx/mm，方块大小筛选会退回像素口径、角码补偿也无从判起。")
        print("  把相机抬高/拉远让四个码连白边一起进画面，再跑一次。")
        return 1
    if qr_px is None:
        print("\n✗ 吸盘码没解出来 —— K 就无从算起，**没有 K 就没有任何坐标**。")
        print("  （这不是「少一条记录」，是整张照片作废。）")
        print("  · 焦距锁对了没？默认 FOCUS_LOCK=%d，先用 tools/measure_focus.py 确认。"
              % qv.FOCUS_LOCK)
        print("  · 臂的停点会不会把吸盘码自己挡住、或者挡在画面外？")
        return 1

    ppm = min(qv.px_per_mm(dec[x]) for x in layout.codes)
    cubes = cvis.detect_cubes(frame, ppm)
    if sorted(cubes) != sorted(cvis.CUBE_COLORS):
        print(f"\n✗ 方块没认齐: 只认到 {sorted(cubes)}，要 {sorted(cvis.CUBE_COLORS)}。")
        print("  · 方块都摆出来了吗？被别的方块挡住没有？")
        print("  · 跑 src/color_vision.py --debug 看 output/color_debug.png "
              "哪个颜色的框没套上。")
        return 1

    K = offset_K(qr_px, arm_xy, A_inv, C)
    M = {"A_inv": np.asarray(A_inv, float).tolist(), "C": np.asarray(C, float).tolist(),
         "c": float(c), "K": K.tolist(), "z_qr": float(z_qr)}
    px2robot = make_px2robot(A_inv, C, c, K)

    print(f"\n  吸盘码像素 ({qr_px[0]:.1f}, {qr_px[1]:.1f})"
          f"   ← 多了 {len(layout.codes)} 帧取中位数压抖动")
    print(f"  臂报的坐标 ({arm_xy[0]:.2f}, {arm_xy[1]:.2f})   （{src}）")
    print(f"  → K = ({K[0]:+.2f}, {K[1]:+.2f}) mm")
    print(f"  ★ 吸盘码抖 1px → 四个坐标一起平移 {px_jitter_mm(A_inv, c):.3f}mm"
          f"（所以它必须取中位数、不能拿单帧）")

    # ── 纸面标尺自检: 同一张照片，两种尺子量出来的距离对不对得上 ──
    rows = paper_ruler(dec, layout, A_inv, z_qr, cam_z)
    worst = max((r["rel"] for r in rows), default=0.0)
    print(f"\n  ── 纸面标尺自检（拿印出来的码间距离当尺子）──")
    for r in rows:
        print(f"    {r['pair']:<22} A 量出 {r['robot_mm']:7.2f}mm"
              f"   纸面 {r['paper_mm']:7.2f}mm   差 {r['rel'] * 100:5.2f}%")
    if worst > RULER_WARN:
        print(f"  ⚠ 最大相对差 {worst * 100:.1f}% > {RULER_WARN * 100:.0f}% —— "
              f"A 的**尺度**和纸对不上。")
        print(f"     四个方块会**整体**偏，而且不报错。重跑 ①（measure_suction_map"
              f".py --go）再拍。")
    else:
        print(f"  ✅ 最大相对差 {worst * 100:.2f}% ≤ {RULER_WARN * 100:.0f}%"
              f" —— A 的尺度和纸对得上")

    state = cvis.build_world_state(cubes, "robot", dec, layout,
                                   robot=(M, px2robot), prior=cvis.prior_z_levels())
    flat = cvis.reset_levels_if_flat(state)
    if flat:
        print(f"  ★ {flat}")

    cvis.print_table(state, "robot")
    print(f"\n  z_level 是从 output/world_state.json 抄的（相机看不出层数）。"
          f"摆平了再拍，上面那条会自动归零。")
    print(f"  ★ 吸盘码→吸嘴的结构偏移在这里是个**常数**（机器人 mm），"
          f"量一次就永久有效；\n    它不会像手眼矩阵那样随「方块摆在纸的"
          f"哪一角」变向。")

    if dry:
        print("\n  --dry: 没有写盘。去掉 --dry 才会落 output/world_state.json。")
        return 0
    p = cvis.save_world_state(state)
    print(f"\n  ✅ 已写入 {p}（旧的自动备份成 .bak）")
    print(f"\n▶ 到这里相机的活就干完了 —— 关掉它，剩下交给 main.py:"
          f"\n    python3 src/main.py")
    return 0


# ─────────────────────────── 主流程 ───────────────────────────
def run_hardware(args) -> int:
    mp, why = aim.load_map(args.map)
    if why:
        print(f"✗ {why}")
        return 1
    A_inv_map, C, cam_z = mp["A_inv"], mp["C"], mp["cam_z"]
    mo = mp["raw"].get("motion") or {}
    home = mo.get("robot_home")
    if not home:
        print(f"✗ 地图里没记 robot_home —— 那是「拍照位」。重跑 ①。")
        return 1
    table_z = (args.table_z if args.table_z is not None
               else (mo.get("table_z") if mo.get("table_z") is not None
                     else msm.load_table_z()))
    if table_z is None:
        print("✗ 不知道纸面 Z。先跑 python3 src/step4_pick_test.py --probe，"
              "或给 --table-z。")
        return 1

    layout = load_paper_layout(msm.PAPER_JSON)
    suction = load_suction_code()
    if not suction:
        print("✗ 读不到吸盘码内容（data/suction_qr.json）—— 先跑 tools/gen_suction_qr.py")
        return 1

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
        dType.SetPTPJointParams(api, aim.SPEED_MM_S, aim.SPEED_MM_S, aim.SPEED_MM_S,
                                aim.SPEED_MM_S, aim.ACC, aim.ACC, aim.ACC, aim.ACC,
                                isQueued=0)
        tc._set_coord_params(api, dType, aim.SPEED_MM_S, aim.ACC)
        dType.SetPTPCommonParams(api, aim.RATIO, aim.RATIO, isQueued=0)

        limits = {a: tuple(v) for a, v in tc.DEF_LIMITS.items()}
        if args.limits:
            for spec in args.limits.split(","):
                ax, lo, hi = spec.split(":")
                limits[ax.strip()] = (float(lo), float(hi))
        limits["z"] = (table_z + args.obj_h, limits["z"][1])

        # ── 停到 home 位（拍照位）。★ 高度也钉死: z_qr 由它决定，而 z_qr 进了 c ──
        target_z = min(table_z + args.obj_h + args.hover, limits["z"][1])
        cur = tc.read_pose_stable(api, dType, 5)
        print(f"\n[当前] {tc.fmt(cur)}")
        far = (abs(cur["x"] - float(home["x"])) > 1.0
               or abs(cur["y"] - float(home["y"])) > 1.0
               or abs(cur["z"] - target_z) > 0.5)
        if far:
            print(f"[停到拍照位] home ({float(home['x']):.2f},{float(home['y']):.2f})"
                  f"  Z={target_z:.2f}"
                  f"（纸面 {table_z:.2f} + 方块高 {args.obj_h:.0f} + 悬停 {args.hover:.0f}）")
            print("  ★ 拍这一张时**臂不能动**，也不该挡住任何码或方块 —— "
                  "K 就是「拍的那一瞬臂在哪」，动了这张就作废。")
            if not args.yes:
                input("       回车开始（Ctrl-C 中止）… ")
            tgt = {"x": float(home["x"]), "y": float(home["y"]),
                   "z": target_z, "r": cur["r"]}
            ok, why2 = tc.check_target(tc.read_pose_stable(api, dType, 5), tgt, limits)
            if ok:
                ok, why2 = tc.move_to(api, dType, tgt, limits, args.mode)
            if not ok:
                print(f"✗ 走位失败: {why2}")
                return 1
            tc.verify_arrival(api, dType, tgt)
        else:
            print("[拍照位] 臂已经在 home 位（XY 差 ≤1mm、Z 差 ≤0.5mm），不动它。")

        z_tip = target_z - table_z     # 这张照片的高度 —— 系数由 compute_and_report 现折
        print(f"\n[拍照高度] 吸嘴离纸面 z_tip={z_tip:.1f}mm"
              f"（① 量 A 时是 {mo.get('z_tip_mm')}mm —— 两个不一样就得折 A⁻¹）")

        cap = open_camera(args.cam, focus=args.focus)
        if not cap.isOpened():
            print("✗ 打不开摄像头")
            return 1
        detector = cv2.QRCodeDetector()

        before = tc.read_pose_stable(api, dType, 5)
        print(f"\n[拍一张] 停点上连拍 {args.frames} 帧压吸盘码抖动，**臂不动** …")
        frame, qr_px, qr_n = shoot(cap, detector, layout, suction, args.frames)
        after = tc.read_pose_stable(api, dType, 5)
        if frame is None:
            print("✗ 相机读不到帧")
            return 1
        moved = float(np.hypot(after["x"] - before["x"], after["y"] - before["y"]))
        if moved > POSE_TOL_MM:
            print(f"\n✗ 拍这张的**前后**臂动了 {moved:.2f}mm > {POSE_TOL_MM}mm —— "
                  f"K 不可信（K 的定义是「拍的那一瞬臂在哪」）。")
            print("  重拍一次；老是这样就查一下是不是有人碰了臂、或者刚才那个"
                  "move_to 还没停稳。")
            return 1
        print(f"  ✅ 拍完了，臂没动（{moved:.3f}mm ≤ {POSE_TOL_MM}mm）"
              f"，吸盘码 {qr_n}/{args.frames} 帧命中")
        arm_xy = (before["x"], before["y"])

        # ★ 把这张**真机**照片存下来 —— 事后唯一能复查的东西。
        #   为什么必须存: 定位偏了的时候，「偏在哪、偏多少」只能靠这一张回看
        #   （四个码认在哪、吸盘码认在哪、方块认在哪、纸的方位对不对）。现场一过
        #   就再也拿不到同一张了；上一轮排查正卡在这儿 —— 手头只有一张测焦点的图，
        #   只能拿它凑合着当"真机照片"用。
        #   ★ 存**原帧**（不画任何框）: 画了框的调试图是给人看的，回看要的是原始像素。
        #   ★ 存图失败不算失败: 该给的结果照样给，只是下次没得回看，所以单说一句。
        shot_png = ensure_output_dir() / "look_once_last.jpg"
        if cv2.imwrite(str(shot_png), frame):
            print(f"  [存图] 原帧留在 {shot_png} —— 定位不对就拿它回看")
        else:
            print(f"  ⚠ 存图失败（{shot_png}）—— 不影响这次定位，但下次没得回看")

        cap.release()
        cap = None
        return compute_and_report(frame, qr_px, arm_xy, A_inv_map, C, cam_z,
                                  mp["map_z_qr"], z_tip, args.qr_height,
                                  args.obj_h, layout, suction, detector,
                                  args.dry, "臂实测")
    finally:
        if cap is not None:
            cap.release()
        try:
            dType.DisconnectDobot(api)
        except Exception:
            pass


def run_image(args) -> int:
    mp, why = aim.load_map(args.map)
    if why:
        print(f"✗ {why}")
        return 1
    A_inv_map, C, cam_z = mp["A_inv"], mp["C"], mp["cam_z"]
    layout = load_paper_layout(msm.PAPER_JSON)
    suction = load_suction_code()
    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"✗ 读不到图片 {args.image}")
        return 1
    if args.pose is None:
        print("✗ --image 必须同时给 --pose X Y Z（拍那张时臂报的坐标）——")
        print("  没有它就没有 K，算不出任何机械臂坐标。")
        return 1
    px, py, pz = (float(v) for v in args.pose)
    table_z = args.table_z if args.table_z is not None else msm.load_table_z()
    if table_z is None:
        print("✗ 不知道纸面 Z（用于从 pose 的 z 反推 z_qr）。给 --table-z。")
        return 1
    z_tip = pz - table_z
    print(f"[离线] {args.image}")
    print(f"  pose ({px:.2f}, {py:.2f}, {pz:.2f}) → 吸嘴离纸面 z_tip={z_tip:.1f}")
    detector = cv2.QRCodeDetector()
    dec, _ = detect_sweep(frame, detector,
                          want=list(layout.codes) + ([suction] if suction else []))
    qr_px = quad_center(dec[suction]) if (suction and suction in dec) else None
    return compute_and_report(frame, qr_px, (px, py), A_inv_map, C, cam_z,
                              mp["map_z_qr"], z_tip, args.qr_height,
                              args.obj_h, layout, suction, detector,
                              True, "命令行给的")


# ─────────────────────────── 离线自检 ───────────────────────────
def selftest() -> int:
    print("look_once.py 自检")
    n = [0, 0]

    def check(name, cond, detail=""):
        n[1] += 1
        if cond:
            n[0] += 1
        print(f"  {'✅' if cond else '❌'} {name}" + (f"   {detail}" if detail else ""))

    Zc, s = 323.0, 4.575          # 相机高度 / 纸面 px每mm（和 ① 同一套口径）
    z_qr, obj_h = 176.0, 26.0
    c = aim.c_coef(Zc, z_qr, obj_h)
    k_qr = msm.k_at_height(z_qr, Zc)
    k_obj = msm.k_at_height(obj_h, Zc)
    th = np.deg2rad(37.0)                       # 机位是转的 —— 转着测才有意义
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = s * k_qr * R
    A_inv = np.linalg.inv(A)
    C = np.array([1004.6, 589.1])
    # ★ P_C = 光轴正下方那个**机械臂坐标**。真机上没人去量它（也不用），
    #   这里给个任意值，专门用来证明「不量也一样准」。
    P_C = np.array([175.0, 25.0])
    W = {"red": (42.28, 35.20), "yellow": (-30.0, 55.0),
         "green": (10.0, -40.0), "blue": (-60.0, -20.0)}
    layout = load_paper_layout(msm.PAPER_JSON)
    suction = load_suction_code()        # 端到端那一段要拿它的**内容**去生成码图

    def scene(u, eps=np.zeros(2)):
        """正演: 臂停在 u，码相对臂有 eps 的结构偏移 → 一帧里各自的像素。

        ★ P_C（光轴正下方那个机械臂坐标）故意给个**没人知道**的值: 真实系统里
          没有人会去量它，而下面的自检要证明「不量也一样准」。
        """
        qr = C + A @ (np.asarray(u, float) + eps - P_C)
        cubes = {k: C + s * k_obj * R @ (np.asarray(v, float) - P_C)
                 for k, v in W.items()}
        return qr, cubes

    def recover(u, eps=np.zeros(2)):
        qr, cubes = scene(u, eps)
        K = offset_K(qr, u, A_inv, C)
        px2 = make_px2robot(A_inv, C, c, K)
        return K, {k: np.array(px2(None, p[0], p[1])) for k, p in cubes.items()}

    # 1) ★ 一张照片 → 四个方块一起拿到（不是只拿目标那一个）
    _, got = recover((199.345, 0.0))
    err = max(float(np.linalg.norm(got[k] - np.asarray(W[k], float))) for k in W)
    check("★★ 一张照片 → **四个**方块的机械臂坐标一起拿到",
          err < 1e-9,
          f"四个里最大的误差 {err:.2e}mm（红 "
          f"{got['red'][0]:.2f},{got['red'][1]:.2f}）")

    # 2) ★★ 光轴正下方是哪个机械臂坐标（P_C）**不用知道** —— 它整体被 K 吃掉
    P_C[:] = (120.0, 70.0)                   # 换一个「没人知道」的光轴落点
    _, got2 = recover((199.345, 0.0))
    d2 = max(float(np.linalg.norm(got2[k] - np.asarray(W[k], float))) for k in W)
    check("★★ 光心落在哪个机械臂坐标不用量 —— P_C 被 K 整体吃掉",
          d2 < 1e-9, f"P_C 换成 (120,70) 结果照样准，最大差 {d2:.2e}mm")

    # 3) ★★ 臂换个停点 → 码的像素跟着变，两下一减就消掉，坐标**一模一样**
    qr_a, _ = scene((199.345, 0.0))
    qr_b, _ = scene((150.0, -80.0))
    Ka, gota = recover((199.345, 0.0))
    Kb, gotb = recover((150.0, -80.0))
    db = max(float(np.linalg.norm(gotb[k] - np.asarray(W[k], float))) for k in W)
    check("★★ 臂换个停点 → 码像素挪了、K 抵消掉，坐标**一模一样**",
          db < 1e-9 and float(np.linalg.norm(gotb["red"] - gota["red"])) < 1e-9
          and float(np.linalg.norm(qr_b - qr_a)) > 1.0,
          f"码像素挪了 {float(np.linalg.norm(qr_b - qr_a)):.0f}px，"
          f"K=({Ka[0]:+.1f},{Ka[1]:+.1f})→({Kb[0]:+.1f},{Kb[1]:+.1f})，"
          f"坐标最大差 {db:.2e}mm")

    # 4) ★ 码→吸嘴的结构偏移 = 四个坐标**整体**平移同一个常数（可标定、不会变向）
    eps = np.array([3.0, -4.0])
    _, got3 = recover((199.345, 0.0), eps)
    shifts = {k: got3[k] - np.asarray(W[k], float) for k in W}
    spread = max(float(np.linalg.norm(shifts[k] - shifts["red"])) for k in W)
    check("★ 码→吸嘴的结构偏移 → 四个坐标**整体**平移同一个常数（能一次标定掉）",
          spread < 1e-9
          and abs(float(np.linalg.norm(shifts["red"] + eps))) < 1e-9,
          f"四块都平移 ({shifts['red'][0]:+.2f},{shifts['red'][1]:+.2f})mm"
          f" = −eps，彼此差 {spread:.1e}mm")

    # 5) ★ 纸面标尺: A 的尺度错 5% 就会被它抓出来（K 不参与，只查尺度）
    ppm_paper = s                            # ★ 纸躺在**桌面**上，k(0)=1 → 就是 s
    off = np.array([520.0, 410.0])
    th2 = np.deg2rad(20.0)
    R2 = np.array([[np.cos(th2), -np.sin(th2)], [np.sin(th2), np.cos(th2)]])

    def fake_quads():
        out = {}
        for code in layout.codes:
            cx, cy = layout.mm_for("center")[code]
            p = R2 @ (np.array([cx, cy]) * ppm_paper) + off
            h = 20.0
            out[code] = np.array([[p[0] - h, p[1] - h], [p[0] + h, p[1] - h],
                                  [p[0] + h, p[1] + h], [p[0] - h, p[1] + h]])
        return out

    quads = fake_quads()
    good = max(r["rel"] for r in paper_ruler(quads, layout, A_inv, z_qr, Zc))
    bad_inv = np.linalg.inv(1.05 * A)
    bad = max(r["rel"] for r in paper_ruler(quads, layout, bad_inv, z_qr, Zc))
    check("★ 纸面标尺: A 对 → 差 ≈0；A 尺度错 5% → 差 ≈4.8%",
          good < 1e-6 and 0.03 < bad < 0.08,
          f"对 {good * 100:.4f}% / 错5% → {bad * 100:.2f}%（闸值 {RULER_WARN * 100:.0f}%）")

    # 6) ★ 抖动换算: 这个数印出来是给人看的，别让它悄悄变
    j = px_jitter_mm(A_inv, c)
    check("★ 吸盘码抖 1px → 四个坐标一起偏 ~0.20mm（所以要取中位数）",
          abs(j - 1.0 / (s * k_obj)) < 1e-9,
          f"{j:.3f}mm/px = 1/({s:.3f}×{k_obj:.4f})")

    # 7) c 用的是 ③ 那个 c_coef，不是另抄一份 —— 抄错了 c 会整体偏
    c_tip = msm.k_at_height(15.0, Zc) / k_obj
    check("★ c 走 aim_suction.c_coef（和 ③/④ 同一个），不是另抄一份",
          abs(c - k_qr / k_obj) < 1e-12 and abs(c_tip - c) > 0.4,
          f"c={c:.4f}；拿错的那个（k_tip 口径）是 {c_tip:.4f}，"
          f"会整体偏到 {c_tip / c * 100:.0f}%")

    # ═══════════ 8~11) 端到端: 画一张合成照片，整条通路真跑一遍 ═══════════
    # ★ 上面 1~7 喂进去的都是「按公式算出来的像素」—— 那只验了算术。这一段画一张
    #   **真的图片**，让**真** cv2.QRCodeDetector 解码、**真** HSV 找色块、**真** 的
    #   ppm 换算全走一遍。接错线（ppm 拿错、方块像素没进 c、K 拿错停点、码和块顺序
    #   颠倒）只有这一段会红。
    if not hasattr(cv2, "QRCodeEncoder_create") or not suction:
        print("  ⏭ 缺 QRCodeEncoder（OpenCV 4.5.3+ 才有）或 data/suction_qr.json"
              " → 跳过 8~11 端到端几条")
    else:
        W_s = {"red": (42.0, 35.0), "yellow": (-30.0, 55.0),
               "green": (10.0, -40.0), "blue": (-60.0, -20.0)}
        # 每种颜色取 HSV 区间正中间那一档来画（画偏了就落出 HSV_RANGES，自检会红）
        HSV_MID = {"red": (5, 200, 200), "yellow": (27, 200, 200),
                   "green": (60, 200, 200), "blue": (115, 200, 200)}
        P_C_s = np.mean([np.asarray(v, float) for v in W_s.values()], axis=0)
        HOME_s = P_C_s + np.array([25.0, -10.0])   # 码得在画面里（① 才量得到它）
        enc = cv2.QRCodeEncoder_create()
        mod = 6                                     # 每模块几像素（整张图的唯一尺度）

        def render(text):
            m = enc.encode(text)
            m = cv2.copyMakeBorder(m, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
            return cv2.resize(m, None, fx=mod, fy=mod, interpolation=cv2.INTER_NEAREST)

        def paste(fr, img, cx, cy):
            h, w = img.shape
            x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
            fr[y0:y0 + h, x0:x0 + w] = img[:, :, None] if fr.ndim == 3 else img

        # ★ s 由「画出来的码、检测到的四边形边长」反推 —— 不假设，量出来。
        #   这样图、A、c 三者天生自洽，剩下的误差就只剩解码本身那零点几毫米。
        probe = np.full((400, 400), 255, np.uint8)
        paste(probe, render(layout.codes[0]), 200, 200)
        d0, _ = detect_sweep(probe, cv2.QRCodeDetector(), want=[layout.codes[0]])
        s_s = qv.px_per_mm(d0[layout.codes[0]])
        A_s = s_s * k_qr * R
        A_inv_s = np.linalg.inv(A_s)

        def synth(with_suction=True, drop_cube=None, A_qr=None):
            """正演一张整个桌面的照片（就是本脚本公式的反向）。

            ★ 方块**顶面**用 k_obj、码那层用 k_qr —— 和 make_px2robot 里 c 的口径
              是同一个，所以画出来的图对不上就说明 c 用错了。
            ★ A_qr: 吸盘码那一层的 px/机械臂mm。默认 A_s（码高 = z_qr）。
              11) 拿它把「① 量 A 的高度」和「这次拍照的高度」分开。
            """
            fr = np.full((1080, 1920, 3), 40, np.uint8)
            mid_p = np.mean([layout.mm_for("center")[x] for x in layout.codes], axis=0)
            th_p = np.deg2rad(6.0)                  # 纸自己在画面里大概是正的
            R_p = np.array([[np.cos(th_p), -np.sin(th_p)],
                            [np.sin(th_p), np.cos(th_p)]])
            for code in layout.codes:
                q = np.asarray(layout.mm_for("center")[code], float) - mid_p
                px = C + s_s * R_p @ q
                paste(fr, render(code), px[0], px[1])
            for color, w in W_s.items():
                if color == drop_cube:
                    continue
                p = C + s_s * k_obj * R @ (np.asarray(w, float) - P_C_s)
                half = 0.5 * 30.0 * s_s * k_obj     # 30mm 的方块，顶面上看
                bgr = cv2.cvtColor(np.uint8([[[*HSV_MID[color]]]]),
                                   cv2.COLOR_HSV2BGR)[0, 0]
                cv2.rectangle(fr, (int(p[0] - half), int(p[1] - half)),
                              (int(p[0] + half), int(p[1] + half)),
                              tuple(int(v) for v in bgr), -1)
            qp = None
            if with_suction:
                qp = C + (A_s if A_qr is None else A_qr) @ (HOME_s - P_C_s)
                paste(fr, render(suction), qp[0], qp[1])
            return fr, qp

        def run_report(fr, qp, A_inv_map=A_inv_s, map_z_qr=z_qr, z_tip=0.0):
            """跑真 compute_and_report，把它的打印收进字符串、把那张表截下来。

            ★ 截的是**传进 print_table 的那个 state**，不是 stdout 的文本 ——
              文本格式改一下（对齐、加列）不该让自检红。
            ★ 8~10 默认「地图高度 = 拍照高度」（z_tip=0 + 码高 176 = 176 = z_qr），
              于是折与不折一样（g=1）—— 故意这么设，好让那三条只验通路本身。
              高度不一致那一档归 11) 专管。
            ★ 崩了**不往外抛**，收成 -1 交回去: 变体（比如把闸门拆了）崩在
              compute_and_report 里时，整条自检会当场死掉、连 ❌ 都来不及印 ——
              那就看不出是「哪一条」在报警了。收成 -1，判红的那条照样判红，
              而且「拒」和「崩」还能分开数（-1 ≠ 1）。
            """
            seen = {}
            raw_table = cvis.print_table

            def spy(state, mode):
                seen.update(state)
                raw_table(state, mode)

            cvis.print_table = spy
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    rc = compute_and_report(fr, qp, HOME_s, A_inv_map, C, Zc,
                                            map_z_qr, z_tip, z_qr - z_tip,
                                            obj_h, layout, suction,
                                            cv2.QRCodeDetector(), True, "合成")
            except Exception as e:                  # 崩 = 最响的失败，不是「没事」
                rc = -1
                buf.write(f"\n[这里本该「拒」却崩了] {type(e).__name__}: {e}\n")
            finally:
                cvis.print_table = raw_table
            return rc, buf.getvalue(), seen

        def max_err(got):
            """四块里最大的坐标误差（mm）。"""
            return max((float(np.hypot(got[k]["x"] - W_s[k][0],
                                       got[k]["y"] - W_s[k][1]))
                        for k in W_s if k in got), default=float("nan"))

        # 8) ★★ 好照片: 一次拍摄 → 四块坐标，误差就是解码本身的零点几毫米
        frame_ok, qr_ok = synth()
        rc8, out8, got8 = run_report(frame_ok, qr_ok)
        err8 = max_err(got8)
        K8 = offset_K(qr_ok, HOME_s, A_inv_s, C)
        check("★★ 端到端（合成照片）: 一次拍摄 → 四块坐标全对",
              rc8 == 0 and len(got8) == 4 and err8 < 1.0,
              f"返回 {rc8}，四块最大误差 {err8:.3f}mm"
              f"；K=({K8[0]:+.2f},{K8[1]:+.2f}) 正好等于合成里那个"
              f"**没人量过**的光心落点 P_C=({P_C_s[0]:+.2f},{P_C_s[1]:+.2f})")

        # 9) ★ 端到端负例: 吸盘码不在画面里 → 整张作废（不是「少一条记录」）
        frame_ns, _ = synth(with_suction=False)
        rc9, out9, _ = run_report(frame_ns, None)
        check("★ 端到端: 吸盘码不在画面里 → 整张作废、不硬算",
              rc9 == 1 and "吸盘码没解出来" in out9,
              "四个纸面码都在、方块也在，照样拒 —— 没有 K 就没有坐标")

        # 10) ★ 端到端负例: 少一块颜色 → 整张作废
        frame_nc, qr_nc = synth(drop_cube="blue")
        rc10, out10, _ = run_report(frame_nc, qr_nc)
        check("★ 端到端: 只认到 3 块颜色 → 整张作废、不硬算",
              rc10 == 1 and "方块没认齐" in out10,
              "少一块就不写盘 —— 宁可空跑一次，也不留半套坐标给 main.py")

        # 11) ★★ 拍照高度 ≠ ① 量 A 的高度 → 折过 A⁻¹ 后四块照样准；不折就整体放大
        #   ★ 为什么这条非有不可: 2026-09-20 真机上那 2cm 就是这个 ——
        #     run_hardware 只把 g 算出来、印在「折到这儿（÷1.279）」里，却把**没折**的
        #     A⁻¹ 传了下去。8~10 三条都让「地图高度 = 拍照高度」（g=1），所以照不出来；
        #     把两个高度**分开**才是这件事的结构特征。δ 律那边折不折都对，所以这条
        #     只有在**一发定**的绝对式子上才立得住 —— 正是本脚本用的那条。
        z_qr_map11 = 135.0                 # ① 量 A 时: 吸嘴 15 + 码高 120
        A_inv_map11 = np.linalg.inv(s_s * msm.k_at_height(z_qr_map11, Zc) * R)
        A_inv_fold11, c_fold11, z_qr_fold11, g11 = photo_coeffs(
            A_inv_map11, z_qr_map11, Zc, 56.0, 120.0, obj_h)
        fr11, qp11 = synth(A_qr=s_s * msm.k_at_height(z_qr_fold11, Zc) * R)
        rc11a, _, got11a = run_report(fr11, qp11, A_inv_map11, z_qr_map11, 56.0)
        err11a = max_err(got11a)

        # 「没折」会有多偏: 拿**生产件**（offset_K + make_px2robot）喂**没折**的 A⁻¹，
        # 四块像素按 synth 里那套正演摆出来，直接量。这一步是为了把数字钉死 ——
        # 真机上看到的是 2cm，这里必须量出同一个量级，否则这条自检就没有说服力。
        K_raw = offset_K(qp11, HOME_s, A_inv_map11, C)
        px2_raw = make_px2robot(A_inv_map11, C, c_fold11, K_raw)
        err11raw = max(
            float(np.linalg.norm(
                np.array(px2_raw(None, *(C + s_s * k_obj * R
                                        @ (np.asarray(w, float) - P_C_s))))
                - np.asarray(w, float)))
            for w in W_s.values())

        # 老地图没记 z_qr（折不了）→ 必须**拒**，不许硬算一套偏掉的坐标。
        #   ★ 钉的是「拒」而且要**说清为什么**、不是 rc != 0 就算过: 把闸门拆了
        #     （或把折的那行改坏）时 compute_and_report 会崩在印 z_qr 那一句上，
        #     崩出来也是非 0 —— 只看 rc 的话，「说清了理由的拒」和「崩了」就分不开。
        rc11b, out11b, _ = run_report(fr11, qp11, A_inv_map11, None, 56.0)
        check("★★ 拍照高度 ≠ ① 量 A 的高度: 折了照样准；折不了就拒（不硬算）",
              rc11a == 0 and err11a < 1.0
              and rc11b == 1 and "折不了" in out11b and 10.0 < err11raw < 30.0,
              f"折了 {err11a:.3f}mm（g=k(176)/k(135)={g11:.4f}）；"
              f"没折会偏 {err11raw:.1f}mm —— 离拍照位越远越偏、四个象限一起朝外，"
              f"真机上就是那 2cm；地图缺 z_qr_mm 时拒（返回 {rc11b}）")

    # 12) ★ shoot() 的调用处和签名对不对得上（真机那条路的第一步就是它）
    #   ★ 为什么非钉不可，而且必须**查调用处**: 2026-09-20 真机上栽的就是这个 ——
    #     shoot 的形参删掉一个，run_hardware 里的调用没跟着删，于是在
    #     「停点上连拍 30 帧」那一刻才 TypeError 炸掉，前面停位、开相机全白跑。
    #     8~11 走的是 compute_and_report，根本不经过 shoot，所以那几条拦不住；
    #     flake8 / pyflakes **也不看调用元数**（静态检查只查名字）。所以这里直接
    #     拿 ast 把本文件里每个 shoot(...) 抠出来，用真签名 bind 一遍。
    import ast
    import inspect as _inspect
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    sig = _inspect.signature(shoot)
    calls = [nd for nd in ast.walk(tree)
             if isinstance(nd, ast.Call) and isinstance(nd.func, ast.Name)
             and nd.func.id == "shoot"]
    bad12 = []
    for nd in calls:
        if any(isinstance(a, ast.Starred) for a in nd.args):
            continue                      # 解包出来的没法数，跳过
        try:
            sig.bind(*[None] * len(nd.args),
                     **{k.arg: None for k in nd.keywords if k.arg})
        except TypeError as e:
            bad12.append(f"第 {nd.lineno} 行: {e}")
    check("★ 每个 shoot(...) 调用处都能被它的签名接住（真机第一步）",
          bool(calls) and not bad12,
          f"查到 {len(calls)} 个调用处，都对得上"
          if not bad12 else "对不上 → " + "；".join(bad12))

    # 13) ★ 每个 compute_and_report(...) 调用处，都得把「① 量 A 时的高度」**真**传进去
    #   ★ 为什么 11) 不够、非查调用处不可: 11) 只证明「传进去之后折得对」和「传 None
    #     会拒」—— 它够不着 run_hardware（那要真机）。而 2026-09-20 那个 bug 恰恰栽在
    #     调用处: run_hardware 把 g 算了、印了「折到这儿（÷1.279）」，却没让它落到
    #     A⁻¹ 上。把「传了没有」在**原地**钉住，就不必等真机才暴露。
    sig2 = _inspect.signature(compute_and_report)
    k_map = list(sig2.parameters).index("map_z_qr")
    calls2 = [nd for nd in ast.walk(tree)
              if isinstance(nd, ast.Call) and isinstance(nd.func, ast.Name)
              and nd.func.id == "compute_and_report"]
    bad13 = []
    for nd in calls2:
        if any(isinstance(a, ast.Starred) for a in nd.args):
            continue
        try:
            sig2.bind(*[None] * len(nd.args),
                      **{k.arg: None for k in nd.keywords if k.arg})
        except TypeError as e:
            bad13.append(f"第 {nd.lineno} 行: {e}")
            continue
        if len(nd.args) > k_map:
            arg = nd.args[k_map]
        else:
            arg = next((k.value for k in nd.keywords
                        if k.arg == "map_z_qr"), None)
        if arg is None:
            bad13.append(f"第 {nd.lineno} 行: 没传 map_z_qr —— A⁻¹ 就折不了")
        elif isinstance(arg, ast.Constant) and arg.value is None:
            bad13.append(f"第 {nd.lineno} 行: map_z_qr 写死成 None —— 折不了")
    check("★ 每个 compute_and_report(...) 调用处都把「① 量 A 的高度」真传进去",
          len(calls2) >= 2 and not bad13,
          f"查到 {len(calls2)} 个调用处（真机 / 离线 / 自检各一），都传了高度"
          if not bad13 else "有漏 → " + "；".join(bad13))


    print(f"\n{'✅ 全部通过' if n[0] == n[1] else '❌ 有失败'}（{n[0]}/{n[1]}）")
    return 0 if n[0] == n[1] else 1


# ─────────────────────────── 命令行 ───────────────────────────
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="拍一张照片 → 四个方块的机械臂坐标 → output/world_state.json。"
                    "之后相机就不用了，交给 src/main.py。",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="离线: 用现成照片（要配 --pose）")
    ap.add_argument("--pose", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="--image 时必给: 拍那张时**臂报的**坐标")
    ap.add_argument("--dry", action="store_true", help="算完只打印，不写盘")
    ap.add_argument("--hover", type=float, default=msm.DEFAULT_HOVER_MM,
                    help=f"拍照高度 = 方块顶面 + 这个（默认 {msm.DEFAULT_HOVER_MM:.0f}）")
    ap.add_argument("--obj-h", type=float, default=msm.DEFAULT_OBJ_H_MM,
                    help=f"方块高 mm（默认 {msm.DEFAULT_OBJ_H_MM:.0f}）—— c 用它")
    ap.add_argument("--qr-height", type=float, default=msm.DEFAULT_QR_HEIGHT_MM,
                    help=f"吸盘码中心到吸嘴尖 mm（默认 {msm.DEFAULT_QR_HEIGHT_MM:.0f}）")
    ap.add_argument("--table-z", type=float, default=None,
                    help="纸面 Z，默认读地图 / output/pick_test_result.json")
    ap.add_argument("--frames", type=int, default=msm.ACCUM_FRAMES,
                    help=f"压吸盘码抖动用几帧（默认 {msm.ACCUM_FRAMES}）")
    ap.add_argument("--cam", type=int, default=None, help="摄像头编号，默认自动挑")
    ap.add_argument("--focus", type=int, default=qv.FOCUS_LOCK,
                    help=f"锁死手动焦距（默认 {qv.FOCUS_LOCK}）")
    ap.add_argument("--map", default=None, help="suction_map.json（默认 output/ 下那个）")
    ap.add_argument("--mode", default="movl", choices=("movl", "movj"),
                    help="走位模式（默认 movl）")
    ap.add_argument("--port", default=None, help="串口，默认自动找")
    ap.add_argument("--limits", default=None, help="轴限位覆盖: x:lo:hi,y:lo:hi")
    ap.add_argument("--clear-alarms", action="store_true", help="自动清报警")
    ap.add_argument("--yes", action="store_true", help="不问直接走")
    ap.add_argument("--selftest", action="store_true", help="纯算自检，不碰硬件")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.selftest:
        return selftest()
    if args.image:
        return run_image(args)
    return run_hardware(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[已中止]")
        sys.exit(130)
