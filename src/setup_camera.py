#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_camera.py —— 重新架好摄像头之后，跑**这一条**就够了

干的事（一条命令里做完，中间不用管）:
  1) 手眼标定: 看一眼四个二维码 → 重算「像素 → 机械臂」矩阵
  2) 拍初始方块位置: 同一台相机、同一个矩阵，把四个色块换算成机械臂坐标
  3) 顺手清掉过期的层数（「桌面全平放」判据，见 color_vision.reset_levels_if_flat）

跑完就可以直接说人话了:
    python3 src/main.py "把红色方块放到绿色方块右侧" --go

──────────────────────── 什么时候要跑 ────────────────────────
  · 你把摄像头重新摆过 / 挪过 / 动过 → **必须**跑
  · 你重新摆了方块、想从干净状态开始 → 跑
  · 只是换了条指令、方块没动 → 不用跑，直接 main.py

★ 为什么摄像头一动就必须重标: 矩阵绑死「这台机械臂 + 这个机位」。摄像头一挪，
  旧矩阵就废了 —— 而它**不会报错**，只会让机械臂每次都偏几毫米。
  这是全项目最危险的失败方式，所以宁可多跑一次。

──────────────────────── 设计上的三条硬规矩 ────────────────────────
A. 同一个相机句柄贯穿全程。标定和拍方块共用一次 open_camera 的结果 ——
   中间 release 再 open，可能被挑到另一个 /dev/videoN（内建 vs 外接），
   两帧的像素口径就完全不同，矩阵立刻错。

B. 全有或全无。矩阵和记忆库要么都更新，要么一个字节都不动。
   ★ 落盘顺序是**先记忆库、后矩阵**，不是随便定的:
     记忆库里存的是**机械臂 mm**，main.py 抓取时直接用它、不再过一遍矩阵。
     万一第二步写失败 → 记忆库是新的（坐标对），矩阵是旧的（下次扫描才用得上），
     这一轮抓取仍然是对的。
     反过来先写矩阵的话，失败时会留下「新矩阵 + 旧坐标」，而那套旧坐标是
     **旧矩阵**换算出来的 —— 每一次抓取都偏，还不报错。

C. 不重复实现。标定链路复用 step3，识别链路复用 color_vision。
   抄第二份的下场是两边迟早走样，而走样的方式是**默默算出一个偏掉的结果**。

──────────────────────── 参数 ────────────────────────
  python3 src/setup_camera.py                 # 正常跑（摄像头 + 机械臂就位）
  python3 src/setup_camera.py --image a.jpg   # 不开相机，用一张照片跑完整条链路
  python3 src/setup_camera.py --skip-calib    # 只重拍方块位置（摄像头没动）
  python3 src/setup_camera.py --selftest      # 无硬件自检
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import color_vision as cvv
import qr_vision as qv
from paths import COLOR_DEBUG_PNG, MATRIX_JSON, PAPER_JSON, ROBOT_POINTS_JSON, WORLD_STATE_JSON
from step3_hand_eye_calib import (StillImage, calibrate, check_scale_drift,
                                  load_matrix, pixel_to_robot, ppm_of_points,
                                  report_matrix, resolve_teach_coords, save_matrix)

# ★ 默认容差**不是** step3 的 COORD_TOL_MM(3.0)。step3 自己的 --tol 说明里写着
#   「本次实测四点偏差 7.45mm，放宽到 8.0 才放行」。用 3.0 会让这条命令**每次**
#   都失败在坐标自洽校验上 —— 而那台机器的偏差本来就是这么大。
#   ★ 放宽只是「让它别挡路」，不等于数据自洽: 这个判据在本机已经基本失去
#     分辨力（把 P2/P3 单点挪 20mm 也只报 7.2~7.8mm，与真数据同量级）。
TOL_DEFAULT = 8.0

# 机位漂移告警门槛: 标定帧 和 拍方块帧 的 px/mm 差多少算「摄像头被碰过」。
PPM_DRIFT_WARN = 0.20


# ═══════════════════════ 一、前置检查（开相机之前） ═══════════════════════
def _read_json(path) -> object | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_problems(path=None) -> list[str]:
    """读 robot_points.json 里 step2 对刀时记下的遗留告警（字符串列表）。"""
    d = _read_json(path or ROBOT_POINTS_JSON)
    if not isinstance(d, dict):
        return []
    return [str(x) for x in (d.get("problems") or [])]


def read_matrix_ppm(path=None) -> float | None:
    """读上次标定记下的 px/mm（老文件没有这个键 → None）。"""
    d = _read_json(path or MATRIX_JSON)
    v = d.get("px_per_mm") if isinstance(d, dict) else None
    return float(v) if isinstance(v, (int, float)) else None


def preflight(layout, coords_path, strict: bool):
    """
    开相机之前能查的全查掉 —— 省得白等一轮对焦才报「坐标没填」。
    返回 (coords, scale_json)；不过关返回 None（调用方 return 2）。
    """
    r = resolve_teach_coords(layout, coords_path, title="重架机位 —— 标定 + 拍初始位置")
    if r is None:
        return None
    coords, scale_json = r

    check_scale_drift(layout, coords, scale_json)

    probs = read_problems()
    if not probs:
        print("  对刀记录: robot_points.json 里没有遗留告警 ✅")
        return coords, scale_json
    print(f"\n⚠️ robot_points.json 里还挂着 {len(probs)} 条对刀告警"
          f"（step2 当时记下的）:")
    for p in probs:
        print(f"   {p}")
    print("   ☆ 它们**不拦**本次标定 —— 但记的就是「那四个坐标本身可能偏了几毫米」，"
          "标定会把这个偏差原样搬进矩阵。在意的话回去重跑 step2 对刀。")
    if strict:
        print("\n❌ --strict: 有告警就不往下走。")
        return None
    return coords, scale_json


# ═══════════════════════ 二、采集 ═══════════════════════
def _drain(cap, n: int = 5) -> None:
    """丢掉几帧，让自动曝光 / 白平衡收敛（静止图上也照做，行为才一致）。"""
    for _ in range(n):
        cap.read()


def grab_cube_frame(cap, det, layout, tries: int = 3, timeout: float = 3.0):
    """
    抓一帧「四个码齐全」的画面 —— 只有这种帧才能把色块换算成机械臂坐标。
    返回 (frame, decoded)；凑不齐就抛 RuntimeError（带可操作提示）。

    ★ 为什么先判码、再交给 analyze: analyze 在缺码的帧上会先打一串告警再抛异常，
      重试三轮就是三串 —— 而真正的原因（画面里压根没码/缺一个码）反而淹在里面。

    ★ wait_for_focus 必须给 timeout: 它的循环只受时间约束，画面全黑或静止时
      会一直转下去。不给 timeout 就等于把这条命令挂死在等对焦上。
    """
    decoded: dict = {}
    missing = list(layout.codes)
    for i in range(max(1, tries)):
        _drain(cap, 5)
        frame, _blur, _counts = qv.wait_for_focus(cap, det, layout.codes, timeout=timeout)
        if frame is None:
            print(f"  ⚠ 第 {i + 1}/{tries} 轮：摄像头读不出画面")
            continue
        decoded, _located = qv.detect(frame, det)
        missing = [c for c in layout.codes if c not in decoded]
        if not missing:
            return frame, decoded
        print(f"  ⚠ 第 {i + 1}/{tries} 轮：只解出 {sorted(decoded)}，缺 {missing}")

    raise RuntimeError(
        f"试了 {tries} 轮，四个码还是没凑齐，最后缺 {missing}。\n"
        f"    · 四个码必须**完整**出现在画面里 —— 少一个就换算不出坐标\n"
        f"    · 有码贴到画面边缘被裁掉 → 摄像头抬高/拉远一点\n"
        f"    · 先把机位调好再来: python3 tools/test_camera_qr.py\n"
        f"    · 也可以先拍张照片离线跑: python3 src/setup_camera.py --image 照片.jpg")


def _report_ppm_drift(calib_ppm, cube_ppm, out_matrix) -> None:
    """
    比「标定那一帧」和「拍方块那一帧」的机位（px/mm）。

    ★ 这条只能抓「两步之间摄像头被碰了」—— 两步只隔几秒，通常不会发生。
      真正危险的是**纸/底座被挪过**（那会让矩阵静默偏掉，四点拟合残差恒为 0，
      任何重投影检查都看不出来）。那个脚本无能为力，只能在收尾时再提醒一次。
    """
    old = read_matrix_ppm(out_matrix)

    if calib_ppm is None:
        # --skip-calib: 这次没有标定帧，就拿上次标定记下的 ppm 当基准 ——
        # 这正好是 --skip-calib 押的那个注（「摄像头没动」），能量出来就别放过。
        if old:
            rel = abs(cube_ppm - old) / max(1e-9, old)
            tag = "一致 ✅" if rel <= PPM_DRIFT_WARN else "⚠️ 对不上"
            print(f"\n  机位自检: 上次标定记的 {old:.2f}px/mm，"
                  f"这次拍方块 {cube_ppm:.2f}px/mm（差 {rel * 100:.1f}%）{tag}")
            if rel > PPM_DRIFT_WARN:
                print("     --skip-calib 的前提是「摄像头没动」，但这个数对不上 —— "
                      "去掉 --skip-calib 重新标定一次。")
        return

    rel = abs(cube_ppm - calib_ppm) / max(1e-9, calib_ppm)
    tag = "一致 ✅" if rel <= PPM_DRIFT_WARN else "⚠️ 差得有点多"
    print(f"\n  机位自检: 标定帧 {calib_ppm:.2f}px/mm，"
          f"拍方块帧 {cube_ppm:.2f}px/mm（差 {rel * 100:.1f}%）{tag}")
    if rel > PPM_DRIFT_WARN:
        print("     摄像头在两步之间被碰过？画面糊了？建议重跑一次。")

    if old:
        r2 = abs(calib_ppm - old) / max(1e-9, old)
        tail = ("→ 摄像头确实被挪过，这次重标是必要的 ✅" if r2 > PPM_DRIFT_WARN
                else "→ 机位和上次几乎一样")
        print(f"  和历史比: 上次标定 {old:.2f}px/mm，这次 {calib_ppm:.2f}px/mm"
              f"（差 {r2 * 100:.1f}%）{tail}")


# ═══════════════════════ 三、主流程 ═══════════════════════
def run_once(*, cap, layout, coords, out_matrix: Path, out_world: Path,
             tol: float = TOL_DEFAULT, duration: float = 3.0,
             block_tries: int = 3, block_timeout: float = 3.0,
             skip_calib: bool = False, prior: dict | None = None,
             debug: bool = False) -> int:
    """
    拍 → 算 → 落盘。返回 0 成功 / 1 失败（失败时**一个字节都不写**）。

    参数全部显式传入（而不是去读全局配置）是为了 --selftest 能拿临时目录
    和一个假 cap 把整条链路跑一遍，不给自检留「手动改全局变量」的后门。
    """
    det = cv2.QRCodeDetector()
    M, pixel_pts, calib_ppm = None, None, None

    # ── 1. 标定 ──
    if skip_calib:
        try:
            M = load_matrix(out_matrix)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"\n❌ --skip-calib: 读不出 {out_matrix} 里的矩阵: {e}")
            return 1
        print(f"\n  --skip-calib: 沿用 {out_matrix} 里的矩阵，不重新标定。")
        print("  ★ 只有「摄像头没碰、只挪了方块」时才该这么用；"
              "动过摄像头就得老老实实重标。")
    else:
        print("\n  ── 第 1 步: 手眼标定 ──")
        try:
            M, pixel_pts = calibrate(cap, det=det, layout=layout, coords=coords,
                                     duration=duration, tol_mm=tol)
        except (RuntimeError, ValueError) as e:
            print(f"\n❌ 标定失败: {e}")
            print("   ★ 什么都没写 —— 矩阵和记忆库都保持原样。")
            return 1
        calib_ppm = ppm_of_points(layout, pixel_pts)
        report_matrix(M, layout, coords, pixel_pts)

    # ── 2. 拍方块 ──
    print("\n  ── 第 2 步: 拍方块的初始位置 ──")
    try:
        frame, decoded = grab_cube_frame(cap, det, layout,
                                         tries=block_tries, timeout=block_timeout)
    except RuntimeError as e:
        print(f"\n❌ {e}")
        print("   ★ 什么都没写 —— 矩阵和记忆库都保持原样。")
        return 1

    cube_ppm = ppm_of_points(layout, [qv.reference_point(decoded[c], layout.reference)
                                      for c in layout.codes])
    _report_ppm_drift(calib_ppm, cube_ppm, out_matrix)

    # ── 3. 换算 ──
    #   prior 不传就去读记忆库 —— reset_levels_if_flat 要靠它才知道「哪些层数是过期的」
    if prior is None:
        prior = cvv.prior_z_levels()
    try:
        state, cubes, _ppm, _dec = cvv.analyze(frame, "robot", layout,
                                               robot=(M, pixel_to_robot), prior=prior)
    except (cvv.VisionError, RuntimeError, ValueError) as e:
        print(f"\n❌ 方块没认全: {e}")
        print("   ★ 什么都没写 —— 矩阵和记忆库都保持原样。")
        return 1

    print("\n  算出来的机械臂坐标:")
    cvv.print_table(state, "robot")

    probs = cvv.sanity_check(state, "robot", layout)
    if probs:
        print("\n  ⚠️ 体检发现:")
        for p in probs:
            print(f"     · {p}")
        print("     （不拦路。多半是真有一块被挡住，或两块贴得比方块还近）")
    else:
        print("\n  ✅ 体检没看出毛病")

    if debug:
        try:
            COLOR_DEBUG_PNG.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(COLOR_DEBUG_PNG), cvv.draw_debug(frame, cubes, state, layout))
            print(f"  [调试] 认色框线图 → {COLOR_DEBUG_PNG}")
        except OSError as e:
            print(f"  [调试] 存不下调试图: {e}")

    # ── 4. 落盘（全有或全无；顺序见文件头「规矩 B」）──
    try:
        cvv.save_world_state(state, path=out_world)
        if not skip_calib:
            save_matrix(M, layout, coords, out_matrix, ppm=calib_ppm)
    except OSError as e:
        print(f"\n❌ 写盘失败: {e}")
        print("   注意: 记忆库可能已经写进去了，矩阵还是旧的 —— "
              "再跑一次本命令即可（不会越跑越坏）。")
        return 1

    print("\n" + "═" * 72)
    print("  ✅ 全部完成")
    print("═" * 72)
    print(f"  记忆库（方块在哪）→ {out_world}")
    if not skip_calib:
        print(f"  手眼矩阵          → {out_matrix}")
    print("\n  现在可以直接说人话了:")
    print('     python3 src/main.py "把红色方块放到绿色方块右侧" --go')
    print("\n  两点提醒:")
    print("   · 标定纸和机械臂底座必须是 step2 对刀时的**同一个物理位置**。"
          "挪过的话矩阵会静默偏掉（四点拟合残差恒为 0，看不出来）。")
    print("   · output/last_plan.json 里的旧坐标是上一个矩阵算的，已经作废；"
          "main.py 发现桌面状态变了会重新问 DeepSeek，不用手工删。")
    return 0


# ═══════════════════════ 四、命令行 ═══════════════════════
def main() -> int:
    ap = argparse.ArgumentParser(
        description="重新架好摄像头后跑这一条: 重算手眼矩阵 + 拍初始方块位置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="跑完就能: python3 src/main.py \"把红色方块放到绿色方块右侧\" --go")
    ap.add_argument("--selftest", action="store_true",
                    help="不需要硬件，用合成图验证整条链路（不碰 output/）")
    ap.add_argument("--reference", default=qv.DEFAULT_REFERENCE,
                    choices=qv.REFERENCE_MODES,
                    help=f"参考点，必须和 step2_teach_coords.py 用的一致"
                         f"（默认 {qv.DEFAULT_REFERENCE}）")
    ap.add_argument("--cam", type=int, default=None,
                    help="摄像头序号；默认不填 = 按设备名自动挑外接摄像头")
    ap.add_argument("--image", default=None,
                    help="不开摄像头，改用一张已拍好的照片跑完整条链路"
                         "（照片要能同时看见四个码，且最好是垂直俯拍）")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--coords", default=None,
                    help="机械臂坐标 JSON 文件(可选，覆盖文件内 DOBOT_COORDS)")
    ap.add_argument("--tol", type=float, default=TOL_DEFAULT,
                    help=f"坐标自洽校验的容差(mm)，默认 {TOL_DEFAULT}。"
                         f"★ 不用 step3 的 3.0 —— 这台机器实测四点偏差 7.45mm，"
                         f"3.0 会让这条命令每次都失败。放宽只是不挡路，不等于数据自洽。")
    ap.add_argument("--duration", type=float, default=3.0,
                    help="标定时多帧累积的时长(秒)")
    ap.add_argument("--block-tries", type=int, default=3,
                    help="拍方块那一步重试几轮（默认 3）")
    ap.add_argument("--block-timeout", type=float, default=3.0,
                    help="拍方块时单轮等对焦的上限(秒)，默认 3.0")
    ap.add_argument("--skip-calib", action="store_true",
                    help="不重新标定，沿用磁盘上的矩阵（只挪了方块、没碰摄像头时用）")
    ap.add_argument("--out-matrix", default=None,
                    help=f"矩阵存哪（默认 {MATRIX_JSON.name}）")
    ap.add_argument("--out-world", default=None,
                    help=f"world_state 存哪（默认 {WORLD_STATE_JSON.name}）")
    ap.add_argument("--strict", action="store_true",
                    help=f"{ROBOT_POINTS_JSON.name} 里有遗留告警时直接停（默认只警告）")
    ap.add_argument("--debug", action="store_true",
                    help=f"存一张认色框线图到 {COLOR_DEBUG_PNG.name}")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    layout = qv.load_paper_layout(PAPER_JSON, reference=args.reference)

    r = preflight(layout, args.coords, args.strict)
    if r is None:
        return 2
    coords, _scale_json = r

    out_matrix = Path(args.out_matrix) if args.out_matrix else MATRIX_JSON
    out_world = Path(args.out_world) if args.out_world else WORLD_STATE_JSON

    if args.skip_calib and not out_matrix.exists():
        print(f"\n❌ --skip-calib 但 {out_matrix} 不存在 —— 没有矩阵可以沿用。")
        print("   去掉 --skip-calib 重新标定一次。")
        return 2

    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            print(f"❌ 读不出图片: {args.image}")
            return 2
        cap = StillImage(img)
        print(f"[图片] {args.image}   {img.shape[1]}x{img.shape[0]} px")
    else:
        cap = qv.open_camera(args.cam, args.width, args.height)
        if not cap.isOpened():
            print("❌ 打不开摄像头" + (f" /dev/video{args.cam}" if args.cam is not None
                                      else "（自动挑选失败）"))
            print("   当前系统里的视频节点:")
            print(qv.describe_cameras())
            print("   （也可以先拍一张照片，用 --image 照片.jpg 离线跑）")
            return 2
        print(f"[相机] 已打开 video{args.cam if args.cam is not None else '（自动挑的）'}"
              f"   {args.width}x{args.height}")

    try:
        return run_once(cap=cap, layout=layout, coords=coords,
                        out_matrix=out_matrix, out_world=out_world,
                        tol=args.tol, duration=args.duration,
                        block_tries=args.block_tries,
                        block_timeout=args.block_timeout,
                        skip_calib=args.skip_calib, debug=args.debug)
    except KeyboardInterrupt:
        print("\n⚠️ 被 Ctrl-C 打断 —— 什么都没写。")
        return 130
    finally:
        cap.release()


# ═══════════════════════ 五、离线自检 ═══════════════════════
def _run_capture(**kw):
    """跑一次 run_once 并把它的输出抓下来 —— 自检要断言「层数按 0 算」那行有没有出现。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = run_once(**kw)
    return code, buf.getvalue()


def selftest() -> int:
    """
    用合成图把整条链路跑通 —— 不用相机、不连机械臂、**不碰 output/**。

    ★ 关键设计: 像素→机械臂 用一个**已知的仿射映射**（两轴比例故意不同:
      X +0.5、Y -0.8）。标定会精确恢复出这个仿射，于是「world_state 的 x/y」
      有一条可对答案的真值线。两处刻意的选择:
        · 两轴比例不同 → 把缩放写死成 1:1 的实现在这里立刻露馅；
        · **Y 取负** → 仿射行列式为负。这台的几何是「纸面 y 向下 + 机械臂右手系」，
          正确时行列式**必须是负的**（qr_vision.affine_winding_bad）。取正的会被
          绕向检查当成「P2/P3 标反了」拦下来 —— 那一下正好证明这条检查是活的。
    """
    print("=" * 72)
    print("  setup_camera 离线自检（合成图，不用相机、不连机械臂、不碰 output/）")
    print("=" * 72)
    ok_all = True

    def check(name, cond, extra=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  {'✅' if cond else '❌'} {name}" + (f"   {extra}" if extra else ""))

    layout = qv.load_paper_layout(PAPER_JSON)
    det = cv2.QRCodeDetector()

    frame, want_px, _ = cvv.synth_sheet(layout, cvv.SYNTH_CUBES_MM)
    # ★ 仿射定义在**纸面 mm → 机械臂 mm**上，不是像素上。
    #   为什么: check_coord_consistency 拿「纸面布局」和「机械臂坐标」做仿射拟合，
    #   只有这条链本身是仿射的，拟合残差才是 0 —— 定义在像素上会掺进透视，
    #   一个好端端的合成数据会被判成「某一点教歪了」，白折腾。
    #   两轴取 0.81 / 0.94 = 这台 Magician 实测的机器常数；
    #   Y 取负 → 行列式为负，符合「纸面 y 向下 + 机械臂右手系」，否则绕向检查会拦。
    AX, AY, ATX, ATY = 0.81, -0.94, 150.0, 120.0

    def affine(x_mm, y_mm):
        return AX * x_mm + ATX, AY * y_mm + ATY

    decoded, _ = qv.detect(frame, det)
    if not all(c in decoded for c in layout.codes):
        check("合成图上四个码都解得出", False, f"只解出 {sorted(decoded)}")
        print("\n  ❌ 连合成图都解不出码，后面的不用跑了")
        return 1

    # 「对刀坐标」= 四个码的参考点在纸上的 mm，经已知仿射映射后的值。
    mm_ref = layout.mm_for(layout.reference)
    coords = {c: list(affine(*mm_ref[c])) for c in layout.codes}

    print("\n[1] 全链路: 标定 → 拍方块 → 落盘")
    with tempfile.TemporaryDirectory() as td_s:
        td = Path(td_s)
        out_m, out_w = td / "m.json", td / "w.json"

        # 故意塞一份「层数全是 3」的过期记忆库 —— 图上四块隔得开，判据必须清掉它
        stale = {c: 3 for c in cvv.CUBE_COLORS}
        code, out = _run_capture(
            cap=StillImage(frame), layout=layout, coords=coords,
            out_matrix=out_m, out_world=out_w, tol=TOL_DEFAULT,
            duration=0.3, block_tries=1, block_timeout=0.5, prior=stale)

        check("整条链路跑通，退出码 0", code == 0, f"退出码 {code}")
        got = out_m.exists() and out_w.exists()
        check("矩阵和记忆库都落盘了", got)
        if not got:
            print(out[-2000:])
            return 1

        M = load_matrix(out_m)
        world = json.loads(out_w.read_text(encoding="utf-8"))

        print("\n[2] 坐标对不对（真值 = 已知仿射）")
        detail = {}
        for c in cvv.CUBE_COLORS:
            ex, ey = affine(*cvv.SYNTH_CUBES_MM[c])
            detail[c] = round(float(np.hypot(world[c]["x"] - ex, world[c]["y"] - ey)), 2)
        worst = max(detail.values())
        check(f"world_state 的 x/y 与真值一致（两轴 0.81/0.94 不同，最大差 {worst}mm）",
              worst < 2.0, str(detail))
        # 对照: 要是谁把「纸面 mm」直接当成机械臂坐标喂进去，结果会差得远得多
        swapped = max(abs(world[c]["x"] - cvv.SYNTH_CUBES_MM[c][0])
                      for c in cvv.CUBE_COLORS)
        check("（对照）结果明显不是纸面 mm —— 上一条不是空转", swapped > 20.0,
              f"最小差 {swapped:.0f}mm")

        print("\n[3] 过期的层数要清掉，且必须真的读了你给的记忆库")
        check("四块层数全归 0",
              all(world[c]["z_level"] == 0 for c in cvv.CUBE_COLORS),
              str({c: world[c]["z_level"] for c in cvv.CUBE_COLORS}))
        check("打印里出现了「层数一律按 0 算」并点名原层数",
              "层数一律按 0 算" in out and "原记第 3 层" in out)

        # 对照: 记忆库里本来就是 0 → 不该刷那行无意义的提示（证明那行确实来自 prior）
        _c2, out2 = _run_capture(
            cap=StillImage(frame), layout=layout, coords=coords,
            out_matrix=td / "m2.json", out_world=td / "w2.json", tol=TOL_DEFAULT,
            duration=0.3, block_tries=1, block_timeout=0.5,
            prior={c: 0 for c in cvv.CUBE_COLORS})
        check("（对照）层数本来就是 0 → 不刷提示", "层数一律按 0 算" not in out2)

        print("\n[4] 失败路径: 一个字节都不许写")
        blank = np.full((600, 800, 3), 255, np.uint8)
        codeA, outA = _run_capture(
            cap=StillImage(blank), layout=layout, coords=coords,
            out_matrix=td / "mA.json", out_world=td / "wA.json", tol=TOL_DEFAULT,
            duration=0.2, block_tries=1, block_timeout=0.2, prior={})
        check("负例A（画面里没码）: 退出码 1", codeA == 1, f"退出码 {codeA}")
        check("负例A: 标定就失败了 → 两个文件都没写",
              not (td / "mA.json").exists() and not (td / "wA.json").exists())

        # 负例B: 码全在、标定能过，但方块少一个 → analyze 会抛错
        #        ★ 这条才是「全有或全无」的真正守门测试: 矩阵明明已经算出来了，
        #          只要方块那步失败，它也必须**留在内存里不落盘**。
        partial = frame.copy()
        bx, by = (int(round(v)) for v in want_px["blue"])
        cv2.rectangle(partial, (bx - 160, by - 160), (bx + 160, by + 160),
                      (255, 255, 255), -1)
        codeB, outB = _run_capture(
            cap=StillImage(partial), layout=layout, coords=coords,
            out_matrix=td / "mB.json", out_world=td / "wB.json", tol=TOL_DEFAULT,
            duration=0.2, block_tries=1, block_timeout=0.2, prior={})
        check("负例B（少一块方块）: 退出码 1", codeB == 1, f"退出码 {codeB}")
        check("负例B: 标定成功了，但**矩阵也没被写**（全有或全无）",
              not (td / "mB.json").exists())
        check("负例B: 记忆库也没被写", not (td / "wB.json").exists())
        check("负例B: 报的是「方块没认全」，不是别的错",
              "方块没认全" in outB)

        print("\n[5] 落盘格式")
        check("矩阵能原样读回（load_matrix 往返）", M.shape == (3, 3))
        check("矩阵文件记下了 px/mm（给下次做机位漂移对比）",
              isinstance(read_matrix_ppm(out_m), float))
        check("save_world_state 会留 .bak（再写一次就有了）",
              (lambda: (cvv.save_world_state(world, path=out_w),
                        out_w.with_suffix(".json.bak").exists())[1])())

    print("\n" + "=" * 72)
    if ok_all:
        print("  ✅ 自检全部通过")
        print("  下一步: 把摄像头架好、方块摆平，然后跑 python3 src/setup_camera.py")
    else:
        print("  ❌ 自检未通过 —— 先别看真机，见上面 ❌ 那几行")
    print("=" * 72)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
