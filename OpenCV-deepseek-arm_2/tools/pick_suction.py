#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_suction.py —— ④ 按指令抓方块: ③ 瞄 → 降到方块顶面 → 吸 → 抬 → 移到放置点 → 放
==============================================================================
前三个都是「量」: ① 量 A 和 k，② 量光心 C，③ 把吸嘴瞄到方块正上方（只横移）。
本脚本是这条路线**第一次真的碰东西** —— 第一次下降、第一次开真空、第一次放料。

★ 相对运动路线在这里收口: **抓取点不需要任何映射**
  ③ 收敛的那一刻，吸嘴已经在方块正上方 —— 于是「抓取点」就是**机械臂自己报的
  当前 XY**（同一次读数还给出 Z，直接当升降的起点）。不读 hand_eye_matrix.json、
  不要四个教点、不算绝对坐标。这就是当初加吸盘码的目的: 瞄到了，就不必知道方块
  在机器人坐标里是几。
  · ③ 负责「方块在画面哪儿」→ 换算成「该走多少」
  · ④ 负责「走够了之后的那些动作」—— 它**不关心方块在哪**，只知道自己在哪

★ 抓取那一段**一行都没重写**，全部复用 src/step4_pick_test.py 里已经上机验证过的:
  · s4.suction(api, on)                  → SetEndEffectorSuctionCup（SDK 没包，直接调 .so）
  · s4.cube_top_z(table_z, obj_h, level, press) → 该降到哪个 Z
  连那两条用血换来的规矩也一起继承:

  ★★ 吸盘停在**方块顶面**，不是纸面（step4_pick_test.py:508-513）。
     「降到纸面」就是往方块里扎 --obj-h 那么多: 软唇口顶得住、看着"还能抓"，
     但方块被压 + 机械臂丢步 → 之后零点就不准了。

  ★★ --obj-h 默认 26 里的那 1mm 是**吸盘唇口的压缩量**，不是拿卡尺量的净高
     （step4_pick_test.py:515-517: 26 才吸得住，28 抬空 1mm 吸不到）。
     所以 --press 默认 0 —— 压缩量已经在 obj_h 里了，别再压第二次。

★ 为什么下降**必须**在 XY 收敛之后（本脚本最重要的那条顺序）:
  没收敛就下降 = 吸嘴扎在方块边上、或者扎进两个方块中间。所以 solve_loop 不返回
  到位，这里**一步都不降** —— 见 run_pick 里 aim_ok 那一段，它不是提醒，是闸门。

★ ④ 的悬停高度（方块顶面+30 ≈ 离纸面 56）和 ① 量 A 那个高度（z_tip=15）**不一样**，
  这**不影响落点**，只影响走几步:
    ① 给的 A 里含 k(z_qr_①=135)=1.718，④ 那个高度是 k(176)=2.197
    → k 比 g = 2.197/1.718 = 1.279，即 **A 偏小 28%**，每步超调 28%。
    但代入律里看: (码像素−C) 和 c·(方块像素−C) **都**带同一个 k(z_qr)，
    δ = A⁻¹·s·k(z_qr)·(w_c − u) —— δ=0 仍然正好在 u=w_c。
    即 A 的任何**标量**误差只改每步走多远，**不改不动点**，所以瞄上之后照样压在
    方块上。代价是步数: 超调 28% → 误差每步乘 (g−1)=0.279，
    起步差 50mm 要 5 步才进 0.3mm（13.9→3.9→1.1→0.30→0.084），而 MAX_ITER=8
    只留了 3 步余量。所以 run_pick 里**照折**（aim.rescale_A_inv 把 A⁻¹ ÷g）——
    折完就是「一步到位」，和 ① 那个高度一样。
    能歪掉落点的只有 A 的**方向**误差（轴向/旋转写错了），那属于重跑 ① 的事。

★ 为什么要「先规划成一串腿、再执行」（plan_pick / do_pick 分开）:
  抓取的风险全在**顺序**上（下降前有没有先悬停到位、抬起前有没有先横移、
  放料前有没有先降到放置高度）。把顺序写成一个能打印、能离线断言的列表，比埋在
  一串 if 里安全 —— selftest 就是在逐条盯这个列表。

用法（在**项目根目录**下执行）
----
    python3 tools/pick_suction.py --color red --hold              # ★ 第一次跑这个: 抓起来就停
    python3 tools/pick_suction.py --color red --drop 240,140 --go  # 抓到绝对 XY
    python3 tools/pick_suction.py --color red --drop-rel 0,-60 --go # 放到抓取点旁边 60mm
    python3 tools/pick_suction.py --selftest                      # 离线自检（不连相机不连臂）

  ★ 第一次上机先用 --hold: 它只验证「瞄得准不准 + 吸不吸得住」，不放置。
    放置那一步成功与否会掩盖瞄准的误差 —— 先分开验，才知道偏的是哪一段。

  想看全景（5 个码 + 4 个方块一起认）用:
    python3 src/color_vision.py --camera --suction --paper-mm

安全
----
    · 不给表（--table-z 或 json 里也没有）→ **一步都不动**
    · 丢步报警（0x50~0x5f）是硬闸门: 零点已不可信 → 拒绝，**没有 force 后门**
    · Z 下限锁在 table_z + obj_h − press: 唯一放开的一点是 --press
    · 放置点**在下降之前**就查够不够得着 —— 免得方块举在手上才发现放不下
    · 要交互式终端（stdin 是 tty），不接受管道喂参数直接动臂
    · 退出时一定关真空（finally），不会把泵开着丢下就走
    · 不回零、不写 SetHOMEParams / SetEndEffectorParams（那两项出过事故）
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
sys.path.insert(0, str(HERE))                 # 同目录的 aim_suction / measure_suction_map
sys.path.insert(0, str(HERE.parent / "src"))

import aim_suction as aim                                                    # noqa: E402
import color_vision as cvis                                                  # noqa: E402
import measure_suction_map as msm                                            # noqa: E402
import step4_pick_test as s4                                                 # noqa: E402
from dobot_sdk import find_port, load_sdk                                    # noqa: E402
from paths import PICK_SUCTION_JSON, ensure_output_dir                       # noqa: E402
from qr_vision import (FOCUS_LOCK, load_paper_layout, load_suction_code,     # noqa: E402
                       open_camera)

# ═════════════════════════════ 可调参数 ═════════════════════════════
DEFAULT_PUMP_S = 0.6          # 开真空后等这么久再抬（和 step4 一致）
DEFAULT_PRESS_MM = 0.0        # ★ 额外的预压量。压缩量已经含在 obj_h 里了，默认不加
RELEASE_S = 0.4               # 放气后等这么久再抬（和 step4 一致）


# ─────────────────────── 纯计算（可离线自检） ───────────────────────
def parse_xy(text: str) -> tuple[float, float]:
    """把 "240,140" / "240 140" 解析成两个 float。解析不了就抛 ValueError。"""
    parts = str(text).replace("，", ",").replace(" ", ",").split(",")
    parts = [p for p in parts if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"要写成 x,y 两个数，收到 {text!r}")
    return float(parts[0]), float(parts[1])


def resolve_drop(args, grab_xy) -> tuple[tuple[float, float] | None, str, str | None]:
    """
    把 --drop / --drop-rel / --hold 归结成一个放置点 + 一句人话。

    ★ 三条路互斥、一条都不能少: 不明确说要放哪就不许动臂。默认值最危险 ——
      一个「顺手放到纸心」的默认值，会在人还没想好之前就把方块挪走。
    → (放置点 | None, 描述, 出错原因)
    """
    given = [(bool(args.drop), "--drop"), (bool(args.drop_rel), "--drop-rel"),
             (bool(args.hold), "--hold")]
    picked = [n for f, n in given if f]
    if len(picked) > 1:
        return None, "", f"{'、'.join(picked)} 是三条互斥的路，只能给一个"
    if not picked:
        return None, "", ("要说明抓到之后怎么办: --hold（抬起来就停，第一次先用这个）"
                          " / --drop x,y（绝对 XY） / --drop-rel dx,dy（相对抓取点）")
    if args.hold:
        return (float(grab_xy[0]), float(grab_xy[1])), "原地（--hold: 不放置）", None
    if args.drop:
        try:
            x, y = parse_xy(args.drop)
        except ValueError as e:
            return None, "", f"--drop {e}"
        return (x, y), f"绝对 ({x:.2f}, {y:.2f})", None
    try:
        dx, dy = parse_xy(args.drop_rel)
    except ValueError as e:
        return None, "", f"--drop-rel {e}"
    return ((float(grab_xy[0]) + dx, float(grab_xy[1]) + dy),
            f"抓取点 {dx:+.1f},{dy:+.1f} → ({float(grab_xy[0]) + dx:.2f},"
            f" {float(grab_xy[1]) + dy:.2f})", None)


def z_floor(table_z: float, obj_h: float, press: float) -> float:
    """
    Z 的下限 = 方块顶面（含吸盘该有的那点挤压）。

    ★ 和 step4_pick_test.py:636-639 同一条:「只要知道纸面高度，就把 Z 锁在纸面上，
      抓取时会去碰方块顶面（比纸面高 obj_h），唯一放开的一点是 press」。
      所以这个值既是**能到的最低点**，也正好是**该停的那一点** —— 再往下就是
      往方块里扎，没有第二种解释。
    """
    return s4.cube_top_z(table_z, obj_h, 0, press)


def plan_pick(grab_xy, grab_z, hover_z, drop_xy, drop_z, hold: bool = False) -> list:
    """
    把整个抓取排成一串腿。**先规划、再执行** —— 顺序能在离线自检里逐条断言。

    腿有两种:
      {"kind": "move", "x","y","z","why"}       一步位姿（XY 和 Z 一起下，绝对坐标）
      {"kind": "pump", "on","why","wait"}       开关真空，等 wait 秒

    ★ 三条顺序铁律（selftest 逐条盯着）:
      1. 第一腿一定是**悬停到抓取点上方**（hover_z）—— 绝不允许从半空直接下降。
      2. 开真空那一腿一定在「降到 grab_z」之后。
      3. 放料那一腿一定在「降到 drop_z」之后，且**抬起离开之前**。
    """
    gx, gy = float(grab_xy[0]), float(grab_xy[1])
    legs = [
        {"kind": "move", "x": gx, "y": gy, "z": float(hover_z),
         "why": "悬停到抓取点上方"},
        {"kind": "move", "x": gx, "y": gy, "z": float(grab_z),
         "why": "下降贴住方块顶面"},
        {"kind": "pump", "on": True, "why": "开真空", "wait": None},
        {"kind": "move", "x": gx, "y": gy, "z": float(hover_z),
         "why": "抬起（带着方块）"},
    ]
    if hold:
        return legs
    dx, dy = float(drop_xy[0]), float(drop_xy[1])
    legs += [
        {"kind": "move", "x": dx, "y": dy, "z": float(hover_z),
         "why": "移到放置点上方"},
        {"kind": "move", "x": dx, "y": dy, "z": float(drop_z),
         "why": "下降放置"},
        {"kind": "pump", "on": False, "why": "关真空（放料）", "wait": RELEASE_S},
        {"kind": "move", "x": dx, "y": dy, "z": float(hover_z),
         "why": "抬起离开"},
    ]
    return legs


def do_pick(legs, move_xyz, pump, pump_s: float = DEFAULT_PUMP_S,
            log=print) -> tuple[bool, str | None]:
    """
    执行 plan_pick 排出来的那一串腿。任何一腿失败就**立刻停在那**并返回原因。

    ★ 为什么失败就停、不重试、不"接着走完": 后面每一腿的前提都是前面成了
      （没吸住就横移 = 拖着方块走；没降到放置点就放气 = 方块从半空掉）。
      停住 + 让 finally 关真空，是唯一不用猜的状态。

    pump(on) -> int（0 = 成功）；move_xyz(x, y, z, why) -> bool
    """
    for i, leg in enumerate(legs, 1):
        if leg["kind"] == "move":
            if not move_xyz(leg["x"], leg["y"], leg["z"], leg["why"]):
                return False, f"第 {i} 腿失败: {leg['why']}"
            continue
        rc = pump(leg["on"])
        log(f"  {'✓' if rc == 0 else '✗'} {leg['why']} result={rc}")
        if rc != 0:
            return False, f"{leg['why']} 失败 result={rc}"
        wait = pump_s if leg["wait"] is None else leg["wait"]
        if wait:
            time.sleep(wait)
    return True, None


# ─────────────────────────── 硬件那一段 ───────────────────────────
def run_pick(args) -> int:
    """连臂 → ③ 瞄 → 降到方块顶面 → 吸 → 抬 → 移到放置点 → 放。"""
    mp, why = aim.load_map(args.map)
    if why:
        print(f"✗ {why}")
        return 1
    A_inv, C, cam_z = mp["A_inv"], mp["C"], mp["cam_z"]

    table_z = args.table_z if args.table_z is not None else msm.load_table_z()
    if table_z is None:
        print("✗ 不知道纸面 Z —— 没有它，Z 就没有下限，本脚本一步都不动。")
        print("  先跑 python3 src/step4_pick_test.py --probe，或给 --table-z。")
        return 1

    import step2_teach_coords as tc

    limits = {a: tuple(v) for a, v in tc.DEF_LIMITS.items()}
    if args.limits:
        for spec in args.limits.split(","):
            ax, lo, hi = spec.split(":")
            limits[ax.strip()] = (float(lo), float(hi))
    floor = z_floor(table_z, args.obj_h, args.press)
    limits["z"] = (floor, limits["z"][1])
    print(f"[Z 下限] {floor:.2f} = 纸面 {table_z:.2f} + 方块高 {args.obj_h:.1f}"
          f" − 预压 {args.press:.1f}"
          f"   ★ 降到这里**方块顶面**，再往下没有第二种解释")

    # ★ 绝对放置点能在连臂之前就查掉 —— 免得白跑一趟（step4 的教训）
    pre_drop = None
    if args.drop:
        try:
            pre_drop = parse_xy(args.drop)
        except ValueError as e:
            print(f"✗ --drop {e}")
            return 2
        bad = s4.why_unreachable(limits, pre_drop)
        if bad:
            print(f"✗ 放置点 ({pre_drop[0]:.1f}, {pre_drop[1]:.1f}) 在软限位之外: {bad}")
            print(f"  软限位 x {limits['x'][0]:.0f}~{limits['x'][1]:.0f}"
                  f"   y {limits['y'][0]:.0f}~{limits['y'][1]:.0f}")
            return 2
    if args.go and not sys.stdin.isatty():
        print("✗ 要动机械臂就得在**交互式终端**里跑（抓取要按回车确认）。")
        print("  现在 stdin 不是终端（被管道/重定向/后台了），一步都不动。")
        return 1
    # ★ 放置点的语法/互斥也在连臂之前验掉 —— 别等 ③ 把吸嘴挪过去之后、
    #   方块就在眼皮底下时才说「你没说放到哪」。
    _, _, why2 = resolve_drop(args, (0.0, 0.0))
    if why2:
        print(f"✗ {why2}")
        return 2

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
    state = {"held": False}          # 真值只有一处: finally 只看它
    try:
        dType.SetCmdTimeout(api, 5000)
        ares, alist = tc.read_alarms(api, dType)
        print(f"[报警] result={ares}  {tc.describe_alarms(alist)}")
        if tc.has_alarms(alist):
            tc.report_alarm_detail(alist)
            if tc.needs_homing(alist):
                print("\n✗ 有丢步报警: 零点已经不可信。"
                      "**这次比 ③ 更硬** —— ④ 要按绝对 Z 降到方块顶面，")
                print("  零点一漂，那个「方块顶面」就是错的，吸嘴会扎进方块"
                      "（方块被压 + 继续丢步）。先回零再跑。")
                return 1
            ok, alist = tc.resolve_alarms(api, dType, alist, args.clear_alarms)
            if not ok:
                return 1

        dType.SetQueuedCmdClear(api)
        dType.SetQueuedCmdStartExec(api)
        dType.SetPTPJointParams(api, aim.SPEED_MM_S, aim.SPEED_MM_S, aim.SPEED_MM_S,
                                aim.SPEED_MM_S, aim.ACC, aim.ACC, aim.ACC, aim.ACC,
                                isQueued=0)
        tc._set_coord_params(api, dType, aim.SPEED_MM_S, aim.ACC)
        dType.SetPTPCommonParams(api, aim.RATIO, aim.RATIO, isQueued=0)

        cur = tc.read_pose_stable(api, dType, 5)
        home = dict(cur)
        print(f"\n[当前] {tc.fmt(cur)}")
        z_tip = home["z"] - table_z
        print(f"  纸面 Z={table_z:.2f} → 吸嘴离纸面 z_tip={z_tip:.1f}mm")

        # ── 先升到悬停高度（纯升降，XY 不动），再查防撞 ──
        # ★ 顺序不能反: 闸门摆在 raise 前面的话，吸嘴只要一开始就低就永远被拒，
        #   而这个 raise 恰恰走不到 —— ③ 上就是这么卡住的（① 收工时吸嘴停在
        #   z_tip=15，比 26mm 的方块顶面还低 11mm，所以「一开始就低」是常态）。
        hover_z = table_z + args.obj_h + msm.DEFAULT_HOVER_MM
        if args.hover is not None:
            hover_z = min(table_z + args.obj_h + args.hover, limits["z"][1])
        if abs(hover_z - cur["z"]) > 0.5:
            verb = "升到" if hover_z > cur["z"] else "降到"
            print(f"\n[{verb}悬停高度] Z={hover_z:.2f}"
                  f"（纸面 {table_z:.2f} + 方块高 {args.obj_h:.0f} + "
                  f"{hover_z - table_z - args.obj_h:.0f}）")
            if not args.yes:
                input("       回车开始（Ctrl-C 中止）… ")
            tgt = {"x": cur["x"], "y": cur["y"], "z": hover_z, "r": cur["r"]}
            ok, why2 = tc.check_target(tc.read_pose_stable(api, dType, 5), tgt, limits)
            if ok:
                ok, why2 = tc.move_to(api, dType, tgt, limits, args.mode)
            if not ok:
                print(f"✗ 升降失败: {why2}")
                return 1
            tc.verify_arrival(api, dType, tgt)
            home = tc.read_pose_stable(api, dType, 5)
            z_tip = home["z"] - table_z
            print(f"  现在 Z={home['z']:.2f} → 吸嘴离纸面 z_tip={z_tip:.1f}mm")

        # ── 只横移那一段的防撞线（③ 的同一道闸；摆在 raise 之后才是最后的裁决）──
        need_z = table_z + args.obj_h + msm.MIN_CLEAR_ABOVE_OBJ_MM
        if home["z"] < need_z:
            print(f"\n✗ 吸嘴现在 Z={home['z']:.2f}（离纸面 {z_tip:.1f}mm），"
                  f"比方块顶面只高 {z_tip - args.obj_h:.1f}mm ——")
            print(f"  横移过去会撞到方块。要求 Z ≥ {need_z:.2f}"
                  f"（纸面 {table_z:.2f} + 方块高 {args.obj_h:.0f} + "
                  f"余量 {msm.MIN_CLEAR_ABOVE_OBJ_MM:.0f}）。")
            if args.hover is None:
                print(f"  悬停高度本身不够（--hover 默认 "
                      f"{msm.DEFAULT_HOVER_MM:.0f}，这里要 ≥ "
                      f"{msm.MIN_CLEAR_ABOVE_OBJ_MM:.0f}）—— 检查纸面 Z 对不对。")
            else:
                print(f"  ★ 你给的 --hover {args.hover:.0f} 不够: 得 ≥ "
                      f"{msm.MIN_CLEAR_ABOVE_OBJ_MM:.0f}"
                      f"（吸嘴至少要高出方块顶面这么多）。")
            return 1

        grab_z = z_floor(table_z, args.obj_h, args.press)
        drop_z = s4.cube_top_z(table_z, args.obj_h, 0)     # ★ 放置不预压（step4:758-759）
        print(f"\n[抓取高度] 降到 Z={grab_z:.2f}"
              f"（= 纸面 {table_z:.2f} + obj_h {args.obj_h:.1f}"
              f"{'' if args.press == 0 else f' − 预压 {args.press:.1f}'}）")
        if args.press == 0:
            print("  ★ 那 1mm 唇口压缩量**已经含在 obj_h 里**了（step4 实测 26 才吸得住），"
                  "所以这里不再多压。")
        print(f"[放置高度] 放下降到 Z={drop_z:.2f}（不预压 —— 让它底面正好落回纸面）")

        # ── ③: 先瞄，横移到位之前一步都不降 ──
        layout = load_paper_layout(msm.PAPER_JSON)
        suction = load_suction_code()
        if not suction:
            print("✗ 读不到吸盘码内容（data/suction_qr.json）—— 先跑 tools/gen_suction_qr.py")
            return 1
        z_qr = z_tip + args.qr_height
        c = aim.c_coef(cam_z, z_qr, args.obj_h)
        print(f"\n[闭环系数] Z_cam={cam_z:.1f}  z_tip={z_tip:.1f} → z_qr={z_qr:.1f}")
        print(f"  c = k({z_qr:.0f})/k({args.obj_h:.0f}) = {c:.4f}")
        # ★ ① 是在 z_tip=15 那个高度量 A 的，④ 却在「方块顶面+悬停」上瞄 —— k 不一样。
        #   不折也落得对（见文件头），只是每步超调，可能要 5 步；折了就是 1 步。
        A_inv, g = aim.rescale_A_inv(A_inv, mp["map_z_qr"], z_qr, cam_z)
        if g is None:
            print("  ⚠ 地图里没记 ① 量 A 时的码高度 → A 不折算。"
                  "落点不受影响，但可能要**多走三四步**（见 rescale_A_inv）。")
        elif abs(g - 1.0) > 0.02:
            print(f"  A 折算到本高度: k 比 {g:.4f}（① 在 "
                  f"z_qr={mp['map_z_qr']:.0f}，现在 {z_qr:.0f}）"
                  f"→ A⁻¹ ÷{g:.3f}，不然每步超调 {abs(1 - g) * 100:.0f}%")

        cap = open_camera(args.cam, focus=args.focus)
        if not cap.isOpened():
            print("✗ 打不开摄像头")
            return 1
        detector = cv2.QRCodeDetector()
        sample, move = aim.make_arm_io(api, dType, tc, cap, detector, layout, suction,
                                       limits, home, args.mode, args.color, cam_z,
                                       args.obj_h, args.frames)

        print(f"\n[第 1 段: 瞄] 只横移，最多 {args.max_iter} 步。"
              f"**收敛之前一步都不降。**")
        if not args.yes:
            if input("  确认开始？(y/N) ").strip().lower() != "y":
                print("已取消，未发任何运动指令。")
                return 1
        hist, aim_ok, why2 = aim.solve_loop(sample, move, A_inv, C, c,
                                            tol_mm=args.tol, max_step=args.max_step,
                                            max_iter=args.max_iter,
                                            lost_ok_mm=args.lost_ok)
        for r in hist:
            print(f"  · 第 {r['i']} 次量: 码({r['qr_px'][0]:7.1f},{r['qr_px'][1]:7.1f})"
                  f"  方块({r['cube_px'][0]:7.1f},{r['cube_px'][1]:7.1f})"
                  f"  → 还差 {r['raw_norm']:6.2f}mm"
                  + ("  （钳到 %.0fmm）" % r["norm"] if r["clamped"] else ""))

        # ★★ 这道闸是 ④ 的根本: 没瞄上就绝不下降
        #    ★ aim_ok=True 有两种: why2=None 是真收敛；why2 有字面是「近距丢失」
        #      收的（吸嘴把目标挡住了）—— 后者**允许下降**，但落点凭的是上次那个
        #      读数而不是 tol，所以下面必须把它说出来，不能混过去。
        if not aim_ok:
            print(f"\n✗ 没瞄上（{why2}）—— **不下降、不吸**，直接退出。")
            print("  没收敛就下降 = 吸嘴扎在方块边上或两个方块之间，比抓空危险得多。")
            print("  先不加 --go 跑一遍看 δ，或者重跑 ①/②（相机或纸动过就重量）。")
            return 1
        if hist and why2:
            r1l = hist[-1]
            print(f"  ⚠ 算瞄上（**不是**收敛）: {why2}")
            print(f"    吸嘴压到 {args.color} 上方必然挡掉目标色，最后那几毫米量不到。"
                  f"落点精度 = **预计还差 {r1l['raw_norm'] - r1l['norm']:.2f}mm**"
                  f"（上次真量到 {r1l['raw_norm']:.2f}mm、朝它走了 {r1l['norm']:.2f}mm）。"
                  f"\n    ★ 这条**放行**，下面照常下降 —— 几何使然，没有更准的量法。")
        elif hist:
            print(f"  ✅ 瞄上: 最后还差 {hist[-1]['raw_norm']:.2f}mm"
                  f"（{aim.steps_taken(hist)} 步）")

        # ── 抓取点 = 机械臂**自己报的当前 XY**（不需要任何映射）──
        grab = tc.read_pose_stable(api, dType, 5)
        grab_xy = (grab["x"], grab["y"])
        drop_xy, drop_desc, why2 = resolve_drop(args, grab_xy)
        if why2:
            print(f"\n✗ {why2}")
            return 2
        # ★ 放置点在下**降之前**就要查: 方块举在手上才发现放不下，是最难收场的局面
        bad = s4.why_unreachable(limits, drop_xy)
        if bad:
            print(f"\n✗ 放置点 ({drop_xy[0]:.1f}, {drop_xy[1]:.1f}) 够不着: {bad}")
            print("  还没下降，方块还在原地 —— 换个 --drop 重跑。")
            return 2

        legs = plan_pick(grab_xy, grab_z, hover_z, drop_xy, drop_z, hold=args.hold)

        print(f"\n[第 2 段: 抓取] 抓取点 ({grab_xy[0]:.2f}, {grab_xy[1]:.2f})"
              f" ← 机械臂自己报的，没经过任何映射")
        print(f"  放置点 {drop_desc}")
        print(f"  悬停 Z={hover_z:.2f}   抓取 Z={grab_z:.2f}   放置 Z={drop_z:.2f}")
        print("\n  动作表:")
        for i, leg in enumerate(legs, 1):
            if leg["kind"] == "move":
                print(f"    {i}. 走到 ({leg['x']:7.2f},{leg['y']:7.2f},"
                      f" Z={leg['z']:7.2f})  {leg['why']}")
            else:
                print(f"    {i}. 真空 {'开' if leg['on'] else '关'}"
                      f"                {leg['why']}")
        if args.hold:
            print("\n  ★ --hold: 抬到悬停高度就停。退出时**一定关真空**"
                  "（不会把泵开着丢下就走），方块会从悬停高度掉回原位。")
        print(f"\n⚠ 这是**第一次真的下降 + 真空**。确认: 方块在抓取点、"
              f"吸盘干净、泵接好、周围无障碍。")
        if not args.yes:
            if input("  回车开始抓取（Ctrl-C 中止）… ").strip().lower() not in ("", "y"):
                print("已取消。")
                return 1

        r_now = grab["r"]

        def move_xyz(x, y, z, why3) -> bool:
            tgt = {"x": float(x), "y": float(y), "z": float(z), "r": r_now}
            ok, bad2 = tc.check_target(tc.read_pose_stable(api, dType, 5), tgt, limits)
            if not ok:
                print(f"     ✗ {why3}: {bad2}")
                return False
            ok, bad2 = tc.move_to(api, dType, tgt, limits, args.mode)
            if not ok:
                print(f"     ✗ {why3}: {bad2}")
                return False
            ok, bad2 = tc.verify_arrival(api, dType, tgt)
            print(f"     {'✓' if ok else '⚠'} {why3}" + (f"  {bad2}" if bad2 else ""))
            return ok

        def pump(on) -> int:
            rc = s4.suction(api, on)
            if rc == 0:
                state["held"] = bool(on)
            return rc

        ok, why2 = do_pick(legs, move_xyz, pump, pump_s=args.pump_s)
        if not ok:
            print(f"\n✗ {why2}")
            print("  已经停在这一步没再往下走。"
                  + ("方块还在吸盘上 —— 收尾时会关真空。" if state["held"] else ""))
            return 1

        print("\n✅ 抓取流程走完。")
        if args.hold:
            print("   方块现在吸在吸盘上、悬停着 —— 收尾关真空后它会掉回原位。")
        else:
            print(f"   看落点: 目标 {drop_desc}")
            print("   偏了多少、往哪偏 → 就是 ③ 在这一处的瞄准误差 + 吸附时的横向拉动。")
        ensure_output_dir()
        resid = aim.resid_mm(hist)
        PICK_SUCTION_JSON.write_text(json.dumps({
            "tool": "pick_suction.py",
            "note": "相对运动路线（吸盘码闭环）的抓取记录",
            "color": args.color,
            "table_z": round(float(table_z), 3),
            "obj_h": args.obj_h, "press": args.press,
            "grab_z": round(float(grab_z), 3), "drop_z": round(float(drop_z), 3),
            "grab_xy": [round(float(grab_xy[0]), 2), round(float(grab_xy[1]), 2)],
            "drop_xy": [round(float(drop_xy[0]), 2), round(float(drop_xy[1]), 2)],
            "drop_spec": args.drop or args.drop_rel or "--hold",
            "hold": bool(args.hold),
            # ★ 这两个数都别自己算 —— 语义在 ③ 那边定好了（steps_taken/resid_mm）。
            #   len(hist)-1 在「近距丢失」收工时少数一步；hist[-1]["raw_norm"] 是
            #   **走之前**的账，不是残差（那一步真走掉了，人要的是走完还差多少）。
            "aim_steps": aim.steps_taken(hist),
            "aim_resid_mm": round(resid, 3) if resid is not None else None,
            "z_qr": round(float(z_qr), 2), "c_coef": round(float(c), 4),
            "cam_z": round(float(cam_z), 2),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   （参数已记到 {PICK_SUCTION_JSON.name}）")
        return 0
    finally:
        # ★ 收尾三件事，顺序钉死: 先松吸盘（别举着东西）、再停队列、最后断开。
        if state["held"]:
            try:
                s4.suction(api, False, isQueued=0)
                print("[收尾] 已关真空（松开吸盘）")
            except Exception:
                pass
        try:
            dType.SetQueuedCmdForceStopExec(api)
            dType.SetQueuedCmdClear(api)
        except Exception:
            pass
        if cap is not None:
            cap.release()
        try:
            dType.DisconnectDobot(api)
        except Exception:
            pass


# ─────────────────────────── 自检 ───────────────────────────
def selftest() -> int:
    """离线自检: 把「顺序」和「算术」拿假臂、假泵逐条钉一遍。不连相机不连臂。"""
    bad = 0

    def check(name, cond, extra=""):
        nonlocal bad
        print(f"  {'✅' if cond else '❌'} {name}" + (f"   {extra}" if extra else ""))
        if not cond:
            bad += 1

    print("pick_suction.py 自检")
    table_z, obj_h, press = -18.416, 26.0, 0.0
    hover_z = table_z + obj_h + msm.DEFAULT_HOVER_MM
    grab_z = z_floor(table_z, obj_h, press)
    drop_z = s4.cube_top_z(table_z, obj_h, 0)
    grab_xy, drop_xy = (210.0, 40.0), (240.0, 140.0)

    # 1) Z 的几个数: 抓取停在方块顶面（不是纸面）、放置与抓取同高
    check("④ 抓取高度 = 纸面 + obj_h（**不是纸面**）",
          abs(grab_z - (table_z + obj_h)) < 1e-9,
          f"{grab_z:.2f} = {table_z:.2f}+{obj_h:.0f}，纸面是 {table_z:.2f}")
    check("④ 放置与抓取同高（同一层）",
          abs(s4.cube_top_z(table_z, obj_h, 1) - (table_z + 2 * obj_h)) < 1e-9)
    check("④ 悬停比方块顶面高", hover_z > grab_z, f"{hover_z:.1f} > {grab_z:.1f}")
    check("④ press 是**往下**压（下限更低）",
          z_floor(table_z, obj_h, 1.5) < z_floor(table_z, obj_h, 0.0),
          f"press 1.5 → {z_floor(table_z, obj_h, 1.5):.2f}")

    # 2) 动作表的顺序 —— 三条铁律
    legs = plan_pick(grab_xy, grab_z, hover_z, drop_xy, drop_z)
    kinds = [l["kind"] for l in legs]
    zs = [l["z"] for l in legs if l["kind"] == "move"]
    check("④ 第一腿是悬停（绝不从半空直接降）",
          legs[0]["kind"] == "move" and abs(legs[0]["z"] - hover_z) < 1e-9,
          f"Z={legs[0]['z']:.2f}")
    i_on = next(i for i, l in enumerate(legs)
                if l["kind"] == "pump" and l["on"])
    i_grab = next(i for i, l in enumerate(legs)
                  if l["kind"] == "move" and abs(l["z"] - grab_z) < 1e-9)
    check("④ 开真空在「降到抓取高度」**之后**", i_on > i_grab,
          f"第 {i_grab + 1} 腿降下 → 第 {i_on + 1} 腿开泵")
    i_off = next(i for i, l in enumerate(legs)
                 if l["kind"] == "pump" and not l["on"])
    i_drop = next(i for i, l in enumerate(legs)
                  if l["kind"] == "move" and abs(l["z"] - drop_z) < 1e-9)
    check("④ 放气在「降到放置高度」**之后**", i_off > i_drop,
          f"第 {i_drop + 1} 腿降下 → 第 {i_off + 1} 腿放气")
    check("④ 放气之后还有一腿抬起离开",
          legs[-1]["kind"] == "move" and abs(legs[-1]["z"] - hover_z) < 1e-9)
    check("④ 每一腿 move 的 Z 都 ≥ 抓取高度（没有一腿往方块里扎）",
          all(z >= grab_z - 1e-9 for z in zs),
          f"最低 {min(zs):.2f}，抓取高度 {grab_z:.2f}")
    check("④ 抬起来之前没有任何横移（先抬后走）",
          legs[3]["why"].startswith("抬起")
          and abs(legs[3]["x"] - grab_xy[0]) < 1e-9
          and abs(legs[3]["y"] - grab_xy[1]) < 1e-9)
    check("④ 只横移到放置点那一腿才换 XY",
          all(abs(l["x"] - drop_xy[0]) < 1e-9 and abs(l["y"] - drop_xy[1]) < 1e-9
              for l in legs[4:] if l["kind"] == "move"))

    # 3) --hold: 不放置、也不放气（抬起来停在吸盘上）
    h = plan_pick(grab_xy, grab_z, hover_z, drop_xy, drop_z, hold=True)
    check("④ --hold 不收尾放气", not any(l["kind"] == "pump" and not l["on"] for l in h))
    check("④ --hold 不横移到放置点",
          all(abs(l["x"] - grab_xy[0]) < 1e-9 for l in h if l["kind"] == "move"))
    check("④ --hold 也走「先降 → 开泵 → 抬」这三步",
          [l["kind"] for l in h] == ["move", "move", "pump", "move"])

    # 4) do_pick 按表执行，且**一腿失败就停在那一腿**
    def runner(fail_at=None):
        seen = []

        def mv(x, y, z, why):
            seen.append(("move", round(z, 3)))
            return len(seen) != fail_at

        def pm(on):
            seen.append(("pump", on))
            return 0

        ok, why = do_pick(legs, mv, pm, pump_s=0.0, log=lambda *a, **k: None)
        return seen, ok, why

    seen, ok, why = runner()
    check("④ 顺序执行: 降→开泵→抬→横移→降→放气→抬",
          ok and [s[1] if s[0] == "pump" else "m" for s in seen]
          == ["m", "m", True, "m", "m", "m", False, "m"],
          f"{len(seen)} 腿")
    seen, ok, why = runner(fail_at=2)          # 第 2 腿（下降）失败
    check("④ 下降失败 → 停在那里、**泵都没开**",
          not ok and ("pump", True) not in seen and len(seen) == 2, why or "")
    seen, ok, why = runner(fail_at=5)          # 第 5 腿（移到放置点）失败
    check("④ 带着方块横移失败 → 停住、泵**保持开**（由收尾关）",
          not ok and ("pump", True) in seen and ("pump", False) not in seen, why or "")
    check("④ 失败之后**没有再往下走**（后面的腿一腿都没执行）",
          not ok and len(seen) == 5, f"执行了 {len(seen)} 腿，共 {len(legs)} 腿")

    # 5) --drop / --drop-rel / --hold 的解析
    class A:
        drop = drop_rel = None
        hold = False

    a = A()
    a.drop = "240,140"
    xy, desc, why2 = resolve_drop(a, grab_xy)
    check("④ --drop 绝对点解析", xy == (240.0, 140.0) and why2 is None, desc)
    a = A()
    a.drop = "240，140"                      # 中文逗号也得认（人常打错）
    xy, desc, why2 = resolve_drop(a, grab_xy)
    check("④ --drop 认中文逗号", xy == (240.0, 140.0) and why2 is None)
    a = A()
    a.drop_rel = "0,-60"
    xy, desc, why2 = resolve_drop(a, grab_xy)
    check("④ --drop-rel 相对抓取点", xy == (210.0, -20.0) and why2 is None, desc)
    a = A()
    a.hold = True
    xy, desc, why2 = resolve_drop(a, grab_xy)
    check("④ --hold 就是抓取点", xy == grab_xy and why2 is None)
    a = A()
    xy, desc, why2 = resolve_drop(a, grab_xy)
    check("④ 什么都不给 → 说人话、不动臂", xy is None and "要说明" in (why2 or ""))
    a = A()
    a.drop, a.hold = "1,2", True
    xy, desc, why2 = resolve_drop(a, grab_xy)
    check("④ 互斥的两条一起给 → 拦下", xy is None and "互斥" in (why2 or ""))
    a = A()
    a.drop = "240"
    xy, desc, why2 = resolve_drop(a, grab_xy)
    check("④ --drop 少一个数 → 拦下", xy is None and "--drop" in (why2 or ""))

    # 6) 没瞄上就绝不下降 —— 用假臂钉住这条
    #    ③ 的 solve_loop 不收敛时 (False, 原因)，run_pick 在它之后就 return 了；
    #    这里用「计划表的第一腿必须是悬停」+「不收敛的返回」两个可测事实来钉。
    _, _, aim_why = aim.solve_loop(lambda: None, lambda d, n: True,
                                   np.eye(2), np.array([1004.6, 589.1]), 1.5798,
                                   log=lambda *a, **k: None)
    check("④ 没量到 → ③ 不收敛（④ 就会在下降之前退出）",
          aim_why is not None and "没认出" in aim_why, aim_why or "")

    # 7) 地图缺了就报人话（和 ③ 同一条道）
    mp, why2 = aim.load_map("/tmp/definitely_not_here_suction_map.json")
    check("④ 地图读不到时报人话", mp == {} and bool(why2), (why2 or "")[:36])

    # 8) ★ 落盘那两个「瞄准账」必须走 ③ 的 steps_taken/resid_mm，不许就地算。
    #    ★ 为什么非查源码不可: 那两行在 run_pick 里 —— 只有真机跑得到，离线自检
    #      够不着（和 look_once 的 check 12/13 同一个处境）。而这两个数正是排查时
    #      最要信的两个，写错了没人看得出来:
    #        · len(hist)-1   → 「近距丢失」收工时少数一步（实测把 1 步报成 0 步）
    #        · raw_norm      → 是**走之前**的账，不是残差（实测报 14.66，其实已到 0）
    #      ③ 的自检只证明「这函数算得对」，证明不了「④ 用了它」—— 所以在这儿钉住。
    import ast
    d = None
    for nd in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8"))):
        if (isinstance(nd, ast.Call) and isinstance(nd.func, ast.Attribute)
                and nd.func.attr == "dumps" and nd.args
                and isinstance(nd.args[0], ast.Dict)):
            d = nd.args[0]
    vals = ({k.value: v for k, v in zip(d.keys, d.values)
             if isinstance(k, ast.Constant)} if d is not None else {})
    steps_src = ast.unparse(vals["aim_steps"]) if "aim_steps" in vals else ""
    resid_src = ast.unparse(vals["aim_resid_mm"]) if "aim_resid_mm" in vals else ""
    check("④ ★ 落盘的步数走 aim.steps_taken（不是 len(hist)-1）",
          steps_src == "aim.steps_taken(hist)", f"aim_steps = {steps_src or '（没找到）'}")
    check("④ ★ 落盘的残差走 aim.resid_mm（不是 hist[-1]['raw_norm']）",
          "resid" in resid_src and "raw_norm" not in resid_src,
          f"aim_resid_mm = {resid_src or '（没找到）'}")

    print(f"\n{'✅ 全部通过' if not bad else f'❌ {bad} 项没过'}")
    return 1 if bad else 0


# ─────────────────────────── CLI ───────────────────────────
def build_argparser():
    ap = argparse.ArgumentParser(
        description="④ 抓方块: ③ 瞄 → 降到方块顶面 → 吸 → 抬 → 移到放置点 → 放")
    ap.add_argument("--color", default=None,
                    help=f"抓哪个颜色的方块（{'/'.join(cvis.CUBE_COLORS)}）")
    ap.add_argument("--go", action="store_true",
                    help="★ 真动机械臂（下降 + 真空）。不给只做离线检查")
    ap.add_argument("--hold", action="store_true",
                    help="★ 第一次先用这个: 抓起来抬到悬停高度就停，不放置")
    ap.add_argument("--drop", default=None,
                    help="放置点的**绝对**机械臂 XY，如 240,140")
    ap.add_argument("--drop-rel", default=None,
                    help="放置点相对**抓取点**的偏移，如 0,-60")
    ap.add_argument("--map", default=None, help=f"①/② 的地图，默认 {aim.SUCTION_MAP_JSON}")
    ap.add_argument("--obj-h", type=float, default=msm.DEFAULT_OBJ_H_MM,
                    help=f"方块「高」（默认 {msm.DEFAULT_OBJ_H_MM:.0f}）—— "
                         "**含吸盘唇口压缩量**，25mm 的方块要填 26")
    ap.add_argument("--press", type=float, default=DEFAULT_PRESS_MM,
                    help=f"**额外**预压 mm（默认 {DEFAULT_PRESS_MM:.0f}）。"
                         "压缩量已在 --obj-h 里，别重复加")
    ap.add_argument("--qr-height", type=float, default=msm.DEFAULT_QR_HEIGHT_MM,
                    help=f"吸盘码中心到吸嘴尖 mm（默认 {msm.DEFAULT_QR_HEIGHT_MM:.0f}）")
    ap.add_argument("--hover", type=float, default=None, nargs="?", const=30.0,
                    help=f"悬停高度 = 方块顶面 + 这么多 mm（默认 "
                         f"{msm.DEFAULT_HOVER_MM:.0f}）")
    ap.add_argument("--pump-s", type=float, default=DEFAULT_PUMP_S,
                    help=f"开真空后等多久再抬（默认 {DEFAULT_PUMP_S}）= 吸住的时间")
    ap.add_argument("--tol", type=float, default=aim.DEFAULT_TOL_MM,
                    help=f"③ 说「再走不到这么多 mm」就算瞄上（默认 {aim.DEFAULT_TOL_MM}）")
    ap.add_argument("--max-step", type=float, default=aim.MAX_STEP_MM,
                    help=f"③ 单步钳幅 mm（默认 {aim.MAX_STEP_MM:.0f}）")
    ap.add_argument("--lost-ok", type=float, default=aim.LOST_OK_MM, metavar="MM",
                    help=f"「近距丢失 = 够准了」的闸值（默认 {aim.LOST_OK_MM:.0f}mm）: "
                         f"曾经量到过 ≤MM 之后目标被吸嘴挡住，就算瞄上、继续下降。"
                         f"给负数 = 关掉（看不见就中止，最保守）")
    ap.add_argument("--max-iter", type=int, default=aim.MAX_ITER,
                    help=f"③ 迭代上限（默认 {aim.MAX_ITER}）")
    ap.add_argument("--frames", type=int, default=msm.ACCUM_FRAMES,
                    help=f"每次量积累多少帧（默认 {msm.ACCUM_FRAMES}）")
    ap.add_argument("--table-z", type=float, default=None,
                    help="纸面 Z，默认读 output/pick_test_result.json")
    ap.add_argument("--cam", type=int, default=None, help="摄像头编号，默认自动挑")
    ap.add_argument("--focus", type=int, default=FOCUS_LOCK,
                    help=f"**锁死手动焦距**（UVC 值，小=对远、大=对近）。"
                         f"默认 {FOCUS_LOCK}，和 ③ 同一个值 —— 瞄准那一段和 ③ 共用，"
                         f"锁定理由见 qr_vision.lock_focus。给负数 = 不锁（等自动对焦）。")
    ap.add_argument("--port", default=None, help="机械臂串口，默认自动查找")
    ap.add_argument("--mode", default="movl", choices=("movl", "movj"),
                    help="movl 直线（默认）；抓取/放置走直线别用 movj")
    ap.add_argument("--limits", default=None,
                    help='覆盖软限位，如 "x:50:380,y:-310:310"（★ Z 下限由本脚本自己锁）')
    ap.add_argument("--clear-alarms", action="store_true",
                    help="报警清不掉时强制清（丢步 0x50~0x5f 清不掉，那是硬闸门）")
    ap.add_argument("--yes", action="store_true", help="跳过所有确认回车")
    ap.add_argument("--selftest", action="store_true", help="离线自检，不连相机不连臂")
    return ap


def preflight(args) -> int:
    """
    离线预检: 不连臂、不开相机，只把「动臂要用的那几个数」摆出来对一遍。

    ★ --go 是**唯一**会动的开关（和 ③ 一致）。--hold/--drop 都只是形态，
      不是执行开关 —— 少一个 --go 就一步都不动，这样就不会有"我以为它只是看看"
      的那种事故。
    """
    print("=" * 68)
    print("  ④ 离线预检（机械臂一步没动）")
    print("=" * 68)
    mp, why = aim.load_map(args.map)
    if why:
        print(f"✗ {why}")
        return 1
    print(f"  地图: {args.map or aim.SUCTION_MAP_JSON}")
    print(f"  光心 C = ({mp['C'][0]:.1f}, {mp['C'][1]:.1f})px"
          f"   Z_cam = {mp['cam_z']:.1f}mm")
    print(f"  焦距锁 FOCUS={args.focus}（{'手动锁死' if args.focus >= 0 else '不锁，等自动对焦'}）"
          f" —— 不锁死的话吸盘码会「时解得出时解不出」，瞄准那一下就卡住")
    table_z = args.table_z if args.table_z is not None else msm.load_table_z()
    if table_z is None:
        print("✗ 读不到纸面 Z（--table-z 或 output/pick_test_result.json）"
              "—— 没有它 Z 就没有下限，真跑时一步都不动。")
        print("  先跑 python3 src/step4_pick_test.py --probe")
        return 1
    floor = z_floor(table_z, args.obj_h, args.press)
    print(f"\n  [Z 账] 纸面 {table_z:.2f}")
    print(f"         方块「高」--obj-h {args.obj_h:.1f}"
          f"（★ 含唇口压缩量: 25mm 的方块要填 26）")
    print(f"         预压 --press {args.press:.1f}"
          f"（★ 已含在 obj-h 里，默认不加）")
    print(f"    → 抓取/放置高度 Z={floor:.2f}"
          f"（= 纸面 + obj_h）★ 是**方块顶面**，不是纸面 {table_z:.2f}")
    print(f"    → 悬停高度     Z={table_z + args.obj_h + (args.hover if args.hover is not None else msm.DEFAULT_HOVER_MM):.2f}")
    print(f"    → Z 下限       Z={floor:.2f}（本脚本自己锁，放开的一点只有 --press）")

    # 放置点那条路先验语法/互斥，再验够不够得着
    xy, desc, why2 = resolve_drop(args, (0.0, 0.0))
    if why2:
        print(f"\n✗ {why2}")
        return 2
    print(f"\n  [放置] {desc}")
    if args.drop:
        lim = {a: tuple(v) for a, v in __import__("step2_teach_coords").DEF_LIMITS.items()}
        bad = s4.why_unreachable(lim, xy)
        if bad:
            print(f"  ✗ 在软限位之外: {bad}")
            return 2
        print("  ✓ 在软限位之内")
    elif args.drop_rel:
        print("  （相对抓取点 —— 绝对位置要等 ③ 收敛后才知道，那时会再查一次）")

    print("\n  [动作表] 抓取点用符号 (X,Y) 代替，真跑时它 = 机械臂自己报的当前 XY")
    for i, leg in enumerate(plan_pick((0.0, 0.0), floor,
                                      table_z + args.obj_h + msm.DEFAULT_HOVER_MM,
                                      xy, s4.cube_top_z(table_z, args.obj_h, 0),
                                      hold=args.hold), 1):
        if leg["kind"] == "move":
            print(f"    {i}. 走到 ({leg['x']:7.2f},{leg['y']:7.2f},"
                  f" Z={leg['z']:7.2f})  {leg['why']}")
        else:
            print(f"    {i}. 真空 {'开' if leg['on'] else '关'}                {leg['why']}")

    hold = "--hold" if args.hold else f'--drop {args.drop or args.drop_rel}'
    print(f"\n  真跑（会下降 + 开真空）:")
    print(f"    python3 tools/pick_suction.py --color {args.color or 'red'} "
          f"{hold} --go")
    print("  ★ 第一次先加 --hold: 只验证「瞄得准不准 + 吸不吸得住」，不放置。")
    return 0


def main() -> int:
    args = build_argparser().parse_args()
    if args.selftest:
        return selftest()
    if not args.color:
        print(f"✗ 要指定抓哪个颜色: --color {'|'.join(cvis.CUBE_COLORS)}")
        return 2
    if args.color not in cvis.CUBE_COLORS:
        print(f"✗ 不认识的颜色 {args.color!r}，只有 {'/'.join(cvis.CUBE_COLORS)}")
        return 2
    # ★ --go 是唯一会动的开关。缺它就只做离线预检，绝不连臂。
    if not args.go:
        return preflight(args)
    if not (args.hold or args.drop or args.drop_rel):
        print("✗ 要说明抓到之后怎么办: --hold / --drop x,y / --drop-rel dx,dy")
        return 2
    return run_pick(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[已中止] 退出前会关真空、停队列。")
        sys.exit(130)
