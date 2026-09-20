#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_teach_coords.py —— 逐个「对刀」量出 P1~P4 在机械臂坐标系里的 XY
=================================================================
对应《方案.md》第二阶段。做完这一步，step3_hand_eye_calib.py 才有东西可拟合。

为什么要写这个脚本（而不是让你看着 DobotStudio 手抄）:
  · 手抄 4 组数字 → 容易抄错/看错行，而且错了要等标定跑完才暴露；
  · 这里每记一个点就立即和「纸上已知布局」对距离，错了当场报；
  · 结果直接落成 output/robot_points.json，step3_hand_eye_calib.py 自动读，不用复制粘贴。

──────────────────────── 怎么用（在**项目根目录**下执行）────────────────────────
  1) 机械臂上电、USB 插好，标定纸固定在硬纸板上、机械臂底座卡进缺口
  2) 先看一眼 output/ref_point_guide.png —— 图上把 5 个候选参考点全标了（TL/TR/BR/BL/C），
     按 --reference 选的那个瞄（默认 corner_tr = 数据区右上角）。
     ★ 千万别瞄白边外框角（图上的红叉）: 那是一片纯白、看不出位置，差 5.42mm。
  3) python3 src/step2_teach_coords.py
     → 先「触底」：把 Z 慢慢降下来，吸盘刚碰到纸面时记一下
       （这样后面 Z 就被锁在「纸面 − BELOW_PAPER_MM」上，压不下去太多）
     → 然后按 P1→P2→P3→P4 逐个微调对准、按空格记录
  4) 跑 python3 src/step3_hand_eye_calib.py 做正式标定

──────────────────────── 按键 ────────────────────────
  移动（每次一小步，绝对坐标，不会累积漂移）:
     w / s     Y + / Y -
     a / d     X - / X +
     r / f     Z + / Z -
     t / g     R + / R -
   其他:
     1 / 2 / 3  步长 = 10mm / 2mm / 0.2mm
     m          切换直线(MOVL)/关节(MOVJ)运动方式
     j          回到本码已记录的位置（想重来时用）
     space      记录当前码的 XY
     n / p      下一个码 / 上一个码
     l          列出已记录的点 + 当前校验结果
     h          帮助
     q          结束并保存（没记全也保存，会告诉你缺哪些）

──────────────────────── 安全设计（重要，别删） ────────────────────────
  ★ 全程只发 PTP 运动指令。**绝不调用** SetHOMEParams / SetHOMECmd /
    SetEndEffectorParams —— 这台机器上次卡死事故就是 demo_all 里那两个
    脏参数（末端偏移 71.6、回零原点 200）造成的（见 rescue2.py 的注释）。
    本脚本只在连接时**读**这些参数并打印出来，一个字都不写。
  ★ 软限位：每个目标点先检查是否在允许盒子里；已经越界时只允许「往盒子里
    走」，不允许继续往外。Z 在触底之后被锁在纸面附近
    （z >= 纸面 − BELOW_PAPER_MM，见文件顶部那个常量）。
  ★ 绝对坐标而非增量：目标由「实测当前位姿 + 步长」算出。某条指令丢了也
    不会像增量那样悄悄累积误差 —— 下一步依然回到正确的绝对位置。
  ★ 单步慢速 + 每步回读校验：到位偏差过大就报警，连续失败直接停。
  ★ Ctrl-C 随时中止 → 立即 ForceStop + 清队列，再断开。终端也一定复原。
  ★ 报警处理:
      · 上电必然置位 0x00「系统复位」→ 自动 ClearAllAlarmsState() 清掉再开工
        （固件设计如此，不清的话每次开机都会被自己挡住）
      · 丢步 0x50~0x5f → **拒绝清除、拒绝运动**，要求先重新回零。
        丢步后编码器零点已不可信，硬量出来的坐标是「稳定地错」，比报错更危险。
      · 限位 0x40~0x49 等其它报警 → 默认不动，需显式 --clear-alarms

──────────────────────── 关于「先回零」────────────────────────
  ★ 本脚本不替你回零（回零会大幅运动，风险高，不放进自动流程）。
    开机第一件事请自己确认机械臂已回零 —— home_arm.py 或 DobotStudio。
    原因: 丢步/受撞/手掰过之后，控制器并不总是报警，但关节零位已经偏了，
    而它照样能读出一个「看起来很正常」的 XY。整套标定会因此整体平移或旋转，
    并且在残差里**看不出来**（4 点拟合残差恒为 0）。

依赖: 同目录的 qr_vision.py；data/calib_A4_qr.json；SDK 在 sdk/dobot/
      （随项目自带，不用改路径；加载细节见 src/dobot_sdk.py）
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import json
import sys
import tempfile
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np

from qr_vision import (DEFAULT_REFERENCE, REFERENCE_LABEL, REFERENCE_MODES,
                       SCALE_TOL, affine_fit, affine_winding_bad, load_paper_layout,
                       robot_scale_note, shape_diagnosis,
                       similarity_fit)
from paths import PAPER_JSON, ROBOT_POINTS_JSON as OUT_JSON, ensure_output_dir
from dobot_sdk import find_port, load_sdk

# ─────────────────────────── 参数 ───────────────────────────
STEPS = {"1": ("粗", 10.0), "2": ("中", 2.0), "3": ("细", 0.2)}
DEFAULT_STEP_KEY = "2"

# 软限位（Magician 工作范围留余量后的默认值；可用命令行覆盖）
# ★ 2026-09-18 按用户要求把每条区间**上下各放宽 50mm**。原值:
#     x(100, 330)  y(-260, 260)  z(-60, 160)  r(-180, 180)
#   为什么要动 x 的下限: 纸上 X 最小的可指认点是 P3.bl = 78.3mm（其次 P3.br 79.0、
#   P4.bl 83.7、P4.br 84.3、P3.c 89.2、P4.c 93.9）。原来的 100 把这些点全挡在外面，
#   而机械臂离底座明明还有余量 —— 所以下限放到 50，整张纸都在盒子里。
#   ⚠ 放宽后有几条已经**超出机器物理能力**，等于该轴没有限制:
#     x>320（Magician 臂展约 320mm）、y±310、r 超过 ±180 的一整圈。
#     真正还在起作用的只剩 x 下限和 z 区间 —— 别再以为"有软限位就撞不着"。
DEF_LIMITS = {
    "x": (50.0, 380.0),
    "y": (-310.0, 310.0),
    "z": (-110.0, 210.0),    # z 的下限 = 工作时不许低于这里（触底后被换成本文件的 BELOW_PAPER_MM）
    "r": (-230.0, 230.0),
}
# ★ 触底量出纸面之后，Z 下限 = 纸面 − BELOW_PAPER_MM，也就是**允许压到纸面以下多少**。
#   写成"往下多少"的正数，别写成负的 slack —— 这个数原来是 `TOUCH_SLACK = 0.0`
#   并且用 `z_floor + TOUCH_SLACK` 相加，想往下放得填负数，符号是反的、很容易看错。
#   2026-09-18 由 0 放宽到 20（用户要求 +2cm）:
#     0 的时候纸面就是硬底。粗档一步 10mm，吸盘离纸 6mm 时按一下就被拒 ——
#     连"把吸盘轻轻压到纸上"都做不到，报错「✗ 会低于纸面 Z=… → 拒绝」。
#   ⚠ 这是**仅剩的几道护桌子的闸之一**：调大就是拿桌面和吸盘冒险。
#     纸面本身就是靠触底时"目视碰到"估的，实际有几个 mm 的误差（唇口会被压扁），
#     所以留一点余量是合理的；但不要因为"还能再加"就继续往上加。
BELOW_PAPER_MM = 20.0

# ★ 触底时的 Z 探测下限，和工作下限 DEF_LIMITS["z"][0] 是**两回事**。
#   为什么必须分开: 触底这一步的**全部目的**就是找出纸面在哪个 Z。
#   要是探测下限比纸面还高，机械臂根本够不到纸，这一步永远做不完 ——
#   实测就撞上了: 默认 -60 时吸盘离纸还有一截就被自己的软限位拦住。
#   探测期间放宽到下面这个值，由你用眼睛盯着吸盘兜底（触底本来就是目视操作）；
#   一旦按下 space 记下纸面，Z 立刻被锁回「纸面 − BELOW_PAPER_MM」，保护照旧生效。
#   参考: 本项目《认识这台机械臂》记录的已验证高度带是 -33.8 ~ +70.0 mm
#   （贴桌约 -35），但那是"至少能到"、不是极限，所以这里给足余量。
# ★ 2026-09-18: 工作下限一起放宽 50mm 后（-60 → -110），这个探测下限**必须跟着走**
#   （-95 → -145），否则它就比工作下限还高，触底时反而比平时更受限 —— 那就本末倒置了。
#   自检里 `PROBE_Z_FLOOR < DEF_LIMITS["z"][0]` 这条就是在钉死这个关系。
PROBE_Z_FLOOR = -145.0

# 到位判定：实测位姿与目标的最大偏差超过它就算没到位
ARRIVE_TOL_MM = 3.0
ARRIVE_TOL_DEG = 5.0

SETTLE_S = 0.35      # 运动完成后等机械臂停稳再读位姿
POSE_SAMPLES = 7     # 读位姿取中位数，抗单帧抖动

# 校验容差：机械臂坐标算出的两两距离 vs 纸上距离，允许差多少 mm
COORD_TOL_MM = 3.0

# 纸↔机械臂的等比缩放容差 SCALE_TOL 在 qr_vision.py —— 那儿是定义处，
# hand_eye_calib 也要用它，两边必须同一个数（见 similarity_fit）。


# ─────────────────────────── SDK 加载 ───────────────────────────
# 实现在 src/dobot_sdk.py（全项目唯一入口），此处 import 进来。
# ★ step4_pick_test 用的是 `import step2_teach_coords as tc` 再调 tc.load_sdk()，
#   所以这个名字必须继续在这里可见 —— 不要删。
# （load_sdk / find_port 均在文件头的 import 处引入。）


def read_pose(api, dType) -> dict:
    """读一次实时位姿（GetPose 内部会重试到成功）。"""
    p = dType.GetPose(api)
    return {"x": p[0], "y": p[1], "z": p[2], "r": p[3],
            "j1": p[4], "j2": p[5], "j3": p[6], "j4": p[7]}


def read_pose_stable(api, dType, n: int = POSE_SAMPLES) -> dict:
    """取 n 次读数的**中位数**。单次 GetPose 偶尔会读到正在更新的半帧。"""
    xs, ys, zs, rs = [], [], [], []
    for _ in range(n):
        p = read_pose(api, dType)
        xs.append(p["x"]); ys.append(p["y"]); zs.append(p["z"]); rs.append(p["r"])
    return {"x": float(np.median(xs)), "y": float(np.median(ys)),
            "z": float(np.median(zs)), "r": float(np.median(rs)),
            "j1": p["j1"], "j2": p["j2"], "j3": p["j3"], "j4": p["j4"]}


def read_alarms(api, dType) -> tuple[int, list[int]]:
    """读报警。返回 (result, 字节列表)；非空表示有报警，不该运动。"""
    r = dType.GetAlarmsState(api)
    return r[0], list(r[1])


# ── 报警位编码 ──
# 协议《Dobot-Communication-Protocol-V1.1.5》1.5.1 原话:
#   「数组 alarmsState 中的每一个字节可以标识 8 个报警项的报警状态，
#     且 MSB 在高位，LSB 在低位」
# 所以是 uint8_t[16] = 128 个报警位，编号线性铺开:
#
#     报警号 = 字节序号 * 8 + bit 位      （bit 0 = 该字节的 LSB）
#
# ★ 这里曾经写成 *16，是错的，而且是**自检抓出来的**:
#   ALARM 文档给的「丢步报警 0x50~0x5f」= 80~95，若按 *16 算，
#   0x58~0x5f（88~95）根本落不进 8 个 bit，直接无法表示 —— 一个自洽的
#   编码方案不可能有表示不出来的合法编号。改成 *8 之后 0x40~0x49 与
#   0x50~0x5f 全部落在 bit 0~7 内，自洽。
#
# ★ 注意: 本机 越疆机器人/ 目录里**没有** Dobot 的报警说明文档，只有通信协议。
#   所以下面这套名字只覆盖了能查到的常见项；查不到的只能给编号，不要瞎猜 ——
#   报错信息宁可只说「0x43 未知」，也不能给个错的中文名把人带偏。
ALARM_CODES = {
    0x00: "系统复位（上电后自动置位，属正常现象，用协议指令清除即可）",
    0x01: "未定义指令",
    0x02: "文件系统错误",
    0x03: "MCU 与 FPGA 通信失败",
    0x04: "角度传感器读数异常",
    0x11: "规划目标点不在工作空间内（逆解失败）",
    0x12: "逆解超出关节限位",
}

# 上电后必然出现、且无害、按设计就该被清掉的报警
BENIGN_CODES = {0x00}


def alarm_code_name(code: int) -> str:
    if code in ALARM_CODES:
        return ALARM_CODES[code]
    if 0x40 <= code <= 0x49:
        return "关节限位报警（J1~J4 正/负限位、平行四边形限位）→ 检查姿态"
    if 0x50 <= code <= 0x5F:
        return "丢步报警（轴 1~4）→ **必须重新回零**，否则坐标不可信"
    return "未知（本机没有 Dobot 报警说明文档，只能给编号）"


def decode_alarms(blist: list[int]) -> list[tuple[int, str]]:
    """把报警字节展开成 [(报警号, 说明), ...]。编码见上方注释: 号 = 字节*8 + bit。"""
    out = []
    for bi, b in enumerate(blist):
        for bit in range(8):
            if b & (1 << bit):
                code = bi * 8 + bit
                out.append((code, alarm_code_name(code)))
    return out


def has_alarms(blist: list[int]) -> bool:
    """
    有没有报警。★ 只看**有没有置位**，不看列表长不长。

    ★★ 这里踩过一个很贵的坑，务必理解:
       SDK 的 GetAlarmsState(api) 返回的是 `list(buf)[:length]`，而 length 是
       协议固定的 **16**（uint8_t[16]），所以**永远返回 16 个字节**，没有任何
       报警时就是 16 个 0x00。原来写成 `if alist:` —— 16 个零的列表非空，
       Python 判为 True，于是「没报警」也被当成「有报警」，
       这个门禁**永远过不去**，每次开机都被自己挡住。
       正确判据是 `any(blist)`（有没有任何一位被置 1）。
    """
    return any(blist)


def describe_alarms(blist: list[int]) -> str:
    """一行摘要: 字节的十六进制 + 解出的报警号。"""
    if not has_alarms(blist):
        return "（无）"
    hexs = " ".join(f"{b:02x}" for b in blist)
    codes = " ".join(f"0x{c:02x}" for c, _ in decode_alarms(blist))
    return f"字节[{hexs}]  报警号[{codes}]"


def alarms_are_benign_only(blist: list[int]) -> bool:
    """是不是只有「上电复位」这种按设计就该清掉的报警。"""
    codes = {c for c, _ in decode_alarms(blist)}
    return bool(codes) and codes <= BENIGN_CODES


def needs_homing(blist: list[int]) -> bool:
    """
    有没有「丢步」报警。

    ★ 这是个硬闸门: 电机丢步之后断电/被撞，编码器读数与实际关节角已经对不上，
      零点漂了。这时候量出来的 XY 每一个都是错的，而且还量得很「稳」——
      不会报任何错，只会让整张标定纸悄悄歪掉。所以见到丢步一律不许继续。
    """
    return any(0x50 <= c <= 0x5F for c, _ in decode_alarms(blist))


def clear_alarms(api, dType) -> int:
    """调 ClearAllAlarmsState() 清报警。返回 result，0 表示成功。"""
    return int(dType.ClearAllAlarmsState(api))


def report_alarm_detail(blist: list[int]) -> None:
    """把报警逐条列出来（带中文说明），比一行十六进制好读。"""
    print("\n[报警明细]")
    for code, name in decode_alarms(blist):
        print(f"    0x{code:02x}  {name}")


def resolve_alarms(api, dType, alist: list[int], force_clear: bool) -> tuple[bool, list[int]]:
    """
    拿到报警后决定能不能开工，必要时清掉。返回 (能否运动, 剩余报警)。

    ★ 为什么要「清」而不是直接拒绝: Dobot 控制器上电时**必然**置位
      0x00「系统复位」，这是固件设计如此，清掉是标准流程（协议里 ClearAllAlarmsState
      就是干这个的，非队列、立即生效）。原脚本只读不清，于是每次开机都被自己
      挡住 —— 这正是实际遇到的情况。

    ★ 但「清」不能无脑做: 0x50~0x5f 是丢步，清掉它只会把「零点已经不可信」
      这条唯一线索抹掉，然后量出一整套错的坐标。所以丢步单独拦。

    ★ 其余报警（限位 0x40~0x49 等）默认不动，要用户显式 --clear-alarms ——
      因为那类报警通常意味着物理上还卡着，先得把机械臂挪开再清才有意义。
    """
    benign = alarms_are_benign_only(alist)
    if not (benign or force_clear):
        print("\n✗ 有未清除的报警，拒绝运动。")
        print("  处理办法:")
        print("    1) 若上面是 0x40~0x49 限位报警 → 先断电，用手把机械臂挪回工作范围内")
        print("    2) 确认吸盘/线缆没有卡住、也没有东西挡住关节")
        print("    3) 然后重跑:")
        print("         python3 src/step2_teach_coords.py --clear-alarms")
        print("    4) 若还是清不掉 → 机械臂断电，等 10 秒再上电（上电会重新自检）")
        return False, alist

    why = "只有上电复位报警，按固件设计直接清" if benign else \
          "--clear-alarms: 强制清除（确认机械臂已脱离卡住状态）"
    print(f"\n[清除] {why} → ClearAllAlarmsState()")
    rc = clear_alarms(api, dType)
    time.sleep(0.3)
    _, left = read_alarms(api, dType)
    print(f"[清除] result={rc}  清除后: {describe_alarms(left)}")
    if has_alarms(left):
        # 清不掉一般有两种原因: 物理急停还按着，或者触发报警的条件仍然存在
        print("  ⚠️ 报警没清干净 —— 通常是: ①急停按钮还按着（顺时针转一下弹起）；"
              "②触发报警的原因还在（比如还压在限位上）。")
        return False, left
    print("  ✅ 报警已清空")
    return True, left


# ─────────────────────────── 软限位 ───────────────────────────
def violation(v: float, lo: float, hi: float) -> float:
    """点到区间 [lo,hi] 的距离；在区间内为 0，越界为正。"""
    return max(lo - v, 0.0, v - hi)


def check_target(cur: dict, tgt: dict, limits: dict) -> tuple[bool, str]:
    """
    目标点能不能去？

    规则: 每个轴的目标值都要落在 [lo,hi] 内；若某个轴目标仍越界，则要求
    「越界程度比当前**更小**」—— 也就是只允许往盒子里走、不允许继续往外。
    这样即使机械臂开机时就在限位外（比如上次停在奇怪姿态），也不会把人
    卡死在「一步都动不了」的境地。

    ★ 这里曾经写成 `>=`（原话是「不比当前更小就拒绝」），看着更严、其实是个
      **致命 bug**，实测把人卡死了整整一轮：
        按下 a/d/w/s 想动 XY 时，Z 根本没被指令改动，于是 v1 == v0、
        越界程度完全相等 → `>=` 成立 → **连 XY 都拒绝**。
        屏幕上却只打印 Z 越界，让人以为是 Z 的问题，越查越歪。
      注意这恰好违背了本函数存在的**全部理由**（"不会卡死在一步都动不了"）：
      用一个"防卡死"的规则把机械臂彻底卡死。
    ★ 正确语义只有两条，缺一不可:
        1) 本轴没被指令改动（v1 == v0）→ 不归这条规则管，直接放行；
        2) 本轴改了，且越界程度**变大**（严格 >）→ 才是"继续往外走"，拒绝。
    """
    for ax, (lo, hi) in limits.items():
        v0, v1 = cur[ax], tgt[ax]
        if violation(v1, lo, hi) <= 0:
            continue
        if abs(v1 - v0) < 1e-9:      # 规则 1: 这一轴原地不动 → 放行（★ 别删）
            continue
        if violation(v1, lo, hi) > violation(v0, lo, hi) + 1e-9:   # 规则 2
            return False, (f"{ax.upper()}={v1:.1f} 超出软限位 [{lo:.0f}, {hi:.0f}]"
                           f"（当前 {v0:.1f}）→ 拒绝执行")
    return True, ""


# ─────────────────────────── 运动 ───────────────────────────
def wait_finish(api, dType, timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if dType.GetQueuedCmdMotionFinish(api)[1]:
            return True
        time.sleep(0.03)
    return False


def move_to(api, dType, tgt: dict, limits: dict, mode: str,
            timeout: float = 30.0) -> tuple[bool, str]:
    """
    走一个绝对目标点。返回 (成功, 说明)。

    mode: "movl" 直线(工具走直线，XY 微调时 Z 不会跑) / "movj" 关节
    """
    ptp = (dType.PTPMode.PTPMOVLXYZMode if mode == "movl"
           else dType.PTPMode.PTPMOVJXYZMode)
    r = dType.SetPTPCmd(api, ptp, tgt["x"], tgt["y"], tgt["z"], tgt["r"], isQueued=1)
    if r[0] != 0:
        hint = ""
        if mode == "movl":
            hint = "  （直线模式被拒 → 按 m 切到关节模式再试）"
        return False, f"指令入队失败 result={r[0]}{hint}"
    if not wait_finish(api, dType, timeout):
        return False, f"运动 {timeout:.0f}s 未完成（卡住？按 Ctrl-C，或跑 rescue.py）"
    return True, ""


def verify_arrival(api, dType, tgt: dict) -> tuple[bool, str]:
    """回读位姿，确认真的到位了。"""
    time.sleep(SETTLE_S)
    p = read_pose_stable(api, dType, 5)
    dp = max(abs(p[a] - tgt[a]) for a in ("x", "y", "z"))
    dr = abs(p["r"] - tgt["r"])
    if dp > ARRIVE_TOL_MM or dr > ARRIVE_TOL_DEG:
        return False, (f"到位偏差过大: ΔXYZ={dp:.2f}mm ΔR={dr:.2f}° "
                       f"→ 实测({p['x']:.1f},{p['y']:.1f},{p['z']:.1f})")
    return True, ""


def step_once(api, dType, axes: dict, limits: dict, mode: str):
    """
    在当前位姿基础上加一个增量并走过去。
    axes: {"x": +2.0} 这种，只写要动的轴（其他轴保持不动）。
    返回 (成功, 说明, 新位姿)
    """
    cur = read_pose_stable(api, dType, 5)
    tgt = dict(cur)
    for a, d in axes.items():
        tgt[a] = cur[a] + d

    ok, why = check_target(cur, tgt, limits)
    if not ok:
        return False, why, cur

    ok, why = move_to(api, dType, tgt, limits, mode)
    if not ok:
        return False, why, cur

    ok, why = verify_arrival(api, dType, tgt)
    if not ok:
        # 没到位不算致命：可能是负载/间隙。给出提示但仍让用户继续操作，
        # 只是让他知道这一步没走准。
        return True, "⚠ " + why, read_pose_stable(api, dType, 5)
    return True, "", read_pose_stable(api, dType, 5)


# ─────────────────────────── 校验 ───────────────────────────
def validate_points(points: dict, layout, tol_mm: float = COORD_TOL_MM):
    """
    拿机械臂坐标算出的两两距离，和纸上已知距离比。

    这是唯一能抓住「抄错数字 / P2 P3 记反 / 对错了角」的检查 —— 四点拟合
    本身残差恒为 0，不会告诉你任何错。

    ★★ 比之前多了一层：**先估整体缩放，再比**。
      实测机械臂的 GetPose 和真实毫米不成 1:1（一台 Magician 是 0.833，见
      qr_vision.robot_scale_note），于是六个距离会**一起**少掉同样的比例 ——
      第一版直接拿原始值比 3mm 容差，结果六条一起报警、每条都差几十毫米，
      完全看不出真正的问题在哪，还把人往「先回零」的错路上带。

      所以这里用「纸上的距离 × 缩放」当基准比。缩放是机器固有的、两边（对刀
      与执行）都用同一个映射，会自己抵消，**不是要修的东西**；剩下真正要看的
      是去掉缩放和旋转之后还差多少。

    返回 (problems, details)：problems 是问题字符串列表（空=没问题），
    details 是每对点的 (名称, 纸上mm, 机械臂mm, 原始差mm) —— 表格照旧显示原始值，
    人工核对时比的是原始值。
    """
    problems, details = [], []
    codes = list(layout.codes)
    ref_mm = layout.reference_mm          # 和标定时用的参考点一致
    xy = {c: [points[c]["x"], points[c]["y"]] for c in codes if c in points}

    # 1) 有没有重复点（忘了移动 / 又记了一次同一个位置）
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            if a in xy and b in xy:
                d = float(np.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1]))
                if d < 5.0:
                    problems.append(f"{a} 和 {b} 几乎重合（相距 {d:.1f}mm）"
                                    f"→ 有一次是忘了移动就记录了？")

    # 2) 先看四点形状对不对。判据是**仿射**残差，不是等比缩放的残差。
    #    ★★ 为什么必须是仿射: 机械臂报的 XY 不成 1:1 时，**两个轴的比例还能不一样**
    #       （实测这台 X 方向约 0.81、Y 方向约 0.93）。拿"等比缩放"去拟合，这种
    #       各向异性会全部落进残差里，于是把"机械臂模型差异"误判成"某个点教歪了"，
    #       指错方向、还让人白重对一遍。
    #       仿射（各轴可分别缩放 + 允许剪切）能把这些让掉；让完还剩大残差，
    #       那才是真有点教错了。
    #    ★ 只有 2 个点时仿射定不出来（要 ≥3），退回等比缩放当基准 —— 那条路上
    #      「点没瞄准」和「机械臂不成 1:1」本来就分不开，见下面第 5 段。
    aff = affine_fit(ref_mm, xy, codes)
    aff_worst = aff[1] if aff else 0.0
    sim = similarity_fit(ref_mm, xy, codes)
    n = sim[3] if sim else 0
    scale = sim[0] if sim else 1.0
    if aff is not None:
        L, t = aff[0][:, :2], aff[0][:, 2]
        expect = {c: L @ np.array(ref_mm[c], float) + t
                  for c in codes if c in ref_mm}
    else:
        expect = {c: np.array(ref_mm[c], float) * scale
                  for c in codes if c in ref_mm}

    # 3) 两两距离：基准 = 按"教出来的映射"**应该**量到的机械臂距离。
    #    ★ 不用"纸上距离 × 缩放"当基准 —— 各轴缩得不一样时，单一个缩放乘不出
    #      正确的基准（这正是把上下两条边误报 20mm 的原因, 见 shape_diagnosis）。
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            if a not in xy or b not in xy or a not in expect or b not in expect:
                continue
            want = float(np.hypot(ref_mm[a][0] - ref_mm[b][0],
                                  ref_mm[a][1] - ref_mm[b][1]))
            pred = float(np.linalg.norm(expect[a] - expect[b]))
            got = float(np.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1]))
            ok = abs(got - pred) <= tol_mm
            details.append((f"{a}-{b}", want, pred, got, got - pred, ok))
            if not ok:
                problems.append(f"{a}-{b}: 纸上 {want:.2f}mm，机械臂实测 {got:.2f}mm，"
                                f"比「已教出的形状」该有的 {pred:.2f}mm 差 {got - pred:+.2f}mm"
                                f"（容差 {tol_mm:.0f}mm）")

    # 4) Z 一致性：四个角应该在同一张纸面上，Z 差太多说明测的时候高度不一致
    zs = [points[c]["z"] for c in codes if c in points]
    if len(zs) >= 2 and max(zs) - min(zs) > 5.0:
        problems.append(f"四个角的 Z 相差 {max(zs) - min(zs):.1f}mm —— 纸是平的，"
                        f"Z 应该几乎一样；是不是有的点没降到位就记了？")

    # 5) 形状对不上时，指出到底是**哪一个点**错了（详见 shape_diagnosis）
    #    ★ 「机械臂不成 1:1」（整体缩放 / 各轴比例不同）**不进 problems** ——
    #      它是说明不是错误，混进来会被读成「⚠ 还有问题」，又把方向带偏
    #      （第一版就是这么错的）。要显示它的地方调 scale_note()，单独一块。
    #    ★ 这里只剩下一种情况: 让掉各轴缩放和剪切之后**还**对不上 —— 那才是真错了。
    winding = aff is not None and affine_winding_bad(aff)
    if winding and len(xy) == 4:
        # ★ 先判绕向: 对角相邻的两个角标反（P2↔P3 或 P1↔P4）得到的是原矩形的
        #   对角镜像 —— 而镜像是仿射变换，残差≈0，**能骗过仿射残差检查**
        #   （实测 P2↔P3 互换残差只有 0.29mm）。但**这张纸配这台机器**正确时
        #   行列式必为负（纸面 y 向下 + 机械臂右手系，见 qr_vision.
        #   affine_winding_bad），标反了才变正，一眼就能分开。
        #   ★ 别按"负就是镜像"的直觉写反 —— 那正是第一版的错，会把正常数据拦下。
        problems.append(
            f"四点绕向反了（左转变右转）: 仿射行列式 "
            f"{float(np.linalg.det(aff[0][:, :2])):+.3f} > 0，"
            f"而这张纸配这台机器**正确时必为负**"
            f"（纸面 y 向下 + 机械臂右手系 —— 推导见 qr_vision.affine_winding_bad）。\n"
            f"  ★ 多半是**相邻两个码的标签写反了**（最常见 P2 和 P3、"
            f"其次 P1 和 P4）: 这两对互换个位置，四点形状仍然自洽，但整张纸被镜像了，"
            f"标定会把物体放到对角线的另一侧去。核对一下每个码到底对着纸上哪个角。")
    elif aff is not None and aff_worst > tol_mm and len(xy) == 4:
        problems.append(
            f"四点形状对不上: 让掉「各轴分别缩放 + 剪切」之后**仍然**差 "
            f"{aff_worst:.1f}mm → 是**某一个点**教歪了"
            f"（机器比例的事，不是这里的问题）。")
        problems.extend(shape_diagnosis(layout, xy, codes))
    elif sim is not None and n == 2 and abs(scale - 1.0) > SCALE_TOL:
        # 2 个点怎么拟合残差都是 0，「点没瞄准」和「机械臂不成 1:1」长得一模一样。
        # 这条之所以留在 problems: 此刻**校验是空转的**，得说清"现在看不出什么"。
        problems.append(
            f"只量了 2 个点，比例 {scale:.4f} → 现在分不清是「其中一个点没瞄准」"
            f"还是「机械臂本来就不成 1:1」（2 个点怎么拟合残差都是 0）。\n"
            f"  ★ 再教一个码就能分开: 第三个码配出来的比例还是 {scale:.3f} 左右\n"
            f"    → 是机械臂的固有差异（正常，不影响标定，接着教完就行）；\n"
            f"    否则就是那一个点歪了，重对。")

    not_measured = [c for c in codes if c not in points]
    if not_measured:
        problems.append(f"还没量: {not_measured}")

    return problems, details


def scale_note(points: dict, layout, short: bool = False) -> str:
    """
    「机械臂报的 XY 不是毫米」的**说明**文字（正常时、或四点形状不自洽时返回空串）。

    ★ 单独一个函数，是为了一处定义、各处显示一致（表格、记录时、结尾）。
      不自洽时返回空串: 那种情况下比例本身没有意义（实测「P2/P3 记反」能让它
      算出 0.004），该由 validate_points 去说「哪个点教歪了」。
    ★ 判据用**仿射**残差而不是等比缩放的残差 —— 机械臂两个轴的比例本来就可以
      不一样（实测 X 0.81 / Y 0.93）。用等比去卡，正常机器会被误判成"不自洽"，
      这段说明就永远显示不出来。
    """
    xy = {c: [points[c]["x"], points[c]["y"]]
          for c in layout.codes if c in points}
    aff = affine_fit(layout.reference_mm, xy, layout.codes)
    if aff is None or aff[1] > COORD_TOL_MM:
        return ""
    if affine_winding_bad(aff):
        # 绕向反了（多半是 P2↔P3 标反）时残差可能很小，比例值却是假的 —— 不能说。
        return ""
    L = aff[0][:, :2]
    axes = (float(np.linalg.norm(L[:, 0])), float(np.linalg.norm(L[:, 1])))
    # ★★ "等比缩放"这个数必须从**仿射**的两个轴比例来，**不能**用相似拟合的:
    #    纸上坐标 y 向下 → 这套数据里带一个镜像，而相似拟合只会（等比缩放+旋转），
    #    复数拟合出来的 |c| 跟真实比例毫无关系。实测: 一套完全自洽、两轴都 1:1 的
    #    坐标，它能报出 0.453（RMS 111mm）→ 于是**凭空**说「机械臂不成 1:1、走
    #    10mm 只报 4.5mm」，把一台好机器说成坏的。两轴接近时几何平均就是那个"等比"。
    scale = float(np.sqrt(axes[0] * axes[1]))
    return robot_scale_note(scale, len(xy), short=short, axes=axes)


# ─────────────────────────── 终端单键输入 ───────────────────────────
class RawKeys:
    """
    单键读取（不用敲回车，操作手感才跟得上）。

    ★ 用 cbreak 而不是 raw：cbreak 保留 ISIG，**Ctrl-C 仍然能中断**。
      用 raw 的话 Ctrl-C 会被当成普通字节吞掉，机器人动起来就停不住了。
    ★ __exit__ 里务必复原终端，否则 Ctrl-C 之后终端会变成不显示回显。
    """

    def __init__(self, stream=sys.stdin):
        self.stream = stream
        self.fd = stream.fileno()
        self.saved = None

    def __enter__(self):
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
        return False

    def get(self) -> str:
        try:
            ch = self.stream.read(1)
        except (KeyboardInterrupt, EOFError):
            raise
        return ch


# ─────────────────────────── 交互式对刀 ───────────────────────────
def fmt(p: dict) -> str:
    return (f"X={p['x']:8.2f}  Y={p['y']:8.2f}  Z={p['z']:8.2f}  R={p['r']:7.2f}")


KEY_HELP = """
按键说明
  移动（一小步，绝对坐标）
    w / s   Y + / Y -          a / d   X - / X +
    r / f   Z + / Z -          t / g   R + / R -
  其他
    1 / 2 / 3   步长 10mm / 2mm / 0.2mm
    m           切换 直线(MOVL) / 关节(MOVJ)
    j           回到本码已记录的位置
    space       记录当前码的 XY
    n / p       下一个码 / 上一个码
    l           列出已记录的点 + 校验结果
    h           帮助        q  结束并保存
"""

AXIS_KEYS = {
    "w": {"y": +1}, "s": {"y": -1},
    "a": {"x": -1}, "d": {"x": +1},
    "r": {"z": +1}, "f": {"z": -1},
    "t": {"r": +1}, "g": {"r": -1},
}


def touch_off(api, dType, args, keys: RawKeys, limits: dict):
    """
    触底：慢慢降 Z，直到吸盘刚碰到纸面，记下这个 Z 当「纸面」。

    为什么要单独做一步: 之后所有运动都把 z 锁在「纸面 − BELOW_PAPER_MM」上，
    机械臂就不可能把吸盘深深压进桌面。不给它一个已知的地板，软限位只是猜的。
    """
    print("\n" + "─" * 68)
    print("第 0 步: 触底（确定纸面高度）")
    print("─" * 68)
    print("接下来只让 Z 下降。请一直盯着吸盘和纸面之间的缝:")
    print("  · 缝隙刚消失（吸盘软胶轻轻接触纸面）就按  space  记下纸面高度")
    print("  · 一旦发现压下去了，立刻按  r  抬起来")
    print("  · 按  q  跳过触底 → 之后 Z 的下限就是工作软限位，可能够不到纸；"
          "更推荐下次直接用 --table-z 指定纸面高度")
    print("\n当前位姿: " + fmt(read_pose_stable(api, dType)))

    step = 2.0
    while True:
        print(f"\r[触底·只调Z] 步长 {step:.1f}mm   s 下降 / r 上升 / 1粗 2中 3细"
              f" / space 确认 / q 跳过   （左右前后请先结束触底） ", end="")
        k = keys.get()

        if k == "q":
            print("\n已跳过触底。")
            return None
        if k in STEPS:
            step = STEPS[k][1]
            continue
        if k == " ":
            p = read_pose_stable(api, dType)
            print(f"\n✓ 纸面高度记为 Z = {p['z']:.2f} mm")
            print(f"   → 对刀期间 Z 下限 = {p['z'] - BELOW_PAPER_MM:.2f}"
                  f"（纸面再往下 {BELOW_PAPER_MM:.0f}mm）")
            return float(p["z"])
        if k not in ("s", "r"):
            # ★ 别静默吞键: 触底这一步**只有** s/r 能让机械臂动，用户按 a/d/w
            #   会「毫无反应」，看着像脚本坏了。实际上只是还没进对刀步骤。
            #   实测就有人卡在这里，以为按键失灵。
            if k in AXIS_KEYS:
                print(f"\n  ⚠ 现在还在「触底」——这一步只调 Z 高度。"
                      f"{k} 是左右前后，得先结束触底。")
                print("     结束方法: 按 space 记下纸面高度，或按 q 跳过触底。")
            continue

        # 只动 Z
        d = (-step if k == "s" else +step)
        ok, why, cur = step_once(api, dType, {"z": d}, limits, args.mode)
        if not ok:
            print(f"\n  ✗ {why}")
        else:
            print(f"\r  {fmt(read_pose_stable(api, dType))}" + ("   " + why if why else ""))


def teach_loop(api, dType, args, layout, limits: dict, z_floor: float | None):
    """P1→P4 逐个微调对准并记录。"""
    codes = list(layout.codes)
    ref_mm = layout.reference_mm
    points: dict[str, dict] = {}
    step_key = DEFAULT_STEP_KEY
    mode = args.mode
    i = 0
    fails = 0

    print("\n" + "─" * 68)
    print("开始逐个对刀")
    print("─" * 68)
    print(f"参考点: {layout.reference}（{REFERENCE_LABEL[layout.reference]}）")
    print(f"运动方式: {'直线 MOVL' if mode == 'movl' else '关节 MOVJ'}"
          f"   软限位: " + "  ".join(f"{a.upper()}[{lo:.0f},{hi:.0f}]"
                                    for a, (lo, hi) in limits.items()))
    if z_floor is not None:
        print(f"Z 下限: {z_floor:.2f}（纸面）—— 不会压进桌面")
    print("\n" + KEY_HELP)

    with RawKeys() as keys:
        while True:
            code = codes[i]
            print(f"\n▶ 请把吸盘【正中心】对准 {code} 的参考点"
                  f"（纸上坐标 X={ref_mm[code][0]:.2f} Y={ref_mm[code][1]:.2f} mm）")
            if code in points:
                print(f"  （已记录: X={points[code]['x']:.2f} Y={points[code]['y']:.2f}，"
                      f"按 j 可回到那里）")
            print(f"  当前: {fmt(read_pose_stable(api, dType))}   步长 {STEPS[step_key][1]}mm")
            print("  > ", end="")
            k = keys.get()

            # ── 退出 ──
            if k in ("q", "\x03", "\x04"):
                print("\n结束对刀。")
                return points

            # ── 帮助 / 列表 ──
            if k == "h":
                print(KEY_HELP)
                continue
            if k == "l":
                print_points_table(points, layout)
                problems, details = validate_points(points, layout)
                print_detail(details)
                for p in problems:
                    print(f"  ⚠ {p}")
                if not problems:
                    print("  ✅ 校验通过")
                continue

            # ── 步长 / 运动方式 ──
            if k in STEPS:
                step_key = k
                print(f"  步长 = {STEPS[k][1]}mm ({STEPS[k][0]})")
                continue
            if k == "m":
                mode = "movj" if mode == "movl" else "movl"
                print(f"  运动方式 → {'直线 MOVL' if mode == 'movl' else '关节 MOVJ'}")
                continue

            # ── 换码 ──
            if k == "n":
                i = min(i + 1, len(codes) - 1)
                continue
            if k == "p":
                i = max(i - 1, 0)
                continue

            # ── 回到已记录位置 ──
            if k == "j":
                if code not in points:
                    print("  这个码还没记录过。")
                    continue
                tgt = {a: points[code][a] for a in ("x", "y", "z", "r")}
                ok, why = move_to(api, dType, tgt, limits, mode)
                print(f"  {'✓ 回到' if ok else '✗ '} (X={tgt['x']:.2f} Y={tgt['y']:.2f})"
                      + ("" if ok else f" {why}"))
                continue

            # ── 记录 ──
            if k == " ":
                p = read_pose_stable(api, dType)
                if p["z"] < (limits["z"][0] - 0.01):
                    print(f"  ✗ 当前 Z={p['z']:.2f} 低于安全下限，先抬起来再记录")
                    continue
                points[code] = p
                print(f"  ✓ {code} 记录: X={p['x']:.3f}  Y={p['y']:.3f}  (Z={p['z']:.2f})")
                # 立即和纸上布局比一下已量到的点
                problems, details = validate_points(points, layout)
                hard = [q for q in problems if "还没量" not in q]
                if hard:
                    for q in hard:
                        print(f"    ⚠ {q}")
                    print("    → 按 space 重记，或 j 回去微调；确认无误可直接 n 继续")
                else:
                    print("    距离校验: ✅ 与纸上布局一致")
                    # 缩放是说明、不是错误，所以放在这行**后面**、不跟 ⚠ 一起
                    # （第一版跟错误混着打，被读成故障了）。一行版，全文按 l。
                    sn = scale_note(points, layout, short=True)
                    if sn:
                        print(f"    {sn}")
                # 自动跳到下一个未记录的码
                nxt = next((c for c in codes[i + 1:] if c not in points), None)
                if nxt:
                    i = codes.index(nxt)
                continue

            # ── 移动 ──
            if k in AXIS_KEYS:
                step = STEPS[step_key][1]
                axes = {a: s * step for a, s in AXIS_KEYS[k].items()}
                # Z 的下限锁在「纸面 − BELOW_PAPER_MM」上（触底之后）
                if "z" in axes and z_floor is not None:
                    cur = read_pose_stable(api, dType, 5)
                    if cur["z"] + axes["z"] < z_floor - BELOW_PAPER_MM:
                        ax_lo = z_floor - BELOW_PAPER_MM
                        print(f"\r  ✗ 会低于下限 Z={ax_lo:.2f}"
                              f"（纸面 {z_floor:.2f} 再往下 {BELOW_PAPER_MM:.0f}，"
                              f"当前 {cur['z']:.2f}）→ 拒绝")
                        continue
                ok, why, cur = step_once(api, dType, axes, limits, mode)
                if not ok:
                    fails += 1
                    print(f"\r  ✗ {why}")
                    if fails >= 3:
                        print("  连续 3 次失败 → 停下来。请检查: 是否到工作范围边缘？"
                              "换个方向，或按 m 换运动方式。")
                        fails = 0
                else:
                    fails = 0
                    print(f"\r  {fmt(cur)}" + ("   " + why if why else "") + "      ")
                continue

            # 其它键（回车等）: 忽略，但**要出声** —— 静默吞键会让人以为按键失灵
            if k not in ("\r", "\n"):
                print(f"  ⚠ 未识别的键 {k!r}。移动键只有 w/s/a/d/r/f/t/g，按 h 看完整说明。")


def print_points_table(points: dict, layout) -> None:
    print(f"\n  {'码':<4}{'X(mm)':>10}{'Y(mm)':>10}{'Z(mm)':>10}   纸上参考点(mm)")
    for c in layout.codes:
        if c in points:
            p, mm = points[c], layout.reference_mm[c]
            print(f"  {c:<4}{p['x']:>10.3f}{p['y']:>10.3f}{p['z']:>10.3f}"
                  f"   X={mm[0]:.2f} Y={mm[1]:.2f}")
        else:
            print(f"  {c:<4}{'—':>10}{'—':>10}{'—':>10}   (未量)")

    # 纸↔机械臂的比例：一眼看出「机械臂坐标成不成 1:1」。
    # ★ 别写成 ❌ —— 实测这个比例本来就不是 1.000（一台 Magician 是 X 0.81 / Y 0.93），
    #   对刀和执行两边用同一个映射、会自己抵消，不是要修的东西。
    #   真正要看的是**让掉各轴缩放和剪切之后的残差**（那才是"点有没有教错"）。
    # ★★ 分两个轴看，不能只给一个数: 两轴比例本来就可以不一样，只报一个"缩放"
    #   会让人以为 X/Y 一样缩，后面按 mm 推 Z 下降量就全错了。
    xy = {c: [points[c]["x"], points[c]["y"]] for c in layout.codes if c in points}
    aff = affine_fit(layout.reference_mm, xy, layout.codes)
    sim = similarity_fit(layout.reference_mm, xy, layout.codes)
    if aff is not None:
        L = aff[0][:, :2]
        kx = float(np.linalg.norm(L[:, 0]))
        ky = float(np.linalg.norm(L[:, 1]))
        far = max(abs(kx - 1.0), abs(ky - 1.0)) > SCALE_TOL
        rot_s = f"   旋转 {sim[1]:+.1f}°（纸怎么摆都行，与坐标系无关）" if sim else ""
        print(f"\n  纸→机械臂: 各轴比例 X {kx:.4f} / Y {ky:.4f}"
              f"{'  ⚠ 与 1 差得多（一般正常: 机械臂的模型差异）' if far else '  ✅ ≈1'}"
              f"{rot_s}")
        ok = aff[1] <= COORD_TOL_MM
        if affine_winding_bad(aff):
            # ★ 绕向反了时残差可能很小（P2↔P3 标反只有 0.29mm），打 ✅ 会骗人
            print(f"    去掉「各轴比例 + 剪切」后残差 {aff[1]:.2f}mm"
                  f"  ⚠ 但绕向反了（行列式 "
                  f"{float(np.linalg.det(aff[0][:, :2])):+.3f} > 0，正确必为负）"
                  f"→ 标签写反了，上面的比例别信")
        else:
            print(f"    去掉「各轴比例 + 剪切」后残差 {aff[1]:.2f}mm"
                  f"{'  ✅ 四点自洽' if ok else '  ⚠ 形状不对 → 是点教错了，不是比例的事'}")
        note = scale_note(points, layout)
        if note:
            print("\n  " + note.replace("\n", "\n  "))
    elif sim is not None:
        print(f"\n  纸→机械臂: 缩放 {sim[0]:.4f}  旋转 {sim[1]:+.1f}°"
              f"（只 {sim[3]} 个点: 分不清是「点没瞄准」还是「机械臂不成 1:1」）")


def print_detail(details) -> None:
    """
    ★ 「差」列是**实测 − 预期**，预期来自"贴到已教出的四点上的那个映射"
      （各轴比例、剪切都算进去了）。⚠ 就是这个差超了容差 → 形状对不上的那几条。

    ★★ 为什么不比「实测 − 纸上」: 机械臂报的 XY 本来就不是毫米（而且 X/Y 两个轴
      的比例还不一样 —— 实测 0.81 / 0.93）。拿纸上的毫米直接减，六条会一起差
      几十毫米、条条亮 ⚠，完全看不出真正是哪一条不对（第一版就是这么错的，
      还把人往「先回零」带）。上面 print_points_table 会打出各轴比例，对着看就明白。
    """
    if not details:
        return
    print(f"\n  {'点对':<9}{'纸上(mm)':>11}{'预期(mm)':>11}{'实测(mm)':>11}{'差(mm)':>10}")
    print("   （预期 = 按已教出的四点映射该量到的值；⚠ = 实测与预期超差）")
    for name, want, pred, got, d, ok in details:
        print(f"  {name:<9}{want:>11.2f}{pred:>11.2f}{got:>11.2f}{d:>10.2f}"
              f"{'' if ok else '  ⚠'}")


def collected_coords(layout, points: dict) -> dict:
    """整理成 {码: [X, Y]}，即 hand_eye_calib 要的那份。"""
    return {c: [round(points[c]["x"], 4), round(points[c]["y"], 4)]
            for c in layout.codes if c in points}


# ─────────────────────────── 保存 ───────────────────────────
def save(points: dict, layout, meta: dict) -> None:
    coords = collected_coords(layout, points)
    # 把这次量到的「纸→机械臂」比例一起存下来（见 qr_vision.affine_fit）。
    # ★ 它是个**机器常数**: 同一台机器正常就该一直是这两个数。
    #   以后换纸/重对刀再量一次，若它变了 → 机器动过了（撞过、皮带松、联轴器
    #   打滑…），不用等标定歪掉才发现。放进 JSON 就是留个底，不参与任何计算。
    # ★ 存**两个轴**的比例: 实测这台 X 方向 0.81、Y 方向 0.93，两个轴并不一样，
    #   只存一个"缩放"是把各向异性抹掉了 —— 而各向异性恰恰是后面按 mm 推
    #   Z 下降量、算速度时最容易踩的坑。
    # ★★ 只在四点形状自洽时才存: 有点教歪的时候，拟合出来的比例是没意义的
    #   （实测"P2/P3 记反"能让它算出 0.004），存下去就是留了个假常数。
    aff = affine_fit(layout.reference_mm, coords, layout.codes)
    sim = similarity_fit(layout.reference_mm, coords, layout.codes)
    scale_info = None
    #   ★ 绕向反了时也要拦住: 残差可能很小（P2↔P3 标反 0.29mm），但比例是假的
    #     （那时行列式>0，两个"轴比例"已经不代表纸的 X/Y 了）。
    if aff is not None and aff[1] <= COORD_TOL_MM and not affine_winding_bad(aff):
        L = aff[0][:, :2]
        scale_info = {
            "scale_x": round(float(np.linalg.norm(L[:, 0])), 5),   # 报告值 ÷ 真实毫米
            "scale_y": round(float(np.linalg.norm(L[:, 1])), 5),
            "rot_deg": round(sim[1], 4) if sim else None,
            "rms_mm": round(aff[1], 3),   # 去掉各轴比例/剪切后的残差: ≈0 = 四点自洽
            "points_used": len(coords),
            "note": ("机器常数，仅作参考，标定里会自己抵消；"
                     "换纸重测时若明显变化说明机器动过"),
        }
    data = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "reference": layout.reference,
        "paper_json": PAPER_JSON.name,
        "source": "step2_teach_coords.py（吸盘中心对准参考点，读 GetPose）",
        "dobot_coords": coords,
        "robot_scale": scale_info,
        "raw": {c: {k: round(float(v), 4) for k, v in p.items()}
                for c, p in points.items()},
        **meta,
    }
    ensure_output_dir()
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n已保存: {OUT_JSON}")
    if scale_info:
        print(f"  记下机器常数 各轴比例 X={scale_info['scale_x']:.5f} "
              f"Y={scale_info['scale_y']:.5f} "
              f"(残差 {scale_info['rms_mm']:.2f}mm / {scale_info['points_used']}点) "
              f"—— 换纸重测时拿它比对，变了说明机器动过")
    else:
        print("  （四点形状不自洽，这次不记机器常数 —— 教歪的时候算出来的比例是假的）")

    print("\n" + "═" * 62)
    print("  粘进 step3_hand_eye_calib.py 的 DOBOT_COORDS（或者直接跑，它会自动读本文件）")
    print("═" * 62)
    print("DOBOT_COORDS = {")
    for c in layout.codes:
        v = coords.get(c)
        s = f"[{v[0]:.3f}, {v[1]:.3f}]" if v else "[None, None]"
        mm = layout.reference_mm[c]
        print(f'    "{c}": {s},   # 纸上参考点 X={mm[0]:.2f} Y={mm[1]:.2f}')
    print("}")
    print("═" * 62)


# ─────────────────────────── 主流程 ───────────────────────────
def run(args) -> int:
    layout = load_paper_layout(PAPER_JSON, reference=args.reference)
    limits = {a: tuple(v) for a, v in DEF_LIMITS.items()}
    if args.limits:
        for spec in args.limits.split(","):
            ax, lo, hi = spec.split(":")
            limits[ax.strip()] = (float(lo), float(hi))

    print("=" * 68)
    print("  逐个对刀：量 P1~P4 在机械臂坐标系里的 XY")
    print("=" * 68)
    print(f"标定纸: {PAPER_JSON.name}  ({layout.paper_mm[0]:.0f}x{layout.paper_mm[1]:.0f}mm)")
    print(f"参考点: {layout.reference} —— {REFERENCE_LABEL[layout.reference]}"
          f"（ref_point_guide.png 里对应颜色的十字）")
    print("\n⚠ 安全提醒:")
    print("  · 清空机械臂周围 60cm，手别放在运动路径上")
    print("  · 全程低速；Ctrl-C 随时中止（会立即停住）")
    print("  · 本脚本不会回零、不会改末端偏移参数")
    print("  · ★ 开机后**先确认已回零**（home_arm.py 或 DobotStudio 点「回零」）:")
    print("      丢步、被撞、或手掰过关节之后，编码器零点就不对了，")
    print("      这时量出来的 XY 会「稳定地错」——不报错、也看不出，但整张标定纸会歪。")
    if not sys.stdin.isatty():
        print("\n✗ 需要在交互式终端里运行（要读单键）。")
        return 1
    input("按回车开始连接… ")

    api, dType = load_sdk()
    port = find_port(dType, args.port)
    if not port:
        print("✗ 没找到机械臂串口。电源开了吗？USB 插好了吗？")
        print("  换个口/换根线试试；也可以用 --port /dev/ttyUSB0 手动指定")
        return 1
    print(f"\n[连接] {port} @ 115200 …")
    ret = dType.ConnectDobot(api, port, 115200)
    if ret[0] != 0:
        print(f"✗ 连接失败 state={ret[0]} (1=未找到设备 2=端口被占用)")
        return 1
    print(f"[连接] 成功  fwType={ret[1]}  version={ret[2]}")

    z_floor = None
    try:
        dType.SetCmdTimeout(api, 5000)

        # ── 只读: 报警 / 末端偏移 / 当前位姿 ──
        ares, alist = read_alarms(api, dType)
        # ★ 判据是 any(alist) 不是 alist —— 见 has_alarms() 的注释，
        #   GetAlarmsState 就算没报警也会返回 16 个 0x00。
        print(f"[报警] result={ares}  {describe_alarms(alist)}")
        # 只读打印，绝不写回 —— 这两个参数是上次卡死事故的根源
        ep = dType.GetEndEffectorParams(api)
        print(f"[末端偏移] (只读) result={ep[0]} xBias={ep[1]:.2f} "
              f"yBias={ep[2]:.2f} zBias={ep[3]:.2f}")
        print(f"[当前位姿] {fmt(read_pose_stable(api, dType))}")

        if has_alarms(alist):
            report_alarm_detail(alist)
            if needs_homing(alist):
                # 丢步 = 编码器零点已经不可信，这时候量出来的坐标是「稳定地错」
                print("\n✗ 出现「丢步报警」: 电机丢步时关节实际角度和编码器读数已经对不上，")
                print("  零点漂了 —— 此时量出来的 XY 会全部带同一个固定偏移，而且不会任何报错。")
                print("  必须先重新回零，再回来对刀:")
                print("     python3 tools/home_arm.py")
                print("  或用 DobotStudio 点「回零」，完成后重跑本脚本。")
                return 1
            ok, alist = resolve_alarms(api, dType, alist, args.clear_alarms)
            if not ok:
                return 1

        dType.SetQueuedCmdClear(api)
        dType.SetQueuedCmdStartExec(api)
        # 低速：关节与坐标参数都压下来
        dType.SetPTPJointParams(api, args.speed, args.speed, args.speed, args.speed,
                                args.acc, args.acc, args.acc, args.acc, isQueued=0)
        _set_coord_params(api, dType, args.speed, args.acc)
        dType.SetPTPCommonParams(api, args.ratio, args.ratio, isQueued=0)
        print(f"[参数] 关节/坐标速度={args.speed:.0f} 加速度={args.acc:.0f} "
              f"速度比例={args.ratio:.0f}%")

        with RawKeys() as keys:
            if args.table_z is not None:
                z_floor = args.table_z
                print(f"[纸面] 用命令行给的 Z={z_floor:.2f}（跳过触底）")
            else:
                # ★ 触底期间用「探测下限」而不是工作下限 —— 理由见 PROBE_Z_FLOOR
                #   的注释: 探测下限若高于纸面，这一步永远做不完。
                probe = {**limits, "z": (args.probe_z, limits["z"][1])}
                print(f"\n[触底] Z 临时放宽到 {args.probe_z:.0f}"
                      f"（工作下限 {limits['z'][0]:.0f}）")
                print("       这段时间**只有你的眼睛**在保护桌面 —— 盯住吸盘与纸的缝隙")
                z_floor = touch_off(api, dType, args, keys, probe)

        if z_floor is not None:
            limits["z"] = (z_floor - BELOW_PAPER_MM, limits["z"][1])
            print(f"[软限位] Z 下限锁定为 {limits['z'][0]:.2f}"
                  f"（纸面 {z_floor:.2f} 往下 {BELOW_PAPER_MM:.0f}）")

        points = teach_loop(api, dType, args, layout, limits, z_floor)

        # ── 收尾: 停队列 ──
        dType.SetQueuedCmdStopExec(api)

        # ── 汇总校验 ──
        print("\n" + "═" * 68)
        print("  结果")
        print("═" * 68)
        print_points_table(points, layout)
        problems, details = validate_points(points, layout)
        print_detail(details)
        if problems:
            print("\n⚠ 还有问题:")
            for p in problems:
                print(f"  · {p}")
        else:
            print("\n✅ 四点齐全，两两距离与纸上布局一致。")

        save(points, layout, {
            "port": port,
            "fw": f"{ret[1]} {ret[2]}",
            "end_effector": [ep[1], ep[2], ep[3]],
            "mode": args.mode,
            "table_z": z_floor,
            "limits": {a: list(v) for a, v in limits.items()},
            "problems": problems,
        })

        if problems:
            print("\n→ 有问题就别往下跑标定。改完重跑本脚本（已记录的点会提示，"
                  "按 j 可回到原处微调）。")
        else:
            print("\n→ 下一步: python3 src/step3_hand_eye_calib.py")
        return 0

    except KeyboardInterrupt:
        print("\n[中止] Ctrl-C —— 正在紧急停止…")
        return 130
    finally:
        # ★ 不管怎么退出（正常/异常/Ctrl-C），都先把队列强制停掉再断开：
        #   否则 Python 被中断后控制器可能还在执行残留指令 —— 这正是
        #   rescue.py 存在的原因，这里从一开始就不留这种尾巴。
        try:
            dType.SetQueuedCmdForceStopExec(api)
            dType.SetQueuedCmdClear(api)
        except Exception:
            pass
        try:
            dType.DisconnectDobot(api)
            print("[断开] 完成（队列已强制停止并清空）")
        except Exception:
            pass


def _set_coord_params(api, dType, xyz_v, xyz_a) -> None:
    """MOVL(直线) 需要坐标参数，否则用固件默认值，行为不可预期。"""
    class PTPCoordinateParams(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("xyzVelocity", ctypes.c_float), ("rVelocity", ctypes.c_float),
                    ("xyzAcceleration", ctypes.c_float), ("rAcceleration", ctypes.c_float)]
    p = PTPCoordinateParams(xyz_v, xyz_v / 2, xyz_a, xyz_a / 2)
    api.SetPTPCoordinateParams(ctypes.byref(p), False, ctypes.byref(ctypes.c_uint64(0)))


# ─────────────────────────── 离线自检 ───────────────────────────
class _FakeSdk:
    """
    离线自检用的假 SDK：不用机械臂也能验证「限位 / 步长 / 到位校验」的逻辑。

    ★ 之所以能这样测，是因为运动相关函数都把 dType 当参数传进来 ——
      不是去 import 全局模块。副作用是这里可以塞一个替身进来。
    """
    class PTPMode:
        PTPMOVJXYZMode = 1
        PTPMOVLXYZMode = 2
    PTPMOVLXYZMode = 2
    PTPMOVJXYZMode = 1

    def __init__(self, pose=None, refuse=(), alarms=None):
        self.pose = dict(pose or {"x": 200.0, "y": 0.0, "z": 50.0, "r": 0.0,
                                  "j1": 0.0, "j2": 45.0, "j3": 45.0, "j4": 0.0})
        self.sent = []          # 记录收到的所有目标点
        self.refuse = set(refuse)   # 让某些 PTP 模式返回错误
        self.alarms = list(alarms) if alarms is not None else [0] * 16
        self.cleared = 0        # ClearAllAlarmsState 被调用了几次

    def GetAlarmsState(self, api, maxLen=32):
        return [0, list(self.alarms)]

    def ClearAllAlarmsState(self, api):
        self.cleared += 1
        self.alarms = [0] * len(self.alarms)
        return 0

    def SetPTPCmd(self, api, ptp, x, y, z, r, isQueued=0):
        if ptp in self.refuse:
            return [3, 0]
        self.sent.append({"ptp": ptp, "x": x, "y": y, "z": z, "r": r})
        self.pose.update(x=x, y=y, z=z, r=r)
        return [0, len(self.sent)]

    def GetQueuedCmdMotionFinish(self, api):
        return [0, True]

    def GetPose(self, api):
        p = self.pose
        return [p["x"], p["y"], p["z"], p["r"], p["j1"], p["j2"], p["j3"], p["j4"]]


def selftest() -> int:
    layout = load_paper_layout(PAPER_JSON, reference=DEFAULT_REFERENCE)
    limits = {a: tuple(v) for a, v in DEF_LIMITS.items()}
    fake = _FakeSdk()
    api = object()
    ok_all = True

    def check(name, cond, extra=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  {'✅' if cond else '❌'} {name}" + (f"   {extra}" if extra else ""))

    print("=" * 68)
    print("  对刀脚本离线自检（不需要机械臂）")
    print("=" * 68)

    print("\n[1] 软限位")
    # ★ 边界一律**从当前 limits 现算**，不写死数字。
    #   2026-09-18 把 x 下限从 100 放宽到 50 时，原来写死的 X=350 由「盒外」
    #   变成了「盒内」，"[1]" 这一节当场失效 —— 测试数字必须跟着参数走。
    xlo, xhi = limits["x"]
    xm = (xlo + xhi) / 2.0
    x_out = xhi + 10.0
    ok, why = check_target({"x": xm, "y": 0, "z": 50, "r": 0},
                           {"x": xhi + 20.0, "y": 0, "z": 50, "r": 0}, limits)
    check(f"目标 X>{xhi:.0f} 超出上限 → 拒绝", not ok, why)
    ok, why = check_target({"x": xm, "y": 0, "z": 50, "r": 0},
                           {"x": xm + 10.0, "y": 0, "z": 50, "r": 0}, limits)
    check("目标在盒内 → 放行", ok)
    # 已在限位外时，只准往回走
    ok_out, _ = check_target({"x": x_out, "y": 0, "z": 50, "r": 0},
                             {"x": x_out + 10.0, "y": 0, "z": 50, "r": 0}, limits)
    ok_in, _ = check_target({"x": x_out, "y": 0, "z": 50, "r": 0},
                            {"x": xm, "y": 0, "z": 50, "r": 0}, limits)
    check("已越界时继续往外走 → 拒绝", not ok_out)
    check("已越界时往盒里走 → 放行（避免卡死）", ok_in)

    # ★★ 回归测试：Z 贴在纸上（正好等于下限）时，动 XY 必须放行。
    #    真机实测: 机械臂停在纸面 Z=-77.97、下限也是 -77.97，此时按 a/d/w/s
    #    想对刀 XY，却被判「Z 超出软限位 → 拒绝执行」—— 一个键都动不了。
    #    根因是旧代码用 `>=` 判越界：Z 没被指令改动，越界程度相等 → 误判为
    #    "继续往外走"。这条测试就是把那个场景钉死。
    zt = {"x": 100.0, "y": -260.0, "z": -78.0, "r": -180.0}
    zb = {"x": 330.0, "y": 260.0, "z": 160.0, "r": 180.0}
    zlim = {"x": (100.0, 330.0), "y": (-260.0, 260.0),
            "z": (-77.97, 160.0), "r": (-180.0, 180.0)}
    on_paper = {"x": 199.36, "y": 10.0, "z": -77.97, "r": 0.0}
    for ax, d in (("x", +2.0), ("x", -2.0), ("y", +2.0), ("y", -2.0)):
        t = dict(on_paper); t[ax] += d
        ok_xy, why_xy = check_target(on_paper, t, zlim)
        check(f"贴在纸面时动 {ax}{d:+.0f} → 放行（曾经被误拒）", ok_xy, why_xy)
    # 但同时必须仍然挡住"往纸里扎"和"往上抬不越界"
    t_down = dict(on_paper); t_down["z"] -= 2.0
    ok_d, _ = check_target(on_paper, t_down, zlim)
    check("贴在纸面时继续下降 → 仍然拒绝（保护没被削弱）", not ok_d)
    t_up = dict(on_paper); t_up["z"] += 2.0
    ok_u, _ = check_target(on_paper, t_up, zlim)
    check("贴在纸面时抬升 → 放行", ok_u)
    # 全轴都越界的极端情形下，原地不动也不该被卡死
    ok_stuck, why_stuck = check_target(zt, dict(zt), zlim)
    check("所有轴都在限位外且原地不动 → 仍放行（不自锁）", ok_stuck, why_stuck)
    _ = zb

    # ★ 探测下限必须比工作下限更松，否则触底时够不到纸、这一步永远做不完
    #   （实测被这条坑过: 默认 -60 时吸盘离纸还有一截就被软限位拦住）
    check("触底探测下限比工作下限更低（不然触底没法做）",
          PROBE_Z_FLOOR < DEF_LIMITS["z"][0],
          f"探测 {PROBE_Z_FLOOR:.0f} < 工作 {DEF_LIMITS['z'][0]:.0f}")

    print("\n[2] 步长与绝对坐标")
    fake = _FakeSdk()
    for _ in range(3):
        step_once(api, fake, {"x": +2.0}, limits, "movl")
    check("连走 3 步 +2mm 恰好到 X=206（绝对坐标不漂）",
          abs(fake.pose["x"] - 206.0) < 1e-6, f"实测 {fake.pose['x']}")
    check("每次都发的是绝对目标点", [round(s["x"], 3) for s in fake.sent] == [202.0, 204.0, 206.0])

    print("\n[3] 到位校验 / 错误上报")
    fake = _FakeSdk(refuse=(2,))          # 让 MOVL 失败
    ok, why, _ = step_once(api, fake, {"x": +2.0}, limits, "movl")
    check("MOVL 被拒 → 报错并提示可切 MOVJ", not ok and "m" in why, why)
    ok, why, _ = step_once(api, fake, {"x": +2.0}, limits, "movj")
    check("同一目标换 MOVJ → 成功", ok, why)

    print("\n[4] 距离校验（抓抄错/P2P3 记反）—— 这是唯一有识别力的检查")
    ang = np.deg2rad(37.0)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    off = np.array([-250.0, 80.0])

    # ★★ 造数据必须照**真实物理摆法**造，否则自检会假红/假绿。
    #    · 纸上坐标是 y **向下**的（calib_A4_qr.json: P1 左上 y=39.5、P3 左下 y=170.5）；
    #    · 纸正面朝上摊在桌上，机械臂 XY 是右手系（Z 朝上）。
    #    从上往下看: 机械臂 X→Y 逆时针，纸面 u→v（右→下）顺时针 —— 顺逆改不了，
    #    所以「纸 → 机械臂」的行列式**必为负**（就是下面 flip 里那个 −1）。
    #    ydown=True（默认）造的就是这套**正确**坐标；写成 [u·kx, v·ky] 会造出一份
    #    物理上不可能出现的坐标（行列式为正），会被（正确地）判成"标签绕反"。
    #    ydown=False 只给**相似拟合自己**的自检用: 相似拟合只有一个等比缩放，
    #    遇到真实摆法（含镜像）还原不出缩放/角度，那是"判据该用仿射"的原因，
    #    不是相似拟合算错了。要单测它的还原能力，就得喂它同向数据。
    def mk(perm=None, scale=1.0, dup=None, axes=None, ydown=True):
        mm = {c: np.array(layout.reference_mm[c], float) for c in layout.codes}
        if perm:
            for a, b in perm:
                mm[a], mm[b] = mm[b].copy(), mm[a].copy()
        k = np.array(axes if axes else (1.0, 1.0), float) * scale
        flip = np.array([1.0, -1.0]) if ydown else np.array([1.0, 1.0])
        pts = {}
        for c, v in mm.items():
            w = (R @ (v * k * flip)) + off
            pts[c] = {"x": float(w[0]), "y": float(w[1]), "z": 30.0, "r": 0.0}
        if dup:
            pts[dup[1]] = dict(pts[dup[0]])
        return pts

    problems, details = validate_points(mk(), layout)
    check("正确的一整套坐标 → 无问题", not problems, str(problems))

    # ★ P2/P3 互换**不能**指望"两两距离"抓住 —— 它是个仿射自洽的镜像
    #   （实测六条距离偏差都在 0.5mm 内，逐对比距离全放行）。抓住它的是
    #   validate_points 里的**行列式符号**那条，见下面 [4] 里专门的用例。
    p2, _ = validate_points(mk(perm=[("P2", "P3")]), layout)
    check("P2/P3 记反 → 被抓住", bool(p2), (p2[0] if p2 else "一条都没报"))
    p3, _ = validate_points(mk(dup=("P1", "P2")), layout)
    check("两个码记成同一点 → 被抓住", any("重合" in x for x in p3), (p3[0] if p3 else ""))
    p5b, _ = validate_points({k: v for k, v in mk().items() if k != "P4"}, layout)
    check("缺一个码 → 提示还没量", any("还没量" in x for x in p5b))

    # ★★ 纸↔机械臂的缩放: 用来把「某个点教歪了」和「机械臂本来就不成 1:1」分开。
    #    ★ 分开之后**两边都不是故障** —— 后者的处理是"知道它不是毫米，别当毫米用"，
    #      不是回零重来（第一版写成了"先回零"，方向完全错）。
    #      所以这里要钉住的正是「别对缩放本身报错」这条。
    s0, r0, rms0, n0 = similarity_fit(layout.reference_mm, mk(ydown=False), layout.codes)
    check("同向摆法 → 相似拟合还原出 1.0000 / 37° / RMS≈0（单测它自己的还原能力）",
          abs(s0 - 1.0) < 1e-9 and abs(abs(r0) - 37.0) < 1e-6 and rms0 < 1e-9,
          f"缩放 {s0:.6f} 旋转 {r0:.4f}° RMS {rms0:.2e}")

    bad = mk()
    bad["P4"]["x"] += 20.0                    # 只把 P4 一个点教歪 20mm
    bad_flat = mk(ydown=False)                # 同一场景的同向版，专给相似拟合看缩放
    bad_flat["P4"]["x"] += 20.0
    s1, _r1, rms1, _ = similarity_fit(layout.reference_mm, bad_flat, layout.codes)
    check("只歪一个点 → 缩放仍≈1，但 RMS 变大（不是缩放问题）",
          abs(s1 - 1.0) < SCALE_TOL and rms1 > COORD_TOL_MM,
          f"缩放 {s1:.4f}(内) RMS {rms1:.2f}mm(外)")
    pb, _ = validate_points(bad, layout)
    check("只歪一个点 → 报「某一个点教歪」，不扯到「不成 1:1」（免得误导）",
          any("某一个点" in x for x in pb) and not any("不成 1:1" in x for x in pb),
          (pb[0] if pb else ""))

    # 整体缩到 82%: 实测机械臂就是这个量级(5/6)，必须**只报告、不报警**。
    # 逐对距离也不再六条一起叫 —— 那正是第一版把人带偏的地方。
    shrink = mk(scale=0.82)
    # 缩放那个数由相似拟合读出，所以这一条要喂它同向数据（见 mk 的 ydown 说明）
    s2, _r2, rms2, _ = similarity_fit(layout.reference_mm,
                                      mk(scale=0.82, ydown=False), layout.codes)
    check("整体缩到 82% → 缩放被量出来，且 RMS 仍≈0（是缩放不是单点）",
          abs(s2 - 0.82) < 1e-6 and rms2 < 1e-6,
          f"缩放 {s2:.4f} RMS {rms2:.2e}")
    ps, ds = validate_points(shrink, layout)
    check("整体缩到 82% → 逐对距离不再逐条误报（那是缩放的账，不是点的账）",
          not any(x.startswith("P1-P2:") or x.startswith("P2-P4:") for x in ps),
          " / ".join(x[:24] for x in ps))
    check("整体缩到 82% → 表格里也不打 ⚠（差列是原始值，⚠ 由归一化后判）",
          all(ok for *_x, ok in ds))
    # ★ 缩放要作为**说明**单独给出，绝不混进 problems —— 混进去会被读成
    #   「⚠ 还有问题」，人就去回零了（第一版真发生过）。这条专门钉住这个分寸。
    note = scale_note(shrink, layout)
    check("整体缩到 82% → 说明里点明「不成 1:1」且**不影响手眼标定**",
          "不成 1:1" in note and "不影响手眼标定" in note, note.splitlines()[0])
    check("整体缩到 82% → 给出「连走三次量一量」的可重复性验证法",
          "可重复" in note)
    check("整体缩到 82% → 缩放**不许**出现在 problems 里（会把说明读成故障）",
          not any("缩放" in x or "1:1" in x for x in ps),
          " / ".join(x[:28] for x in ps))
    check("整体缩到 82% → 短版说明只有一行（记录时用，不刷屏）",
          len(scale_note(shrink, layout, short=True).splitlines()) == 1)
    check("正常情况 → 说明为空串（别没事找事）", scale_note(mk(), layout) == "")
    check("拟合不自洽(P4 歪 20mm) → 说明为空（此时缩放值没有意义）",
          scale_note(bad, layout) == "")

    # ★★ 各向异性: X 缩 0.81、Y 缩 0.94 —— 这台 Magician 的实测行为。
    #    这是把判定从「相似残差」改成「仿射残差」的原因:
    #    相似拟合只有一个缩放，遇到两轴不同必然留下大残差 → 一台**好**机器
    #    会被永远判成「某个点教歪了」→ 又变成「误拦卡死标定」，跟缩放那轮同病。
    aniso = mk(axes=(0.81, 0.94))
    # 相似残差那一条喂**同向**数据: 要单独看"各向异性"这一件事，
    # 不能和"纸面 y 反向"（它同样会让相似拟合残差变大）混在一起说。
    _sa, _ra, rmsa, _na = similarity_fit(layout.reference_mm,
                                        mk(axes=(0.81, 0.94), ydown=False), layout.codes)
    _aa, aff_worst_a = affine_fit(layout.reference_mm, aniso, layout.codes)
    check("两轴不同 → 相似残差大、仿射残差≈0（说明该用仿射判）",
          rmsa > COORD_TOL_MM and aff_worst_a < 1e-6,
          f"相似RMS {rmsa:.2f}mm / 仿射残差 {aff_worst_a:.2e}mm")
    pa, da = validate_points(aniso, layout)
    check("两轴比例不同 X0.81/Y0.94 → **不许**报问题（真实现象，得放行）",
          not pa, " / ".join(x[:30] for x in pa))
    check("两轴不同 → 表格里也不打 ⚠（⚠ 按仿射残差判）",
          all(ok for *_x, ok in da))
    notea = scale_note(aniso, layout)
    check("两轴不同 → 说明里点明「两个轴的比例还不一样」并给出两个数",
          "两个轴" in notea and "0.81" in notea and "0.94" in notea,
          notea.splitlines()[0] if notea else "(空)")
    check("两轴不同且拟合自洽 → 也**不许**出现在 problems 里",
          not any("缩放" in x or "1:1" in x for x in pa))

    # ★★ 对角相邻两角标反（P2↔P3）: 得到的四点正好是原矩形的**对角镜像**。
    #    镜像是仿射变换 → 残差≈0（实测 0.29mm），能骗过仿射残差检查 ——
    #    但这个错**很致命**（整张纸镜像，物体被放到对角线另一侧）。
    #    判据只能是**行列式符号**: 这张纸配这台机器正确时必为负（y 向下 + 右手系）。
    pswap, _ = validate_points(mk(perm=[("P2", "P3")]), layout)
    check("P2/P3 标反（对角镜像，仿射残差查不出）→ 靠行列式 >0 抓住",
          any("绕向反了" in x for x in pswap), (pswap[0] if pswap else "没报"))
    check("P2/P3 标反 → 绕向反了时不许记机器常数（那时的比例是假的）",
          scale_note(mk(perm=[("P2", "P3")]), layout) == "")

    # ★ 四点形状对不上时，要点名「哪个点教歪了」（只说「有一个点」没用）。
    #   模拟真实故障: P4 瞄到了数据区**左上角**，不是右上角 —— 偏一个数据区宽度，
    #   方向沿纸面 x（机器人坐标里就是 R·x̂；沿机器人 x̂ 偏是另一个方向，
    #   落点不在任何角上，那就成不了"铁证"，测试也没意义）。
    diag = mk()
    _d4 = R @ np.array([-26.0, 0.0])
    diag["P4"]["x"] += float(_d4[0])
    diag["P4"]["y"] += float(_d4[1])
    pd, _ = validate_points(diag, layout)
    check("四点形状不自洽 → 报「某一个点教歪了」并点名 P4",
          any("某一个点" in x for x in pd) and any(x.strip().startswith("★")
                                                 and "P4" in x for x in pd),
          next((x for x in pd if x.strip().startswith("★")), " / ".join(pd[:2])))
    # ★ 判据是「修正后落在同码的哪个已知参考点」，不是「谁挪得最少」——
    #   对角的 P3 往往也只要挪差不多（实测两个都正好是 26.00mm，比不出来）。
    check("点名 P4 的依据是落点对上「数据区左上角」（不是靠比大小）",
          any("左上角" in x and "P4" in x for x in pd),
          next((x for x in pd if "左上角" in x), " / ".join(pd[:3])))

    # 小幅缩水(1.5%)落在 SCALE_TOL 内: 量得出来，但不值得报警 ——
    # 机械臂自己就常不成 1:1，这个量级根本分不清是尺子还是模型。
    p4, _ = validate_points(mk(scale=0.985), layout)
    s4, _r4, _rms4, _ = similarity_fit(layout.reference_mm,
                                       mk(scale=0.985, ydown=False), layout.codes)
    check("整体缩水 1.5% → 缩放量成 0.9850，但不报警（分不清尺子还是模型）",
          abs(s4 - 0.985) < 1e-6 and not p4, (p4[0] if p4 else f"缩放{s4:.4f}"))

    # 2 个点时缩放是精确解、残差恒 0，绝不能拿 RMS 当证据
    two = {c: mk()[c] for c in ("P1", "P2")}
    s3, _r3, _rms3, n3 = similarity_fit(layout.reference_mm, two, layout.codes)
    check("只有 2 个点 → 仍能量出缩放但点数为 2（不该拿残差说事）",
          n3 == 2 and abs(s3 - 1.0) < 1e-9, f"n={n3} 缩放 {s3:.6f}")
    p6, _ = validate_points({c: v for c, v in mk(scale=0.82).items()
                             if c in ("P1", "P2")}, layout)
    check("2 个点且比例 0.82 → 提示「再教一个码就能分开」",
          any("再教一个码" in x for x in p6), (p6[-1] if p6 else ""))

    print("\n[5] 纸上布局与参考点（供你操作时核对）")
    for c in layout.codes:
        mm = layout.reference_mm[c]
        print(f"  {c}: {layout.reference}({REFERENCE_LABEL[layout.reference]})"
              f"  X={mm[0]:7.2f}  Y={mm[1]:7.2f} mm")

    # ★ 怎么验「五个参考点没串」？
    #   ★★ 注意: **不能**用「两两距离」去验 —— 五个模式之间是纯平移关系，
    #      平移不改变任何两点间距离，所以五种模式的相邻码间距全都一样
    #      （实测都是 227.76）。这正是四点拟合残差恒为 0 的同一个退化性。
    #      拿距离去验，不管串没串都会通过 = 假绿灯。（我第一版就是这么写的，错的。）
    #   正确的验法: 以 corner_tl 为基准算偏移向量，必须满足
    #      tr - tl = (S, 0)   bl - tl = (0, S)   br - tl = (S, S)   c - tl = (S/2, S/2)
    #   且 S 对四个码都相同。这样"tr 读成了 top_left"之类的映射错会立刻露馅。
    offs, sides = [], []
    for c in layout.codes:
        tl = np.array(layout.mm_for("corner_tl")[c], float)
        o = {m: np.array(layout.mm_for(m)[c], float) - tl for m in REFERENCE_MODES}
        offs.append(o)
        sides.append(float(np.linalg.norm(o["corner_tr"])))
    S = float(np.mean(sides))
    check("五个参考点都能取到（四角 + 中心）", len(layout.points_mm) == len(REFERENCE_MODES))
    check("四个码的数据区边长一致（同一套码）",
          max(sides) - min(sides) < 0.05, f"{min(sides):.2f}~{max(sides):.2f} mm")
    # 每个码的偏移向量都必须正好落在那四个位置（容差 0.05mm）
    want = {"corner_tr": (S, 0.0), "corner_bl": (0.0, S),
            "corner_br": (S, S), "center": (S / 2, S / 2)}
    worst = 0.0
    for o in offs:
        for m, (wx, wy) in want.items():
            worst = max(worst, float(np.linalg.norm(o[m] - np.array([wx, wy]))))
    check("各参考点的相对位置正确（tr 在 +X / bl 在 +Y / c 在正中）",
          worst < 0.05, f"最大偏差 {worst:.3f} mm，数据区边长 S={S:.2f}mm")
    check("数据区边长 ≈ 二维码边长 × 21/29（说明取的是数据区、不是含白边外框）",
          abs(S - layout.qr_side_mm * 21 / 29) < 0.05,
          f"S={S:.2f} vs 按比例 {layout.qr_side_mm * 21 / 29:.2f} mm")
    # 默认参考点必须是有效值，且真正被用上
    check(f"默认参考点 {DEFAULT_REFERENCE} 有效且已生效",
          DEFAULT_REFERENCE in REFERENCE_MODES
          and layout.reference == DEFAULT_REFERENCE
          and layout.reference_mm is layout.mm_for(DEFAULT_REFERENCE),
          f"{DEFAULT_REFERENCE} = {REFERENCE_LABEL[DEFAULT_REFERENCE]}")

    print("\n[6] 报警解码与处理")
    # 报警号 → 位: 字节序号 = 号 // 8，字节内第 (号 % 8) 位
    def mkbits(*codes):
        b = [0] * 16
        for c in codes:
            b[c // 8] |= 1 << (c % 8)
        return b

    # ★★ 回归测试: 真实 SDK 没报警时返回的是 **16 个 0x00**（协议定长），
    #    曾经用 `if alist:` 判空 → 16 个零非空 → 永远误判为「有报警」。
    check("全零 16 字节 = 真的没报警（不能用 len 判空）",
          not has_alarms([0] * 16) and describe_alarms([0] * 16) == "（无）")
    check("全零 16 字节 ≠ 「只有上电复位」",
          not alarms_are_benign_only([0] * 16))
    check("0x00 上电复位 → 能被解出且判为无害",
          [c for c, _ in decode_alarms(mkbits(0x00))] == [0x00]
          and alarms_are_benign_only(mkbits(0x00)))
    check("0x11 → 落在字节[2] 的 bit1（不是字节[1]）",
          mkbits(0x11)[2] == 0x02 and [c for c, _ in decode_alarms(mkbits(0x11))] == [0x11])
    check("同一字节里两个报警能同时解出",
          [c for c, _ in decode_alarms(mkbits(0x00, 0x03))] == [0x00, 0x03])
    check("丢步 0x50~0x5f → 触发「必须先回零」闸门",
          needs_homing(mkbits(0x50)) and needs_homing(mkbits(0x5F)))
    check("限位 0x40 不算丢步（回零救不了它）",
          not needs_homing(mkbits(0x40)))
    # ★ 这条是当初抓出「*16 写错」的那个不变量，别再删:
    #   ALARM 文档列出的合法编号必须**全部**能被编码表示出来。
    doc_ranges = [(0x00, 0x04, "基础报警"), (0x11, 0x12, "规划/逆解"),
                  (0x40, 0x49, "关节限位"), (0x50, 0x5F, "丢步")]
    unrepresentable = [c for lo, hi, _ in doc_ranges for c in range(lo, hi + 1)
                       if not decode_alarms(mkbits(c))]
    check("文档里所有合法报警号都能被本编码解出（*8 而不是 *16）",
          not unrepresentable, f"表示不出来的: {unrepresentable}")

    # ★ 只在调用 resolve_alarms 的那一行静音 —— 别把 check() 也一起吞了，
    #   否则里面的 ❌ 会被写进 StringIO，自检「失败但看不到是哪条」。
    def resolve_quiet(fake, bits, force):
        with contextlib.redirect_stdout(io.StringIO()):
            return resolve_alarms(api, fake, bits, force_clear=force)

    f = _FakeSdk(alarms=mkbits(0x00))
    okc, left = resolve_quiet(f, mkbits(0x00), False)
    check("只有上电复位 0x00 → 自动清除后放行",
          okc and not has_alarms(left) and f.cleared == 1)

    f = _FakeSdk(alarms=mkbits(0x40))
    okc, _ = resolve_quiet(f, mkbits(0x40), False)
    check("限位 0x40、没给 --clear-alarms → 拒绝且不写控制器",
          (not okc) and f.cleared == 0)
    okc, left = resolve_quiet(f, mkbits(0x40), True)
    check("限位 0x40、给了 --clear-alarms → 清除后放行",
          okc and not has_alarms(left) and f.cleared == 1)

    # 清不掉的场景: 让 ClearAllAlarmsState 假装无效（急停还按着）
    class _StuckSdk(_FakeSdk):
        def ClearAllAlarmsState(self, api):
            self.cleared += 1
            return 0            # 报警位不消失

    f = _StuckSdk(alarms=mkbits(0x00))
    okc, left = resolve_quiet(f, mkbits(0x00), False)
    check("清不掉（急停还按着）→ 不放行，如实报告剩余报警",
          (not okc) and has_alarms(left) and f.cleared == 1)

    # 走一遍 run() 里的真实取值路径: read_alarms(fake) → 门禁判空
    _, fresh = read_alarms(api, _FakeSdk())
    check("run() 门禁: 真实路径读回来的全零 16 字节 → 放行（原 bug 就死在这）",
          not has_alarms(fresh) and len(fresh) == 16)

    print("\n[7] 机器常数落盘（robot_points.json 里的 robot_scale）")
    # ★ 这个数的用途: 换纸/重对刀时拿它对一下 —— 变了说明机器动过
    #   （撞过/拆过/皮带松/联轴器打滑），不是拿来参与标定计算的。
    #   所以它必须**跟着每次对刀自动更新**，不能靠手填（手填的会过期）。
    global OUT_JSON
    with tempfile.TemporaryDirectory() as td:
        saved, OUT_JSON = OUT_JSON, Path(td) / "robot_points.json"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                save(shrink, layout, {"problems": []})
            d = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        finally:
            OUT_JSON = saved
    rs = d.get("robot_scale") or {}
    check("落盘时自动记下**各轴**比例（不用手填）",
          abs(rs.get("scale_x", 0) - 0.82) < 1e-6
          and abs(rs.get("scale_y", 0) - 0.82) < 1e-6,
          f"scale_x={rs.get('scale_x')} scale_y={rs.get('scale_y')}")
    check("一并记下残差和用到的点数（判断这个数可不可信）",
          "rms_mm" in rs and rs.get("points_used") == 4,
          f"rms={rs.get('rms_mm')} n={rs.get('points_used')}")
    check("缩放不进 problems（它是说明，不是故障）",
          not any("缩放" in x for x in d.get("problems", [])))

    # ★ 两轴不同的时候，落盘的两个数必须**分别是**两轴的比例 ——
    #   只记一个综合值的话，X 0.81 / Y 0.94 会被平均成 0.875 混过去，
    #   漂移比对也就跟着失灵（撞过之后 X 涨 Y 跌正是这个样子）。
    with tempfile.TemporaryDirectory() as td:
        saved, OUT_JSON = OUT_JSON, Path(td) / "robot_points.json"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                save(mk(axes=(0.81, 0.94)), layout, {"problems": []})
            d2 = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        finally:
            OUT_JSON = saved
    rs2 = d2.get("robot_scale") or {}
    check("两轴不同 → 落盘分别是两轴的比例（不平均成一个数）",
          abs(rs2.get("scale_x", 0) - 0.81) < 1e-6
          and abs(rs2.get("scale_y", 0) - 0.94) < 1e-6,
          f"scale_x={rs2.get('scale_x')} scale_y={rs2.get('scale_y')}")
    check("两轴不同的这套 → 照样不进 problems",
          not any("缩放" in x for x in d2.get("problems", [])))

    print("\n" + "═" * 68)
    print("  ✅ 自检全部通过" if ok_all else "  ❌ 有失败项")
    print("═" * 68)
    return 0 if ok_all else 1


# ─────────────────────────── 入口 ───────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="逐个对刀：量 P1~P4 在机械臂坐标系里的 XY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python3 src/step2_teach_coords.py --selftest        # 离线自检，不用机械臂\n"
               "  python3 src/step2_teach_coords.py                   # 交互式对刀（含触底）\n"
               "  python3 src/step2_teach_coords.py --table-z -70     # 已知纸面高度，跳过触底\n"
               "  python3 src/step2_teach_coords.py --probe-z -110    # 纸面比 -95 还低时\n"
               "  python3 src/step2_teach_coords.py --clear-alarms    # 报警清不掉时\n")
    ap.add_argument("--selftest", action="store_true", help="离线自检，不连接机械臂")
    ap.add_argument("--clear-alarms", action="store_true",
                    help="清除报警（上电复位报警会自动清，无需本开关；"
                         "限位等其它报警需显式加此开关；丢步报警拒绝清除）")
    ap.add_argument("--port", default=None, help="串口，默认自动查找")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    choices=REFERENCE_MODES,
                    help=f"对刀瞄哪个点（默认 {DEFAULT_REFERENCE}=数据区右上角）: "
                         + " / ".join(f"{m}={REFERENCE_LABEL[m]}" for m in REFERENCE_MODES))
    ap.add_argument("--table-z", type=float, default=None,
                    help="纸面高度 Z（省掉交互触底，并直接锁定 Z 下限）")
    ap.add_argument("--probe-z", type=float, default=PROBE_Z_FLOOR,
                    help=f"触底期间的 Z 探测下限，默认 {PROBE_Z_FLOOR:.0f}"
                         "（纸面比这还低就调更低，例如 -110）")
    ap.add_argument("--mode", default="movl", choices=("movl", "movj"),
                    help="运动方式：movl 直线(默认) / movj 关节")
    ap.add_argument("--speed", type=float, default=40.0,
                    help="速度 mm/s（关节/坐标），默认 40，越小越安全")
    ap.add_argument("--acc", type=float, default=40.0, help="加速度，默认 40")
    ap.add_argument("--ratio", type=float, default=30.0,
                    help="PTP 速度比例 %%（0~100），默认 30")
    ap.add_argument("--limits", default=None,
                    help='覆盖软限位，如 "x:50:380,y:-310:310"（格式 轴:下限:上限）')
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断]")
        sys.exit(130)
