#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dobot_check.py —— 只读体检：连上机械臂看一眼关键状态，**不发任何运动指令**
=========================================================================
为什么单独有这么一个脚本:
  · step2_teach_coords.py 要在交互式终端里跑、还会占住串口，不适合用来排查问题；
  · 它也不打印 HOMEParams（回零原点），而那个数正是"Z 读数整体偏了"的头号嫌疑
    （本项目出过一次事故: 有人把回零原点写成了 (200,200,200,200)）。
  · 这个脚本不需要 tty，可以在任何时候跑，纯读。

网上查不到结论的时候，先跑这个，把输出贴出来 —— 比猜快得多。

用法:
  python3 tools/dobot_check.py                 # 自动找串口
  python3 tools/dobot_check.py --port /dev/ttyUSB1

★ 安全: 全程只调 GetXxx()。唯一的两条写指令是最后清空命令队列
  （SetQueuedCmdForceStopExec + Clear）—— 那是为了防止残留指令继续执行，
  属于安全收尾，不是运动指令。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))     # 公共库在 src/

from dobot_sdk import find_port, load_sdk        # noqa: E402
from qr_vision import DATA_SIDE_MM, QR_SIDE_MM   # noqa: E402  只用常量，顺便验证能 import

# 报警号 = 字节序号*8 + bit（协议 1.5.1: 每字节 8 个报警项）
ALARM_CODES = {
    0x00: "系统复位（上电自动置位，正常，可清）",
    0x01: "未定义指令", 0x02: "文件系统错误",
    0x03: "MCU 与 FPGA 通信失败", 0x04: "角度传感器读数异常",
    0x11: "规划目标点不在工作空间内（逆解失败）", 0x12: "逆解超出关节限位",
}


def alarm_name(code: int) -> str:
    if code in ALARM_CODES:
        return ALARM_CODES[code]
    if 0x40 <= code <= 0x49:
        return "关节限位报警 → 检查姿态"
    if 0x50 <= code <= 0x5F:
        return "丢步报警 → 必须重新回零，否则坐标不可信"
    return "未知（本机无报警说明文档，只能给编号）"


def get_alarms(dType, api) -> list[int]:
    r = dType.GetAlarmsState(api)
    return list(r[1])


def decode(blist: list[int]) -> list[tuple[int, str]]:
    return [(i * 8 + j, alarm_name(i * 8 + j))
            for i, b in enumerate(blist) for j in range(8) if b & (1 << j)]


def main() -> int:
    ap = argparse.ArgumentParser(description="只读体检（不发运动指令）")
    ap.add_argument("--port", default=None, help="串口，默认自动查找")
    args = ap.parse_args()

    api, dType = load_sdk()
    port = find_port(dType, args.port)
    if not port:
        print("✗ 没找到串口。机械臂电源开了吗？USB 插好了吗？")
        print("  ls /dev/serial/by-id/  看看有没有 CP210x")
        return 1

    print("=" * 66)
    print("  机械臂只读体检（不会动）")
    print("=" * 66)
    print(f"[连接] {port} @ 115200 …")
    ret = dType.ConnectDobot(api, port, 115200)
    if ret[0] != 0:
        print(f"✗ 连接失败 state={ret[0]}"
              f"（1=未找到设备；2=端口被占用 —— 先关掉 DobotStudio / 其它脚本）")
        return 1
    print(f"[连接] 成功  fwType={ret[1]}  version={ret[2]}")

    try:
        dType.SetCmdTimeout(api, 3000)

        # ── 1. 报警 ──
        blist = get_alarms(dType, api)
        items = decode(blist)
        print(f"\n[报警] 有报警 = {bool(items)}")
        print(f"  16 字节: {' '.join(f'{b:02x}' for b in blist)}")
        if items:
            for code, name in items:
                print(f"  0x{code:02x}  {name}")
            if any(0x50 <= c <= 0x5F for c, _ in items):
                print("  ⚠️ 有丢步报警 → 先回零，否则下面所有坐标都不可信")
        else:
            print("  （无）")

        # ── 2. 回零原点（头号嫌疑）──
        hp = dType.GetHOMEParams(api)
        print(f"\n[回零原点 HOMEParams] result={hp[0]}")
        print(f"  X={hp[1]:.2f}  Y={hp[2]:.2f}  Z={hp[3]:.2f}  R={hp[4]:.2f}")
        # Magician 的常规出厂回零点就是 (200, 0, 0, 0)。事故里被写成 (200,200,200,200)。
        sane = (abs(hp[1] - 200) < 1 and abs(hp[2]) < 1
                and abs(hp[3]) < 1 and abs(hp[4]) < 1)
        if sane:
            print("  ✅ 看起来是出厂值 (200, 0, 0, 0)")
        else:
            print("  ⚠️ 不是常规的 (200,0,0,0)。回零会按这套原点去跑 ——")
            print("     若这几个数被改过，回零后所有坐标读数都会整体偏移。")
            print("     本项目的事故记录里出现过 (200,200,200,200)。")

        # ── 3. 末端工具偏移 ──
        ep = dType.GetEndEffectorParams(api)
        print(f"\n[末端工具偏移 EndEffectorParams] result={ep[0]}")
        print(f"  xBias={ep[1]:.2f}  yBias={ep[2]:.2f}  zBias={ep[3]:.2f}")
        if abs(ep[1]) > 1 or abs(ep[2]) > 1 or abs(ep[3]) > 1:
            print("  ⚠️ 有明显非零偏移。zBias 非零会直接让 Z 读数整体平移；")
            print("     事故记录里出现过 xBias=71.6。吸盘若垂直装在法兰中心，这里应接近 0。")
        else:
            print("  ✅ 接近 0，符合「吸盘直接装在法兰上」")

        # ── 4. 当前位姿 / 关节角 ──
        p = dType.GetPose(api)
        print(f"\n[当前位姿] X={p[0]:.2f}  Y={p[1]:.2f}  Z={p[2]:.2f}  R={p[3]:.2f}")
        print(f"[关节角  ] J1={p[4]:.2f}  J2={p[5]:.2f}  J3={p[6]:.2f}  J4={p[7]:.2f}")
        if abs(p[0] - 200) < 1 and abs(p[1]) < 1 and abs(p[2]) < 1:
            print("  ↑ 正好是回零点 (200,0,0)")

        # ── 5. 队列 ──
        q = dType.GetQueuedCmdCurrentIndex(api)
        print(f"\n[命令队列] result={q[0]}  当前索引={q[1]}")
        print("  （索引不为 0 只说明以前跑过指令，不代表还在动）")

        print("\n" + "=" * 66)
        print("  读完了。把上面整段贴出来即可定位问题。")
        print(f"  参考: 标定纸数据区边长 {DATA_SIDE_MM:.2f}mm，码含白边 {QR_SIDE_MM:.2f}mm")
        print("=" * 66)
        return 0
    finally:
        try:
            dType.SetQueuedCmdForceStopExec(api)
            dType.SetQueuedCmdClear(api)
        except Exception:
            pass
        try:
            dType.DisconnectDobot(api)
            print("\n[断开] 完成（队列已停并清空；全程未发运动指令）")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
