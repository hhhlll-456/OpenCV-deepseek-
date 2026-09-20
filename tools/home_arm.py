#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
home_arm.py —— 回零（让机械臂回到零位）
=============================================================================
★ 这是一个**会真动**的脚本，而且回零是幅度最大的一种运动:
  机械臂会先展开、扫掠一段，再收拢到零位。跑之前把周围和上方 60cm 清空。

什么时候必须回零:
  · 出现「丢步报警」（0x50~0x5F）—— 不回零的话，之后读到的所有坐标都是
    「稳定地错」，而且不报错、残差里也看不出来；
  · 手掰过机械臂、撞过、急停过；
  · 每次重新上电后不确定零位对不对。

零位 = 关节角 (0°, 45°, 45°, 0°)。回零成功的标志: 零位附近微调一下、
蜂鸣器响、指示灯变绿。

★ 本脚本**只发 SetHOMECmd，绝不设置 HOMEParams**。
  这是从一次事故里留下来的规矩: 曾经有人把回零原点 HOMEParams 写成了
  (200,200,200,200)，结果回零按那套原点去跑，之后所有坐标读数整体偏移。
  出厂值应是 (200,0,0,0) —— 想核对就跑 `python3 tools/dobot_check.py` 看，
  那是只读的。要改 HOMEParams 请用 DobotStudio，不要在脚本里顺手改。

用法:
  python3 tools/home_arm.py                    # 自动找串口，8 秒倒计时后回零
  python3 tools/home_arm.py --port /dev/ttyUSB0
  python3 tools/home_arm.py --yes              # 跳过倒计时（确定周围没人时）

回零失败/中途想停:
  Ctrl-C 只是**停止本脚本**，已经下发的回零指令可能还在控制器里继续执行。
  要真正停住，重跑本脚本时选 n，或用 DobotStudio 的停止。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))     # 公共库在 src/

from dobot_sdk import find_port, load_sdk        # noqa: E402

ZERO_JOINTS = (0.0, 45.0, 45.0, 0.0)             # 越疆 Magician 零位
TOL_DEG = 3.0                                    # 判定「到位」的容差
WAIT_TIMEOUT_S = 120                             # 回零最久等这么久
COUNTDOWN_S = 8


def main() -> int:
    ap = argparse.ArgumentParser(description="回零（会让机械臂大幅运动）")
    ap.add_argument("--port", default=None, help="串口，默认自动查找")
    ap.add_argument("--yes", action="store_true", help="跳过倒计时确认")
    args = ap.parse_args()

    api, dType = load_sdk()
    port = find_port(dType, args.port)
    if not port:
        print("✗ 没找到机械臂串口。电源开了吗？USB 插好了吗？")
        print("  换个口/换根线试试；也可以用 --port /dev/ttyUSB0 手动指定")
        return 1

    print(f"[连接] {port} @ 115200 …")
    ret = dType.ConnectDobot(api, port, 115200)
    if ret[0] != 0:
        print(f"✗ 连接失败 state={ret[0]} (1=未找到设备 2=端口被占用"
              f" —— 先关掉 DobotStudio / 其它脚本)")
        return 1
    print(f"[连接] 成功  fwType={ret[1]}  version={ret[2]}")

    try:
        dType.SetCmdTimeout(api, 5000)

        # ── 先只读: 报警 + 当前姿态（让人知道现在是什么状态）──
        p = dType.GetPose(api)
        print(f"\n[当前] X={p[0]:.1f} Y={p[1]:.1f} Z={p[2]:.1f} "
              f"J=({p[4]:.1f}, {p[5]:.1f}, {p[6]:.1f}, {p[7]:.1f})")
        print(f"[目标] 零位 J={ZERO_JOINTS}")

        ares, blist = dType.GetAlarmsState(api)
        if any(blist):
            print(f"[报警] 有报警（result={ares}）: "
                  f"{' '.join(f'{b:02x}' for b in blist)}")
            print("  （回零正是丢步报警的处理手段，所以这里不拦；")
            print("    但如果是「急停还按着」，回零也不会动 —— 先松开急停）")
        else:
            print("[报警] 无")

        # ── 确认 + 倒计时（照抄原脚本的行为，别省）──
        if not args.yes:
            print("\n⚠ 即将回零! 机械臂会展开扫掠再收拢到零位，")
            print("  请清空机械臂四周及上方 60cm 空间")
            ans = input("  确认继续？(y/N) ").strip().lower()
            if ans != "y":
                print("已取消，未发任何运动指令。")
                return 0
            for i in range(COUNTDOWN_S, 0, -1):
                print(f"  {i} 秒后开始… (Ctrl-C 中止)", end="\r", flush=True)
                time.sleep(1)
            print(" " * 40, end="\r")

        # ── 发回零指令 ──
        # 只发 SetHOMECmd，不带 HOMEParams（见文件头的事故说明）。
        dType.SetQueuedCmdClear(api)
        dType.SetQueuedCmdStartExec(api)
        dType.SetPTPJointParams(api, 60, 60, 60, 60, 60, 60, 60, 60, isQueued=0)

        print("[执行] 回零中…")
        r = dType.SetHOMECmd(api, isQueued=1)
        if r[0] != 0:
            print(f"✗ 回零指令入队失败 result={r[0]}")
            return 1

        # 等运动结束（回零比较久，给 120s）
        t0 = time.time()
        while time.time() - t0 < WAIT_TIMEOUT_S:
            if dType.GetQueuedCmdMotionFinish(api)[1]:
                break
            time.sleep(0.05)
        else:
            print(f"⚠ 等了 {WAIT_TIMEOUT_S}s 还没结束 —— 可能被卡住/碰到东西了，")
            print("  请立刻查看机械臂，必要时急停。")
        time.sleep(2.0)          # 等零位附近的微调做完

        # ── 回读判定：不能只看「指令发成功了」就报成功 ──
        p = dType.GetPose(api)
        j = (p[4], p[5], p[6], p[7])
        print(f"\n[回零结果] J1={j[0]:7.2f}° J2={j[1]:7.2f}° "
              f"J3={j[2]:7.2f}° J4={j[3]:7.2f}°")
        d = max(abs(a - b) for a, b in zip(j, ZERO_JOINTS))
        if d <= TOL_DEG:
            print(f"✓ 回零成功（与零位最大偏差 {d:.2f}° ≤ {TOL_DEG}°）"
                  f"，应有蜂鸣 + 绿灯")
            print("  下一步可以重新对刀: python3 src/step2_teach_coords.py"
                  " --reference corner_tr")
            return 0
        print(f"⚠ 偏差较大: {d:.2f}° (> {TOL_DEG}°) → 回零没真正到位。")
        print("  先跑 python3 tools/dobot_check.py 看报警和回零原点 HOMEParams。")
        return 1
    finally:
        try:
            dType.SetQueuedCmdStopExec(api)
        except Exception:
            pass
        try:
            dType.DisconnectDobot(api)
            print("\n[断开] 完成")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[已中止] ⚠ 回零指令可能仍在控制器里继续执行 —— "
              "请看着机械臂，必要时急停或在 DobotStudio 里停止。")
        sys.exit(130)
