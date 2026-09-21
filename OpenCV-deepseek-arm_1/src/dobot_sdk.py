#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dobot_sdk.py —— Dobot SDK 的**唯一**加载入口
=============================================================================
谁用: step2_teach_coords / step4_pick_test / tools/dobot_check / tools/home_arm

为什么要单独有这么一个模块（而不是各脚本各写一遍）:
  原先 SDK 路径是**写死的绝对路径**（/home/hanli/越疆机器人/...），别人拿到
  项目第一件事就得改它，改漏一处就跑不起来。现在:
    · SDK 随项目放在 sdk/dobot/，所以默认路径对谁都成立；
    · 三处重复的加载代码收成一处在改。

★ 为什么不用 SDK 自带的 DobotDll.load():
  它写死了相对路径 "./libDobotDll.so"，只在"当前工作目录 == SDK 目录"时才
  找得到库。本项目要从仓库根目录跑，所以按**绝对路径**自己 CDLL。
  （DobotDll.py 里的封装都把 api 当第一个参数收，所以照样能用。）

★ 新机器上最常见的失败 —— .so 加载不了:
  libDobotDll.so 动态链接 Qt5（libQt5SerialPort/Qt5Network/Qt5Core）。
  目标机器没装 Qt5，CDLL 会抛 "cannot open shared object file"。
  Ubuntu/Debian:  sudo apt install libqt5serialport5 libqt5network5 libqt5core5a
  另外它是 x86-64 专用二进制 → ARM（树莓派）、Mac、Windows 都用不了。
  下面 load_sdk() 会把这两种情况翻成人话。
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from paths import SDK_DIR

# 目标平台（写在这里是为了报错时能直接说清楚，而不是让人猜）
PLATFORM_NOTE = "x86-64 Linux + Dobot Magician + Qt5 运行库"

_QT_HINT = (
    "  这个 .so 动态链接了 Qt5（libQt5SerialPort / Qt5Network / Qt5Core）。\n"
    "  检查: ldd sdk/dobot/libDobotDll.so | grep 'not found'\n"
    "  安装: sudo apt install libqt5serialport5 libqt5network5 libqt5core5a"
)


def sdk_dir() -> Path:
    """
    SDK 目录。优先项目自带的 sdk/dobot/，其次环境变量 $DOBOT_SDK_DIR。

    环境变量那条是给"想用另一份 SDK 解压目录"的人留的出口，不填也没关系。
    """
    if (SDK_DIR / "libDobotDll.so").exists() and (SDK_DIR / "DobotDll.py").exists():
        return SDK_DIR
    env = os.environ.get("DOBOT_SDK_DIR")
    if env and (Path(env) / "libDobotDll.so").exists():
        return Path(env)
    raise SystemExit(
        f"✗ 找不到 Dobot SDK。\n"
        f"  期望: {SDK_DIR}/ 里有 DobotDll.py 和 libDobotDll.so\n"
        f"  （这两个文件应随项目一起交付；丢了就去越疆官网下 "
        f"'Dobot Demo V2.3-zh' 解压后拷进来）\n"
        f"  也可以用环境变量另指一份: export DOBOT_SDK_DIR=/path/to/run-linux"
    )


def load_sdk():
    """
    加载 SDK，返回 (api, dType)。

      api   —— CDLL 实例，所有 dType.Xxx(api, ...) 调用的第一个参数
      dType —— DobotDll 模块，封装了各指令的 ctypes 结构体

    只在本函数里做绝对路径加载；失败时给出可操作的报错，不留 traceback。
    """
    d = sdk_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    try:
        import DobotDll as dType
    except ImportError as e:
        raise SystemExit(f"✗ 导入 {d}/DobotDll.py 失败: {e}") from None

    so = d / "libDobotDll.so"
    try:
        api = ctypes.CDLL(str(so), ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise SystemExit(
            f"✗ 加载 {so} 失败: {e}\n{_QT_HINT}\n"
            f"  本项目的 SDK 只适用于: {PLATFORM_NOTE}"
        ) from None
    return api, dType


def find_port(dType, explicit: str | None = None) -> str | None:
    """
    找机械臂串口。**找不到返回 None**，不抛异常 —— 让调用方印友好提示。

    ★ 为什么要包一层: SDK 那份 DobotDll.find_port() 找不到设备时是 raise
      RuntimeError 的，调用方原本写的 `if not port:` 友好提示根本走不到，
      新机器上没插臂就只会看到一坨 traceback。这里统一成返回 None。
    """
    if explicit:
        return explicit
    try:
        return dType.find_port()
    except Exception:
        return None
