#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_env.py —— 环境自检：一条命令告诉你"还缺什么、怎么装"
=============================================================================
谁该跑: **刚拿到这个项目的人**，在动机械臂之前先跑这个。
        「我这边能跑」证明不了「他那台能跑」—— 缺的东西里有两样（Qt5、串口
        权限）不在文件夹里，而且报错信息极易被误判成别的毛病。

★ 全程只读、不连机械臂、不发任何指令。没插硬件也能跑，那几项会标成「跳过」。

级别:
  ✅ 有           没问题
  ⚠ 建议         不影响主流程，但会影响体验（如缺中文字体，图例会变方块）
  ❌ 缺           必须解决，后面跑不动
  ⊘ 跳过         需要硬件/等他做，现在验不了
  ─ 待办         不是环境问题，是流程还没走到（如还没对刀）

退出码: 有 ❌ 就返回 1，否则 0 —— 可以直接拿去做脚本/CI 的闸门。

用法:
  python3 tools/check_env.py
"""

from __future__ import annotations

import argparse
import glob
import grp
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))            # 公共库在 src/

OK, WARN, FAIL, SKIP, TODO = "✅", "⚠ ", "❌", "⊘ ", "─ "

# 逐项结果：(级别, 标题, 说明行列表)
RESULTS: list[tuple[str, str, list[str]]] = []


def add(level: str, title: str, *lines: str) -> None:
    RESULTS.append((level, title, [ln for ln in lines if ln]))


# ─────────────────────────── 1. Python ───────────────────────────
def check_python() -> None:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 8):
        add(OK, f"Python {ver}", f"解释器: {sys.executable}")
    else:
        add(FAIL, f"Python {ver} 太旧",
            "本项目用到 f-string、`from __future__ import annotations` 等 3.7+ 特性。",
            "建议装 Python 3.8 以上（本机验证过 3.12）。")


# ────────────────────── 2. Python 第三方包 ──────────────────────
def check_pip_packages() -> None:
    missing = []
    for mod, pipname, verattr in (("cv2", "opencv-python", "__version__"),
                                  ("numpy", "numpy", "__version__"),
                                  ("PIL", "pillow", "__version__")):
        try:
            m = __import__(mod)
            add(OK, f"{pipname} {getattr(m, verattr, '?')}")
        except ImportError:
            missing.append(pipname)
            add(FAIL, f"缺 {pipname}（import {mod} 失败）")
    if missing:
        add(FAIL, "补装 Python 包", "pip install -r requirements.txt",
            "或: pip install " + " ".join(missing))


# ─────────── 3. SDK 加载（顺便就把 Qt5 一起验了）───────────
def _ldd_missing(so: Path) -> list[str]:
    """用 ldd 找出没解析到的动态库名。ldd 不存在就返回空。"""
    if not shutil.which("ldd"):
        return []
    try:
        out = subprocess.run(["ldd", str(so)], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return []
    return [ln.split("=>")[0].strip()
            for ln in out.splitlines() if "not found" in ln]


def check_sdk() -> None:
    from paths import SDK_DIR
    dll, py = SDK_DIR / "libDobotDll.so", SDK_DIR / "DobotDll.py"
    if not (dll.exists() and py.exists()):
        add(FAIL, "缺 SDK 文件",
            f"期望 {SDK_DIR}/ 里有 DobotDll.py 和 libDobotDll.so",
            "这两个本该随项目一起给你。丢了就去越疆官网下 "
            "'Dobot Demo V2.3-zh'，把 run-linux/ 里的这两个文件拷进来。",
            "或用环境变量另指一份: export DOBOT_SDK_DIR=/path/to/run-linux")
        return

    # 真的加载一次 —— 这是 Qt5 是否齐全的**唯一可靠判据**。
    from dobot_sdk import load_sdk
    try:
        load_sdk()
        add(OK, "Dobot SDK 能加载",
            f"{SDK_DIR}/  (含 libDobotDll.so {dll.stat().st_size // 1024}K)")
    except (SystemExit, Exception) as e:
        msg = str(e) if str(e).strip() else repr(e)
        miss = _ldd_missing(dll)
        lines = [f"加载 {dll} 失败。", *msg.splitlines()]
        if miss:
            lines.append("ldd 说这些库没解析到: " + ", ".join(miss))
            lines.append("→ 这就是缺 Qt5。装:")
            lines.append("   sudo apt install libqt5serialport5 "
                         "libqt5network5 libqt5core5a")
        else:
            lines.append("排查: ldd sdk/dobot/libDobotDll.so | grep 'not found'")
        add(FAIL, "SDK 加载失败", *lines)


# ───────────────────── 4. 串口 / dialout 组 ─────────────────────
def check_serial() -> None:
    devs = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    by_id = "/dev/serial/by-id"
    has_byid = os.path.isdir(by_id) and os.listdir(by_id) if os.path.isdir(by_id) else False

    # 组权限：新机器上最常见的"连接失败"根因，且报错很不像权限问题
    groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
    in_dialout = "dialout" in groups

    if devs:
        readable = all(os.access(d, os.R_OK | os.W_OK) for d in devs)
        add(OK if readable else FAIL,
            f"串口设备: {', '.join(devs)}",
            "" if readable else "设备在，但当前用户没有读写权限 → 下面这条必做。")
        if has_byid:
            add(OK, f"稳定软链: {by_id}/ 有内容（脚本优先用它，插拔不乱）")
    else:
        add(SKIP, "串口：没插机械臂",
            "插上并上电后重跑本脚本。机械臂是 CP210x 芯片，会出现在 "
            "/dev/ttyUSB0。")

    if in_dialout:
        add(OK, "当前用户在 dialout 组（串口权限没问题）")
    else:
        add(FAIL, "当前用户不在 dialout 组",
            "Linux 上 /dev/ttyUSB0 属于 root:dialout，不在组里会 "
            "'连接失败 state=1'，看着像没插好。",
            "sudo usermod -aG dialout $USER",
            "★ 加完必须**注销重新登录**（或重启）才生效。")


# ───────────────────────── 5. 摄像头 ─────────────────────────
def check_camera() -> None:
    from qr_vision import find_camera, list_cameras
    try:
        cams = list_cameras()
    except Exception as e:
        add(WARN, f"摄像头枚举失败: {e}")
        return
    if not cams:
        add(FAIL, "没找到摄像头",
            "一个 /dev/video* 都没有。检查 USB 连接；",
            "tools/qr_camera.py --list 看详细信息。")
        return

    add(OK, f"找到 {len(cams)} 个视频节点")
    for c in cams:
        tag = "   [metadata，不是摄像头]" if c["metadata"] else ""
        add(OK, f"  · /dev/video{c['index']}  {c['name']}{tag}")

    # ★ 光"有摄像头"不够 —— 脚本会自动挑一个，挑到笔记本内置的就有问题了:
    #   内置摄像头一般对着人，不是俯拍桌面，标定必然失败。要提前说清楚。
    pick = find_camera()
    if pick is None:
        add(FAIL, "没有可用作摄像头打开的节点")
        return
    is_external = not any(x in pick["name"].lower()
                          for x in ("integrated", "ir camera", "infrared"))
    if is_external:
        add(OK, f"脚本会自动选: /dev/video{pick['index']}  {pick['name']}")
        add(OK, "  （外接摄像头 ✓）")
    else:
        add(FAIL, "只能选到笔记本内置摄像头",
            f"自动选中的是 /dev/video{pick['index']}  {pick['name']}",
            "流水线要的是**俯拍桌面**的外接摄像头（一声一视 X6L / DCX-5MAF 等）。",
            "笔记本内置对着人是拍不到标定纸的 —— 插上外接 USB 摄像头再重跑本脚本。",
            "（工具仍会退回内置，好让你能先跑 tools/qr_camera.py 看画面。）")


# ──────────────────── 6. 中文字体（画图例用）────────────────────
def check_font() -> None:
    # tools/measure_guide.py 要用 CJK 字体画中文标注；缺了会渲染成方块
    if any(Path(p).exists() for p in (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")):
        add(OK, "中文字体（画图例要用）")
    else:
        add(WARN, "缺中文字体",
            "不影响流水线，但 tools/measure_guide.py 出的图例中文会变方块。",
            "sudo apt install fonts-noto-cjk")


# ────────────────── 7. 标定纸（输入数据）──────────────────
def check_paper() -> None:
    import json
    from paths import PAPER_JSON, PAPER_PNG
    if not PAPER_JSON.exists():
        add(FAIL, f"缺 {PAPER_JSON.relative_to(ROOT)}",
            "这是整个项目的地基（纸面几何）。没有它所有脚本都跑不了。",
            "确认交接时没漏掉 data/ 目录。")
        return
    try:
        from qr_vision import DATA_MODULES, MODULES_TOTAL, load_paper_layout
        layout = load_paper_layout(PAPER_JSON)
        meta = json.loads(PAPER_JSON.read_text(encoding="utf-8"))["meta"]
        # ★ layout.qr_side_mm 存的是**含白边**的边长（29 模块），不是数据区。
        #   数据区（黑色图案，21 模块）= 它的 21/29 —— 两者差 8 个模块，
        #   是本项目最容易混的一对尺寸（见 README「参考点约定」）。
        data_side = layout.qr_side_mm * DATA_MODULES / MODULES_TOTAL
        add(OK, "标定纸定义可读",
            f"{PAPER_JSON.relative_to(ROOT)}  source={meta.get('source', '?')}",
            f"{len(layout.codes)} 个码  参考点={layout.reference}  "
            f"数据区 {data_side:.1f}mm（{DATA_MODULES} 模块）"
            f" / 含白边 {layout.qr_side_mm:.1f}mm（{layout.modules} 模块）")
    except Exception as e:
        add(FAIL, f"标定纸定义读不了: {e}")
        return

    # ★ PNG 和 json 不一定配套：json 是 step1 排出来的，PNG 才是「印这张纸」；
    #   json 来自实测（measured-sheet）时，PNG 只是早期模板 —— 照着它印出来的纸
    #   和 json 的毫米数**对不上**。这两种情况必须说不同的话，否则会把人引到
    #   「重印一张纸」那条错路上（README「先看清 json 和 PNG 是不是同一张纸」）。
    has_png = PAPER_PNG.exists()
    if meta.get("source") == "measured-sheet":
        note = (f"（{PAPER_PNG.relative_to(ROOT)} 只是早期模板，**别拿它去印** —— "
                f"印出来的纸和 json 对不上）" if has_png else "")
        add(OK, "标定纸来源: 按卷尺实测重建（measure_layout_from_sheet）", note,
            "→ 用**你手上那张纸**，它和 json 里的毫米数是配套的。",
            "换纸才需要重量: tools/measure_guide.py → make_layout_from_sheet.py")
    else:
        note = (f"（{PAPER_PNG.relative_to(ROOT)} — 打印要 100% 比例，"
                f"不能选「适应页面」）" if has_png else "")
        add(OK, "标定纸来源: 由 step1 排版生成", note)


# ─────────────── 8. DeepSeek API key（⑤ 决策层要用）───────────────
# ★ 为什么是 ⚠ 不是 ❌: step1~step4（标定 + 抓取测试）**完全不需要联网**，
#   一个刚拿到项目的人很可能还没配 key 就要先对刀。要是这里报 ❌，
#   check_env 的退出码会变成 1，把「其实可以开始对刀了」误判成「环境没装好」。
def check_deepseek() -> None:
    env = "DEEPSEEK_API_KEY"
    key = os.environ.get(env, "").strip()

    if not key:
        add(WARN, f"没设 {env}（只有主程序 main.py 需要它）",
            "step1~step4 标定/测试流程不需要联网，可以先不管这一条。",
            f"要跑 main.py 就得设: export {env}='sk-你的key'",
            "★ 把它写进 ~/.bashrc 才会一直有效（只 export 一次，重开终端就没了）。",
            "★ 别把它写进项目里任何文件 —— 这个项目是要交付给别人的，"
            "对方要用**自己的** key。")
        return

    if not key.startswith("sk-"):
        add(WARN, f"{env} 看着不像 DeepSeek 的 key",
            "正常以 'sk-' 开头。（不打印内容，只提醒格式。）",
            "别在这里贴真 key —— 本脚本不读文件、不回显。")
    else:
        # ★ 只报长度，**绝不回显内容** —— 自检输出经常被人整段贴到聊天里求助。
        add(OK, f"{env} 已设置（长度 {len(key)}，以 sk- 开头）",
            "内容不回显（免得被贴出去）。")

    # 只做 TCP 连通性探测，**不发任何 API 请求** —— 不花钱、不算「发出指令」，
    # 但能提前抓到「防火墙/代理挡住了」这个很常见、报错又很难懂的失败。
    import socket
    try:
        with socket.create_connection(("api.deepseek.com", 443), timeout=4):
            add(OK, "能连上 api.deepseek.com:443（只做了 TCP 握手，没调接口）")
    except OSError as e:
        add(WARN, "连不上 api.deepseek.com:443",
            f"{type(e).__name__}: {e}",
            "标定流程不受影响；只有 main.py 会失败。",
            "常见原因: 需要代理、公司网络拦截、或 DNS 不通。")


# ───────────────── 9. 进度：该做什么了（非环境问题）─────────────────
def check_progress() -> None:
    from paths import MATRIX_JSON, PICK_RESULT_JSON, PLAN_JSON, ROBOT_POINTS_JSON
    if ROBOT_POINTS_JSON.exists():
        add(OK, f"已对刀: {ROBOT_POINTS_JSON.relative_to(ROOT)}")
    else:
        add(TODO, "还没对刀（这步得人工做）",
            "下一步: python3 src/step2_teach_coords.py --reference corner_tr",
            "★ 交互式：把吸盘中心逐个对准四个码的参考点，**按 q 存盘**"
            "（Ctrl-C 退出不存）。")
    if MATRIX_JSON.exists():
        add(OK, f"已标定: {MATRIX_JSON.relative_to(ROOT)}")
    else:
        add(TODO, "还没标定",
            "对刀完成后: python3 src/step3_hand_eye_calib.py")

    # ★ 主程序 main.py 的两块前置: 没『纸面 Z』它就算不出该降到哪；
    #   没『计划文件』说明还没跑过（--go 默认执行存下的计划，所以必须先跑一次不带 --go 的）。
    if PICK_RESULT_JSON.exists():
        add(OK, f"已知纸面 Z: {PICK_RESULT_JSON.relative_to(ROOT)}")
    else:
        add(TODO, "还不知道纸面 Z（main.py 动臂前必须有）",
            "量一次: python3 src/step4_pick_test.py --probe",
            "★ 换过底座/桌面就得重新量 —— 高度变了，纸面 Z 跟着变。",
            "临时用一次也行: python3 src/main.py \"指令\" --table-z -33.8")
    if PLAN_JSON.exists():
        add(OK, f"存有上次的抓放计划: {PLAN_JSON.relative_to(ROOT)}")
        add(TODO, "  ★ 真动臂前先看一遍这份计划（--go 默认执行的就是它）",
            "python3 src/main.py \"你上次那句指令\"   # 不带 --go，只打印计划")
    else:
        add(TODO, "还没跑过主程序（先只出计划、不动臂）",
            "python3 src/main.py \"把绿色方块放到红色方块右侧\"",
            "★ 不带 --go 时 DeepSeek 会算，但机械臂一步都不动。")


# ─────────────────────────── 主流程 ───────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="环境自检（只读；不连机械臂、不发指令）",
        epilog="无参数 = 跑全部检查。退出码 1 = 有必须解决的项（可当脚本闸门用）。")
    ap.parse_args()

    print("=" * 72)
    print("  环境自检（只读；不连机械臂、不发指令）")
    print("=" * 72)

    for fn in (check_python, check_pip_packages, check_sdk, check_serial,
               check_camera, check_font, check_paper, check_deepseek,
               check_progress):
        try:
            fn()
        except Exception as e:                      # 单项崩了不该拖垮整个自检
            add(FAIL, f"{fn.__name__} 自身出错: {e!r}")

    for level, title, lines in RESULTS:
        print(f"\n{level} {title}")
        for ln in lines:
            print(f"     {ln}")

    n_fail = sum(1 for lv, _, _ in RESULTS if lv == FAIL)
    n_warn = sum(1 for lv, _, _ in RESULTS if lv == WARN)
    print("\n" + "=" * 72)
    if n_fail:
        print(f"  ❌ 有 {n_fail} 项必须解决（⚠ {n_warn} 项建议）—— 按上面的命令装")
        print("     全绿之前先别接机械臂跑 step2/step4。")
    else:
        print(f"  ✅ 环境齐了（{n_warn} 项建议可忽略）")
        print("     下一步: python3 tools/dobot_check.py   # 只读体检，不动")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
