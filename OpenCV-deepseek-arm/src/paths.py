#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paths.py —— 全项目路径集中定义（**唯一一处**）

为什么要有这个文件:
  以前每个脚本各自写 `HERE = Path(__file__).resolve().parent`，再拼
  `HERE / "calib_A4_qr.json"`。脚本一挪目录，七个文件全得改。
  现在这里算一次，别处只 import。

目录约定:
    <root>/data/    标定纸定义与图 —— 生成一次后当**输入**用，要跟着项目走
    <root>/output/  运行产物 —— 每次跑出来，交付时清空
    <root>/docs/    文档
    <root>/src/     标定流水线 step1~step4 + 公共库 qr_vision
    <root>/tools/   辅助工具（拍图、回零、查 SDK、生成图例）
    <root>/sdk/     Dobot SDK 随项目自带 —— 这样别人拿到就能跑，不用改路径

★ output/ 不存在时自动建，调用方不用管。
"""
from __future__ import annotations

from pathlib import Path

# ─────────────────────────── 目录 ───────────────────────────
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
TOOLS_DIR = ROOT / "tools"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DOCS_DIR = ROOT / "docs"

# ──────────────── 机械臂 SDK（sdk/，随项目交付） ────────────────
# ★ 为什么把 SDK 放进仓库: 原先 SDK_DIR 写死成 /home/hanli/... 的绝对路径，
#   别人拿到项目第一件事就是改这个路径。现在 SDK 跟着项目走，开箱即用。
#   加载逻辑在 src/dobot_sdk.py（优先这里，其次 $DOBOT_SDK_DIR）。
SDK_DIR = ROOT / "sdk" / "dobot"

# ──────────────── 输入: 标定纸（data/，要交付） ────────────────
# ★ 这两个是**所有脚本的地基**: 纸面几何(mm)。纸没换、底座没换就不用重新生成。
PAPER_JSON = DATA_DIR / "calib_A4_qr.json"      # 纸面几何定义（step1 生成，step2/3/4 读）
PAPER_PNG = DATA_DIR / "calib_A4_qr.png"        # 打印用图（step1 生成）

# ──────────── 运行产物（output/，每次跑出来，不交付） ────────────
ROBOT_POINTS_JSON = OUTPUT_DIR / "robot_points.json"       # step2 落盘
MATRIX_JSON = OUTPUT_DIR / "hand_eye_matrix.json"          # step3 落盘
PICK_RESULT_JSON = OUTPUT_DIR / "pick_test_result.json"    # step4 落盘（纸面 Z）
# ★ main.py 落盘: 「看过的那份计划」= 「要执行的那份计划」。
#   为什么必须存盘而不是执行时重新问一遍 DeepSeek: 模型是推理模型、有随机性，
#   重问可能得到不同的坐标 —— 那「人工过目」就白过了。--go 默认读这个文件。
PLAN_JSON = OUTPUT_DIR / "last_plan.json"
# ★ 桌面「记忆库」（方案第五阶段第 3 条「刷新记忆」）: 方块现在在哪、在第几层。
#   为什么必须落盘而不是每次重新识别: 相机只能看到 XY，**看不出摞了几层**。
#   z_level 只能靠「这条指令执行完之后 +1」一路记下来，断了就全错。
WORLD_STATE_JSON = OUTPUT_DIR / "world_state.json"

# ★ 记忆库里每条记录的**来源**标记（键名 + 相机那个值）。
#   "src": "camera" = 这条 x/y 是相机刚拍的、机械臂**还没碰过**这块方块；
#   没有这个键 = 机械臂确认过的（refresh_world_state 重写时自然就没了），按真实值走。
#   为什么需要它: 实测「相机拍完的第一次抓取」吸盘总会偏左一点，第一次抓要补偿
#   5mm，之后就该按真实值 —— 「第一次」只能靠"这条坐标是谁给的"判断。
#   ★ 为什么放在 paths.py 而不是 color_vision.py: 写的一方是 color_vision（相机）、
#     读的一方是 main（规划/执行），而 main **不能** import color_vision（那会把
#     cv2 拖进纯算术的规划链路，见 main.persist_world_state 的说明）。
#     paths.py 是两边都 import 且不带任何重依赖的地方 —— 这个字符串只能放这儿，
#     不然就得抄两份（抄两份的第一个后果就是改了一处忘了另一处）。
WORLD_SRC_KEY = "src"
WORLD_SRC_CAMERA = "camera"

# ★ 记忆库里每条记录的**半侧**标记（键名 + 两个值）。
#   "half": "left"  = 这块方块落在**纸的左半边**（P1/P3 那一侧）；
#   "half": "right" = 纸的右半边（P2/P4 那一侧）；没有这个键 = 判不出来。
#   为什么需要它: 用户要求那 5mm 的第一抓补偿**只对左半边的方块**做
#   （2026-09-18 实测：右半边的第一抓本来就准）。「哪半边」必须在**纸自己的
#   坐标系**里判（paper x < paper_mm[0]/2），不能拿机械臂坐标去猜 —— 纸一挪、
#   机位一变，机械臂坐标跟着变，纸面的左右却是钉死的。
#   ★ 判不出来的情况（少二维码，拟合不出像素→纸面）：**不写这个键**，
#     规划那边也就不补偿 —— 宁可少补 5mm，也不赌错方向白偏。
#   ★ 和 WORLD_SRC_KEY 同样的理由放在 paths.py: 写的一方是 color_vision，
#     读的一方是 main，而 main 不能 import color_vision（cv2）。
WORLD_HALF_KEY = "half"
WORLD_HALF_LEFT = "left"
WORLD_HALF_RIGHT = "right"

COLOR_DEBUG_PNG = OUTPUT_DIR / "color_debug.png"           # color_vision.py 的调试图
SHOTS_DIR = OUTPUT_DIR / "shots"                           # tools/test_camera_qr.py 存图
REF_GUIDE_PNG = OUTPUT_DIR / "ref_point_guide.png"         # step1 出的参考角图例
MEASURE_GUIDE_PNG = OUTPUT_DIR / "measure_guide.png"       # tools/measure_guide.py 出的量尺图


def ensure_output_dir() -> Path:
    """确保 output/ 存在（写文件前调一次）。返回该目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def ensure_data_dir() -> Path:
    """确保 data/ 存在（step1 写纸面定义时调一次）。返回该目录。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
