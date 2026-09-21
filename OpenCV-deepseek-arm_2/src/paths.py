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

# ★ 吸盘上的「第 5 个二维码」（tools/gen_suction_qr.py 生成）。
#   贴在末端法兰/吸盘金属块正上方，用来算出「吸嘴此刻在桌面哪儿」，
#   从而不必依赖机械臂基座的绝对精度（见《WPS文字文档.wps》一、二节）。
#   JSON 里记的是它的**实际物理尺寸**（数据区 mm），后面做视差校正要用到。
SUCTION_QR_JSON = DATA_DIR / "suction_qr.json"

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
#   ★★ 这就是 **XY 补偿那一档的闸门**: 只有 src == "camera" 的那一次抓取才补
#     （见 main.GRASP_OFFSET_MM 的注释）。为什么必须卡这一下: 那几毫米补的是
#     **相机坐标**的系统偏差，机械臂自己确认过的落点**没有**这个偏差 ——
#     再补一次等于把它推离真实位置 5mm，现象是"抓过之后再抓就抓不住"。
#     用户 2026-09-26 实测后让撤销"每次都补"的那一版，原话:「靠近P4的，除了一开始
#     需要修改一下坐标位置，之后每次不需要修改了。只改动第一次的即可」。
#   ★ 高度补偿（GRASP_OFFSET_Z_MM）**不看它** —— 那一档按"此刻在哪个角附近"判，
#     搬到 P1 的也算，见下面的 WORLD_CODES_KEY。
#   ★ 为什么放在 paths.py 而不是 color_vision.py: 写的一方是 color_vision（相机）、
#     读的一方是 main（规划/执行），而 main **不能** import color_vision（那会把
#     cv2 拖进纯算术的规划链路，见 main.persist_world_state 的说明）。
#     paths.py 是两边都 import 且不带任何重依赖的地方 —— 这个字符串只能放这儿，
#     不然就得抄两份（抄两份的第一个后果就是改了一处忘了另一处）。
WORLD_SRC_KEY = "src"
WORLD_SRC_CAMERA = "camera"

# ★ 记忆库里每条记录的**最近角码**标记（键名 + 四个值）。
#   "near": "P1" = 这块方块离纸面上**那个角码**最近（P2/P3/P4 同理）；
#   没有这个键 = 判不出来。
#   为什么需要它: 用户实测偏差**四个角各不一样**（见 main.GRASP_OFFSET_MM
#   的注释）—— 按「纸的左/右半边」两半地补，等于把两个角各自的偏差平均掉，
#   补完中间那两处仍然偏。「离哪个角最近」必须在**纸自己的坐标系**里判
#   （比纸面 mm 距离），不能拿机械臂坐标去猜 —— 纸一挪、机位一变，机械臂坐标
#   跟着变，纸面的四个角却是钉死的。
#   ★★ 它只用来**选 XY 补偿表里那个角**（P1→哪几个 mm），本身不进入任何几何。
#     高度那一档要的是"**此刻**离哪个角最近"，跟它**不是一回事** ——
#     那个必须拿 WORLD_CODES_KEY 里四个码的位置现算（两者可能是两个角）。
#   ★ 判不出来的情况（少二维码，拟合不出像素→纸面）：**不写这个键**，
#     规划那边 XY 那一档也就不补偿 —— 宁可少补这几毫米，也不赌错方向白偏。
#   ★★ 生命周期: 它和 src 戳一样是**相机那一眼的属性**，机械臂碰过这块方块之后
#     refresh_world_state 把整条记录重写成 {x, y, z_level} —— near 和 src 一起没。
#     （XY 那一档本来就只在相机刚给的那一次抓取补，所以"没了"正合语义。）
#   ★ 和 WORLD_SRC_KEY 同样的理由放在 paths.py: 写的一方是 color_vision，
#     读的一方是 main，而 main 不能 import color_vision（cv2）。
WORLD_NEAR_KEY = "near"

# ★ 四个角码的**名字**，必须和 data/calib_A4_qr.json 里 qr 的键一字不差。
#   为什么抄成常量: 读的一方是 main（纯算术，不能 import qr_vision → cv2），
#   main 那张补偿表只能拿字面量当键。写的一方 color_vision 有一条自检钉住
#   「这四个名字 == json 里那四个」—— 重新生成标定纸换了码名，会在那儿炸，
#   而不是让补偿**悄没声地失效**（查不到键 → 不补 → 每次都偏 5mm，还看不出来）。
WORLD_NEAR_P1 = "P1"
WORLD_NEAR_P2 = "P2"
WORLD_NEAR_P3 = "P3"
WORLD_NEAR_P4 = "P4"
WORLD_NEAR_CODES = (WORLD_NEAR_P1, WORLD_NEAR_P2, WORLD_NEAR_P3, WORLD_NEAR_P4)

# ★ 记忆库里那张「四个角码**此刻**在机械臂坐标系的哪儿」的**保留键**
#   （值 = {"P1": [x, y], ...}，机械臂 mm）。
#   为什么需要它: 方块被机械臂搬走之后，主程序要能回答「它**现在**离哪个角码最近」。
#   用户 2026-09-26 的原话: 「所有在P1附近的物体，不管是一开始[在]还是后来被移过去的，
#   Z 都减少 3mm」。而方块那条记录里的 near 戳是**相机那一眼**给的、
#   机械臂一碰就被 refresh_world_state 抹掉，根本回答不了"现在在哪" ——
#   所以必须另有一份**码的位置**现算。主程序跑不了视觉（cv2），只能由写的一方
#   把这一帧量到的码坐标存下来。
#   ★ 判不出/不齐时（少二维码）→ main.nearest_code_of_xy 返回 None，高度那一档
#     退回按记录里的 near 判；两个都没有就不补。
#   ★ 为什么存进 world_state.json 而不是另开一个文件: 它和四条方块记录是**同一帧**的
#     产物，分开存迟早一份新一份旧；而且它跟着记忆库一起落盘/备份（save_world_state），
#     不用再造一套读写。
#   ★★ 键名以 "_" 开头是一条**约定**: 它**不是一条方块记录**，凡是要遍历记忆库、
#     把每条都当"方块"用的地方（path_blockers / deepseek_brain 发 prompt / 自检）
#     都必须跳过带下划线的键 —— 照搬会 KeyError（它里面没有 "x"）或者把码名
#     当成方块名发给模型。执行这条约定的地方只有三处: main.cube_records、
#     deepseek_brain.build_messages、color_vision 的自检；新增遍历时记得照办。
WORLD_CODES_KEY = "_codes"

# ★ 「机械臂走 mm ↔ 画面动像素」的实测映射（tools/measure_suction_map.py 落盘）。
#   这是**相对运动**（动态基座定位）那条路的地基: 不让机械臂去够某个绝对坐标，
#   只看吸盘码在画面里怎么动，反解出该走多少。
#   ★ 为什么不放 data/: 它随「相机架在哪、吸盘码贴多高」而变，是一台机器一次
#     测量的产物，和 hand_eye_matrix.json 同性 —— 换机位就得重量，别当输入交付。
SUCTION_MAP_JSON = OUTPUT_DIR / "suction_map.json"

# ★ ④ tools/pick_suction.py（相对运动路线上的真抓取）落盘。
#   ★ 为什么不写进 pick_test_result.json: 那个文件的 table_z 是 load_table_z() 的
#     来源之一，写它就有可能把「纸面 Z」这条地基覆盖掉。而且两个工具记的是两回事 ——
#     step4 记「哪套纸面→机械臂映射赢了」，④ 记「相对路线这一次抓在哪、放哪」。
#     宁可多一个文件，也不让两条路的落盘互相踩。
PICK_SUCTION_JSON = OUTPUT_DIR / "pick_suction_result.json"

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
