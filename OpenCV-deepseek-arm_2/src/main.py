#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py —— 主程序（⑥ 执行）：一句中文 → DeepSeek 决策 → 机械臂抓放
=============================================================================
整条链路的最后一环，把前面几步串起来:

    world_state（OpenCV 算出的方块机械臂坐标）
        + 一句中文
        ↓  src/deepseek_brain.py（⑤ 决策，不碰硬件）
    [{"obj","grasp":{x,y,z_level},"place":{x,y,z_level}}, ...]
        ↓  本文件
    机械臂真的动起来

★★ 本文件**不做决策，只做翻译和执行**。
   「该抓哪、该放哪」全部由 DeepSeek 给，这里只负责把 x/y/z_level 变成
   吸盘该走的坐标和该做的动作。决策错了是 brain 的事，执行错了才是这里的事。

──────────────────────────── 两个必须知道的前提 ────────────────────────────

★ 1. world_state 里的坐标是**机械臂坐标（mm）**，不是像素。
     OpenCV 那半边负责「像素 → 机械臂坐标」（那才是手眼标定的用处），
     本文件拿到的是已经换算好的结果。所以主程序**根本不需要**
     hand_eye_matrix.json —— 它只要方块在机械臂坐标系里的 x/y。

     OpenCV 那半边就是 src/color_vision.py，它把结果写成同形 JSON
     （output/world_state.json）。本文件默认读那个文件。

★ 2. 必须知道纸面 Z(table_z)，否则**一步都不动**。
     z_level 只是「第几层」，真正要走的 Z = 纸面 Z + 方块高度 × 层数。
     纸面 Z 由 `step4_pick_test.py --probe` 触底量出来（写进
     output/pick_test_result.json），本文件从那儿读。
     没有它就只能出计划、不能动臂 —— 这是有意的，不是缺陷。

★ 3. 桌面「记忆库」output/world_state.json 是**有状态的**，这一点最反直觉。
     相机能给的是 x/y；**z_level（摞在第几层）相机永远看不出来** ——
     俯拍图上摞在第二层的方块和躺在地上的长得一模一样。
     所以 z_level 只能「这条指令执行完之后改一次」一路记下来（《方案.md》
     第五阶段的「刷新记忆库」）。执行完由 refresh_world_state 就地更新、
     再落盘；这份记忆一断，后面所有叠放指令的高度就全错，而且**不报错**。
     你手工重摆过方块，就必须 `--reset-world`，否则记忆和桌面对不上。

────────────────────────── 两种跑法，自己挑一种 ──────────────────────────

  ① 一句话直接干（默认就按这个用）:
       python3 src/main.py "把绿色方块放到红色方块右侧" --go
     现问 DeepSeek → 打印计划 → 按一次回车 → 动臂。**一条命令，一次回车。**

  ② 想先目测一遍再动:
       python3 src/main.py "把绿色方块放到红色方块右侧"      # 只打印，不动臂
       python3 src/main.py "..." --go                        # 同一句话 → 跑刚看过的那份
     第 2 条能跑的就是第 1 条存下的那份计划（模型有随机性，不重问才叫"过目过"）。

   ★ 复用上次计划**只在「同一句话 + 桌面没变过」时发生**（见 main 里的 use_saved）。
     换了一句话、或上次执行已经把方块挪走了 → 一律重新问 DeepSeek。
     这条是踩出来的: 曾经不管理由一律复用，于是 `"把X放Y右边" --go` 跑的是
     上一条指令的计划 —— 看着像"代码改了没生效"，实际跑错了东西。

────────────────────────── 安全上的几条硬规矩 ──────────────────────────
  · 丢步报警(0x50~0x5F)一律拒绝运动 —— 零点已不可信，没有任何后门。
  · Z 下限锁在「抓取时最低的那一点」，吸盘不会扎进方块（扎进去 → 丢步 → 零点漂移）。
  · 会动臂就必须有人在终端前（tty），因为 DeepSeek 给的坐标可能是错的。
  · 退出时**一定关真空泵**（finally 里兜底），不会吸着方块卡在那儿。
  · 软限位在**连臂之前**就查一遍，不白跑一趟。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import deepseek_brain as brain          # noqa: E402  ⑤ 决策
import step2_teach_coords as tc         # noqa: E402  运动/限位/报警那一套
import step4_pick_test as s4            # noqa: E402  吸盘/cube_top_z/可达性
from paths import (PLAN_JSON, ROOT, WORLD_CODES_KEY,   # noqa: E402
                   WORLD_NEAR_CODES, WORLD_NEAR_KEY, WORLD_NEAR_P1,
                   WORLD_NEAR_P2, WORLD_NEAR_P3, WORLD_NEAR_P4,
                   WORLD_SRC_CAMERA, WORLD_SRC_KEY, WORLD_STATE_JSON,
                   ensure_output_dir)

# ─────────────────────────── 默认参数 ───────────────────────────
# 数值尽量对齐 step4_pick_test.py（那边真机跑通过），别在这里另立一套。
# ★ 唯一一处**故意不一致**: 搬运高度。step4 是单块验证工具，用的是
#   「抓/放顶面 + hover」，看不见桌上摞了几层；本文件会看**搬运路径上**有没有
#   方块要越过（见 path_blockers / DEFAULT_CARRY_CLEAR）。真要拿 step4 吸着
#   方块从别的方块上方掠过，它照样会蹭 —— 别用它跑这种场景。
DEFAULT_OBJ_H = 26.0     # 吸盘接触面到「已记纸面 Z」的距离（见 cube_top_z 的注释）

# ★ 方块**净高**（不含 1mm 吸盘唇口压缩量）的默认值，mm。
#   为什么写死 25 而不是留空: 用户 2026-09-19 明确要求「别让我每次都敲 --net-h 25」。
#   它只在两种时候被用到: ① 要叠放（有 z_level>0）；② 搬运路径会**经过**摞着的方块。
#   平放（第 0 层）压根不碰这个数 —— 见 nozzle_z 里那条单独的分支。
#   ★ 代价说清楚: 这个 25 是个**假设**（假设方块真实净高就是 25mm）。拿卡尺量出来
#     不是 25，就得显式 --net-h <实测值>，否则每叠一层就差 (net_h − 真实净高)。
#     库函数本身仍**拒绝**在没给 net_h 时算叠放（见 nozzle_z / cube_top_z），
#     所以这个默认值只影响命令行，不影响 t_need_net_h 那两条守门自检。
DEFAULT_NET_H = 25.0

DEFAULT_HOVER = 30.0     # 悬停高度下限: 抓取点/放置点各自再往上多少（见 plan_action）
DEFAULT_PUMP_S = 0.6     # 开真空后等多久再抬（抽气需要时间）
RELEASE_S = 0.4          # 关真空后等多久再走（放料需要时间）
DEFAULT_SPEED = 40.0
DEFAULT_ACC = 40.0
DEFAULT_RATIO = 30.0

# ★ 搬运(吸着方块平移)时，被吸那块方块的**底面**至少要高出
#   「要越过的那个顶面」这么多。
#   为什么不能沿用「抓取 Z + 固定 hover」那套: 那个高度跟桌上摆了几摞毫无关系 ——
#   实测纸面 Z=-18.49 时，底面只到 Z=12.51，即离纸面 (obj_h+hover−净高)=31mm：
#     · 旁边是**平放**的方块（顶面 = 纸面+26）→ 只剩 5mm，扫过去就蹭到（用户报的那次）
#     · 旁边摞到第 1 层（顶面 = 纸面+50）→ 底面比它还低 19mm，必撞
#   净空写死 35 而不是"跟着 hover 走"，就是要让这个间距跟方块多高**无关地**成立。
#   方块越高、吸盘抬得越高 —— 见 path_blockers()。
DEFAULT_CARRY_CLEAR = 35.0

# ★ 「要越过去的那块是**平放**的（第 0 层）」时净空只要这么多（mm）。
#
#   为什么原来一刀切 35 会出事（2026-09-19 用户实测）: 桌上只要有**一块平放的**
#   方块横在路径上，搬运高度就被顶到「平放顶面 + 35 + 25」。纸面 Z=-18.49 时那是
#   Z=67.51 —— 而这条路最后要走到离底座只有 117mm 的降低点，那个半径上机械臂
#   实测最多只够到 Z≈64.5（就是下面 DEFAULT_CARRY_CLEAR 里 68.51→64.54 那次）。
#   于是"抬到一半就停住、底座报警"。
#
#   平放的方块只有一层高，从它上面过要的净空跟"摞了两三层"完全是两码事。
#   ★ 净空本身仍然**恒等于这个数**（见 carried_body 的注释: 净空 =
#     (吸盘 Z − body) − 方块顶面 = clear），不是"越矮净空越小"。
#   ★ 20mm 不是拍脑袋: 当年真正蹭上去的那次间距是 5mm（见上面
#     DEFAULT_CARRY_CLEAR 的实测），20mm 是它的 4 倍，且仍高于方块顶面。
#   ★ 只对**第 0 层**放宽; 第 1 层起仍用 carry_clear(35) —— 摞起来的方块顶上
#     没有"平放"这个退路，宁可贵一点。见 carry_clear_mm()。
CARRY_CLEAR_LOW_MM = 20.0

# ★ 搬运路径离某块方块多近，才算「会撞上、得抬过去」。
#   方块是 30mm 见方（半宽 15），吸着一块从旁边过时两块中心距 < 30mm 就真贴上了；
#   取 45mm 是再留 15mm 余量（原 35mm；2026-09-18 用户要求放宽到 4.5cm）。
#   判据是「方块中心 → 路径线段的最短距离」。
#   ★ 它同时决定三件事，放宽的代价要一起看:
#     ① path_blockers —— 判定范围变大 ⇒ 更多路径被判成"要越障"⇒ 更容易需要先升；
#     ② rise_point    —— 升高点离放点更远 ⇒ 低空那段更短、升得更早；
#     ③ escape_point  —— 低空能走的那段更短 ⇒ 升高点更靠近抓点、半径涨得更少。
#   三条都指向同一个方向: 放宽它会让「顶关节限位」**更容易**发生，不是更不容易。
#   真正治那个毛病的是 escape_point（见它的注释），不是这个数 —— 别指望调大它。
CARRY_NEAR_MM = 45.0

# ★ 「快到底座那一圈了」的半径（mm）—— 进去之前就得把搬运高度降下来。
#   ★★ 这不是关节限位，SDK 里既没有逆解也没有限位可查（见 execute.joints_note）。
#      它是一条**照实测画出来的**"该开始小心"的线:
#        半径 151.7mm → 随便走;
#        半径 130mm  → Z=94.51 走到一半停住（ΔXYZ=6.21）;
#        半径 117.5mm → 最多够到 Z≈64.5（命令 68.51、差 3.97mm）。
#      画在 150 是为了给 early_descent 一个动手的地方，不是"到了这儿就一定
#      够得着/够不着"。真实的分界线只能靠在现场记 (半径, Z, J2, J3) 攒出来。
NEAR_BASE_R_MM = 150.0

# ★ 想在某个 XY 降下来之前，那个点周围至少要空出这么多（mm）。
#   判据是「方块中心到该点的距离」。用户口径是 2cm，但那是**边到边** ——
#   方块 30mm 见方，两边各留 10mm 才是真正想表达的"空出一块地方"，
#   所以按中心距取 30 + 10×2 = 50。
#   ★ 比 CARRY_NEAR_MM(45) 略严: 这里问的是"能不能停下、原地垂直降下去"，
#     比"能不能横着蹭过去"更该留余量。
NEAR_CLEAR_MM = 50.0

# ★ 抓取时吸盘点在检测位置上再偏这么多（dx, dy），单位 mm。
#   **按方块记录里的 near 戳（它离哪个角码最近）分别给**（键 = paths.WORLD_NEAR_P1..P4）。
#
#   为什么有这一条（用户实测）: 相机给的方块坐标在纸面四个角上各偏一点，量还不一样
#   （下面那四个数就是逐角量出来的）。
#
#   ★★ **只在"相机刚拍完的那一次抓取"补** —— 判据是记录里的 **src 戳**
#     （paths.WORLD_SRC_KEY == "camera"，color_vision 写记录时盖上的）。
#     机械臂碰过这块方块之后，refresh_world_state 重写这条记录会把 src 戳去掉，
#     再抓它就按机械臂确认过的真实坐标走，**一个毫米都不补**。
#     ★★ 为什么不能每次都补（用户 2026-09-26 实测后让撤销的就是这一条）: 这几毫米
#       补的是「**相机给的坐标**」的系统偏差。机械臂自己确认过的落点**没有**这个
#       偏差，再补一次等于把它推离真实位置 5mm —— 现象是"头一次抓得住、抓过之后再
#       抓就抓不住了"。用户原话:「靠近P4的，除了一开始需要修改一下坐标位置，之后
#       每次不需要修改了。只改动第一次的即可」。
#     ★ 「第一次」是**每条相机记录**的第一次抓取，不是"这块方块的第一次": 相机重新
#       拍一次桌面就是一份新记录（src 戳重新盖上），那一次照样补。
#
#   ★★ 方向口径（用户给定，别再自己推）: 表里的数就是**加在吸盘点上的量**，
#     即「命令机械臂到达的位置 = 记录里的位置 + 这个偏移」。
#     ★ 它**不是**「观测到往哪偏、再取反」—— 那要多走一步反号，四个角里最容易在这
#       一步上把符号搞反（本文件早先放过一版就是这么推反的）。
#     ★ dx/dy 都是**机械臂坐标**，正负直接照机械臂来，不折算画面方向。
#   四个角各自的值:
#     · near = P1 的方块: 到达位置 = 记录里的位置 + (+5,  +1)
#     · near = P2 的方块: 到达位置 = 记录里的位置 + (+1,   0)    （Y 不动）
#     · near = P3 的方块: 到达位置 = 记录里的位置 + (+7,  +1)
#     · near = P4 的方块: 到达位置 = 记录里的位置 + (+4,  +3)
#   ★ 四个角**互不相同**，别再假设「纸同一列的两个角一样」: P1 和 P3 起初实测同值，
#     后来按实测各自微调过（现在 X 差 2mm）—— 这些数是量出来的，不是从几何推的。
#   ★ 四个角**全是往 +X/+Y 挪**（相机给的坐标整体偏 −X/−Y），只是量各不相同:
#     X 方向挪得最多的是 P3（7mm），Y 方向是 P4（3mm）—— 别以为"最大的"是同一个角。
#     ★ 别因为"现在清一色是正的"就把这张表退化成**一个常数**再拍在四个角上:
#       四个角的值差到 7mm（P3 的 X）比 P2 的 1mm 大出好几倍，平均掉就白补了。
#
#   ★ 「离哪个角最近」由 color_vision 在**纸自己的坐标系**里算好、随记录写进
#     world_state 的 near 戳（paths.WORLD_NEAR_KEY），不是在这儿拿机械臂坐标猜 ——
#     机位一变，机械臂坐标的"四个区域"就跟着变，纸面的四个角却是钉死的。
#   ★★ near 判不出来（少二维码）时**不补**: 下面是按 near **下标取值**，取不到就是
#     None。宁可少补这几毫米，也不赌错方向白偏。
GRASP_OFFSET_MM = {
    WORLD_NEAR_P1: (+5.0,  +1.0),   # 到达位置 = 记录里的位置 + (+5, +1)
    WORLD_NEAR_P2: (+1.0,   0.0),   # Y 不动，四个角里补得最少
    WORLD_NEAR_P3: (+7.0,  +1.0),   # ★ 和 P1 不再同值（X 差 2mm）；X 补得最多
    WORLD_NEAR_P4: (+4.0,  +3.0),   # Y 补得最多（3mm）
}

# ★★ 抓取**高度**的补偿。★ 判据和上面那张**不一样**，别照抄:
#     上面那张**只在相机刚拍完的那一次抓取**补、认的是**哪条记录**（src 戳），
#     角来自 near 戳；这一张**每一抓都补**、认的是方块「**此刻**离哪个角码最近」——
#     用方块记录现在的 x/y 去跟这一帧四个角码的实际位置比
#     （保留键 WORLD_CODES_KEY，color_vision 存下来的）。
#     用户 2026-09-26 的原话: 「所有在P1附近的物体，**不管是一开始[在]还是后来被
#     移过去的**，Z都减少3mm」。所以机械臂把它搬到 P1 那一带之后再抓，这一档也吃得上。
#     ★ 和 XY 那一档相反，这一档**不受 src 戳限制**: 它的成因（见下）跟"坐标是谁给的"
#       无关，搬过去之后照样存在。
#
#   ★ 表里**没列出来的角 = 不补（0.0）**，所以现在只有 P1、P2 两档。
#   ★ 为什么单独立一张表、不并成 (dx, dy, dz) 三元组: 这两件事**判据都不一样**
#     （哪条记录 vs 此刻在哪儿，见上），而且**各量各的** —— XY 那点偏差是"相机给的
#     坐标"的系统偏差，高度这个是**那一带的地面/纸面不平**（吸盘贴不到底、吸不住）。
#     并成一个元组的话，凡是读 dx/dy 的地方（describe_plan、自检、真机演示）
#     全都得跟着改成三元素，而它们本来只关心 XY。
#   ★ 单位/正负: 同样是**加在命令 Z 上的量**，正负照机械臂来。
#     −3.0 = 这一抓命令臂**再往下扎 3mm**（比公式算出来的低 3mm）。
#   ★ 只把 grasp_z 挪下去: place_z 不动；grasp_hover / travel_z 跟着 grasp_z 走
#     （它们本来就是从 grasp_z 推的，见 plan_action ①③）。用户原话「其他位置不变」
#     指的是**别的角不动** —— 所以这张表只列实测出问题的那两档。
GRASP_OFFSET_Z_MM = {
    WORLD_NEAR_P1: -3.0,   # 命令 Z 再降 3mm（往下扎），补 P1 那一带吸盘贴不到底
    WORLD_NEAR_P2: -3.0,   # 同上: P2 那一带实测也贴不到底，一样再降 3mm
}


def nearest_code_of_xy(codes, x: float, y: float) -> str | None:
    """★ 「(x, y) 这个**机械臂坐标**此刻离哪个角码最近」—— 判不出来返回 None。

    ★ 和 color_vision.nearest_code_in_paper 是**同一件事的两种坐标系**，别互相顶替:
      · 那边在**纸面 mm** 里判、用的是纸钉死的四个角 → 给"**相机看见那一刻**它在哪个
        角"定性，结果盖进 near 戳（GRASP_OFFSET_MM 拿它选角）。
      · 这边在**机械臂 mm** 里判、用的是**这一帧**四个码实际在哪 → 给"它**此刻**在
        哪儿"定性（GRASP_OFFSET_Z_MM 用它）。
      两件事的输入不同，所以必须各判一次，不能共用一份结果。
    ★★ 四个码**缺一不可**: 少一个码，落在它附近的方块就会被判给隔壁那个码 ——
      于是把 P1 的 −3mm 补到本来在 P2 那一带的方块上。判不出来宁可不补，
      和 near 戳一个口径（见上面 ★★）。
    """
    if not isinstance(codes, dict) or set(codes) != set(WORLD_NEAR_CODES):
        return None
    best, best_d = None, None
    for code, xy in codes.items():
        try:
            cxy = (float(xy[0]), float(xy[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            return None                      # 记坏了就当判不出来，别猜
        d = math.hypot(x - cxy[0], y - cxy[1])
        if best_d is None or d < best_d:
            best, best_d = code, d
    return best


def _here_xy(entry: dict | None, gx: float, gy: float) -> tuple[float, float]:
    """方块**此刻**在哪儿（判高度那一档用）。

    ★ 优先信记忆库那条记录: 机械臂搬过之后 refresh_world_state 已经把它改成了
      机械臂确认过的落点，那才是"现在在哪儿"。
    ★ 记录里没有坐标（空世界 / --virtual 之类）就退回模型给的抓点 —— 两者本该一致。
    """
    if isinstance(entry, dict) and isinstance(entry.get("x"), (int, float)) \
            and isinstance(entry.get("y"), (int, float)):
        return float(entry["x"]), float(entry["y"])
    return float(gx), float(gy)


# ★ 竖着升的时候一次升多少（mm）。
#   这台机器「够不着」**不会报错**，只会停在半路 —— 2026-09-18 实测: 命令升到
#   Z=68.51，实际停在 64.54（差 3.97mm），只有 verify_arrival 的「到位偏差过大」
#   能看出来。整段一次走的话，等你发现时已经停在半路了；分段走，短掉的那一小步
#   就是边界，能当场说清「到 Z=xx 就上不去了、当时离底座 xx mm」。
RISE_STEP_MM = 10.0

# ★ 换升高点时，半径至少要涨这么多（mm）才值得多走一个路点。
#   涨不了多少就回原地升 —— 多一个路点就是多一次「吸着方块悬在半空」的机会。
ESCAPE_MIN_GAIN_MM = 10.0

# ★ 几何判据的比较容差（mm）。别拿它当"精度"，它只负责一件事:
#   让「刚好等于」这种边界情形不因浮点末位（1e-14 那种）翻面。
EPS = 1e-9


class PlanError(Exception):
    """计划层的问题（缺参数、算不出来）—— 报错要说人话，不要 traceback。"""


# ═══════════════════════════ 一、规划（不碰硬件）═══════════════════════════
def nozzle_z(table_z: float, level: int, obj_h: float,
             net_h: float | None, press: float = 0.0) -> float:
    """
    吸盘接触面该降到哪个 Z。

    level = 那个位置**已有几层**方块。抓取时是「它现在在第几层」，
    放置时是「它将落在第几层」（两者同层时 Z 是同一个数）。

    ★ 为什么 level>0 不能直接用 step4.cube_top_z 的 level 参数:
        cube_top_z(T, obj_h, level) = T + (level+1)*obj_h
      它每加一层就加一个 **obj_h**。但 obj_h(默认 26) 里含了 1mm 的
      「吸盘唇口密封压缩量」—— 那 1mm 只在**最上面那一层贴到吸盘**时才存在。
      叠放时每多一层，真实增加的高度是方块**净高**（比如 25），不是 26。

      算式样例（T 随便取一个数都成立，这里借用 step4 注释里那个**换底座之前**
      的旧读数 -78.518 —— 它只是个样例数，不是现在的纸面 Z，别拿去用）:
          cube_top_z(T, 26, 1) = T + 52      = -26.518   ← 照抄会**高 1mm，吸空**
          正确                 = T + 26 + 25 = -27.518
      step4:501-503 的注释专门警告过这一点（「别照抄 level 参数」），
      所以这里 level>0 一律用净高叠，且 net_h 必须由用户明确给出。

    ★ level==0 时本式退化成 table_z + obj_h - press，与 step4 的
      cube_top_z(table_z, obj_h, 0, press) **完全相等** —— 自检里有一条
      专门钉住这个等价，防止哪天两边改歪。
    """
    # ★ 第 0 层**根本不碰 net_h**，所以要单独一条分支返回。
    #   写成 `return table_z + obj_h + level * net_h - press` 会被 Python
    #   先算 `level * net_h` —— level=0、net_h=None（就是默认值）时直接
    #   TypeError。自检里那条「0 层不许崩」就是钉这个的。
    if level <= 0:
        return table_z + obj_h - press
    if net_h is None:
        raise PlanError(
            f"有方块要放到第 {level} 层，但没给方块净高 --net-h，算不出该降到哪。\n"
            f"  为什么不能拿 --obj-h({obj_h}) 凑: 它含 1mm 吸盘密封压缩量，\n"
            f"  只在最上层成立。照它叠会每层多算 1mm → 吸盘吸空。\n"
            f"  请用卡尺量一个方块的**真实净高**（不含压缩量），例如 --net-h 25"
        )
    return table_z + obj_h + level * net_h - press


def cube_top_z(table_z: float, lv: int, obj_h: float,
               net_h: float | None) -> float:
    """
    停在第 lv 层的方块，它的**顶面**在哪个 Z。lv = 它底下垫了几块。

    ★ lv==0 用 obj_h 而不是净高: obj_h 里那 1mm 吸盘唇口压缩量让结果偏高 1mm，
      对「别撞上」来说是安全方向。别顺手改成净高 —— 那就少算了 1mm。
    """
    if lv <= 0:
        return table_z + obj_h
    if net_h is None:
        raise PlanError(
            f"搬运路径会经过第 {lv} 层的方块，但没给方块净高 --net-h，"
            f"算不出它顶面多高。\n"
            f"  为什么不能拿 --obj-h({obj_h}) 凑: 它含 1mm 吸盘密封压缩量，"
            f"只在最上层成立。\n"
            f"  请用卡尺量一个方块的**真实净高**，例如 --net-h 25")
    return table_z + (lv + 1) * net_h


def carried_body(net_h: float | None, obj_h: float) -> float:
    """
    被吸那块方块自身多厚（吸盘接触面 → 它底面）。

    ★ 给了净高就用净高: obj_h 里那 1mm 是吸盘唇口压出来的，只在「方块平放在纸上、
      吸盘压上去」那一刻存在；吸着一块方块在半空平移时没有那 1mm。
    ★ 没给净高就退回 obj_h（比真实厚度多 1mm、偏安全）—— 老行为，别改成 0。

    ★ 注意这个数**影响的是绝对高度，不是净空**:
        净空 = (吸盘 Z − body) − 方块顶面 = (顶面 + clear + body) − body − 顶面 = clear
      所以 body 取 25 还是 26，净空恒等于 clear，不会因为这里差 1mm 就撞上。
      别拿"改 body 会撞"当理由拦这次重构。
    """
    return net_h if net_h is not None else obj_h


def carry_clear_mm(level: int, carry_clear: float = DEFAULT_CARRY_CLEAR) -> float:
    """
    要越过第 level 层方块时，被吸方块的**底面**至少该高出它顶面多少（mm）。

    ★ 平放（第 0 层）用 CARRY_CLEAR_LOW_MM(20)，摞起来（≥1 层）用 carry_clear(35)。
      为什么分档、为什么偏偏是 20 —— 见 CARRY_CLEAR_LOW_MM 的注释。
    ★ 分档只影响**高度**，不影响安全裕量的定义: 净空恒等于返回的这个数
      （见 carried_body 的注释），不是"层数低净空就小"。
    """
    return CARRY_CLEAR_LOW_MM if level <= 0 else carry_clear


def need_z(top: float, level: int, body: float,
           carry_clear: float = DEFAULT_CARRY_CLEAR) -> float:
    """
    要让被吸方块的底面从「顶面在 top、第 level 层」的那块方块上方过去，
    吸盘得抬到哪个 Z。 = 顶面 + 该层该留的净空 + 方块自身厚度。

    ★ 抽出来是为了让「抓点上空」和「平移高度」用**同一个算式** —— 它们本来就
      是同一件事（带着方块从一块方块上方过），两处各写一份迟早会改歪一处。
    """
    return top + carry_clear_mm(level, carry_clear) + body


def _pt_seg_dist(ax: float, ay: float, bx: float, by: float,
                 px: float, py: float) -> float:
    """点 P 到线段 AB 的最短距离（mm）。A、B 重合时退化成两点距离。"""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _is_exempt(table_z: float, b: dict, obj_h: float, net_h: float | None,
               near_mm: float, exempt_xy: tuple[float, float] | None,
               exempt_top: float | None) -> bool:
    """
    「这块方块算不算落点那一摞」—— 算就不用管它（垂直降下去即可）。

    ★ 抽成函数是为了**只有一份**: path_blockers（哪几块挡路）和 _zones_on_path
      （线段的哪几段挡路）必须用同一个豁免判据，否则「有障碍」和「哪一段脏了」
      会在同一组坐标上给出互相矛盾的答案 —— 那是查不出来的那类错。
    """
    if exempt_xy is None or exempt_top is None:
        return False
    if math.hypot(float(b["x"]) - exempt_xy[0], float(b["y"]) - exempt_xy[1]) > near_mm + EPS:
        return False
    return cube_top_z(table_z, int(b.get("z_level", 0)), obj_h, net_h) <= exempt_top + EPS


def cube_records(world: dict | None):
    """★ 遍历记忆库里**方块记录**的唯一口子 —— 产出 (名字, 记录)。

    ★ 为什么不能直接 `world.items()`: 记忆库里除了四条方块记录，还有带 "_" 的
      **保留键**（现在只有 _codes = 四个角码此刻在机械臂坐标系的哪儿，见
      paths.WORLD_CODES_KEY）。它不是一条 "x/y/z_level" 记录 —— 当成方块用会
      KeyError（它里面没有 "x"），或者把码名当成方块名报给用户。
    ★ 为什么要收成一个口子: 三处几何（path_blockers / _zones_on_path /
      early_descent）都要遍历方块，各写一遍 `if name.startswith("_")` 迟早漏一处 ——
      而漏了的那处会**在真机上**才炸（只有带了 _codes 的记忆库才踩得到）。
      新写遍历记忆库的代码，走这里，别再自己 `world.items()`。
    """
    for name, rec in (world or {}).items():
        if name.startswith("_"):
            continue
        yield name, rec


def path_blockers(table_z: float, world: dict | None, obj: str,
                  ax: float, ay: float, bx: float, by: float,
                  obj_h: float, net_h: float | None,
                  near_mm: float = CARRY_NEAR_MM,
                  exempt_xy: tuple[float, float] | None = None,
                  exempt_top: float | None = None) -> list[tuple[str, int, float]]:
    """
    「方块会不会挡住从 A(ax,ay) 走到 B(bx,by) 这条水平线段」—— 返回挡路的
    [(名字, 层数, 顶面 Z), ...]，按顶面从高到低排。**空列表 = 这一段清净**。

    ★ 为什么只看线段附近: 「抬到比桌上最高那摞还高」在摞离得很远时纯属白抬 ——
      实测就是这么撞上关节限位的: 抓取点在 (119.8, -55.1)、离底座才 132mm，
      却因为落点要放到第 3 层而把搬运高度顶到 Z=120.5，机械臂被逼成一个很紧的
      姿态（贴着底座 + 抬到最高），J2/J3 顶到限位、底座亮红灯。
      离得远的方块根本碰不着，不该影响高度。

    ★ 为什么 A=B 也允许传: 那就是「这个点附近有东西吗」—— 用来查抓取点上空。
      见 plan_action 的 ①。

    ★ exempt_xy / exempt_top: 落点那一摞的特例。就在落点附近、**顶面不高于
      exempt_top** 的方块不算障碍 —— 那是要落上去的那一摞，垂直降下去就行，
      不是横着掠过去的。
    """
    out: list[tuple[str, int, float]] = []
    for name, b in cube_records(world):
        if name == obj:                      # 要被吸走的那块，不是障碍
            continue
        cx, cy = float(b["x"]), float(b["y"])
        # ★ 两个距离判据都带 EPS: 不加就是「刚好等于 near_mm」时结果由浮点末位
        #   决定。实测踩到过 —— 升高点本来就在离落点**正好** near_mm 的地方，
        #   摆在那儿的方块算出来 35.00000000000001 > 35.0，于是豁免不掉、
        #   被当成路径障碍。高度那个判据一直带着 +1e-9，距离这边却裸着，
        #   同一段代码里两套口径。
        if _pt_seg_dist(ax, ay, bx, by, cx, cy) > near_mm + EPS:
            continue                         # 离这一段远 —— 碰不着，不管
        if _is_exempt(table_z, b, obj_h, net_h, near_mm, exempt_xy, exempt_top):
            continue                         # 落点正下方，垂直降下去就行
        lv = int(b.get("z_level", 0))
        out.append((name, lv, cube_top_z(table_z, lv, obj_h, net_h)))
    out.sort(key=lambda t: -t[2])
    return out


def rise_point(table_z: float, world: dict | None, obj: str,
               gx: float, gy: float, px: float, py: float,
               obj_h: float, net_h: float | None, landing: float,
               near_mm: float = CARRY_NEAR_MM) -> tuple[float, float] | None:
    """
    「搬到一半要在哪儿竖着升上去」—— 返回那个点，站不了就返回 None。

    ★ 为什么不在抓取点原地升: 高 Z + 离底座近 = 关节折得最紧的姿势，最容易顶
      限位（2026-09-18 实测: 抓点在 (119.8,-55.1)、离底座才 132mm，原地升到
      Z=120.5 时 J2/J3 顶限位、底座亮红灯）。先贴着桌面平移到离放点 near_mm 的
      地方再升，那儿离底座远得多。

    ★ 这个点在离放点 near_mm 处，所以升到一半、被吸方块底面越过往落点那一摞的
      顶面时，两者还隔着 near_mm 那么远（方块半宽 15mm，所以真正的余量是
      near_mm − 15mm）。

    ★ 只保证**这个点**旁边没方块。低空平移那一段的安全性由调用方保证
      （blockers 非空时走 escape_point，不走这里）。

    ★★ 判"这个点旁边有没有方块"必须走 path_blockers，不能自己写个距离比较 ——
      因为**落点那一摞按定义就正好贴在升高点旁边**: 升高点离放点 near_mm，
      而那一摞就在放点上。落点那一摞是被豁免的（垂直降下去即可），所以要是
      自己写 `距离 <= near_mm 就否决`，只要放点上有摞，升高点就永远"踩着方块"、
      白白退化回抓点原地升 —— 恰好又是顶关节限位那个姿势。
      以前这里写的是裸的 `<`（正好 near_mm 不算贴着），"能用"纯属运气，
      没人写下来为什么。现在把豁免显式接进来，理由就落在纸面上了。
    """
    seg = math.hypot(px - gx, py - gy)
    if seg <= near_mm:
        return None                       # 抓放点本来就挨着，没有中途可站
    t = (seg - near_mm) / seg
    sx, sy = gx + t * (px - gx), gy + t * (py - gy)
    # 退化成点的线段 = 纯距离比较，顺带把「落点那一摞豁免」这条规则原样继承过来
    near = path_blockers(table_z, world, obj, sx, sy, sx, sy, obj_h, net_h,
                         exempt_xy=(px, py), exempt_top=landing)
    return None if near else (sx, sy)


def _zones_on_path(table_z: float, world: dict | None, obj: str,
                   ax: float, ay: float, bx: float, by: float,
                   obj_h: float, net_h: float | None,
                   near_mm: float = CARRY_NEAR_MM,
                   exempt_xy: tuple[float, float] | None = None,
                   exempt_top: float | None = None) -> list[tuple[float, float]]:
    """
    「沿着 A→B 这条线走，会在**哪几段弧长**里蹭到方块」—— 返回 [(进, 出), ...]（mm）。

    ★ 和 path_blockers 是同一个判据的两副面孔: 那边回答"哪几块挡路"，这边回答
      "线段的哪几段脏"。两处必须由同一份几何推出来，否则「有障碍」和「哪一段干净」
      会在同一组坐标上互相矛盾 —— 那种错没有任何现场迹象，只能靠共用一份来杜绝。

    ★ 只返回**向前**的区间（出点 ≤ 0 的直接丢掉）: 抓点身后的方块挡不住往前走，
      不该影响"能走多远"。区间端点都夹到 [0, L] 里，方便调用方直接取极值。
    """
    L = math.hypot(bx - ax, by - ay)
    if L <= EPS:
        return []
    ux, uy = (bx - ax) / L, (by - ay) / L
    zones: list[tuple[float, float]] = []
    for name, b in cube_records(world):
        if name == obj:                      # 要被吸走的那块，不是障碍
            continue
        if _is_exempt(table_z, b, obj_h, net_h, near_mm, exempt_xy, exempt_top):
            continue
        # 「方块中心到这条线的距离 ≤ near_mm」对应线上的一段弧长区间，
        # 一元二次直接解出来: s² - 2·proj·s + (|w|² - r²) ≤ 0
        wx, wy = float(b["x"]) - ax, float(b["y"]) - ay
        proj = wx * ux + wy * uy             # 方块中心在这条线上的投影位置
        perp2 = (wx * wx + wy * wy) - proj * proj
        r = near_mm + EPS                    # 略微放大: 别卡在"刚好等于"上
        if perp2 >= r * r:
            continue                         # 整根线都离它够远
        half = math.sqrt(r * r - perp2)
        if proj + half <= EPS:
            continue                         # 它在这段线的**身后**，不影响
        # ★ 进点夹到 0.0 而不是夹成 1e-12: 调用方有 `s <= EPS` 的判据，也有
        #   `== 0.0` 的等值判据（自检），夹成微量小数会让后者莫名其妙地红。
        entry = 0.0 if proj - half <= EPS else proj - half
        out = min(L, proj + half)
        if out <= EPS:
            continue
        zones.append((entry, out))
    return zones


def clear_run_end(table_z: float, world: dict | None, obj: str,
                  gx: float, gy: float, px: float, py: float,
                  obj_h: float, net_h: float | None,
                  near_mm: float = CARRY_NEAR_MM) -> float:
    """
    从抓点出发、沿着「抓点 → 放点」这条线，**贴着桌面**最多能走多远（mm）。

    ★ 判据就是 path_blockers 用的那一条: 线上每一点离每块方块都要 > near_mm。
      所以返回值以内的任何一点，低空平移过去都不算「从方块旁边蹭过去」，
      不需要额外的高度 —— 这正是"先平移、再升"敢贴着桌面走的原因。

    ★ 抓点自己就贴着方块（或抓放点重合）时返回 0.0 —— 一步都走不了，
      调用方据此退回原地升。

    ★ 这里**不带**落点豁免: 它算的是"往前走多远开始撞"，一律从严。
    """
    L = math.hypot(px - gx, py - gy)
    if L <= EPS:
        return 0.0
    zones = _zones_on_path(table_z, world, obj, gx, gy, px, py, obj_h, net_h, near_mm)
    return min((e for e, _ in zones), default=L)


def clear_run_start(table_z: float, world: dict | None, obj: str,
                    gx: float, gy: float, px: float, py: float,
                    obj_h: float, net_h: float | None, landing: float,
                    near_mm: float = CARRY_NEAR_MM) -> float:
    """
    反着问: 从**放点**往回看，最后一个障碍的"出点"在哪儿（mm）＝ 后段从哪儿干净。

    ★ 这是 clear_run_end 的镜像。前段要问"走到哪儿为止干净"（好在那儿升），
      后段要问"从哪儿起干净"（好在那儿降）—— 抬不高的是**靠近底座的那一头**，
      所以真正需要低着走的是后段。

    ★ 必须走**落点豁免**（传 landing）: 落点那一摞就压在终点上，它的区间出点就是
      L；不带豁免的话"后段"恒为空、降低点永远算不出来 —— 而"摞上去"恰恰是最常见
      的那类指令。落点那一摞本来就是在 place_hover 高度上垂直降下去的，不是障碍。

    ★ 返回值是 [0, L] 里的弧长。等于 L 表示**没有**干净的后段（一路脏到终点），
      调用方据此放弃在主路下降。
    """
    L = math.hypot(px - gx, py - gy)
    if L <= EPS:
        return L
    zones = _zones_on_path(table_z, world, obj, gx, gy, px, py, obj_h, net_h,
                           near_mm, exempt_xy=(px, py), exempt_top=landing)
    return max((o for _, o in zones), default=0.0)


def escape_point(table_z: float, world: dict | None, obj: str,
                 gx: float, gy: float, px: float, py: float,
                 obj_h: float, net_h: float | None,
                 near_mm: float = CARRY_NEAR_MM) -> tuple[float, float] | None:
    """
    「路径上**有方块要越**」时的升高点 —— 返回在哪儿竖着升；没有更好的就 None。

    ★ 为什么不能像以前那样「有障碍就在抓点原地升」:
      有障碍 ⇒ travel_z 是**最高**的（= 最高那块顶面 + carry_clear + body），
      而抓点往往离底座**最近**。最高 × 最近 = 这台机器折得最紧的姿势，最容易
      顶关节限位。2026-09-18 实测就是这么栽的:
        抓点 (113.01, 32.32)、离底座 117.5mm，要升到 Z=68.51，
        机械臂只爬到 64.54 就上不去了（ΔZ=3.97mm、底座报警）。
      而"把绿方块放到红方块左边"这句话**常常**产生一个障碍: 红方块就站在落点
      旁边（中心距 = deepseek_brain.SIDE_GAP_MM），抓点在红方块的**另一头**时，
      这条线直接从它身上过去（距离 ≈ 0 < near_mm）。这条最常走的指令于是经常
      走在这个姿势上。
      ★ 2026-09-18 把旁边的中心距从 30mm 提到 50mm 之后，它**不再必然**是障碍
        （50 > near_mm 45，抓点在同侧时整条线离它够远）—— 上面那句从"必然"改成
        "常常"就是这个原因。有障碍才走本函数，没有的话 rise_point 就够用了。

    ★ 怎么办: 先贴着桌面往放点方向走一段（那一段是干净的，见 clear_run_end），
      走到**还能走的最远处**再竖着升 —— 那儿离底座更远，同样的高度就够得着。
      同一组坐标: 半径从 117.5mm 涨到 151.7mm。

    ★ 为什么取「半径最大」的那个点、而不是「走得最远」的那个点:
      落点完全可能比抓点更靠近底座，这时候沿线段往前走反而是往底座**钻**。
      半径沿一条线段是凸的，所以最大值必在两个端点之一 —— 比一下就够了。

    ★ 只保证**走过去那一路**和**那一点本身**都离方块 ≥ near_mm（低空平移的
      安全性）。升到 travel_z 之后就高于所有方块了，后面随便走。
    """
    L = math.hypot(px - gx, py - gy)
    if L <= EPS:
        return None
    s = clear_run_end(table_z, world, obj, gx, gy, px, py, obj_h, net_h, near_mm)
    if s <= EPS:
        return None                          # 一步都走不开，只能原地升
    t = s / L
    ex, ey = gx + t * (px - gx), gy + t * (py - gy)
    if math.hypot(ex, ey) <= math.hypot(gx, gy) + ESCAPE_MIN_GAIN_MM:
        return None                          # 半径没涨多少，不值当多这一趟
    return (ex, ey)


def sink_point(table_z: float, world: dict | None, obj: str,
               gx: float, gy: float, px: float, py: float,
               obj_h: float, net_h: float | None, landing: float,
               travel_z: float, place_hover: float,
               near_mm: float = CARRY_NEAR_MM) -> tuple[float, float] | None:
    """
    「搬到一半该在哪儿**先降下来**」—— 返回那个点，不用降就返回 None。

    ★ 这是 escape_point 的镜像，治的是**同一类故障的另一半**。
      escape_point 管"升"这件事（在离底座远的地方升），可 2026-09-18 实测证明
      **升上去之后横着走一样会栽**: 抓点 (188.16, 18.97) 离底座 189mm，要横着
      走到放点 (118.71, -40.34)——离底座只有 125mm。为了越障，搬运高度被顶上
      Z=94.51，跑到一半就停在 (124.9, -35.0)：ΔXYZ=6.21mm、底座报警。
      原因还是那个: 离底座越近越抬不高。横着走的那条路，越靠近底座就越该低着走。

    ★ 高度为什么本来要那么高: travel_z 是按**路径上最高那块**的顶面
      + carry_clear + body 算出来的，为的是"从它旁边蹭过去"。可那块方块只在
      路径的**前一段**。越过去之后，这个高度就只剩坏处（离底座近的地方抬不起来），
      一点好处都没有了。

    ★ 所以: 走到**最后一个障碍的出点**（后段从这儿起干净，见 clear_run_start）
      就降到 place_hover，剩下的路低着走。降低点是"越完障、还能低飞"的最早位置。

    ★ 降到 place_hover 是**已知可达**的，不是新赌一把: 那个高度、那个 XY，
      计划本来就要去（"放完抬起离开"就停在 (px,py,place_hover)）。所以这次改动
      不会把机械臂送到一个原本到不了的地方 —— 它只是把"贴近底座的那一段"
      从高位挪到低位。

    ★ 为什么**不**再判一次"这儿是不是快到边缘了": 判不了。SDK 里没有逆解、也没有
      关节限位可查（见 execute.joints_note），"多近才算边缘"画不出一条诚实的线 ——
      画错了就是"该降的时候不降"，也就是原来那个报警。而反过来的代价几乎是零:
      低空那一段的安全性由**水平间距**保证（方块半宽 15mm，两块中心距 ≥ near_mm
      = 45mm ⇒ 横向至少留 15mm），跟高度无关 —— 所以在哪儿降都不会撞，
      降下来只会让后面那段**更容易够得着**。既然"多降一次"没有坏处、
      "少降一次"就是报警，那就一律降。

    ★ 什么时候不做: 本来就不用升（travel_z ≤ place_hover，降下去只是白多两步）、
      抓放点几乎重合、或者一路脏到终点没有干净的后段。
    """
    if travel_z <= place_hover + EPS:
        return None                          # 没有可降的高度，这一步是白加的
    L = math.hypot(px - gx, py - gy)
    if L <= EPS:
        return None
    s = clear_run_start(table_z, world, obj, gx, gy, px, py, obj_h, net_h,
                        landing, near_mm)
    if s <= EPS:
        return None                          # 前段就脏，没有"越完之后"可言
    if s >= L - EPS:
        return None                          # 一路脏到终点，没有干净的后段可低飞
    t = s / L
    return (gx + t * (px - gx), gy + t * (py - gy))


def _enter_circle_t(ax: float, ay: float, bx: float, by: float,
                    r: float) -> float | None:
    """
    从 A(ax,ay) 沿直线走到 B(bx,by)，**第一次进到**半径 r 的圆里时的 t ∈ [0,1]。
    起点就已经在圆里 → 0.0；这条路压根不碰这个圆 → None。

    ★ 只取**小的那个根**（−b−√disc）: 二次方程两个根是"先进去"和"又出来"，
      要的是先进去的那个。
    ★ 解出来落在 (0,1) 之外的一律当 None —— 那表示"进入点在起点身后"或
      "整段走完都没进去"，两种都不该拿来当路点。
    """
    dx, dy = bx - ax, by - ay
    a = dx * dx + dy * dy
    if a <= 1e-12:
        # 退化成一个点: 要么就在圆里，要么永远进不去
        return 0.0 if math.hypot(ax, ay) <= r + EPS else None
    b = 2.0 * (ax * dx + ay * dy)
    c = ax * ax + ay * ay - r * r
    if c <= EPS:
        return 0.0                            # 起点本来就在圆里（或圆上）
    disc = b * b - 4.0 * a * c
    if disc <= EPS:
        return None                           # 碰不到这个圆（含相切）
    t = (-b - math.sqrt(disc)) / (2.0 * a)
    if t <= EPS or t >= 1.0 - EPS:
        return None
    return t


def early_descent(table_z: float, world: dict | None, obj: str,
                  gx: float, gy: float, px: float, py: float,
                  obj_h: float, net_h: float | None, landing: float,
                  travel_z: float, place_hover: float,
                  rise: tuple[float, float] | None,
                  near_base_r: float = NEAR_BASE_R_MM,
                  near_clear: float = NEAR_CLEAR_MM) -> tuple[float, float] | None:
    """
    「降得比 sink_point 更早」—— 返回新的降低点，不该更早降就返回 None。

    ★ 治的是什么（2026-09-19 用户实测）: sink_point 只按**障碍**决定在哪儿降，
      完全没看"降点在哪儿"。路径干净（甚至一个障碍都没有）时它是 None，
      于是机械臂从头到尾都在 travel_z 上横着走 —— 而这条路可能一直走到离底座
      只有 117mm 的地方（那次实测的降低点就是半径 117.4mm）。半径那么小时
      travel_z 根本够不着，于是"抬到一半停住、底座报警"。
      用户的原话: 「（有物体/没有物体）还是会一直抬高平移」「处于工作区边界时
      就不要抬这么高了，应该先放低一点再平移」。

    ★ 怎么办: 放点落在近底座那一圈（半径 < near_base_r）时，从抓点往放点走，
      **第一次踏进那个圈**的那一点就先把高度降下来，剩下的路低着走。
      比 sink_point 更早 → 只会在它没得降（None）或降得太晚时接手。

    ★ 只在三个条件同时成立时才降（缺一个就 None，宁可维持原样）:
      ① 真的需要降（travel_z > place_hover）—— 否则这一步纯属白加;
      ② 候选点周围空得下（没有任何方块中心在 near_clear 之内）—— 用户那条
         「周围 2cm 没有物体」，见 NEAR_CLEAR_MM。降下来是要**停一下**的，
         比横着过去更该留余量;
      ③ 从候选点到放点这一路，在 place_hover 高度上确实干净 —— 拿
         path_blockers 原样判（横向间距 ≥ CARRY_NEAR_MM 就撞不着，与高度无关，
         理由见 sink_point 的注释）。
      ④ 候选点不能**早于升高点** —— 那一段本来就是低空走的，而且升高点之前
         还没升上去，在那儿"降"没有意义。

    ★ 为什么敢"更早降"而不担心撞: 低空平移的安全性由**水平间距**保证
      （方块半宽 15mm、中心距 ≥ 45mm ⇒ 横向至少 15mm），跟高度无关 ——
      所以在哪儿降都不会撞，降下来只会让后面那段更容易够得着。这条论证
      和 sink_point 用的是同一条。
    """
    if travel_z <= place_hover + EPS:
        return None                           # ① 本来就不用升
    L = math.hypot(px - gx, py - gy)
    if L <= EPS:
        return None                           # 抓放点重合，没有"一段路"可言
    if math.hypot(px, py) > near_base_r + EPS:
        return None                           # 放点都不在近底座那一圈，不关这事
    t = _enter_circle_t(gx, gy, px, py, near_base_r)
    if t is None:
        return None
    if rise is not None:
        # ④ 升高点的 t: 它一定在抓→放这条线上（rise_point/escape_point 都在线上取）
        t_rise = math.hypot(rise[0] - gx, rise[1] - gy) / L
        if t <= t_rise + EPS:
            return None                       # 进圈的地方不晚于升高点，够不着手
    cx, cy = gx + t * (px - gx), gy + t * (py - gy)
    # ② 候选点周围要空 —— 这就是用户说的「周围 2cm 没有物体」
    for name, b in cube_records(world):
        if name == obj:                       # 手上那块，不在桌上
            continue
        if math.hypot(float(b["x"]) - cx, float(b["y"]) - cy) < near_clear - EPS:
            return None
    # ③ 剩下的路在低高度上要真的干净
    if path_blockers(table_z, world, obj, cx, cy, px, py, obj_h, net_h,
                     exempt_xy=(px, py), exempt_top=landing):
        return None
    return (cx, cy)


def plan_action(action: dict, table_z: float, obj_h: float, net_h: float | None,
                press: float, hover: float, world: dict | None = None,
                carry_clear: float = DEFAULT_CARRY_CLEAR) -> dict:
    """把一个 DeepSeek 动作翻译成吸盘要走的一串路点。纯算术，不碰硬件。"""
    obj = str(action["obj"])
    gx, gy, gl = float(action["grasp"]["x"]), float(action["grasp"]["y"]), int(action["grasp"]["z_level"])
    px, py, pl = float(action["place"]["x"]), float(action["place"]["y"]), int(action["place"]["z_level"])

    if gl < 0 or pl < 0:
        raise PlanError(f"{obj}: z_level 不能是负数（抓到 {gl} / 放到 {pl}）")

    # ⑥ 按方块记录的 near 戳把吸盘点挪一下（GRASP_OFFSET_MM），补偿实测的
    #    「相机给的坐标在四个角上各偏一点」。
    #    ★ 只改 gx/gy 这**两个局部变量**，绝不写回 world / action —— 「不影响后面对
    #      该物体实际位置的计算」全靠这一条: 记忆库、last_plan.json、发给 DeepSeek
    #      的世界状态，三处都还是记录里的原值，一分都没被这几毫米污染。
    #    ★★ 只在**相机刚拍完的那一次抓取**补 —— 判据是记录里的 src 戳
    #      （WORLD_SRC_CAMERA）。机械臂碰过之后 refresh_world_state 会把这条记录
    #      整条重写成机械臂确认过的坐标（src 戳没了，见它自己的说明），之后再抓
    #      就**一个毫米都不补**。
    #      ★ 为什么（用户 2026-09-26 实测后让撤销"每次都补"）: 这几毫米补的是
    #        「相机给的坐标」的系统偏差，机械臂自己确认过的落点没有这个偏差 ——
    #        再补一次等于把它推离真实位置，现象是"头一次抓得住、抓过之后再抓就抓不住"。
    #      用户原话:「靠近P4的，除了一开始需要修改一下坐标位置，之后每次不需要修改了。
    #      只改动第一次的即可」。
    #    ★ 判哪个角用 near 戳（相机那一眼量到的最近角码）；没有这个戳
    #      （少二维码判不出来）也**不补** —— 判据缺了时保守不动，比赌一个方向强。
    #    ★ 放在这几行（而不是抓取那一步）是为了让后面所有几何都跟着挪:
    #      grasp_near / blockers / rise / sink 全是拿 gx/gy 算的 —— 臂实际去的是
    #      挪过的点，避障当然得按挪过的点算。
    grasp_adj = None
    grasp_adj_near = None
    entry = (world or {}).get(obj)
    entry = entry if isinstance(entry, dict) else None

    # ▸ 高度那一档（GRASP_OFFSET_Z_MM）: 判据和 XY **不一样** —— 认的是
    #   「它**此刻**离哪个角码最近」，拿记录现在的 x/y 跟这一帧四个码的实际位置比
    #   （保留键 WORLD_CODES_KEY）。用户原话「所有在P1附近的物体，不管是一开始[在]
    #   还是后来被移过去的，Z都减少3mm」—— 所以被机械臂搬到 P1 那一带的也吃得上。
    #   ★ 这一档**不看 src 戳**: 那一带的纸面/地面不平时时都在（跟坐标是谁给的无关），
    #     所以**每一抓**都补，和上面只有第一次补的 XY 不同。
    #   ★ 先算这一档、后算 XY: 这里要的是方块**自己**的位置，而下面那几行会把 gx/gy
    #     改成"命令臂去的点"。
    #   ★ 拿不到码的位置（旧记忆库没这个键 / 少码）→ 退回用 near 戳（"不知道它挪没挪，
    #     就按它老家算"）—— 保守方向: 少补一次，而不是把几毫米补到别的角上。
    here_x, here_y = _here_xy(entry, gx, gy)
    now_near = entry.get(WORLD_NEAR_KEY) if entry else None
    grasp_adj_z = GRASP_OFFSET_Z_MM.get(now_near, 0.0)

    # ▸ XY 那一档: 只认**相机刚拍完那一次**（src 戳）。表里没那个角 / 不是相机给的
    #   → 不补。★ 机械臂搬过之后 src 戳就没了（refresh_world_state），所以这一档
    #   天然只作用一次 —— 这不是"顺手"，是它的定义（见上面 ★★）。
    near = entry.get(WORLD_NEAR_KEY) if entry else None
    if entry is not None and entry.get(WORLD_SRC_KEY) == WORLD_SRC_CAMERA:
        grasp_adj = GRASP_OFFSET_MM.get(near)
    if grasp_adj is not None:
        grasp_adj_near = near
        gx, gy = gx + grasp_adj[0], gy + grasp_adj[1]

    # ★ 高度补偿加在**最后**（nozzle_z 算出来的那个数上），不改 nozzle_z 本身 ——
    #   那个函数是"吸盘接触面按几何该到哪"，自检里还钉着它和 step4 的等价性；
    #   这 3mm 是**这台机器那个角落**实测出来的修正，不是几何的一部分。
    grasp_z = nozzle_z(table_z, gl, obj_h, net_h, press) + grasp_adj_z
    # ★ 放置**不带 press**: 抬着的那块方块底面本来就正好落在支撑面上，
    #   再下压就是把方块摁进纸面/下面那块（吸盘放气前压住 → 方块被推歪）。
    #   这一条照抄 step4:730-731。
    place_z = nozzle_z(table_z, pl, obj_h, net_h, 0.0)

    # 三个高度各管各的 —— 2026-09-18 从「整条动作共用一个高度」改过来。
    #   旧版把「抓点上空」和「放点上空」绑成同一个值，于是「放到第 3 层」会把
    #   **抓取点上空**也顶到 Z=115.5；抓点在离底座只有 132mm 的地方时，那个
    #   姿势就是顶关节限位的原因。现在只有真正需要高的那一段才高。
    body = carried_body(net_h, obj_h)
    # 落点现有那一摞的顶面（pl 层之下）。不高于它的方块由 place_z 负责。
    # ★ 这里**必须**回 cube_top_z 去算，不能写 `table_z + pl * net_h`:
    #   方块顶面在 cube_top_z 里是「第 0 层用 obj_h」的规矩（T + obj_h），
    #   而 T + 1*net_h 恰好比它**低 1mm** —— 就是 obj_h 里那 1mm 吸盘唇口压缩量。
    #   差这 1mm 的后果不是"数字难看"，而是**「摞到平放方块上」**（最常见的那条
    #   叠放指令）时，下面那块方块豁免不掉、被当成路径障碍: 搬运高度被凭空顶高、
    #   升高点也被迫退回抓点原地升 —— 正好是顶关节限位那个姿势。
    #   （pl==1 时两者等价，pl>=2 时才真正走 net_h 那条分支。）
    landing = table_z if pl <= 0 else cube_top_z(table_z, pl - 1, obj_h, net_h)

    # ① 抓点上空: 只受**抓点附近**的方块限制。
    #    去抓一块平放的方块、旁边又没别的方块时，这里就是最普通的 grasp_z + hover。
    grasp_near = path_blockers(table_z, world, obj, gx, gy, gx, gy, obj_h, net_h,
                               exempt_xy=(px, py), exempt_top=landing)
    grasp_hover = grasp_z + hover
    if grasp_near:
        # ★ 取所有近邻里最费的那个高度。blockers 返回时已按顶面从高到低排，
        #   而 need_z 随层数单调（层高且净空大），所以 [0] 就是最大值 ——
        #   写成 max() 是为了别把这条单调性当隐含前提埋着。
        grasp_hover = max(grasp_hover,
                          max(need_z(top, lv, body, carry_clear)
                              for _, lv, top in grasp_near))

    # ② 放点上空: place_z 本身已经把落点那一摞算进去了，再留一个 hover，
    #    保证「横着过来接垂直下降」时被吸方块不会蹭到那一摞。
    place_hover = place_z + hover

    # ③ 平移高度: 要越过路径上的障碍，也不能低于两头的悬停高度。
    blockers = path_blockers(table_z, world, obj, gx, gy, px, py, obj_h, net_h,
                             exempt_xy=(px, py), exempt_top=landing)
    travel_z = max(grasp_hover, place_hover)
    if blockers:
        # ★ 同 ① 的道理，但这里更关键: 平放的障碍只留 20mm 净空（见
        #   CARRY_CLEAR_LOW_MM）。原来一刀切 35 会把这个高度顶到够不着的地方。
        travel_z = max(travel_z,
                       max(need_z(top, lv, body, carry_clear)
                           for _, lv, top in blockers))

    # ④ 升高这一下放在哪儿做。**两种情况都要挪走**，区别只是挪到哪儿:
    #      路径清净 → 挪到离放点 near_mm 的地方（那儿离底座远）—— rise_point
    #      路径有障碍 → 走不到放点附近，但可以沿这条线走到"还能走的最远处"再升
    #                   —— escape_point
    #    为什么"有障碍"这一支原来写的是**原地升**、现在必须改:
    #      有障碍 ⇒ travel_z 被顶到最高（障碍顶面 + carry_clear + body），而抓点
    #      往往正是离底座最近的地方。最高 × 最近 = 关节折得最紧的姿势。
    #      更要命的是「放到某块方块旁边」这类指令**常常**产生障碍: 旁边那块就站在
    #      落点旁（中心距 = deepseek_brain.SIDE_GAP_MM），抓点若在它的另一头，
    #      这条线直接从它身上过去。于是这话最常走的指令经常栽在同一个姿势上。
    #      ★ 2026-09-18 中心距从 30mm 提到 50mm 后**不再必然**（50 > near_mm 45），
    #        但"另一头"那种摆放依然会撞上 —— 所以这一支还得留着。
    #    两支都做不到（一步都走不开、或半径涨不了多少）才退回原地升。
    if blockers:
        rise = escape_point(table_z, world, obj, gx, gy, px, py, obj_h, net_h)
    else:
        rise = rise_point(table_z, world, obj, gx, gy, px, py,
                          obj_h, net_h, landing)

    # ⑤ 降到哪儿再把剩下的路走完。
    #    ★ 光有 ④ 不够 —— 2026-09-18 实测: 就算把"升"挪到了离底座远的地方，
    #      **升上去之后横着走**照样栽（抓点 189mm → 放点 125mm，Z=94.51 走到一半
    #      停在半径 130mm、ΔXYZ=6.21mm）。所以越完障就把高度降下来，贴近底座的
    #      那一段低着走。高度是**为了越障**才那么高的，越完了它就没用了。
    #    降到 place_hover 特别稳当: 那个 XY、那个高度，计划本来就一定要去
    #    （"放完抬起离开"就停在那儿），不是新开的赌注。
    sink = sink_point(table_z, world, obj, gx, gy, px, py, obj_h, net_h,
                      landing, travel_z, place_hover)

    # ⑦ 降得**更早**一点 —— 只看障碍的 ⑤ 管不到"降点在哪儿"。
    #    ★ 2026-09-19 用户实测: 路径上只有一块平放的方块时，⑤ 算出来的降低点
    #      落在离底座只有 117mm 的地方，而搬运高度在那个半径上够不着 ——
    #      "抬到一半停住、底座报警"。用户原话「（有物体/没有物体）还是会一直
    #      抬高平移」「处于工作区边界时……应该先放低一点再平移」。
    #    ★ 只在 ⑦ 比 ⑤ 更早、且候选点周围空得下时才接手（判据全在 early_descent
    #      里）。晚于 ⑤ 的一律不要 —— 那等于把 ⑤ 挣来的低空段又还回去。
    early = early_descent(table_z, world, obj, gx, gy, px, py, obj_h, net_h,
                          landing, travel_z, place_hover, rise)
    sink_early = False
    if early is not None and (
            sink is None
            or math.hypot(early[0] - gx, early[1] - gy)
            < math.hypot(sink[0] - gx, sink[1] - gy) - EPS):
        sink = early
        sink_early = True

    return {
        "obj": obj,
        "grasp_xy": (gx, gy), "grasp_level": gl, "grasp_z": grasp_z,
        "place_xy": (px, py), "place_level": pl, "place_z": place_z,
        # 这一抓的 XY 按"老家"补偿过没有（None = 没补，坐标是记录里的原值）。
        # ★ grasp_xy 是**补偿后**的、臂实际会去的点；原始值 = grasp_xy - grasp_adj。
        # ★ grasp_adj_near 是**为什么补了这么多**（"P1".."P4"）—— 只给 describe_plan
        #   打印用: 不补时它和 grasp_adj 一起是 None，不会出现"有 near 没 adj"。
        "grasp_adj": grasp_adj,
        "grasp_adj_near": grasp_adj_near,
        # 这一抓的高度补了多少 mm（0.0 = 没补）。
        # ★ 判据**独立于上面两个**: 按"它此刻离哪个角最近"算（被机械臂搬过去的也算），
        #   所以完全可能出现"XY 没补、高度补了"（老家不在表里 / 没 near 戳，
        #   但此刻正在 P1 那一带）—— 打印和自检都别把两者绑在一起。
        # ★ grasp_z 是**已经加过它**的最终命令值；减掉它才是公式值。
        # ★ grasp_adj_z_near 是"按哪个角补的"（和 grasp_adj_near 可能是**两个角**）。
        "grasp_adj_z": grasp_adj_z,
        "grasp_adj_z_near": now_near,
        "grasp_hover": grasp_hover,    # 去抓 / 抓起来都停这个高度
        "place_hover": place_hover,    # 放完抬起离开停这个高度
        "travel_z": travel_z,          # 吸着方块平移时走的高度
        "rise_xy": rise,               # None = 就在抓点原地升
        "sink_xy": sink,               # None = 一路都在 travel_z 上横着走
        # 这个降低点是 ⑦ 为了"别在近底座那一圈还抬着"提前挪的吗（见 early_descent）。
        # 只给 describe_plan 说明用，执行时两条路走的是同一串动作。
        "sink_early": sink_early,
        # 下面几个只为了让人看懂「高度是怎么来的」，执行时不用
        "carry_body": body,
        "blockers": blockers,          # 平移路径上的障碍 [(名字, 层数, 顶面Z), ...]
        "grasp_near": grasp_near,      # 抓点附近的方块（同上格式）
    }


def build_plan(actions: list[dict], table_z: float, obj_h: float,
               net_h: float | None, press: float, hover: float,
               world: dict | None = None,
               carry_clear: float = DEFAULT_CARRY_CLEAR) -> list[dict]:
    return [plan_action(a, table_z, obj_h, net_h, press, hover, world, carry_clear)
            for a in actions]


def preflight(plan: list[dict], limits: dict, table_z: float) -> list[str]:
    """
    连臂**之前**把能查的都查掉，返回问题列表（空 = 可以动）。

    ★ 为什么要在连臂前查: 走到一半被软限位拒掉，既白跑一趟，也让人以为
      "机械臂坏了"。真正的原因往往只是「DeepSeek 给的坐标在限位外」。
    """
    bad = []
    for i, st in enumerate(plan, 1):
        for what, xy in (("抓取点", st["grasp_xy"]), ("放置点", st["place_xy"])):
            why = s4.why_unreachable(limits, xy)
            if why:
                bad.append(f"[{i}] {st['obj']} 的{what} ({xy[0]:.1f}, {xy[1]:.1f}) {why}")
        for what, z in (("抓取 Z", st["grasp_z"]), ("放置 Z", st["place_z"]),
                        ("抓点悬停 Z", st["grasp_hover"]),
                        ("放点悬停 Z", st["place_hover"]),
                        ("搬运 Z", st["travel_z"])):
            lo, hi = limits["z"]
            if z < lo - 1e-9 or z > hi + 1e-9:
                bad.append(f"[{i}] {st['obj']} 的{what}={z:.2f} 超出软限位 Z [{lo:.0f}, {hi:.0f}]")

    # ★ 两个方块被放到同一个位置 = 物理上撞车。DeepSeek 不知道空位有没有被占
    #   （world_state 里没有"空位"信息），所以这里只能靠自检发现，属于**警告**
    #   而不是拦截 —— 因为"叠上去"和"撞车"在某些坐标下看起来一样，由人判断。
    seen: dict[tuple, str] = {}
    for i, st in enumerate(plan, 1):
        key = (round(st["place_xy"][0], 1), round(st["place_xy"][1], 1), st["place_level"])
        if key in seen:
            bad.append(f"[{i}] {st['obj']} 的放置点和第 {seen[key]} 步落在一起 "
                       f"({key[0]}, {key[1]}, 第{key[2]}层) → 会撞车，请人工确认")
        else:
            seen[key] = st["obj"]
    return bad


def describe_plan(plan: list[dict], table_z: float | None) -> str:
    if not plan:
        return "（空计划: 没有需要执行的动作）"
    lines = []
    for i, st in enumerate(plan, 1):
        lines.append(f"  [{i}] {st['obj']}")
        lines.append(f"      抓 ({st['grasp_xy'][0]:7.2f}, {st['grasp_xy'][1]:7.2f})"
                     f" 第{st['grasp_level']}层  → 降到 Z={st['grasp_z']:.2f}")
        # ★ 补偿过就得写出来: 上面那行是**臂实际会去的点**，和记忆库里记的对不上
        #   （差那几毫米），不说明白就成了"计划里的数字和识别结果对不上"。
        #   没补偿的那种情况不吭声 —— 那才是"就该这样"。
        adj = st.get("grasp_adj")
        if adj:
            raw = (st["grasp_xy"][0] - adj[0], st["grasp_xy"][1] - adj[1])
            # ★ 打印里必须带上**是哪个角**（P1..P4）—— 用户是靠这一行核对"补的
            #   方向对不对"的: 四个角的值不一样，只写"补了 5mm"看不出补的是哪一个。
            near = st.get("grasp_adj_near")
            lines.append(f"      ★ 这块方块在角码 **{near}** 那一带、而且这条坐标"
                         f"**是相机刚给的** —— 吸盘点先挪 ({adj[0]:+.1f}, {adj[1]:+.1f})mm"
                         f"（补 {near} 那一带的实测偏差）: "
                         f"记忆库里 ({raw[0]:.2f}, {raw[1]:.2f}) → 实际去 "
                         f"({st['grasp_xy'][0]:.2f}, {st['grasp_xy'][1]:.2f})")
            lines.append(f"        （只在**相机刚拍完的第一次抓取**补 —— "
                         f"机械臂碰过之后按它确认过的真实坐标走，不再补；"
                         f"这几毫米也不进记忆库，记忆库里仍是原来的值）")
        # ★ 高度那一档**单独判**（判据是"此刻在哪"，跟上面那个"是不是相机刚给的"
        #   是两回事），所以它写在 if 外面: XY 没补、高度补了是很正常的一种组合。
        #   ★ 上面「降到 Z=」那行印的是**已经补过**的数，不写出来用户拿它跟公式
        #     算的对不上，会以为高度算错了。
        dz = st.get("grasp_adj_z") or 0.0
        if dz:
            z_near = st.get("grasp_adj_z_near")
            lines.append(f"      ★ 这一抓的**高度**也补了 {dz:+.1f}mm（它**现在**离 "
                         f"**{z_near}** 最近，那一带吸盘贴不到底）: "
                         f"公式算是 Z={st['grasp_z'] - dz:.2f}"
                         f" → 实际降到 Z={st['grasp_z']:.2f}")
        lines.append(f"      放 ({st['place_xy'][0]:7.2f}, {st['place_xy'][1]:7.2f})"
                     f" 第{st['place_level']}层  → 降到 Z={st['place_z']:.2f}")
        # ★ 三个高度分开写 —— 「这个高度怎么来的」是撞了东西时第一个要看的。
        gz, tz, pz = st["grasp_hover"], st["travel_z"], st["place_hover"]
        body = st["carry_body"]

        gn = st.get("grasp_near") or []
        if gn:
            name, lvl, top = gn[0]
            lines.append(f"      抓点上空 Z={gz:.2f}"
                         f"  ← 抓点旁边就是 {name} 第 {lvl} 层（顶面 Z={top:.2f}），"
                         f"得抬过它")
        else:
            lines.append(f"      抓点上空 Z={gz:.2f}"
                         f"  ← 抓点附近没有别的方块，就一个最普通的高度"
                         f"（抓取 Z + 悬停）")

        bl = st.get("blockers") or []
        if bl:
            name, lvl, top = bl[0]
            belly = tz - body          # 平移时被吸方块的**底面**在哪个 Z
            more = (f"（路径上另外还有 {len(bl) - 1} 块，都比它矮）"
                    if len(bl) > 1 else "")
            # ★ 净空直接从数字里反算（吸盘 Z − body − 顶面），不另存一个值 ——
            #   省得"打印的净空"和"算高度用的净空"哪天变成两个数。
            cle = belly - top
            kind = "平放" if lvl <= 0 else f"第 {lvl} 层"
            lines.append(f"      搬运   Z={tz:.2f}"
                         f"  ← 平移会经过 {name} {kind}（顶面 Z={top:.2f}），"
                         f"按它留净空算的{more}")
            lines.append(f"        （那时被吸方块底面在 Z={belly:.2f}，比它高 "
                         f"{cle:.0f}mm"
                         + (f" —— 平放只留 {CARRY_CLEAR_LOW_MM:.0f}mm，"
                            f"摞起来的才要 {DEFAULT_CARRY_CLEAR:.0f}mm"
                            if lvl <= 0 else "") + "）")
        else:
            lines.append(f"      搬运   Z={tz:.2f}"
                         f"  ← 路径上没有别的方块，用两端悬停里更高的那个")

        lines.append(f"      放点上空 Z={pz:.2f}"
                     f"  ← 放置 Z + 悬停（横着过来接垂直下降，蹭不到落点那一摞）")

        # ★ 「升高点在哪儿」是这个计划里最该看的一行: 高 Z + 离底座近 = 关节折得
        #   最紧、最容易顶限位的姿势。所以不管有没有，都要把**离底座多远**写出来。
        rise = st.get("rise_xy")
        grasp_r = math.hypot(*st["grasp_xy"])
        if rise:
            rise_r = math.hypot(*rise)
            why_rise = ("路径上有方块要越，走不到放点附近 —— 走到还能走的最远处再升"
                        if bl else "路径清净，挪到放点附近再升")
            lines.append(f"      升高点 ({rise[0]:.2f}, {rise[1]:.2f})"
                         f"  离底座 {rise_r:.0f}mm（抓点只有 {grasp_r:.0f}mm）"
                         f"  ← {why_rise}")
        elif tz > gz + EPS:
            # ★ 只陈述事实，不喊狼来了。这里**不能**断言"危险" —— 抓点离底座
            #   300mm 时原地升完全没事，117mm 时才是真危险，而 SDK 里既没有逆解
            #   也没有关节限位可查（见 execute.joints_note），没有一个诚实的分界线
            #   可画。编一个阈值出来只会让人对着假警报麻木 —— 那比不报警更糟。
            lines.append(f"      升高点 无  ← 要升 {tz - gz:.1f}mm，但没地方可挪"
                         f"（一步都走不开 / 半径涨不了多少），只能在抓点原地升")
            lines.append(f"        抓点离底座 {grasp_r:.0f}mm，要抬到 Z={tz:.2f}"
                         f" —— 半径越小、抬得越高，越容易顶关节限位；"
                         f"这两个数差得越开越没事")

        # ★ 「降低点」＝ 横着走的路也拆成两段: 高着越障的那一段，和降到低位走的
        #   那一段。2026-09-18 实测: 只把"升"挪走还不够 —— 升上去之后横着走，
        #   越靠近底座越抬不高，一样会停在半路（ΔXYZ=6.21mm、底座报警）。
        sink = st.get("sink_xy")
        if sink:
            sink_r = math.hypot(*sink)
            if st.get("sink_early"):
                why_sink = (f"← 再往前就是离底座 {NEAR_BASE_R_MM:.0f}mm 以内那一圈，"
                            f"Z={tz:.2f} 在那儿够不着 —— 提前降下来")
            else:
                why_sink = f"← Z={tz:.2f} 只是为了越障，越完就降下来"
            lines.append(f"      降低点 ({sink[0]:.2f}, {sink[1]:.2f})"
                         f"  离底座 {sink_r:.0f}mm  {why_sink}")
            lines.append(f"        剩下的路从 Z={tz:.2f} 降到 Z={pz:.2f} 再横着走"
                         f"（离底座越近越抬不高，贴底座那一段得低着走）")
        elif (math.hypot(*st["place_xy"]) <= NEAR_BASE_R_MM + EPS
                and tz > pz + EPS):
            # 该降而没降: 放点在近底座那一圈、又确实要抬那么高，却没找出安全
            # 的提前降点（周围不够空 / 后段不干净）。如实说，不掩盖。
            lines.append(f"      降低点 无  ← 放点离底座只有"
                         f" {math.hypot(*st['place_xy']):.0f}mm，"
                         f"要抬着 Z={tz:.2f} 走进去；"
                         f"但找不到能安全提前降下来的点（附近不空 / 后段不干净）")
    if table_z is not None:
        lines.append(f"  （纸面 Z={table_z:.2f}）")
    return "\n".join(lines)


# ═══════════════════════════ 二、执行（碰硬件）═══════════════════════════
def arm_ready(api, dType, clear_alarms: bool, log=print) -> tuple[bool, str]:
    """
    报警门禁: 返回 (能不能动, 不能动的原因)。单独成函数是为了能离线测。

    ★ 丢步报警(0x50~0x5F)是**硬闸门** —— 零点已经不可信，清除它只会把
      唯一线索抹掉，然后量出一整套错的坐标。所以这里没有 force 后门。
    """
    ares, alist = tc.read_alarms(api, dType)
    log(f"[报警] result={ares}  {tc.describe_alarms(alist)}")
    if not tc.has_alarms(alist):
        return True, ""
    tc.report_alarm_detail(alist)
    if tc.needs_homing(alist):
        return False, ("有丢步报警: 零点已经不可信，所有坐标读数都作废。\n"
                       "  先跑 tools/home_arm.py 回零，再回来。")
    ok, _left = tc.resolve_alarms(api, dType, alist, clear_alarms)
    if not ok:
        return False, "报警没清干净，拒绝运动。"
    return True, ""


def execute(api, dType, plan: list[dict], args, limits: dict,
            mode: str = "movl", log=print, holder: dict | None = None,
            progress: dict | None = None) -> int:
    """
    真的把 plan 走一遍。返回 0 成功 / 1 失败。

    ★ api / dType 是**参数**，不是全局 import —— 这样离线自检可以塞替身进来
      （手法抄自 step2._FakeSdk，见文件尾的 _FakeDobot）。

    ★ holder 是「真空泵现在开着吗」的共享状态（一个 dict，就地更新）。
      调用方的 finally 靠它决定要不要兜底关泵。**不能用返回值代替** ——
      中途失败时返回值是"失败"，但泵可能正开着，那才是必须关泵的时候。
      （这正是 step4 用 held 变量记着的原因。）

    ★ progress 是「哪些方块**真的**放下去了」的共享状态（同一个道理: 就地更新）。
      它跟刷新记忆库有关，所以判据必须是「关真空那一刻」而不是「整个动作跑完」:
      关真空之后方块已经落在桌上，哪怕紧接着的"抬起离开"失败，它的位置也已经
      是确定的。反过来，若在**吸着方块**的时候中止，那块掉在哪就没人知道了 ——
      这种情况它不会进 progress，由 holder["held"] 另行报警。
      也不能只看返回值: 5 个动作跑完 3 个才失败，那 3 个是有效记忆。

    每个动作的路径:
        悬停到抓取点上方 → 垂直下降到方块顶面 → 开真空 → 抬起
        → 平移到放置点上方 → 垂直下降 → 关真空 → 抬起离开
    水平段永远走在一个「算好的高度」上，不会刮到桌上的方块。

    ★ 但那个「算好的高度」不等于全程一个高度（2026-09-18 改）: 高是为了**越过
      路径上的障碍**，障碍越完就没用了，而离底座越近越抬不高。所以横着走这条
      路会拆成两段 —— 高着越障，然后在降低点（sink_xy）降到 place_hover，
      贴着底座的那一段低着走。见 sink_point()。

    ★ 高度不是一个常数（2026-09-18 改）: 抓点上空走 grasp_hover、平移走 travel_z、
      放点上空走 place_hover。去抓一块平放的方块时 grasp_hover 就是最普通的高度，
      不会因为「放点要摞到第 3 层」把机械臂在离底座很近的地方顶到 Z=115+。

    ★ rise_xy 非空时，平移拆成两段: 先在**低处**贴着桌面走到 rise_xy，在那儿竖着
      升到 travel_z，再横着到放点。见 rise_point() —— 就是为了别在离底座近的地方
      摆出「贴着底座 + 抬到最高」那个顶关节限位的姿势。
    """
    if holder is None:
        holder = {}
    holder.setdefault("held", False)
    if progress is None:
        progress = {}
    progress.setdefault("placed", [])
    progress["total"] = len(plan)

    cur = tc.read_pose_stable(api, dType, 5)
    r_now = cur["r"]

    def joints_note() -> str:
        """
        「离底座多远 + 四个关节角」—— 够不着的时候只有这几个数说得清。

        ★ 为什么专门打印 J2/J3: 这一整类故障（高 Z + 小半径 = 折得最紧的姿势）
          目前只能靠"走不到"间接发现，因为 SDK 里**没有逆解、也没有关节限位查询**。
          把实测的 (半径, Z, J2, J3) 记几趟，才有依据给 J2 定一个提前预警的阈值 ——
          在那之前，任何拍出来的阈值都不可信。
        """
        p = tc.read_pose_stable(api, dType, 3)
        return (f"离底座 {math.hypot(p['x'], p['y']):.0f}mm  "
                f"J1={p['j1']:.1f}° J2={p['j2']:.1f}° "
                f"J3={p['j3']:.1f}° J4={p['j4']:.1f}°")

    def goto(x, y, z, why) -> bool:
        tgt = {"x": float(x), "y": float(y), "z": float(z), "r": r_now}
        ok, why_bad = tc.check_target(tc.read_pose_stable(api, dType, 5), tgt, limits)
        if not ok:
            log(f"  ✗ {why}\n    {why_bad}")
            return False
        ok, why_bad = tc.move_to(api, dType, tgt, limits, mode)
        if not ok:
            log(f"  ✗ {why}: {why_bad}")
            return False
        ok, why_bad = tc.verify_arrival(api, dType, tgt)
        log(f"  {'✓' if ok else '⚠'} {why}" + (f"  {why_bad}" if why_bad else ""))
        if not ok:
            log(f"      {joints_note()}")
        return ok

    def climb(x, y, z_from, z_to, why) -> bool:
        """
        竖着升到 z_to —— **分成 RISE_STEP_MM 一段一段走**，每段都验一次到位。

        ★ 为什么要分段: 这台机器「够不着」**不报错、只停在半路**。整段一次走，
          等你发现的时候已经停在半路了（2026-09-18 实测: 命令 Z=68.51、
          实测 64.54，差 3.97mm 才被 verify_arrival 抓到）。分段之后，短掉的
          那一小段就是边界，能当场说清"升到 Z=xx 就上不去了"。

        ★ 为什么不硬顶: 某一段没走满就**立刻停手、如实返回失败**。继续往上加目标值
          去顶，结果就是关节顶限位、底座亮红灯 —— 那正是要避免的东西。

        ★ 「升不动」不等于坐标算错，多半是"这个半径上就够不到这么高"。所以报错里
          一定带上离底座多远和 J2/J3，那才是下一次该往哪儿改路线的依据。
        """
        if z_to <= z_from + EPS:
            return goto(x, y, z_to, why)     # 不用升，也照走一步（保持动作的形状）
        z = z_from
        while z_to - z > EPS:
            z = min(z + RISE_STEP_MM, z_to)
            if goto(x, y, z, why if z >= z_to - EPS else
                    f"{why} … 升到 Z={z:.2f}"):
                continue
            log(f"    ★ 到 Z={z:.2f} 就上不去了（目标 Z={z_to:.2f}，"
                f"还差 {z_to - z:.2f}mm）。")
            log("      这不是坐标算错，是**这个位置够不到这么高** —— "
                "要么换更远离底座的升高点，要么别在这个位置抬这么高。")
            return False
        return True

    for i, st in enumerate(plan, 1):
        gx, gy = st["grasp_xy"]
        px, py = st["place_xy"]
        log(f"\n── [{i}/{len(plan)}] {st['obj']} ──")
        if not goto(gx, gy, st["grasp_hover"], f"悬停到抓取点上方 ({gx:.2f}, {gy:.2f})"):
            return 1
        if not goto(gx, gy, st["grasp_z"], f"下降贴住方块顶面 Z={st['grasp_z']:.2f}"):
            return 1

        rc = s4.suction(api, True)
        log(f"  {'✓' if rc == 0 else '✗'} 开真空 result={rc}")
        if rc != 0:
            return 1
        holder["held"] = True          # ★ 从这一刻起，泵是开着的 —— 退出必须关
        log(f"  … 抽气 {args.pump_s:.1f}s")
        time.sleep(args.pump_s)

        if not goto(gx, gy, st["grasp_hover"], "抬起（离开方块顶面）"):
            return 1
        rise = st.get("rise_xy")
        # ★ 名字必须两样。以前两个分支都叫「抬起」—— 用户贴日志时看到的
        #   「✓ 抬起」和「⚠ 抬起」其实是**两个不同的动作**，谁也说不清是哪一步
        #   出了问题。日志是这台机器唯一的"事发现场"，不能有同名不同物。
        if rise:
            # ★ 先在低处走开，再竖着升 —— 别在离底座近的地方摆出折得最紧的姿势。
            if not goto(rise[0], rise[1], st["grasp_hover"],
                        f"贴着桌面移到升高点 ({rise[0]:.2f}, {rise[1]:.2f})"):
                return 1
            if not climb(rise[0], rise[1], st["grasp_hover"], st["travel_z"],
                         f"竖着升到搬运高度 Z={st['travel_z']:.2f}"):
                return 1
        elif not climb(gx, gy, st["grasp_hover"], st["travel_z"],
                       f"在抓点原地升到搬运高度 Z={st['travel_z']:.2f}"):
            return 1
        sink = st.get("sink_xy")
        # ★ 名字必须两样（同 ④ 的道理）: 「平移到放置点上方」和「压低平移到放置点
        #   上方」是两件事，日志里混成一个名字就没人说得清是哪一步出的问题。
        if sink:
            # ★ 高只是为了越障 —— 越完就**先降下来再把剩下的路走完**。
            #   2026-09-18 实测: 升上去之后横着走，越靠近底座越抬不高，
            #   Z=94.51 走到半径 130mm 就停了（ΔXYZ=6.21mm、底座报警）。
            #   降到 place_hover 是**已知可达**的: 那个 XY、那个高度，
            #   计划本来就要去（下面的"抬起离开"就停在那儿）。
            if not goto(sink[0], sink[1], st["travel_z"],
                        f"平移到降低点 ({sink[0]:.2f}, {sink[1]:.2f})"):
                return 1
            if not goto(sink[0], sink[1], st["place_hover"],
                        f"在降低点降到 Z={st['place_hover']:.2f}（越完障了）"):
                return 1
            if not goto(px, py, st["place_hover"], "压低平移到放置点上方"):
                return 1
        elif not goto(px, py, st["travel_z"], "平移到放置点上方"):
            return 1
        if not goto(px, py, st["place_z"], f"下降放置 Z={st['place_z']:.2f}"):
            return 1

        rc = s4.suction(api, False)
        holder["held"] = False         # 放完了，泵已关
        log(f"  {'✓' if rc == 0 else '✗'} 关真空（放料） result={rc}")
        # ★ 记在这里、不是循环末尾: 关真空这一瞬间方块就落在 (px,py) 了。
        #   等一下的"抬起离开"哪怕失败，记忆库该记的还是这个位置。
        progress["placed"].append(i - 1)
        time.sleep(RELEASE_S)
        if not goto(px, py, st["place_hover"], "抬起离开"):
            return 1
    return 0


def run_arm(args, plan: list[dict], progress: dict | None = None,
            holder: dict | None = None) -> int:
    """
    连臂 → 门禁 → 参数 → execute → 收尾。

    ★ progress / holder 由调用方传进来、就地填。为什么不直接返回它们:
      本函数的返回值是**退出码**（main 直接 sys.exit 出去），而刷新记忆库
      需要的信息在**失败**时同样重要（甚至更重要）—— 跑挂了也要把"已经
      放好的那几个"记下来。
    """
    if progress is None:
        progress = {}
    progress.setdefault("placed", [])
    if holder is None:
        holder = {}                    # 「泵现在开着吗」——由 execute 就地更新
    limits = s4.limits_from_args(args)
    # ★ Z 下限**替换**成「本计划里最低的那一点」，不是取 min 求交集。
    #   为什么必须替换: 默认下限是 -60，而纸面 Z 常常比它还低（样例 -78.5）。
    #   若写成 min(-60, 方块顶面-52.5) = -60，吸盘就还能往下走 7.5mm
    #   —— 扎进方块、压坏它、机械臂丢步，之后零点全不可信。
    #   换成「最深就到这里」之后，任何一条算错的坐标都会被 check_target 拦下。
    #   （step4:610 就是这么做的，语义一致。）
    deepest = min(min(st["grasp_z"], st["place_z"]) for st in plan)
    limits["z"] = (deepest, limits["z"][1])

    api, dType = tc.load_sdk()
    port = tc.find_port(dType, args.port)
    if not port:
        print("✗ 没找到机械臂串口。电源开了吗？USB 插好了吗？")
        return 1

    print(f"[连接] {port} @ 115200 …")
    ret = dType.ConnectDobot(api, port, 115200)
    if ret[0] != 0:
        print(f"✗ 连接失败 state={ret[0]} (1=未找到设备 2=端口被占用)")
        return 1
    print(f"[连接] 成功  fwType={ret[1]}  version={ret[2]}")

    holder.setdefault("held", False)
    try:
        dType.SetCmdTimeout(api, 5000)
        ok, why = arm_ready(api, dType, args.clear_alarms)
        if not ok:
            print(f"\n✗ {why}")
            return 1

        dType.SetQueuedCmdClear(api)
        dType.SetQueuedCmdStartExec(api)
        dType.SetPTPJointParams(api, args.speed, args.speed, args.speed, args.speed,
                                args.acc, args.acc, args.acc, args.acc, isQueued=0)
        tc._set_coord_params(api, dType, args.speed, args.acc)
        dType.SetPTPCommonParams(api, args.ratio, args.ratio, isQueued=0)

        return execute(api, dType, plan, args, limits, mode=args.mode,
                       holder=holder, progress=progress)
    finally:
        try:
            # ★ 判据是 holder["held"]，不是返回值。中途失败时返回值是"失败"，
            #   但泵可能正开着吸着方块 —— 那才是最必须关泵的时候。
            if holder["held"]:
                s4.suction(api, False, isQueued=0)
                print("[收尾] 已关真空（松开吸盘）")
        except Exception:
            pass
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


# ═══════════════════════════ 三、数据进出 ═══════════════════════════
def _rel(p: Path) -> str:
    """
    打印路径时尽量用相对项目根的样子，好认。

    ★ 为什么不是直接 p.relative_to(ROOT): 那个函数**不在 ROOT 底下就抛异常**。
      记忆库的位置来自 paths.py，正常当然在项目里；但 `--world-json` 可以指向
      任何地方，自检里也会把它换成临时目录 —— 一个纯打印用的函数因为路径在
      别处而把整个流程炸掉，不值得。所以在项目外就原样打印绝对路径。
    """
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _read_state_file(p: Path) -> dict:
    if not p.exists():
        raise SystemExit(f"✗ 找不到 {p}。")
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict) or not d:
        raise SystemExit(f"✗ {p.name} 应该是一个 {{\"red\": {{\"x\":..,\"y\":..,\"z_level\":..}}}} 的字典。")
    return d


def load_world_state(args) -> tuple[dict, str, bool]:
    """
    方块坐标从哪来。★ 这是接 OpenCV 的**唯一替换点**，优先级:

      1. --world-json FILE        显式指定的文件（OpenCV 的直接输出）
      2. output/world_state.json  桌面记忆库 —— **默认**（有就用）
      3. --virtual / 前两者都没有  项目自带的假坐标，底座没好时跑通链路用

    返回 (坐标, 来源说明, 是不是**真实**坐标)。

    ★ 为什么默认改成「记忆库优先」而不是「虚拟优先」:
      坐标本身每次都该由相机重新看（color_vision.py 会重写这个文件），
      但 z_level（摞在第几层）**相机永远看不出来** —— 它只存在于记忆库里。
      所以要是不读记忆库，叠放指令的层数每次都会退回 0，吸盘会按"在地上"
      去抓一个摞在第二层的方块 → 吸空。
      真要用假坐标跑链路，显式加 --virtual；想从零开始，加 --reset-world。

    ★ 第三个返回值（真假）不是装饰: 虚拟坐标是**编的**，绝不能写进记忆库。
      否则下一条指令读记忆库时会当真，而来源说明里还写着"x/y 是上次识别的"
      —— 那就成了一句谎话。执行完要不要落盘，就看这个值。
      这个场景很现实: --virtual 本来就是给"接相机之前先跑通链路"用的，
      而跑通链路自然要 --go。
    """
    if args.world_json:
        p = Path(args.world_json)
        return _read_state_file(p), f"{p.name}（--world-json 指定）", True
    if args.virtual:
        # ★ 必须复制一份: 下面 refresh_world_state 是**就地改**这个 dict 的，
        #   直接返回 brain 里那个模块级常量等于把它改脏了（自检里会串）。
        return (copy.deepcopy(brain.VIRTUAL_WORLD_STATE),
                "★ 虚拟坐标（--virtual 强制；底座/OpenCV 还没接时用来跑通链路）",
                False)
    if WORLD_STATE_JSON.exists():
        return (_read_state_file(WORLD_STATE_JSON),
                f"{_rel(WORLD_STATE_JSON)}（桌面记忆库 —— "
                f"x/y 是上次识别的、z_level 靠它一路记下来）", True)
    return (copy.deepcopy(brain.VIRTUAL_WORLD_STATE),
            "★ 虚拟坐标（还没有 "
            f"{_rel(WORLD_STATE_JSON)}；先用假坐标跑通链路）", False)


def refresh_world_state(world_state: dict, plan: list[dict],
                        placed, holding: bool = False) -> list[str]:
    """
    把「真的放下去了」的动作写回记忆库（就地改 world_state），返回人话说明。

    ★ 为什么必须刷新（《方案.md》「刷新记忆库（极度重要）」）:
      相机只能看到 XY，**看不出摞了几层** —— 俯拍图上摞在第二层的方块和躺在
      地上的长得一模一样。z_level 只能「这条指令执行完 +1」一路记下来；
      这份记忆一断，后面所有叠放指令的高度就全错，而且错得**不报错**。

    ★ 为什么层数用 place_level，而不是《方案.md》写的「参照物 z_level + 1」:
      那句只在「摞上去」时对。「放到旁边」时方块是落回桌面的，层数该是 0，
      +1 会把它记成 1 —— 下一条指令就按第 1 层去抓，吸盘多抬 25mm 吸空。
      而 place_level 是 DeepSeek 在看过世界状态之后给出的、并经 deepseek_brain
      复核过的结果（brain.py 会拒绝和它自己算的不一致的答案），
      「放到旁边」那条它算的就是 0。两者本该相等，以 place_level 为准更稳。

    ★ 只信 placed（关真空那一刻记下的），不信返回值: 5 个动作跑完 3 个才失败，
      那 3 个的位置是确定的；剩下的没动过，记忆库里原来的值依然有效。
      但**正在吸着**的那一个例外 —— 它已经被吸离原位了，旧记录是错的，
      见下面 holding 那一段（直接删掉，不留错的）。

    ★★ 重写时**两个戳一个都不留**（src / near 一起丢掉）:
      · src = 「这条 x/y 是相机给的、机械臂还没碰过」—— 搬完这句话就不成立了
        （现在的位置是机械臂确认过的落点），必须去掉。去掉它 XY 补偿
        （GRASP_OFFSET_MM）那一档就不再作用 —— 这正是要的: 那几毫米补的是**相机
        坐标**的系统偏差，机械臂自己确认过的落点没有这个偏差。
      · near = 「相机那一眼量到它离哪个角最近」—— 同上，是**相机那次记录**的属性；
        记录已经被整条换成机械臂确认过的了，留着这个戳只会让人以为"它还在那个角"。
      ★ 高度那一档（GRASP_OFFSET_Z_MM）不受影响: 它按记录**现在的 x/y** 跟
        WORLD_CODES_KEY（顶层保留键，这里不碰）现算"此刻离哪个角最近"。
    """
    notes: list[str] = []
    for idx in placed:
        st = plan[idx]
        obj = st["obj"]
        x, y = st["place_xy"]
        lvl = int(st["place_level"])
        old = world_state.get(obj)
        if not isinstance(old, dict):
            notes.append(f"{obj}: 记忆库里原来是空的 → "
                         f"({x:.2f}, {y:.2f}, 层{lvl})")
        else:
            notes.append(f"{obj}: ({old.get('x')}, {old.get('y')}, "
                         f"层{old.get('z_level')}) → ({x:.2f}, {y:.2f}, 层{lvl})")
        world_state[obj] = {"x": float(x), "y": float(y), "z_level": lvl}

    if holding and len(placed) < len(plan):
        # 中止时吸盘上还吸着东西: 它掉在哪、落到第几层**没人知道**。
        # ★ 处理方式是**把它从记忆库里删掉**，不是留着。
        #   为什么不能留: 留着的那个位置是"抓取前"的 —— 它已经被吸离原位了，
        #   那条记录是错的。留着它，下一次指令就会对着空地抓，而且**不报错**
        #   （全项目最危险的失败方式）。删掉之后模型看不到这个颜色，
        #   会明说"没有这个方块"，逼人来处理。
        who = plan[len(placed)]["obj"]
        world_state.pop(who, None)
        notes.append(
            f"⚠⚠ 中止时吸盘上还吸着 {who} —— 它现在掉在哪、在第几层**没人知道**，"
            f"所以已把它从记忆库里**删掉**（留着的旧位置是错的，会害下一次对空地抓）。\n"
            f"     人工把它摆到一个确定的位置，然后重跑 color_vision.py 重新识别；"
            f"或者 --reset-world 从零开始。")
    return notes


def reset_world_state() -> None:
    """删掉记忆库，层数从 0 重来。★ 你手工重摆过方块之后必须来一下 ——
    否则记忆库里还是「上次指令之后」的位置/层数，和眼前的桌面对不上。"""
    if WORLD_STATE_JSON.exists():
        WORLD_STATE_JSON.unlink()
        print(f"[记忆] 已删 {_rel(WORLD_STATE_JSON)} —— "
              f"所有方块的层数从 0 重新记（按都躺在地上算）。")
    else:
        print(f"[记忆] 本来就没有 {WORLD_STATE_JSON.name}，不用删。")


def report_world_refresh(world_state: dict, plan: list[dict],
                        progress: dict, holding: bool,
                        world_real: bool = True) -> None:
    """
    执行完之后刷新记忆库并落盘。★ 成功、失败都要调 —— 跑挂时"已经放好的
    那几个"的位置是确定的，把它记下来才不至于下一次指令对着空地抓。

    world_real=False（坐标是 --virtual 编的）时**不落盘**: 那些坐标不是
    相机看到的，写进记忆库会让下一条指令当真。
    """
    placed = list(progress.get("placed", []))
    total = progress.get("total", len(plan))
    print(f"\n── 刷新记忆库（{len(placed)}/{total} 个动作真的把方块放下了）──")
    notes = refresh_world_state(world_state, plan, placed, holding)
    for n in notes:
        print(("  ⚠ " if n.startswith("⚠") else "  · ") + n.lstrip("⚠ "))
    if not placed and not holding:
        print("  没有「已放好」的方块 —— 记忆库不落盘（原文件一个字没动）。")
        return
    if not world_real:
        print(f"  ⚠ 坐标是虚拟的（不是相机看到的）—— **不写进记忆库**，"
              f"{_rel(WORLD_STATE_JSON)} 保持原样。")
        print("    等 color_vision.py 真的看过桌面，它自己会把这份记忆写出来。")
        return
    p = persist_world_state(world_state)
    print(f"  已存 → {_rel(p)}"
          f"（上一份留在 {p.name}.bak，覆盖错了还能捞回来）")
    print("  下一条指令的「第几层」就靠这份记忆 —— 别再手工挪方块，"
          "挪了就 --reset-world。")


def _resolve_table_z(args) -> float | None:
    return args.table_z if args.table_z is not None else s4.load_table_z()


def persist_world_state(state: dict) -> Path:
    """
    记忆库落盘。★ 写盘这件事**只有一处实现** —— 在 color_vision.save_world_state
      （带 .bak 备份），这里只是懒导入地调它，免得两份代码各写一半、格式跑偏。

    懒导入（不是写到文件头的 import）: main.py 的规划/自检路径本来不碰 OpenCV，
      没必要因为「执行完要存个 JSON」就把 cv2 拖进所有调用点。
    """
    from color_vision import save_world_state
    return save_world_state(state)


def save_plan(instruction: str, model: str, world_state: dict,
              actions: list[dict], args, world_src: str = "",
              world_real: bool = True) -> Path:
    ensure_output_dir()
    # ★ table_z 必须一起存: --go 会**重新**解析一次纸面 Z（当时给的和现在
    #   读到的可能不是同一个数）。Z 差一点，吸盘就多下降一点 —— 这是会撞到
    #   桌面的量，所以不存下来就没法比对「你过目的」和「要跑的」是不是同一份。
    #   （可能在 None: 计划模式下不给 --table-z 也允许存。）
    PLAN_JSON.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "instruction": instruction,
        "model": model,
        "world_state": world_state,
        # ★ 坐标来源也要存: 你过目的那份计划是从哪来的（真相机 / 记忆库 / 编的），
        #   是"这份计划值不值得信"的一半。复用分支靠 world_real 决定要不要
        #   在跑完之后把坐标写进记忆库 —— 编的坐标绝不能写进去。
        "world_src": world_src,
        "world_real": world_real,
        # ★ 编这份计划时的左右/前后换算指纹。复用前比对，方向改过就拒绝重跑
        #   —— 方向改动不会体现在下面的 place 坐标里，只比对坐标是看不出来的。
        "brain_rules": brain.rules_fingerprint(),
        "actions": actions,
        "params": {"obj_h": args.obj_h, "net_h": args.net_h,
                   "press": args.press, "hover": args.hover,
                   "carry_clear": args.carry_clear,
                   "table_z": _resolve_table_z(args)},
        "note": "--go 默认执行这份计划，不会重新问 DeepSeek。要重问加 --fresh。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return PLAN_JSON


def plan_rules_mismatch(saved: dict) -> str | None:
    """
    旧计划是不是在**另一套左右/前后换算**下编的。是就返回原因（人话），否则 None。

    ★ 为什么必须拦（2026-09-18 真事）: 把 right 从 Y+30 改成 Y−30 之后重跑，
      方块还是落在原来那边。原因不是没改对，而是 `--go` **默认复用**旧计划
      （output/last_plan.json），那份坐标是改之前编的 —— 方向类改动
      一个字都不会体现在旧坐标里，所以必须先比对、对不上就拒绝重编。
      这和纸面 Z 那条是同一类：宁可让你重看一遍计划，也不许「看的是一份、
      跑的其实是另一份」。
    ★ 间距（SIDE_GAP_MM）是同一类里的第二个: 30mm 改成 50mm 之后，旧计划里的
      place 坐标**看起来完全合法**（±50 也是正常数字），不拦就会照老间距摆、
      两块面贴面。所以 `rules_fingerprint()` 里也带着 gap。
    """
    saved_fp = saved.get("brain_rules")
    if saved_fp is None:
        return (f"{PLAN_JSON.name} 是**加指纹之前**存的 —— "
                f"不知道它按哪套左右/前后算的")
    if saved_fp != brain.rules_fingerprint():
        return "左右/前后的换算规则（或方块边长、摆放间距）和存计划时不一样了"
    return None


def load_plan_file() -> dict:
    """读上一份计划。调用点必须自己先判 exists() —— 文件不在是正常情况。"""
    return json.loads(PLAN_JSON.read_text(encoding="utf-8"))


# ═══════════════════════════ 四、离线自检 ═══════════════════════════
class _FakeApi:
    """假的 CDLL。项目里只有吸盘和 MOVL 坐标参数是直接对着 .so 调的。"""

    def __init__(self):
        self.suction: list[bool] = []

    def SetEndEffectorSuctionCup(self, enable, suck, isQueued, idx):
        self.suction.append(bool(suck))
        return 0

    def SetPTPCoordinateParams(self, p, isQueued, idx):
        return 0


class _FakeDobot(tc._FakeSdk):
    """在 step2 的假 SDK 上补齐主程序会用到的那几个函数。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.connected = False
        self.joint_params = None

    def ConnectDobot(self, api, port, baud):
        self.connected = True
        return [0, 1, "fake"]

    def DisconnectDobot(self, api):
        self.connected = False
        return 0

    def SetCmdTimeout(self, api, t):
        return 0

    def SetQueuedCmdClear(self, api):
        return 0

    def SetQueuedCmdStartExec(self, api):
        return 0

    def SetQueuedCmdForceStopExec(self, api):
        return 0

    def SetPTPJointParams(self, api, *a, **kw):
        self.joint_params = a
        return 0

    def SetPTPCommonParams(self, api, *a, **kw):
        return 0


class _RefuseAfter(_FakeDobot):
    """前 n 次 PTP 放行，之后一律拒绝 —— 模拟「走到一半失败」。"""

    def __init__(self, n: int, *a, **kw):
        super().__init__(*a, **kw)
        self.n = n

    def SetPTPCmd(self, api, ptp, x, y, z, r, isQueued=0):
        if len(self.sent) >= self.n:
            return [3, 0]
        return super().SetPTPCmd(api, ptp, x, y, z, r, isQueued)


class _ZLimit(_FakeDobot):
    """
    「这个位置够不到这么高」—— **不报错、只停在半路**。

    ★ 这是这台机器最阴的一种失败，也正是 2026-09-18 那次的真实形态:
      命令升到 Z=68.51，固件不拒绝（返回值 0），实测却停在 64.54（差 3.97mm），
      只有随后核到位的那一步能看出来。
    ★ 所以这里: 指令**照收**（sent 里记的是命令值，方便断言"发了什么"），
      但 pose 的 z 被压在 z_max —— 逼着被测代码必须去**核到位**，
      光看 SetPTPCmd 的返回值是发现不了的。
    """

    def __init__(self, z_max: float, *a, **kw):
        super().__init__(*a, **kw)
        self.z_max = float(z_max)

    def SetPTPCmd(self, api, ptp, x, y, z, r, isQueued=0):
        ret = super().SetPTPCmd(api, ptp, x, y, z, r, isQueued)
        if ret[0] == 0 and self.pose["z"] > self.z_max:
            self.pose["z"] = self.z_max            # ← 悄悄停在半路
        return ret


def selftest() -> int:
    """不用机械臂、不联网，验证「动作 → 路点 → 动作序列」这套账算得对。"""
    ok_all = True

    def ck(name, cond, extra=""):
        nonlocal ok_all
        # ★ 传进来的是函数就当场调用 —— 这里踩过一次很难看的坑:
        #   本文件 29 处 ck() **全部**写成了 `ck("...", t_xxx)`（漏了尾部括号），
        #   传进去的是**函数对象**而不是调用结果，bool(函数) 恒为 True，
        #   于是每一条自检一次都没执行过，却一路打印 ✅。
        #   最糟的是其中包含「完整走一遍动作序列」「丢步报警必须拦下」这些
        #   最该跑的安全条目 —— 安全网看上去是好的，实际上是空的。
        #   现在「是函数就调用」，两种写法都对；以后漏括号也不会再静默通过。
        if callable(cond):
            # 判定口径 = **「跑起来不抛异常」就算过**，不是看返回值:
            #   下面每个 t_xxx 都是以 assert 报错、以 return 提前收工（成功路径
            #   什么都不返回）写的。所以不能拿返回值去 bool()。
            # 一条挂了也不能连坐: 异常在这里收住、记成「这条没过」，后面的
            #   条目照跑 —— 否则第一次跑起来只看得见最前面那个错。
            try:
                cond()
                cond = True
            except Exception as e:                      # noqa: BLE001
                cond = False
                extra = f"{type(e).__name__}: {e}"
        ok_all &= bool(cond)
        print(f"  {'✅' if cond else '❌'} {name}" + (f"   {extra}" if extra else ""))

    print("=" * 72)
    print("  主程序离线自检（不连机械臂、不联网）")
    print("=" * 72)

    T = -78.518
    OBJ_H, NET_H = 26.0, 25.0
    lim = {a: tuple(v) for a, v in tc.DEF_LIMITS.items()}

    # 1. level==0 必须和 step4 的式子**逐位相等**（那是真机跑通的式子，不能改歪）
    def t_level0_same():
        for press in (0.0, 1.0, 2.5):
            mine = nozzle_z(T, 0, OBJ_H, NET_H, press)
            theirs = s4.cube_top_z(T, OBJ_H, 0, press)
            assert abs(mine - theirs) < 1e-12, f"press={press}: {mine} vs {theirs}"
    ck("第 0 层 Z 与 step4 式子完全一致（真机验证过的行为不许变）", t_level0_same)

    # 1b. ★ 最常走的那条路: 不叠放、也就没给 --net-h（默认就是 None）。
    #     第 0 层压根用不到净高 —— 要是代码顺手写成 `level * net_h`，
    #     这里直接 TypeError，而上面那条测试因为传了 NET_H 根本发现不了。
    def t_level0_no_net_h():
        for press in (0.0, 1.0):
            mine = nozzle_z(T, 0, OBJ_H, None, press)
            theirs = s4.cube_top_z(T, OBJ_H, 0, press)
            assert abs(mine - theirs) < 1e-12, f"press={press}: {mine} vs {theirs}"
        # 整条计划也得能建出来（build_plan 是最外层入口）
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        p = build_plan([act], T, OBJ_H, None, 0.0, DEFAULT_HOVER)
        assert len(p) == 1 and p[0]["grasp_z"] == T + OBJ_H
    ck("不给 --net-h（默认）时的单层抓放也不许崩", t_level0_no_net_h)

    # 2. 叠放必须按**净高**叠，不能拿含压缩量的 obj_h 凑
    def t_stack():
        z1 = nozzle_z(T, 1, OBJ_H, NET_H)
        want = T + OBJ_H + NET_H
        assert abs(z1 - want) < 1e-12, f"{z1} != {want}"
        wrong = s4.cube_top_z(T, OBJ_H, 1)          # 照抄 level 参数的错误写法
        assert abs(z1 - wrong) > 0.9, "两种写法差得应该正好是 1mm 压缩量"
        assert z1 < wrong, "按净高算应该比错误写法**更低**，否则会吸空"
    ck("第 1 层按净高叠（比照抄 level 参数低 1mm，不会吸空）", t_stack)

    # 3. 要叠放却没给 --net-h → 必须明确拒绝，不许拿 obj_h 顶替
    def t_need_net_h():
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 1}}
        try:
            plan_action(act, T, OBJ_H, None, 0.0, DEFAULT_HOVER)
        except PlanError as e:
            assert "net-h" in str(e), "报错里要说清该给哪个参数"
            return
        raise AssertionError("缺 net_h 本该报错却通过了")
    ck("要叠放但没给 --net-h → 拒绝并说清怎么办", t_need_net_h)

    # 4. 放置不带 press（照抄 step4 的理由: 预压会把方块摁进支撑面）
    def t_place_no_press():
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        st = plan_action(act, T, OBJ_H, NET_H, 2.0, DEFAULT_HOVER)
        assert abs(st["grasp_z"] - (T + OBJ_H - 2.0)) < 1e-12, "抓取该带 press"
        assert abs(st["place_z"] - (T + OBJ_H)) < 1e-12, "放置不该带 press"
        assert st["place_z"] > st["grasp_z"], "放置应比预压后的抓取高"
    ck("放置不带 press，抓取带 press", t_place_no_press)

    # 5. ★ 三个高度各算各的: 路径清净时，抓点上空 / 放点上空各自等于
    #    「自己的那个点 + hover」，搬运取两者较高的。旧版是三者共用一个值。
    def t_three_heights():
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        assert st["blockers"] == [], f"没有世界状态，不该有障碍: {st['blockers']}"
        assert st["grasp_near"] == [], st["grasp_near"]
        assert abs(st["carry_body"] - NET_H) < 1e-12
        assert abs(st["grasp_hover"] - (st["grasp_z"] + DEFAULT_HOVER)) < 1e-12
        assert abs(st["place_hover"] - (st["place_z"] + DEFAULT_HOVER)) < 1e-12
        assert abs(st["travel_z"] - max(st["grasp_hover"], st["place_hover"])) < 1e-12
    ck("路径清净 → 抓点上空/放点上空各按各的点算，搬运取较高者", t_three_heights)

    # 5b. ★★ 用户 2026-09-18 报的那条，就是这次改动的靶心:
    #      **去抓一块平放的方块**、旁边没别的东西，但落点要摞到第 3 层。
    #      抓取点上空必须还是普普通通的 grasp_z + hover，
    #      绝不能被放置高度顶到 Z=112+ —— 那样机械臂会在离底座只有 132mm 的
    #      地方折成「贴着底座 + 抬到最高」的姿势，J2/J3 顶限位、底座亮红灯。
    def t_grasp_hover_not_dragged():
        T2 = -18.494                     # 那次真机量到的纸面 Z
        world = {"red": {"x": 200.38, "y": -26.51, "z_level": 0},
                 "blue": {"x": 200.38, "y": -26.51, "z_level": 1},
                 "yellow": {"x": 200.38, "y": -26.51, "z_level": 2},
                 "green": {"x": 119.83, "y": -55.14, "z_level": 0}}
        act = {"obj": "green",
               "grasp": {"x": 119.83, "y": -55.14, "z_level": 0},
               "place": {"x": 200.38, "y": -26.51, "z_level": 3}}
        st = plan_action(act, T2, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert st["blockers"] == [], f"落点那摞不该算障碍: {st['blockers']}"
        assert st["grasp_near"] == [], f"抓点旁边没东西: {st['grasp_near']}"
        want = st["grasp_z"] + DEFAULT_HOVER
        assert abs(st["grasp_hover"] - want) < 1e-12, \
            f"抓取点上空被放置高度顶到 {st['grasp_hover']:.2f} 了（应该是 {want:.2f}）"
        assert st["grasp_hover"] < 45.0, f"{st['grasp_hover']:.2f} 还是太高"
        # 抬起来之后仍然要爬到搬运高度 —— 但那个动作挪到离底座更远的升高点上做
        assert st["travel_z"] > st["grasp_hover"], "这组数据本就该抬高了再搬"
        assert st["rise_xy"] is not None, "路径清净时该挪到升高点再升"
        rise_r = math.hypot(*st["rise_xy"])
        grasp_r = math.hypot(*st["grasp_xy"])
        assert rise_r > grasp_r + 20, \
            f"升高点 {rise_r:.1f} 没比抓点 {grasp_r:.1f} 明显更远离底座"
    ck("去抓平放方块 → 抓取点上空就是普通高度，不被放置高度顶上去",
       t_grasp_hover_not_dragged)

    # 5c. 抓点**旁边**就是一块摞起来的方块 → 抓取点上空得抬过它
    def t_grasp_near_lifts():
        world = {"blue": {"x": 225, "y": 150, "z_level": 1}}
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert [b[0] for b in st["grasp_near"]] == ["blue"], st["grasp_near"]
        want = st["grasp_near"][0][2] + DEFAULT_CARRY_CLEAR + st["carry_body"]
        assert abs(st["grasp_hover"] - want) < 1e-12, \
            f"{st['grasp_hover']:.2f} != {want:.2f}"
        assert st["grasp_hover"] > st["grasp_z"] + DEFAULT_HOVER, \
            "抓点旁边就摞着东西，却还是用了普通高度"
    ck("抓点旁边就摞着方块 → 抓取点上空抬过它（不无脑用普通高度）",
       t_grasp_near_lifts)

    # 5d. ★ 搬运路径经过某摞 → 抬到它上面（这就是撞方块那次的成因）
    def t_carry_over_stack():
        world = {"yellow": {"x": 235, "y": 135, "z_level": 1}}     # 正在路径中点
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert [b[0] for b in st["blockers"]] == ["yellow"], st["blockers"]
        assert st["blockers"][0][1] == 1
        assert abs(st["blockers"][0][2] - (T + 2 * NET_H)) < 1e-12, \
            "那摞顶面该按净高叠两层"
        assert abs(st["carry_body"] - NET_H) < 1e-12, "给了净高就用净高"
        belly = st["travel_z"] - st["carry_body"]
        gap = belly - st["blockers"][0][2]
        assert gap >= DEFAULT_CARRY_CLEAR - 1e-9, f"平移净空只有 {gap:.1f}mm"
        clear = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, None)
        assert st["travel_z"] > clear["travel_z"], "路径上有摞却没抬得比清净时高"
        # 钉住「普通高度确实会撞」: 抓/放顶面 + hover 低于那摞顶面
        floor = max(st["grasp_z"], st["place_z"]) + DEFAULT_HOVER
        assert floor - st["carry_body"] < st["blockers"][0][2], \
            "普通高度本该撞上那摞 —— 说明这条测试没测到点上"
        # ★ 这里只能原地升 —— 但**不是因为"有障碍"，而是因为"一步都走不开"**:
        #   yellow 几乎就压在路径起点上（离抓点 15mm < near_mm），抓点本身
        #   已经在它的近旁区间里了，低空没有任何一段是干净的。
        #   （有障碍但这个理由不成立时会挪走 —— 那正是 5j 测的那条。）
        assert st["rise_xy"] is None, \
            f"抓点本身就贴着障碍、没有可走的低空段，只能原地升，实为 {st['rise_xy']}"
        assert clear_run_end(T, world, "red", 210, 150, 260, 120,
                             OBJ_H, NET_H) == 0.0, \
            "★ 这条自检的对照前提没了: yellow 本来该是「一步都走不开」"
    ck("搬运路径经过第 2 层的摞 → 抬到它上面（普通高度确实会撞）", t_carry_over_stack)

    # 5e. ★ 落点那一摞**不算**搬运障碍 —— 我们是垂直降下去的，place_z 已经算好了。
    def t_landing_stack_ok():
        T2 = -18.494
        world = {"blue": {"x": 200.38, "y": -26.51, "z_level": 1},
                 "yellow": {"x": 200.38, "y": -26.51, "z_level": 2}}
        act = {"obj": "green",
               "grasp": {"x": 119.83, "y": -55.14, "z_level": 0},
               "place": {"x": 200.38, "y": -26.51, "z_level": 3}}
        st = plan_action(act, T2, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert st["blockers"] == [], f"落点那摞不该算障碍: {st['blockers']}"
        # 落点那一摞的顶面 = 第 3 层的支撑面，place_z 正好把吸盘停在它上面
        assert abs(st["place_z"] - (T2 + OBJ_H + 3 * NET_H)) < 1e-12
    ck("落点那一摞不算搬运障碍（垂直降下去就行）", t_landing_stack_ok)

    # 5f. ★ 路径经过**平放**的方块也要抬 —— 不然就退回"只剩几毫米扫过去"的老毛病。
    #     ★★ 而且要按**平放那一档**留净空（CARRY_CLEAR_LOW_MM=20），不是一刀切 35。
    #        这是 2026-09-19 那次"抬到一半停住、底座报警"的根治: 路径上只有一块
    #        平放方块时，一刀切 35 会把搬运高度顶到 Z=67.51，而降低点在半径 117mm、
    #        那儿实测最多够到 Z≈64.5。分成两档之后同一场景只要 Z=52.51。
    def t_flat_cube_on_path():
        world = {"blue": {"x": 235, "y": 135, "z_level": 0}}
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert [b[0] for b in st["blockers"]] == ["blue"], st["blockers"]
        belly = st["travel_z"] - st["carry_body"]
        gap = belly - st["blockers"][0][2]
        assert abs(gap - CARRY_CLEAR_LOW_MM) < 1e-9, \
            f"平放那一档该正好留 {CARRY_CLEAR_LOW_MM}mm，实为 {gap:.1f}mm"
        assert gap < DEFAULT_CARRY_CLEAR, "平放竟然没比摞起来省高度？"
        assert gap >= 15.0, f"净空 {gap:.1f}mm 离当年蹭上去的 5mm 太近了"
        # 绝对高度也要对得上: 顶面 + 平放档净空 + 方块厚度
        assert abs(st["travel_z"] - (st["blockers"][0][2] + CARRY_CLEAR_LOW_MM
                                     + st["carry_body"])) < 1e-9
        # 档位由**层数**决定，不是由调用方传的那个 carry_clear 决定
        assert carry_clear_mm(0) == CARRY_CLEAR_LOW_MM
        assert carry_clear_mm(1) == DEFAULT_CARRY_CLEAR
        assert carry_clear_mm(3, 40.0) == 40.0, "第 1 层起该用传进来的 carry_clear"
    ck("路径经过平放的方块只留 20mm 净空（不是一刀切 35）", t_flat_cube_on_path)

    # 5g. ★ 离路径很远的摞**不参与**。
    def t_far_stack_ignored():
        world = {"yellow": {"x": 185.4, "y": 12.0, "z_level": 2}}
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert st["blockers"] == [], f"离得远不该算障碍: {st['blockers']}"
        clear = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, None)
        assert st["travel_z"] == clear["travel_z"], "远处的摞不该抬高搬运高度"
    ck("离路径很远的摞不抬高搬运高度（旧版会白抬到最高摞）", t_far_stack_ignored)

    # 5h. 路径上真有摞却没给 --net-h → 拒绝，不许拿 obj_h 偷着顶替
    def t_blocker_need_net_h():
        world = {"yellow": {"x": 235, "y": 135, "z_level": 1}}
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        try:
            plan_action(act, T, OBJ_H, None, 0.0, DEFAULT_HOVER, world)
        except PlanError as e:
            assert "net-h" in str(e), "报错里要说清该给哪个参数"
            return
        raise AssertionError("路径上有摞、没给 --net-h 本该报错却通过了")
    ck("路径上有摞但没给 --net-h → 拒绝而不是按平放高度搬", t_blocker_need_net_h)

    # 5i. ★ 升高点: 挪到「离放点 CARRY_NEAR_MM」的地方再竖着升。
    def t_rise_point():
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        rx, ry = st["rise_xy"]
        d = math.hypot(rx - 260, ry - 120)
        assert abs(d - CARRY_NEAR_MM) < 1e-9, \
            f"升高点该离放点 {CARRY_NEAR_MM:.0f}mm，实为 {d:.2f}"
        assert math.hypot(rx - 260, ry - 120) < math.hypot(210 - 260, 150 - 120), \
            "升高点该比抓点更靠近放点"

        # ★★ 落点那一摞**按构造就贴**在升高点旁边: 升高点离放点正好 near_mm，
        #    而那一摞就在放点上。所以判「升高点有没有被占」必须沿用落点豁免那套
        #    规则 —— 否则只要放点上有摞（「摞上去」那种最常见的指令），升高点就
        #    永远被判成"踩着方块"、白白退回抓点原地升，也就是顶关节限位那个姿势。
        #    这条自检专门钉它: 放点下面那块支撑方块（第 0 层、顶面不高于落点面）
        #    不算占位。
        act2 = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
                "place": {"x": 260, "y": 120, "z_level": 1}}
        sx, sy = plan_action(act2, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)["rise_xy"]
        assert abs(math.hypot(sx - 260, sy - 120) - CARRY_NEAR_MM) < 1e-9, \
            "★ 这条自检的前提是「升高点离放点正好 near_mm」，前提没了它就白测"
        world2 = {"blue": {"x": 260, "y": 120, "z_level": 0}}   # 落点的支撑方块
        st2 = plan_action(act2, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world2)
        assert st2["blockers"] == [], "落点那一摞不该算搬运障碍"
        assert st2["rise_xy"] is not None, \
            "★ 落点那一摞按构造就贴在升高点旁，不能被判成「升高点被占」而退回原地升"

        # 反过来: 升高点上真的站着一块**不豁免**的方块（第 1 层，顶面高过落点面）
        # → 必须交出 None，不能把人送去那儿升。plan_action 那边这种方块会先被算成
        # blockers 走 escape_point，所以只能直接问 rise_point —— 但它是 rise_point
        # 自己的安全阀，坏了没人看得出来，得单独钉住。
        world3 = {"blue": {"x": rx, "y": ry, "z_level": 1}}
        assert rise_point(T, world3, "red", 210, 150, 260, 120,
                          OBJ_H, NET_H, T + OBJ_H) is None, \
            "升高点上站着不豁免的方块，rise_point 不该还把它交出来"

        # 抓放点本来就挨着（< near_mm）→ 没有中途可站，直接原地升
        assert rise_point(T, None, "red", 210, 150, 220, 150,
                          OBJ_H, NET_H, T) is None, \
            "抓放点相距不到 near_mm，本来就没有中途点可站"
    ck(f"升高点: 挪到离放点 {CARRY_NEAR_MM:.0f}mm 处；只有**不豁免**的方块才否决它",
       t_rise_point)

    # 5j. ★★ 这条自检就是本次改动的**起因**，用真机上栽过的那组坐标钉死。
    #     抓点 (113.01, 32.32) 离底座只有 117.5mm；放点旁边 30mm 站着红方块，
    #     所以要越障 ⇒ 搬运高度是最高的。旧判据「有障碍就 rise=None」于是让它
    #     **在抓点原地升**到 Z=68.51 —— 那是这台机器折得最紧的姿势，
    #     真机只爬到 64.54 就上不去了（日志 ΔXYZ=3.97mm、底座报警）。
    def t_escape_point():
        T7 = -18.494
        world = {"red": {"x": 187.88, "y": 48.51, "z_level": 0},
                 "yellow": {"x": 199.54, "y": -48.39, "z_level": 0},
                 "blue": {"x": 119.25, "y": -40.07, "z_level": 0}}
        act = {"obj": "green", "grasp": {"x": 113.01, "y": 32.32, "z_level": 0},
               "place": {"x": 187.88, "y": 78.51, "z_level": 0}}
        st = plan_action(act, T7, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert [b[0] for b in st["blockers"]] == ["red"], \
            f"离放点只有 30mm 的红方块该算障碍: {st['blockers']}"

        rise = st["rise_xy"]
        assert rise is not None, \
            "★ 有障碍但低空还有得走 —— 必须挪开再升，绝不能退回原地升"
        r_grasp = math.hypot(113.01, 32.32)
        r_rise = math.hypot(*rise)
        assert r_rise >= r_grasp + ESCAPE_MIN_GAIN_MM, \
            f"升高点该明显远离底座: 抓点 {r_grasp:.1f} → 升高点 {r_rise:.1f}"

        # 低空那一段必须**真的干净**: 全程离红方块 ≥ near_mm。
        # 判据和路径不能自相矛盾 —— 挪过去的那段自己要是贴着方块，等于白挪。
        s = clear_run_end(T7, world, "green", 113.01, 32.32, 187.88, 78.51,
                          OBJ_H, NET_H)
        walked = math.hypot(rise[0] - 113.01, rise[1] - 32.32)
        assert walked <= s + 1e-9, \
            f"升高点 ({rise[0]:.2f}, {rise[1]:.2f}) 跑到了不干净的那一段上"
        assert math.hypot(rise[0] - 187.88, rise[1] - 48.51) >= CARRY_NEAR_MM - 1e-9, \
            "升高点离红方块太近，在那儿升会撞上"

        # ★ 换升高点**不许**顺手动搬运高度 —— 高度是按障碍顶面算的，跟"在哪儿升"无关。
        #   红方块是**平放**的 ⇒ 走 CARRY_CLEAR_LOW_MM 那一档。2026-09-19 之后
        #   这组数据只要 Z=52.51（原来是 68.51）—— 换句话说，就算不挪升高点，
        #   这个高度在 117.5mm 半径上也够得着了; 挪走仍然更好，两条都留着。
        nn, nl, ntop = st["blockers"][0]
        want_tz = ntop + carry_clear_mm(nl, DEFAULT_CARRY_CLEAR) + st["carry_body"]
        assert abs(st["travel_z"] - want_tz) < 1e-9, \
            f"搬运高度该只由{nn}顶面决定（{want_tz:.2f}），实为 {st['travel_z']:.2f}"
        assert st["travel_z"] > T7 + OBJ_H + DEFAULT_HOVER, \
            "这组数据本就该被红方块顶到普通悬停高度之上"
        # 也钉住「真机上栽的那个姿势确实是最紧的」: 抓点半径小、高度却是最高的
        assert r_grasp < math.hypot(*st["place_xy"]), \
            "★ 这条自检的前提是「抓点比放点更贴底座」，前提没了就别信它"

        # 代价说明白: 低空多走这么多，换来的就是「升高时离底座远了这么多」。
        # 真机那次原地升的姿势是 117.5mm 半径抬到 Z=68.51；现在挪到 150mm 开外再抬。
        assert walked > 30.0, f"只多走了 {walked:.1f}mm，等于没挪开"
        assert r_rise - r_grasp > 30.0, \
            f"半径只多了 {r_rise - r_grasp:.1f}mm，顶限位的姿势没改善"

        # 计划打印里要能一眼看出「这次是走了最远处再升」，不能和"路径清净"混为一谈
        text = describe_plan([st], T7)
        assert "走到还能走的最远处再升" in text, \
            f"计划说明该讲清这次为什么在半路升:\n{text}"
    ck("抓点贴底座 + 路径有障碍 → 走到最远处再升（不再被迫原地升）",
       t_escape_point)

    # 5k. ★ 「原地升」那段说明**只陈述事实**，不许一刀切喊危险。
    #     抓点离底座 300mm 时原地升完全没事，117mm 时才是真危险；而 SDK 里既没有
    #     逆解也没有关节限位可查，画不出一个诚实的分界线。编个阈值出来只会让人
    #     对着假警报麻木 —— 那比不报警更糟。
    def t_rise_note_honest():
        act = {"obj": "red", "grasp": {"x": 300, "y": 0, "z_level": 0},
               "place": {"x": 200, "y": 0, "z_level": 0}}
        world = {"blue": {"x": 200, "y": 15, "z_level": 0}}
        st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert st["rise_xy"] is None, "这组数据本该是「没地方可挪」"
        assert st["travel_z"] > st["grasp_z"] + EPS, "这组数据本该要升"
        text = describe_plan([st], T)
        assert "没地方可挪" in text, f"该讲清为什么在原地升:\n{text}"
        assert "顶关节限位" in text, "该说清「原地升」和顶限位的关系"
        assert "最容易" not in text, \
            "★ 抓点离底座 300mm 时原地升根本没事 —— 不许一律说成「最容易顶限位」"
        assert "300mm" in text, "该把抓点离底座多远摆出来，由人自己判断"
    ck("「原地升」那段只陈述事实（抓点离底座远时不喊狼来了）", t_rise_note_honest)

    # 5l. ★★ 这条自检是**第二次**报警的起因: 「升」挪走了还不够，**横着走**照样栽。
    #     抓点 (188.16, 18.97) 离底座 189mm，要横着走到放点 (118.71, -40.34)（只有
    #     125mm）。为了越过 red/green 那两摞，搬运高度被顶到 Z=94.51 —— 真机走到
    #     (124.9, -35.0) 就停了: ΔXYZ=6.21mm、底座报警。
    #     ★ 高度是**为了越障**才那么高的，越完就该降下来；离底座越近越抬不高，
    #       贴底座的那一段必须低着走。
    def t_sink_point():
        T8 = -18.494239807128906
        world = {"red": {"x": 188.16, "y": 18.97, "z_level": 1},
                 "green": {"x": 188.16, "y": 18.97, "z_level": 0},
                 "blue": {"x": 118.71, "y": -40.34, "z_level": 0}}
        act = {"obj": "yellow",
               "grasp": {"x": 188.16, "y": 18.97, "z_level": 2},
               "place": {"x": 118.71, "y": -40.34, "z_level": 1}}
        st = plan_action(act, T8, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)

        # 前提一: 这次确实要越障，搬运高度确实被顶得老高（不然这条自检没测到点上）
        assert [b[0] for b in st["blockers"]] == ["red", "green"], st["blockers"]
        assert abs(st["travel_z"] - (st["blockers"][0][2] + DEFAULT_CARRY_CLEAR
                                     + st["carry_body"])) < 1e-9
        assert st["travel_z"] > st["place_hover"] + 20.0, \
            f"★ 这条自检的前提是「搬运高度明显过高」，实为 {st['travel_z']:.2f}"

        # 前提二（和真机日志一致）: red/green 就压在**抓点**上，低空一步都走不开，
        # 所以"升"这一半这次没法挪，只能原地升 —— 栽的就是这个姿势。
        assert st["rise_xy"] is None, \
            "★ 抓点自己就贴着障碍、低空没有可走的段 —— 和真机日志一致"

        # 「降」这一半才是这次的修法: 越完障先降下来，剩下的路低着走
        sink = st["sink_xy"]
        assert sink is not None, \
            "★ 越完障之后明明有干净的后段，却还打算一路高着横过去"
        gx, gy = st["grasp_xy"]
        px, py = st["place_xy"]
        L = math.hypot(px - gx, py - gy)
        walked = math.hypot(sink[0] - gx, sink[1] - gy)
        assert 0.0 < walked < L - 1e-9, "降低点必须在抓点和放点**之间**"

        # 降低点自己得干净，剩下那一段也得干净 —— 不然降下去就撞上了
        for name, _, _ in st["blockers"]:
            b = world[name]
            assert math.hypot(sink[0] - b["x"], sink[1] - b["y"]) >= CARRY_NEAR_MM - 1e-9, \
                f"降低点 ({sink[0]:.2f}, {sink[1]:.2f}) 离 {name} 太近，在那儿降会撞上"
            assert _pt_seg_dist(sink[0], sink[1], px, py, b["x"], b["y"]) \
                >= CARRY_NEAR_MM - 1e-9, \
                f"降低点到放点这一段还贴着 {name}，低飞照样撞"

        # 省下来的高度就是这次改动的全部收益
        gain = st["travel_z"] - st["place_hover"]
        assert gain > 25.0, f"只降了 {gain:.1f}mm，等于没改"
        assert math.hypot(*sink) > math.hypot(px, py) + 20.0, \
            (f"降低点该比放点离底座远得多（在宽绰的地方才降得下来）: "
             f"降 {math.hypot(*sink):.0f}mm / 放 {math.hypot(px, py):.0f}mm")

        # 计划打印里要能一眼看出「横着走拆成了两段」
        text = describe_plan([st], T8)
        assert "降低点" in text and "越完就降下来" in text, \
            f"计划说明该讲清这次在半路降高度:\n{text}"

        # ---- 不该降的两种情况 ----
        # (a) 本来就不用升: 平放搬到平放，travel_z 就是 place_hover
        act_flat = {"obj": "yellow", "grasp": {"x": 188.16, "y": 18.97, "z_level": 0},
                    "place": {"x": 118.71, "y": -40.34, "z_level": 0}}
        flat = plan_action(act_flat, T8, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, {})
        assert flat["travel_z"] <= flat["place_hover"] + 1e-9, "平放搬平放本来就一样高"
        assert flat["sink_xy"] is None, "没有可降的高度就不该白插两步"

        # (b) 一路脏到终点: 放点旁边就立着**不豁免**的一摞（第 1 层，高过落点面）
        #     → 到终点都得高着，没有能低飞的后段
        dirty = {"red": {"x": 270, "y": 125, "z_level": 1}}
        act_b = {"obj": "yellow", "grasp": {"x": 210, "y": 150, "z_level": 0},
                 "place": {"x": 260, "y": 120, "z_level": 0}}
        st_b = plan_action(act_b, T8, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, dirty)
        assert st_b["travel_z"] > st_b["place_hover"] + 1e-9, \
            "★ 这条自检的前提是「确实要抬」，前提没了它就没测到点上"
        assert st_b["sink_xy"] is None, \
            "★ 终点附近就有障碍、没有干净的后段 —— 不许在那儿降"
    ck("搬运高度只为越障: 越完就先降下来再横着走（治第二次报警）", t_sink_point)

    # 5l2. ★★ 「路径上**一个障碍都没有**，却还是一路抬着走」—— 2026-09-19 用户原话
    #      「（有物体/没有物体）还是会一直抬高平移」。⑤ sink_point 只按**障碍**决定
    #      在哪儿降，没有障碍时它返回 None，于是机械臂从头到尾都在 travel_z 上横着走；
    #      而这条路可能一直走进离底座很近的那一圈 —— 半径小 + 抬得高 = 大臂过陡、报警。
    #      ★ 什么时候真的需要⑧（early_descent）: 抓的是**摞起来**的方块（gl > pl，
    #        所以 grasp_hover 高过 place_hover）、放点在近底座那一圈、路径干净。
    def t_early_descent():
        T9 = -18.494
        # 抓点在 300mm 外、从第 2 层上取；放到离底座只有 110mm 的桌上。
        act = {"obj": "yellow",
               "grasp": {"x": 300, "y": 0, "z_level": 2},
               "place": {"x": 110, "y": 0, "z_level": 0}}
        world = {"yellow": {"x": 300, "y": 0, "z_level": 2}}   # 只有手上那块，路径全空

        st = plan_action(act, T9, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)

        # 前提: 确实"没有障碍却要抬着走"，不然这条自检没测到点上
        assert st["blockers"] == [], f"这组数据本该路径全空: {st['blockers']}"
        assert st["travel_z"] > st["place_hover"] + EPS, \
            "★ 前提没了: 没有障碍时若不比 place_hover 高，就没什么可提前降的"
        assert math.hypot(*st["place_xy"]) <= NEAR_BASE_R_MM, "放点该在近底座那一圈里"

        # 升高点（45mm 处）必须**在圈外** —— 这样"提前降"才有得降（④ 那条约束）
        rise = st["rise_xy"]
        assert rise is not None, "路径清净时该走 rise_point 挪到放点附近再升"
        assert math.hypot(*rise) > NEAR_BASE_R_MM, \
            f"升高点该在近底座那一圈之外: {math.hypot(*rise):.0f}mm"

        # ⑧ 生效: 降低点被挪到**刚踏进那一圈**的地方，而不是一路高到放点
        sink = st["sink_xy"]
        assert sink is not None, \
            "★ 没有障碍就一路抬着走进近底座那一圈 —— 正是这次要修的"
        assert st["sink_early"] is True, "该标成「为了近底座提前降的」"
        assert abs(math.hypot(*sink) - NEAR_BASE_R_MM) < 0.5, \
            f"降低点该正好落在 {NEAR_BASE_R_MM:.0f}mm 那一圈上: {math.hypot(*sink):.2f}mm"
        # 降点必须在升高点**之后**（不然等于降到一半再回去升）
        gx, gy = st["grasp_xy"]
        assert math.hypot(sink[0] - gx, sink[1] - gy) \
            > math.hypot(rise[0] - gx, rise[1] - gy) + EPS, \
            "降低点跑到升高点前面去了"
        # 圈外那一小段还是高的，圈里那一段必须低着走
        text = describe_plan([st], T9)
        assert "提前降下来" in text, f"计划说明该讲清为什么降得这么早:\n{text}"

        # ---- 不该降的情况（逐条钉住 early_descent 的判据）----
        # (a) 放点根本不在近底座那一圈 → 不关这事
        far = plan_action({"obj": "yellow",
                           "grasp": {"x": 300, "y": 0, "z_level": 2},
                           "place": {"x": 260, "y": 0, "z_level": 0}},
                          T9, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert far["sink_xy"] is None and far["sink_early"] is False, far["sink_xy"]

        # (b) 本来就不用降（平放搬平放）→ 不白插两步
        flat = plan_action({"obj": "yellow",
                            "grasp": {"x": 300, "y": 0, "z_level": 0},
                            "place": {"x": 110, "y": 0, "z_level": 0}},
                           T9, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert flat["travel_z"] <= flat["place_hover"] + EPS
        assert flat["sink_xy"] is None

        # (c) 候选点旁边 48mm 就杵着一块（< NEAR_CLEAR_MM 50）→ 不在那儿停、不降。
        #     ★ 48 特意卡在 45(算障碍) 和 50(能不能停) 之间: 它**不是**路径障碍
        #       （不改变高度），但足够近到不该在那儿垂直降落。
        near = dict(world)
        near["blue"] = {"x": 150, "y": 48, "z_level": 0}
        st_n = plan_action(act, T9, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, near)
        assert st_n["blockers"] == [], \
            f"48mm > CARRY_NEAR_MM，它不该算路径障碍: {st_n['blockers']}"
        assert st_n["sink_xy"] is None and st_n["sink_early"] is False, \
            "★ 旁边 48mm 有方块还降下来 —— 用户那条「周围 2cm 没物体」被绕过了"

        # (d) 进圈的点早于升高点 → 够不着手（那一段本来就是低空走的）
        #     把升高点顶到圈里（放点更靠里 ⇒ rise 只有 45mm 处、半径 145）
        inner = plan_action({"obj": "yellow",
                             "grasp": {"x": 300, "y": 0, "z_level": 2},
                             "place": {"x": 100, "y": 0, "z_level": 0}},
                            T9, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert inner["rise_xy"] is not None \
            and math.hypot(*inner["rise_xy"]) < NEAR_BASE_R_MM, \
            "★ 这条的前提是「升高点落在圈里」，前提没了它就没测到点上"
        assert inner["sink_early"] is False, \
            "★ 升高点自己就在圈里时，先把高度降下来再走到它那儿升，是白折腾"

        # (e) 直接钉住 _enter_circle_t 的边界语义
        assert _enter_circle_t(300, 0, 100, 0, 150) is not None
        assert abs(_enter_circle_t(300, 0, 100, 0, 150) - 0.75) < 1e-9
        assert _enter_circle_t(300, 0, 200, 0, 150) is None, "整段都在圈外"
        assert _enter_circle_t(100, 0, 300, 0, 150) == 0.0, "起点就在圈里"
        assert _enter_circle_t(50, 0, 60, 0, 150) == 0.0, "起点在圈里（哪怕终点也在）"
        assert _enter_circle_t(300, 0, 110, 0, 20) is None, "圈太小，根本碰不到"
    ck("近底座那一圈: 提前降到圈外再横着走（治「没障碍也一路抬着走」）",
       t_early_descent)

    # 5l3. ★★ 用**真机那份失败的 last_plan.json** 钉死 fix①的收益。
    #      病情: 路径上只有 blue 一块**平放**的方块，一刀切 35 把搬运高度顶到
    #      Z=67.51；而降低点在半径 117mm、那儿实测最多够到 Z≈64.5 ⇒ 抬到一半停住、报警。
    #      分成两档（平放 20）之后同一场景只要 Z=52.51 —— 半径 117mm 够得着。
    #      ★ 这里把那份计划的世界状态照抄进来，不改它 —— 那几个数是事发现场。
    def t_real_plan_fixed():
        Tr = -18.494239807128906
        world = {"red": {"x": 212.0, "y": -3.32, "z_level": 0},
                 "blue": {"x": 162.0, "y": -3.32, "z_level": 0}}
        act = {"obj": "yellow",
               "grasp": {"x": 171.76, "y": -49.97, "z_level": 0},
               "place": {"x": 112.0, "y": -3.32, "z_level": 0}}
        st = plan_action(act, Tr, OBJ_H, 25.0, 0.0, 30.0, world, 35.0)

        assert [b[0] for b in st["blockers"]] == ["blue"], st["blockers"]
        assert abs(st["travel_z"] - 52.51) < 0.02, \
            f"应该只要 Z≈52.51（原来 67.51），实为 {st['travel_z']:.2f}"
        # 关键结论: 降低点那个半径上，52.51 够得着而 67.51 够不着
        #   （实测: 半径 117.5mm 抬 68.51 只到 64.54）
        assert st["sink_xy"] is not None
        assert math.hypot(*st["sink_xy"]) < 120.0, \
            "★ 这份计划的降低点本来就贴着底座，前提没了这条自检就没意义了"
        assert st["travel_z"] < 64.54, \
            "★ 必须落在实测可达的高度以内（117.5mm 半径实测上限约 64.5）"
        # 平放那一档的净空确实是 20，不是 35
        belly = st["travel_z"] - st["carry_body"]
        assert abs((belly - st["blockers"][0][2]) - CARRY_CLEAR_LOW_MM) < 1e-9
    ck("真机那份计划: 平放障碍只留 20mm → 搬运 Z 52.51（原来 67.51，够不着）",
       t_real_plan_fixed)

    # 5m. ★ 抓取补偿**两张表、两套判据**（这一段最要紧的就是别把两者搞混）:
    #     · XY（GRASP_OFFSET_MM）**只在"相机刚拍完的那一次抓取"补** —— 判据是记录里
    #       的 src 戳，角来自 near 戳: P1→(+5,+1) / P2→(+1,0) / P3→(+7,+1) /
    #       P4→(+4,+5)。机械臂碰过之后 src 戳就没了（refresh_world_state 整条重写），
    #       **之后一毫米都不补**。用户 2026-09-26 的原话:「靠近P4的，除了一开始需要
    #       修改一下坐标位置，之后每次不需要修改了。只改动第一次的即可」。
    #     · 高度（GRASP_OFFSET_Z_MM）**每一抓都补**，看它「**此刻**离哪个角码最近」——
    #       拿记录现在的 x/y 跟这一帧四个角码的实际位置比。现在只有 P1、P2 → −3mm。
    #       「不管是一开始[在]还是后来被移过去的」都算 —— 不看 src 戳。
    #     ★ 两件事都得钉住，缺一个都会**静默**走偏:
    #       · 某个角该挪却没挪 → 这条改动白写；
    #       · 值写错方向 / 四个角写成一样 → 那几个角白补，还多带一个方向的误差；
    #       · 判不出角码（少二维码）却挪了 → 赌错方向；
    #       · **搬过之后 XY 还在补** → 把已经准了的落点又推偏 5mm，现象是"抓过之后
    #         再抓就抓不住"，而且不报错；
    #       · **相机刚给的那次不补** → 每次头一抓都偏，白量了那四个数；
    #       · 高度错用 src 戳 / near 戳判 → 被机械臂搬到 P1 的方块补不上、
    #         或者该补的没补上；
    #       · 高度反过来漏进 XY 的闸门（只在第一次补）→ 搬过去的方块不再降 3mm，
    #         吸盘贴不到底、吸不住。
    #     ★ 高度那一档同样只认表里列的角: 没列的角**一个数都不许动**（0.0），
    #       而且放置高度必须纹丝不动 —— 它是"抓不到底"，跟放下去没关系。
    def t_grasp_offset():
        # ★ 表里必须**刚好这四个角**，一个不多一个不少。将来谁想把"判不出来"也
        #   塞进这个表当默认值，这里会先炸。
        assert set(GRASP_OFFSET_MM) == set(WORLD_NEAR_CODES), \
            f"补偿表只该有四个角码: {sorted(GRASP_OFFSET_MM)}"
        # ★★ 四个值全是**用户实测钉死**的，不是"随便取 5"。口径: 值 = 加在吸盘点上的量
        #     （命令位置 − 记录里的位置），**不取反**；正负都是机械臂坐标的。
        #     ★ 四个角**互不相同**（P1 和 P3 现在 X 差 2mm）—— 别再假设同一列同值。
        assert GRASP_OFFSET_MM == {
            WORLD_NEAR_P1: (+5.0, +1.0), WORLD_NEAR_P2: (+1.0, 0.0),
            WORLD_NEAR_P3: (+7.0, +1.0), WORLD_NEAR_P4: (+4.0, +3.0)}, \
            f"四个角的值是用户实测定的，要改先重新量: {GRASP_OFFSET_MM}"

        # ── 测试用的几何 ──────────────────────────────────────────────────
        # 四个角码摆在四个角上（机械臂 mm）。方块默认摆 (210,150) —— 正好在
        # **P1 那一带**（离 P1 121mm，离 P2 197mm）。(300,150) 则在 **P2 那一带**
        # （离 P2 112mm，离 P1 206mm）。这两处离分界线都够远，改动判据不会翻。
        CODES = {WORLD_NEAR_P1: [100.0, 100.0], WORLD_NEAR_P2: [400.0, 100.0],
                 WORLD_NEAR_P3: [100.0, 400.0], WORLD_NEAR_P4: [400.0, 400.0]}

        def rec(x=210.0, y=150.0, near=WORLD_NEAR_P1, src=WORLD_SRC_CAMERA):
            """一条方块记录。near/src=None = 不写那个键（判不出来）。"""
            e = {"x": x, "y": y, "z_level": 0}
            if src is not None:
                e[WORLD_SRC_KEY] = src
            if near is not None:
                e[WORLD_NEAR_KEY] = near
            return e

        def w(entry, codes=None):
            """一份世界状态。"red" 一条 + 可选的保留键 _codes。"""
            d = {"red": entry}
            if codes is not None:
                d[WORLD_CODES_KEY] = copy.deepcopy(codes)
            return d

        def act_at(x, y):
            return {"obj": "red", "grasp": {"x": x, "y": y, "z_level": 0},
                    "place": {"x": 260, "y": 120, "z_level": 0}}

        act = act_at(210, 150)
        act_before = copy.deepcopy(act)
        # 没补过的那份（空世界）—— 高度、几何的对照。
        plain = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, {})

        # (a) 四个角各挪各的: 挪出去的正好是表里那个数，放置点一律不跟着变，
        #     传进来的两份数据一个字都不许被改。
        #     ★ 这一组**故意不放 _codes**（模拟旧记忆库 / 判不出码的位置）——
        #       高度那一档此时退回按 near 戳（老家）判，也就是老行为；
        #       带码的位置该怎么判见 (e)。
        sts = {}
        for near in WORLD_NEAR_CODES:
            cam = w(rec(near=near))
            cam_before = copy.deepcopy(cam)
            st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, cam)
            ax, ay = GRASP_OFFSET_MM[near]
            dz = GRASP_OFFSET_Z_MM.get(near, 0.0)
            assert st["grasp_adj"] == GRASP_OFFSET_MM[near], st["grasp_adj"]
            assert st["grasp_adj_near"] == near, st["grasp_adj_near"]
            assert st["grasp_xy"] == (210.0 + ax, 150.0 + ay), st["grasp_xy"]
            assert st["grasp_adj_z"] == dz, f"{near}: 高度补偿该是 {dz}: {st['grasp_adj_z']}"
            assert st["place_xy"] == (260.0, 120.0), "放置点不该被这几毫米碰到"
            assert cam == cam_before, f"记忆库被就地改脏了: {cam}"
            assert act == act_before, f"动作（含模型给的原始坐标）被改脏了: {act}"
            sts[near] = st

        # (a2) XY 补偿不该漏进高度 —— 四个角的**高度推导**里唯一该动的是表里明写的
        #      那一档 Z（现在只有 P1、P2）。拿"没补的那份"（空 world）比:
        #      ★ 高度是拿挪过的 gx/gy 一路算出来的，方向和值错了会连带把高度带偏；
        #        所以这里不只比"四个角互相一样"，还要比"和没补的那份一样"。
        for near, st in sts.items():
            dz = GRASP_OFFSET_Z_MM.get(near, 0.0)
            # 抓取高度: 只差表里那一档 Z；别的角 dz=0 → 一个数都不许动。
            assert st["grasp_z"] == plain["grasp_z"] + dz, \
                f"{near}: 抓取高度只该差表里那一档 Z（{dz}）: {st['grasp_z']}"
            # 抓点上空是从 grasp_z 推的，必须一起走 —— 不是"顺便"，是它的定义。
            assert st["grasp_hover"] == plain["grasp_hover"] + dz, \
                f"{near}: 抓点上空该跟着 grasp_z 一起走: {st['grasp_hover']}"
            # ★ 放置高度**一定**不许动: 这一档是"抓不到底"，跟放下去没关系。
            assert st["place_z"] == plain["place_z"], \
                f"{near}: 放置高度不该跟着动: {st['place_z']}"
            # 搬运高度在两头上空之间取大 —— 空世界里没有障碍，就是这个关系。
            assert st["travel_z"] == max(st["grasp_hover"], st["place_hover"]), \
                f"{near}: 空世界里搬运高度就该是两头上空取大: {st['travel_z']}"

        # (a2b) ★ 高度补偿表本身: 只认表里列的角，没列的一律 0.0（不补）。
        #      现在只有 P1、P2 —— 谁以后加角码，这里会先把"没给值"炸出来。
        assert set(GRASP_OFFSET_Z_MM) <= set(WORLD_NEAR_CODES), \
            f"高度补偿表里有不认识的角码: {sorted(GRASP_OFFSET_Z_MM)}"
        assert GRASP_OFFSET_Z_MM == {WORLD_NEAR_P1: -3.0, WORLD_NEAR_P2: -3.0}, \
            f"这一档 Z 是用户实测定的，要改先重新量: {GRASP_OFFSET_Z_MM}"
        for near in (WORLD_NEAR_P1, WORLD_NEAR_P2):
            assert sts[near]["grasp_adj_z"] == -3.0, sts[near]["grasp_adj_z"]
        for near in (WORLD_NEAR_P3, WORLD_NEAR_P4):
            assert sts[near]["grasp_adj_z"] == 0.0, \
                f"{near} 没在高度表里，就该是 0.0: {sts[near]['grasp_adj_z']}"

        # (a3) 判不出角码（少二维码 → 没有这个键）→ 也不许挪。判据缺了要保守，
        #      不能"默认当某一个角" —— 那等于把一个赌注藏在默认值里。
        unknown = w(rec(near=None))
        st_u = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, unknown)
        assert st_u["grasp_adj"] is None and st_u["grasp_adj_near"] is None \
            and st_u["grasp_adj_z"] == 0.0 \
            and st_u["grasp_xy"] == (210.0, 150.0), \
            f"角码判不出来时不该补: {st_u['grasp_xy']}"
        assert st_u["grasp_z"] == plain["grasp_z"], \
            f"判不出角码时高度也不许动: {st_u['grasp_z']}"

        # (b) 机械臂确认过的坐标（什么戳都没有）→ 一点都不许挪
        st2 = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, w(rec(src=None, near=None)))
        assert st2["grasp_adj"] is None and st2["grasp_adj_near"] is None \
            and st2["grasp_adj_z"] == 0.0 \
            and st2["grasp_xy"] == (210.0, 150.0), st2
        assert st2["grasp_z"] == plain["grasp_z"], \
            f"机械臂确认过的坐标不该补高度: {st2['grasp_z']}"

        # (c) ★★ **没有 src 戳**（= 机械臂已经碰过这块方块）→ XY **一个毫米都不补**，
        #     哪怕 near 戳还留着。这是用户 2026-09-26 实测后让撤销"每次都补"的那条:
        #     这几毫米补的是**相机坐标**的系统偏差，机械臂自己确认过的落点**没有**这个
        #     偏差 —— 再补一次等于把它推离真实位置 5mm（现象: "抓过之后再抓就抓不住"）。
        st_arm = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER,
                             w(rec(near=WORLD_NEAR_P4, src=None)))
        assert st_arm["grasp_adj"] is None and st_arm["grasp_adj_near"] is None \
            and st_arm["grasp_xy"] == (210.0, 150.0), \
            f"机械臂碰过的方块不该再补 XY: {st_arm}"

        # (d) ★★ 高度按「**此刻**在哪儿」判，和 near 戳**分开**算:
        #     near=P4 但人在 P1 那一带 → 高度按 P1 补 −3mm（XY 那一档此时按 P4 补，
        #     因为这条还是**相机刚给的**）。两套判据互不影响。
        moved = w(rec(x=210.0, y=150.0, near=WORLD_NEAR_P4), CODES)
        st_m = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, moved)
        assert st_m["grasp_adj"] == GRASP_OFFSET_MM[WORLD_NEAR_P4], st_m["grasp_adj"]
        assert st_m["grasp_adj_near"] == WORLD_NEAR_P4, st_m["grasp_adj_near"]
        assert st_m["grasp_adj_z"] == -3.0 and st_m["grasp_adj_z_near"] == WORLD_NEAR_P1, \
            f"高度该按它此刻在哪儿（P1）判，不是 near 戳（P4）: {st_m}"

        # (d1b) ★★ 这一条最要紧: 机械臂**已经碰过**（没有 src 戳）的方块，XY 不补了，
        #      但**高度照样补** —— 它此刻人在 P1 那一带。两档的闸门**不一样**:
        #      XY 认"是不是相机刚给的"，高度认"此刻在哪儿"。
        #      用户原话的后半句:「所有在P1附近的物体，**不管是一开始[在]还是后来被
        #      移过去的**，Z都减少3mm」—— 这一句就是这条。
        st_arm2 = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER,
                              w(rec(x=210.0, y=150.0, near=WORLD_NEAR_P4, src=None), CODES))
        assert st_arm2["grasp_adj"] is None and st_arm2["grasp_xy"] == (210.0, 150.0), \
            f"机械臂碰过的方块 XY 不该补: {st_arm2}"
        assert st_arm2["grasp_adj_z"] == -3.0 \
            and st_arm2["grasp_adj_z_near"] == WORLD_NEAR_P1 \
            and st_arm2["grasp_z"] == plain["grasp_z"] - 3.0, \
            f"搬到 P1 那一带之后，高度这一档照样要补（它不看 src 戳）: {st_arm2}"
        assert st_m["grasp_z"] == plain["grasp_z"] - 3.0, st_m["grasp_z"]
        # 反过来: 老家在 P1、但此刻已经被挪到 P2 那一带 → 高度**不补**（不该白扎 3mm）。
        act_far = act_at(300, 150)
        st_f = plan_action(act_far, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER,
                           w(rec(x=300.0, y=150.0, near=WORLD_NEAR_P1), CODES))
        assert st_f["grasp_adj_near"] == WORLD_NEAR_P1 \
            and st_f["grasp_adj"] == GRASP_OFFSET_MM[WORLD_NEAR_P1], st_f
        assert st_f["grasp_adj_z"] == 0.0 and st_f["grasp_adj_z_near"] == WORLD_NEAR_P2, \
            f"老家在 P1、人已经在 P2 那一带，就不该再扎下去: {st_f}"
        # 没判出老家（没有 near 戳）也不影响高度那一档 —— 两套判据是独立的。
        st_n = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER,
                           w(rec(near=None), CODES))
        assert st_n["grasp_adj"] is None and st_n["grasp_xy"] == (210.0, 150.0), st_n
        assert st_n["grasp_adj_z"] == -3.0 and st_n["grasp_adj_z_near"] == WORLD_NEAR_P1, \
            f"XY 没判出老家，高度照样该按此刻在哪儿判: {st_n}"

        # (d2) ★ 码的位置**不全 / 记坏了** → 不许猜: 退回按 near 戳判（"不知道它挪没挪，
        #      就按老家算"）。少一个码就可能把 P1 的 −3mm 补到隔壁那个角上。
        partial = {c: v for c, v in CODES.items() if c != WORLD_NEAR_P2}
        st_p = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER,
                           w(rec(near=WORLD_NEAR_P4), partial))
        assert st_p["grasp_adj_z"] == 0.0 and st_p["grasp_adj_z_near"] == WORLD_NEAR_P4, \
            f"码不齐时该退回按老家判（P4 → 不补），不该猜: {st_p}"
        broken = dict(CODES, **{WORLD_NEAR_P3: [1.0]})       # 缺一个分量
        st_b = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER,
                           w(rec(near=WORLD_NEAR_P4), broken))
        assert st_b["grasp_adj_z"] == 0.0 and st_b["grasp_adj_z_near"] == WORLD_NEAR_P4, \
            f"码的位置记坏了就当判不出来，别猜: {st_b}"

        # (e) 计划说明里得能看出「记录里是哪个、实际去哪个、是**哪个角**」——
        #     不写出来就成了"计划里的数字和识别结果对不上"；不写角码，用户就没法
        #     核对补的方向（四个角的值不一样，只说"补了 5mm"看不出补的是哪一个）。
        #     ★ 高度那一档同理: 上面印的「降到 Z=」是**已经补过**的数，不说明来历，
        #       用户拿它跟公式对不上会以为高度算错了。
        for near, st in sts.items():
            ax, ay = GRASP_OFFSET_MM[near]
            dz = GRASP_OFFSET_Z_MM.get(near, 0.0)
            text = describe_plan([st], T)
            assert f"**{near}**" in text, f"{near}: 说明里没写是哪个角:\n{text}"
            assert "记忆库里 (210.00, 150.00)" in text \
                and f"实际去 ({210.0 + ax:.2f}, {150.0 + ay:.2f})" in text, \
                f"{near}: 没讲清这一抓挪去哪了:\n{text}"
            if dz:
                assert f"高度**也补了 {dz:+.1f}mm" in text, \
                    f"{near}: 补了高度却没在说明里写出来:\n{text}"
                # 印出来的公式值必须是**没补的**那个数 —— 否则这一行只是把
                # 补过的数抄一遍，等于没说。
                assert f"公式算是 Z={st['grasp_z'] - dz:.2f}" in text, \
                    f"{near}: 高度那行没讲清原来的数是多少:\n{text}"
            else:
                assert "高度**也补了" not in text, \
                    f"{near}: 没补高度就别冒出高度补偿的说明:\n{text}"
        for other, why in ((st_u, "角码判不出来"), (st2, "机械臂碰过了")):
            assert "记忆库里" not in describe_plan([other], T), \
                f"{why}：没挪的时候不该冒出补偿说明"
        # ★ 老家和"此刻"是两个角时，说明里**两个名字都得有** —— 否则用户看到
        #   "按 P4 补"，再看高度那行写的还是 P4，就以为高度也用错了判据。
        txt_m = describe_plan([st_m], T)
        assert "**P4**" in txt_m and "**P1**" in txt_m \
            and "高度**也补了 -3.0mm" in txt_m, \
            f"老家(P4)和此刻(P1)不同时，说明里必须两个角都写出来:\n{txt_m}"
        # XY 没补、只补了高度时，高度那行**也**要印出来（它不在 XY 那个分支里）。
        txt_n = describe_plan([st_n], T)
        assert "高度**也补了 -3.0mm" in txt_n and "**P1**" in txt_n \
            and "记忆库里" not in txt_n, \
            f"只有高度补了的时候，也得说明白:\n{txt_n}"

        # (f) ★★ 抓过之后（refresh_world_state 整条重写）: **两个戳一个都不留**，
        #     记录只剩 x/y/z_level ⇒ 再抓它 **XY 一个毫米都不补**（src 没了 = 这条
        #     坐标是机械臂确认过的，本来就没有相机那个系统偏差）。
        #     ★ 高度那一档**照样补**: 它不看 src，只看这条记录**现在**的 x/y 落在
        #       哪个角码旁边 —— 落点 (210,150) 在 P1 那一带 → 仍然 −3mm。
        ws = w(rec(near=WORLD_NEAR_P4), CODES)
        refresh_world_state(ws, [{"obj": "red", "place_xy": (210.0, 150.0),
                                  "place_level": 0}], [0])
        assert WORLD_SRC_KEY not in ws["red"], f"抓过之后来源戳必须消失: {ws['red']}"
        assert WORLD_NEAR_KEY not in ws["red"], \
            f"抓过之后 near 戳也该消失（它是「相机那一眼」的属性）: {ws['red']}"
        assert ws["red"] == {"x": 210.0, "y": 150.0, "z_level": 0}, ws["red"]
        # ★ 世界状态里那条保留键（码的位置）不许被 refresh 碰掉 —— 后面每一抓都还要用它。
        assert ws.get(WORLD_CODES_KEY) == CODES, f"保留键被改脏了: {ws.get(WORLD_CODES_KEY)}"
        st5 = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, ws)
        assert st5["grasp_adj"] is None and st5["grasp_xy"] == (210.0, 150.0), \
            f"抓过之后再抓它，XY 就不该再补了（那几毫米是相机坐标的偏差）: {st5}"
        assert st5["grasp_adj_z"] == -3.0 and st5["grasp_adj_z_near"] == WORLD_NEAR_P1, \
            f"它现在在 P1 那一带，高度就该按 P1 补（这一档不看 src 戳）: {st5}"
        # 再抓它一次，结果**一模一样** —— 这两档都不会一次一次叠上去。
        st6 = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, ws)
        assert (st6["grasp_xy"], st6["grasp_z"]) == (st5["grasp_xy"], st5["grasp_z"]), \
            f"同一个世界状态下重算必须得到同一个结果: {st6}"
    ck("抓取补偿两套判据: XY **只在相机刚拍完的那一次抓取**补（按 near 戳那个角，"
       "机械臂碰过之后一毫米都不补）、高度**每一抓都补**（按它此刻离哪个角码最近，"
       "P1/P2 两档 −3mm，搬到 P1 的也算）；码不齐/角码判不出就不补，放置高度纹丝不动",
       t_grasp_offset)

    # 5n. ★ 记忆库顶层的**保留键**（paths.WORLD_CODES_KEY = "_codes"）**不是**一条方块
    #     记录。它和四条方块记录睡在同一个字典里，于是「遍历记忆库」的每一处都得按
    #     约定跳过它 —— 现在有三处几何要遍历（path_blockers / _zones_on_path /
    #     early_descent），全走 cube_records 这一个口子（见它的说明）。
    #     ★ 这不是假想的坑: _codes 里**没有 "x"**，几何函数第一句 float(b["x"]) 就
    #       KeyError。而且**只有带了 _codes 的记忆库才踩得到** —— 那个键要相机写过
    #       一次桌面才有，所以早先那些用空记忆库 / 手写记忆库的自检全照绿，
    #       一到真机才炸。
    #     ★ 光断言 cube_records 自己的行为**还不够**: 谁绕过它、自己 world.items()，
    #       那条断言抓不到。所以这里拿一份**带保留键 + 真有一块挡路的方块**的记忆库
    #       把 plan_action 整条链路跑一遍 —— 它内部正好走全那三处几何。
    def t_reserved_keys():
        codes = {WORLD_NEAR_P1: [100.0, 100.0], WORLD_NEAR_P2: [400.0, 100.0],
                 WORLD_NEAR_P3: [100.0, 400.0], WORLD_NEAR_P4: [400.0, 400.0]}
        # ★ 必须**真摆一块挡路的**（blue 顶到第 1 层）: 记忆库里只有手上那一块、
        #   或干脆是空的时，那三处几何会提前 return —— 自检就等于没跑。
        world = {"red": {"x": 210.0, "y": 150.0, "z_level": 0},
                 "blue": {"x": 260.0, "y": 150.0, "z_level": 1},
                 WORLD_CODES_KEY: codes}
        assert [n for n, _ in cube_records(world)] == ["red", "blue"], \
            f"cube_records 把保留键当成方块了: {[n for n, _ in cube_records(world)]}"
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 150, "z_level": 1}}
        st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER, world)
        assert [b[0] for b in st["blockers"]] == ["blue"], \
            f"带保留键的记忆库里，挡路的只该是 blue: {st['blockers']}"
    ck("记忆库顶层的保留键 _codes 不被当成方块（三处几何都跳过它，"
       "带保留键的记忆库照样算得出障碍）", t_reserved_keys)

    # 6. 限位外的坐标必须在**连臂前**被拦下
    def t_unreachable():
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 40, "y": 0, "z_level": 0}}     # x=40 < 下限 50
        st = plan_action(act, T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        probs = preflight([st], lim, T)
        assert probs and "放置点" in probs[0], f"该拦下却放行: {probs}"
    ck("够不着的坐标在连臂前就被拦下", t_unreachable)

    # 7. 两个方块落到同一个位置 → 要报出来（DeepSeek 不知道空位有没有被占）
    def t_collide():
        mk = lambda o: {"obj": o, "grasp": {"x": 210, "y": 150, "z_level": 0},
                        "place": {"x": 260, "y": 120, "z_level": 0}}
        plan = build_plan([mk("red"), mk("blue")], T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        probs = preflight(plan, lim, T)
        assert any("撞车" in p for p in probs), f"该报撞车: {probs}"
    ck("两个方块落到同一处 → 报出撞车", t_collide)

    # 8. 丢步报警必须拦住，且**不提供任何后门**
    def t_lost_step():
        fake = _FakeDobot(alarms=[0] * 16)
        fake.alarms[10] = 1 << 2          # 报警号 10*8+2 = 82 = 0x52 → 丢步
        ok, why = arm_ready(_FakeApi(), fake, clear_alarms=True, log=lambda *a: None)
        assert not ok, "丢步报警本该拦住"
        assert "回零" in why, f"该提示回零: {why}"
        assert fake.cleared == 0, "★ 丢步报警绝不能被清除掉"
    ck("丢步报警拦下且拒绝清除（没有 force 后门）", t_lost_step)

    # 9. 上电复位报警(0x00)是良性，按固件设计自动清
    def t_benign():
        fake = _FakeDobot(alarms=[0] * 16)
        fake.alarms[0] = 1 << 0           # 0x00
        ok, _ = arm_ready(_FakeApi(), fake, clear_alarms=False, log=lambda *a: None)
        assert ok and fake.cleared == 1, "上电复位该被自动清掉"
    ck("上电复位报警按设计自动清", t_benign)

    # 10. 走一遍完整动作: 抓 → 抬 → 平移 → 放 → 抬，且吸盘顺序正确
    def t_execute():
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 1}}
        plan = build_plan([act], T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        st = plan[0]
        fake, fapi = _FakeDobot(), _FakeApi()
        class A:
            pump_s, mode = 0.0, "movl"
        rc = execute(fapi, fake, plan, A(), lim, log=lambda *a: None)
        assert rc == 0, f"execute 返回 {rc}"
        pts = fake.sent
        zs = [p["z"] for p in pts]
        assert zs[0] == st["grasp_hover"], "第 1 步该悬停在**抓取点**上方"
        assert zs[1] == st["grasp_z"], "第 2 步该降到抓取 Z"
        assert fapi.suction == [True, False], f"吸盘顺序该是 开→关，实为 {fapi.suction}"
        assert zs[-2] == st["place_z"], "倒数第二步该降到放置 Z"
        assert zs[-1] == st["place_hover"], "最后一步该在放点上空抬起离开"

        # 吸着方块期间（开真空之后、关真空之前）走过的每一个路点:
        #   · XY 只允许出现在 抓点 / 升高点 / 放点 这三处，不许有额外的中间点
        #   · 高度只允许 ≥ 抓点悬停，绝不许一边吸着一边往下降
        allowed = {st["grasp_xy"], st["place_xy"]}
        if st["rise_xy"]:
            allowed.add(st["rise_xy"])
        if st["sink_xy"]:
            allowed.add(st["sink_xy"])
        for p in pts[2:-1]:
            assert (p["x"], p["y"]) in allowed, f"多出一个路点 ({p['x']:.1f}, {p['y']:.1f})"
            # ★ 「下降放置」那一步是**故意**往下降的，要放它过去。
            #   这里原来只挖掉最后一个点，没挖掉倒数第二个 —— 于是这条自检
            #   一跑就红。它不是发现了 bug，是判据写漏了一种**合法**的点。
            if p["z"] <= st["place_z"] + 1e-9:
                continue
            assert p["z"] >= st["grasp_hover"] - 1e-9, f"吸着方块还在下降 Z={p['z']:.2f}"

        # ★ 本次改动的核心: 先在**低处**走到升高点，在那儿竖着升到搬运高度，
        #   最后一段才在搬运高度上横着到放点。绝不能反过来在抓点原地升。
        if st["rise_xy"]:
            at_rise = [p for p in pts if (p["x"], p["y"]) == st["rise_xy"]]
            assert at_rise[0]["z"] == st["grasp_hover"], "该先在低处走到升高点"
            assert at_rise[-1]["z"] == st["travel_z"], "到了升高点才竖着升到搬运高度"
            # ★ 升是**分段**的（每 RISE_STEP_MM 一步，见 climb()），所以升高点上
            #   不止两个点。这里钉住三件事: 中途不许回头下降、不许超过目标高度、
            #   终点就是搬运高度。写成 at_rise[1] == travel_z 会被分段一改就误报。
            zs_rise = [p["z"] for p in at_rise]
            assert zs_rise == sorted(zs_rise), f"升的过程中高度不该回落: {zs_rise}"
            assert max(zs_rise) <= st["travel_z"] + 1e-9, \
                f"不该升过搬运高度: {max(zs_rise)} > {st['travel_z']}"
            assert any(p["z"] > st["grasp_hover"] + 1e-9 for p in at_rise) or \
                st["travel_z"] <= st["grasp_hover"] + 1e-9, \
                "升高点上该真的升高了（除非本来就不需要升）"
        first_place = [p for p in pts if (p["x"], p["y"]) == st["place_xy"]][0]
        assert first_place["z"] == st["travel_z"], \
            f"横着到放点必须在搬运高度，实为 {first_place['z']:.2f}"
    ck("完整走一遍: 低处走开→升高→平移→下降→放→抬（不在抓点原地升）", t_execute)

    # 11. 中途失败必须把真空泵关掉（否则吸着方块停在那儿）
    def t_fail_closes_pump():
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        plan = build_plan([act], T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        fake, fapi = _FakeDobot(), _FakeApi()
        fake.refuse = {dType_PTPMOVL()}          # 让第一个 PTP 就被拒
        class A:
            pump_s, mode = 0.0, "movl"
        rc = execute(fapi, fake, plan, A(), lim, log=lambda *a: None)
        assert rc == 1, "被拒该返回失败"
        assert fapi.suction == [], "一步都没走成，不该开真空"
    ck("运动被拒 → 返回失败且不开真空", t_fail_closes_pump)

    # 11b. ★ 已经吸住之后再失败 —— 泵必须被记为「还开着」，收尾才关得掉。
    #      这一条是整份代码里最要命的一条: 漏了它，脚本会吸着方块停在那儿。
    def t_pump_on_then_fail():
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 0}}
        plan = build_plan([act], T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        # 前 2 步（悬停、下降）成功 → 开真空 → 第 3 步（抬起）失败
        fake, fapi = _RefuseAfter(n=2), _FakeApi()
        holder = {}

        class A:
            pump_s, mode = 0.0, "movl"
        rc = execute(fapi, fake, plan, A(), lim, log=lambda *a: None, holder=holder)
        assert rc == 1, f"该失败，实为 {rc}"
        assert fapi.suction == [True], f"失败前该开过真空，实为 {fapi.suction}"
        assert holder["held"] is True, "★ 泵还开着 —— 没标记的话收尾不会关泵"
    ck("吸住之后中途失败 → 标记「泵还开着」（收尾据此关泵）", t_pump_on_then_fail)

    # 11c. ★★ 上升**分段走、每段核到位**；够不到就当场停手，绝不继续往上顶。
    #      这是 2026-09-18 那次报警的正面修法: 那次整段一次升到 Z=68.51，
    #      实测停在 64.54、差 3.97mm，报警出来的时候已经晚了。
    def t_climb_stepped():
        # 放点在第 1 层 → 搬运高度比抓点悬停高 26mm，一定不止一段
        act = {"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 1}}
        plan = build_plan([act], T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        st = plan[0]
        span = st["travel_z"] - st["grasp_hover"]
        assert span > RISE_STEP_MM, \
            f"★ 这条自检要的是「不止一段」的情形，可升高度只有 {span:.1f}mm"

        class A:
            pump_s, mode = 0.0, "movl"

        # (a) 一路够得着: 分几段、每段不超过 RISE_STEP_MM、最后正好落在搬运高度
        fake, fapi = _FakeDobot(), _FakeApi()
        rc = execute(fapi, fake, plan, A(), lim, log=lambda *a: None)
        assert rc == 0, f"一路够得着就该成功，实为 {rc}"
        at_rise = [p["z"] for p in fake.sent if (p["x"], p["y"]) == st["rise_xy"]]
        assert at_rise[0] == st["grasp_hover"], "第一下该是**贴着桌面**走到升高点"
        assert at_rise[-1] == st["travel_z"], f"最后该正好升到搬运高度: {at_rise}"
        steps = [b - a2 for a2, b in zip(at_rise, at_rise[1:])]
        assert len(steps) >= 2, f"可升 {span:.1f}mm 却只走了一段: {at_rise}"
        assert all(0.0 < s <= RISE_STEP_MM + 1e-9 for s in steps), \
            f"每段都该是 (0, {RISE_STEP_MM:.0f}]mm: {steps}"

        # (b) 够不着: 只够到「悬停 + 一档」，再往上纹丝不动
        #     → 必须当场返回失败，而且**只允许**多发出一条更高的指令就停手。
        z_stall = st["grasp_hover"] + RISE_STEP_MM
        fake2, fapi2 = _ZLimit(z_stall), _FakeApi()
        rc2 = execute(fapi2, fake2, plan, A(), lim, log=lambda *a: None)
        assert rc2 == 1, "★ 升不动了却报成功 —— 那才会让方块停在半空"
        above = [p["z"] for p in fake2.sent if p["z"] > z_stall + 1e-9]
        assert len(above) == 1, \
            f"★ 该在够不到之后只多发 1 条指令就停手，实为 {above}（在硬顶）"
        assert not [p for p in fake2.sent if (p["x"], p["y"]) == st["place_xy"]], \
            "★ 升不上去却还继续往放点走 —— 那是吸着方块去撞障碍"
        assert fapi2.suction == [True], \
            "该在**吸着方块**的时候失败（不然这条自检测的不是危险情形）"
    ck("竖着升分段走、每段核到位；够不到就当场停手不硬顶",
       t_climb_stepped)

    # 12. 空计划（方块名不存在）不该动臂
    def t_empty():
        assert build_plan([], T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER) == []
        assert "空计划" in describe_plan([], T)
    ck("空计划（方块不存在）不动臂", t_empty)

    # 13. ★ 刷新记忆库: 「放到旁边」层数是 0，「摞上去」才是 1。
    #     这是《方案.md》105-110 行那句话唯一会出错的地方，专门钉住。
    def t_refresh_level():
        def fresh():
            return {"red": {"x": 210.0, "y": 150.0, "z_level": 0},
                    "green": {"x": 220.0, "y": 200.0, "z_level": 0}}

        # ① 放到红色**旁边**（这里取 y=180；左右哪边都一样，本条只验层数）
        #    → 落在桌面上，层数必须还是 0
        ws = fresh()
        plan = build_plan([{"obj": "green", "grasp": {"x": 220, "y": 200, "z_level": 0},
                            "place": {"x": 210, "y": 180, "z_level": 0}}],
                          T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        notes = refresh_world_state(ws, plan, [0])
        assert ws["green"] == {"x": 210.0, "y": 180.0, "z_level": 0}, ws["green"]
        assert ws["red"] == {"x": 210.0, "y": 150.0, "z_level": 0}, "参照物不该被改"
        naive = ws["red"]["z_level"] + 1        # 《方案.md》写的那种算法
        assert naive == 1, "★ 若这里不是 1，说明这条自检失去对照意义了"
        assert "层0" in notes[0], notes

        # ② 摞到红色**上面** → 层数 1
        plan2 = build_plan([{"obj": "green", "grasp": {"x": 210, "y": 180, "z_level": 0},
                             "place": {"x": 210, "y": 150, "z_level": 1}}],
                           T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        notes2 = refresh_world_state(ws, plan2, [0])
        assert ws["green"] == {"x": 210.0, "y": 150.0, "z_level": 1}, ws["green"]
        assert "层1" in notes2[0], notes2
    ck("刷新记忆库: 放旁边→层0（不是「参照物+1」），摞上去→层1", t_refresh_level)

    # 13b. ★ 左右/前后/间距改过之后，旧计划不许被**静默复用**。
    #      坐标里看不出来（±50 都是正常数字），只能靠指纹拦。
    def t_stale_rules():
        now = brain.rules_fingerprint()
        assert plan_rules_mismatch({"brain_rules": now}) is None, "同规则的计划不该被拦"
        # 换个方向后的指纹（模拟「改了表、计划还是旧的」）
        g = brain.SIDE_GAP_MM
        old = now.replace(f"right:0,-{g}", f"right:0,{g}")
        assert old != now, "★ 这条测试失去对照意义了: 指纹里找不到 right"
        # ★ 间距单独改过也要拦: 旧计划的 place 坐标长得完全合法（就是老间距那个值），
        #   不拦就会照老间距摆 —— 两块面贴面。这条钉的就是 30→50 这次改动。
        old_gap = now.replace(f"gap:{g}", "gap:30")
        assert old_gap != now, "★ 这条测试失去对照意义了: 指纹里找不到 gap"
        assert plan_rules_mismatch({"brain_rules": old_gap}), \
            "间距改了却放行 —— 会按老间距（面贴面）摆"
        assert plan_rules_mismatch({"brain_rules": old}), "方向变了却放行 —— 会重跑老方向"
        assert plan_rules_mismatch({}), "没有指纹的旧计划必须拦下（不知道按哪套算的）"
        # 光看坐标是发现不了的: 两份计划的 place 长得一模一样
        p_old = {"brain_rules": old, "actions": [{"place": {"y": 65.65}}]}
        p_new = {"brain_rules": now, "actions": [{"place": {"y": 65.65}}]}
        assert p_old["actions"] == p_new["actions"] and plan_rules_mismatch(p_old)
    ck("方向规则改过 → 旧计划拒绝复用（不静默重跑）", t_stale_rules)

    # 14. 只刷新**真的放下去了**的那些；中止时吸着的那块绝不能瞎记
    def t_refresh_partial():
        ws = {"red": {"x": 210.0, "y": 150.0, "z_level": 0},
              "green": {"x": 220.0, "y": 200.0, "z_level": 0}}
        plan = build_plan(
            [{"obj": "red", "grasp": {"x": 210, "y": 150, "z_level": 0},
              "place": {"x": 260, "y": 120, "z_level": 0}},
             {"obj": "green", "grasp": {"x": 220, "y": 200, "z_level": 0},
              "place": {"x": 180, "y": 180, "z_level": 0}}],
            T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)

        # 只有第 1 个放下了（第 2 个还没走到）→ 第 2 个必须原样不动
        refresh_world_state(ws, plan, [0])
        assert ws["red"] == {"x": 260.0, "y": 120.0, "z_level": 0}, ws["red"]
        assert ws["green"] == {"x": 220.0, "y": 200.0, "z_level": 0}, \
            "★ 没放下的那块不能被写成计划里的目标位置"

        # 中止时吸盘上还吸着 green → 它的旧记录是错的，必须**删掉**。
        # ★ 不许"留着不管": 留着的旧位置会让下一次指令对着空地抓，而且不报错。
        ws2 = {"red": {"x": 210.0, "y": 150.0, "z_level": 0},
               "green": {"x": 220.0, "y": 200.0, "z_level": 0}}
        notes = refresh_world_state(ws2, plan, [0], holding=True)
        assert "green" not in ws2, f"★ 吸着的那块必须从记忆库删掉，实为 {ws2}"
        assert "red" in ws2, "别的方块不该受影响"
        joined = "".join(notes)
        assert "green" in joined and "没人知道" in joined, notes
    ck("中止时只记「已放好」的；吸着的那块从记忆库删掉（不留错位置）",
       t_refresh_partial)

    # 15. execute 要如实记下「第几步真的放下了」——含「抬不起来但确实已放下」
    def t_execute_progress():
        def mk(o):
            return {"obj": o, "grasp": {"x": 210, "y": 150, "z_level": 0},
                    "place": {"x": 260, "y": 120, "z_level": 0}}
        plan = build_plan([mk("red"), mk("green")], T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)

        class A:
            pump_s, mode = 0.0, "movl"

        prog = {}
        rc = execute(_FakeApi(), _FakeDobot(), plan, A(), lim,
                     log=lambda *a: None, progress=prog)
        assert rc == 0, f"execute 返回 {rc}"
        assert prog["placed"] == [0, 1], prog
        assert prog["total"] == 2, prog

        # ★ 第 2 个动作的"抬起离开"失败 —— 但它的方块**已经放下了**（关真空在前）。
        #   记忆库必须记它，否则那块方块会被永久遗忘在原地。
        #   ★ 次数是**数出来的**，不是拍的: 这里原来写死 6，按的是老版「每个动作 4 步」
        #     的数；现在每个动作 8 步（悬停/下降/抬起/移到升高点/竖升/平移/下降放置/
        #     抬起离开），整份计划 16 步。硬写数字的真正问题是: 动作形状一改，数错
        #     一步，测的就不是"放完之后才失败"而是"还没放下就失败"，整条自检看着还
        #     是绿的、其实已经不在测它该测的东西了。所以先跑一遍**成功**的，
        #     拿它真的发了几条指令当基准。
        base = _FakeDobot()
        rc0 = execute(_FakeApi(), base, plan, A(), lim, log=lambda *a: None)
        assert rc0 == 0, f"基准那遍该成功，实为 {rc0}"
        n_last = len(base.sent)
        assert n_last >= 8, f"两个动作至少 8 条指令，只发了 {n_last} 条 —— 基准不对"
        prog2 = {}
        rc2 = execute(_FakeApi(), _RefuseAfter(n=n_last - 1), plan, A(), lim,
                      log=lambda *a: None, progress=prog2)
        assert rc2 == 1, f"该失败，实为 {rc2}"
        assert prog2["placed"] == [0, 1], \
            f"★ 关真空之后才失败，两个都该算放下，实为 {prog2}"

        # 反过来: 第 1 个动作**吸着方块、还在低空走**那步就挂了 → 一个都没放下。
        #   （第 4 条指令 = 贴着桌面移到升高点；此刻真空已开、还没到放点。）
        #   ★ 顺手把"这一步真的是吸着的时候挂的"也钉住 —— 不然这条自检随时可能
        #     因为动作形状变了而退化成"根本没吸起来就失败了"，那它就白测了。
        fake3, api3 = _RefuseAfter(n=4), _FakeApi()
        prog3 = {}
        rc3 = execute(api3, fake3, plan, A(), lim, log=lambda *a: None, progress=prog3)
        assert rc3 == 1, f"该失败，实为 {rc3}"
        assert api3.suction == [True], \
            f"★ 这一步本该是吸着方块的时候挂的，实为 {api3.suction}"
        assert prog3["placed"] == [], prog3
    ck("execute 如实回报「哪些方块真的放下了」（含放完才抬不起来的情况）",
       t_execute_progress)

    # 16. 坐标来源优先级，以及「虚拟坐标必须是副本」
    def t_source_precedence():
        import tempfile
        saved = WORLD_STATE_JSON
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "world_state.json"
            tmp.write_text(json.dumps({"red": {"x": 1.0, "y": 2.0, "z_level": 3}}),
                           encoding="utf-8")
            globals()["WORLD_STATE_JSON"] = tmp        # 别去碰真的 output/
            try:
                class A:
                    world_json, virtual = None, False
                st, src, real = load_world_state(A())
                assert st["red"]["z_level"] == 3, f"该读记忆库，实为 {st}"
                assert "记忆库" in src and real, (src, real)

                # --virtual 强制; 而且拿到的必须是**副本**
                A.virtual = True
                st2, src2, real2 = load_world_state(A())
                assert st2 == brain.VIRTUAL_WORLD_STATE, st2
                assert not real2, "★ 虚拟坐标必须被标成「不是真的」—— 否则会被写进记忆库"
                assert "虚拟" in src2, src2
                st2["red"]["x"] = 99999
                assert brain.VIRTUAL_WORLD_STATE["red"]["x"] != 99999, \
                    "★ 虚拟坐标必须是副本 —— 就地改会弄脏 brain 的模块常量"

                # --world-json 优先于记忆库
                other = Path(td) / "cam.json"
                other.write_text(json.dumps({"blue": {"x": 5, "y": 6, "z_level": 1}}),
                                 encoding="utf-8")
                A.virtual, A.world_json = False, str(other)
                st3, src3, real3 = load_world_state(A())
                assert "blue" in st3 and "cam.json" in src3 and real3, (st3, src3)
            finally:
                globals()["WORLD_STATE_JSON"] = saved
    ck("坐标来源优先级: --world-json > 记忆库 > 虚拟（虚拟是副本且标为「非真实」）",
       t_source_precedence)

    # 17. 刷新 + 落盘全程不许碰坏 brain 的虚拟常量（跑两条指令之后仍要原样）
    def t_virtual_not_dirtied():
        before = copy.deepcopy(brain.VIRTUAL_WORLD_STATE)

        class A:
            world_json, virtual = None, True
        ws, _, real = load_world_state(A())
        plan = build_plan([{"obj": "green", "grasp": {"x": 220, "y": 200, "z_level": 0},
                            "place": {"x": 260, "y": 200, "z_level": 1}}],
                          T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
        refresh_world_state(ws, plan, [0])
        assert ws["green"]["z_level"] == 1, ws
        assert brain.VIRTUAL_WORLD_STATE == before, "brain 的虚拟常量被改脏了"
        assert not real, "虚拟坐标不该被当成真的"

        # ★ 虚拟坐标**不许**写进记忆库。用临时路径试，绝不碰真的 output/。
        import tempfile
        saved = WORLD_STATE_JSON
        with tempfile.TemporaryDirectory() as td:
            globals()["WORLD_STATE_JSON"] = Path(td) / "world_state.json"
            try:
                report_world_refresh(ws, plan, {"placed": [0], "total": 1},
                                     holding=False, world_real=False)
                assert not globals()["WORLD_STATE_JSON"].exists(), \
                    "★ 虚拟坐标绝不能被写进记忆库（下一条指令会当真）"
            finally:
                globals()["WORLD_STATE_JSON"] = saved
    ck("虚拟坐标: 不弄脏 brain 常量，也不写进记忆库", t_virtual_not_dirtied)

    # 18. 真坐标的**落盘**那条路必须整条走通（上面两条都只走到「不落盘」的分支）
    #     ★ 为什么专门来一条: 打印路径时用了 p.relative_to(ROOT)，而记忆库一旦
    #       不在项目目录里它就直接抛 ValueError —— 整个刷新把流程炸掉。
    #       之前两条自检都提前 return，把这条分支绕过去了，等于没测。
    def t_persist_branch():
        import tempfile
        import color_vision as cv
        saved_m, saved_cv = WORLD_STATE_JSON, cv.WORLD_STATE_JSON
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "world_state.json"          # ★ 故意放在项目目录之外
            globals()["WORLD_STATE_JSON"] = p
            cv.WORLD_STATE_JSON = p                    # 落盘走的是 color_vision 那份
            try:
                plan1 = build_plan(
                    [{"obj": "yellow", "grasp": {"x": 260, "y": 120, "z_level": 0},
                      "place": {"x": 210, "y": 150, "z_level": 2}}],
                    T, OBJ_H, NET_H, 0.0, DEFAULT_HOVER)
                ws = {"red": {"x": 210.0, "y": 150.0, "z_level": 1}}
                report_world_refresh(ws, plan1, {"placed": [0], "total": 1},
                                     holding=False, world_real=True)
                assert p.exists(), "★ 说好了「已存」，文件却没写出来"
                got = json.loads(p.read_text(encoding="utf-8"))
                assert got["yellow"]["z_level"] == 2, got
                assert got["red"]["z_level"] == 1, "没被碰过的那块必须原样留着"

                # 再刷一次 → 覆盖前要留 .bak（记忆库被一次误跑盖掉就补不回来了）
                report_world_refresh(ws, plan1, {"placed": [0], "total": 1},
                                     holding=False, world_real=True)
                bak = p.with_suffix(".json.bak")
                assert bak.exists(), "★ 覆盖前必须留一份 .bak"
                assert json.loads(bak.read_text(encoding="utf-8"))["yellow"]["z_level"] == 2

                # 项目外的路径也得能打印出来（就是不许抛异常）
                assert str(p) in _rel(p), _rel(p)
            finally:
                globals()["WORLD_STATE_JSON"] = saved_m
                cv.WORLD_STATE_JSON = saved_cv
    ck("真坐标落盘: 记忆库不在项目目录里也照样写得出（含 .bak），不因打印路径炸掉",
       t_persist_branch)

    # 19. --net-h 默认写死 25（用户 2026-09-19: 「不要让我每次都敲一遍」）
    def t_default_net_h():
        assert DEFAULT_NET_H == 25.0, DEFAULT_NET_H
        # ★ 这条只有把解析器抽成 build_argparser() 才测得到 —— 之前它藏在 main()
        #   肚子里，自检根本够不着「默认值到底是多少」这笔账。
        got = build_argparser().parse_args([]).net_h
        assert got == DEFAULT_NET_H, f"不给 --net-h 时默认值是 {got}，应为 {DEFAULT_NET_H}"
        # 显式给的值必须仍然压过默认（写死的是默认，不是把参数焊死）
        assert build_argparser().parse_args(["--net-h", "31"]).net_h == 31.0
        # 默认值得**真能用**: 有它，叠放才不算不出来
        assert abs(nozzle_z(T, 1, OBJ_H, DEFAULT_NET_H)
                   - (T + OBJ_H + DEFAULT_NET_H)) < 1e-9
        # ★ 写死的只是**命令行默认值**，库函数对 None 的拒绝一步没让
        #   （否则「没量净高就叠放」会从"拒绝"变成"悄悄按 25 算"）
        try:
            nozzle_z(T, 1, OBJ_H, None)
        except PlanError:
            pass
        else:
            raise AssertionError("★ 库函数不该因为命令行有了默认值就放宽 net_h=None 这条")
    ck("--net-h 默认写死 25（显式给仍能覆盖；库函数对 None 的拒绝不变）",
       t_default_net_h)

    # 20. run_instruction: 没指令只返回码，不许 SystemExit（否则交互会被带走）
    def t_run_instruction_guard():
        class A:
            instruction = ""
        assert run_instruction(A()) == 2
    ck("run_instruction: 没指令 → 退出码 2（不再 ap.error 直接 SystemExit）",
       t_run_instruction_guard)

    # 21. 交互循环: 空行忽略 / 坏指令不带走整场 / quit 收工 / reset-world 只清一次
    def t_interactive_loop():
        import io

        class FakeIn(io.StringIO):
            def isatty(self):
                return True         # 装成交互式终端，否则会被那道 TTY 闸口挡住

        seen = []

        def fake_run(a):
            # 记下**每条指令看到的 reset_world** —— 真 run_instruction 清完标志
            # 才会走后面的流程（见它的注释），这里照做，验的是 interactive 那侧:
            # 它必须**共用同一个 args**、不许每条都把 --reset-world 又置回 True。
            seen.append((a.instruction, a.reset_world))
            a.reset_world = False
            if a.instruction == "坏":
                raise SystemExit(2)          # 模拟「坐标文件坏了」那条 sys.exit 的路
            return 2 if a.instruction == "没做成" else 0

        class A:
            go, reset_world, instruction = False, True, None

        old_in, old_out = sys.stdin, sys.stdout
        old_run = globals()["run_instruction"]
        sys.stdin, sys.stdout = FakeIn("一\n\n坏\n没做成\nquit\n"), io.StringIO()
        globals()["run_instruction"] = fake_run
        try:
            rc = interactive(A())
        finally:
            sys.stdin, sys.stdout = old_in, old_out
            globals()["run_instruction"] = old_run
        assert rc == 0, f"交互正常收工该是 0，实为 {rc}"
        assert seen == [("一", True), ("坏", False), ("没做成", False)], seen
    ck("交互模式: 空行忽略、坏指令不带走整场、quit 收工、--reset-world 只清一次",
       t_interactive_loop)

    # 22. `-i --go "第一句"` 里命令行上那句必须**先跑**，不许被静默丢掉
    def t_interactive_initial():
        import io

        class FakeIn(io.StringIO):
            def isatty(self):
                return True

        seen = []

        def fake_run(a):
            seen.append(a.instruction)
            return 0

        class A:
            go, reset_world, instruction = False, False, "开头那句"

        old_in, old_out = sys.stdin, sys.stdout
        old_run = globals()["run_instruction"]
        sys.stdin, sys.stdout = FakeIn("后面那句\nquit\n"), io.StringIO()
        globals()["run_instruction"] = fake_run
        try:
            rc = interactive(A())
        finally:
            sys.stdin, sys.stdout = old_in, old_out
            globals()["run_instruction"] = old_run
        assert rc == 0, rc
        assert seen == ["开头那句", "后面那句"], seen
    ck('交互模式: 命令行上带的那句先跑（`-i --go "第一句"` 不会被静默丢掉）',
       t_interactive_initial)

    # 23. 交互模式在管道/重定向里跑 → 明确拒绝，别假装在那儿读指令
    def t_interactive_not_tty():
        import io

        class FakeIn(io.StringIO):
            def isatty(self):
                return False

        old_in, old_out = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = FakeIn(""), io.StringIO()
        try:
            rc = interactive(type("A", (), {})())
        finally:
            sys.stdin, sys.stdout = old_in, old_out
        assert rc == 1, f"非交互环境该明确失败（1），实为 {rc}"
    ck("交互模式在管道里跑 → 明确拒绝（1），不去逐行读", t_interactive_not_tty)

    print()
    print("=" * 72)
    if ok_all:
        print("  ✅ 自检全部通过")
        print("  下一步: python3 src/main.py \"把绿色方块放到红色方块右侧\"")
    else:
        print("  ❌ 有项目没通过，先别连机械臂")
    print("=" * 72)
    return 0 if ok_all else 1


def dType_PTPMOVL() -> int:
    """MOVL 模式号（2）。抽成函数是为了自检里造「MOVL 被拒」的场景。"""
    return 2


# ═══════════════════════════ 五、命令行 ═══════════════════════════
def build_argparser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    ★ 单独抽出来，是为了自检能验「只有 argparser 才知道」的账 ——
      比如 `--net-h` 的默认值到底是不是 25（见 selftest 的 t_default_net_h）。
      main() 只管 parse + 分发，不再把解析器藏在函数体里。
    """
    ap = argparse.ArgumentParser(
        description="主程序: 一句中文 → DeepSeek 决策 → 机械臂抓放",
        epilog="一句话直接干:   python3 src/main.py \"把绿色方块放到红色方块右侧\" --go\n"
               "只想先看看计划: 去掉 --go（不动臂）\n"
               "连着下好几条:   python3 src/main.py -i --go"
               "（开着不用退，一行一条指令）")
    ap.add_argument("instruction", nargs="?", default=None,
                    help="中文指令，如 \"把绿色方块放到红色方块右侧\"。"
                         "一句话里可以带好几步（「把红的放到绿的右边，再把蓝的摞到红的上面」）；"
                         "也可以不给 —— 不给就直接进 -i 交互模式")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="★ 交互模式: 开一条命令，之后一行一条指令，参数只给这一次，"
                         "不用每条都重敲。quit / exit / 退出 或 Ctrl-D 结束；"
                         "给了 --go 的话每条指令仍会先打计划、按回车确认后才动臂")
    ap.add_argument("--go", action="store_true",
                    help="★ 真动机械臂（打完计划按一次回车就开始）。"
                         "不给就只打印计划、不动臂")
    ap.add_argument("--selftest", action="store_true", help="离线自检（不连臂、不联网）")
    ap.add_argument("--fresh", action="store_true",
                    help="--go 时**强制**重新问 DeepSeek（否则只在同一句话、"
                         "且桌面没变过时才复用上次计划）")

    # 数据来源
    ap.add_argument("--world-json", default=None,
                    help="方块坐标 JSON（OpenCV 的输出）。不给则优先读桌面记忆库")
    ap.add_argument("--virtual", action="store_true",
                    help="强制用项目自带的虚拟坐标（有记忆库时默认读记忆库）")
    ap.add_argument("--reset-world", action="store_true",
                    help=f"删掉 {WORLD_STATE_JSON.name} 从零开始（你手工重摆过方块时用）")
    ap.add_argument("--model", default=brain.DEFAULT_MODEL, help="DeepSeek 模型")
    ap.add_argument("--table-z", type=float, default=None,
                    help="纸面 Z（不给则读 output/pick_test_result.json）")
    ap.add_argument("--clear-alarms", action="store_true",
                    help="显式清除限位等报警（丢步报警一律拒绝，没有后门）")

    # 尺寸/运动参数（默认值全部对齐 step4_pick_test.py）
    ap.add_argument("--obj-h", type=float, default=DEFAULT_OBJ_H,
                    help=f"吸盘接触面到已记纸面 Z 的距离，默认 {DEFAULT_OBJ_H}")
    ap.add_argument("--net-h", type=float, default=DEFAULT_NET_H,
                    help=f"方块**净高**（不含 1mm 吸盘压缩量），默认 "
                         f"{DEFAULT_NET_H:.0f}（用户 2026-09-19 要求写死，不用每次敲）。"
                         f"只在两种时候用得上: ① 要叠放（有 z_level>0）；"
                         f"② 搬运路径会**经过**摞着的方块（要算它顶面多高）。"
                         f"★ 卡尺量出来不是 {DEFAULT_NET_H:.0f} 就显式给实测值，"
                         f"否则每叠一层都差 (这个值 − 真实净高)")
    ap.add_argument("--press", type=float, default=0.0, help="抓取时的额外预压量")
    ap.add_argument("--hover", type=float, default=DEFAULT_HOVER,
                    help=f"悬停高度下限，默认 {DEFAULT_HOVER}。三个悬停各自算: "
                         f"抓点上空 = 抓取 Z + 这个值（抓点旁边有方块才会更高）；"
                         f"放点上空 = 放置 Z + 这个值；搬运取两者较高者，"
                         f"路径上真会撞到方块时才再往上抬")
    ap.add_argument("--carry-clear", type=float, default=DEFAULT_CARRY_CLEAR,
                    help=f"吸着方块平移时，方块**底面**至少要高出"
                         f"「路径上要越过的那块」顶面多少 mm（默认 {DEFAULT_CARRY_CLEAR:.0f}）。"
                         f"方块越高、吸盘自动抬得越高；路径上没东西就不抬。"
                         f"这个数越小越贴地、越容易蹭到方块")
    ap.add_argument("--pump-s", type=float, default=DEFAULT_PUMP_S, help="抽气等待秒数")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    ap.add_argument("--acc", type=float, default=DEFAULT_ACC)
    ap.add_argument("--ratio", type=float, default=DEFAULT_RATIO)
    ap.add_argument("--mode", choices=("movl", "movj"), default="movl",
                    help="movl=直线（默认，XY 平移时 Z 不会跑）")
    ap.add_argument("--port", default=None, help="串口，默认自动查找")
    ap.add_argument("--limits", default=None,
                    help='覆盖软限位，如 "x:50:380,y:-310:310"（格式 轴:下限:上限）')

    return ap


# ★ 交互模式下打了这几个词就收工（大小写不敏感；空行＝忽略，不算退出）。
QUIT_WORDS = {"q", "quit", "exit", "退出", "结束", "不干了"}


def run_instruction(args) -> int:
    """跑**一条**指令的完整流程（拿坐标 → 决策/复用 → 规划 → 过目 → 动臂）。

    ★ 交互模式（interactive）每读一行就调一次；单发模式 main() 只调一次。
      所以这里**不能**碰 argparse（ap 出不了 main）—— 参数不对一律 return 退出码，
      绝不调 ap.error，那会直接 SystemExit、把整场交互一起带走。
    ★ 返回值沿用本项目的退出码约定: 0 成功 / 1 运行期失败 / 2 前置条件或用法问题。
    """
    if not args.instruction:
        print("✗ 没有指令。")
        return 2

    if args.reset_world:
        reset_world_state()
        # ★ 只清一次: 交互模式下要是每条指令都清，第二条会把第一条刚写进记忆库的
        #   结果又删掉 —— 用户看着就成了「刚放好，下一句却说方块不存在」。
        args.reset_world = False

    # ── 拿到动作: 要么复用它，要么现问 DeepSeek ──
    # ★ 复用只发生在一种情况下: **同一句话 + 桌面没变过**。
    #   那正是「不带 --go 看过一遍、再 --go 跑同一句」那条路 —— 跑的就是你看过的
    #   那一份。其余情况（换了一句话、或上次执行把方块挪走了）一律重问。
    #
    #   ⚠ 为什么不能「不管理由一律复用上次的计划」: 你会看到
    #     `python3 src/main.py "把X放到Y右边" --go` 完全没按你说的做 —— 它跑的是
    #     上一条指令的计划。2026-09-18 真踩过（左右改了却没生效）。
    #   为什么不能「不管桌面变没变都复用」: 方块已经被上次执行挪走了，
    #     旧计划里的抓取点就是**空位** —— 吸个空。
    saved_table_z = None        # 只有复用分支会填；下面拿它钉「你过目的 Z」没变
    world_src, world_real = "", True   # 坐标是哪来的 / 是不是真的（虚拟的不能落盘）
    use_saved = False
    # ★ 上一份计划不存在是**正常**的（第一次跑就是这么回事）—— 不能像
    #   load_plan_file 那样直接 SystemExit 掉，那等于"没跑过就不让跑"。
    if args.go and not args.fresh and PLAN_JSON.exists():
        saved = load_plan_file()
        # ① 这份计划是按哪套左右/前后编的。方向改动**不会**体现在坐标里，
        #    不查就会「代码改了、方向看着一点没变」（2026-09-18 真踩过）。
        why = plan_rules_mismatch(saved)
        if why:
            print(f"\n✗ 存下的计划不能复用: {why}")
            print(f"  现在这套: {brain.rules_fingerprint()}")
            print(f"  存计划时: {saved.get('brain_rules') or '（没有记录）'}")
            print("  左右/前后这类改动**不会**反映在旧计划的坐标里 —— 直接重跑")
            print("  只会把老方向再走一遍。必须重新编一份:")
            print(f"      python3 src/main.py \"{args.instruction}\"        # 重问，先看计划")
            print(f"      python3 src/main.py \"{args.instruction}\" --go   # 确认后再动臂")
            return 2
        # ② 同一句话吗
        if saved.get("instruction") != args.instruction:
            print(f"[决策] 和上次存的计划不是同一句话（那是「{saved['instruction']}」）"
                  f"—— 重新问 DeepSeek")
        else:
            # ③ 桌面从编那份计划之后变过吗（上次执行会把方块挪走）
            cur_state, _, _ = load_world_state(args)
            if saved.get("world_state") != cur_state:
                print("[决策] 桌面记忆库比那份计划新（方块位置/层数变过）"
                      "—— 重新问 DeepSeek")
            else:
                use_saved = True
    if use_saved:
        actions = saved["actions"]
        world_state = saved["world_state"]
        world_src = saved.get("world_src", "")
        world_real = bool(saved.get("world_real", True))
        # ★ 这一行必须打出来: 「跑的就是你过目过的那份」是整个安全设计的核心，
        #   用户得看得见。之前只赋了 src 没打印，等于白存。
        print(f"[坐标来源] {PLAN_JSON.name} —— 用上次存下的计划，"
              f"**没有**重新问 DeepSeek（要重问加 --fresh）")
        if world_src:
            print(f"           （那份计划编它的时候，坐标来源: {world_src}）")
        if not world_real:
            print("           ⚠ 那份计划用的是**虚拟坐标** —— 跑完之后不会写记忆库。")
        model = saved.get("model", args.model)
        p = saved.get("params", {})
        for k, cli_v in (("obj_h", args.obj_h), ("net_h", args.net_h),
                         ("press", args.press), ("hover", args.hover),
                         ("carry_clear", args.carry_clear)):
            if k in p and p[k] is not None and abs(float(p[k]) - float(cli_v or 0)) > 1e-9:
                print(f"⚠ 参数 --{k.replace('_', '-')} 现在是 {cli_v}，"
                      f"存计划时是 {p[k]} —— Z 会按**现在**的值重算。")
        # ★ 净空变小是唯一「越跑越危险」的方向: 其它参数调来调去只是悬停高低，
        #   净空是「平移时方块底面比最高那摞再高多少」—— 调小了就可能重新刮到。
        if ("carry_clear" in p and p["carry_clear"] is not None
                and float(args.carry_clear) < float(p["carry_clear"]) - 1e-9):
            print(f"  ⚠ 净空**调小了**（{p['carry_clear']:.0f} → "
                  f"{args.carry_clear:.0f}mm）—— 你当初过目的是更大的那个间距。")
        saved_table_z = p.get("table_z")
    else:
        world_state, world_src, world_real = load_world_state(args)
        print(f"[坐标来源] {world_src}")
        try:
            actions, resp = brain.run_one(args.instruction, world_state, args.model)
        except SystemExit:
            raise
        except Exception as e:
            print(f"✗ 问 DeepSeek 失败: {e}")
            return 1
        # ★ resp 是 call_api 的**信封** {content, reasoning, usage, finish}，
        #   tokens 在 usage 里面、不是顶层 —— 写成 resp.get('total_tokens') 会永远打出 "?"。
        tok = (resp.get("usage") or {}).get("total_tokens")
        print(f"[决策] 模型={args.model}  tokens={tok if tok is not None else '?'}")
        model = args.model
        save_plan(args.instruction, model, world_state, actions, args,
                  world_src=world_src, world_real=world_real)
        print(f"[计划] 已存 → {PLAN_JSON.name}")

    print(f"\n指令: {args.instruction}")
    print(f"动作: {brain.pretty(actions)}")

    if not actions:
        print("\n（没有需要执行的动作 —— 可能是方块名不存在，或指令无需移动。不动臂。）")
        return 0

    # ── 规划 ──
    table_z = _resolve_table_z(args)

    # ★ 复用旧计划时，纸面 Z 必须是**当初那个数**。
    #   为什么单独拦: 其它参数（obj_h/hover…）变了顶多是悬停高低，纸面 Z 变了
    #   直接改「吸盘降到多深」—— 低了会摁进桌面。而 Z 是**重新解析**出来的
    #   （不给 --table-z 就去读 output/ 里的文件），完全可能和你过目时不是同一个。
    #   真变了就拒绝：宁可让你重看一遍计划，也不许「看的是一份、跑的是另一份」。
    if (saved_table_z is not None and table_z is not None
            and abs(float(saved_table_z) - float(table_z)) > 0.05):
        print("\n✗ 纸面 Z 变了 —— 拒绝执行。")
        print(f"  你过目的那份计划是按 Z={saved_table_z:.2f} 算的，"
              f"现在解析出来是 Z={table_z:.2f}（差 {table_z - float(saved_table_z):+.2f}mm）。")
        print("  Z 决定吸盘降多深，差一点就可能压到桌面。")
        print(f"  想按原来那个数跑: 加 --table-z {saved_table_z:.2f}")
        print("  想按现在的数跑: 先重看一遍计划（不带 --go），确认后再 --go")
        return 2

    if table_z is not None and saved_table_z is None and args.go and not args.fresh:
        # 存计划时不知道 Z（计划模式下不给 --table-z 是允许的），现在知道了。
        # 这种事没危险（当初没按任何 Z 过目），但得说一声免得你以为没变。
        print(f"⚠ 存计划时还不知道纸面 Z；现在解析出 Z={table_z:.2f} 来算。")

    if table_z is None:
        print("\n✗ 不知道纸面 Z，算不出该降到哪 —— 只出计划，不动臂。")
        print("  先量一次:  python3 src/step4_pick_test.py --probe")
        print("  或者直接给: --table-z -33.8")
        return 2
    try:
        plan = build_plan(actions, table_z, args.obj_h, args.net_h, args.press,
                          args.hover, world_state, args.carry_clear)
    except PlanError as e:
        print(f"\n✗ {e}")
        return 2

    print(f"\n运动计划（纸面 Z={table_z:.2f}，方块高 {args.obj_h}"
          + (f"，净高 {args.net_h}" if args.net_h is not None else "")
          + f"，搬运净空 {args.carry_clear:.0f}）:")
    # ★ 把「这套左右/前后是怎么算的」打出来 —— 方向反了的时候，第一个要看的
    #   就是这一行。光看坐标看不出来（±50 都长得像正常数字）。
    print("      方向约定（" + "；".join(
        brain.side_rule_text(s)
        for s in ("right", "left", "front", "back")) + "）")
    print(describe_plan(plan, table_z))

    limits = s4.limits_from_args(args)
    probs = preflight(plan, limits, table_z)
    if probs:
        print("\n⚠ 预检发现问题:")
        for x in probs:
            print(f"    · {x}")

    if not args.go:
        print("\n★ 以上是**计划**，机械臂一步都没动。")
        print("  认真核对抓取点/放置点是不是你想的位置，然后:")
        # ★ 只在**不是默认值**时才提示带上 --net-h —— 默认 25 已经写死在代码里，
        #   再提示就是教你打一个本来就一样的数（用户 2026-09-19 的诉求）。
        print(f"      python3 src/main.py \"{args.instruction}\" --go"
              + (f" --net-h {args.net_h:g}" if args.net_h != DEFAULT_NET_H else ""))
        print("  想接着下别的指令（不用重开命令、参数也不用再敲）:"
              "  python3 src/main.py -i --go")
        return 0

    # ── 真动臂 ──
    if probs:
        print("\n✗ 预检没过（或有人工确认项），拒绝运动。先解决上面那些问题。")
        return 2
    if not sys.stdin.isatty():
        print("✗ 要动机械臂，就得在**交互式终端**里跑（要人工确认）。")
        return 1

    print("\n⚠ 安全: 清空机械臂周围 60cm；Ctrl-C 随时停；本脚本不回零、不改末端参数")
    try:
        input("确认方块位置和上面的坐标对得上、周围没人 → 按回车开始（Ctrl-C 中止）… ")
    except EOFError:
        return 1

    # ★ progress（哪些方块真的放下了）/ holder（泵还开着吗）两个都从这儿传进去,
    #   跑完/跑挂都拿回来刷新记忆库 —— 见 report_world_refresh 的注释。
    progress: dict = {}
    holder: dict = {}
    rc = run_arm(args, plan, progress, holder)
    report_world_refresh(world_state, plan, progress, holder.get("held", False),
                        world_real=world_real)
    return rc


def interactive(args) -> int:
    """交互模式: 开一条命令，之后一行一条指令，参数只给这一次。

    为什么要有它（用户 2026-09-19 的诉求）: 一句中文一条命令，动几下就得改一次
    引号、还得把 --net-h 那串参数再敲一遍，太啰嗦。

    ★ 安全闸口一步都没少: 每条指令照样走 run_instruction 的完整流程 ——
      打印计划、预检、`--go` 时仍在 TTY 里等你按一次回车才动臂。交互模式缩短的
      只是**打字**，不是**确认**。
    ★ 一条指令失败（退出码 2 / SystemExit）只废掉那一条，循环接着读下一行。
      只有 Ctrl-C 会真的终止（中途停臂必须能停）。
    """
    if not sys.stdin.isatty():
        print("✗ 交互模式要在**交互式终端**里跑 —— 它得一行一行读你的指令。")
        print("  只是想在脚本里跑一条指令的话，直接把指令写在命令行上:"
              "  python3 src/main.py \"把红色方块放到绿色方块右侧\"")
        return 1

    print("=" * 72)
    print("  交互模式: 一行一条指令，回车执行；不用退出去重开命令")
    print("=" * 72)
    print("  · 退出: 打 quit / exit / 退出（或按 Ctrl-D）")
    print("  · 一行里也能带好几步，例如:")
    print("      \"把红色方块放到绿色方块右侧，再把黄色方块摞到蓝色方块上面\"")
    if args.go:
        print("  · ★ --go 已开: 每条指令都会先打计划 + 预检，"
              "你按一次回车才动臂")
    else:
        print("  · ⚠ 没给 --go: 只打计划、**不动臂**。想看它真动，退出后重开命令"
              "并加 --go")
    print("  · 参数（--obj-h / --hover …）只在命令行给这一次，后面每条都沿用；"
          "要改就退出重开")
    # ★ `-i --go "第一句"` 这种写法要认: 命令行上带的那句**先跑**，再接着读输入。
    #   不认的话它就被静默丢掉 —— 用户以为跑了，其实只是进了循环在等（最坏的那种坑）。
    pending = (args.instruction or "").strip()
    if pending:
        print(f"  · 命令行上还带了一句，先跑它: {pending}")
    print()

    n_run = 0
    while True:
        if pending:
            line, pending = pending, ""
        else:
            try:
                line = input("指令> ").strip()
            except EOFError:                 # Ctrl-D
                print()
                break
            except KeyboardInterrupt:        # 提示符上按 Ctrl-C ＝ 收工
                print()
                break
            if not line:
                continue                     # 空行只是回车，不算退出
            if line.lower() in QUIT_WORDS:
                break

        args.instruction = line
        print()
        try:
            rc = run_instruction(args)
        except SystemExit as e:
            # ★ run_instruction 里某些「文件坏了」的路径会 sys.exit —— 那是给单发
            #   模式准备的。在这儿收住，否则一条坏指令就把整场交互带走。
            rc = e.code if isinstance(e.code, int) else 1
            print(f"（这条指令中途退出了，退出码 {rc}；交互继续）")
        if rc == 0:
            n_run += 1
        print()
        print("─" * 72)
        print()

    print(f"交互结束（这次成功处理了 {n_run} 条指令）。")
    return 0


def main() -> int:
    args = build_argparser().parse_args()

    if args.selftest:
        return selftest()

    # ★ 交互模式两条入口: 显式 -i；或者**压根没给指令**。
    #   没给指令时不再 ap.error 退出 —— 用户明说了「一次动作用一个指令过于麻烦」，
    #   不给指令就是「我想一条一条来」的意思。非 TTY 环境由 interactive 自己挡下。
    if args.interactive or not args.instruction:
        return interactive(args)

    return run_instruction(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断] 用户按了 Ctrl-C。")
        sys.exit(130)
