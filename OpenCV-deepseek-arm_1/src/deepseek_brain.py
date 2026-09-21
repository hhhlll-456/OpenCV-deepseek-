#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepseek_brain.py —— 阶段四：把「方块在哪 + 用户说了什么」变成「抓哪、放哪」
=============================================================================
谁在用: 主程序（等底座做好后）。本文件只负责**决策**，不碰摄像头也不碰机械臂。

流程（对应《方案.md》第三~五阶段的接口）:
    OpenCV 标定+颜色识别  →  world_state（方块名 → 机械臂 XY + 层级）
    用户一句话            →  ┐
    world_state           →  ┴→  DeepSeek  →  动作 JSON  →  主程序 → 机械臂

★ 为什么现在就能做: 阶段四只吃 world_state 这个**字典**，不吃摄像头。
  底座还没做好，就先塞一份虚拟坐标进来（见 VIRTUAL_WORLD_STATE），
  把「指令 → 抓取点/放置点」这一段单独跑通、单独验证。底座好了之后，
  把虚拟字典换成 OpenCV 的输出即可，本文件一行都不用改。

───────────────────────── SIDE_OFFSET_MM: 已按真机对调 ─────────────────────────
下面这套左右前后的换算约定（SIDE_OFFSET_MM）**用虚拟坐标测不出对错** ——
虚拟数据下无论左右写反没写反，自检都是全绿的。

★ 2026-09-18 真机实测: 原来那版（right = Y+30）是以**摄像头画面**的左右为准，
  吸到绿方块往"右侧"放，实际落在机械臂的左侧。按用户要求改成**以机械臂的
  左右为准**，于是 right/left 的符号对调（见下表）。

⚠ 只对了 left/right。front/back 仍是原样:**没有在真机上验过**。
  真出现"前后反了"，同样只改这张表，不要去改 prompt 里的散文描述
  （prompt 的文字是从这张表推不出来的，改表才是唯一真源）。

  接真机后第一次跑，仍然**只放两个方块、只走一次**，人眼看一次方向。

用法:
  python3 src/deepseek_brain.py --dry-run              # 只打印要发出去的 prompt，不花钱
  python3 src/deepseek_brain.py --selftest             # 离线自检（不联网）
  python3 src/deepseek_brain.py --demo                 # 拿虚拟坐标跑一批用例，比对标准答案
  python3 src/deepseek_brain.py --say "把绿色方块放到红色方块右侧"
  python3 src/deepseek_brain.py --say "..." --model deepseek-flash --show-reasoning

需要环境变量 DEEPSEEK_API_KEY。本文件**不读、不写、不打印**任何密钥内容。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from paths import WORLD_HALF_KEY, WORLD_SRC_KEY  # noqa: E402

# ★ 记忆库里那些**主程序内部**的戳，发给模型之前一律剥掉（见 build_messages）。
#   它们不属于输出协议（grasp/place 只有 x/y/z_level），发过去只会让模型多一个
#   要照抄的字段（flash 实测爱照抄输入）。加新戳时**记得加到这里**。
INTERNAL_STATE_KEYS = (WORLD_SRC_KEY, WORLD_HALF_KEY)

API_URL = "https://api.deepseek.com/chat/completions"
# 默认 flash: 实测比 v4-pro 快约 25%、正确率没差别（见 README「模型选择」）。
# ★ 注意 tokens 两边基本一样（9221 vs 9228），flash 省的是时间不是 tokens。
# 想换回来: 改这一行，或命令行加 --model deepseek-v4-pro。
DEFAULT_MODEL = "deepseek-flash"
KEY_ENV = "DEEPSEEK_API_KEY"

# 方块边长。方案里写 30mm。★ 必须等于你实际量的高度，抓取 Z 全靠它。
CUBE_MM = 30

# ★ 「放到某块旁边」时，两个方块**中心**的最小距离。2026-09-18 用户要求 ≥50mm。
#   为什么不能等于边长: 中心距 = 边长（30mm）时两块**面贴面**，一点缝隙都没有 ——
#   相机有误差、机械臂有误差，任何一点偏差都是撞上去。留 20mm 缝才够容错。
#   ★ 必须与 CUBE_MM 分开: CUBE_MM 是方块**物理尺寸**（抓取高度 / 障碍高度 /
#     叠放高度全靠它），SIDE_GAP_MM 是**摆放间距**，两者是两件事，别再合回去。
SIDE_GAP_MM = 50

# ─────────────────────── 左右前后的换算约定 ───────────────────────
# ★★ 这张表是**唯一真源**。prompt 里的散文和下面的自检都只是它的复述 ——
#    方向要改就只改这里，别去改 prompt 的措辞（那样两边会打架）。
#
# 原先照抄《方案.md》第 84 行「左边 Y−30，右边 Y+30」，真机一跑发现那是**摄像头
# 画面**的左右，机械臂实际做出来是反的。2026-09-18 按要求改成**以机械臂的左右
# 为准**，于是 right/left 符号对调。
#   · 判据: 机械臂自身的 +Y 方向往哪边，哪边才叫"右"。
#   · front/back 没动（用户只报了左右反），**仍未经真机验证**。
# ★ 距离用 SIDE_GAP_MM（50mm）而不是 CUBE_MM（30mm）—— 见上面 SIDE_GAP_MM 的说明。
SIDE_OFFSET_MM = {
    "right": (0, -SIDE_GAP_MM),      # 右边（机械臂的右手边）→ Y - 50
    "left":  (0, +SIDE_GAP_MM),      # 左边（机械臂的左手边）→ Y + 50
    "front": (+SIDE_GAP_MM, 0),      # 前边 → X + 50
    "back":  (-SIDE_GAP_MM, 0),      # 后边 → X - 50
}

SIDE_CN = {"right": "右侧", "left": "左侧", "front": "前面", "back": "后面"}


def rules_fingerprint() -> str:
    """
    「左右/前后怎么算、放多远」的指纹：换算表 + 方块边长 + 摆放间距。

    ★ 为什么需要它: 这几条规则**不会**体现在旧计划的坐标里 ——
      改了 right 的符号之后，`output/last_plan.json` 里那些 place 坐标一个字都不变。
      主程序默认 `--go` 是**复用**旧计划、不问 DeepSeek，于是看起来
      「代码改了、方向一点没变」，实际跑的还是改之前那份。
      2026-09-18 真踩过: 左右对调后重跑，方块照样落在另一边。
      存计划时把这个指纹一起存下来，复用前比对，对不上就拒绝。

    ★ 把 gap 也算进来同样是为这件事: 间距从 30 改成 50 之后，旧计划里的
      place 坐标**看起来完全合法**（就是老间距那个值），复用它会照老间距摆，
      两块面贴面。指纹里带 gap，旧计划自动作废、重新问一次模型。
    """
    parts = [f"{k}:{v[0]},{v[1]}" for k, v in sorted(SIDE_OFFSET_MM.items())]
    parts.append(f"cube:{CUBE_MM}")
    parts.append(f"gap:{SIDE_GAP_MM}")
    return "|".join(parts)


def _delta_phrase(dx: int, dy: int) -> str:
    """把 (dx, dy) 说成人话: 「x 不变，y 减去 50」。"""
    xy = [f"x {'加上' if dx > 0 else '减去'} {abs(dx)}" if dx else "x 不变",
          f"y {'加上' if dy > 0 else '减去'} {abs(dy)}" if dy else "y 不变"]
    return "，".join(xy)


def side_rule_text(side: str) -> str:
    """人话描述一条方向: 「右侧 = x 不变，y 减去 50」。打计划时也用它。"""
    return f"{SIDE_CN[side]} = {_delta_phrase(*SIDE_OFFSET_MM[side])}"


def _side_rule_line(side: str) -> str:
    """
    把 SIDE_OFFSET_MM 的一行翻成人话，喂给 prompt。

    ★ 为什么不让 prompt 自己手写这几句: 这张表**已经改过一次方向**（左右对调）。
      手写的散文不会跟着改，结果是「表说 Y−30、prompt 说 y 加 30」——
      模型照 prompt 算，主程序拿表核对，两边打架而且不报错。
      同源生成，表改了 prompt 自动跟着改。
    """
    return f"  某个方块{SIDE_CN[side]} = 它的 " + _delta_phrase(*SIDE_OFFSET_MM[side])


# ─────────────────── 虚拟 world_state（底座好了就换掉）───────────────────
# 字段抄《方案.md》第 64-71 行。z_level: 0 = 躺在桌面上，1 = 摞在一个方块上。
# ★ 这三个数字只是占位，让决策链路能跑起来；接了真标定后由 OpenCV 写入。
VIRTUAL_WORLD_STATE = {
    "red":    {"x": 210, "y": 150, "z_level": 0},
    "yellow": {"x": 260, "y": 120, "z_level": 0},
    "blue":   {"x": 180, "y": 180, "z_level": 0},
    "green":  {"x": 220, "y": 200, "z_level": 0},
}

COLOR_CN = {"red": "红色", "yellow": "黄色", "blue": "蓝色", "green": "绿色"}


# ─────────────────────────── Prompt ───────────────────────────
def build_messages(world_state: dict, instruction: str) -> list[dict]:
    """
    组 prompt。坐标系和输出格式都在这里写死 —— 它们是**接口**，不是可以自由发挥的东西。

    ★ 输出的动作里同时给「抓取点」和「放置点」两组坐标:
      抓取点 = 那个方块现在在哪；放置点 = 按指令算出来的目标位置。
      这样主程序不需要再猜任何一个位置，拿到就能直接走。
    """
    # ★ 记忆库里每条可能带内部戳 "src"/"half"（x/y 是相机拍的还是机械臂确认的、
    #   在纸的哪半边，见 paths.WORLD_SRC_KEY / WORLD_HALF_KEY）—— 那是**主程序
    #   内部的账**，不是给模型的字段: 输出协议里 grasp/place 只有 x/y/z_level，
    #   把它们一起发过去就是让模型多几个要照抄的字段（flash 实测爱照抄输入），
    #   纯属给它添乱。剥掉再发（列表见 INTERNAL_STATE_KEYS）。
    plain = {k: ({kk: vv for kk, vv in v.items() if kk not in INTERNAL_STATE_KEYS}
                 if isinstance(v, dict) else v)
             for k, v in world_state.items()}
    state_txt = json.dumps(plain, ensure_ascii=False, indent=2)
    # ★ 方块名映射由 world_state 的键自动生成，不手写 —— 换了/加了方块不会漏。
    #   用户说的是中文，但 obj 字段必须回英文键名，否则下游按名字查不到。
    name_map = "\n".join(f"  {COLOR_CN.get(k, k)} → {k}" for k in world_state)
    # ★ 换算规则**从 SIDE_OFFSET_MM 生成**，不手写（见 _side_rule_line 的说明）。
    side_rules = "\n".join(_side_rule_line(s)
                           for s in ("right", "left", "front", "back"))
    # ★ 示例里的坐标也一并算出来，别手写数字 —— 左右一对调，手写的 330 就变成了
    #   反方向的例子，而 prompt 里自相矛盾的示例比没有示例更糟。
    ex_y = 300
    ex_ry = ex_y + SIDE_OFFSET_MM["right"][1]
    system = f"""你是一个机械臂决策大脑。你要把用户的一句中文指令，翻译成一串机械臂动作。

【方块名对照】用户嘴里说的是中文，你在 obj 字段里**必须**写左边的英文键名：
{name_map}

【当前桌面上方块的位置】坐标是机械臂坐标系，单位毫米：
{state_txt}

【方块的物理属性】
每个方块是边长 {CUBE_MM}mm 的正方体。
坐标指的是方块**中心**在桌面上的投影位置。
z_level 表示它摞在第几层：0 = 直接放在桌面上；1 = 摞在一个方块上面；以此类推。

【放置位置的换算规则】（必须严格照算，不许自己估计）
放到某个方块"旁边"时，落点与它的方块**中心**相距 {SIDE_GAP_MM}mm
（方块本身只有 {CUBE_MM}mm 宽，两块之间留 {SIDE_GAP_MM - CUBE_MM}mm 的缝）：
{side_rules}
  放到某个方块"上面" = x、y 和它相同，z_level 比它大 1
放到某个方块"旁边"（左/右/前/后）时，方块是落在**桌面上**的，z_level 恒为 0
—— 即使参照物本身摞在第 1 层（旁边那块地方下面是空的），也是 0。
放到桌面上（不是摞起来）时，z_level = 0。

【多条指令】
如果用户一句话里说了好几个动作，必须**按顺序**处理，并且每一步都要
考虑前面动作已经改变了桌面：方块被挪走以后，它就在新位置了。

【输出格式】
只输出一个 JSON 数组，不要任何解释文字、不要 Markdown 代码块。

★★ 最外层永远是方括号 [ ]，这一条没有例外。
   哪怕整句话只移动 1 个方块，也必须写成 [{{...}}]，**绝对不许**只写 {{...}}。
   （漏掉方括号的返回会被主程序直接判为无效、整条指令作废。）

数组每个元素是一个动作，按执行顺序排列：
[
  {{
    "obj": "要移动的方块名",
    "grasp": {{"x": 数字, "y": 数字, "z_level": 数字}},
    "place": {{"x": 数字, "y": 数字, "z_level": 数字}}
  }}
]
grasp 是这个方块**现在**的位置，place 是它**应该被放到**的位置。
如果指令里提到的方块名不存在，输出空数组 []。

【格式示例】⚠️ 下面示例里的坐标**全是编的**，只为了演示方括号怎么套、动作多时
怎么排。你必须用上面【当前桌面上方块的位置】里的真实数字重新算，**不许照抄**
示例里的数字。

用户说：把红色方块放到黄色方块右侧
（黄色方块在 (400,{ex_y},层0)，所以右侧是 {_delta_phrase(*SIDE_OFFSET_MM['right'])}，落在桌面上）
输出：[{{"obj":"red","grasp":{{"x":100,"y":{ex_y},"z_level":0}},"place":{{"x":400,"y":{ex_ry},"z_level":0}}}}]
—— 注意只有 1 个动作时，外面**照样有**方括号。

用户说：把红色方块放到黄色方块上面，再把蓝色方块放到红色方块右侧
输出：[{{"obj":"red","grasp":{{"x":100,"y":{ex_y},"z_level":0}},"place":{{"x":400,"y":{ex_y},"z_level":1}}}},
      {{"obj":"blue","grasp":{{"x":600,"y":600,"z_level":0}},"place":{{"x":400,"y":{ex_ry},"z_level":0}}}}]
—— 第 2 步用的是红色方块**被挪到新位置之后**的坐标；放到"旁边"时 z_level 是 0，
   即使红色方块本身已经摞到第 1 层。"""

    return [{"role": "system", "content": system},
            {"role": "user", "content": instruction}]


# ─────────────────────────── 调 API ───────────────────────────
def get_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise SystemExit(
            f"✗ 环境变量 {KEY_ENV} 没设置。\n"
            f"  临时用:  export {KEY_ENV}='sk-...'\n"
            f"  ★ 别把它写进任何文件 —— 这个项目是要交付给别人的。")
    return key


def call_api(messages: list[dict], model: str = DEFAULT_MODEL,
             timeout: float = 120.0) -> dict:
    """返回 {content, reasoning, usage}。出错抛 SystemExit，附上可读的原因。"""
    body = {"model": model, "messages": messages, "temperature": 0,
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {get_key()}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        raw = raw.replace(os.environ.get(KEY_ENV, ""), "<key>")   # 万一回显，别打出来
        hint = {401: "key 无效或过期", 402: "余额不足", 429: "请求太频繁",
                400: "请求格式不对（模型名写错了？）"}.get(e.code, "")
        raise SystemExit(f"✗ HTTP {e.code} {hint}\n  {raw[:400]}")
    except Exception as e:
        raise SystemExit(f"✗ 请求失败: {type(e).__name__}: {e}")

    msg = d["choices"][0]["message"]
    return {"content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "usage": d.get("usage", {}),
            "finish": d["choices"][0].get("finish_reason")}


# ─────────────────────── 解析返回的 JSON ───────────────────────
def parse_actions(content: str) -> list[dict]:
    """
    把模型返回的文本解析成动作列表。

    ★ 即使开了 JSON 模式也要容错: 实测模型偶尔仍会裹上一层 ```json 代码块，
      或者前面带一句"好的，以下是…"。这条链路的下游是机械臂，解析失败必须
      **抛错**，绝不能猜一个动作出来 —— 猜错就是撞上去。
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("返回内容为空")

    txt = content.strip()
    # 剥掉可能的 Markdown 代码块
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if m:
        txt = m.group(1).strip()
    # 万一带了前言后语，截取出 JSON
    if not txt.startswith("["):
        i, j = txt.find("["), txt.rfind("]")
        if i >= 0 and j > i:
            txt = txt[i:j + 1]
        else:
            # ★ deepseek-flash 实测约 1/40 会把「只有 1 个动作」直接返回成单个
            #   对象 {...}，而不是协议要求的 [{...}]（v4-pro 没见出现过）。
            #   协议约定是数组，但单个对象的意思**唯一确定** —— 包一层就还原了，
            #   属于还原、不属于猜。不兜这一下，一个括号就会让整条流水线崩掉，
            #   而模型其实算对了（见 README「模型选择」）。注意: 字段照样逐个校验，
            #   残缺的对象仍然会被下面拒掉。
            i, j = txt.find("{"), txt.rfind("}")
            if i < 0 or j <= i:
                raise ValueError(f"返回里找不到 JSON 数组/对象: {content[:200]!r}")
            txt = "[" + txt[i:j + 1] + "]"

    data = json.loads(txt)
    if not isinstance(data, list):
        raise ValueError(f"顶层不是数组，而是 {type(data).__name__}")

    out = []
    for k, a in enumerate(data):
        if not isinstance(a, dict):
            raise ValueError(f"第 {k} 个动作不是对象: {a!r}")
        for field in ("obj", "grasp", "place"):
            if field not in a:
                raise ValueError(f"第 {k} 个动作缺字段 {field!r}: {a!r}")
        pt = {}
        for where in ("grasp", "place"):
            p = a[where]
            if not isinstance(p, dict):
                raise ValueError(f"第 {k} 个动作的 {where} 不是对象: {p!r}")
            try:
                pt[where] = {"x": float(p["x"]), "y": float(p["y"]),
                             "z_level": int(p["z_level"])}
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"第 {k} 个动作的 {where} 字段不对: {p!r} ({e})")
        out.append({"obj": str(a["obj"]).strip().lower(), **pt})
    return out


# ──────────────── 标准答案：确定性地算出「应该是什么」 ────────────────
# ★ 这一层存在的意义: 没有它就无法判断 DeepSeek 到底算得对不对 ——
#   「看着像对的」不是判据。它把同一套换算规则用代码写一遍，
#   拿代码的结果去比对模型的结果。两边都按 SIDE_OFFSET_MM 算，
#   所以它验的是「模型有没有照规则算」，**不是**「规则本身对不对」。
def resolve_place(state: dict, spec: tuple) -> dict:
    """
    spec 有两种:
      ("side",  "red", "right")  → 红色方块右侧，落在**桌面**上
      ("on_top", "yellow")       → 摞到黄色方块上面

    ★★ 「旁边」的层级恒为 0（桌面），**不是**参照物的层级。
      这一点很容易想当然写错: 若参照物在第 1 层（摞在别的方块上），照抄它的
      层级会算出"放在半空中" —— 那块地方下面没有支撑，方块会掉。
      旁边的位置落在桌面上，所以恒为第 0 层。

    ★ 这里的假设: 那个落点下方的桌面是空的。world_state 不记录空位被谁占了，
      所以「放到一个已经有东西的位置」这种指令本层看不出来 —— 见 compare() 之外
      的说明。真机上撞到这种情况，是 world_state 该补的信息，不是这里猜。
    """
    kind = spec[0]
    if kind == "side":
        _, ref, side = spec
        dx, dy = SIDE_OFFSET_MM[side]
        r = state[ref]
        return {"x": r["x"] + dx, "y": r["y"] + dy, "z_level": 0}
    if kind == "on_top":
        _, ref = spec
        r = state[ref]
        return {"x": r["x"], "y": r["y"], "z_level": r["z_level"] + 1}
    raise ValueError(f"不认识的放置方式: {kind!r}")


def expected_actions(world_state: dict, specs: list[dict]) -> list[dict]:
    """
    按 specs 顺序模拟一遍，给出每个动作的标准答案。

    ★ 顺序执行 + 每步更新桌面状态 —— 和机械臂执行完刷新记忆库是同一个逻辑
      （《方案.md》第五阶段第 3 条）。复合指令能不能算对，全看这一步：
      第二个动作必须基于**第一个动作之后**的桌面，而不是初始桌面。
    """
    st = copy.deepcopy(world_state)
    out = []
    for spec in specs:
        obj = spec["obj"]
        if obj not in st:
            raise ValueError(f"用例里提到的方块不在桌面上: {obj!r}")
        grasp = dict(st[obj])
        place = resolve_place(st, spec["place"])
        out.append({"obj": obj, "grasp": grasp, "place": place})
        st[obj] = dict(place)                 # 挪走了 → 记忆库刷新
    return out


def compare(got: list[dict], want: list[dict], tol_mm: float = 0.5) -> list[str]:
    """比对，返回问题列表（空 = 全对）。"""
    problems = []
    if len(got) != len(want):
        return [f"动作条数不对: 模型给了 {len(got)} 条，应该是 {len(want)} 条"]

    for k, (g, w) in enumerate(zip(got, want)):
        tag = f"第{k + 1}个动作({g['obj']})"
        if g["obj"] != w["obj"]:
            problems.append(f"{tag}: 要动的方块不对 —— 模型 {g['obj']}，应为 {w['obj']}")
            continue
        for where in ("grasp", "place"):
            for ax in ("x", "y"):
                if abs(g[where][ax] - w[where][ax]) > tol_mm:
                    problems.append(
                        f"{tag} 的 {where}.{ax}: 模型 {g[where][ax]:.0f}，"
                        f"应为 {w[where][ax]:.0f}（差 {g[where][ax] - w[where][ax]:+.0f}mm）")
            if g[where]["z_level"] != w[where]["z_level"]:
                problems.append(
                    f"{tag} 的 {where}.z_level: 模型 {g[where]['z_level']}，"
                    f"应为 {w[where]['z_level']}")
    return problems


def pretty(actions: list[dict]) -> str:
    lines = []
    for a in actions:
        g, p = a["grasp"], a["place"]
        lines.append(f"    · {a['obj']:6} 抓({g['x']:.0f},{g['y']:.0f},层{g['z_level']})"
                     f" → 放({p['x']:.0f},{p['y']:.0f},层{p['z_level']})")
    return "\n".join(lines) if lines else "    （空：没有需要执行的动作）"


# ────────────────────────── 用例 ──────────────────────────
# 想加用例就往这里加: say = 你说的话，specs = 标准答案（按顺序）。
# specs 用 ("side", 参照物, 方位) / ("on_top", 参照物) 描述，坐标由上面的规则算出来，
# 这样标准答案不是手打的魔法数字，改约定时它会跟着一起变。
DEMO_CASES = [
    {"say": "把绿色方块放到红色方块右侧",
     "specs": [{"obj": "green", "place": ("side", "red", "right")}]},

    {"say": "把蓝色方块放到黄色方块的左边",
     "specs": [{"obj": "blue", "place": ("side", "yellow", "left")}]},

    {"say": "把红色方块放到黄色方块上面",
     "specs": [{"obj": "red", "place": ("on_top", "yellow")}]},

    {"say": "把黄色方块放到绿色方块前面",
     "specs": [{"obj": "yellow", "place": ("side", "green", "front")}]},

    # ★ 复合指令 + 状态更新: 第 2 步必须基于第 1 步之后的桌面，
    #   绿色已经挪到红色右侧了，蓝色要摞在**挪走之后**的绿色上。
    {"say": "先把绿色方块放到红色方块右侧，然后把蓝色方块放到绿色方块上面",
     "specs": [{"obj": "green", "place": ("side", "red", "right")},
               {"obj": "blue", "place": ("on_top", "green")}]},

    # 放在原地不动的参照物旁边，验证它没有把"要动的方块"和"参照物"搞混
    {"say": "把红色方块放到蓝色方块后面",
     "specs": [{"obj": "red", "place": ("side", "blue", "back")}]},

    # ★ 三步连环，而且第 2 步正好踩在上面那条「旁边恒为第 0 层」的规则上:
    #   红色摞到黄色上之后在第 1 层，蓝色放到"红色右侧"必须落回桌面(第 0 层)。
    #   照抄参照物层级的实现会在这里给出 1 —— 这条用例就是冲着这个来的。
    {"say": "把红色方块放到黄色方块上面，然后把蓝色方块放到红色方块右侧，"
            "最后把绿色方块放到蓝色方块上面",
     "specs": [{"obj": "red", "place": ("on_top", "yellow")},
               {"obj": "blue", "place": ("side", "red", "right")},
               {"obj": "green", "place": ("on_top", "blue")}]},
]

# 提示里没提到、也不该杜撰: 让它承认无法处理（应当返回空数组且不改动任何东西）
UNKNOWN_CASE = {"say": "把紫色方块放到红色方块右侧", "specs": []}


# ─────────────────────────── 跑一条 ───────────────────────────
def run_one(instruction: str, world_state: dict, model: str,
            show_reasoning: bool = False) -> tuple[list[dict], dict]:
    msgs = build_messages(world_state, instruction)
    r = call_api(msgs, model=model)
    if show_reasoning and r["reasoning"]:
        print(f"  [模型思考过程]\n{r['reasoning']}\n")
    return parse_actions(r["content"]), r


def cmd_demo(model: str, show_reasoning: bool, repeat: int = 1) -> int:
    """
    跑一批用例并和标准答案比对。

    ★ repeat > 1 是**可靠性**检验，不是凑数: 推理模型同一句话两次可能给不同答案，
      「跑一遍全对」证明不了能上机械臂。多轮只报汇总，不逐条刷屏。
    """
    cases = DEMO_CASES + [UNKNOWN_CASE]
    verbose = repeat == 1

    print("=" * 74)
    print(f"  DeepSeek 决策实验 —— 虚拟坐标，模型={model}，每条跑 {repeat} 轮")
    print("  ★ 虚拟坐标下只能验「模型有没有照规则算」，验不了「规则本身对不对」")
    print("=" * 74)
    print("\n[虚拟桌面]")
    for k, v in VIRTUAL_WORLD_STATE.items():
        print(f"    {k:6}({COLOR_CN[k]})  x={v['x']:>4}  y={v['y']:>4}  层={v['z_level']}")
    print(f"    方块边长 {CUBE_MM}mm；放到旁边时中心距 {SIDE_GAP_MM}mm"
          f"（留 {SIDE_GAP_MM - CUBE_MM}mm 缝）")

    tally = {c["say"]: [0, 0, []] for c in cases}          # [对, 错, 错因...]
    tok = 0
    for it in range(repeat):
        if not verbose:
            print(f"\n### 第 {it + 1}/{repeat} 轮")
        for case in cases:
            want = expected_actions(VIRTUAL_WORLD_STATE, case["specs"])
            if verbose:
                print("\n" + "─" * 74)
                print(f"指令: {case['say']}")
            try:
                got, r = run_one(case["say"], VIRTUAL_WORLD_STATE, model, show_reasoning)
            except (SystemExit, ValueError) as e:
                tally[case["say"]][1] += 1
                tally[case["say"]][2].append(str(e))
                if verbose:
                    print(f"  ❌ 没跑通: {e}")
                else:
                    print(f"  ❌ 第{it + 1}轮 · {case['say']} → {e}")
                continue
            tok += r["usage"].get("total_tokens", 0)
            if verbose:
                print("  模型给的动作:")
                print(pretty(got))
                print("  标准答案:")
                print(pretty(want))

            probs = compare(got, want)
            if probs:
                tally[case["say"]][1] += 1
                tally[case["say"]][2].append("; ".join(probs))
                if verbose:
                    for p in probs:
                        print(f"  ❌ {p}")
                else:
                    print(f"  ❌ 第{it + 1}轮 · {case['say']}")
            else:
                tally[case["say"]][0] += 1
                if verbose:
                    print("  ✅ 与标准答案一致")

    n_ok = sum(v[0] for v in tally.values())
    n_bad = sum(v[1] for v in tally.values())
    print("\n" + "=" * 74)
    print(f"  结果: {n_ok} 对 / {n_bad} 错（共 {n_ok + n_bad} 次调用）  tokens={tok}")
    print()
    for c in cases:
        ok, bad, _ = tally[c["say"]]
        mark = "✅" if bad == 0 else ("⚠️ " if ok else "❌")
        print(f"    {mark} {ok}/{ok + bad}  {c['say']}")
    bad_cases = [(c["say"], tally[c["say"]][2]) for c in cases if tally[c["say"]][1]]
    if bad_cases:
        print("\n  错的地方:")
        for say, whys in bad_cases:
            print(f"    · {say}")
            for w in dict.fromkeys(whys):                  # 同样的错只报一次
                print(f"        {w}")
    print("=" * 74)
    return 1 if n_bad else 0


# ─────────────────────────── 离线自检 ───────────────────────────
def cmd_selftest() -> int:
    print("=" * 74)
    print("  离线自检（不联网、不花钱）")
    print("=" * 74)
    checks = []

    def ck(name, fn):
        try:
            fn()
            checks.append((True, name))
            print(f"  ✅ {name}")
        except AssertionError as e:
            checks.append((False, name))
            print(f"  ❌ {name} → {e}")

    # 1. 换算规则。（★ 这条钉的是「表没被人手滑改回去」，不是「方向对」——
    #    方向对错只能上真机看，虚拟坐标验不出来。）
    def t_offsets():
        g = SIDE_GAP_MM
        # ★ 2026-09-18 真机对调: 以**机械臂**的左右为准，right 是 Y−。
        assert SIDE_OFFSET_MM["right"] == (0, -g), f"右边该是 Y-{g}（机械臂的右手边）"
        assert SIDE_OFFSET_MM["left"] == (0, +g), f"左边该是 Y+{g}（机械臂的左手边）"
        assert SIDE_OFFSET_MM["front"] == (g, 0), f"前边该是 X+{g}"
        assert SIDE_OFFSET_MM["back"] == (-g, 0), f"后边该是 X-{g}"
        # 左右必须正好相反，否则"换个方向放"就变成"放到同一处"
        assert (SIDE_OFFSET_MM["right"][0] == SIDE_OFFSET_MM["left"][0]
                and SIDE_OFFSET_MM["right"][1] == -SIDE_OFFSET_MM["left"][1]), \
            "左右必须互为反方向，且走同一条轴"
        assert SIDE_OFFSET_MM["right"][0] == 0, "左右只动 Y，不动 X"
        # ★ 摆放间距必须**大于**方块边长 —— 相等就是面贴面，一动就撞。
        #   这条钉的是 2026-09-18 的改动: 间距曾经等于 CUBE_MM（30mm）。
        assert SIDE_GAP_MM > CUBE_MM, \
            f"旁边落点的中心距 {SIDE_GAP_MM}mm 没有大于边长 {CUBE_MM}mm —— 两块会贴在一起"
        assert SIDE_GAP_MM >= 50, f"用户要求中心距至少 5cm，现在是 {SIDE_GAP_MM}mm"
    ck("左右前后换算规则（左右已按机械臂对调，前后未验；旁边中心距 50mm）", t_offsets)

    # 1b. prompt 里的散文必须是**从表生成**的，不能手写。
    #     手写过一次，表改了 prompt 没改 —— 模型照 prompt 算、拿表核对，必错。
    def t_prompt_side_same_source():
        s = build_messages(VIRTUAL_WORLD_STATE, "测试")[0]["content"]
        for side in ("right", "left", "front", "back"):
            assert _side_rule_line(side).strip() in s, \
                f"prompt 里的「{SIDE_CN[side]}」不是从 SIDE_OFFSET_MM 生成的"
        # 示例里的落点也得跟着表走（曾经手写成 +30，对调后就成了反例）
        _dx, dy = SIDE_OFFSET_MM["right"]
        assert f'"y":{300 + dy}' in s, "格式示例里的右侧落点数字没跟着表走"
    ck("prompt 的换算规则/示例都从 SIDE_OFFSET_MM 生成", t_prompt_side_same_source)

    # 2. 解析器能吃下正常 JSON
    def t_parse_ok():
        got = parse_actions(json.dumps([{"obj": "green",
                                         "grasp": {"x": 1, "y": 2, "z_level": 0},
                                         "place": {"x": 3, "y": 4, "z_level": 1}}]))
        assert got[0]["place"]["z_level"] == 1 and got[0]["obj"] == "green"
    ck("解析正常 JSON", t_parse_ok)

    # 3. 解析器能剥掉 Markdown 代码块（模型偶尔还是会裹）
    def t_parse_fence():
        got = parse_actions('好的，以下是结果：\n```json\n[{"obj":"red",'
                            '"grasp":{"x":0,"y":0,"z_level":0},'
                            '"place":{"x":1,"y":1,"z_level":0}}]\n```\n希望有帮助')
        assert got[0]["obj"] == "red"
    ck("解析带代码块/前后言的返回", t_parse_fence)

    # 4. 解析失败必须抛错，绝不能猜一个动作出来
    def t_parse_bad():
        for bad in ("", "   ", "我不知道", "[{\"obj\":\"red\"}]",
                    "[1,2,3]", '{"obj":"red"}'):
            try:
                parse_actions(bad)
            except (ValueError, json.JSONDecodeError):
                continue
            raise AssertionError(f"{bad!r} 本该报错却通过了")
    ck("畸形返回一律报错（不猜动作）", t_parse_bad)

    # 4b. flash 的格式打滑: 单个动作返回单个对象（实测约 1/40）→ 包成单元素数组。
    #     意思唯一确定，属于还原；但字段仍要齐全。
    def t_parse_lone_object():
        one = {"obj": "red",
               "grasp": {"x": 210, "y": 150, "z_level": 0},
               "place": {"x": 260, "y": 120, "z_level": 1}}
        for form in (json.dumps(one),
                     "好的：" + json.dumps(one),          # 带前言
                     json.dumps(one, ensure_ascii=False)):
            got = parse_actions(form)
            assert len(got) == 1, f"应还原成 1 个动作，得到 {len(got)} 个: {form!r}"
            assert got[0]["place"]["z_level"] == 1
    ck("单个对象还原成单元素数组（flash 格式打滑）", t_parse_lone_object)

    # 5. 复合指令的标准答案：第二步必须基于第一步之后的桌面
    def t_sequential():
        specs = [{"obj": "green", "place": ("side", "red", "right")},
                 {"obj": "blue", "place": ("on_top", "green")}]
        exp = expected_actions(VIRTUAL_WORLD_STATE, specs)
        red = VIRTUAL_WORLD_STATE["red"]
        ry = red["y"] + SIDE_OFFSET_MM["right"][1]     # 右侧落在哪，跟表走
        assert exp[0]["place"]["x"] == red["x"], "绿应放在红的 x 上"
        assert exp[0]["place"]["y"] == ry, f"绿应放在红的右侧（y={ry}）"
        assert exp[1]["place"]["x"] == red["x"], "蓝应摞在**挪走后**的绿上（x=红的x）"
        assert exp[1]["place"]["y"] == ry
        assert exp[1]["place"]["z_level"] == 1, "摞起来该是第 1 层"
    ck("复合指令：第二步基于第一步之后的桌面", t_sequential)

    # 5b. 「旁边」恒为第 0 层。★ 这是本项目真犯过一次的错: 照抄参照物的层级，
    #     在参照物摞在第 1 层时会算出"放在半空中"。最初的用例参照物全在第 0 层，
    #     把它盖住了 —— 所以这条用例单独把参照物放到第 1 层。
    def t_side_level():
        st = {"red": {"x": 100, "y": 100, "z_level": 1}}
        p = resolve_place(st, ("side", "red", "right"))
        assert p["z_level"] == 0, f"参照物在第1层时，旁边该落回桌面(0)，算出的是 {p['z_level']}"
        assert p["y"] == 100 + SIDE_OFFSET_MM["right"][1] and p["x"] == 100
        assert resolve_place(st, ("on_top", "red"))["z_level"] == 2, "摞到第1层上该是第2层"
    ck("「旁边」恒为第 0 层（不照抄参照物层级）", t_side_level)

    # 6. 比对函数能抓到错
    def t_compare():
        want = expected_actions(VIRTUAL_WORLD_STATE,
                                [{"obj": "green", "place": ("side", "red", "right")}])
        assert not compare(copy.deepcopy(want), want), "正确答案不该报错"
        bad = copy.deepcopy(want)
        # 放到**左侧**（即方向反了）—— 这正是真机上最容易发生、也最不容易看出来的错
        bad[0]["place"]["y"] = VIRTUAL_WORLD_STATE["red"]["y"] + SIDE_OFFSET_MM["left"][1]
        assert compare(bad, want), "放反方向必须被抓到"
        assert compare([], want), "条数不对必须被抓到"
    ck("比对函数能抓到放反方向 / 条数不对", t_compare)

    # 7. prompt 里必须真的写了换算规则和格式（改了 prompt 别把接口改坏）
    def t_prompt():
        s = build_messages(VIRTUAL_WORLD_STATE, "测试")[0]["content"]
        for must in (str(CUBE_MM), str(SIDE_GAP_MM), "右侧", "grasp", "place", "z_level"):
            assert must in s, f"prompt 里少了 {must!r}"
        # ★ 间距的规矩必须说「中心距 50mm」，不能再说「就是一个边长 30mm」——
        #   后者会让模型按 30mm 算，两块面贴面。
        assert f"中心**相距 {SIDE_GAP_MM}mm" in s, \
            "prompt 没写清「旁边的落点与参照物中心相距 50mm」"
        # ★ 每个方块都得有「中文名 → 英文键名」的对照。少了模型就可能把 obj
        #   写成"红色"，下游按名字查 world_state 会 KeyError。
        for k in VIRTUAL_WORLD_STATE:
            assert f"→ {k}" in s, f"prompt 里少了 {k} 的名字对照"
        # 而且对照表必须是从 world_state 生成的，不是写死的四个颜色
        s2 = build_messages({"purple": {"x": 0, "y": 0, "z_level": 0}}, "测试")[0]["content"]
        assert "→ purple" in s2, "方块名对照没跟着 world_state 走"

        # ★ 记忆库里的内部戳（paths.WORLD_SRC_KEY / WORLD_HALF_KEY）是主程序内部
        #   的账，**不许**发出去: 输出协议里 grasp/place 只有 x/y/z_level，
        #   多一个字段就让模型多一个要照抄的东西。坐标本身照发（剥戳不能连 x/y 一起剥掉）。
        s3 = build_messages({"purple": {"x": 7.0, "y": 9.0, "z_level": 1,
                                        WORLD_SRC_KEY: "camera",
                                        WORLD_HALF_KEY: "left"}}, "测试")[0]["content"]
        for k in INTERNAL_STATE_KEYS:
            assert k not in s3, f"内部戳 {k!r} 漏进 prompt 了"
        assert '"x": 7.0' in s3 and '"y": 9.0' in s3, "剥戳时把坐标也剥掉了"

        # ★ flash 实测会漏掉最外层的方括号（只返回 {{...}}）——prompt 里必须明确禁止。
        assert "绝对不许" in s and "方括号" in s, "没写「不许省略方括号」这条硬规则"
        assert "【格式示例】" in s, "少了格式示例（格式合规主要靠它）"

        # ★ 格式示例里的数字**绝不能和真实坐标撞车**。撞了模型可能直接照抄示例
        #   数字当成答案，而且算出来"看着很合理"，测试才发现得了。
        ex = s[s.find("【格式示例】"):]
        ex_nums = {int(n) for n in re.findall(r"\d+", ex)}
        real_nums = {v[k] for v in VIRTUAL_WORLD_STATE.values() for k in ("x", "y")}
        clash = ex_nums & real_nums
        assert not clash, f"格式示例的数字和真实坐标撞了 {sorted(clash)}，模型可能照抄"
    ck("prompt 含换算规则/格式/方块名对照（且自动生成）", t_prompt)

    # 8. 缺 key 时报的是人话，且不泄露任何东西
    def t_key():
        old = os.environ.pop(KEY_ENV, None)
        try:
            get_key()
            raise AssertionError("没 key 本该报错")
        except SystemExit as e:
            assert KEY_ENV in str(e), "报错信息里该说清是哪个变量"
        finally:
            if old is not None:
                os.environ[KEY_ENV] = old
    ck("缺 key 报人话", t_key)

    bad = [n for ok, n in checks if not ok]
    print("\n" + "=" * 74)
    if bad:
        print(f"  ❌ {len(bad)} 项没过: {bad}")
    else:
        print(f"  ✅ 自检全部通过（{len(checks)} 项）")
        print("     下一步: python3 src/deepseek_brain.py --dry-run   # 看一眼要发出去的 prompt")
    print("=" * 74)
    return 1 if bad else 0


# ─────────────────────────── 主流程 ───────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="阶段四：DeepSeek 决策（虚拟坐标可离线跑）",
        epilog="无参数 = 打印帮助。做实验用 --demo。")
    ap.add_argument("--say", default=None, help="直接用一句话跑一次（要联网）")
    ap.add_argument("--demo", action="store_true", help="用虚拟坐标跑一批用例并比对标准答案")
    ap.add_argument("--dry-run", action="store_true", help="只打印要发出去的 prompt，不联网")
    ap.add_argument("--selftest", action="store_true", help="离线自检，不联网不花钱")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"模型名，默认 {DEFAULT_MODEL}（还有一个 deepseek-flash）")
    ap.add_argument("--show-reasoning", action="store_true", help="打印模型的思考过程")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每条指令跑几轮（>1 用来检验稳定性，会成倍花 tokens）")
    args = ap.parse_args()

    if args.selftest:
        return cmd_selftest()
    if args.dry_run:
        msgs = build_messages(VIRTUAL_WORLD_STATE, args.say or "把绿色方块放到红色方块右侧")
        for m in msgs:
            print(f"\n{'=' * 74}\n[{m['role']}]\n{'=' * 74}\n{m['content']}")
        print(f"\n（--dry-run 没有联网，也没读 {KEY_ENV}）")
        return 0
    if args.demo:
        return cmd_demo(args.model, args.show_reasoning, max(1, args.repeat))
    if args.say:
        got, r = run_one(args.say, VIRTUAL_WORLD_STATE, args.model, args.show_reasoning)
        print(pretty(got))
        print(f"\n  tokens={r['usage'].get('total_tokens')}")
        return 0

    ap.print_help()
    print("\n提示: 先 python3 src/deepseek_brain.py --dry-run")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
