#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_pick_test.py —— 拿**现有的四个测量值**算一张「纸面 → 机械臂」映射，再跑一个小抓取实验
=====================================================================================

为什么有这么一个脚本
--------------------
四个点量出来互不自洽: 仿射残差 6.70mm，形状是个**梯形** —— 下排两点在机械臂坐标里
的横向间距（174 → 148）比上排短了约 26mm，而纸上下排几乎等长（213 / 214mm）。
残差只能说明「有一个点不对劲」，它说不出是**哪个**，更说不出**该怎么改**。

本脚本**不做任何重新测量**，只做两件事:

 1) 把现有四个数**按两种解释**各拟一张映射，摆在一起比 ——
      · A 机: 四点都当作各码的「数据区右上角」（= 现状）
      · B 机: P4 那一笔当作该码的「数据区左上角」（正好差一个数据区宽 26.0mm）
    判据用本项目一直在用的**机器常数**（同一台机器应一成不变: 纸X≈0.81、纸Y≈0.94）:
      A 机算出 纸X×0.7542  → 和 0.81 差 0.056，对不上
      B 机算出 纸X×0.8042  → 和 0.81 差 0.006，几乎重合
    这也是唯一**不用动机械臂**就能分出 A/B 的判据。
    ★ 上面这几个数（6.70mm、0.7542/0.8042）是**最初那次实测**记下来的，
      用来讲清「为什么要比 A/B」；换底座/重测四点后绝对值会变 ——
      当前值以 `--selftest` 打印的「实测四点: A 残差 … / B 残差 …」为准。

    ★ 为什么残差分不出来: 四点里"把 P4 挪一个数据区宽"和"把上排整体挪一点"
      在仿射里是可以互相顶掉的（差一个剪切），残差都落在 1.6mm。
      而机器常数（各轴比例）是物理量，不会被这类"顶账"骗过去。

 2) 给一个抓取小实验，让机械臂把「哪套解释对」**验**出来:
    把方块放在纸上一个看得见、写死的位置，让吸盘分别悬停到 A / B 两套映射给出的点。
    两点相差十几毫米（方块才二十几毫米），哪个对准方块一眼就能看出来。
    定案之后再真正抓一次（默认放到纸面正中）。

★★ 选点曾经有一条硬约束（第一版就栽在这，软限位放宽后已解除）:
   纸面 y 越大 → 机械臂 x 越小，纸下缘 y=196.5 落到 x≈83。
   原来软限位 x 下限是 100，比 83 还高 —— **下排两个码的中心和底边够不着**，
   所以下方那个码的 *中心* (P4.c) 不能当验点，只能挑下排的**上边**
   (P4.tr / P4.tl，y=170.5 → x≈107) 或者上排的点。
   2026-09-18 下限放宽到 50 之后，纸上最低点 P3.bl（X=78.3）也在盒内，
   下排的中心/底边都能去了 —— 不再有被迫跳过的验点。
   下面那段"够不够得着"的检查仍然保留、而且要留着: 它是拿**当前** limits
   现算的，换机器、改软限位都会跟着变，不能靠这段注释代替。

用法（在**项目根目录**下执行）
----
    python3 src/step4_pick_test.py                       # 只看账: 映射对比 + 各验收点预测（不连臂）
    python3 src/step4_pick_test.py --at P4.tr            # 某个纸面点在两套映射下各是什么坐标
    python3 src/step4_pick_test.py --probe               # 连臂触底，量出纸面 Z（不做别的运动）
    python3 src/step4_pick_test.py --at P4.tr --go       # 连臂: 依次悬停 A、B 两点，人工看哪个准
    python3 src/step4_pick_test.py --at P2.br --go       # 另一个独立复核点（离任何教点都远）
    python3 src/step4_pick_test.py --at P4.tr --go --model b --pick
                                                         # 用 B 机真正抓一次，放到纸面中心
    python3 src/step4_pick_test.py --selftest            # 离线自检，不用机械臂

纸面点怎么写（--at / --drop）
    P4.c        右下码的数据区中心     P4.tl / tr / br / bl  该码数据区的四角
    P2.br       右上码数据区右下角（复核点）
    paper.c     整张纸的正中（四个码中间的空白处）
    242.45,183.5   直接写纸面 mm

安全
----
    · 运动全部走 step2_teach_coords.py 里那一套（PTP、软限位、到位回读、Ctrl-C 立即停）
    · **不给 --table-z 就一步都不动** —— 没有纸面高度，Z 就没有下限
    · 不会回零、不会写 SetHOMEParams / SetEndEffectorParams（那两项出过事故）
    · 退出时一定关真空泵
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qr_vision import (REFERENCE_LABEL, affine_fit, homography_apply,   # noqa: E402
                       homography_fit, load_paper_layout)
from paths import (PAPER_JSON, PICK_RESULT_JSON as RESULT_JSON,          # noqa: E402
                   ROBOT_POINTS_JSON as ROBOT_JSON, ensure_output_dir)

# ═══════════════════════════════════════════════════════════════════
#  四个测量值 —— **已清空，等重测**
#
#  ★ 2026-09-17 底座高度换过，旧读数全部作废（X/Y 比例和 Z 都会变），
#    所以这里不再内置兜底值：没有 output/robot_points.json 就直接报错，
#    免得拿旧数悄悄算出一堆"看起来对"的结果。
#
#  重测方法（在项目根目录下）:
#      python3 src/step2_teach_coords.py --reference corner_tr
#      （对完四个码，**按 q 结束** —— 按 q 才落盘，Ctrl-C 不存）
#    落盘后 output/robot_points.json 会被本脚本自动读走。
# ═══════════════════════════════════════════════════════════════════
DEFAULT_ROBOT_XY: dict[str, list[float]] | None = None

# 机器常数（**旧底座**实测: 纸X≈0.81 / 纸Y≈0.94）。
# 纸面 x 方向走 100mm，机械臂报 ≈81mm；纸面 y 方向走 100mm，报 ≈94mm。
# ★ 两个轴**不一样**，所以判据是"两个数分别比"，不能拿几何平均去比。
# ⚠ 换底座后这个比例可能变 —— 重测完照 k_dist() 报的数核对一遍再改这里。
MACHINE_K = (0.81, 0.94)

# ═══════════════════════════════════════════════════════════════════
#  三套映射（都只用那四个数）
#    a / b —— 争的是「P4 那一笔瞄的是哪个角」，各拟一张**仿射**（有残差，能当判据）
#    h     —— 标签同 A，但改用**单应**插值；精确穿过四个教点（残差恒 0，不是判据）
# ═══════════════════════════════════════════════════════════════════
MODELS: dict[str, dict] = {
    "a": {
        "kind": "affine",
        "short": "A 四点都按「数据区右上角」",
        "long": "A 机: 四个点都当作各码的「数据区右上角」（现状）",
        "refs": {"P1": "corner_tr", "P2": "corner_tr",
                 "P3": "corner_tr", "P4": "corner_tr"},
    },
    "b": {
        "kind": "affine",
        "short": "B P4 按「数据区左上角」",
        "long": "B 机: P4 那一笔当作该码的「数据区左上角」（差一个数据区宽）",
        "refs": {"P1": "corner_tr", "P2": "corner_tr",
                 "P3": "corner_tr", "P4": "corner_tl"},
    },
    "h": {
        "kind": "homography",
        "short": "H 四点单应（精确穿过四个教点）",
        "long": "H 机: 标签同 A（四点都是「数据区右上角」），但用**单应**插值 —— 精确穿过四个教点",
        "refs": {"P1": "corner_tr", "P2": "corner_tr",
                 "P3": "corner_tr", "P4": "corner_tr"},
    },
}


@dataclass
class Model:
    key: str
    long: str
    short: str
    refs: dict           # {码: 参考点模式}
    kind: str            # "affine" / "homography"
    M: np.ndarray        # 2x3 仿射（单应机为 None）: XY = 纸面mm @ M[:,:2].T + M[:,2]
    H: np.ndarray        # 3x3 单应（仿射机为 None）
    worst: float         # 四点最大残差 mm（单应机恒为 0，**不是**准确度的证据）
    scale_x: float       # 纸面 X 方向比例（机器常数）: 单应机取纸心的局部比例
    scale_y: float       # 纸面 Y 方向比例
    det: float           # 行列式（正确必为负，见 qr_vision.affine_winding_bad）

    def xy(self, paper_xy) -> tuple[float, float]:
        if self.kind == "homography":
            return homography_apply(self.H, paper_xy)
        p = np.asarray(paper_xy, dtype=np.float64)
        q = p @ self.M[:, :2].T + self.M[:, 2]
        return float(q[0]), float(q[1])


# ─────────────────────────── 四个测量值 ───────────────────────────
def load_robot_xy() -> tuple[dict, str]:
    """读出那四个测量值。返回 ({码: [X, Y]}, 来源说明)。没有就直接报错，不兜底。"""
    if not ROBOT_JSON.exists():
        raise SystemExit(
            f"缺少 {ROBOT_JSON.name} —— 四个机械臂坐标还没重测。\n"
            "  底座高度换过，旧读数已作废，本脚本不再内置兜底值。\n"
            "  请先跑:\n"
            "      python3 src/step2_teach_coords.py --reference corner_tr\n"
            "  对完四个码后**按 q 结束**（只有按 q 才会落盘）。"
        )
    d = json.loads(ROBOT_JSON.read_text(encoding="utf-8"))
    coords = {k: [float(v[0]), float(v[1])] for k, v in d["dobot_coords"].items()}
    # ★ 来源照文件自己写的念 —— 别写死成"step2_teach_coords.py 落盘的"，
    #   手工整理/加过冗余点的文件也走这条路，写死就撒谎了。
    return coords, f"{ROBOT_JSON.name}：{d.get('source', '（文件没写来源）')}"


def load_table_z() -> float | None:
    """纸面 Z: 优先用之前 --probe 记下的，其次用 step2_teach_coords.py 落盘的。"""
    for path, key in ((RESULT_JSON, "table_z"), (ROBOT_JSON, "table_z")):
        if path.exists():
            v = json.loads(path.read_text(encoding="utf-8")).get(key)
            if v is not None:
                return float(v)
    return None


# ─────────────────────────── 拟合 ───────────────────────────
def _local_jac(model_xy, p, eps: float = 0.5) -> np.ndarray:
    """
    映射在纸面点 p 处的局部雅可比（2x2）: 第 0 列 = ∂(机械臂XY)/∂(纸面x)。

    仿射处处一样；单应处处不同，所以单应机报的是**纸心**那一处的比例。
    """
    ex = (np.asarray(model_xy((p[0] + eps, p[1])))
          - np.asarray(model_xy((p[0] - eps, p[1])))) / (2 * eps)
    ey = (np.asarray(model_xy((p[0], p[1] + eps)))
          - np.asarray(model_xy((p[0], p[1] - eps)))) / (2 * eps)
    return np.column_stack([ex, ey])


def fit_models(layout, robot_xy: dict) -> dict[str, Model]:
    """每套解释各拟一张映射。仿射走 qr_vision.affine_fit —— 和项目其它脚本同一套判据。"""
    centre = (layout.paper_mm[0] / 2.0, layout.paper_mm[1] / 2.0)
    out = {}
    for key, spec in MODELS.items():
        paper = {c: layout.points_mm[spec["refs"][c]][c] for c in layout.codes}
        if spec["kind"] == "homography":
            H = homography_fit(paper, robot_xy, layout.codes)
            if H is None:
                raise SystemExit("单应拟合失败（点不够或共线）")
            m = Model(key=key, long=spec["long"], short=spec["short"], refs=spec["refs"],
                      kind="homography", M=None, H=H, worst=0.0,
                      scale_x=0.0, scale_y=0.0, det=0.0)
            J = _local_jac(m.xy, centre)
        else:
            aff = affine_fit(paper, robot_xy, layout.codes)
            if aff is None:
                raise SystemExit("四点拟合失败（点不够或共线）")
            M, worst = aff
            m = Model(key=key, long=spec["long"], short=spec["short"], refs=spec["refs"],
                      kind="affine", M=M, H=None, worst=worst,
                      scale_x=0.0, scale_y=0.0, det=0.0)
            J = M[:, :2]
        m.scale_x = float(np.linalg.norm(J[:, 0]))
        m.scale_y = float(np.linalg.norm(J[:, 1]))
        m.det = float(np.linalg.det(J))
        out[key] = m
    return out


def k_dist(m: Model) -> float:
    """模型的两轴比例和机器常数差多远（越小越像这台机器）。"""
    return abs(m.scale_x - MACHINE_K[0]) + abs(m.scale_y - MACHINE_K[1])


def compare_pair(models: dict, key: str) -> tuple[str, str]:
    """
    悬停对比要停哪两个点。

    · 标签之争（a/b）已经由悬停实验定了案 → 现在默认比的是 **H 与 A**
      （"梯形到底是不是透视造成的"），因为两者只在四个教点之外才分开；
    · 想回头看 a/b 之争就写 --vs b。
    """
    if key == "h":
        return "h", ("a" if "a" in models else "b")
    other = next((k for k in ("h", "a", "b") if k != key and k in models), None)
    return key, other


# ─────────────────────────── 纸面点解析 ───────────────────────────
CORNER_ALIAS = {"tl": "corner_tl", "tr": "corner_tr",
                "br": "corner_br", "bl": "corner_bl", "c": "center"}


def default_limits() -> dict:
    """软限位。只用 step2_teach_coords.py 那一套，不在这里另立一份。"""
    import step2_teach_coords as tc
    return {a: tuple(v) for a, v in tc.DEF_LIMITS.items()}


def limits_from_args(args) -> dict:
    """软限位 = step2_teach_coords.py 那一套 + --limits 覆盖。连接前也要用，所以独立成函数。"""
    limits = default_limits()
    if getattr(args, "limits", None):
        for spec in args.limits.split(","):
            ax, lo, hi = spec.split(":")
            limits[ax.strip()] = (float(lo), float(hi))
    return limits


def why_unreachable(limits: dict, xy) -> str:
    """
    这个机械臂坐标在不在软限位里？在 → 返回空串；不在 → 说清哪个轴、差多少。

    ★ 为什么要在**连接机械臂之前**就查: 四个点都在纸的下半边，而纸面 y 越大
      机械臂 x 越小（纸上往下 = 往底座方向）。纸下缘（y=196）已经落到 x≈83。
      软限位下限原来是 100，比 83 还高，那些点机械臂根本不会去 —— 那时全靠
      这条提前拦下来。现在下限放宽到 50，纸上的点都进盒了，但这个检查不能删:
      limits 是**参数**（--limits 能改、换机器会重设），够不够得着得现算。
      而且一旦有哪个点在盒外（比如以后换张更大的纸），等到连上、动完一半
      再被 check_target 拒掉，既白跑一趟，也让人以为"机械臂坏了"。
      改成先查、并且直接给出一个能去的替代点。
    """
    bad = []
    for ax, v in (("x", float(xy[0])), ("y", float(xy[1]))):
        lo, hi = limits[ax]
        if v < lo:
            bad.append(f"{ax.upper()}={v:.1f} 低于下限 {lo:.0f}")
        elif v > hi:
            bad.append(f"{ax.upper()}={v:.1f} 超过上限 {hi:.0f}")
    return "；".join(bad)


def landmarks(layout) -> dict:
    """纸上能指认的点: 四个码的中心/四角 + 整张纸中心。"""
    out = {"paper.c": (layout.paper_mm[0] / 2, layout.paper_mm[1] / 2)}
    for c in layout.codes:
        for alias, mode in CORNER_ALIAS.items():
            out[f"{c}.{alias}"] = tuple(layout.points_mm[mode][c])
    return out


def landmark_rows(layout, models, limits, pair=("a", "b")) -> list:
    """每个可指认的纸面点: 两套映射的坐标、相差多少、能不能去。按相差从大到小排。"""
    k1, k2 = pair
    rows = []
    for tok, p in landmarks(layout).items():
        a = models[k1].xy(p)
        b = models[k2].xy(p)
        d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
        why = why_unreachable(limits, b) or why_unreachable(limits, a)
        rows.append((d, tok, p, a, b, why))
    rows.sort(reverse=True, key=lambda r: r[0])
    return rows


# 参考点模式 → 角别名（CORNER_ALIAS 的反表）
ALIAS_OF = {v: k for k, v in CORNER_ALIAS.items()}


def taught_tokens(layout, models, pair) -> set:
    """
    两套映射**都**教过的纸面点（token 写法，取交集）—— 拿它当验点是白验。

    ★ 为什么取交集、不是并集:
      · 相同标签的一对（如 h 与 a）都穿过同样四个教点 —— 在这些点上两者的差
        只是"单应精确/仿射偏一个残差"，是**构造**出来的，验不出插值准不准 → 排除。
      · 标签不同的一对（如 a 与 b）恰恰在 P4 那一点上教的是**不同**的纸面角落 ——
        那正是要验的争点（P4.tr vs P4.tl），绝不能排除。
      交集正好把这两种情况分开。
    """
    sets = []
    for k in pair:
        if k in models:
            sets.append({f"{c}.{ALIAS_OF[models[k].refs[c]]}" for c in layout.codes})
    return set.intersection(*sets) if sets else set()


def verify_candidates(layout, models, limits, pair=("a", "b"),
                      n: int = 2, min_delta: float = 5.0,
                      exclude_taught: bool = True) -> list:
    """够得着、两套映射差得够多、且（默认）不是教点的验点，按相差降序取前 n 个。"""
    skip = taught_tokens(layout, models, pair) if exclude_taught else set()
    out = []
    for d, tok, p, a, b, why in landmark_rows(layout, models, limits, pair):
        if not why and d >= min_delta and tok not in skip:
            out.append((tok, d))
        if len(out) >= n:
            break
    return out


def best_landmark(layout, models, limits, pair=("a", "b"), min_delta: float = 5.0):
    """差得够多、又能去的点里，相差最大的那个 → 推荐拿它做实验。"""
    c = verify_candidates(layout, models, limits, pair, n=1, min_delta=min_delta)
    return c[0] if c else (None, 0.0)


def parse_at(layout, token: str) -> tuple[tuple[float, float], str]:
    """把 'P4.c' / 'paper.c' / '242.4,183.5' 解析成 (纸面 mm, 说明)。"""
    t = token.strip()
    if "," in t:
        a, b = t.split(",")
        return (float(a), float(b)), f"纸面 ({float(a):.2f}, {float(b):.2f})mm"
    if "." not in t:
        raise SystemExit(f"看不懂的点 {token!r}：要写成 P4.c / paper.c / 242.4,183.5")
    who, what = t.split(".", 1)
    what = what.lower()
    if what not in CORNER_ALIAS:
        raise SystemExit(f"看不懂的角 {what!r}：可用 tl/tr/br/bl/c（c=中心）")
    if who.lower() in ("paper", "纸"):
        w, h = layout.paper_mm
        if what != "c":
            raise SystemExit("整张纸只支持 paper.c（正中心）")
        return (w / 2.0, h / 2.0), f"纸面正中心 ({w / 2:.1f}, {h / 2:.1f})mm"
    if who not in layout.codes:
        raise SystemExit(f"没有 {who!r} 这个码：可用 {layout.codes}")
    mode = CORNER_ALIAS[what]
    pt = layout.points_mm[mode][who]
    return pt, f"{who} 的「{REFERENCE_LABEL[mode]}」 ({pt[0]:.2f}, {pt[1]:.2f})mm"


# ─────────────────────────── 看账（不连机械臂）───────────────────────────
def report(layout, models: dict[str, Model], robot_xy: dict, src: str, at: str) -> None:
    print("=" * 72)
    print("  step4_pick_test.py —— 现有四点 → 纸面/机械臂映射（不重测）")
    print("=" * 72)
    print(f"标定纸: {PAPER_JSON.name}  {layout.paper_mm[0]:.0f}x{layout.paper_mm[1]:.0f}mm"
          f"   参考点: {REFERENCE_LABEL[layout.reference]}")
    print(f"测量值来源: {src}")
    for c in layout.codes:
        print(f"  {c}  X={robot_xy[c][0]:8.3f}  Y={robot_xy[c][1]:8.3f}")

    print("\n每套映射（都只用上面这四个数）")
    for m in models.values():
        flags = []
        if m.det > 0:
            flags.append("行列式为正 → 绕向反了，坐标标反")
        if m.kind == "affine":
            flags.append(f"和机器常数差 {k_dist(m):.4f}")
        else:
            flags.append("残差恒 0（8 自由度恰好穿过 4 点）—— 不是准确度证据")
        print(f"  {m.long}")
        print(f"      [{'仿射' if m.kind == 'affine' else '单应'}] "
              f"四点残差最大 {m.worst:5.2f}mm   行列式 {m.det:+.4f}   "
              f"纸X×{m.scale_x:.4f}  纸Y×{m.scale_y:.4f}")
        print(f"      [{'; '.join(flags)}]")

    aff_keys = [k for k, m in models.items() if m.kind == "affine"]
    print(f"\n机器常数判据（记录在案: 纸X≈{MACHINE_K[0]}  纸Y≈{MACHINE_K[1]}，"
          f"同一台机器应当一成不变）—— 只在**仿射**之间比，单应不适用")
    for k in aff_keys:
        m = models[k]
        dx, dy = m.scale_x - MACHINE_K[0], m.scale_y - MACHINE_K[1]
        verdict = "对得上" if k_dist(m) < 0.03 else "对不上"
        print(f"  {k.upper()}: 纸X×{m.scale_x:.4f}（差 {dx:+.4f}）  "
              f"纸Y×{m.scale_y:.4f}（差 {dy:+.4f}）  → {verdict}")
    best = min((models[k] for k in aff_keys), key=k_dist)
    print(f"  ★ 机器常数上最像这台机器的是 {best.key.upper()}"
          f"（残差 {best.worst:.2f}mm）")
    if "b" in models and "a" in models:
        print("  ★ 注意: 实测悬停实验已证明「P4 瞄的是右上角」—— A 的标签是对的。"
              "B 贴常数是因为\n     把 P4 挪一个数据区宽恰好把梯形摊平了，属于"
              "「用一个错抵消另一个错」，不是它更准。")

    print("\n各验收点的预测（同一个纸面点，各套映射各给出什么机械臂坐标）")
    print(f"  {'纸面点':10s} {'纸面 mm':>16s} {'A 机':>19s} {'B 机':>19s} {'H 机':>19s} {'H−A':>8s}")
    tokens = [at] + [t for t in ("paper.c", "P2.br", "P4.tr", "P4.tl", "P4.c") if t != at]
    for tok in tokens:
        (x, y), _ = parse_at(layout, tok)
        a = models["a"].xy((x, y))
        b = models["b"].xy((x, y))
        h = models["h"].xy((x, y))
        da = float(np.hypot(a[0] - b[0], a[1] - b[1]))
        dh = float(np.hypot(h[0] - a[0], h[1] - a[1]))
        print(f"  {tok:10s} ({x:6.1f},{y:6.1f}) ({a[0]:7.2f},{a[1]:8.2f})"
              f" ({b[0]:7.2f},{b[1]:8.2f}) ({h[0]:7.2f},{h[1]:8.2f})"
              f" {dh:6.2f}  (A/B 差 {da:.2f})")

    limits = default_limits()
    print("\n哪些纸面点能当验点 —— 现在要比的是 **H 与 A**（a/b 标签之争上一轮已定案）")
    _dd = None
    if "h" in models and "a" in models:
        _dd = float(np.hypot(*(np.subtract(
            models["h"].xy(layout.points_mm["corner_tr"]["P4"]),
            models["a"].xy(layout.points_mm["corner_tr"]["P4"])))))
    print("  ★ 教点上 H 精确落在测量值上、A 偏 "
          + (f"{_dd:.1f}mm" if _dd is not None else "一个残差")
          + "（= A 在该点的残差）——")
    print("    但单应穿过教点是**构造**出来的，只能证明 A 在教点上是错的，")
    print("    证明不了 H 在**教点之间**也对。要验插值，验点必须**不是**教点。")
    rows = landmark_rows(layout, models, limits, pair=("h", "a"))
    rows.sort(key=lambda r: (bool(r[5]), -r[0]))      # 够得着的排前面，组内按相差降序
    print(f"  {'纸面点':10s} {'纸面 mm':>16s} {'H−A':>9s}  {'去得成吗':8s} 说明")
    taught = {f"{c}.tr" for c in layout.codes}
    for d, tok, p, a, b, why in rows[:14]:
        ok_txt = "可达" if not why else "✗ 够不着"
        note = "" if not why else why
        if not why and tok in taught:
            note = ("教点: H 精确 / A 偏 "
                    + (f"{_dd:.1f}mm" if _dd is not None else "一个残差")
                    + "（构造使然，不算验证）")
        print(f"  {tok:10s} ({p[0]:6.1f},{p[1]:6.1f}) {d:7.2f}mm  {ok_txt:8s} {note}")
    cands = verify_candidates(layout, models, limits, pair=("h", "a"))
    if cands:
        tok, d = cands[0]
        print(f"\n  ★ 推荐验点: 「{tok}」—— 不是教点，H 与 A 差 {d:.2f}mm，机械臂够得着。")
        print(f"     python3 {Path(__file__).name} --at {tok} --go")
        print("     （方块放在这个纸面点上，看吸盘停到 H 还是 A 那边正对它）")
    # ★ 这里原来写死"下排码的中心/底边超出软限位下限 100，去不了"。
    #   那句在 x 下限 100 的年代是对的，2026-09-18 放宽到 50 之后就变成假话了。
    #   改成从**当前** limits 现算 —— limits 会变（--limits / 换机器 / 换纸），
    #   写死的结论迟早跟现实对不上。
    unreach = [(tok, why) for _d, tok, _p, _a, _b, why in rows if why]
    if unreach:
        print(f"  ★ 当前软限位下有 {len(unreach)} 个纸面点去不了: "
              + "；".join(f"{t}（{w}）" for t, w in unreach[:3])
              + ("…" if len(unreach) > 3 else ""))
    else:
        print("  ★ 纸面上所有可指认点都在当前软限位内 —— 验点随便挑，没有被迫跳过的。")


# ─────────────────────────── 抓取小实验 ───────────────────────────
def write_result(table_z: float, model_key: str, note: str,
                 extra: dict | None = None) -> None:
    """把「跑通了的那一套参数」记下来 —— 下次不用重猜，下一阶段也照它接。"""
    ensure_output_dir()
    RESULT_JSON.write_text(json.dumps({
        "table_z": round(float(table_z), 3),
        "model_used": model_key,
        "note": note,
        "source": "step4_pick_test.py",
        **(extra or {}),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def cube_top_z(table_z: float, obj_h: float, level: int = 0, press: float = 0.0) -> float:
    """
    吸盘该降到哪个 Z。level = 那个位置**已有几层**方块（0 = 直接放在纸面上）。

    ⚠ 下面 -78.518 / -76.5 是**换底座之前**的读数，只当算式样例看。
      换底座后 Z 全变了，真值请用 `step4_pick_test.py --probe` 重新记一次；
      本函数只做算术，table_z 由调用方传进来。

    ★★ 为什么不能直接用纸面 Z —— 这里最容易写错、后果也最硬:
      吸盘要停在**方块顶面**上，不是纸面上。纸面 Z=-78.5、方块高 25mm 时，
      抓 0 层的方块要停在 -78.5 + 25 = **-53.5**。
      按「降到纸面」写就是让吸盘往方块里扎 25mm —— 软胶嘴能顶住、
      看起来"还能抓"，但方块被压、机械臂丢步（之后零点就不准了）。
      同一层上的抓和放，Z 是同一个数（都是 cube_top_z(level)）。

    ★ 实测: 方块净高 25mm，但 --obj-h 要 **26** 才吸得住（28 抬空 1mm 吸不到）——
      这 1mm 是软吸盘唇口压在方块顶面上、密封所需的压缩量。所以这里的 obj_h 是
      **吸盘接触面到「已记纸面 Z」的距离**，不是拿卡尺量出来的方块净高。

      ★★ 由此可反推出记下的纸面 Z **不是**真实纸面:
          26 能用 → 接触面 = -78.518 + 26 = -52.518 ≈ 方块顶面 + 1mm
          → 方块顶面 ≈ -51.5；净高 25 → **真实纸面 ≈ -76.5**
        而探测记下的是 -78.518，比真实纸面低 2.0mm —— 触底时唇口被压扁约 2mm
        才判定"碰到了"。所以 -78.518 是"唇口压扁后的读数"，不是纸面。
        两种写法完全等价（差 0.02mm）:
          记下纸面 -78.518 + --obj-h 26
          真实纸面 -76.5   + --obj-h 25 --press 1
        **别把 -78.518 当物理纸面去算真实高度**，两套数别混用。

    ⚠ level>0（叠放）别直接用这个式子: 26 里那 1mm 压缩量只在「最上面那一层接触
      吸盘」时才存在，第 2 层往上要按**方块净高**（25）叠。真要叠放先把这个函数
      拆成「净高 + 一次压缩量」，别照抄 level 参数。

    ★ 试值方向: obj_h 越大 = 停得越高 = 越碰不到（会抓空）；
      越小 = 越低 = 越可能压坏方块/丢步。所以标定时**从大往小试**（28 不行试 26），
      别从小往大试。
    """
    return table_z + (level + 1) * obj_h - press


def suction(api, on: bool, isQueued: int = 1) -> int:
    """
    开关真空吸盘。返回 result（0 = 成功）。

    ★ 新版 SDK 的导出是 SetEndEffectorSuctionCup(bool enableCtrl, bool suck,
      bool isQueued, uint64_t*)，而 SDK 自带的 DobotDll.py 只包了旧版那几个，
      没有这个函数 —— 所以这里直接对着 .so 调，参数类型按 DobotDll.h 给。
    """
    idx = ctypes.c_uint64(0)
    return int(api.SetEndEffectorSuctionCup(
        ctypes.c_bool(True), ctypes.c_bool(bool(on)),
        ctypes.c_bool(bool(isQueued)), ctypes.byref(idx)))


def run_arm(args, layout, models, robot_xy) -> int:
    import step2_teach_coords as tc          # 运动/限位/报警那一套全都复用，不另写一份

    model = models[args.model]
    k1, k2 = compare_pair(models, args.model) if args.vs is None else (args.model, args.vs)
    (px, py), desc = parse_at(layout, args.at)
    tgt_xy = model.xy((px, py))

    # ── 连臂之前先查够不够得着（第一版就是动到一半被软限位拒掉）──
    if args.go or args.pick:
        limits0 = limits_from_args(args)
        checks = [(desc, k1, models[k1].xy((px, py))),
                  (desc, k2, models[k2].xy((px, py)))]
        if args.pick:
            (dx, dy), ddesc = parse_at(layout, args.drop)
            checks.append((ddesc, args.model, model.xy((dx, dy))))
        bad = []
        for what, key, xy in checks:
            why = why_unreachable(limits0, xy)
            if why:
                bad.append((what, key.upper(), xy, why))
        if bad:
            print("✗ 这个点在软限位之外，机械臂去不了 —— 还没连臂就拦下了（没白跑）。")
            for what, key, xy, why in bad:
                print(f"    {what}  按 {key} 机 → ({xy[0]:.2f}, {xy[1]:.2f})   {why}")
            print(f"  软限位: x {limits0['x'][0]:.0f}~{limits0['x'][1]:.0f}"
                  f"   y {limits0['y'][0]:.0f}~{limits0['y'][1]:.0f}")
            print("  原因: 纸面 y 越大 → 机械臂 x 越小。下排码的中心/底边（y≥183.5）"
                  "落到 x≤95，低于下限。")
            tok, d = best_landmark(layout, models, limits0, pair=(k1, k2))
            if tok:
                print(f"\n  换个够得着的验点 —— 推荐「{tok}」（两套映射差 {d:.2f}mm）:")
                print(f"      python3 {Path(__file__).name} --at {tok} --go")
            print(f"  或者跑 python3 {Path(__file__).name} 看完整的「验点表」。")
            return 2

    table_z = args.table_z if args.table_z is not None else load_table_z()
    if (args.go or args.pick) and table_z is None:
        print("✗ 要动机械臂就必须知道纸面高度 Z —— 现在没有。")
        print("  先跑一次触底量出来:")
        print(f"      python3 {Path(__file__).name} --probe")
        print("  或者直接给:  --table-z -33.8   （单位 mm，用 GetPose 的读数）")
        return 2

    # ★ 只要会动机械臂，就必须有人在终端前: 悬停对比要目视、抓取要按回车确认。
    #   没有这道闸，一条 `step4_pick_test.py --go` 喂给脚本/管道就会直接连臂动起来
    #   （--table-z 或 pick_test_result.json 里存着纸面 Z 时，前面那些 "没 Z 不动" 的
    #   拦截都拦不住它）。
    if (args.probe or args.go or args.pick) and not sys.stdin.isatty():
        print("✗ 要动机械臂，就得在**交互式终端**里跑（悬停要目视、抓取要确认）。")
        print("  现在 stdin 不是终端（被管道/重定向/后台了），一步都不动。")
        return 1

    print("=" * 72)
    print("  抓取小实验")
    print("=" * 72)
    print(f"用 {model.long}")
    print(f"纸面目标点: {desc}")
    print(f"→ 机械臂 XY = ({tgt_xy[0]:.2f}, {tgt_xy[1]:.2f})"
          f"   （四点残差 {model.worst:.2f}mm）")
    if not args.pick:
        other = models[k2]
        o = other.xy((px, py))
        print(f"   对照 {other.short}: ({o[0]:.2f}, {o[1]:.2f})"
              f"  相差 {np.hypot(o[0] - tgt_xy[0], o[1] - tgt_xy[1]):.2f}mm")
    print("\n⚠ 安全: 清空机械臂周围 60cm；Ctrl-C 随时停；本脚本不回零、不改末端参数")

    api, dType = tc.load_sdk()
    port = tc.find_port(dType, args.port)
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

    limits = limits_from_args(args)
    if table_z is not None:
        # ★ 只要知道纸面高度，就把 Z 下限锁在纸面上 —— 连"只是悬停"也锁。
        #   抓取时会去碰方块顶面（比纸面高 --obj-h），唯一放开的一点是 --press。
        floor = table_z + (args.obj_h - args.press if args.pick else 0.0)
        limits["z"] = (floor, limits["z"][1])

    held = False
    try:
        dType.SetCmdTimeout(api, 5000)
        ares, alist = tc.read_alarms(api, dType)
        print(f"[报警] result={ares}  {tc.describe_alarms(alist)}")
        if tc.has_alarms(alist):
            tc.report_alarm_detail(alist)
            if tc.needs_homing(alist):
                print("\n✗ 有丢步报警: 零点已经不可信，先回零再跑本脚本。")
                return 1
            ok, alist = tc.resolve_alarms(api, dType, alist, args.clear_alarms)
            if not ok:
                return 1

        dType.SetQueuedCmdClear(api)
        dType.SetQueuedCmdStartExec(api)
        dType.SetPTPJointParams(api, args.speed, args.speed, args.speed, args.speed,
                                args.acc, args.acc, args.acc, args.acc, isQueued=0)
        tc._set_coord_params(api, dType, args.speed, args.acc)
        dType.SetPTPCommonParams(api, args.ratio, args.ratio, isQueued=0)

        # ── 触底: 只找纸面 Z，不做别的 ──
        if args.probe:
            with tc.RawKeys() as keys:
                probe_lim = {**limits, "z": (args.probe_z, limits["z"][1])}
                z = tc.touch_off(api, dType, args, keys, probe_lim)
            if z is None:
                print("没记下纸面高度，未写入。")
                return 0
            write_result(z, args.model, "由 --probe 触底量得")
            print(f"\n已记下纸面 Z = {z:.2f}（写进 {RESULT_JSON.name}）")
            print(f"下一步: python3 {Path(__file__).name} --go"
                  "    （不给 --at 会自动挑验点）")
            return 0

        cur = tc.read_pose_stable(api, dType, 5)
        r_now = cur["r"]
        top_z = cube_top_z(table_z, args.obj_h, 0)          # 方块顶面 = 吸盘贴上去的高度
        hover_z = top_z + args.hover
        print(f"\n[当前位姿] {tc.fmt(cur)}")
        print(f"[高度] 纸面 Z={table_z:.2f} + 方块高 {args.obj_h:.1f} = 方块顶面 Z={top_z:.2f}")
        print(f"       吸盘悬停 Z={hover_z:.2f}（顶面 +{args.hover:.0f}）")

        def goto(x, y, z, why) -> bool:
            tgt = {"x": float(x), "y": float(y), "z": float(z), "r": r_now}
            ok, why_bad = tc.check_target(tc.read_pose_stable(api, dType, 5), tgt, limits)
            if not ok:
                print(f"  ✗ {why}")
                print(f"    {why_bad}")
                return False
            ok, why_bad = tc.move_to(api, dType, tgt, limits, args.mode)
            if not ok:
                print(f"  ✗ {why}: {why_bad}")
                return False
            ok, why_bad = tc.verify_arrival(api, dType, tgt)
            print(f"  {'✓' if ok else '⚠'} {why}"
                  + (f"  {why_bad}" if why_bad else ""))
            return ok

        # ── 悬停对比（不抓）: 依次停到两套映射各自的点上，人工看哪个对准方块 ──
        if not args.pick:
            p1 = models[k1].xy((px, py))
            p2 = models[k2].xy((px, py))
            print("\n即将运动（只悬停，不吸、不下降）:")
            print(f"  第 1 停 {models[k1].key.upper()} → ({p1[0]:.2f}, {p1[1]:.2f})")
            print(f"  第 2 停 {models[k2].key.upper()} → ({p2[0]:.2f}, {p2[1]:.2f})"
                  f"   （移动 {np.hypot(p2[0] - p1[0], p2[1] - p1[1]):.2f}mm）")
            print(f"  悬停高度 Z={hover_z:.2f}"
                  f"（方块顶面 {top_z:.2f} + {args.hover:.0f}，不碰方块）")
            input("确认机械臂周围没人、没障碍 → 按回车开始（Ctrl-C 中止）… ")
            order = [k1, k2]
            for i, key in enumerate(order, 1):
                m = models[key]
                x, y = m.xy((px, py))
                print(f"\n[{i}/{len(order)}] {m.short}")
                if not goto(x, y, hover_z, f"悬停到 ({x:.2f}, {y:.2f})"):
                    return 1
                if i < len(order):
                    other = models[order[i]]
                    ox, oy = other.xy((px, py))
                    print("      目视: 吸盘正中心对准方块了吗？偏了多少、往哪偏？")
                    print(f"      （下一个点会挪到 ({ox:.2f}, {oy:.2f})，"
                          f"移动 {np.hypot(ox - x, oy - y):.2f}mm）")
                    input("      看好了按回车 → ")
            print(f"\n{models[k1].key.upper()} 与 {models[k2].key.upper()} 都停过了。")
            if {k1, k2} == {"h", "a"}:
                print("  注意两者在四个教点上是**分开**的（H 精确、A 偏一个残差）——")
                print("  在教点上 H 赢是构造使然，不算数。这一次看的是**教点之间**:")
                print("  · 对准 H → 那个梯形确实是透视/局部比例造成的 → 以后用 --model h")
                print("  · 对准 A → 梯形另有原因（机器非线性），单应插值把它插歪了 → 别用 h")
                print("  · 两次都对不准（都差几个毫米）→ 四个教点撑不起整张纸，"
                      "得再教一个中间点当第五点")
            else:
                print("  哪一次正对方块，那套解释就是对的:")
                print("  · 对准 A → P4 那笔确实是「数据区右上角」→ 那个残差另有原因"
                      "（机器非线性/梯形），得另外查")
                print("  · 对准 B → P4 那笔是「数据区左上角」→ 四点自洽到 1.6mm")
            cands = [t for t, _ in verify_candidates(layout, models, limits, pair=(k1, k2))]
            other_tok = next((t for t in cands if t.lower() != args.at.strip().lower()),
                             None)
            if other_tok:
                print(f"\n★ 复核一次（强烈建议）: 方块挪到纸面「{other_tok}」，再跑一遍")
                print(f"    python3 {Path(__file__).name} --at {other_tok} --go"
                      + ("" if args.vs is None else f" --vs {k2}"))
                print("  两次都指向同一套，才算定案 —— 同一个点万一看花眼了不算数。")
            print(f"\n定案后真抓一次:\n  python3 {Path(__file__).name} --at {args.at} "
                  f"--go --model {k1} --pick --obj-h {args.obj_h:.0f}"
                  f"   （--model 换成看到准的那套）")
            print(f"  ★ 下降停 Z={cube_top_z(table_z, args.obj_h):.2f}"
                  f"（纸面 {table_z:.2f} + 物高 {args.obj_h:.1f}）= 方块顶面；"
                  f"不是纸面 {table_z:.2f}")
            return 0

        # ── 真抓 ──
        (dx, dy), ddesc = parse_at(layout, args.drop)
        drop_xy = model.xy((dx, dy))
        grab_z = cube_top_z(table_z, args.obj_h, 0, args.press)
        # ★ 放置不能带 --press: 抬着的那块方块底面本来就正好落在顶面位置上，
        #   再预压就是把方块摁进纸面（吸盘放气前压住 → 方块被推歪/划出去）。
        drop_z = cube_top_z(table_z, args.obj_h, 0)
        print(f"\n放置点: {ddesc} → ({drop_xy[0]:.2f}, {drop_xy[1]:.2f})")
        print(f"抓取点: ({tgt_xy[0]:.2f}, {tgt_xy[1]:.2f})")
        print(f"抓取下降: Z={grab_z:.2f} = 纸面 {table_z:.2f} + 物高 {args.obj_h:.1f}"
              f"{'' if args.press == 0 else f' − 预测压 {args.press:.1f}'}"
              f"   ★ 停在**方块顶面**，不是纸面")
        if args.press:
            print(f"放置下降: Z={drop_z:.2f}（不预压 —— 让方块底面正好落回纸面）")
        # ★ --pick 是一条道走到黑的，没有悬停对比那一步 —— 提醒一下，别拿没定案的映射真抓
        other = models[k2]
        o = other.xy((px, py))
        gap = float(np.hypot(o[0] - tgt_xy[0], o[1] - tgt_xy[1]))
        if gap > 5.0:
            print(f"\n⚠ 注意: 你选了 {model.key.upper()} 机。同一个纸面点，"
                  f"{other.key.upper()} 机会停在 ({o[0]:.2f}, {o[1]:.2f})，"
                  f"两者差 {gap:.1f}mm ——")
            print(f"   选错了就是偏 {gap:.1f}mm 抓空。没做过悬停对比的话，"
                  f"先 Ctrl-C，改用:")
            print(f"      python3 {Path(__file__).name} --at {args.at} --go"
                  f" --model {model.key}    （先看哪套对准，再回来 --pick）")
        print("\n请确认: 方块正放在纸面目标点上、吸盘干净、真空泵接好。")
        input("回车开始抓取（Ctrl-C 中止）… ")

        if not goto(tgt_xy[0], tgt_xy[1], hover_z, "悬停到抓取点上方"):
            return 1
        if not goto(tgt_xy[0], tgt_xy[1], grab_z, "下降贴住方块顶面"):
            return 1
        rc = suction(api, True)
        print(f"  {'✓' if rc == 0 else '✗'} 开真空 result={rc}")
        if rc != 0:
            return 1
        held = True
        time.sleep(args.pump_s)
        if not goto(tgt_xy[0], tgt_xy[1], hover_z, "抬起"):
            return 1
        if not goto(drop_xy[0], drop_xy[1], hover_z, "移到放置点上方"):
            return 1
        if not goto(drop_xy[0], drop_xy[1], drop_z, "下降放置"):
            return 1
        rc = suction(api, False)
        time.sleep(0.4)
        held = False
        print(f"  {'✓' if rc == 0 else '✗'} 关真空（放料） result={rc}")
        if not goto(drop_xy[0], drop_xy[1], hover_z, "抬起离开"):
            return 1
        write_result(table_z, model.key, "抓取成功收尾", extra={
            "at": args.at, "drop": args.drop,
            "obj_h": args.obj_h, "press": args.press,
            "grab_z": round(grab_z, 3), "drop_z": round(drop_z, 3),
            "grab_xy": [round(tgt_xy[0], 2), round(tgt_xy[1], 2)],
            "drop_xy": [round(drop_xy[0], 2), round(drop_xy[1], 2)],
        })
        print("\n✅ 抓取流程走完。看方块落点:")
        print(f"   目标 {ddesc}")
        print("   落点偏了多少、往哪偏 → 就是这张映射在该位置的误差。")
        return 0
    finally:
        try:
            if held:
                suction(api, False, isQueued=0)
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


# ─────────────────────────── 离线自检 ───────────────────────────
def selftest() -> int:
    """不用机械臂，验证拟合/解析这套账算得对。"""
    layout = load_paper_layout(PAPER_JSON)
    ok_all = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok_all
        ok_all = ok_all and cond
        print(f"  {'✓' if cond else '✗'} {name}")

    # ① 造两套**完全自洽**的数，各按一种解释造 —— 判据必须左右都能分出来:
    #      · 按 A 造（四点都在各码右上角）→ A 残差 ≈0，B 残差必须很大
    #      · 按 B 造（P4 在左上角）      → B 残差 ≈0，A 残差必须很大
    #    ★ 这一步正是"残差能不能认账"的检验: 真出现那种形状时，错的那套必须报大数。
    true_L = np.array([[-0.81, 0.0], [0.0, 0.94]])      # 含镜像: det<0
    true_t = np.array([210.0, -20.0])
    codes = layout.codes

    def synth(which: str) -> dict:
        paper = {c: layout.points_mm[MODELS[which]["refs"][c]][c] for c in codes}
        return {c: (np.asarray(paper[c]) @ true_L.T + true_t).tolist() for c in codes}

    ma = fit_models(layout, synth("a"))
    mb = fit_models(layout, synth("b"))
    print(f"    造数按 A: A 残差 {ma['a'].worst:.2f}mm  B 残差 {ma['b'].worst:.2f}mm")
    print(f"    造数按 B: A 残差 {mb['a'].worst:.2f}mm  B 残差 {mb['b'].worst:.2f}mm")
    check("造数按 A: 认得出 A（残差≈0）、且认得出 B 是错的（残差 >4mm）",
          ma["a"].worst < 1e-6 and ma["b"].worst > 4.0)
    check("造数按 B: 认得出 B（残差≈0）、且认得出 A 是错的（残差 >4mm）",
          mb["b"].worst < 1e-6 and mb["a"].worst > 4.0)
    check("造数按 B: 解回的机器常数正是造数用的 0.81 / 0.94",
          abs(mb["b"].scale_x - 0.81) < 1e-6 and abs(mb["b"].scale_y - 0.94) < 1e-6)
    check("造数按 B: 行列式为负（镜像）", mb["b"].det < 0)
    check("造数按 B: 中间的纸面点预测精确",
          abs(mb["b"].xy((148.5, 105.0))[0] - (np.array([148.5, 105.0]) @ true_L.T
                                               + true_t)[0]) < 1e-6)

    # ② 用**实际量到的那四个数**: B 解释必须比 A 解释更贴合这四点。
    #    ⚠ 这一节吃 robot_points.json，没重测就跳过（上面顶是纯造数，照跑不误）。
    #      ★ 断言一律写成「两套解释**相对**比较」，不写死具体毫米数 —— 重测四点后
    #        绝对残差会变，但「B 更贴」这个相对关系才是这一节真正要守的东西。
    have_real = ROBOT_JSON.exists()
    real = mr = None
    if not have_real:
        print(f"    ⊘ 跳过 ②④⑤⑥（还没有 {ROBOT_JSON.name}，等重测四点）")
    else:
        real, _ = load_robot_xy()
        mr = fit_models(layout, real)
        print(f"    实测四点: A 残差 {mr['a'].worst:.2f}mm / 常数差 {k_dist(mr['a']):.4f}"
              f"   B 残差 {mr['b'].worst:.2f}mm / 常数差 {k_dist(mr['b']):.4f}")
        check("实测四点: 残差 A 大于 B（B 把 P4 当左上角更贴合这四点）",
              mr["a"].worst > mr["b"].worst)
        check("实测四点: B 机比 A 机更贴机器常数",
              k_dist(mr["b"]) < k_dist(mr["a"]))
        check("实测四点: 两套解释的取向都对（行列式为负）",
              mr["a"].det < 0 and mr["b"].det < 0)

    # ③ 纸面点解析
    p, _ = parse_at(layout, "paper.c")
    check("paper.c = 纸面正中心", abs(p[0] - layout.paper_mm[0] / 2) < 1e-9)
    p, _ = parse_at(layout, "P4.c")
    check("P4.c = 右下码数据区中心", abs(p[0] - 242.45) < 0.01 and abs(p[1] - 183.5) < 0.01)
    p, _ = parse_at(layout, "P4.tl")
    check("P4.tl 与 P4.tr 差一个数据区宽 26.0mm",
          abs(layout.points_mm["corner_tr"]["P4"][0] - p[0] - 26.0) < 0.1)
    p, _ = parse_at(layout, "148.5,105")
    check("直接写 mm 也认", abs(p[0] - 148.5) < 1e-9 and abs(p[1] - 105.0) < 1e-9)

    if mr is None:
        print("    ⊘ 跳过 ④⑤⑥（同上，吃实测四点）")
    else:
        # ④ 悬停点差得够大（实验才分得出来）
        (x, y), _ = parse_at(layout, "P4.c")
        d = float(np.hypot(*(np.subtract(mr["a"].xy((x, y)), mr["b"].xy((x, y))))))
        print(f"    验收点 P4.c 上 A/B 相差 {d:.2f}mm")
        check("验收点 P4.c 上两套映射相差 >12mm（肉眼可分辨）", d > 12.0)

        # ⑤ 单应机 H: 必须精确穿过四个教点，**但残差 0 不能当判据** ——
        #    单应 8 自由度 / 四点 8 方程，恰好定死，换套标签也一样精确穿过。
        #    这条自检就是要把这个性质钉住: 用 B 的标签造数、按 B 的标签拟单应，
        #    残差照样 0 —— 说明它**不区分**标签对错。
        check("H 精确穿过四个教点（残差 <1e-9）", mr["h"].worst < 1e-9)
        check("H 在教点上的映射 = 测量值本身",
              all(float(np.hypot(*(np.subtract(mr["h"].xy(layout.points_mm["corner_tr"][c]),
                                               real[c])))) < 1e-9 for c in codes))
        check("H 的取向也对（纸心局部行列式为负）", mr["h"].det < 0)
        hb = fit_models(layout, synth("b"))
        check("★ 同一套数据下单应残差照样 ≈0（无论标签对错）→ 单应残差不能当判据",
              hb["h"].worst < 1e-9)
        check("★ 而仿射会说话: 同一套数据 A 残差 5.26mm（>4mm）", hb["a"].worst > 4.0)
        # H 与 A 在**教点**上差多少: H 精确=测量值，A 差一个（该点的）残差。
        # ★ 不写死具体毫米数（会随重测变），只钉「等于 A 在该点的残差、不超过 A 的 worst」。
        dd_teach = float(np.hypot(*(np.subtract(
            mr["h"].xy(layout.points_mm["corner_tr"]["P4"]),
            mr["a"].xy(layout.points_mm["corner_tr"]["P4"])))))
        print(f"    H 与 A 在教点 P4.tr 上相差 {dd_teach:.2f}mm（= A 在该点的残差）")
        check("★ 教点上 H 精确、A 偏「A 在该点的残差」（0 < 值 ≤ A 的 worst）",
              0.0 < dd_teach <= mr["a"].worst + 1e-6)
        cands = verify_candidates(layout, models=mr, limits=default_limits(), pair=("h", "a"))
        print(f"    H−A 验点候选（非教点、够得着）: {[(t, round(v, 2)) for t, v in cands]}")
        check("★ 有够得着、差 >5mm、且**不是教点**的验点（才验得出插值对不对）",
              bool(cands) and not any(t.endswith(".tr") for t, _ in cands))

        # ⑥ 软限位: x 下限放宽到 50 之后（2026-09-18），纸上的点全进盒 ——
        #    P4.c 这类原来看来"够不着"的点现在都能去。推荐点也得够得着。
        #    ⚠ 这条断言的**方向**是跟着 DEF_LIMITS["x"][0] 走的:
        #      下限 ≤ 78（纸上最低点 P3.bl）→ 全可达；哪天又调回 100 → 又变回"拦下"。
        lim = default_limits()
        (x, y), _ = parse_at(layout, "P4.c")
        why = why_unreachable(lim, mr["b"].xy((x, y)))
        print(f"    P4.c 按 B 机 {mr['b'].xy((x, y))[0]:.1f} → {why or '可达'}"
              f"   （x 下限现在是 {lim['x'][0]:.0f}）")
        check(f"x 下限已放宽 → 纸上所有点都够得着（含 P4.c）"
              f"   下限 {lim['x'][0]:.0f}，P4.c X={mr['b'].xy((x, y))[0]:.1f}"
              + (f"，仍被拦: {why}" if why else ""),
              lim["x"][0] <= 78.0 and not why)
        (x, y), _ = parse_at(layout, "P4.tr")
        check("P4.tr 可达", not why_unreachable(lim, mr["b"].xy((x, y))))
        tok, dd = best_landmark(layout, models=mr, limits=lim)
        print(f"    best_landmark(a/b) → {tok}（差 {dd:.2f}mm）")
        check("推荐验点存在、够得着、且相差 >10mm",
              bool(tok) and dd > 10.0 and not why_unreachable(lim, mr["b"].xy(
                  parse_at(layout, tok)[0])))
        # 默认对比对: h 应配 a
        check("compare_pair(h) → ('h','a')", compare_pair(mr, "h") == ("h", "a"))
        check("compare_pair(b) 里含 h（默认拿 h 当对照）", compare_pair(mr, "b")[1] == "h")
        # 教点排除规则（交集）: 同标签对排除全部四点；a/b 对要留下 P4 那两个争点
        check("taught_tokens(h,a) = 四个 .tr（全排除，插值验点不能用教点）",
              taught_tokens(layout, mr, ("h", "a")) == {"P1.tr", "P2.tr", "P3.tr", "P4.tr"})
        check("taught_tokens(a,b) 只排除共同那三点，留下 P4.tr / P4.tl 两个争点",
              taught_tokens(layout, mr, ("a", "b")) == {"P1.tr", "P2.tr", "P3.tr"})

    # ⑦ 抓取高度: 吸盘停在**物体顶面**，不是纸面
    #    ★ 这条最要命: 写错就是让吸盘往方块里扎一个物高
    #    ⚠ -78.518 是**换底座之前**记下的纸面 Z，这里只当算术样例。
    #      它只做加减法，换成任何数结论都一样；真表高以 --probe 重测为准。
    cz = cube_top_z(-78.518, 25.0)
    print(f"    纸面 -78.518 + 物高 25 → 抓取 Z={cz:.3f}（不是 -78.518）")
    check("cube_top_z: 抓 0 层的方块 = 纸面 + 物高（-53.518，不是 -78.518）",
          abs(cz - (-53.518)) < 1e-6)
    check("cube_top_z: 抓 1 层的方块 = 纸面 + 2×物高",
          abs(cube_top_z(-78.518, 25.0, 1) - (-28.518)) < 1e-6)
    check("cube_top_z: --press 是压过顶面的量（往下压）",
          abs(cube_top_z(-78.518, 25.0, 0, 1.5) - (-55.018)) < 1e-6)
    check("★ 物高 0 时退化成纸面（--obj-h 0 等价于旧行为）",
          abs(cube_top_z(-78.518, 0.0) - (-78.518)) < 1e-6)
    # 实测配方（用户跑通的那一套）: 记下纸面 -78.518 + --obj-h 26 → 接触面 -52.518
    check("★ 实测配方: -78.518 + --obj-h 26 → 吸盘接触面 -52.518",
          abs(cube_top_z(-78.518, 26.0) - (-52.518)) < 1e-6)
    check("★ --obj-h 越大停得越高（28 比 26 高 2mm → 抬空吸不到）",
          cube_top_z(-78.518, 28.0) > cube_top_z(-78.518, 26.0))
    # ★ 两种写法必须给同一个高度 —— 这是"记下纸面 ≠ 真实纸面"的账
    a = cube_top_z(-78.518, 26.0)                 # 记下纸面 + obj_h 26
    b = cube_top_z(-76.5, 25.0, 0, 1.0)           # 真实纸面 + 净高 25 − 压 1
    print(f"    记下纸面-78.518+obj_h26 = {a:.3f} | 真实纸面-76.5+净高25-压1 = {b:.3f}")
    check("★ 两种写法等价（<0.05mm）→ 反推真实纸面 ≈ -76.5（用户原话）",
          abs(a - b) < 0.05)
    check("★ 真实纸面 -76.5 与探测读数 -78.518 差约 2mm（唇口被压扁）",
          abs((-76.5) - (-78.518) - 2.0) < 0.1)

    print("\n" + ("自检通过 ✅" if ok_all else "自检失败 ✗"))
    return 0 if ok_all else 1


# ─────────────────────────── 入口 ───────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="用现有四点算「纸面→机械臂」映射，并跑一个小抓取实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例（在项目根目录下）:\n"
               "  python3 src/step4_pick_test.py                        # 只看账（不连臂）\n"
               "  python3 src/step4_pick_test.py --at P3.tl             # 看某个纸面点各套映射的预测\n"
               "  python3 src/step4_pick_test.py --probe                # 触底量纸面 Z\n"
               "  python3 src/step4_pick_test.py --at P3.tl --go        # 悬停对比 H 与 A（默认）\n"
               "  python3 src/step4_pick_test.py --at P3.tl --go --vs a # 同上，显式指定对比对象\n"
               "  python3 src/step4_pick_test.py --at P4.tr --go --vs b # 回头看 a/b 标签之争\n"
               "  python3 src/step4_pick_test.py --at P3.tl --go --model h --pick --obj-h 25\n"
               "     （--obj-h 25 = 物体高 2.5cm → 下降停在 纸面Z+25 的**顶面**）\n"
               "     ★ 纸面 -78.5 时抓取 Z 是 -53.5，不是 -76.5/-78.5\n")
    ap.add_argument("--selftest", action="store_true", help="离线自检，不连机械臂")
    ap.add_argument("--at", default=None,
                    help="纸面目标点: P3.tl / P4.tr / paper.c / 242.45,183.5"
                         "（默认自动挑「两套映射差最大、又够得着」的验点；"
                         "下排码的中心/底边 y≥183.5 → x≤95 够不着）")
    ap.add_argument("--drop", default="paper.c",
                    help="放料点（仅 --pick 用，默认 paper.c=纸面正中心）")
    ap.add_argument("--model", default="h", choices=sorted(MODELS),
                    help="用哪套映射（默认 h = 单应，精确穿过四个教点；a/b = 两套仿射解释）")
    ap.add_argument("--vs", default=None, choices=sorted(MODELS),
                    help="对比时另一个映射（默认: h↔a，a/b 互比）")
    ap.add_argument("--go", action="store_true", help="连机械臂并运动（默认只算不动）")
    ap.add_argument("--pick", action="store_true", help="真抓一次（含下降/吸/抬起/放）")
    ap.add_argument("--probe", action="store_true", help="只做触底，量出纸面 Z")
    ap.add_argument("--table-z", type=float, default=None,
                    help="纸面高度 Z（不给就既不动也不抓，先去 --probe）")
    ap.add_argument("--hover", type=float, default=30.0, help="悬停高度 mm（默认 30）")
    ap.add_argument("--obj-h", type=float, default=26.0,
                    help="吸盘接触面离**已记纸面 Z** 多高 mm。默认 26 = 实测标定值"
                         "（触底记的 -78.52 + 26 = -52.52）。"
                         "★ 不是方块净高: 方块净高 25，多的 1mm 是吸盘唇口密封的压下量。"
                         "28 会抬空 1mm 吸不到；这个数**越大停得越高**，标定时从大往小试。"
                         "等价写法: --table-z -76.5 --obj-h 25 --press 1（-76.5 = 真实纸面）")
    ap.add_argument("--press", type=float, default=0.0,
                    help="抓取时压过物体顶面的量 mm（默认 0 = 刚好贴住顶面；"
                         "正数=往下压，别给大，会压坏方块/丢步）")
    ap.add_argument("--pump-s", type=float, default=0.6, help="开泵后等多久再抬（秒）")
    ap.add_argument("--probe-z", type=float, default=None,
                    help="触底期间的 Z 探测下限（纸面很低时才需要）")
    ap.add_argument("--port", default=None, help="串口，默认自动查找")
    ap.add_argument("--mode", default="movl", choices=("movl", "movj"),
                    help="运动方式: movl 直线(默认) / movj 关节")
    ap.add_argument("--speed", type=float, default=40.0, help="速度 mm/s（默认 40）")
    ap.add_argument("--acc", type=float, default=40.0, help="加速度（默认 40）")
    ap.add_argument("--ratio", type=float, default=30.0, help="PTP 速度比例 %%（默认 30）")
    ap.add_argument("--limits", default=None,
                    help='覆盖软限位，如 "x:50:380,y:-310:310"（格式 轴:下限:上限）')
    ap.add_argument("--clear-alarms", action="store_true",
                    help="清除报警（丢步报警拒绝清除）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    layout = load_paper_layout(PAPER_JSON)
    robot_xy, src = load_robot_xy()
    missing = [c for c in layout.codes if c not in robot_xy]
    if missing:
        raise SystemExit(f"缺少 {missing} 的测量值")
    models = fit_models(layout, robot_xy)
    # --at 不给就自动挑验点: 两套映射差最大、又不是教点、还够得着
    if args.at is None:
        k1, k2 = compare_pair(models, args.model) if args.vs is None else (args.model, args.vs)
        cands = verify_candidates(layout, models, default_limits(), pair=(k1, k2))
        args.at = cands[0][0] if cands else "P3.tl"
        print(f"[自动选点] --at 没给 → 用「{args.at}」（{k1.upper()} 与 {k2.upper()} "
              f"差最大的非教点）。想换点就显式写 --at。\n")
    # 先把两个纸面点解析一遍: 写错了当场报错，别等到机械臂连上才炸
    parse_at(layout, args.at)
    if args.pick:
        parse_at(layout, args.drop)

    if args.probe or args.go or args.pick:
        if args.probe_z is None:
            import step2_teach_coords as tc
            args.probe_z = tc.PROBE_Z_FLOOR
        return run_arm(args, layout, models, robot_xy)

    report(layout, models, robot_xy, src, args.at)
    k1, k2 = compare_pair(models, args.model)
    cands = verify_candidates(layout, models, default_limits(), pair=(k1, k2))
    pt = cands[0][0] if cands else "P3.tl"
    alt = cands[1][0] if len(cands) > 1 else None
    name = Path(__file__).name
    print("\n下一步（抓取小实验，全程不用重新量点）:")
    print(f"  1) 把方块放到纸面「{pt}」上（这个点看得见、且 {k1.upper()} 与 "
          f"{k2.upper()} 差得最多）")
    print(f"  2) python3 {name} --probe                 # 触底，量出纸面 Z")
    print(f"  3) python3 {name} --at {pt} --go"
          f"      # 只悬停: 依次停 {k1.upper()}、{k2.upper()}，"
          f"**同一个 Z**、只有 XY 变")
    print("       ★ 这一步**不动 Z、不吸、不下降** —— 唯一变量是 XY，"
          "不这样分不出哪套准")
    if alt:
        print(f"  3b) 方块挪到「{alt}」再跑一遍（两次都指向同一套才定案）:"
              f"  --at {alt} --go")
    print(f"  4) python3 {name} --at {pt} --go --model {k1} --pick")
    print("                                        # **这一步才下降**（吸→抬→移→放）")
    # ★ 这里原来写死 "-52.5"，是某一次 --probe 的老读数 —— 换台机器/重测纸面 Z
    #   之后它就成了错的。必须拿当前记下的 table_z 现算。
    _tz = args.table_z if args.table_z is not None else load_table_z()
    if _tz is not None:
        print(f"     ★ --obj-h {args.obj_h:.0f} = 吸盘接触面到纸面的距离，"
              f"下降停在「纸面 Z + {args.obj_h:.0f}」")
        print(f"       当前纸面 Z={_tz:.1f} → 下降停 Z="
              f"{cube_top_z(_tz, args.obj_h):.1f}（方块顶面，不是纸面）")
    else:
        print(f"     ★ --obj-h {args.obj_h:.0f} = 吸盘接触面到纸面的距离，"
              f"下降停在「纸面 Z + {args.obj_h:.0f}」（方块顶面，不是纸面）")
        print("       纸面 Z 还没量过 → 先跑 --probe")
    print("       已经跑通的一套: --at P3.tl --model h --drop paper.c")
    print("\n注: 悬停对比默认比的是 H（单应）与 A（仿射）。四个教点上 H 精确、A 偏一个残差"
          "\n    （构造使然），所以**别拿教点验**——要验的是教点之间（上表推荐的点）。"
          "\n    想看 a/b 标签之争加 --vs b。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断]")
        sys.exit(130)
