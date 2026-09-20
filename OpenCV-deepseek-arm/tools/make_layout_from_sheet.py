#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_layout_from_sheet.py —— 用卷尺量出来的实测几何，重建 calib_A4_qr.json
=============================================================================
什么时候需要它:
  src/step1_gen_paper.py 是「一次打印整张纸」的做法 —— 四个码的位置由脚本算出来，
  png 和 json 天生配套、误差 <0.1mm。但《方案.md》第 15-16 行建议的是另一种做法:
  用二维码生成器生成 4 个码，自己贴/印到一张纸的四个角上。那样手里那张纸的真实
  几何和 src/step1_gen_paper.py 那份 json 是对不上的。

对不上会怎样（重要，别搞反）:
  · 手眼标定**不用**纸上的毫米数 —— 拟合只用「检测到的像素」和「你对刀测的机械臂 XY」，
    纸面 mm 完全不进 compute_homography。所以错配**不会**让标定悄悄算歪。
  · 但一致性守卫 check_coord_consistency（src/step3_hand_eye_calib.py）和
    src/step2_teach_coords.py 的
    validate_points 会拿「纸面两两距离」和「机械臂两两距离」比，容差 3mm，超了就抛异常。
    所以错配的表现是**响亮的拒绝**，不是静默的错误标定。

★ 为什么必须拿卷尺量、不能拿机械臂坐标反推:
    守卫比的就是「纸面距离 vs 机械臂距离」。如果纸面模型是从机械臂坐标推出来的，
    两边必然相等 —— 守卫变成自证，永远绿灯，等于没有守卫。纸面几何必须来自
    一个**独立**的测量源，也就是你的尺子。

★ 参考点是什么（量之前先读这三行，量错这个是最常见的坑）:
    「参考点」= 二维码**黑色图案**的右上角 = 数据区右上角 = 检测器 quad[1]。
    你手里的纸上只有黑色图案是有形的，白边本来就是白的、量不到 —— 所以
    黑框的右上角就是它，不会和「含白边的外框角」混（那两者差 5.42mm）。
    黑色图案的外框 = 数据区: 因为左上/右上/左下三个查找图形正好贴住数据区的
    三条边，所以黑色像素的包围盒就是 21x21 的数据区。

用法（先量，再填；括号里是量法）:
    python3 tools/make_layout_from_sheet.py --paper 297 210 --side 26 \\
        --ab 217.0 --cd 217.2 --ac 142.0 --bd 141.8 --ad 259.5 --bc 259.3 \\
        --left 27.0 --top 21.0 --content P1 P2 P3 P4

      位置代号（按纸怎么摆，不是按码内容）:
          A = 左上    B = 右上    C = 左下    D = 右下
      --ab/--cd  上排 / 下排 两个参考点的直线距离
      --ac/--bd  左列 / 右列 两个参考点的直线距离
      --ad/--bc  两条对角线
      --left/--top  A 的黑框左边界到纸张左边界、黑框上边界到纸张上边界（可省，
                   只影响 tools/test_camera_qr.py「纸有没有出画」那个提示）
      --content  四个位置上的码**内容**，顺序 A B C D（手机扫一下就知道）

自检:
    python3 tools/make_layout_from_sheet.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))     # 公共库在 src/

from paths import PAPER_JSON as OUT_JSON, ensure_data_dir   # noqa: E402

# 位置代号 ↔ 中文（只为了打印好看）
POS = ("A", "B", "C", "D")
POS_CN = {"A": "左上", "B": "右上", "C": "左下", "D": "右下"}

QUIET_MODULES = 4          # 规范白边宽度（模块）。本工具假定它是标准的 4
DATA_MODULES = 21          # 版本 1 二维码的数据区边长（模块）—— 由 --side 量到的就是这个
TOTAL_MODULES = DATA_MODULES + 2 * QUIET_MODULES     # 29 = 含白边的模块数

RESID_WARN_MM = 1.5        # 重建残差超过这个值 → 大概是哪一条量错了


# ─────────────────────────── 三边定位 ───────────────────────────
def trilaterate(a: tuple[float, float], b: tuple[float, float],
                d_ap: float, d_bp: float) -> tuple[tuple[float, float],
                                                   tuple[float, float]]:
    """
    已知 A、B 两点坐标，以及 P 到 A、P 到 B 的距离，求 P —— 返回两个镜像解。

    ★ 两个解是"纸正面看"和"从纸背面看"，物理上只有一个。本脚本不猜:
      靠"哪个位置在哪个方位"（A 左上 / C 左下 / D 右下）把镜像解定下来，
      见 build_quad()。这也是为什么必须告诉脚本码内容对应哪个位置。
    """
    ax, ay = a
    bx, by = b
    dab = math.hypot(bx - ax, by - ay)
    if dab <= 0:
        raise ValueError("A 和 B 重合了，无法定位")
    ex, ey = (bx - ax) / dab, (by - ay) / dab
    # A→B 方向上的投影长度，以及垂直方向的高度
    x = (d_ap ** 2 - d_bp ** 2 + dab ** 2) / (2 * dab)
    h = math.sqrt(max(0.0, d_ap ** 2 - x ** 2))
    px, py = ax + x * ex, ay + x * ey
    return (px - h * ey, py + h * ex), (px + h * ey, py - h * ex)


def _need_triangle(a: str, b: str, d_ab: float, d_ap: float, d_bp: float) -> None:
    """三点要能构成三角形，否则是量错了。提前报出来比算出个负数再崩好。"""
    if min(d_ab, d_ap, d_bp) <= 0:
        raise ValueError(f"{a}/{b} 相关的距离出现非正数: {d_ab} {d_ap} {d_bp}")
    if d_ap + d_bp < d_ab - 1e-6 or d_ap + d_ab < d_bp - 1e-6 or d_bp + d_ab < d_ap - 1e-6:
        raise ValueError(
            f"{a}、{b} 相关的三条距离构不成三角形（{d_ab} / {d_ap} / {d_bp}）——"
            f" 至少有一条量错了")


def build_quad(d: dict) -> dict[str, tuple[float, float]]:
    """
    由 6 条实测距离求出 A/B/C/D 四个参考点的相对坐标。

    做法: 先钉死 A=(0,0)、B=(ab,0)，再用三边定位放下 C、D。
    解的选择靠"方位常识"而不是猜: C 在 A→B 的 +y 侧（纸的下方）、
    D 取 x 更大的那个解（纸的右侧）。
    """
    dab = d["ab"]
    _need_triangle("A", "B", dab, d["ac"], d["bc"])
    _need_triangle("A", "B", dab, d["ad"], d["bd"])

    A = (0.0, 0.0)
    B = (dab, 0.0)

    c1, c2 = trilaterate(A, B, d["ac"], d["bc"])
    C = max((c1, c2), key=lambda p: p[1])      # C 在下方 → y 更大
    _need_triangle("B", "C", math.dist(B, C), d["bd"], d["cd"])
    d1, d2 = trilaterate(B, C, d["bd"], d["cd"])
    D = max((d1, d2), key=lambda p: p[0])      # D 在右侧 → x 更大

    return {"A": A, "B": B, "C": C, "D": D}


def residuals(quad: dict, d: dict) -> list[tuple[str, float, float, float]]:
    """把重建出来的四边形的两两距离和实测值逐条对一遍。"""
    out = []
    for k, (i, j) in (("ab", ("A", "B")), ("cd", ("C", "D")),
                      ("ac", ("A", "C")), ("bd", ("B", "D")),
                      ("ad", ("A", "D")), ("bc", ("B", "C"))):
        got = math.dist(quad[i], quad[j])
        out.append((k, d[k], got, got - d[k]))
    return out


# ─────────────────────────── 生成 json ───────────────────────────
def build(quad: dict, d: dict, paper: tuple[float, float], side: float,
          contents: dict[str, str], left: float | None, top: float | None) -> dict:
    """把四个参考点摆到纸面坐标系里，装配成 calib_A4_qr.json 的结构。"""
    # 纸面坐标系: 原点=纸张左上角，x 向右、y 向下（和 src/step1_gen_paper.py 一致）
    xs = [p[0] for p in quad.values()]
    ys = [p[1] for p in quad.values()]
    if left is None or top is None:
        # 没给绝对位置 → 把四个参考点整体摆到纸的正中（"贴得还算居中"是最合理的假设）。
        # 只影响 tools/test_camera_qr.py 的「纸有没有出画」提示，不影响标定与守卫。
        # ★ 不能就地把 A 当原点: 那样整个纸框会被平移 ~(-55,-27)mm，这个提示反而会误导。
        ox = (paper[0] - (max(xs) + min(xs))) / 2
        oy = (paper[1] - (max(ys) + min(ys))) / 2
    else:
        # A 的参考点 = 黑框右上角 → 比黑框左边界多一个边长
        ox = left + side - quad["A"][0]
        oy = top - quad["A"][1]
    placed = {k: (p[0] + ox, p[1] + oy) for k, p in quad.items()}

    outer = side * TOTAL_MODULES / DATA_MODULES      # 含白边的名义边长
    rep = {
        "meta": {
            "source": "measured-sheet",
            "paper_mm": list(paper),
            "qr_size_mm_nominal": round(side, 2),
            "qr_size_mm_actual": round(outer, 3),
            "quiet_zone_modules": QUIET_MODULES,
            "reference": "corner_tr",
            "note": ("由 tools/make_layout_from_sheet.py 从实测距离重建。"
                     "边长 --side 量的是黑色图案(数据区, 21 模块)；"
                     "含白边的名义边长按 29/21 外推。"),
        },
        "qr": {},
    }
    for p in POS:
        name = contents[p]
        rx, ry = placed[p]
        # ★ 参考点 = 黑框右上角。其余三角按"码大致没转"外推 —— 默认只用 corner_tr，
        #   所以这点外推误差不影响标定；只有换成 --reference center/其它角才会碰到它。
        corners = {
            "top_left":     (rx - side, ry),
            "top_right":    (rx, ry),
            "bottom_right": (rx, ry + side),
            "bottom_left":  (rx - side, ry + side),
        }
        rep["qr"][name] = {
            "where": POS_CN[p],
            "symbol_corners_mm": {k: [round(v[0], 2), round(v[1], 2)]
                                  for k, v in corners.items()},
            "center_mm": [round(rx - side / 2, 2), round(ry + side / 2, 2)],
            "ref_point_mm": [round(rx, 2), round(ry, 2)],
            "module_mm": round(side / DATA_MODULES, 4),
            "modules": TOTAL_MODULES,
        }
    return rep


# ─────────────────────────── 自检 ───────────────────────────
def selftest() -> int:
    """
    拿已知几何正推出一组'实测距离'，再喂回 build_quad，看能不能还原。
    ★ 必须用**非矩形**的例子: 矩形太对称，镜像解选错也看不出来。
    """
    print("═" * 70)
    print("  自检: 已知四边形 → 反推距离 → 再重建 → 比坐标")
    print("═" * 70)
    truth = {"A": (0.0, 0.0), "B": (217.0, 1.5),
             "C": (2.3, 142.0), "D": (220.4, 139.2)}   # 故意歪一点、非矩形

    def ds(i, j):
        # ★ 不取整: 这一步要验的是三边定位的数学。若把距离先 round(…,3)，
        #   输入的 1µm 量化误差会原样出现在残差里（实测 4e-4mm），
        #   于是"残差必须 ~0"这条断言就变成了在测我的取整、而不是测算术。
        return math.dist(truth[i], truth[j])

    d = {"ab": ds("A", "B"), "cd": ds("C", "D"), "ac": ds("A", "C"),
         "bd": ds("B", "D"), "ad": ds("A", "D"), "bc": ds("B", "C")}
    ok = True

    quad = build_quad(d)
    # 重建结果和真值之间差一个刚体变换（旋转+平移），所以比"两两距离"而不是比坐标
    worst = 0.0
    for k, (i, j) in (("ab", ("A", "B")), ("cd", ("C", "D")), ("ac", ("A", "C")),
                      ("bd", ("B", "D")), ("ad", ("A", "D")), ("bc", ("B", "C"))):
        dev = abs(math.dist(quad[i], quad[j]) - d[k])
        worst = max(worst, dev)
        print(f"  {i}-{j}: 实测 {d[k]:8.3f} → 重建 {math.dist(quad[i], quad[j]):8.3f}"
              f"  偏差 {dev:.3e} mm")
    good = worst < 1e-9
    ok &= good
    print(f"  {'✅' if good else '❌'} 六条距离全部精确还原（最大偏差 {worst:.2e} mm）")

    # 方位: A 左上 / B 右上 / C 左下 / D 右下 —— 镜像解选错的话这几条会翻
    checks = [
        ("A 在 B 左边", quad["A"][0] < quad["B"][0]),
        ("C 在 D 左边", quad["C"][0] < quad["D"][0]),
        ("A 在 C 上边", quad["A"][1] < quad["C"][1]),
        ("B 在 D 上边", quad["B"][1] < quad["D"][1]),
    ]
    for name, cond in checks:
        ok &= bool(cond)
        print(f"  {'✅' if cond else '❌'} {name}（镜像解选对了）")

    # 构不成三角形要报错，不能算出一个负数再崩
    try:
        _need_triangle("A", "B", 10.0, 1.0, 1.0)
        print("  ❌ 畸形三角形没被拦下来")
        ok = False
    except ValueError:
        print("  ✅ 畸形三角形被拦下并给出可读错误")

    # 整条链路: 装配 json → 用正式加载器读回来 → 距离自洽
    rep = build(quad, d, (297.0, 210.0), 26.0,
                {"A": "P1", "B": "P2", "C": "P3", "D": "P4"}, 27.0, 21.0)
    tmp = Path(tempfile.mkdtemp()) / "_selftest_layout.json"
    tmp.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    try:
        from qr_vision import load_paper_layout          # noqa: E402
        lay = load_paper_layout(tmp, reference="corner_tr")
        ref = lay.reference_mm
        w = max(abs(math.dist(ref["P1"], ref["P2"]) - d["ab"]),
                abs(math.dist(ref["P1"], ref["P3"]) - d["ac"]),
                abs(math.dist(ref["P1"], ref["P4"]) - d["ad"]))
        good = w < 0.05
        ok &= good
        print(f"  {'✅' if good else '❌'} 落盘的 json 被 load_paper_layout 读回后，"
              f"纸面距离仍自洽（最大偏差 {w:.3f} mm）")
        # 参考点必须是 corner_tr，且和 black 右上角一致
        good = all(abs(lay.mm_for("corner_tr")[c][0] - ref[c][0]) < 1e-6
                   for c in lay.codes)
        ok &= good
        print(f"  {'✅' if good else '❌'} reference=corner_tr 取到的就是黑框右上角")
    finally:
        tmp.unlink(missing_ok=True)

    print("═" * 70)
    print("  ✅ 自检通过" if ok else "  ❌ 自检失败")
    print("═" * 70)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="用卷尺实测的几何重建 calib_A4_qr.json（自己贴的标定纸用这个）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", nargs=2, type=float, metavar=("W", "H"),
                    help="纸张宽 高 mm，例如 --paper 297 210")
    ap.add_argument("--side", type=float,
                    help="二维码**黑色图案**的边长 mm（= 数据区 = 21 模块）")
    ap.add_argument("--ab", type=float, help="上排 两个参考点距离 mm")
    ap.add_argument("--cd", type=float, help="下排 两个参考点距离 mm")
    ap.add_argument("--ac", type=float, help="左列 两个参考点距离 mm")
    ap.add_argument("--bd", type=float, help="右列 两个参考点距离 mm")
    ap.add_argument("--ad", type=float, help="对角线 A-D 距离 mm")
    ap.add_argument("--bc", type=float, help="对角线 B-C 距离 mm")
    ap.add_argument("--left", type=float, default=None,
                    help="A 的黑框左边界 → 纸张左边界 mm（可省）")
    ap.add_argument("--top", type=float, default=None,
                    help="A 的黑框上边界 → 纸张上边界 mm（可省）")
    ap.add_argument("--content", nargs=4, metavar=("A", "B", "C", "D"),
                    default=["P1", "P2", "P3", "P4"],
                    help="四个位置上的码内容，顺序 左上 右上 左下 右下（默认 P1 P2 P3 P4）")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--selftest", action="store_true", help="跑自检")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    need = ["paper", "side", "ab", "cd", "ac", "bd", "ad", "bc"]
    missing = [n for n in need if getattr(args, n) is None]
    if missing:
        ap.error("缺参数: " + ", ".join("--" + m for m in missing)
                 + "\n  六个距离 --ab --cd --ac --bd --ad --bc 都要量（少一个就定不出"
                   "四个点的相对位置，也失去交叉校验）。")

    d = {k: float(getattr(args, k)) for k in ("ab", "cd", "ac", "bd", "ad", "bc")}
    quad = build_quad(d)

    print("═" * 70)
    print("  标定纸几何重建（数据来源: 卷尺实测）")
    print("═" * 70)
    for k, meas, got, dev in residuals(quad, d):
        flag = "  ⚠" if abs(dev) > RESID_WARN_MM else "   "
        print(f"  {k}: 实测 {meas:8.2f}  重建 {got:8.2f}  "
              f"残差 {dev:+6.2f} mm{flag}")
    worst = max(abs(r[3]) for r in residuals(quad, d))
    if worst > RESID_WARN_MM:
        print(f"\n  ⚠️ 最大残差 {worst:.2f}mm 超过 {RESID_WARN_MM}mm。六个距离里至少有"
              f"一条量错了 ——")
        print("     （残差最大的那条就是嫌疑最大的，重量一次再跑。）")
    else:
        print(f"\n  ✅ 六条距离自洽（最大残差 {worst:.2f}mm），可以落盘")

    rep = build(quad, d, tuple(args.paper), args.side,
                dict(zip(POS, args.content)), args.left, args.top)
    ensure_data_dir()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    xs = [q["ref_point_mm"][0] for q in rep["qr"].values()]
    ys = [q["ref_point_mm"][1] for q in rep["qr"].values()]
    print("\n  参考点 = 黑色图案的右上角（corner_tr）")
    for name, q in rep["qr"].items():
        print(f"    {name} ({q['where']}): 纸上坐标 "
              f"X={q['ref_point_mm'][0]:7.2f}  Y={q['ref_point_mm'][1]:7.2f} mm")
    print(f"  工作区跨距: {max(xs) - min(xs):.2f} x {max(ys) - min(ys):.2f} mm")
    if args.left is None or args.top is None:
        print("  ⚠️ 没给 --left/--top: 纸的外框位置是粗略的 —— 只影响"
              " tools/test_camera_qr.py 的「纸有没有出画」提示，不影响标定和守卫。")
    print(f"\n✅ 已写入 {args.out}")
    print("   下一步: python3 src/step2_teach_coords.py --table-z <你量到的桌面Z>"
          "   （和 src/step3_hand_eye_calib.py 都必须用 --reference corner_tr）")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
