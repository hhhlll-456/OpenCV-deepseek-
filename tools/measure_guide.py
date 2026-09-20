#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_guide.py —— 画一张「你自己的标定纸要量哪些数、怎么量」的示意图

为什么要有这张图:
  量数据这件事，用文字描述几何是最容易搞砸的 —— "量 A 到 B 的距离" 里
  A、B 在哪、边界指哪条边、白边算不算，全靠脑补。所以改成画出来。

图里画的是一张**示意**的纸（四个码贴在四角），比例不用和你的纸一致 ——
你的纸只需要"四个角各一个码"这个结构和它一样就够。

★ 白边外框的角到真正的角，差的是 **4 个模块**（不是固定的 5.42mm）:
  4 个模块 = 4 x 边长/21。边长 39mm 的纸是 5.4mm，你那种 26mm 的码只有 5.0mm。
  这个偏移**标定残差查不出来**，所以图上专门标了个红叉提醒别瞄错。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))     # 公共库在 src/
from paths import MEASURE_GUIDE_PNG as OUT, ensure_output_dir   # noqa: E402

W, H, S = 1600, 980, 2          # 逻辑画布 / 超采样倍数（画 2 倍再缩，边缘才不毛）

ORANGE = (225, 120, 20)         # 上排 / 下排
BLUE = (40, 95, 240)            # 左列 / 右列
GREEN = (0, 145, 60)            # 两条对角线
PURPLE = (150, 55, 200)         # 可选的边界间隙
RED = (215, 0, 0)
GRAY = (155, 155, 155)
DARK = (25, 25, 25)

# ── 示意图里那张"纸"的几何（纯示意，不用和你的纸一致）──
PAPER = (120.0, 200.0, 820.0, 692.0)          # 纸的矩形 (x0,y0,x1,y1)
MX, MY, CODE = 64.0, 50.0, 61.0               # 左右边距 / 上下边距 / 一个码的边长

F_LEG = 18                                     # 底部说明字号


def load_font(px: int):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, px)
            except Exception:
                pass
    return ImageFont.load_default()


def main() -> None:
    img = Image.new("RGB", (W * S, H * S), (255, 255, 255))
    d = ImageDraw.Draw(img)

    def sc(p):
        return (int(round(p[0] * S)), int(round(p[1] * S)))

    def line(p, q, color, width=3):
        d.line([sc(p), sc(q)], fill=color, width=int(width * S))

    def rect(box, outline, width=3):
        d.rectangle([sc(box[:2]), sc(box[2:])], outline=outline,
                    width=int(width * S))

    def dot(p, r, color):
        d.ellipse([sc((p[0] - r, p[1] - r)), sc((p[0] + r, p[1] + r))], fill=color)

    def text(p, s, f, fill, anchor="mm"):
        d.text(sc(p), s, font=f, fill=fill, anchor=anchor,
               stroke_width=3 * S, stroke_fill=(255, 255, 255))

    def dashed(p, q, color, width=2.5, dash=11, gap=8):
        (x0, y0), (x1, y1) = p, q
        L = float(np.hypot(x1 - x0, y1 - y0))
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        t = 0.0
        while t < L:
            e = min(t + dash, L)
            line((x0 + ux * t, y0 + uy * t), (x0 + ux * e, y0 + uy * e),
                 color, width)
            t = e + gap

    def cross(p, r, color, width=4):
        line((p[0] - r, p[1] - r), (p[0] + r, p[1] + r), color, width)
        line((p[0] - r, p[1] + r), (p[0] + r, p[1] - r), color, width)

    f_title = load_font(35 * S)
    f_sub = load_font(23 * S)
    f_lab = load_font(20 * S)
    f_tag = load_font(24 * S)
    f_small = load_font(F_LEG * S)

    # ─────────────────── 标题 ───────────────────
    text((40, 42), "自己贴的标定纸：要量哪些数、怎么量", f_title, DARK, "lm")
    text((40, 88), "红点 = 参考点 = 每个二维码【黑色图案】的右上角。"
                   "量的就是红点与红点之间的距离，一共 6 条。",
         f_sub, (60, 60, 60), "lm")

    # ─────────────────── 纸的轮廓 ───────────────────
    x0, y0, x1, y1 = PAPER
    rect(PAPER, GRAY, 3)
    text(((x0 + x1) / 2, y0 - 24), "纸张上边界", f_lab, GRAY)
    text(((x0 + x1) / 2, y1 + 24), "纸张下边界", f_lab, GRAY)
    text((62, (y0 + y1) / 2), "纸张左边界", f_lab, GRAY)
    text((x1 + 12, (y0 + y1) / 2), "纸张右边界", f_lab, GRAY, "lm")

    # ─────────────────── 四个码的位置 ───────────────────
    boxes = {"A": (x0 + MX, y0 + MY),
             "B": (x1 - MX - CODE, y0 + MY),
             "C": (x0 + MX, y1 - MY - CODE),
             "D": (x1 - MX - CODE, y1 - MY - CODE)}
    ref = {k: (bx + CODE, by) for k, (bx, by) in boxes.items()}   # 黑框右上角
    cy = (y0 + y1) / 2

    # ── 6 条测量线 + 各自的标签（位置是手调的，避开线本身）──
    for a, b, col in (("A", "B", ORANGE), ("C", "D", ORANGE),
                      ("A", "C", BLUE), ("B", "D", BLUE),
                      ("A", "D", GREEN), ("B", "C", GREEN)):
        line(ref[a], ref[b], col, 3)
    # ★ 标签位置是手调的: 两条对角线在纸中央交叉，标签必须放在"离自己那条线
    #   够远、又不在另一条线上"的地方，否则会互相压住（第一版就把 B↔C 和 C↔D
    #   叠在一起了）。
    for a, b, col, lx, ly in (("A", "B", ORANGE, 400, y0 + MY - 16),
                              ("C", "D", ORANGE, 500, y1 - MY - CODE - 20),
                              ("A", "C", BLUE, x0 + MX + 18, cy - 52),
                              ("B", "D", BLUE, x1 - MX - 56, cy - 52),
                              ("A", "D", GREEN, 600, cy - 6),
                              ("B", "C", GREEN, 400, cy - 26)):
        text((lx, ly), f"{a}↔{b}", f_tag, col)

    # ── 可选的边界间隙（紫色 ①②）──
    line((x0, y0 + MY + CODE / 2), (x0 + MX, y0 + MY + CODE / 2), PURPLE, 3)
    dot((x0, y0 + MY + CODE / 2), 4, PURPLE)
    text((x0 + MX / 2, y0 + MY + CODE / 2 - 20), "①", f_tag, PURPLE)
    line((x0 + MX + CODE / 2, y0), (x0 + MX + CODE / 2, y0 + MY), PURPLE, 3)
    dot((x0 + MX + CODE / 2, y0), 4, PURPLE)
    text((x0 + MX + CODE / 2 + 26, y0 + MY / 2), "②", f_tag, PURPLE)

    # ─────────────────── 画四个码 ───────────────────
    # 画成"黑图案 + 三个查找图形"的样子，这样"黑框右上角"才有实感
    n = 21
    base = np.random.default_rng(7).random((n, n)) < 0.47
    for (r0, c0) in ((0, 0), (0, n - 7), (n - 7, 0)):        # 三个查找图形
        for i in range(7):
            for j in range(7):
                ring = i in (0, 6) or j in (0, 6)
                core = 2 <= i <= 4 and 2 <= j <= 4
                base[r0 + i, c0 + j] = ring or core

    def paint(bx, by, side):
        cell = side / n
        for i in range(n):
            for j in range(n):
                if base[i, j]:
                    d.rectangle([sc((bx + j * cell, by + i * cell)),
                                 sc((bx + (j + 1) * cell, by + (i + 1) * cell))],
                                fill=DARK)

    for k, (bx, by) in boxes.items():
        paint(bx, by, CODE)
        rect((bx, by, bx + CODE, by + CODE), (90, 90, 90), 2)
        text((bx + CODE / 2, by + CODE / 2), k, f_title, RED)

    for p in ref.values():                       # 红点最后画，压在最上层
        dot(p, 9, (255, 255, 255))
        dot(p, 7, RED)

    # ─────────────────── 右上角放大图 ───────────────────
    text((950, 194), "放大看：参考点到底在哪一点", f_tag, DARK, "lm")

    zc, zx0, zy0 = 400.0, 1040.0, 340.0
    q = zc / n * 4                                           # 白边宽 = 4 模块
    ox0, oy0 = zx0 - q, zy0 - q
    ox1, oy1 = zx0 + zc + q, zy0 + zc + q
    for p, r in (((ox0, oy0), (ox1, oy0)), ((ox1, oy0), (ox1, oy1)),
                 ((ox0, oy1), (ox1, oy1)), ((ox0, oy0), (ox0, oy1))):
        dashed(p, r, GRAY, 2.5)

    paint(zx0, zy0, zc)
    corner = (zx0 + zc, zy0)
    dot(corner, 15, (255, 255, 255))
    dot(corner, 10, RED)
    line(corner, (corner[0] - 90, corner[1] - 38), RED, 3)
    text((corner[0] - 96, corner[1] - 44), "参考点", f_tag, RED, "rm")
    text((corner[0] - 96, corner[1] - 18), "黑框右上角", f_lab, RED, "rm")

    for p in ((zx0, zy0), (zx0, zy0 + zc), (zx0 + zc, zy0 + zc)):
        dot(p, 5, GRAY)

    # 白边外框的角 —— 差 4 个模块，标红叉
    oc = (ox1, oy0)
    cross(oc, 14, RED, 4)
    text((ox1 - 12, oy0 - 40), "白边外框的角", f_small, RED, "rm")
    text((ox1 - 12, oy0 - 18), "别瞄这里（差 4 个模块）", f_small, RED, "rm")

    text((zx0 + zc / 2, oy1 + 24), "白色虚线 = 白边（纸上纯白，肉眼量不到）",
         f_small, GRAY)
    text((zx0 + zc / 2, oy1 + 48), "灰点 = 黑框另外三个角（本项目不瞄它们）",
         f_small, GRAY)

    # ─────────────────── 底部说明 ───────────────────
    rows = [
        (RED, "红色 ✗ = 白边外框的角：纸上是一片纯白、看不出位置，"
              "与真正的角差 4 个模块 ≈ 5mm —— 绝对别瞄它。"),
        (PURPLE, "① ② 紫色（可选）= A 的黑框左边→纸张左边、上边→纸张上边。"
                 "只影响摄像头自检的「纸有没有出画」提示。"),
        (DARK, "橙色 = 上排 / 下排。    蓝色 = 左列 / 右列。    绿色 = 两条对角线。"),
        (DARK, "六条都要量：五条定形状，第六条是交叉校验 —— "
               "数字自相矛盾时，程序会把嫌疑最大的那条指出来。"),
        (DARK, "量法：先在四个红点点小铅笔点 → 尺子贴平纸面 → 量小点与小点之间。"
               "用钢尺别用软尺，每条量两次。"),
        (DARK, "位置代号 A/B/C/D 按纸摆好的方位定（如上图），不是按二维码的内容。"),
    ]
    for i, (col, s) in enumerate(rows):
        text((40, 800 + i * 28), s, f_small, col, "lm")

    ensure_output_dir()
    img.resize((W, H), Image.LANCZOS).save(OUT)
    print(f"✅ 已输出 {OUT}")


if __name__ == "__main__":
    main()
