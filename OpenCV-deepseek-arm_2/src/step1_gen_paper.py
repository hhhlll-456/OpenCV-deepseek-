#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_gen_paper.py —— 生成 A4 手眼标定纸（四角二维码 P1~P4）+ 输出角点坐标

配合《方案.md》使用：
  · 第二阶段：把本脚本打印出来的纸贴在硬纸板上，用吸盘中心依次对准
              P1~P4 的某个「数据区角点」，记录 DobotStudio 的 XY。
              ★ 瞄哪个角由 step2_teach_coords.py / step3_hand_eye_calib.py 的 --reference 决定
                （默认 corner_tr 右上角）。四个角数学上完全等价，只是瞄的位置不同；
                但对刀和标定**必须用同一个**，否则整体偏一个常量且查不出来。
                瞄之前请看本脚本输出的 ref_point_guide.png，上面五个候选点全标了。
  · 第三阶段：程序读一帧，detectAndDecodeMulti 拿到 4 个二维码的像素角点，
              与写死的物理坐标做 getPerspectiveTransform。

⚠ 打印注意：务必用「100% / 实际大小」打印，**不要**勾选"适应页面 / 缩放"，
   否则物理尺寸就不准了（用尺子量一下某个二维码的边长是不是等于 QR_SIZE_MM）。

⚠ 尺寸说明（实测结论，很重要）：
   《方案.md》建议 2cm，但实测发现——当摄像头要「一眼看全整张 A4」时，2cm 的码
   在 1080p 下只有 ~70px（≈2.5 px/模块），只能定位、无法稳定解码；4K 下也只有
   3/4 能解出来。所以本脚本默认改用 40mm（4cm）。经验阈值：
       解码需要 ≥ 4~5 px/模块  →  QR边长mm × (视野像素/纸宽mm) ÷ 29模块 ≥ 4
   实测可用的组合：
       1080p 全幅  → 二维码 ≥ 40mm   （20mm: 0/4,  40mm: 3/4,  50mm: 4/4）
       4K   全幅  → 二维码 ≥ 25mm   （20mm: 3/4,  25mm: 4/4）
   若只想「定位」不需要解出内容，20~50px 就够，2cm 也能用。

依赖：opencv-python (或系统 python3-opencv) + numpy + pillow(仅用于画标注文字)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import cv2
import numpy as np

# ═════════════════════════════ 可调参数 ═════════════════════════════
DPI         = 300        # 打印分辨率（打印店常用 300）

# 纸张方向："landscape" = 横版 (297x210)，"portrait" = 竖版 (210x297)
#   ★ 横版的意义：纸的长边横向铺开，机械臂最远处只需够到 210mm，而不是 297mm
ORIENTATION = "landscape"
PAPER_MM    = {"portrait": (210.0, 297.0),
               "landscape": (297.0, 210.0)}[ORIENTATION]

QR_SIZE_MM  = 40.0       # 单个二维码「含白边」的边长 mm
                         #   方案.md 原建议 2cm，但 1080p 全幅视野下解不出来，改成 4cm
                         #   （只要 4K 或者只需定位不解码，可以调回 20.0）
MARGIN_MM   = 15.0       # 二维码最外边缘 到 纸张边缘 的距离 mm
QUIET       = 4          # 白边宽度（模块数）。规范要求 >=4，不要为了塞大而改小
TICK_MM     = 5.0        # 四角定位标记长度 mm（画在二维码外面，绝不碰白边）
LABEL_MM    = 5.0        # 标注文字高度 mm
DRAW_TICK   = True       # 是否画四角定位标记
DRAW_LABEL  = True       # 是否标注 P1..P4 文字

# 二维码内容 + 摆放位置。想改内容只动这里；比如想直接放坐标就写成 "20.4,20.4"
CORNERS = [
    ("P1", "左上"),
    ("P2", "右上"),
    ("P3", "左下"),
    ("P4", "右下"),
]

# 输出位置统一在 src/paths.py（纸面定义 → data/，图例 → output/）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (PAPER_JSON as OUT_JSON,        # noqa: E402
                   PAPER_PNG as OUT_PNG,
                   REF_GUIDE_PNG as OUT_GUIDE,
                   ensure_data_dir, ensure_output_dir)

# 允许覆盖「实测」的 calib_A4_qr.json（--force 置 True，见 guard_measured_json）
FORCE_JSON_OVERWRITE = False

# ── 对刀参考点示意图 ──
# 这张图把四个码各放大一块，标出**全部 5 个候选参考点**。
# ★ 为什么标全部、不只标默认那一个: 早先这张图是手工画的、只标了左上角，
#   后来默认参考点换成右上角，图就变成"指着错的地方"了 —— 而且是静默的，
#   没人会想到去看一眼它是不是过期。标全部就没有"过期"这回事了。
# ★ 那张红叉是重点: 白边外框角也是一片纯白、肉眼根本看不出在哪，
#   和真正的角差 5.42mm，且这种偏差标定残差查不出来。
GUIDE_MODES = [                       # (模式名, BGR 颜色, 图上的短标签)
    ("corner_tl", (  0,   0, 255), "TL"),
    ("corner_tr", (  0, 150,   0), "TR"),
    ("corner_br", (255,  60,   0), "BR"),
    ("corner_bl", (  0, 170, 255), "BL"),
    ("center",    (200,   0, 200), "C"),
]
GUIDE_CELL = 620                      # 每个码那一格的边长(px)
GUIDE_MARGIN_MM = 5.0                 # 裁切时在码外留多少 mm（要放得下白边角的红叉）
# ════════════════════════════════════════════════════════════════════

MM = 25.4                      # 1 英寸 = 25.4 mm
PPM = DPI / MM                 # 每毫米多少像素


# ─────────────────────────── 二维码生成 ───────────────────────────
def encode_qr_modules(text: str, quiet: int = QUIET) -> np.ndarray:
    """
    把文本编码成「模块级」二维码位图。
    返回 uint8 数组：0 = 黑模块，255 = 白模块。
    注意：OpenCV 的 encode() 自带 2 模块白边，这里补齐到 quiet(默认4) 模块。
    """
    enc = cv2.QRCodeEncoder_create()

    m = None
    try:
        m = enc.encode(text)                       # 2 字符内容走版本 1
    except cv2.error:
        for ver in range(1, 41):                   # 内容太长就逐版本试
            try:
                m = enc.encode(text, ver)
                break
            except cv2.error:
                continue
    if m is None:
        raise ValueError(f"内容太长，二维码装不下: {text!r}")

    m = m.astype(np.uint8)

    builtin_quiet = 2                              # OpenCV 自带的 2 模块白边
    extra = max(0, quiet - builtin_quiet)
    if extra:
        m = np.pad(m, extra, constant_values=255)
    return m


def render_symbol(modules: np.ndarray, target_px: int) -> tuple[np.ndarray, int]:
    """
    把模块级位图放大成目标像素大小。
    模块边长取整，保证每模块像素数一致 → 打印出来边缘绝对锐利、不会被插值糊掉。
    返回 (图像, 每模块像素数)。
    """
    n = modules.shape[0]
    module_px = max(1, int(round(target_px / n)))
    px = module_px * n
    img = cv2.resize(modules, (px, px), interpolation=cv2.INTER_NEAREST)
    return img, module_px


# ─────────────────────────── 标注文字字体 ───────────────────────────
def load_font(size_px: int):
    from PIL import ImageFont
    for p in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size_px)
            except Exception:
                pass
    return ImageFont.load_default()


# ─────────────────────────── 对刀参考点示意图 ───────────────────────────
def _guide_cell(canvas_bgr: np.ndarray, rep: dict, code: str) -> np.ndarray:
    """把某个码连同它的候选参考点裁出来放大成一格。"""
    d = rep["qr"][code]
    x0, y0, x1, y1 = d["outer_rect_px"]
    m = int(round(GUIDE_MARGIN_MM * PPM))
    x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), x1 + m, y1 + m

    crop = canvas_bgr[y0:y1, x0:x1].copy()
    h, w = crop.shape[:2]
    s = GUIDE_CELL / max(h, w)
    crop = cv2.resize(crop, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_NEAREST)

    def cell(pt):
        return (int(round((pt[0] - x0) * s)), int(round((pt[1] - y0) * s)))

    # 数据区四角 + 中心（这才是候选参考点）
    sx, sy, ex, ey = d["symbol_rect_px"]
    cand = {"corner_tl": (sx, sy), "corner_tr": (ex, sy),
            "corner_br": (ex, ey), "corner_bl": (sx, ey),
            "center": ((sx + ex) // 2, (sy + ey) // 2)}
    # 标签往「远离中心」的方向甩，免得压在二维码图样上看不清
    away = {"corner_tl": (-1, -1), "corner_tr": (+1, -1),
            "corner_br": (+1, +1), "corner_bl": (-1, +1), "center": (+1, 0)}

    # 白边外框的四个角 —— 画红叉，提醒"别瞄这里"（差 5.42mm）
    ox0, oy0, ox1, oy1 = d["outer_rect_px"]
    for pt in ((ox0, oy0), (ox1, oy0), (ox1, oy1), (ox0, oy1)):
        cx, cy = cell(pt)
        for dx, dy in ((1, 1), (1, -1)):
            cv2.line(crop, (cx - 9 * dx, cy - 9 * dy), (cx + 9 * dx, cy + 9 * dy),
                     (0, 0, 255), 2, cv2.LINE_AA)

    for mode, color, tag in GUIDE_MODES:
        cx, cy = cell(cand[mode])
        arm = 30 if mode != "center" else 18
        cv2.line(crop, (cx - arm, cy), (cx + arm, cy), color, 3, cv2.LINE_AA)
        cv2.line(crop, (cx, cy - arm), (cx, cy + arm), color, 3, cv2.LINE_AA)
        cv2.circle(crop, (cx, cy), 6, color, -1, cv2.LINE_AA)
        dx, dy = away[mode]
        tx, ty = cx + dx * (arm + 6) - (26 if dx < 0 else 0), cy + dy * (arm + 20)
        cv2.putText(crop, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    color, 2, cv2.LINE_AA)
    return crop


def build_guide(rep: dict, canvas: np.ndarray) -> None:
    """拼出 2x2 的放大示意图 + 图例，写到 OUT_GUIDE。"""
    from PIL import Image, ImageDraw
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cells = [_guide_cell(bgr, rep, code) for code, _ in CORNERS]

    gap, pad, legend_h = 16, 20, 132
    cw = max(c.shape[1] for c in cells)
    ch = max(c.shape[0] for c in cells)
    W = pad * 2 + cw * 2 + gap
    H = pad * 2 + legend_h + ch * 2 + gap
    out = np.full((H, W, 3), 255, np.uint8)

    for i, cell in enumerate(cells):
        r, c = divmod(i, 2)
        y = pad + legend_h + r * (ch + gap)
        x = pad + c * (cw + gap)
        out[y:y + cell.shape[0], x:x + cell.shape[1]] = cell

    pil = Image.fromarray(out[:, :, ::-1])          # BGR → RGB
    d = ImageDraw.Draw(pil)
    font = load_font(30)
    small = load_font(22)
    d.text((pad, 16), "对刀参考点示意（图中是放大后的四个码）", fill=(0, 0, 0), font=font)

    # 图例: 彩色标签 → 模式名
    x = pad
    for mode, color, tag in GUIDE_MODES:
        b, g, r = color
        d.rectangle([x, 68, x + 30, 98], fill=(r, g, b))
        d.text((x + 40, 68), f"{tag} = {mode}", fill=(0, 0, 0), font=small)
        x += 250
    d.text((pad, 104), "红色 ✗ = 白边外框角：一片纯白、看不出位置，且与真正的角相差"
                       " 5.42mm —— 绝对不要瞄这里。", fill=(180, 0, 0), font=small)

    ensure_output_dir()
    Image.fromarray(np.array(pil)).save(OUT_GUIDE)
    print(f"   示意图: {OUT_GUIDE.name}（5 个候选参考点全标出）")


# ─────────────────────────────── 主流程 ───────────────────────────────
def build() -> dict:
    # ★ 第一件事就拦: PNG 和 JSON 是一起出的，把 PNG 重画了、JSON 没动，
    #   两者就不一致了（tools/test_camera_qr.py 会拿这对文件自检）。所以要拦就都别写。
    guard_measured_json()

    W = int(round(PAPER_MM[0] * PPM))
    H = int(round(PAPER_MM[1] * PPM))
    margin = int(round(MARGIN_MM * PPM))
    target = int(round(QR_SIZE_MM * PPM))

    print(f"纸张  : {PAPER_MM[0]} x {PAPER_MM[1]} mm  ->  {W} x {H} px  @ {DPI} DPI")
    print(f"二维码: 含白边 {QR_SIZE_MM} mm，白边 {QUIET} 模块，边距 {MARGIN_MM} mm\n")

    canvas = np.full((H, W), 255, np.uint8)        # 白纸

    # 先都渲染出来，拿到真实像素边长（因为模块取整，会略小于 20mm 的名义值）
    symbols = {}
    for text, _ in CORNERS:
        mods = encode_qr_modules(text)
        img, module_px = render_symbol(mods, target)
        symbols[text] = (img, module_px, mods.shape[0])
        n_data = mods.shape[0] - 2 * QUIET
        version = (n_data - 17) // 4          # 版本 n 的边长 = 17 + 4n
        print(f"  {text}: 数据区 {n_data}x{n_data} 模块(版本 {version})"
              f" | 整块 {mods.shape[0]} 模块 | 每模块 {module_px} px"
              f" | 实际边长 {img.shape[0]/PPM:.2f} mm")

    qp = symbols[CORNERS[0][0]][0].shape[0]        # 所有码同尺寸

    # 四角位置（左上角像素坐标）
    pos = {
        "左上": (margin,               margin),
        "右上": (W - margin - qp,      margin),
        "左下": (margin,               H - margin - qp),
        "右下": (W - margin - qp,      H - margin - qp),
    }

    def px2mm(v):
        return round(v / PPM, 2)

    report = {
        "meta": {
            "dpi": DPI,
            "paper_mm": list(PAPER_MM),
            "paper_px": [W, H],
            "px_per_mm": round(PPM, 6),
            "qr_size_mm_nominal": QR_SIZE_MM,
            "qr_size_mm_actual": round(qp / PPM, 3),
            "quiet_zone_modules": QUIET,
            "margin_mm": MARGIN_MM,
        },
        "qr": {},
    }

    for text, where in CORNERS:
        img, module_px, n_mod = symbols[text]
        x, y = pos[where]
        canvas[y:y + qp, x:x + qp] = img

        quiet_px = module_px * QUIET               # 白边像素宽
        # 数据区（不含白边）的四角 —— 这才是二维码真正的符号范围
        sx, sy = x + quiet_px, y + quiet_px
        ex, ey = sx + qp - 2 * quiet_px, sy + qp - 2 * quiet_px

        corners_px = {
            "top_left":     [sx, sy],
            "top_right":    [ex, sy],
            "bottom_right": [ex, ey],
            "bottom_left":  [sx, ey],
        }
        corners_mm = {k: [px2mm(a), px2mm(b)] for k, (a, b) in corners_px.items()}

        report["qr"][text] = {
            "where": where,
            # 含白边的外框
            "outer_rect_px": [x, y, x + qp, y + qp],
            "outer_rect_mm": [px2mm(x), px2mm(y), px2mm(x + qp), px2mm(y + qp)],
            # 不含白边的数据区
            "symbol_rect_px": [sx, sy, ex, ey],
            "symbol_rect_mm": [px2mm(sx), px2mm(sy), px2mm(ex), px2mm(ey)],
            # 数据区四角坐标（你要的"四个角坐标"）
            "symbol_corners_px": corners_px,
            "symbol_corners_mm": corners_mm,
            # 常用参考点：数据区左上角（方案.md 里吸盘要对准的那个点）
            "ref_point_px": [sx, sy],
            "ref_point_mm": [px2mm(sx), px2mm(sy)],
            "center_mm": [px2mm(x + qp / 2), px2mm(y + qp / 2)],
            "module_px": module_px,
            "modules": n_mod,
        }

        # ── 四角定位标记（画在外面，离白边留 2mm 间隙，绝不影响识别）──
        if DRAW_TICK:
            gap = int(round(2.0 * PPM))
            L = int(round(TICK_MM * PPM))
            ox0, oy0 = x - gap, y - gap
            ox1, oy1 = x + qp + gap, y + qp + gap
            t = max(2, int(round(0.3 * PPM)))       # 线宽 0.3mm
            ink = 120
            for cx, cy, dx, dy in ((ox0, oy0, 1, 1), (ox1, oy0, -1, 1),
                                   (ox0, oy1, 1, -1), (ox1, oy1, -1, -1)):
                cv2.line(canvas, (cx, cy), (cx + dx * L, cy), ink, t)
                cv2.line(canvas, (cx, cy), (cx, cy + dy * L), ink, t)

    # ── 文字标注 ──
    if DRAW_LABEL:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(canvas)
        d = ImageDraw.Draw(pil)
        font = load_font(int(round(LABEL_MM * PPM)))
        for text, where in CORNERS:
            info = report["qr"][text]
            ox0, oy0, ox1, oy1 = info["outer_rect_px"]
            label = f"{text}  {where}"
            bb = d.textbbox((0, 0), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            if where.startswith("上"):              # 上排 → 文字写在下边
                tx, ty = (ox0 + ox1) // 2 - tw // 2, oy1 + int(3 * PPM)
            else:                                    # 下排 → 文字写在上边
                tx, ty = (ox0 + ox1) // 2 - tw // 2, oy0 - th - int(6 * PPM)
            d.text((tx, ty), label, fill=0, font=font)
        canvas = np.array(pil)

    ensure_data_dir()
    cv2.imwrite(str(OUT_PNG), canvas)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    build_guide(report, canvas)

    return report


def guard_measured_json(out: Path = None) -> None:
    """
    别把**实测**的标定纸布局覆盖掉。

    ★ 这台机器上的 calib_A4_qr.json 有两种来源:
      ① 本脚本从 A4 排版算出来的（meta.source 不是 "measured-sheet"）
      ② make_layout_from_sheet.py 按卷尺实测重建的（meta.source == "measured-sheet"）
      纸是自己贴的，真数据是 ② —— 而 ① 那套坐标是**别的纸**的，拿它标定必错。

    ★ 踩过: 顺手跑一下本脚本做"重渲染 guide"，就把实测 json 冲掉了，
      只能拿六条尺子数据重新重建一遍。所以这里直接拦住。
    """
    if FORCE_JSON_OVERWRITE:
        return
    p = Path(out) if out else OUT_JSON
    if not p.exists():
        return
    try:
        src = json.loads(p.read_text(encoding="utf-8")).get("meta", {}).get("source")
    except (OSError, json.JSONDecodeError):
        return
    if src == "measured-sheet":
        raise SystemExit(
            f"✗ 拒绝覆盖 {p.name} —— 它是**按卷尺实测**重建的（meta.source="
            f"measured-sheet），不是本脚本生成的。\n"
            f"  本脚本只会按 A4 排版算坐标，那和你的纸对不上；盖掉就没了。\n"
            f"  · 只想重画 guide/PNG: 没用，本脚本 PNG 和 JSON 是一起出的 —— 别跑。\n"
            f"  · 确实想用排版坐标替换: 先备份，再加 --force 重跑。\n"
            f"  · 想重建实测坐标: python3 tools/make_layout_from_sheet.py --paper 297 210 "
            f"--side 26 --ab 213 --cd 214 --ac 131 --bd 131 --ad 251 --bc 251")


def print_report(rep: dict) -> None:
    m = rep["meta"]
    print("\n" + "═" * 78)
    print(f"  A4 标定纸坐标表   {m['paper_mm'][0]}x{m['paper_mm'][1]}mm @ {m['dpi']}DPI"
          f"  ({m['paper_px'][0]}x{m['paper_px'][1]}px, 1mm={m['px_per_mm']:.3f}px)")
    print(f"  二维码含白边 {m['qr_size_mm_actual']}mm，白边 {m['quiet_zone_modules']} 模块")
    print("═" * 78)
    for text, d in rep["qr"].items():
        c = d["symbol_corners_mm"]
        n_data = d["modules"] - 2 * m["quiet_zone_modules"]
        print(f"\n{text}  ({d['where']})   数据区 {n_data}x{n_data} 模块"
              f"(版本 {(n_data - 17) // 4}), 每模块 {d['module_px']}px")
        print(f"   数据区四角 (mm):  左上{c['top_left']}  右上{c['top_right']}"
              f"  右下{c['bottom_right']}  左下{c['bottom_left']}")
        print(f"   数据区四角 (px):  左上{d['symbol_corners_px']['top_left']}"
              f"  右上{d['symbol_corners_px']['top_right']}"
              f"  右下{d['symbol_corners_px']['bottom_right']}"
              f"  左下{d['symbol_corners_px']['bottom_left']}")
        # ★ 四个角全列出来，不写死某一个: 对刀瞄哪个由 step2_teach_coords.py /
        #   step3_hand_eye_calib.py 的 --reference 决定（默认为右上角，见 qr_vision.py）。
        #   这里写死过一个角，后来默认改了、报告就成了误导 —— 别再写死。
        print("   ★数据区四角 (mm) —— 对刀瞄哪个由 --reference 决定:")
        print(f"       左上 corner_tl {c['top_left']}    右上 corner_tr {c['top_right']}")
        print(f"       左下 corner_bl {c['bottom_left']}    右下 corner_br {c['bottom_right']}")
        print(f"       中心 center    {d['center_mm']}")
        print(f"   外框(含白边,别瞄这里) (mm): {d['outer_rect_mm']}")

    # ── 视野自检：算一下摄像头看到时每个模块有几像素 ──
    pw, ph = rep["meta"]["paper_mm"]
    qp_mm  = rep["meta"]["qr_size_mm_actual"]
    n_mod  = rep["qr"][next(iter(rep["qr"]))]["modules"]

    def fit_ppm(cam_w, cam_h, fill=1.0):
        """纸张按原比例塞进画面时，每毫米占多少像素（取宽/高里更紧的那个）"""
        return min(cam_w * fill / pw, cam_h * fill / ph)

    orient_cn = "横版" if pw > ph else "竖版"
    print("\n" + "─" * 78)
    print(f"  视野自检（{orient_cn} {pw:.0f}x{ph:.0f}mm 塞满 16:9 画面，全幅）：")
    for name, cam_w, cam_h in (("1080p", 1920, 1080), ("4K   ", 3840, 2160)):
        ppm_v  = fit_ppm(cam_w, cam_h)
        px_mod = qp_mm * ppm_v / n_mod
        verdict = "✅可解码" if px_mod >= 4.5 else ("⚠️仅能定位" if px_mod >= 2.0 else "❌太小")
        print(f"    {name}: 1mm≈{ppm_v:.2f}px → 纸张占 {pw*ppm_v:.0f}x{ph*ppm_v:.0f}px,"
              f" 二维码≈{qp_mm*ppm_v:.0f}px, 每模块 {px_mod:.1f}px  {verdict}")

    # ── 工作区尺寸：四个参考点围出的矩形 ──
    xs = [d["center_mm"][0] for d in rep["qr"].values()]
    ys = [d["center_mm"][1] for d in rep["qr"].values()]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    print("─" * 78)
    print(f"  工作区（以二维码中心为参考点）: {span_x:.2f} x {span_y:.2f} mm")
    print(f"    左右跨距 {span_x:.2f} mm，前后跨距 {span_y:.2f} mm")
    print(f"    公式: 跨距 = 纸边长 - 2*边距({MARGIN_MM}) - 二维码边长({qp_mm:.1f})")
    print("    ↳ 想缩小工作区（机械臂够不着时）→ 把二维码调大 或 把边距调大")
    print("─" * 78)
    print("  如果你要手动填 step3_hand_eye_calib.py 的 DOBOT_COORDS（一般不必:"
          " step2_teach_coords.py 会自动落盘）:")
    print("    注释里写的就是「数据区四角」，按你 --reference 选的那个抄。默认 corner_tr（右上角）。")
    print("    DOBOT_COORDS = {")
    for text in rep["qr"]:
        print(f'        "{text}": [  ,   ],   # 四角(mm) = '
              f'{rep["qr"][text]["symbol_corners_mm"]}')
    print("    }")
    print("─" * 78)
    print(f"\n✅ 已输出:\n   {OUT_PNG}\n   {OUT_JSON}\n   {OUT_GUIDE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="生成 A4 标定纸 PNG + 坐标表 json（按排版算，不是实测）")
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖实测重建的 calib_A4_qr.json（默认拒绝，"
                         "见 guard_measured_json 的说明）")
    args = ap.parse_args()
    FORCE_JSON_OVERWRITE = args.force
    print_report(build())
