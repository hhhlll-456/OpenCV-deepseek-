#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_suction_qr.py —— 生成吸盘上的「第 5 个二维码」（动态基座定位用）

用途（见《WPS文字文档.wps》一、硬件与物理准备阶段）：
  把这张码贴在吸盘底座正上方（末端法兰或吸盘金属固定块表面），
  让机械臂每次拍照时都能算出「自己的吸嘴此刻在桌面哪个位置」，
  从而不必依赖机械臂基座的绝对精度。

★ 尺寸口径（很重要，别和 step1 混）：
  本脚本 --sizes 给的是**含白边（静默区）的整块边长**，
  与 step1_gen_paper.py 的 QR_SIZE_MM 同一个口径（那里 40mm 也是含白边）。
  数据区（黑色图案）边长 = 整块边长 × 21/29 —— 21 是数据模块数，29 = 21+2×4。
  ★ 而 data/calib_A4_qr.json 里 qr_size_mm_actual=35.905 是**数据区**，
    由 tools/make_layout_from_sheet.py 按卷尺实测反推 —— 两处口径不同，别对着抄。

★ 打印：务必「100% / 实际大小」，不要勾选"适应页面/缩放"。
  PNG 已写入 300 DPI 的 pHYs 元数据，WPS/Word 插入时会自动按名义尺寸摆放；
  但**打完请用卡尺量数据区**，对照本脚本打印的「数据区实测」一行。

★ 为什么还要跑可解码性自检：
  吸盘码必须和桌面 4 个角码**同框**被认出来，于是它和角码是同一个缩放比例。
  角码数据区 26mm（≈6.4 px/模块 @1080p 全幅）才稳；本码按含白边口径
  做出来数据区只有十几毫米，很可能落在 4.5 px/模块 的门槛之下。
  这个脚本会把 1080p / 4K 两种视野下的实际解码结果跑给你看，不靠估。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paths import (DATA_DIR, SUCTION_QR_JSON,   # noqa: E402
                   ensure_data_dir)
from step1_gen_paper import encode_qr_modules   # noqa: E402

# ═════════════════════════════ 可调参数 ═════════════════════════════
CONTENT = "QR_SUCTION"      # 二维码内容。《WPS文字文档.wps》里写的就是这个串。
                            #   注意 "_" 不在 QR 的字母数字模式字符集里，会退化成
                            #   字节模式 —— 但 10 字节仍装得进版本 1，所以无妨。
DPI     = 300               # 打印分辨率
QUIET   = 4                 # 白边宽度（模块数），规范要求 >=4，别改小
# 要出哪几个尺寸（含白边的整块边长）。四个由小到大，覆盖「贴得住」到「1080p 也解得开」：
#   15mm → 数据区 10.9mm，1080p 只有 2.7 px/模块，基本只能定位
#   30mm → 数据区 21.7mm，1080p 约 5.3 px/模块，能解
DEFAULT_SIZES = ["15mm", "20mm", "25mm", "30mm"]

# 相机视野自检用：A4 横版 (297x210) 塞满画面时每毫米多少像素，取宽/高里更紧的那个
PAPER_MM = (297.0, 210.0)
CAMERAS = (("1080p", 1920, 1080), ("4K   ", 3840, 2160))
DECODE_OK_PX_PER_MODULE = 4.5   # 与 step1_gen_paper.py 同一经验阈值
# ════════════════════════════════════════════════════════════════════

MM_PER_INCH = 25.4
PPM = DPI / MM_PER_INCH     # 每毫米多少像素


def parse_size(tag: str) -> float:
    """把 "15mm" / "2cm" 解析成毫米。"""
    t = tag.strip().lower()
    if t.endswith("mm"):
        return float(t[:-2])
    if t.endswith("cm"):
        return float(t[:-2]) * 10.0
    return float(t)          # 裸数字按 mm 处理


def render_exact(modules: np.ndarray, size_mm: float) -> tuple[np.ndarray, int]:
    """
    把模块级位图缩放成**物理尺寸精确等于 size_mm** 的图（size_mm 是含白边的整块）。

    ★ 和 step1_gen_paper.py 的 render_symbol 不同：那边把每模块像素数取整，
      于是名义 40mm 实际只有 39.x mm —— 换来的好处是每模块像素数一致、边缘绝对锐利。
      本脚本反过来优先保证**物理尺寸精确**：整块 29 模块除不尽时，模块宽度会在
      相邻两个整数值之间交替（15mm@300dpi 是 177px/29 = 6.10 → 6px 和 7px 混排）。
      对 15mm 这种尺度，尺寸精度比"每模块完全等宽"更重要，且 INTER_NEAREST
      只是复制像素、不会糊边。
    """
    n = modules.shape[0]
    target_px = int(round(size_mm * PPM))
    img = cv2.resize(modules, (target_px, target_px), interpolation=cv2.INTER_NEAREST)
    return img, target_px


def decode_selfcheck(img: np.ndarray) -> str | None:
    """整块原图直接解一遍，确认内容没写错（尺寸问题交给视野自检）。"""
    ok, decoded, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(img)
    return decoded[0] if ok and decoded else None


def blur_tolerance(img: np.ndarray, content: str) -> float:
    """
    能扛住多大的高斯模糊还解得出来（σ，像素）。

    ★ 为什么不能只判「缩到目标大小还能不能解」:
      纯 INTER_AREA 缩放出来的图是**理想图** —— 没有镜头像差、没有噪声、
      没有 JPEG 压缩。实测 15mm@1080p 在这种理想图上照样解得开（2.66px/模块），
      可只要加一点点模糊(σ=0.8)就立刻崩。拿理想图给"✅可解码"是自欺欺人，
      贴到吸盘上才发现扫不出来。
      σ 是个粗糙的代理，但"能扛住 σ≥1.0"和"σ=0 才勉强"在真实相机上差别巨大。
    """
    for sigma in (0.0, 0.8, 1.2, 1.6, 2.0):
        probe = img if sigma == 0 else cv2.GaussianBlur(img, (0, 0), sigma)
        if decode_selfcheck(probe) != content:
            return round(max(0.0, sigma - 0.4), 1)   # 返回"最后一档扛住"的 σ
    return 2.0


def verdict_for(px_per_module: float, sigma: float) -> str:
    """把「每模块像素数 + 抗模糊能力」合成一句人话。"""
    if px_per_module < 2.0:
        return "❌太小"
    if sigma >= 1.2:
        return "✅可解码"
    if sigma >= 0.8:
        return "⚠️勉强（真实相机可能扫不出）"
    return "❌仅在理想图上能解，真实相机必失败"


def print_baseline(mods: np.ndarray, n_total: int, content: str) -> None:
    """
    拿**现有纸面码**当基准跑一遍同样的自检。

    ★ 为什么要这个基准: "σ 够不够大"本身没有绝对标准，但现有 P1~P4 是已经
      在用的、确实扫得出来的码。凡是抗模糊能力不如它的尺寸，就知道不能贴。
      calib_A4_qr.json 里 qr_size_mm_actual=35.905 是**数据区**，此处换算成
      同口径的整块边长再比（35.905 x 29/21）。
    """
    data_mm = 26.0
    outer_mm = data_mm * n_total / (n_total - 2 * QUIET)
    img, _ = render_exact(mods, outer_mm)
    print(f"\n── 基准：现有纸面码（数据区 {data_mm:.1f}mm，整块 {outer_mm:.1f}mm）──")
    for name, cw, ch in CAMERAS:
        ppm_v = min(cw / PAPER_MM[0], ch / PAPER_MM[1])
        w = max(8, int(round(outer_mm * ppm_v)))
        sim = cv2.resize(img, (w, w), interpolation=cv2.INTER_AREA)
        per_mod = ppm_v * outer_mm / n_total
        sigma = blur_tolerance(sim, content)
        print(f"   {name.strip()}: 每模块 {per_mod:.2f}px, 抗模糊 σ={sigma}"
              f"  {verdict_for(per_mod, sigma)}")
    print("   ↳ 比这条线差的尺寸，贴上去大概率扫不出来。")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="生成吸盘上的第 5 个二维码（尺寸为含白边的整块边长）")
    ap.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES,
                    help=f"整块边长，写 '15mm' / '2cm' / 裸数字(mm)。默认 {' '.join(DEFAULT_SIZES)}")
    ap.add_argument("--content", default=CONTENT,
                    help=f"二维码内容，默认 {CONTENT!r}")
    ap.add_argument("--no-check", action="store_true",
                    help="跳过可解码性自检（只是生成图片时快一点）")
    args = ap.parse_args()

    mods = encode_qr_modules(args.content, quiet=QUIET)
    n_total = mods.shape[0]                       # 含白边的总模块数
    n_data = n_total - 2 * QUIET                  # 数据模块数（黑色图案）
    version = (n_data - 17) // 4
    print(f"内容 {args.content!r} → 数据区 {n_data}x{n_data} 模块 (版本 {version})，"
          f"含白边 {n_total} 模块")

    ensure_data_dir()
    report = {
        "content": args.content,
        "dpi": DPI,
        "quiet_zone_modules": QUIET,
        "data_modules": n_data,
        "total_modules": n_total,
        "version": version,
        "size_basis": "outer_with_quiet_zone",
        "note": ("边长口径 = 含白边的整块。数据区(mm) = 整块(mm) x "
                 f"{n_data}/{n_total}。与 step1_gen_paper.py 的 QR_SIZE_MM 同口径；"
                 "与 calib_A4_qr.json 的 qr_size_mm_actual(数据区口径) 不同。"),
        "variants": {},
    }

    if not args.no_check:
        print_baseline(mods, n_total, args.content)

    for tag in args.sizes:
        size_mm = parse_size(tag)
        # img 已含白边，直接就是可打印的整块
        canvas, px = render_exact(mods, size_mm)

        out = DATA_DIR / f"suction_qr_{tag}.png"
        # ★ 用 PIL 存，为的是写进 pHYs(300 DPI) —— cv2.imwrite 不写这个块，
        #   没有它 WPS/Word 插入时只能按 96dpi 算，图会大 3 倍多。
        Image.fromarray(canvas).save(out, dpi=(DPI, DPI))

        outer_actual = px / PPM
        data_actual = outer_actual * n_data / n_total
        module_mm = outer_actual / n_total

        report["variants"][tag] = {
            "outer_mm_nominal": size_mm,
            "outer_mm_actual": round(outer_actual, 3),
            "data_mm_actual": round(data_actual, 3),
            "module_mm": round(module_mm, 4),
            "png_px": px,
            "file": out.name,
        }

        print(f"\n{tag}: 整块名义 {size_mm:.2f}mm → 实际 {outer_actual:.3f}mm"
              f" ({px}px) | 数据区 {data_actual:.3f}mm | 每模块 {module_mm:.4f}mm")
        print(f"     → {out}")

        got = decode_selfcheck(canvas)
        print(f"     自检解码: {'✅ ' + repr(got) if got == args.content else '❌ ' + repr(got)}")

        if not args.no_check:
            print("     视野自检（和桌面 4 角码同框 = 同一缩放比例）:")
            for name, cw, ch in CAMERAS:
                ppm_v = min(cw / PAPER_MM[0], ch / PAPER_MM[1])
                width_px = max(8, int(round(outer_actual * ppm_v)))
                per_mod = ppm_v * module_mm
                sim = cv2.resize(canvas, (width_px, width_px),
                                 interpolation=cv2.INTER_AREA)
                sigma = blur_tolerance(sim, args.content)
                verdict = verdict_for(per_mod, sigma)
                print(f"       {name.strip()}: 整块≈{width_px}px,"
                      f" 每模块 {per_mod:.2f}px (经验阈值 {DECODE_OK_PX_PER_MODULE})"
                      f"  {verdict}")
                report["variants"][tag][f"decode_{name.strip()}"] = {
                    "px_per_module": round(per_mod, 2),
                    "blur_tolerance_sigma": sigma,
                    "verdict": verdict,
                }

    SUCTION_QR_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(f"\n✅ 几何已落盘: {SUCTION_QR_JSON}")
    print("   下一步（《WPS文字文档.wps》一）：贴到吸盘底座正上方、与桌面严格平行，"
          "再用游标卡尺量吸嘴中心相对本码中心的偏移 Δx_tip / Δy_tip。")


if __name__ == "__main__":
    main()
