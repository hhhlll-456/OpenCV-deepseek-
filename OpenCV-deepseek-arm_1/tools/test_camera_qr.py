#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_camera_qr.py —— 摄像头 QR 码检测/解码自检工具

验证《方案.md》第三阶段的前置条件：摄像头能不能「一眼看全 + 解出」P1~P4 四个标定码。
只做检测，不算矩阵、不连机械臂 —— 先确认这一步百分百可靠，再跑 step3_hand_eye_calib.py。

用法:
    python3 tools/test_camera_qr.py                      # 实时预览窗口，带检测框
    python3 tools/test_camera_qr.py --shot               # 等对焦+积累30帧，打印结论后退出
    python3 tools/test_camera_qr.py --stress             # ★换地方后先跑这个：量化机位余量
    python3 tools/test_camera_qr.py --width 2592 --height 1944
    python3 tools/test_camera_qr.py --cam 2              # 手动指定 /dev/video2（默认自动挑）
    python3 tools/test_camera_qr.py --loop               # 反复自检，直到4码齐全（调机位时用）

预览窗口按键:
    s  保存当前帧 + 标注图      q / ESC  退出

★ 检测/对焦/取角的逻辑都在 qr_vision.py 里（本文件只负责「看」和「调机位」），
  和 step3_hand_eye_calib.py 共用同一份，别在这里再抄一遍 —— 抄两份必然慢慢跑偏。

★ 实测踩到的三个坑（qr_vision.py 已内建处理，别再踩）:
  1. 自动对焦很慢: 实测 X6L 要 5~7 秒才收住，DCX-5MAF 快一些。预热不足时
     blur≈2~28、4码全解不出；等够 blur≈310 才稳定解出。→ wait_for_focus()
  2. 视角要正: 纸必须完整落在画面内。实测「P3/P4 被下边缘切掉」时它们永远解不出，
     和分辨率无关。→ 边缘裁切告警
  3. 每模块像素 ≥4.5 稳定解码；实测这颗相机在清晰时 ~3.4px/模块 也能解出。

注意: 图像上的文字只用英文（OpenCV 内置字体不支持中文），终端输出是中文。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 本脚本在 tools/ 下，公共库和数据在 src/、data/ —— 先把自己上一级加进来
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from qr_vision import (DATA_SIDE_MM, FOCUS_TIMEOUT, GOOD_PX_PER_MODULE,   # noqa: E402
                       MODULES_TOTAL, OK_PX_PER_MODULE, QR_SIDE_MM, blur_score,
                       describe_cameras, detect, load_paper_layout, open_camera,
                       paper_fit, px_per_mm, px_per_module, quad_center,
                       quad_side_px, touches_edge, wait_for_focus)
from paths import PAPER_JSON, SHOTS_DIR, ensure_output_dir   # noqa: E402

# JSON 读不到时的兜底（与 step1_gen_paper.py 默认值一致）
DEFAULT_CODES = ["P1", "P2", "P3", "P4"]

# 画在图像上的文字只能用英文（OpenCV 内置字体不支持中文），终端输出才用中文
SIDE_EN = {"上": "TOP", "下": "BOTTOM", "左": "LEFT", "右": "RIGHT"}


# ─────────────────────── 读取标定纸信息（可选） ───────────────────────
def load_expected():
    """
    从 calib_A4_qr.json 读标定纸布局。返回 (PaperLayout | None, 码列表, 模块数, 边长mm)。
    读不到就退回默认值（此时拿不到纸面尺寸，"整张纸在不在画面里"这项检查跳过）。
    """
    try:
        lay = load_paper_layout(PAPER_JSON)
        return lay, lay.codes, lay.modules, lay.qr_side_mm
    except Exception as e:
        print(f"[提示] 读 {PAPER_JSON} 失败，改用默认值: {e}")
        return None, DEFAULT_CODES, MODULES_TOTAL, QR_SIDE_MM


# ─────────────────────────── 画面绘制 ───────────────────────────
def annotate(frame: np.ndarray, decoded: dict, located: list,
             expected: list[str], fps: float, blur: float,
             layout=None) -> np.ndarray:
    vis = frame.copy()
    shape = frame.shape

    # ── 构图辅助: 中心十字 + 安全框。把整张纸放进安全框内，4个角就不会被裁掉 ──
    h, w = shape[0], shape[1]
    mx, my = int(w * 0.04), int(h * 0.04)
    cv2.rectangle(vis, (mx, my), (w - mx, h - my), (140, 140, 140), 1)
    cx, cy = w // 2, h // 2
    cv2.line(vis, (cx - 22, cy), (cx + 22, cy), (140, 140, 140), 1)
    cv2.line(vis, (cx, cy - 22), (cx, cy + 22), (140, 140, 140), 1)

    # 只定位到、没解出内容的码 —— 橙色框
    for quad in located:
        col = (0, 165, 255)
        cv2.polylines(vis, [quad.astype(np.int32)], True, col, 2)
        c = quad_center(quad).astype(int)
        tag = "CUT BY EDGE" if touches_edge(quad, shape) else "LOCATED?"
        cv2.putText(vis, tag, (c[0] - 70, c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, col, 2, cv2.LINE_AA)

    # 解出内容的码 —— 绿色框 + 内容 + 每模块像素
    for text, quad in decoded.items():
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 220, 0), 3)
        ppm_mod = px_per_module(quad)
        c = quad_center(quad).astype(int)
        cv2.circle(vis, tuple(c), 5, (0, 220, 0), -1)
        cv2.putText(vis, f"{text}  {ppm_mod:.1f}px/mod",
                    (c[0] - 90, c[1] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"({c[0]},{c[1]})", (c[0] - 60, c[1] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA)

    # ── 整张纸的投影轮廓：能看出来纸有没有出画（出画=青色→红色）──
    if layout is not None:
        fit = paper_fit(decoded, layout, shape)
        if fit is not None:
            ppts, cut, over_mm = fit
            col = (0, 0, 255) if cut else (255, 200, 0)
            cv2.polylines(vis, [ppts.astype(np.int32)], True, col, 2)
            if cut:
                # 写在纸的下边缘附近（左上角被 HUD 占了）。
                # 边名用英文: OpenCV 内置字体画不了中文，会变成一串 ???。
                cv2.putText(vis, f"PAPER CUT {'/'.join(SIDE_EN[c] for c in cut)}"
                                 f" {over_mm:.0f}mm",
                            (max(8, int(ppts[:, 0].min()) + 8),
                             min(shape[0] - 12, int(ppts[:, 1].max()) - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)

    found = [c for c in expected if c in decoded]
    missing = [c for c in expected if c not in decoded]
    if decoded:
        min_ppm = min(px_per_module(q) for q in decoded.values())
    else:
        min_ppm = 0.0
    if min_ppm >= GOOD_PX_PER_MODULE:
        verdict, vcol = "GOOD - stable decode", (0, 220, 0)
    elif min_ppm >= OK_PX_PER_MODULE:
        verdict, vcol = "MARGINAL - get closer", (0, 165, 255)
    else:
        verdict, vcol = "POOR", (0, 0, 255)

    lines = [
        (f"{shape[1]}x{shape[0]}  {fps:4.1f} FPS   blur={blur:5.0f}", (255, 255, 255)),
        (f"found {len(found)}/{len(expected)}   missing: {' '.join(missing) or '-'}",
         (0, 220, 0) if not missing else (0, 0, 255)),
        (f"px/module min = {min_ppm:.1f}  ->  {verdict}", vcol),
        ("press s=save  q=quit", (200, 200, 200)),
    ]
    y = 30
    for txt, col in lines:
        cv2.putText(vis, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        y += 30
    return vis


# ─────────────────────────── 摄像头 ───────────────────────────
# ★ 这里曾经把 qr_vision.open_camera / wait_for_focus 又抄了一份，
#   结果同名函数遮蔽了上面的 import，两边慢慢跑偏 —— 已删掉，统一用 qr_vision 的。


def report(decoded: dict, located: list, expected: list[str],
           shape: tuple[int, ...], blur: float, layout=None) -> tuple[bool, bool | None]:
    """
    打印一帧的诊断结果。
    返回 (4码是否齐全, 整张纸是否完整在画面内)。纸的位置判断不了时第二项为 None。
    """
    found = [c for c in expected if c in decoded]
    missing = [c for c in expected if c not in decoded]

    print(f"  画面 {shape[1]}x{shape[0]}   blur={blur:.0f}")
    print(f"  解出 {len(found)}/{len(expected)} : {found if found else '无'}")
    if missing:
        print(f"  ★ 缺失     : {missing}")

    # 被边缘裁切的码：这是“永远解不出”的硬伤，优先告警
    cut = [q for q in list(located) + list(decoded.values()) if touches_edge(q, shape)]
    if cut:
        print(f"  🚨 有 {len(cut)} 个码贴到画面边缘 → 被裁切了，永远解不出。"
              f"把摄像头抬高/拉远，让整张纸(含白边)完整进入画面。")
    if located:
        print(f"  仅定位未解码: {len(located)} 个（每模块像素不够或对焦没实）")

    if decoded:
        ppms, ppmms = [], []
        for text, quad in decoded.items():
            side = quad_side_px(quad)
            ppm_mod = px_per_module(quad)
            ppm_ms = px_per_mm(quad)
            c = quad_center(quad)
            ppms.append(ppm_mod)
            ppmms.append(ppm_ms)
            print(f"  {text}: 数据区边长 {side:.0f}px, 每模块 {ppm_mod:.2f}px, "
                  f"中心 ({c[0]:.0f},{c[1]:.0f})")
        mn = min(ppms)
        print(f"  最差码每模块 {mn:.2f}px", end="  ")
        if mn >= GOOD_PX_PER_MODULE:
            print("→ ✅ 解码稳定")
        elif mn >= OK_PX_PER_MODULE:
            print("→ ⚠️ 勉强，建议把二维码放大或摄像头靠近")
        else:
            print("→ ❌ 太小，解不出来")
        # 视野覆盖：用真实 px/mm 算，才能判断整张纸装不装得下
        ppm = min(ppmms)
        print(f"  实测 {ppm:.2f} px/mm → 当前画面可覆盖 "
              f"{shape[1] / ppm:.0f} x {shape[0] / ppm:.0f} mm "
              f"(标定纸 297x210mm，够盖住就能一眼看全)")

    # ★ 整张纸在不在画面里 —— 投影法精确判断，比上面那个估算的覆盖范围靠谱
    paper_ok: bool | None = None
    if layout is not None:
        fit = paper_fit(decoded, layout, shape)
        if fit is not None:
            _pts, cut, over_mm = fit
            paper_ok = not cut
            if cut:
                print(f"  🚨 整张纸没进画面: {'/'.join(cut)}边被切掉约 {over_mm:.0f}mm。"
                      f"码现在还能解出，是因为它们离纸边有 15mm —— 但机位已经到极限，"
                      f"稍微再挪一下就会开始丢码。")
                print(f"     → 把摄像头抬高/拉远，让纸的{''.join(cut)}边也进来。")
            else:
                print("  ✅ 整张纸完整在画面内（四角都没出界）。")
    return not missing, paper_ok


def _n_decoded(img: np.ndarray, det, expected: list[str]) -> int:
    decoded, _ = detect(img, det)
    return sum(1 for c in expected if c in decoded)


def stress_test(frame: np.ndarray, det, expected: list[str]) -> float:
    """
    在当前实拍帧上做「降级模拟」，量化这套机位还剩多少余量。

    为什么需要它: 「这台机位能不能用」的实测结论只对当时的距离/光照/对焦成立，
    换个地方架摄像头，最先崩的通常是距离。这个测试把余量变成数字，
    不用靠猜。实测本纸的失效点: px/模块 <~3.5 开始丢码。
    """
    n_all = len(expected)

    # ① 距离：整帧缩小 = 摄像头拉远
    dist = [round(s, 2) for s in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)]
    dist_fail = None
    for s in dist:
        im = frame if s == 1.0 else cv2.resize(
            frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if _n_decoded(im, det, expected) < n_all:
            dist_fail = s
            break

    # ② 光照：整帧乘系数变暗
    bright = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.15]
    bright_fail = None
    for g in bright:
        im = frame if g == 1.0 else np.clip(
            frame.astype(np.float32) * g, 0, 255).astype(np.uint8)
        if _n_decoded(im, det, expected) < n_all:
            bright_fail = g
            break

    # ③ 对焦：高斯模糊
    blur = [0.0, 0.6, 1.0, 1.4, 1.8, 2.2, 2.8]
    blur_fail = None
    for sg in blur:
        im = frame if sg == 0 else cv2.GaussianBlur(frame, (0, 0), sg)
        if _n_decoded(im, det, expected) < n_all:
            blur_fail = sg
            break

    # ④ 低光噪点：加性高斯噪声
    rng = np.random.default_rng(0)
    noise = [0, 8, 16, 25, 40, 60]
    noise_fail = None
    for sd in noise:
        im = frame if sd == 0 else np.clip(
            frame.astype(np.float32) + rng.normal(0, sd, frame.shape),
            0, 255).astype(np.uint8)
        if _n_decoded(im, det, expected) < n_all:
            noise_fail = sd
            break

    # 基线 px/模块（用整帧直接算的平均值）
    decoded, _ = detect(frame, det)
    if not decoded:
        print("\n  ⚠️ 基线帧就没解出，无法做余量测试。先调好机位再跑 --stress。")
        return 0.0
    ppm_mod = min(px_per_module(q) for q in decoded.values())
    ppm_true = min(px_per_mm(q) for q in decoded.values())

    print("\n" + "─" * 70)
    print("  机位余量压力测试（在刚才这一帧上模拟“换地方”会变的4件事）")
    print(f"  基线: {ppm_mod:.2f} px/模块, 解出 {len(decoded)}/{n_all}")
    print("─" * 70)

    # 每项给出「还能恶化多少」的直观说法 + 充裕/尚可/偏薄
    thin = []
    print("  ① 距离（摄像头拉远）")
    if dist_fail is None:
        print("     拉到 0.5x 仍全解                              ✅ 充裕")
    else:
        px_lim = ppm_mod * dist_fail
        # px/模块是距离的直接后果，用实测失效点 3.5 作判据
        v = "✅ 充裕" if px_lim >= 4.5 else ("⚠️ 尚可" if px_lim >= 3.5 else "❌ 偏薄")
        if px_lim < 4.5:
            thin.append("距离")
        print(f"     拉远到 {dist_fail:.2f}x（{(1 - dist_fail) * 100:.0f}%）开始丢码，"
              f"此时 {px_lim:.2f} px/模块   {v}")

    print("  ② 光照变暗")
    if bright_fail is None:
        print("     暗到 0.15x 仍全解                             ✅ 充裕")
    else:
        v = "✅ 充裕" if bright_fail <= 0.45 else "⚠️ 尚可"
        if bright_fail > 0.45:
            thin.append("光照")
        print(f"     暗到 {bright_fail * 100:.0f}% 亮度开始丢码                {v}")

    print("  ③ 失焦（模糊）")
    if blur_fail is None:
        print("     sigma 2.8 仍全解                              ✅ 充裕")
    else:
        v = "✅ 充裕" if blur_fail >= 1.4 else "⚠️ 尚可"
        if blur_fail < 1.4:
            thin.append("对焦")
        print(f"     模糊 sigma={blur_fail:.1f} 开始丢码                     {v}")

    print("  ④ 低光噪点（ISO）")
    if noise_fail is None:
        print("     sd 60 仍全解                                  ✅ 充裕")
    else:
        v = "✅ 充裕" if noise_fail >= 25 else "⚠️ 尚可"
        if noise_fail < 25:
            thin.append("噪点")
        print(f"     噪声 sd={noise_fail} 开始丢码                        {v}")

    print("─" * 70)
    # 结论以实测的「还能拉远多少」为准 —— 距离是换地方后最先崩的一项，
    # 不能再拿另一个理论阈值去说，否则会和上面①自相矛盾。
    dist_room = (1 - dist_fail) if dist_fail is not None else 0.5
    if dist_room >= 0.3:
        print(f"  结论: {ppm_mod:.2f} px/模块，实测还能拉远 {dist_room * 100:.0f}% 才丢码"
              f" → 余量充足，换地方重架后大概率照样能用。")
    elif dist_room >= 0.15:
        print(f"  结论: {ppm_mod:.2f} px/模块，实测只能拉远 {dist_room * 100:.0f}% 就会丢码"
              f" → 够用但偏薄，换地方时摄像头别架得比现在远。")
    else:
        print(f"  结论: {ppm_mod:.2f} px/模块，实测只能拉远 {dist_room * 100:.0f}% 就会丢码"
              f" → 太薄，先把摄像头架近一点再继续。")
    if thin:
        print(f"  先崩的项: {'、'.join(thin)} —— 换地方时优先盯这几个。")
    if dist_room < 0.3:
        # 覆盖宽度用真实 px/mm 算（不是 px/模块那个历史口径，混用会差 38%）
        cur_w = frame.shape[1] / ppm_true
        tgt_w = frame.shape[1] / (5.0 * MODULES_TOTAL / DATA_SIDE_MM)   # 目标 5 px/模块
        print(f"  建议: 把摄像头靠近，让画面覆盖宽度从 ~{cur_w:.0f}mm 缩到 "
              f"~{tgt_w:.0f}mm（纸 297mm），即 px/模块 从 {ppm_mod:.1f} 提到 ≥5。")
    print("─" * 70)
    return ppm_mod


def save_pair(save_dir: Path, frame: np.ndarray, vis: np.ndarray) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw = save_dir / f"shot_{stamp}.jpg"
    ann = save_dir / f"shot_{stamp}_annotated.jpg"
    cv2.imwrite(str(raw), frame)
    cv2.imwrite(str(ann), vis)
    return raw, ann


# ─────────────────────────── 主流程 ───────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="摄像头 QR 码检测自检")
    ap.add_argument("--cam", type=int, default=None,
                    help="摄像头序号；默认不填 = 按设备名自动挑外接摄像头"
                         "（换摄像头/换USB口后索引会变，别写死）")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--shot", action="store_true",
                    help="等对焦收敛 + 积累若干帧，打印结论后退出")
    ap.add_argument("--frames", type=int, default=30,
                    help="--shot 模式下积累多少帧（默认30，约3秒）")
    ap.add_argument("--loop", action="store_true",
                    help="反复自检直到4码齐全（调整机位时用，Ctrl-C 退出）")
    ap.add_argument("--stress", action="store_true",
                    help="跑机位余量压力测试：模拟拉远/变暗/失焦/噪点，"
                         "判断换地方后会不会失误")
    ap.add_argument("--focus-timeout", type=float, default=FOCUS_TIMEOUT)
    ap.add_argument("--save", default=None, help="保存目录，默认 output/shots")
    args = ap.parse_args()

    lay, expected, modules, qr_mm = load_expected()
    ensure_output_dir()
    save_dir = Path(args.save) if args.save else SHOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 70)
    print("  摄像头 QR 码自检")
    print(f"  期望内容: {expected}   每个码 {modules} 模块(含4模块白边), 边长 {qr_mm}mm")
    print("═" * 70)

    cap = open_camera(args.cam, args.width, args.height)
    if not cap.isOpened():
        print("❌ 打不开摄像头" + (f" /dev/video{args.cam}" if args.cam is not None
                                  else "（自动挑选失败）"))
        print("   当前系统里的视频节点:")
        print(describe_cameras())
        print("   排查: 1) 摄像头插好了吗(lsusb 能看到吗)"
              "  2) 是否被其它程序占用  3) 手动指定 --cam N")
        return 2
    ok, f0 = cap.read()
    if not ok or f0 is None:
        cap.release()
        print("❌ 摄像头打开了但取不到画面")
        return 2
    res = f0.shape[1], f0.shape[0]
    print(f"✅ 摄像头已打开，实际输出 {res[0]}x{res[1]}")

    det = cv2.QRCodeDetector()

    print("  等自动对焦收敛…")
    frame, blur, focus_counts = wait_for_focus(cap, det, expected,
                                               timeout=args.focus_timeout)
    if frame is None:
        cap.release()
        print("❌ 一直取不到画面")
        return 2

    try:
        # ── 一次性自检：积累多帧取并集 ──
        if args.shot or args.loop or args.stress:
            while True:
                union: dict[str, np.ndarray] = {}
                located_seen = []
                best_frame, best_vis = frame, None
                nframes = max(1, args.frames)
                for i in range(nframes):
                    ok, f = cap.read()
                    if not ok or f is None:
                        continue
                    decoded, located = detect(f, det)
                    for k, v in decoded.items():
                        union.setdefault(k, v)
                    if located:
                        located_seen.append(located)
                    if len(union) >= len(expected):
                        best_frame = f
                        break
                decoded_all = union
                located_flat = [q for sub in located_seen for q in sub]
                good, paper_ok = report(decoded_all, located_flat, expected,
                                        frame.shape, blur, layout=lay)
                best_vis = annotate(best_frame, decoded_all, located_flat,
                                    expected, 0.0, blur, layout=lay)
                raw, ann = save_pair(save_dir, best_frame, best_vis)
                print(f"  已保存: {raw}\n          {ann}")
                # 「能用」= 4码齐全 且 整张纸没被切（切了就离丢码只差一点点挪动）
                ready = good and paper_ok is not False
                if args.stress:
                    if good:
                        stress_test(best_frame, det, expected)
                    else:
                        print("\n  ⚠️ 当前机位连基线都解不全，先修好再跑 --stress。")
                if ready:
                    print("\n🎉 4 个码全部解出、整张纸都在画面里 —— 标定的前置条件满足了。")
                    print("   下一步: ① 用 DobotStudio 对刀，把 P1~P4 的 XY 填进"
                          " step3_hand_eye_calib.py 的 DOBOT_COORDS")
                    print("           ② python3 src/step3_hand_eye_calib.py --selftest  (先验证数学)")
                    print("           ③ python3 src/step3_hand_eye_calib.py           (正式标定)")
                    return 0
                if not args.loop:
                    if good and paper_ok is False:
                        print("\n⚠️ 码是解全了，但纸被切掉一条边（见上面的 🚨）—— "
                              "现在能用，但机位到极限了，\n   稍微再挪一下就会丢码。"
                              "建议按提示把摄像头抬高/拉远一点再重跑。")
                    else:
                        print("\n⚠️ 没解全。按上面提示(边缘裁切 / 每模块像素 / 对焦)"
                              "调整后重跑；\n   或加 --loop 反复自检边调边看。")
                    return 1
                print("\n   (--loop 模式，2 秒后继续自检… Ctrl-C 退出)\n")
                time.sleep(2.0)
                frame, blur, focus_counts = wait_for_focus(
                    cap, det, expected, timeout=args.focus_timeout, quiet=True)

        # ── 实时预览 ──
        win = "QR check  (s=save  q=quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        t0, n, fps = time.time(), 0, 0.0
        last_print = 0.0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("取帧失败，退出")
                break
            decoded, located = detect(frame, det)
            blur = blur_score(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            n += 1
            if n % 15 == 0:
                fps = 15 / max(1e-6, time.time() - t0)
                t0 = time.time()
            vis = annotate(frame, decoded, located, expected, fps, blur, layout=lay)
            cv2.imshow(win, vis)

            # 终端每 2 秒同步一次状态，方便没窗口/远程时也能看
            now = time.time()
            if now - last_print > 2.0:
                found = [c for c in expected if c in decoded]
                missing = [c for c in expected if c not in decoded]
                msg = f"解出 {len(found)}/{len(expected)}  blur={blur:.0f}"
                if missing:
                    msg += f"，缺 {' '.join(missing)}"
                if any(touches_edge(q, frame.shape) for q in located):
                    msg += "  🚨有码被边缘裁切"
                if lay is not None:
                    fit = paper_fit(decoded, lay, frame.shape)
                    if fit is not None and fit[1]:
                        msg += f"  🚨纸的{'/'.join(fit[1])}边出画({fit[2]:.0f}mm)"
                    elif fit is not None:
                        msg += "  ✅整张纸在画面内"
                print(f"[{time.strftime('%H:%M:%S')}] {msg}")
                last_print = now

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                raw, ann = save_pair(save_dir, frame, vis)
                print(f"已保存: {raw}\n        {ann}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
