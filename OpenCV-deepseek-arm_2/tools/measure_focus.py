#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
measure_focus.py —— 量出「手动焦距该锁在哪个值」，也就是 qr_vision.FOCUS_LOCK。

★ 这个脚本在量什么、为什么非量不可
  吸盘码和纸面**不在同一个焦面**: 吸盘码离相机 ~150mm、纸面 ~340mm。
  这颗相机（GP_Flip_Mirror 8M USB camera）的自动对焦会停在一个**随它高兴**的
  位置上，而且它"对好了"的判据（prime 的是纸面码能不能解）太宽松 —— 纸面码在
  FOCUS=220 和 255 都解得出来，它就不管吸盘码死活。结果吸盘码「时而解得出、
  时而解不出」，30 帧里一帧都撞不上时整个流程就卡在「没解出吸盘码」。

  唯一的解是**别让它自己决定** —— 手动把焦距锁死在一个「5 个码一帧全中」的值上。
  这个值跟「相机架在哪、机械臂工作在多高」绑定，换机位就得重量。

★ 怎么量
  扫描整段 FOCUS（UVC V4L2_CID_FOCUS_ABSOLUTE 口径，0~1023），每个值拍若干帧，
  数「吸盘码中几帧」和「纸面 4 码**全中**几帧」。两个数同时满的那一段就是答案。

  ★ 只开相机、**不动机械臂**（臂停在哪儿就量哪儿）。结束时把自动对焦还原。
  ★ 量的时候把吸嘴摆到**实际工作高度**上 —— 这个值是对着那个高度量的。

用法:
    python3 tools/measure_focus.py                    # 全量程粗扫
    python3 tools/measure_focus.py 210 370 10 12      # 细扫 210~370 步长 10，每值 12 帧
    python3 tools/measure_focus.py --restore          # 只把自动对焦打开就退出
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import cv2                                                            # noqa: E402
import numpy as np                                                    # noqa: E402
import qr_vision as qv                                                # noqa: E402
from paths import OUTPUT_DIR, PAPER_JSON, ensure_output_dir           # noqa: E402
from qr_vision import blur_score, detect_sweep, load_paper_layout     # noqa: E402

TILE = 200                  # 清晰度网格边长: 取「最清晰那小块」，不依赖码在哪个像素
COARSE_N = 6
COARSE_POINTS = 21


def tile_max(gray):
    """
    把画面切成 TILE 的格子，返回 (最大清晰度, 那个格子的左上角)。

    ★ 为什么取「最大块」而不是「码那个框」: 臂停在高处时脚本并不知道吸盘码
      落在哪个像素，预测一偏、框就框错地方，测出来的是背景。网格取最大则
      不管码在哪都能扫到，臂高度和脚本假设不一致也不怕。
    """
    best, where = -1.0, (0, 0)
    h, w = gray.shape
    for y in range(0, h - TILE + 1, TILE):
        for x in range(0, w - TILE + 1, TILE):
            s = blur_score(gray[y:y + TILE, x:x + TILE])
            if s > best:
                best, where = s, (x, y)
    return best, where


def sweep(cap, det, want_paper, vals, n_frames):
    """扫一遍 vals，返回每行结果。顺带把命中帧存成图（结果里 hits>0 的那些值）。"""
    ensure_output_dir()
    print(f"  {'FOCUS':>6} {'吸盘码':>8} {'纸面全中':>9} {'整幅清晰':>9} "
          f"{'最清晰块':>9}  格子")
    print("  " + "-" * 62)
    rows = []
    for v in vals:
        cap.set(cv2.CAP_PROP_FOCUS, v)
        time.sleep(0.6)                       # 等镜头走完
        for _ in range(3):                    # 丢缓冲（MJPG 里留着上一档的画面）
            cap.read()
        hits_s, hits_p, gs, tm, tw, last = 0, 0, [], -1.0, (0, 0), None
        for _ in range(n_frames):
            ok, f = cap.read()
            if not ok or f is None:
                continue
            last = f
            got, _ = detect_sweep(f, det, want=want_paper + ["QR_SUCTION"])
            if "QR_SUCTION" in got:
                hits_s += 1
            if sum(1 for c in want_paper if c in got) == len(want_paper):
                hits_p += 1
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            gs.append(blur_score(g))
            t, w = tile_max(g)
            if t > tm:
                tm, tw = t, w
        rows.append({"v": v, "s": hits_s, "p": hits_p,
                     "gf": float(np.mean(gs)) if gs else 0.0, "tm": tm})
        print(f"  {v:>6} {hits_s:>6}/{n_frames} {hits_p:>7}/{n_frames} "
              f"{rows[-1]['gf']:>9.1f} {tm:>9.1f}  ({tw[0]},{tw[1]})")
        if last is not None and hits_s > 0:
            cv2.imwrite(str(OUTPUT_DIR / f"focus_hit_{v}.jpg"), last)
    return rows


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--restore" in sys.argv:
        cap = qv.open_camera(None)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            print(f"自动对焦已打开: AUTOFOCUS={cap.get(cv2.CAP_PROP_AUTOFOCUS):g}")
            cap.release()
        return 0

    if len(argv) >= 2:
        lo, hi = int(argv[0]), int(argv[1])
        step = int(argv[2]) if len(argv) > 2 else 10
        n_frames = int(argv[3]) if len(argv) > 3 else 12
        fine = True
    else:
        lo = hi = step = None
        n_frames = COARSE_N
        fine = False

    layout = load_paper_layout(PAPER_JSON)
    want_paper = list(layout.codes)
    det = cv2.QRCodeDetector()

    cap = qv.open_camera(None, focus=-1)      # 先别锁，本脚本要自己扫
    if not cap.isOpened():
        print("✗ 打不开摄像头。别的程序占着？/dev/video* 在不在？")
        return 1
    rows = []
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        probe = []
        for v in (0, 255, 500, 1023, 2047):
            cap.set(cv2.CAP_PROP_FOCUS, v)
            probe.append((v, cap.get(cv2.CAP_PROP_FOCUS)))
        fmax = int(max(b for _, b in probe))
        print("── FOCUS 量程 ──")
        print("  设→读回: " + "  ".join(f"{a}→{b:g}" for a, b in probe)
              + f"   → 上界约 {fmax}")

        if fine:
            vals = list(range(lo, hi + 1, step))
            print(f"\n── 细扫 {lo}~{hi} 步长 {step}，每值 {n_frames} 帧 ──")
        else:
            s = max(1, fmax // COARSE_POINTS)
            vals = list(range(0, fmax + 1, s))
            if vals[-1] != fmax:
                vals.append(fmax)
            print(f"\n── 粗扫 {len(vals)} 个值，每值 {n_frames} 帧 ──")
        rows = sweep(cap, det, want_paper, vals, n_frames)

        print()
        tms = [r["tm"] for r in rows]
        spread = max(tms) - min(tms)
        rel = spread / max(1e-9, float(np.mean(tms)))
        # ★ 判据不能是「中过一帧」—— 边缘那些值（比如纸面 1/8）也能算"中过"，
        #   但锁上去一抖就掉出去。要求**九成以上**的帧都中，才算稳的那一段。
        need = max(1, int(round(n_frames * 0.9)))
        good = [r for r in rows if r["s"] >= need and r["p"] >= need]
        loose = [r for r in rows if r["s"] > 0 and r["p"] > 0]
        if not loose:
            print("★ 没找到「吸盘码和纸面 4 码同时全中」的值。")
            if rel < 0.10:
                print("  最清晰块几乎不随 FOCUS 变（起伏 "
                      f"{rel * 100:.0f}%）→ 相机只是**回读**这个值、镜头没动，"
                      "这颗相机不吃 UVC 焦点控制，锁焦距这条路走不通。")
            else:
                peak = max(rows, key=lambda r: r["tm"])
                print(f"  焦点**真的在动**（起伏 {rel * 100:.0f}%），"
                      f"最清晰在 FOCUS={peak['v']}。")
                if peak["v"] in (vals[0], vals[-1]):
                    print("  但峰值压在**量程端点**上 → 镜头行程够不到吸盘码那个距离。")
                else:
                    print("  但两个焦面凑不到一起 → 看 output/focus_hit_*.jpg，"
                          "或者把吸嘴摆到别的高度再量一次。")
            return 1
        if not good:
            print(f"★ 只有「中过」的值（每值 {n_frames} 帧里到不了九成），没有稳的: "
                  f"FOCUS {[r['v'] for r in loose]}")
            print("  三个可能: 步长太大跨过了那段窗口 / 每值帧数太少 / 这个高度上"
                  "两个焦面本来就凑不到一起。先细扫一遍（步长 5~10、每值 ≥12 帧）。")
            return 1

        lo_v, hi_v = good[0]["v"], good[-1]["v"]
        pick = good[len(good) // 2]["v"]
        print(f"★ 稳的区间（每值 {n_frames} 帧里吸盘码和纸面 4 码都 ≥{need} 帧）: "
              f"FOCUS {lo_v} ~ {hi_v}（{len(good)} 个值）")
        print(f"★★ 建议锁在中间值: FOCUS={pick}")
        print(f"   改 src/qr_vision.py 里的 FOCUS_LOCK = {pick}（现在那行有实测表的注释）")
        print(f"   临时试: ③ python3 tools/aim_suction.py --color red --focus {pick}")
        print(f"   存下来的命中帧: output/focus_hit_*.jpg")
    finally:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        print(f"\n已还原自动对焦: AUTOFOCUS={cap.get(cv2.CAP_PROP_AUTOFOCUS):g}")
        cap.release()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[已中止]")
        sys.exit(130)
