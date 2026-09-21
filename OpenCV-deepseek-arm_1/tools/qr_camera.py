#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_camera.py —— USB 摄像头自动发现 + 二维码捕获/解码

为什么要有这个文件：
  之前测试时写死了 /dev/video2，但换摄像头、换 USB 口之后索引会变。
  这里改成「按名称自动找摄像头」，并自动跳过笔记本内置摄像头和 metadata 节点。

实测通过：一声一视 X6L (eba4:1303)、DCX-5MAF-V1 5MP (0bda:5842)

用法:
    python3 tools/qr_camera.py --list                    # 列出所有摄像头
    python3 tools/qr_camera.py --probe                   # 打印能力 + 实测帧率
    python3 tools/qr_camera.py                            # 实时检测（只看有没有码）
    python3 tools/qr_camera.py --decode                   # 实时检测 + 解码内容
    python3 tools/qr_camera.py --width 2592 --height 1944 # 指定分辨率
    python3 tools/qr_camera.py --index 2                  # 手动指定节点
    python3 tools/qr_camera.py --snapshot a.jpg           # 抓一帧存图后退出
    python3 tools/qr_camera.py --seconds 30               # 只跑 30 秒
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ★ 摄像头发现（list_cameras / find_camera）统一走 src/qr_vision.py ——
#   那是全项目唯一一处实现。这里以前抄过一份，两边的 NAME_PREFER 很快就走样
#   （这边 "dcx-5maf"，qr_vision 是 "dcx"/"x6l"），X6L 摄像头优先挑不中。
#   别再把发现逻辑抄回来；本文件只保留 qr_vision 里没有的多尺度解码那部分。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qr_vision import find_camera, list_cameras   # noqa: E402


# ─────────────────────────── 打开摄像头 ───────────────────────────
def open_camera(index: int, width: int = 0, height: int = 0,
                fourcc: str = "MJPG", warmup: int = 20,
                retries: int = 3) -> cv2.VideoCapture:
    """
    打开摄像头并预热。反复失败会重试（廉价 UVC 相机有时要等一会儿才就绪）。
    """
    last_err = None
    for attempt in range(retries):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # 预热：丢掉前面几帧（自动曝光/白平衡/对焦需要时间收敛）
            ok_any = False
            for _ in range(warmup):
                ok, f = cap.read()
                if ok and f is not None:
                    ok_any = True
            if ok_any:
                return cap
            last_err = "预热期间一帧都没读到"
        else:
            last_err = "打不开设备"
        cap.release()
        if attempt < retries - 1:
            time.sleep(1.5)
    raise RuntimeError(f"打不开 /dev/video{index}: {last_err}")


# ─────────────────────────── 二维码识别 ───────────────────────────
WORK_LONG = 1400          # 小画面时上采样到的基准长边（保证二维码够大）
REL_SCALES = (0.5, 0.7, 1.0)   # 相对原帧的缩放比例
"""
尺度组实测对比（2592x1944 合成图，含模糊+噪声，每格 = 4 个码解出几个）：
    尺度组                    耗时     300px  230px  180px  140px  100px
    (0.35,0.5,0.7,1.0)        261ms    4/4    4/4    4/4    3/4    0/4
    (0.4,0.6,0.8,1.0)         271ms    4/4    4/4    4/4    3/4    0/4
    (0.5,0.7,1.0)             244ms    4/4    4/4    4/4    3/4    0/4   ← 选它
    加 1.4 倍                  457ms    4/4    4/4    4/4    3/4    0/4   （没用，只是更慢）
结论：多加尺度并不能救更小的码，反而白白变慢。
      QR 在画面里 ≥180px 才能稳定解出 → 拍摄时让标定纸尽量铺满画面。
"""
"""
多尺度扫描 —— 这不是可选项，是必须的。
实测：cv2.QRCodeDetector 对同一画面会随缩放比例在 0/4 和 4/4 之间乱跳，
      单尺度完全不可靠；固定跑几个比例再合并结果才稳。

但尺度不能写死：2592x1944 的帧如果按 2.0 倍上采样会变成 5184x3888，
单帧检测要 775ms。所以改成「自适应」——把长边归一化到 WORK_LONG 附近，
再按 REL_SCALES 铺开，这样不管 640x480 还是 5MP，耗时都差不多。
"""


def auto_scales(frame: np.ndarray,
                work_long: int = WORK_LONG,
                rel=REL_SCALES) -> tuple[float, ...]:
    """
    根据帧尺寸算出该用哪几个缩放比例。
      · 大画面（长边 ≥ work_long）：直接降采样 + 原图，不做无用的上采样（省时间）
      · 小画面（如 640x480）：整体上采样，否则二维码太小根本找不到
    """
    long_side = max(frame.shape[:2])
    if long_side >= work_long:
        return tuple(rel)
    k = min(2.0, work_long / long_side)
    out = {round(min(2.5, r * k), 4) for r in rel}
    return tuple(sorted(out))


def _scaled(frame: np.ndarray, s: float) -> np.ndarray:
    if s == 1.0:
        return frame
    fn = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(frame, None, fx=s, fy=s, interpolation=fn)


def _merge_boxes(boxes: list[np.ndarray], tol: float = 0.6) -> list[np.ndarray]:
    """
    合并中心距离过近的候选框。
    同一个二维码在不同缩放尺度下被找到时，中心会差几个像素，
    光靠「取整后当 key」去重会漏掉一部分，所以改成按距离合并。
    """
    out: list[np.ndarray] = []
    for p in boxes:
        c = p.mean(axis=0)
        size = float(np.linalg.norm(p[1] - p[0])) or 1.0
        for q in out:
            if float(np.linalg.norm(q.mean(axis=0) - c)) < size * tol:
                break
        else:
            out.append(p)
    return out


def qr_locate(frame: np.ndarray, det=None, scales=None) -> list[np.ndarray]:
    """
    只「定位」二维码，不解码内容。速度快、对模糊更宽容。
    返回: [ 4x2 角点数组, ... ]，坐标为原始帧尺度
    """
    det = det or cv2.QRCodeDetector()
    scales = scales or auto_scales(frame)
    raw: list[np.ndarray] = []
    for s in scales:
        ok, pts = det.detectMulti(_scaled(frame, s))
        if not ok or pts is None:
            continue
        for p in pts:
            raw.append(np.asarray(p, dtype=np.float64).reshape(-1, 2) / s)
    return _merge_boxes(raw)


def _order_quad(quad) -> np.ndarray:
    """把 4 个角点排成「左上, 右上, 右下, 左下」（旋转 <45° 时可靠）。"""
    p = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    s = p.sum(axis=1)                 # x+y : 最小=左上, 最大=右下
    d = p[:, 1] - p[:, 0]             # y-x : 最小=右上, 最大=左下
    return np.array([p[np.argmin(s)], p[np.argmin(d)],
                     p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)


def _warp_quad(frame: np.ndarray, quad: np.ndarray,
               out_size: int = 360, pad: float = 0.18):
    """
    把斜着的二维码区域裁出来并「摆正」（单应变换 → 正方形）。
    pad: 向外多扩的比例，保证白边完整（白边不够会直接解码失败）。
    """
    c = quad.mean(axis=0)
    q = c + (quad - c) * (1.0 + pad)
    dst = np.array([[0, 0], [out_size - 1, 0],
                    [out_size - 1, out_size - 1], [0, out_size - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    return cv2.warpPerspective(frame, M, (out_size, out_size),
                               flags=cv2.INTER_LINEAR), M


def _decode_patch(patch: np.ndarray, det, wdet) -> str | None:
    """对一个已摆正的小图尝试解码，两个解码器都试。"""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    if wdet:
        try:
            texts, _ = wdet.detectAndDecode(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            for t in texts:
                if t:
                    return t
        except cv2.error:
            pass
    try:
        txt, _, _ = det.detectAndDecode(gray)
        if txt:
            return txt
    except cv2.error:
        pass
    return None


def qr_decode(frame: np.ndarray, det=None, wdet=None,
              scales=None, patch: int = 360) -> dict[str, np.ndarray]:
    """
    「检测 + 解码」，返回 {内容: 4x2 原帧角点}。

    策略：先用便宜的 qr_locate() 在多个尺度上定位，得到候选框；
          再把每个候选框裁出来、摆正、放大到 patch×patch 后解码。
    为什么这么做？实测直接在 2592x1944 全图上跑 5 尺度 × 2 解码器只要 0.7fps；
    而定位只需 ~180ms，单个小图解码只需 ~10ms。
    """
    det = det or cv2.QRCodeDetector()
    if wdet is None:
        try:
            wdet = cv2.wechat_qrcode_WeChatQRCode()
        except Exception:
            wdet = False

    found: dict[str, np.ndarray] = {}
    for quad in qr_locate(frame, det, scales):
        p = _order_quad(quad)
        txt = _decode_patch(_warp_quad(frame, p, patch)[0], det, wdet)
        if txt and txt not in found:
            found[txt] = p.astype(np.float64)

    # 候选框没解出来的（比如白边被截断），用全图解码兜底
    if not found:
        for s in (0.5, 1.0):
            r = det.detectAndDecodeMulti(_scaled(frame, s))
            if r and r[0]:
                for txt, q in zip(r[1], r[2]):
                    if txt and txt not in found:
                        found[txt] = np.asarray(q, np.float64).reshape(-1, 2) / s
    return found


# ─────────────────────────── 各种模式 ───────────────────────────
def cmd_list() -> None:
    print(f"{'节点':<12}{'索引':>4}  {'metadata':<9} 名称")
    print("-" * 72)
    for c in list_cameras():
        mark = "是" if c["metadata"] else ""
        print(f"{c['path']:<12}{c['index']:>4}  {mark:<9} {c['name']}")
    pick = find_camera()
    print("\n自动选择:", f"{pick['path']}  ({pick['name']})" if pick else "没找到")


def cmd_probe(cam: dict, w: int, h: int) -> None:
    print(f"探测 {cam['path']}  ({cam['name']})")
    cap = open_camera(cam["index"], w, h)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    print(f"  实际分辨率 : {real_w} x {real_h}")
    print(f"  FOURCC     : {fourcc}")
    print(f"  亮度/曝光   : {cap.get(cv2.CAP_PROP_BRIGHTNESS):.0f} / {cap.get(cv2.CAP_PROP_EXPOSURE):.0f}")

    n, bad, t0 = 0, 0, time.time()
    frame = None
    while time.time() - t0 < 3.0:
        ok, f = cap.read()
        if ok and f is not None:
            n += 1
            frame = f
        else:
            bad += 1
    dt = time.time() - t0
    print(f"  实测帧率   : {n/dt:.1f} fps  (成功 {n}, 失败 {bad})")

    if frame is not None:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"  画面亮度   : {g.mean():.1f}   清晰度(拉普拉斯方差): {cv2.Laplacian(g, cv2.CV_64F).var():.1f}")
        sc = auto_scales(frame)
        print(f"  扫描尺度   : {sc}")
        t0 = time.time()
        n_qr = len(qr_locate(frame))
        print(f"  定位耗时   : {(time.time()-t0)*1000:.0f} ms/帧   (定位到 {n_qr} 个)")
    cap.release()


def cmd_snapshot(cam: dict, path: str, w: int, h: int) -> None:
    """
    抓一帧存图。存两张：
      <path>           原始画面
      <path>.annotated.jpg  标出检测到的二维码位置 + 四等分网格
    """
    cap = open_camera(cam["index"], w, h)
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        sys.exit("读帧失败")

    cv2.imwrite(path, frame)
    H, W = frame.shape[:2]

    vis = frame.copy()
    boxes = qr_locate(frame)
    for i, p in enumerate(boxes):
        p = p.astype(int)
        cv2.polylines(vis, [p], True, (0, 0, 255), 6)
        c = p.mean(axis=0).astype(int)
        cv2.putText(vis, f"#{i+1}", (c[0] - 50, c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), 5)
    cv2.line(vis, (W // 2, 0), (W // 2, H), (255, 0, 0), 3)
    cv2.line(vis, (0, H // 2), (W, H // 2), (255, 0, 0), 3)
    for j in range(1, 4):
        cv2.line(vis, (W * j // 4, 0), (W * j // 4, H), (0, 255, 0), 1)
        cv2.line(vis, (0, H * j // 4), (W, H * j // 4), (0, 255, 0), 1)

    ann = str(Path(path).with_suffix("")) + ".annotated.jpg"
    cv2.imwrite(ann, vis)

    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"已保存 {path}   ({W}x{H}, 亮度 {g.mean():.1f}, "
          f"清晰度 {cv2.Laplacian(g, cv2.CV_64F).var():.1f})")
    print(f"已保存 {ann}   (红框=检测到的二维码, 绿线=四等分网格, 蓝线=画面中心)")
    print(f"检测到 {len(boxes)} 个二维码")
    for i, p in enumerate(boxes):
        print(f"   #{i+1} 中心({p[:,0].mean():6.0f},{p[:,1].mean():6.0f}) "
              f"边长≈{np.linalg.norm(p[1]-p[0]):.0f}px")


def cmd_live(cam: dict, w: int, h: int, decode: bool, seconds: float,
             save_hit: str | None) -> None:
    cap = open_camera(cam["index"], w, h)
    real_w, real_h = int(cap.get(3)), int(cap.get(4))
    mode = "检测 + 解码" if decode else "只检测（定位）"
    print(f"摄像头 : {cam['path']}  ({cam['name']})")
    print(f"分辨率 : {real_w} x {real_h}    模式: {mode}    时长: {seconds:g}s  (Ctrl-C 退出)")
    print(f"扫描尺度: {auto_scales(np.zeros((real_h, real_w, 3), np.uint8))}\n")

    det = cv2.QRCodeDetector()
    wdet = None
    if decode:
        try:
            wdet = cv2.wechat_qrcode_WeChatQRCode()
        except Exception:
            pass

    t_start = time.time()
    frames = hits = 0
    seen_all: dict[str, int] = {}
    last_report = 0
    hit_frame = None

    try:
        while time.time() - t_start < seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames += 1

            if decode:
                res = qr_decode(frame, det, wdet)
                n_qr, labels = len(res), sorted(res)
                for k in res:
                    seen_all[k] = seen_all.get(k, 0) + 1
            else:
                pts = qr_locate(frame, det)
                n_qr = len(pts)
                labels = [tuple(p.mean(0).round(0).astype(int)) for p in pts]

            if n_qr:
                hits += 1
                if hit_frame is None:
                    hit_frame = frame.copy()

            el = time.time() - t_start
            if el - last_report >= 1.0:
                last_report = int(el)
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flag = "看到 ✅" if n_qr else "—"
                print(f"  {int(el):>3}s  帧{frames:>5}  {frames/max(el,1e-3):5.1f}fps"
                      f"  亮度{g.mean():5.1f}  QR×{n_qr}  {flag}  {labels}")
    except KeyboardInterrupt:
        print("\n(手动中断)")
    finally:
        cap.release()

    el = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"总帧数 {frames}，有二维码的帧 {hits}  ({hits/max(frames,1)*100:.0f}%)   平均 {frames/max(el,1e-3):.1f} fps")
    if decode and seen_all:
        print("识别到的内容:")
        for k, v in sorted(seen_all.items(), key=lambda kv: -kv[1]):
            print(f"   {k!r:<28} 出现 {v} 帧")
    elif not decode and hits:
        print("(只检测模式：没解内容。需要内容请加 --decode)")
    if hit_frame is not None and save_hit:
        cv2.imwrite(save_hit, hit_frame)
        print(f"命中帧已保存 -> {save_hit}")
    print("=" * 72)


# ─────────────────────────────── main ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="USB 摄像头自动发现 + 二维码捕获/解码")
    ap.add_argument("--list", action="store_true", help="列出所有摄像头")
    ap.add_argument("--probe", action="store_true", help="打印能力并实测帧率")
    ap.add_argument("--index", type=int, help="手动指定 /dev/videoN")
    ap.add_argument("--name", type=str, help="按名字片段挑摄像头")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--decode", action="store_true", help="顺便解码内容（慢一些）")
    ap.add_argument("--seconds", type=float, default=20.0, help="运行时长，0=一直跑")
    ap.add_argument("--snapshot", type=str, help="抓一帧存图后退出")
    ap.add_argument("--save-hit", type=str, help="把第一帧命中的画面存下来")
    args = ap.parse_args()

    if args.list:
        cmd_list()
        return

    if args.index is not None:
        # 名字从 list_cameras() 查，别再自己读 sysfs —— 那正是要收掉的重复
        cam = next((c for c in list_cameras() if c["index"] == args.index), None)
        if cam is None:
            cam = {"index": args.index, "path": f"/dev/video{args.index}",
                   "name": "(手动指定)"}
    else:
        cam = find_camera(args.name)
        if not cam:
            print("没找到可用的摄像头。当前系统里的节点：", file=sys.stderr)
            cmd_list()
            sys.exit(1)

    if args.probe:
        cmd_probe(cam, args.width, args.height)
    elif args.snapshot:
        cmd_snapshot(cam, args.snapshot, args.width, args.height)
    else:
        cmd_live(cam, args.width, args.height, args.decode,
                 args.seconds if args.seconds > 0 else 1e9, args.save_hit)


if __name__ == "__main__":
    main()
