#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_hand_eye_calib.py —— 手眼标定（对应《方案.md》第三阶段 3.1）

干的事: 摄像头看一眼四个二维码 → 算出「像素坐标 → 机械臂坐标」的透视变换矩阵。
之后世界状态里的方块像素坐标就能换算成机械臂能走的物理坐标。

──────────────────────── 用法（在**项目根目录**下执行）────────────────────────
1) 先自检（不需要机械臂，验证整套数学是对的）:
       python3 src/step3_hand_eye_calib.py --selftest

2) 对刀测坐标（《方案.md》第二阶段，只做一次）:
   - 把标定纸固定在硬纸板上，机械臂底座卡进缺口
   - 推荐走 step2_teach_coords.py（它当场落盘 output/robot_points.json，还做距离校验）:
         python3 src/step2_teach_coords.py --reference corner_tr
     ★ 对完四个码后**按 q 结束** —— 只有按 q 才落盘，Ctrl-C 不存。
   - 若手填下面的 DOBOT_COORDS，务必先确认瞄的是**哪个角**:
     默认参考点是 corner_tr =「数据区右上角」（qr_vision.DEFAULT_REFERENCE），
     而本文件下面那几行旧注释写的是「左上角」—— 两处说的**不是同一个角**。
     ⚠ 别混: 瞄哪个角，就得让 --reference / robot_points.json 的 reference
       说同一个角，否则标定出来的矩阵会整体偏一个数据区宽（26mm）。
   - 把读到的 4 组 (X, Y) 填进下面的 DOBOT_COORDS
     ★★ 只记 X 和 Y，**不要 Z**。标定是平面(2D)透视变换，Z 用不上；
        多写一个数程序会直接报错拦住你。
   - 想改成用「码的中心」当参考点更省事对准，就把 --reference 换成 center，
     并把 DOBOT_COORDS 改成你量的中心坐标（两边必须一致！）

3) 正式标定（每次开机跑一次）:
       python3 src/step3_hand_eye_calib.py
   成功后会把矩阵写进 output/hand_eye_matrix.json。
   解不出四个码时**直接报错退出**，绝不带着错的矩阵往下走 —— 那会让机械臂
   走到错误坐标，可能撞坏东西。

4) 在别的程序里用（★ 这个模块**可以被 import**，所以文件名的 step3_ 前缀
   是合法标识符 —— Python 不允许 `import 3_xxx` 这种以数字开头的方式）:
       import sys; sys.path.insert(0, "src")
       from step3_hand_eye_calib import load_matrix, pixel_to_robot
       M = load_matrix()
       x, y = pixel_to_robot(M, 640, 360)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from qr_vision import (DATA_MODULES, DEFAULT_REFERENCE, GOOD_PX_PER_MODULE,
                       PaperLayout, REFERENCE_LABEL, REFERENCE_MODES, SCALE_TOL,
                       acquire_codes, affine_fit, affine_winding_bad, contrast_mode,
                       describe_cameras, detect, load_paper_layout, open_camera,
                       px_per_module, reference_point, robot_scale_note,
                       shape_diagnosis, similarity_fit, touches_edge, wait_for_focus)

from paths import (MATRIX_JSON, PAPER_JSON, PAPER_PNG,  # noqa: E402
                   ROBOT_POINTS_JSON)

# ═══════════════════════════════════════════════════════════════════
#  ★★★ 手动测绘结果填这里（《方案.md》第二阶段，只做一次）★★★
#
#  用 DobotStudio 把吸盘中心依次对准 P1~P4 的**同一个角**，记下 (X, Y)。
#  单位 mm。没测之前留 None —— 程序会明确报错，不会瞎跑。
#
#  ⚠ 这个角必须和 --reference 说的一致（默认 corner_tr = 数据区右上角）。
#    本文件早期版本写的是「数据区左上角」，那是另一个角 —— 差一个数据区宽
#    26mm，混用会让整个标定矩阵偏掉。以 --reference 说的为准。
# ═══════════════════════════════════════════════════════════════════
DOBOT_COORDS: dict[str, list] = {
    "P1": [None, None],      # 例如 [150.0, 100.0]
    "P2": [None, None],
    "P3": [None, None],
    "P4": [None, None],
}

COORD_TOL_MM = 3.0     # 机械臂坐标之间的互相距离 vs 纸上距离，允许差多少 mm
# ★ 这个默认值**故意不动** —— --selftest 用的就是它，3mm 才能抓住
#   「某一点偏 20mm」那种真错误（那套合成几何里 20mm 显成 9.26mm）。
#   真机跑这台 Magician 时用 --tol 单独放宽，见 --tol 的说明。


# ─────────────────────────── 坐标校验 ───────────────────────────
def check_coords_filled(coords: dict, codes: list[str]) -> None:
    """
    坐标没填/填错就抛异常 —— 绝不用半个坐标去算矩阵。

    这里把三种常见手误分别拦下来，各给一句能看懂的话（不拦的话，
    getPerspectiveTransform 只会报「需要 4 组对应点，收到 4 和 6」之类，
    完全看不出问题在哪）:
      · 没填（None）        → 提醒去对刀
      · 写成标量 150.0      → 忘了方括号
      · 抄了 Z（3 个数）    → 说明标定是平面变换，用不到 Z
      · 非数字（"150mm"）   → 提醒别写单位
    """
    missing = [c for c in codes if c not in coords]
    if missing:
        raise ValueError(f"DOBOT_COORDS 缺少这些码: {missing}")

    blank, bad_shape, bad_num = [], [], []
    for c in codes:
        v = coords[c]
        # 整条留空: None，或者 [None, None]
        if v is None or (hasattr(v, "__len__") and tuple(v) == (None, None)):
            blank.append(c)
            continue
        if not hasattr(v, "__len__"):        # 写成了标量，多半是忘了方括号
            bad_shape.append((c, v))
            continue
        if len(v) != 2:
            bad_shape.append((c, v))
            continue
        if v[0] is None or v[1] is None:     # [None, 100] 这种半截
            blank.append(c)
            continue
        try:
            float(v[0]), float(v[1])
        except (TypeError, ValueError):
            bad_num.append((c, v))

    if blank:
        raise ValueError(
            f"DOBOT_COORDS 里这些还没填: {blank}\n"
            f"  请先用 DobotStudio 对刀测出 XY 再填（见本文件顶部说明）。")
    if bad_shape:
        detail = "; ".join(f"{c}={v!r}" for c, v in bad_shape)
        raise ValueError(
            f"DOBOT_COORDS 每个点必须写成 [X, Y] 两个数，但: {detail}\n"
            f"  ★ 若是写了 3 个数: Z 不参与标定 —— 像素→机械臂是**平面**透视"
            f"变换，只需要 X Y。\n"
            f"    桌面高度 TABLE_Z、方块高度 CUBE_H、安全高度 SAFE_Z 是执行阶段"
            f"另设的参数，和标定无关，别抄进这里。\n"
            f"  ★ 若是写成标量 150.0: 补上方括号和第二个数 → [150.0, 100.0]。")
    if bad_num:
        detail = "; ".join(f"{c}={v!r}" for c, v in bad_num)
        raise ValueError(
            f"DOBOT_COORDS 里有不是数字的值: {detail}\n"
            f"  形如 [150.0, 100.0]，别写单位（150mm）。")


def check_coord_consistency(layout: PaperLayout, coords: dict,
                            tol_mm: float = COORD_TOL_MM) -> float:
    """
    校验「机械臂坐标」和「纸上已知布局」是否自洽。

    原理: 纸上 4 个参考点的间距(mm) 和机械臂坐标的间距(mm) 描述的是同一批
    物理点。填错数字 / 把 P2 P3 写反 / 量的时候看错点，都会让距离对不上。

    ★★ 但两边的**单位不一样**，不能直接比:
      机械臂报的 XY 是按它自己的运动学模型算的，实测与真实毫米不成 1:1
      （一台 Magician 实测 X≈0.81、Y≈0.94，见 qr_vision.robot_scale_note）。
      直接比的话六条距离会一起超差几十毫米 —— 把「模型差异」误报成
      「抄错坐标」，而这个误报会**卡住整条标定流程**。

      ★ 而且两个轴的比例可以**不一样**（各向异性），所以基准不能用单一的
        「纸距 × 缩放」，要用**仿射拟合**预测出来的距离：仿射有 6 个自由度
        （两轴各自缩放 + 剪切 + 旋转 + 平移），足以吃掉机械臂整个线性模型
        差异。只有「某一个点真的没对准」才会剩下仿射吃不掉的残差。
        用相似拟合（只有 1 个缩放）判的话，一台各向异性的好机械臂会被
        永远判成「教歪了」，跟之前「缩放卡死标定」是同一类误报。

    返回最大偏差(mm)，超过 tol_mm 抛异常。
    """
    codes = layout.codes
    ref = layout.reference_mm

    # ★ 残差大时缩放不可信 —— 整体对错位（比如 P2/P3 写反）会让最小二乘吐出
    #   一个毫无意义的缩放（实测能到 0.004），拿它归一化会把真正的错误抹掉。
    #   所以只在拟合自洽时才归一化，否则退回原始值比，让错误原样露出来。
    aff = affine_fit(ref, coords, codes)
    fit = similarity_fit(ref, coords, codes)
    n = fit[3] if fit else 0
    aff_worst = aff[1] if aff else 0.0
    winding_bad = aff is not None and affine_winding_bad(aff)
    coherent = aff is not None and aff_worst <= tol_mm and not winding_bad
    # ★ 相似拟合只取它的点数 n（下面那句说明要用），以及点不足 3 个时的退化比例。
    #   它的**缩放值在本项目里没有意义**（纸面 y 向下 → 数据带镜像），
    #   所以只有 aff 定不出来时才拿它顶上，见下面的 note_scale。
    scale = fit[0] if fit else 1.0

    if aff is not None:
        L, t = aff[0][:, :2], aff[0][:, 2]
        expect = {c: L @ np.array(ref[c], float) + t for c in codes}
    else:                                   # 点太少，退回单缩放
        expect = {c: np.array(ref[c], float) * scale for c in codes}

    worst, worst_pair = 0.0, None
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            d_paper = float(np.linalg.norm(
                np.array(ref[a], float) - np.array(ref[b], float)))
            d_robot = float(np.linalg.norm(
                np.array(coords[a], float) - np.array(coords[b], float)))
            d_pred = float(np.linalg.norm(expect[a] - expect[b]))
            diff = abs(d_robot - d_pred)
            if diff > worst:
                worst, worst_pair = diff, (a, b, d_paper, d_pred, d_robot)

    axes = None
    if aff is not None:
        L = aff[0][:, :2]
        axes = (float(np.linalg.norm(L[:, 0])), float(np.linalg.norm(L[:, 1])))
    # ★ "等比缩放"那个数从**仿射**的两个轴比例来，不用相似拟合的:
    #   纸上坐标 y 向下 → 数据里带一个镜像，相似拟合的 |c| 与真实比例无关
    #   （实测一套两轴都 1:1 的自洽坐标，它能报 0.453 → 凭空说"机器不成 1:1"）。
    #   `scale` 只留给点不足 3 个、仿射定不出来时的退化路径用。
    note_scale = float(np.sqrt(axes[0] * axes[1])) if axes else scale
    print(f"  坐标自洽校验: 实测距离与「已教出的形状」最大偏差 {worst:.2f} mm "
          f"(容差 {tol_mm} mm)")
    # ★ 比例说明**只在四点自洽时**打: 不对齐/镜像的时候，拟合出来的"各轴比例"
    #   是垃圾（实测 P2↔P3 标反能给出 0.614 / 1.630），打出来只会把人带偏 ——
    #   那种情况该由下面的报错说清"标签写反了"。
    if coherent:
        note = robot_scale_note(note_scale, n, short=True, axes=axes)
        if note:
            print("  " + note.replace("\n", "\n  "))
    if winding_bad and len(coords) == 4:
        raise ValueError(
            "机械臂坐标和纸上布局**绕向反了**（左转变右转）: 仿射行列式 "
            f"{float(np.linalg.det(aff[0][:, :2])):+.3f} > 0，"
            "而这张纸配这台机器**正确时必为负**"
            "（纸面 y 向下 + 机械臂右手系 —— 推导见 qr_vision.affine_winding_bad）。\n"
            "  ★ 多半是**相邻两个码的标签写反了**（最常见 P2 和 P3、其次 P1 和 P4）:"
            "这两对互换后四点形状仍然自洽（残差可能只有零点几毫米），但整张纸被镜像，"
            "标定会把物体放到对角线的另一侧。核对每个码到底对着纸上哪个角。")
    if worst > tol_mm:
        a, b, dp, dpr, dr = worst_pair
        msg = (
            f"机械臂坐标和纸上布局对不上！最差的是 {a}-{b}: "
            f"纸上 {dp:.1f}mm，机械臂实测 {dr:.1f}mm，"
            f"比「已教出的形状」该有的 {dpr:.1f}mm 差 {worst:.1f}mm。\n"
            f"  常见原因: ①某个坐标抄错/漏了小数点 ②P2 和 P3 写反了 "
            f"③对刀时看错了角(用了中心或白边角) ④某一个点根本没对准。")
        if aff is not None and len(coords) == 4 and aff_worst > tol_mm:
            msg += (f"\n  ★ 让掉「各轴分别缩放 + 剪切」之后**仍然**差 "
                    f"{aff_worst:.1f}mm → 是**某一个点**教歪了"
                    f"（机器比例的事，不是这里的问题）。")
            for line in shape_diagnosis(layout, coords, codes):
                msg += "\n  " + line
        raise ValueError(msg)
    return worst


# ─────────────────────────── 标定数学 ───────────────────────────
def compute_homography(pixel_pts: np.ndarray, robot_pts: np.ndarray) -> np.ndarray:
    """
    求 像素 → 机械臂 的透视变换矩阵。返回 3x3。
    四点求单应矩阵是精确解（无残差），所以不能靠重投影误差自证对错，
    要靠 check_coord_consistency 和 --selftest 来把关。
    """
    src = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(robot_pts, dtype=np.float64).reshape(-1, 2)
    if len(src) != 4 or len(dst) != 4:
        raise ValueError(f"需要 4 组对应点，收到 {len(src)} 和 {len(dst)}")
    return cv2.getPerspectiveTransform(src.astype(np.float32),
                                       dst.astype(np.float32))


def pixel_to_robot(M: np.ndarray, px, py) -> tuple[float, float]:
    """把图像上一个像素点换算成机械臂 (X, Y) mm。"""
    p = np.array([px, py, 1.0], dtype=np.float64)
    q = np.asarray(M, dtype=np.float64) @ p
    if abs(q[2]) < 1e-12:
        raise ValueError("透视变换退化（分母为 0）")
    return float(q[0] / q[2]), float(q[1] / q[2])


def save_matrix(M: np.ndarray, layout: PaperLayout, coords: dict,
                path: Path = MATRIX_JSON, ppm: float | None = None) -> None:
    """
    落盘矩阵。★ 带上一份就备份 —— 这个文件以前**没有** .bak，
    一次误跑（比如机位没架好就重标）就把能用的矩阵覆盖掉了，回不去。

    ppm 是标定那一帧量到的「每毫米多少像素」。它只被 setup_camera.py 用来做
    机位漂移对比（标定帧 vs 抓方块帧差太多 = 摄像头在两步之间被碰过）。
    ★ 写成 JSON 的一个新键，老文件没有这个键也照读（load_matrix 只取
      matrix_row_major），所以是向后兼容的。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        try:
            p.with_suffix(".json.bak").write_bytes(p.read_bytes())
        except OSError:
            pass          # 备份失败不该拦住落盘本身
    p.write_text(json.dumps({
        "note": "像素→机械臂坐标 的 3x3 透视变换矩阵；由 step3_hand_eye_calib.py 生成",
        "reference": layout.reference,
        "matrix_row_major": np.asarray(M).reshape(-1).tolist(),
        "px_per_mm": None if ppm is None else float(ppm),
        "dobot_coords": {k: list(map(float, v)) for k, v in coords.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_matrix(path: Path = MATRIX_JSON) -> np.ndarray:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(d["matrix_row_major"], dtype=np.float64).reshape(3, 3)


# ─────────────────────────── 从相机采集并标定 ───────────────────────────
def gather_pixel_points(layout: PaperLayout, cap, det,
                        duration: float = 3.0, quiet: bool = False):
    """
    等对焦 → 多帧累积，凑齐四个码 → 取出参考点像素坐标。
    凑不齐就抛异常（含具体缺哪个码、以及当前每模块像素数）。
    """
    print("  等自动对焦收敛…")
    frame, blur, _ = wait_for_focus(cap, det, layout.codes, quiet=quiet)
    if frame is None:
        raise RuntimeError("摄像头取不到画面")

    decoded, frames, counts = acquire_codes(cap, det, layout.codes,
                                            duration=duration, quiet=quiet)
    missing = [c for c in layout.codes if c not in decoded]
    if missing:
        # 给出可操作的诊断，而不是干巴巴一句失败
        ppm_mod = None
        if decoded:
            ppm_mod = min(px_per_module(q) for q in decoded.values())
        hint = []
        if any(touches_edge(q, frame.shape) for q in decoded.values()):
            hint.append("有码贴到画面边缘被裁切 → 把摄像头抬高/拉远")
        if ppm_mod is not None and ppm_mod < GOOD_PX_PER_MODULE:
            hint.append(f"每模块只有 {ppm_mod:.1f}px (<{GOOD_PX_PER_MODULE}) → 摄像头靠近些")
        if not decoded:
            hint.append("一个都没解出 → 先跑 tools/test_camera_qr.py 调好机位")
        if counts and max(counts) < len(layout.codes):
            hint.append(f"最好的一帧也只解出 {max(counts)} 个，说明机位本身不稳")
        raise RuntimeError(
            f"四个二维码没凑齐，缺 {missing}。标定中止（不会用错矩阵继续）。\n"
            + "".join(f"    · {h}\n" for h in hint))

    pixel_pts = np.array([reference_point(decoded[c], layout.reference)
                          for c in layout.codes], dtype=np.float64)
    return frame, pixel_pts


def evaluate_view(pixel_pts: np.ndarray) -> None:
    """
    看一眼这四个点的几何是否合理（围得出面积、四边尺度别差太多）。

    ★ 别把 layout.codes 的相邻两项当边: 它的顺序是 P1(左上) P2(右上)
      P3(左下) P4(右下) —— 按码的编号排的，不是绕一圈排的，P2→P3 是对角线。
      直接拿来算面积会得到自交"蝴蝶结"的零面积（这个坑真踩过：
      明明四个码都认出来了，却被判成"识别错乱"而中止标定）。
      所以先按绕质心的角度重排成真正的环形顺序再算。
    """
    pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    c = pts.mean(axis=0)
    ring = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]

    # 鞋带公式自己算，不用 cv2.contourArea —— 后者只吃 CV_32F/CV_32S
    # （float64 直接抛断言），而且遇到自交图形返回 0，容易误导。
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 1.0:
        raise RuntimeError(
            f"四个参考点围出的面积只有 {area:.1f}px²（≈0），不可能是一张纸。"
            f"标定中止。多半是四个码的位置认错/认重了。")

    sides = [float(np.linalg.norm(ring[(i + 1) % 4] - ring[i])) for i in range(4)]
    ratio = max(sides) / max(1e-9, min(sides))
    print(f"  视野几何: 四边形面积 {area:.0f}px², 四边长比 {ratio:.2f}")
    if ratio > 2.0:
        print(f"  ⚠️ 四边长相差 {ratio:.1f} 倍 —— 摄像头太斜了。"
              f"反正透视变换能纠正，但越斜精度越差，建议改成垂直俯拍。")


def calibrate(cap, det, layout: PaperLayout, coords: dict,
              duration: float = 3.0, quiet: bool = False,
              tol_mm: float = COORD_TOL_MM) -> tuple[np.ndarray, np.ndarray]:
    """完整走一遍: 采集 → 校验 → 求矩阵。返回 (矩阵, 像素点)。"""
    check_coords_filled(coords, layout.codes)
    check_coord_consistency(layout, coords, tol_mm=tol_mm)
    frame, pixel_pts = gather_pixel_points(layout, cap, det, duration, quiet)
    evaluate_view(pixel_pts)

    robot_pts = np.array([coords[c] for c in layout.codes], dtype=np.float64)
    M = compute_homography(pixel_pts, robot_pts)

    # 自洽检查：把参考点映射回去，应该回到原位（四点拟合必然为 0，
    # 这里只是确认矩阵没被写坏，不当作精度证据）
    errs = [np.linalg.norm(np.array(pixel_to_robot(M, *p)) - r)
            for p, r in zip(pixel_pts, robot_pts)]
    print(f"  映射自洽: 参考点回代最大偏差 {max(errs):.4f} mm (四点拟合应≈0)")
    return M, pixel_pts


# ─────────────────────────── 自检（不需要硬件） ───────────────────────────
def _warp_paper(dst_quad, frame_wh=(1920, 1080), jitter=None):
    """
    把标定纸 PNG 按给定四边形透视投影，模拟一个架好的俯拍摄像头。
    jitter 给一个 numpy Generator 时，把四个目标角点各随机抖 ±8px ——
    避免只验证「某一个特定构图」，让检验覆盖到不同的画面位置。

    ★ PNG 是 step1_gen_paper.py 按 A4 排版出的**模板**，和 calib_A4_qr.json 的实测
      布局不一定是同一张纸（自己贴的纸就是这样: 码 26mm vs 模板 40mm）。
      这没关系 —— 本函数只借它"产生一张有四个码的图"，像素位置对错无所谓；
      真值一律来自 layout，拟合和检验用的是同一套，绝对尺寸自然抵消。
    """
    img = cv2.imread(str(PAPER_PNG))
    if img is None:
        raise FileNotFoundError(f"找不到 {PAPER_PNG}，先跑 step1_gen_paper.py")
    h, w = img.shape[:2]
    dst = np.asarray(dst_quad, dtype=np.float64)
    if jitter is not None:
        dst = dst + jitter.uniform(-8.0, 8.0, size=dst.shape)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(img, M, frame_wh, borderValue=(80, 80, 80))


def _fit_and_check(dec: dict, layout: PaperLayout,
                   fit_mode: str, verify_mode: str) -> list[float]:
    """
    用 fit_mode 的点拟合 像素→纸面mm 矩阵，再用它预测 verify_mode 的点，
    返回每个码的预测误差(mm)。verify_mode 必须与 fit_mode 不同，否则恒为 0。
    """
    px = np.array([reference_point(dec[c], fit_mode) for c in layout.codes])
    gt = np.array([layout.mm_for(fit_mode)[c] for c in layout.codes], float)
    M = compute_homography(px, gt)

    truth = layout.mm_for(verify_mode)
    errs = []
    for c in layout.codes:
        pred = np.array(pixel_to_robot(M, *reference_point(dec[c], verify_mode)))
        errs.append(float(np.linalg.norm(pred - np.array(truth[c], float))))
    return errs


def selftest(reference: str = DEFAULT_REFERENCE, seed: int = 0) -> int:
    """
    不需要机械臂：用「已知答案」的合成画面验证整条标定链路。

    做法: 把标定纸按几种透视投进一个 1080p 画面 → 走**真实的检测代码**
    → 用选定的参考点拟合「像素 → 纸面 mm」的矩阵 → 再用它预测**另一种点**
    （选了角点就预测中心，选了中心就预测角点），和 json 里的已知值比。

    ★ 预测的那套点**没有参与拟合**，所以这是独立检验，不是自证:
      如果角点含义搞错了（比如把白边算进去、或者取了图像左上而不是码自己的
      左上），预测出来的另一种点就会偏好几个 mm，立刻露馅。
      （若两边用同一套点，四点拟合残差恒为 0，测试会永远通过 = 假绿灯。）
    """
    layout = load_paper_layout(PAPER_JSON, reference=reference)
    det = cv2.QRCodeDetector()
    rng = np.random.default_rng(seed)

    # 拟合用 reference，检验用「另一种点」——保证两者不同，测试才有意义
    verify = contrast_mode(reference)
    truth_mm = layout.mm_for(verify)

    print("═" * 72)
    print("  标定链路自检")
    print(f"  拟合用: {reference}    独立检验用: {verify}（未参与拟合）")
    print("═" * 72)

    # 几种构图。★ 别再加「极端斜视」: 透视被压扁的那一侧，码被缩到几像素，
    #   检测器解不出来 —— 那是分辨率的物理极限，不是标定数学的问题，
    #   只会让用例被跳过、白占一行。
    #   "纸张转12度" 这一条最有价值: 它端到端回归了「corner[0] 是码自己的
    #   左上角、与图像旋转无关」这个关键约定。若哪天换了检测器、角点改成
    #   按图像方向排序，这一条会立刻报错。
    base = [[300, 60], [1620, 60], [1620, 1030], [300, 1030]]
    t = np.radians(12.0)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot12 = ((R @ (np.array(base, float) - [960, 540]).T).T + [960, 540]).tolist()
    cases = [
        ("正对俯拍",  base),
        ("轻微斜视",  [[400, 110], [1600, 55],  [1640, 1020], [300, 930]]),
        ("纸张转12度", rot12),
        ("画面偏心",  [[520, 240], [1750, 160], [1810, 980],  [600, 1060]]),
    ]
    worst_all = 0.0
    skipped, ran = [], 0
    for name, quad in cases:
        frame = _warp_paper(quad, frame_wh=(1920, 1080), jitter=rng)
        dec, _loc = detect(frame, det)
        missing = [c for c in layout.codes if c not in dec]
        if missing:
            ppm = min(px_per_module(q) for q in dec.values()) if dec else 0.0
            print(f"  {name}: 检测缺 {missing}（最小的码只有 {ppm:.1f} px/模块），跳过")
            skipped.append(name)
            continue
        ran += 1

        errs = _fit_and_check(dec, layout, reference, verify)
        worst = max(errs)
        worst_all = max(worst_all, worst)
        flag = "✅" if worst < 1.0 else "❌"
        print(f"  {name}: 预测{verify}误差 最大 {worst:.2f} mm  "
              f"(逐个 {' '.join(f'{e:.2f}' for e in errs)})  {flag}")

    # ── 反向对照: 故意用错参考点，看这个检验有没有识别能力 ──
    # 把「数据区左上角」往外挪 4 个模块，模拟成「含白边的外框角」的错觉。
    # 真实偏移方向沿码自己的两条边，不是图像对角线（码可能转了角度）。
    print("\n  反向对照（故意把参考点挪 4 个模块，检验上面这个测试有没有识别能力）:")
    frame = _warp_paper(cases[0][1])
    dec, _loc = detect(frame, det)
    if all(c in dec for c in layout.codes):
        px_bad = []
        for c in layout.codes:
            q = np.asarray(dec[c], np.float64).reshape(4, 2)
            u, v = q[1] - q[0], q[3] - q[0]        # 码自己的两条边
            nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
            mpx = (nu + nv) / (2 * DATA_MODULES)
            px_bad.append(reference_point(dec[c], reference)
                          - 4.0 * mpx * (u / nu + v / nv))     # 往外挪 4 模块
        gt = np.array([layout.reference_mm[c] for c in layout.codes], float)
        M_bad = compute_homography(np.array(px_bad), gt)
        errs = [float(np.linalg.norm(
            np.array(pixel_to_robot(M_bad, *reference_point(dec[c], verify)))
            - np.array(truth_mm[c], float))) for c in layout.codes]
        print(f"    偏移 4 模块后，预测{verify}误差 = 最大 {max(errs):.2f} mm "
              f"→ {'测试有识别能力 ✅' if max(errs) > 3 else '测试无效 ❌'}")
    else:
        print("    检测失败，反向对照跳过")

    # ── 坐标自洽校验: 重点是**别把「机械臂不成 1:1」当成抄错坐标拦下来** ──
    # 实测这台 Magician 走 10mm 只报 8.2mm（缩放 5/6）。若按原始值比 3mm 容差，
    # 六条距离会一起超差几十毫米 → 直接 raise → **整条标定被卡死**（真发生过）。
    print("\n  坐标自洽校验（关键: 机械臂不成 1:1 时必须能放行）:")
    ref_mm = {c: [float(v[0]), float(v[1])] for c, v in layout.reference_mm.items()}
    csc_ok = True

    def _csc(coords, label, want_pass):
        nonlocal csc_ok
        try:
            check_coord_consistency(layout, coords)
            got, detail = True, "放行"
        except ValueError as e:
            got, detail = False, str(e).splitlines()[0]
        good = got == want_pass
        csc_ok = csc_ok and good
        print(f"    {'✅' if good else '❌'} {label}: "
              f"{'放行' if got else '拦下'} —— {detail}")
        return good

    # ★★ 造数据必须照**真实物理摆法**造，否则自检会假红/假绿:
    #    纸上坐标 y 向下（calib_A4_qr.json: P1 左上 y=39.5、P3 左下 y=170.5）、
    #    纸正面朝上、机械臂 XY 右手系（Z 朝上）→ 「纸 → 机械臂」的行列式**必为负**
    #    （推导见 qr_vision.affine_winding_bad）。原来这里直接写 [u·kx, v·ky]，
    #    等于假设行列式为正 —— 那是**物理上不可能出现**的坐标，判据改成
    #    "正确必为负"之后它们会被（正确地）全部拦下。
    _ang = np.deg2rad(23.0)
    _R = np.array([[np.cos(_ang), -np.sin(_ang)], [np.sin(_ang), np.cos(_ang)]])
    _off = np.array([180.0, -40.0])

    def _place(kx=1.0, ky=1.0, swap=None, shift=None):
        """按真实摆法造一套坐标: 旋转 + 各轴缩放 + 纸面 y 反向 + 平移。"""
        out = {}
        for c in layout.codes:
            p = np.array(ref_mm[swap[c]] if swap and c in swap else ref_mm[c], float)
            q = _R @ (p * np.array([kx, -ky])) + _off          # ← 那个负号就是"纸面 y 向下"
            if shift and c in shift:
                q = q + np.array(shift[c], float)
            out[c] = [float(q[0]), float(q[1])]
        return out

    _swap2 = {"P2": "P3", "P3": "P2"}
    _csc(_place(), "纸→机械臂 理想等比 1:1（只差旋转平移）", True)
    _csc(_place(5 / 6, 5 / 6), "整体缩放 5/6（这台机器的真实行为）", True)
    # ★ 各向异性: X 缩 0.81、Y 缩 0.94。这正是这台 Magician 的实测行为。
    #   相似拟合在这组输入上残差 ~11mm（会被误判成「教歪了」），仿射拟合 ~0。
    _csc(_place(0.81, 0.94), "两轴比例不同 X 0.81 / Y 0.94（真实现象，必须放行）", True)
    _csc(_place(swap=_swap2), "P2/P3 写反（真错，必须拦下）", False)
    # ★★ 这一条是把判定从"相似残差"换成"仿射残差"之后**新出现的**洞:
    #    P2↔P3 互换得到的是矩形的对角镜像，而镜像是仿射变换 ——
    #    仿射残差只有 0.29mm，一路放行。补的判据是行列式符号
    #    （这张纸配这台机器**正确必为负**，标反了才变正）。
    _mirror = _place(swap=_swap2)
    _aff_m = affine_fit(layout.reference_mm, _mirror, layout.codes)
    _mir_ok = (_aff_m is not None and abs(_aff_m[1]) <= 3.0
               and affine_winding_bad(_aff_m))
    csc_ok = csc_ok and _mir_ok
    print(f"    {'✅' if _mir_ok else '❌'} P2/P3 互换是仿射自洽的镜像"
          f"（残差 {_aff_m[1]:.2f}mm）→ 只能靠行列式 >0 判:")
    _csc(_mirror, "      于是必须拦下", False)
    # ★ 报错之外还得**别乱说话**: 镜像时拟合出的"各轴比例"是垃圾
    #   （实测 0.614 / 1.630），打出来会让人以为是机器的事。
    import contextlib as _cl
    import io as _io
    _buf = _io.StringIO()
    try:
        with _cl.redirect_stdout(_buf):
            check_coord_consistency(layout, _mirror)
    except ValueError:
        pass
    _junk = "各轴比例" in _buf.getvalue() or "两个轴比例" in _buf.getvalue()
    csc_ok = csc_ok and not _junk
    print(f"    {'✅' if not _junk else '❌'} 镜像时**不许**打「各轴比例」说明"
          f"（那时算出来的比例是垃圾）")
    _csc(_place(shift={"P4": (20.0, 0.0)}),
         "P4 没对准、偏了 20mm（真错，必须拦下）", False)

    # ★ 点名嫌疑点: P4 偏一个数据区宽度(≈21 模块)时，要指出是 P4。
    #   判据是「修正后正好落在 P4 的另一个已知角上」，不是「谁挪得最少」——
    #   实测 P3/P4 的修正量都是 26.00mm，比大小是分不出来的。
    print("\n  形状诊断（四点不自洽时点名「哪个点教歪了」）:")
    #    ★ 这一条**故意**用 ref_mm 当机械臂坐标（不套 _place）: shape_diagnosis
    #      只看"第 4 点在前 3 点撑出的仿射里的位置"，与绕向无关；用恒等映射
    #      才能让"机械臂 x 偏 26mm"正好等于"纸面 x 偏一个数据区宽"。
    _diag = {c: list(v) for c, v in ref_mm.items()}
    _diag["P4"][0] -= 26.0                       # 纸面 x 方向，一个数据区宽
    _lines = shape_diagnosis(layout, _diag, layout.codes)
    _named = any(l.strip().startswith("★") and "P4" in l for l in _lines)
    _hit = any("左上角" in l and "P4" in l for l in _lines)
    csc_ok = csc_ok and _named and _hit
    print(f"    {'✅' if _named and _hit else '❌'} P4 偏 26mm（一个数据区宽）: "
          f"{'点出了 P4 且对上是左上角' if _named and _hit else '没点出来/没说清是哪个角'}")
    for _l in _lines:
        print("        " + _l)

    # ── 机器常数（对刀时记下的各轴比例）: 读得出来、变了要吭声 ──
    print("\n  机器常数记录 / 漂移比对:")
    import contextlib
    import io
    import tempfile

    def _drift(recorded, label, want_word):
        nonlocal csc_ok
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rp.json"
            payload = {"dobot_coords": {c: list(v) for c, v in ref_mm.items()}}
            if recorded is not None:
                payload["robot_scale"] = {"scale_x": recorded[0],
                                          "scale_y": recorded[1]}
            p.write_text(json.dumps(payload), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                # ★ 要喂**物理上成立**的一套坐标（_place），否则 check_scale_drift
                #   会（正确地）判定绕向反了而直接沉默，这几条就全成了假绿。
                check_scale_drift(layout, _place(), p)
            out = buf.getvalue()
        good = want_word in out
        csc_ok = csc_ok and good
        print(f"    {'✅' if good else '❌'} {label}: "
              f"{out.strip().splitlines()[0].strip() if out.strip() else '（没说话）'}")

    _drift(None, "老文件没有 robot_scale → 安静跳过（不能报错）", "")
    _drift((1.0, 1.0), "记的是 1:1、这次也是 1:1 → 一致", "一致")
    _drift((0.6, 0.6), "记的是 0.6/0.6、这次 1:1 → 必须提醒机器动过",
           "机器常数变了")
    # ★ X 涨 Y 跌: 综合缩放看着没变（0.81+0.94 与 0.94+0.81 平均相同），
    #   只看一个数会漏掉，分轴看才发现 —— 撞过之后很典型。
    _drift((0.94, 0.81), "记的是 X 0.94 / Y 0.81（两轴对调）→ 必须提醒",
           "机器常数变了")

    print("─" * 72)
    if ran == 0:
        print("  ❌ 所有用例都没跑起来 —— 检测本身有问题，不是标定的问题。")
        return 1
    if skipped:
        print(f"  ⚠️ 有 {len(skipped)} 个用例因解不出码被跳过: {skipped}")
    if worst_all < 1.0 and csc_ok:
        print(f"  ✅ 自检通过（{ran} 个用例）：最差 {worst_all:.2f} mm。"
              f"标定数学与角点约定都正确。")
        print("     下一步只需对刀测 DOBOT_COORDS，然后跑 python3 src/step3_hand_eye_calib.py")
        return 0
    if not csc_ok:
        print("  ❌ 自检未通过：坐标自洽校验的行为不对（见上面 ❌ 那几行）—— "
              "要么会误拦正常的缩放，要么会放行真的写错。")
    print(f"  ❌ 自检未通过：最差 {worst_all:.2f} mm，别急着上机械臂。")
    return 1


def load_robot_points(path: Path = None) -> tuple[dict, str] | None:
    """
    读 step2_teach_coords.py 产出的 robot_points.json，返回 (coords, reference)；
    文件不存在或没有坐标则返回 None。

    ★ 顺带把 reference 也读出来 —— 见下面 live() 里的「参考点一致性」检查。
    """
    p = Path(path) if path else ROBOT_POINTS_JSON
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ⚠️ 读不了 {p.name}: {e}")
        return None
    coords = d.get("dobot_coords") or {}
    return (coords, d.get("reference", "")) if coords else None


def recorded_scale(path: Path = None) -> tuple[float, float] | None:
    """
    读对刀时记下的机器常数 各轴比例 (kx, ky)（step2_teach_coords.save() 写的）。
    """
    p = Path(path) if path else ROBOT_POINTS_JSON
    if not p.exists():
        return None
    try:
        rs = json.loads(p.read_text(encoding="utf-8")).get("robot_scale") or {}
    except (OSError, json.JSONDecodeError):
        return None
    kx, ky = rs.get("scale_x"), rs.get("scale_y")
    if not all(isinstance(v, (int, float)) for v in (kx, ky)):
        return None
    return float(kx), float(ky)


def check_scale_drift(layout: PaperLayout, coords: dict, path: Path = None) -> None:
    """
    把这次现算的各轴比例 和 对刀时记下的 比一比 —— 这才是记那个数的用处。

    同一台机器它应该**一成不变**（这台 Magician 实测 X≈0.81、Y≈0.94）。
    变了就说明机器被改过: 撞过、拆过、皮带松了、联轴器打滑…… 这类问题
    不会让标定报错（比例会自己抵消），但会让「报的毫米」和真实毫米的
    关系悄悄变掉，早发现比晚发现好。

    ★ 两个轴要**分别**比: 只看一个综合缩放的话，X 涨 Y 跌（撞过之后很典型）
      会被平均掉，看着没变，其实模型已经歪了。
    """
    old = recorded_scale(path)
    aff = affine_fit(layout.reference_mm, coords, layout.codes)
    if aff is None or old is None:
        return
    if affine_winding_bad(aff):
        # 绕向反了（多半 P2↔P3 标反）时残差可能很小，但这两个"轴比例"是假的 ——
        # 拿它比漂移只会瞎报「机器动过」，真正的问题由 check_coord_consistency 说。
        return
    L = aff[0][:, :2]
    new = (float(np.linalg.norm(L[:, 0])), float(np.linalg.norm(L[:, 1])))
    if max(abs(new[0] - old[0]), abs(new[1] - old[1])) <= SCALE_TOL:
        print(f"  机器常数比对: 各轴比例 X {new[0]:.4f} / Y {new[1]:.4f}，"
              f"与上次记录的 X {old[0]:.4f} / Y {old[1]:.4f} 一致 ✅")
        return
    print(f"\n⚠️ 机器常数变了: 这次算出 X {new[0]:.4f} / Y {new[1]:.4f}，"
          f"上次记录的是 X {old[0]:.4f} / Y {old[1]:.4f}"
          f"（差 X {abs(new[0] - old[0]) * 100:.1f}%、"
          f"Y {abs(new[1] - old[1]) * 100:.1f}%）")
    print("   标定本身不受影响（比例会自己抵消），但机器很可能动过:")
    print("   撞过/拆装过/皮带松/联轴器打滑 —— 建议回零后重查一遍再继续。")
    print("   若确实是换了纸重新量的，对着新数把它改掉即可。")


def _coords_filled(coords: dict, codes: list[str]) -> bool:
    try:
        check_coords_filled(coords, codes)
        return True
    except ValueError:
        return False


# ─────────────────────────── 现场标定 ───────────────────────────
class StillImage:
    """
    把一张静态图片伪装成 cv2.VideoCapture，好让 calibrate() 那套原样复用。

    ★ 公开（不再是 _StillImage）给 setup_camera.py 的 --image / --selftest 用。

    ★ 为什么是"伪装"而不是另写一条「从文件标定」的支路: 采集、对焦等待、多帧累积、
      解码、取参考点、求单应 —— 这一串是同一个算法。复制一份出来，两条路迟早走样，
      而走样的方式是**悄悄算出一个偏掉的矩阵**，最难发现。这里只把"帧从哪来"换掉。
    ★ 静止图片没有对焦收敛可言，但行为要和真摄像头一致:
      解得出 4 个码就立刻返回；解不出就等超时、然后照常报「没凑齐，缺哪个」，
      不会静默拿残缺数据继续算。
    ★ 只实现 calibrate 真正会调的方法。故意不实现 get/set ——
      万一哪天有代码在这个路径上调用它们，宁可当场 AttributeError，
      也不要给一个"看起来能用、其实返回错值"的实现。
    """

    def __init__(self, img):
        self._img = img

    def isOpened(self) -> bool:
        return True

    def read(self):
        return True, self._img.copy()

    def release(self) -> None:
        pass


def resolve_teach_coords(layout: PaperLayout, coords_path=None,
                         title: str = "手眼标定") -> tuple[dict, Path] | None:
    """
    定出这次标定用哪套「对刀坐标」，并把该查的前置检查全查完。
    全部通过 → 返回 (coords, scale_json)；任一关不过 → 打印人话原因后返回 None。

    ★ 从 live() 里原样搬出来的，一行逻辑没动。搬是因为 setup_camera.py 也要走
      同一条检查链（参考点一致性、四个坐标填全没）。抄第二份的下场是两边迟早
      走样 —— 而走样的方式是**默默拿一套偏掉的坐标去标定**，最难发现。

    ★ 为什么是「返回 None」而不是抛异常: 这些检查的失败形态是**给操作员看提示**
      （怎么补、按哪个键），不是程序内部错误。抛异常会把这些提示冲掉。
    """
    # 坐标来源优先级: --coords 指定文件 > 文件内 DOBOT_COORDS > step2_teach_coords.py 的产物
    coords, src, ref_used = DOBOT_COORDS, "step3_hand_eye_calib.py 里的 DOBOT_COORDS", None
    scale_json = ROBOT_POINTS_JSON          # 机器常数记在这个文件里，见 recorded_scale()
    if coords_path:
        d = json.loads(Path(coords_path).read_text(encoding="utf-8"))
        coords = d.get("dobot_coords", d) if isinstance(d, dict) else d
        ref_used = d.get("reference") if isinstance(d, dict) else None
        src = str(coords_path)
        scale_json = Path(coords_path)
    elif not _coords_filled(DOBOT_COORDS, layout.codes):
        rp = load_robot_points()
        if rp:
            coords, ref_used = rp
            src = f"{ROBOT_POINTS_JSON.name}（step2_teach_coords.py 量的）"

    print("═" * 72)
    print(f"  {title}")
    print(f"  参考点: {layout.reference}   码: {layout.codes}")
    print(f"  坐标来源: {src}")
    print("═" * 72)

    # ★★ 参考点必须两边一致。对刀瞄了一个角、标定却按另一个点去拟合的话，
    #    四个点整体平移一个常量 —— 而且四点拟合残差恒为 0，
    #    这个错误不会以任何形式报出来，只会让机械臂每次都偏那几毫米。
    if ref_used and ref_used != layout.reference:
        # 偏移量按实际两套点算，别写死 5.42mm —— 那只对「数据区角 ↔ 白边角」
        # 成立；角 ↔ 中心是另一个数（约 14mm）。写死会让人按错误的量级去排查。
        try:
            shift = min(float(np.linalg.norm(
                np.array(layout.mm_for(ref_used)[c], float)
                - np.array(layout.mm_for(layout.reference)[c], float)))
                for c in layout.codes)
            shift_s = f"{shift:.2f}mm"
        except ValueError as e:
            shift_s = f"（算不出来: {e}）"
        print(f"\n❌ 参考点不一致: {src} 是按 {ref_used!r} 量的，"
              f"但这次标定用 {layout.reference!r}")
        print(f"   {REFERENCE_LABEL.get(ref_used, ref_used)} vs "
              f"{REFERENCE_LABEL[layout.reference]} 相差 {shift_s}，"
              f"会变成固定偏移且**不会**在残差里暴露。")
        print("   二选一，让两边一致:")
        print(f"     a) 沿用你量的时候瞄的点: python3 src/step3_hand_eye_calib.py "
              f"--reference {ref_used}")
        print(f"     b) 重新按 {layout.reference} 瞄一遍: "
              f"python3 src/step2_teach_coords.py --reference {layout.reference}")
        return None

    try:
        check_coords_filled(coords, layout.codes)
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\n  对刀步骤:")
        print(f"   推荐: python3 src/step2_teach_coords.py --reference {layout.reference}"
              "   # 交互式逐个对刀，自动落盘")
        print("         ★ 对完按 q 结束 —— 只有按 q 才落盘")
        print("   或者手动（《方案.md》第二阶段）:")
        print("   1. 标定纸固定在硬纸板上，机械臂底座卡进缺口")
        print(f"   2. DobotStudio 点动，吸盘正中心对准 P1 的"
              f"「{REFERENCE_LABEL[layout.reference]}」")
        print("   3. 记下屏幕上的 (X, Y)（只记 X Y，不要 Z）")
        print("   4. P2/P3/P4 同理，瞄**同一个角**，填进 DOBOT_COORDS 或 output/robot_points.json")
        return None

    return coords, scale_json


def report_matrix(M: np.ndarray, layout: PaperLayout, coords: dict,
                  pixel_pts: np.ndarray) -> None:
    """打印矩阵 + 四码抽查表（「像素→机械臂」对「你填的」）。"""
    print("\n  ✅ 标定完成，矩阵 (像素 → 机械臂 mm):")
    for row in M:
        print("     [" + "  ".join(f"{v: .6f}" for v in row) + "]")

    print("\n  抽查几个点（像素 → 机械臂坐标）:")
    for c, p in zip(layout.codes, pixel_pts):
        x, y = pixel_to_robot(M, *p)
        print(f"     {c}: 像素({p[0]:.0f},{p[1]:.0f}) → 机械臂({x:.1f},{y:.1f})"
              f"   你填的是({coords[c][0]},{coords[c][1]})")
    print("     ↑ 这三列应该一致；不一致说明矩阵有问题。")


def ppm_of_points(layout: PaperLayout, pixel_pts) -> float:
    """
    四个参考点反推「每毫米多少像素」—— 取两两之间 像素距/纸面距 的中位数。

    ★ 为什么不用码的模块尺寸（color_vision 那条路）: 这里的两个比较对象是
      「标定帧」和「抓方块帧」，只要**两边用同一个尺子**就能比出差值。用两两距离
      还能顺带把斜视引起的非均匀缩放摊平，比单看一条边稳。
    """
    mm = layout.mm_for(layout.reference)
    P = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    D = np.array([mm[c] for c in layout.codes], dtype=np.float64)
    cands = [float(np.linalg.norm(P[i] - P[j]) / float(np.linalg.norm(D[i] - D[j])))
             for i in range(4) for j in range(i + 1, 4)]
    return float(np.median(cands))


def live(args) -> int:
    layout = load_paper_layout(PAPER_JSON, reference=args.reference)

    r = resolve_teach_coords(layout, args.coords)
    if r is None:
        return 2
    coords, scale_json = r

    # 机器常数有没有变（撞过/拆过/打滑）—— 只看一眼，不拦标定
    check_scale_drift(layout, coords, scale_json)

    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            print(f"❌ 读不出图片: {args.image}")
            return 2
        cap = StillImage(img)
        print(f"[图片] {args.image}   {img.shape[1]}x{img.shape[0]} px")
    else:
        cap = open_camera(args.cam, args.width, args.height)
        if not cap.isOpened():
            print("❌ 打不开摄像头" + (f" /dev/video{args.cam}" if args.cam is not None
                                      else "（自动挑选失败）"))
            print("   当前系统里的视频节点:")
            print(describe_cameras())
            print("   （也可以先拍一张照片，用 --image 照片.jpg 离线标定）")
            return 2
    try:
        M, pixel_pts = calibrate(cap, det=cv2.QRCodeDetector(), layout=layout,
                                 coords=coords, duration=args.duration,
                                 tol_mm=args.tol)
    except (RuntimeError, ValueError) as e:
        print(f"\n❌ 标定失败: {e}")
        return 1
    finally:
        cap.release()

    report_matrix(M, layout, coords, pixel_pts)

    save_matrix(M, layout, coords, Path(args.out) if args.out else MATRIX_JSON,
                ppm=ppm_of_points(layout, pixel_pts))
    out = Path(args.out) if args.out else MATRIX_JSON
    print(f"\n  已写入 {out}")
    print("  其它程序里这样用:")
    print("     from step3_hand_eye_calib import load_matrix, pixel_to_robot")
    print("     M = load_matrix();  x, y = pixel_to_robot(M, px, py)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="手眼标定（像素→机械臂坐标）")
    ap.add_argument("--selftest", action="store_true",
                    help="不需要硬件，验证标定链路数学是否正确")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    choices=REFERENCE_MODES,
                    help=f"参考点，必须和 step2_teach_coords.py 用的一致"
                         f"（默认 {DEFAULT_REFERENCE}=数据区右上角）: "
                         + " / ".join(f"{m}={REFERENCE_LABEL[m]}" for m in REFERENCE_MODES))
    ap.add_argument("--cam", type=int, default=None,
                    help="摄像头序号；默认不填 = 按设备名自动挑外接摄像头")
    ap.add_argument("--image", default=None,
                    help="不开摄像头，改用一张已拍好的照片算（走的是同一条链路）。"
                         "摄像头没插、或想反复用同一张图对比时用这个。"
                         "★ 照片要能同时看见四个码，且是俯视 —— "
                         "斜着拍即使码能解出，透视越强精度越差。")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--duration", type=float, default=3.0,
                    help="多帧累积的时长(秒)")
    ap.add_argument("--tol", type=float, default=COORD_TOL_MM,
                    help=f"坐标自洽校验的容差(mm)，默认 {COORD_TOL_MM}。"
                         f"★ 本次实测四点偏差 7.45mm，放宽到 8.0 才放行。"
                         f"但请注意: 这个判据在本机已基本失去分辨力 —— "
                         f"把 P2/P3 单点挪 20mm 也只报 7.2~7.8mm，与真数据同量级。"
                         f"所以放宽只是「让它别挡路」，不等于数据自洽。")
    ap.add_argument("--coords", default=None,
                    help="机械臂坐标 JSON 文件(可选，覆盖文件内 DOBOT_COORDS)")
    ap.add_argument("--out", default=None, help="矩阵输出路径")
    args = ap.parse_args()

    if args.selftest:
        return selftest(reference=args.reference)
    return live(args)


if __name__ == "__main__":
    sys.exit(main())
