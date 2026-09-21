# -*- coding: utf-8 -*-
"""
DobotDll.py — 新版 Magician SDK (magiciandll) 的 Python ctypes 封装
================================================================
背景: 官网 demo 自带的 DobotDllType.py 是【旧版 SDK】封装(ConnectDobot 传
      ConnectInfo 结构体、带 masterId/slaveId);
      而 run-linux 里的 libDobotDll.so 是【新版 SDK】(源码见
      build/magiciandll-master/src/DobotDll.h),其导出为:
          ConnectDobot(port, baud, fwType*, version*, time*)
          GetPose(Pose*)  等,均【无 masterId 参数】。
      两者混用导致读到乱码("Dobo"/"reSt")与段错误。
本文件按新版头文件逐一对应实现,结构体均为 #pragma pack(1)。
"""
import ctypes
import glob
import os
import time
from ctypes import (CDLL, Structure, RTLD_GLOBAL, byref, c_bool, c_byte,
                    c_char_p, c_float, c_int, c_uint8, c_uint32, c_uint64,
                    create_string_buffer, POINTER)

# ---------------------------------------------------------------------------
# 常量 / 枚举(与 DobotType.h 一致)
# ---------------------------------------------------------------------------
DobotConnect_NoError = 0
DobotConnect_NotFound = 1
DobotConnect_Occupied = 2

DobotCommunicate_NoError = 0
DobotCommunicate_BufferFull = 1
DobotCommunicate_Timeout = 2
DobotCommunicate_InvalidParams = 3

# PTP 运动模式(与 DobotType.h 中 enum PTPMode 一致)
class PTPMode:
    PTPJUMPXYZMode      = 0   # 跳跃: XYZ 绝对坐标
    PTPMOVJXYZMode      = 1   # 关节运动: XYZ 绝对坐标
    PTPMOVLXYZMode      = 2   # 直线运动: XYZ 绝对坐标
    PTPJUMPANGLEMode    = 3   # 跳跃: 关节角绝对坐标
    PTPMOVJANGLEMode    = 4   # 关节运动: 关节角绝对坐标
    PTPMOVLANGLEMode    = 5   # 直线运动: 关节角绝对坐标
    PTPMOVJANGLEINCMode = 6   # 关节运动: 关节角增量
    PTPMOVLXYZINCMode   = 7   # 直线运动: XYZ 增量
    PTPMOVJXYZINCMode   = 8   # 关节运动: XYZ 增量

# ---------------------------------------------------------------------------
# 结构体(全部 #pragma pack(1))
# ---------------------------------------------------------------------------
class Pose(Structure):            # 实时位姿
    _pack_ = 1
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float), ("r", c_float),
                ("jointAngle", c_float * 4)]

class HOMEParams(Structure):      # 回零后的坐标原点 (即 home 位姿对应的用户坐标)
    _pack_ = 1
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float), ("r", c_float)]

class HOMECmd(Structure):         # 回零命令
    _pack_ = 1
    _fields_ = [("reserved", c_uint32)]

class PTPJointParams(Structure):  # PTP 各关节速度/加速度
    _pack_ = 1
    _fields_ = [("velocity", c_float * 4), ("acceleration", c_float * 4)]

class PTPCommonParams(Structure): # PTP 全局速度/加速度比例 (%)
    _pack_ = 1
    _fields_ = [("velocityRatio", c_float), ("accelerationRatio", c_float)]

class PTPCmd(Structure):          # PTP 指令
    _pack_ = 1
    _fields_ = [("ptpMode", c_uint8),
                ("x", c_float), ("y", c_float), ("z", c_float), ("r", c_float)]

def find_port():
    """自动查找越疆机械臂的串口设备 (CP210x), 兼容端口号变化"""
    # 优先用稳定符号链接
    byid = "/dev/serial/by-id/"
    try:
        for name in os.listdir(byid):
            low = name.lower()
            if "cp210" in low or "dobot" in low or "uart" in low:
                return os.path.realpath(os.path.join(byid, name))
    except OSError:
        pass
    # 回退: 扫描 ttyUSB/ttyACM
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise RuntimeError("未找到机械臂串口设备, 请检查 USB 连接")


# ---------------------------------------------------------------------------
# 底层加载
# ---------------------------------------------------------------------------
def load():
    """加载当前目录下的 libDobotDll.so, 返回 CDLL 实例"""
    return CDLL("./libDobotDll.so", RTLD_GLOBAL)


def dSleep(ms):
    time.sleep(ms / 1000.0)


def SetDebugEnable(api, flag=False):
    api.SetDebugEnable(c_bool(flag))


# ---------------------------------------------------------------------------
# 连接 / 断开
# ---------------------------------------------------------------------------
def ConnectDobot(api, portName, baudrate):
    """新版 API: 单设备, 直接连接。
    返回 [result, fwType, fwVersion, runTime]"""
    fwType = create_string_buffer(64)
    version = create_string_buffer(64)
    t = c_float(0.0)
    result = api.ConnectDobot(c_char_p(portName.encode("utf-8")),
                              c_uint32(baudrate), fwType, version, byref(t))
    return [result, fwType.value.decode("utf-8", "ignore"),
            version.value.decode("utf-8", "ignore"), t.value]


def DisconnectDobot(api):
    return api.DisconnectDobot()


def SetCmdTimeout(api, ms=3000):
    """设置单条命令响应超时(毫秒), 官方推荐 3000"""
    return api.SetCmdTimeout(c_uint32(ms))


def SearchDobot(api, maxLen=1000):
    buf = create_string_buffer(maxLen)
    api.SearchDobot(buf, maxLen)
    return buf.value.decode("utf-8", "ignore")


# ---------------------------------------------------------------------------
# 状态查询(均为只读, 不会引起运动)
# ---------------------------------------------------------------------------
def GetDeviceVersion(api):
    major, minor, revision, hw = c_uint8(0), c_uint8(0), c_uint8(0), c_uint8(0)
    result = api.GetDeviceVersion(byref(major), byref(minor),
                                  byref(revision), byref(hw))
    return [result, major.value, minor.value, revision.value, hw.value]


def GetPose(api):
    """读实时位姿。返回 [x, y, z, r, j1, j2, j3, j4]"""
    pose = Pose()
    while True:
        result = api.GetPose(byref(pose))
        if result == DobotCommunicate_NoError:
            break
        dSleep(5)
    return [pose.x, pose.y, pose.z, pose.r] + list(pose.jointAngle)


def GetQueuedCmdCurrentIndex(api):
    idx = c_uint64(0)
    result = api.GetQueuedCmdCurrentIndex(byref(idx))
    return [result, idx.value]


def GetQueuedCmdMotionFinish(api):
    isFinish = c_bool(False)
    result = api.GetQueuedCmdMotionFinish(byref(isFinish))
    return [result, bool(isFinish.value)]


# ---------------------------------------------------------------------------
# 报警
# ---------------------------------------------------------------------------
def GetAlarmsState(api, maxLen=32):
    buf = (c_uint8 * maxLen)()
    length = c_uint32(0)   # 注意: C 头文件是 uint32_t*, 不能用 c_uint8(会越界写坏内存)
    result = api.GetAlarmsState(buf, byref(length), maxLen)
    return [result, list(buf)[:length.value]]


def ClearAllAlarmsState(api):
    return api.ClearAllAlarmsState()


# ---------------------------------------------------------------------------
# 末端工具参数 (影响 XYZ 运动规划, 偏移量必须与实际安装的工具一致)
# ---------------------------------------------------------------------------
class EndEffectorParams(Structure):
    _pack_ = 1
    _fields_ = [("xBias", c_float), ("yBias", c_float), ("zBias", c_float)]


def SetEndEffectorParams(api, xBias, yBias, zBias, isQueued=0):
    p = EndEffectorParams(xBias, yBias, zBias)
    idx = c_uint64(0)
    result = api.SetEndEffectorParams(byref(p), c_bool(isQueued), byref(idx))
    return [result, idx.value]


def GetEndEffectorParams(api):
    p = EndEffectorParams()
    result = api.GetEndEffectorParams(byref(p))
    return [result, p.xBias, p.yBias, p.zBias]


# ---------------------------------------------------------------------------
# 回零 HOME
# ---------------------------------------------------------------------------
def GetHOMEParams(api):
    p = HOMEParams()
    result = api.GetHOMEParams(byref(p))
    return [result, p.x, p.y, p.z, p.r]


# ---------------------------------------------------------------------------
# 指令队列控制
# ---------------------------------------------------------------------------
def SetQueuedCmdClear(api):
    return api.SetQueuedCmdClear()


def SetQueuedCmdStartExec(api):
    return api.SetQueuedCmdStartExec()


def SetQueuedCmdStopExec(api):
    return api.SetQueuedCmdStopExec()


def SetQueuedCmdForceStopExec(api):
    return api.SetQueuedCmdForceStopExec()


# ---------------------------------------------------------------------------
# 回零 HOME
# ---------------------------------------------------------------------------
def SetHOMEParams(api, x, y, z, r, isQueued=0):
    p = HOMEParams(x, y, z, r)
    queuedCmdIndex = c_uint64(0)
    result = api.SetHOMEParams(byref(p), c_bool(isQueued), byref(queuedCmdIndex))
    return [result, queuedCmdIndex.value]


def SetHOMECmd(api, isQueued=0):
    """执行回零。isQueued=1 时放入指令队列, 由 StartExec 触发执行"""
    cmd = HOMECmd(0)
    queuedCmdIndex = c_uint64(0)
    result = api.SetHOMECmd(byref(cmd), c_bool(isQueued), byref(queuedCmdIndex))
    return [result, queuedCmdIndex.value]


# ---------------------------------------------------------------------------
# PTP 运动参数与指令
# ---------------------------------------------------------------------------
def SetPTPJointParams(api, v1, v2, v3, v4, a1, a2, a3, a4, isQueued=0):
    """设置 PTP 模式下 J1~J4 的速度与加速度"""
    p = PTPJointParams()
    p.velocity[0], p.velocity[1], p.velocity[2], p.velocity[3] = v1, v2, v3, v4
    p.acceleration[0], p.acceleration[1], p.acceleration[2], p.acceleration[3] = a1, a2, a3, a4
    queuedCmdIndex = c_uint64(0)
    result = api.SetPTPJointParams(byref(p), c_bool(isQueued), byref(queuedCmdIndex))
    return [result, queuedCmdIndex.value]


def SetPTPCommonParams(api, velocityRatio, accelerationRatio, isQueued=0):
    """设置 PTP 全局速度/加速度比例(0~100 %)"""
    p = PTPCommonParams(velocityRatio, accelerationRatio)
    queuedCmdIndex = c_uint64(0)
    result = api.SetPTPCommonParams(byref(p), c_bool(isQueued), byref(queuedCmdIndex))
    return [result, queuedCmdIndex.value]


def SetPTPCmd(api, ptpMode, x, y, z, r, isQueued=0):
    """PTP 运动指令。坐标模式(绝对 XYZ / 绝对关节角 / 增量等)由 ptpMode 决定。
    返回 [result, queuedCmdIndex]"""
    cmd = PTPCmd(ptpMode, x, y, z, r)
    queuedCmdIndex = c_uint64(0)
    result = api.SetPTPCmd(byref(cmd), c_bool(isQueued), byref(queuedCmdIndex))
    return [result, queuedCmdIndex.value]


def GetPTPTime(api, ptpMode, x, y, z, r):
    """估算 PTP 运动耗时(毫秒), 只读不运动"""
    cmd = PTPCmd(ptpMode, x, y, z, r)
    t = c_uint32(0)
    result = api.GetPTPTime(byref(cmd), byref(t))
    return [result, t.value]
