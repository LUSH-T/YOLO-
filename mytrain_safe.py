r"""安全版训练脚本：先检查系统内存余量，再按余量自动选 batch，避免训练中途 OOM。.

与 mytrain.py 的区别：
  1. 开跑前读取 Windows 的「提交上限 / 可用提交」，不够就直接拦下并给操作指引，
     不会再跑到第 10 轮才崩；
  2. batch 由可用内存自动决定（不再用 -1 自动探测到 25）；
  3. cache 关闭、workers=0；
  4. 从 train-9/weights/best.pt（mAP50=0.930）继续微调，学习率调低；
  5. degrees 从 180 降到 15（地面机器人没必要倒立）。

用法：在仓库目录下
    D:\\Anaconda\\envs\\yolo\\python.exe mytrain_safe.py
"""

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------- 1. 内存预检 ----------------


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def commit_status():
    """返回 (提交上限GB, 可用提交GB)。可用提交才是决定能不能训练的那个数。."""
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPageFile / 2**30, m.ullAvailPageFile / 2**30


def pick_batch(avail_gb: float) -> int:
    """按可用提交内存选 batch。经验值：batch 越大，显存/内存占用越高。."""
    if avail_gb >= 13:
        return 12
    if avail_gb >= 9:
        return 8
    if avail_gb >= 6.5:
        return 4
    return 0


def main() -> int:
    limit, avail = commit_status()
    print(f"[内存] 提交上限 {limit:.1f} GB，当前可用 {avail:.1f} GB")

    batch = pick_batch(avail)
    if batch == 0:
        print(
            "\n[中止] 可用提交内存不足 6.5 GB，训练必定中途崩溃。请二选一：\n"
            "  A. 临时方案：关掉 PyCharm、Visual Studio、微信、浏览器后重跑本脚本；\n"
            "  B. 根治方案：给 D 盘加 32 GB 页面文件后重启（见下方步骤）。\n"
            "     设置 → 系统 → 系统信息 → 高级系统设置 → 性能『设置』→ 高级 →\n"
            "     虚拟内存『更改』→ 取消『自动管理』→ 选 D 盘 → 自定义 32768 / 65536\n"
            "     → 『设置』→ 确定 → 重启。"
        )
        return 1

    print(f"[内存] 按当前余量选用 batch={batch}")
    if batch <= 4:
        print("[内存] 余量偏紧：建议先关掉 PyCharm / Visual Studio / 微信 / 浏览器再跑，否则仍可能在第 10 轮验证时崩。")

    from ultralytics import YOLO

    ckpt = ROOT / "runs/detect/train-9/weights/best.pt"
    print(f"[模型] 从 {ckpt} 继续微调" if ckpt.is_file() else f"[模型] {ckpt} 不存在，改用 yolo11n.pt")

    model = YOLO(str(ckpt) if ckpt.is_file() else "yolo11n.pt")
    model.train(
        # ---------- 数据 ----------
        data="robo.yaml",
        imgsz=640,
        # ---------- 内存控制 ----------
        batch=batch,  # 由上面的预检决定，不再用 -1
        cache=False,  # 不缓存到内存；405 张图加速有限，却要多吃内存
        workers=0,  # Windows 上必须 0，否则容易 WinError 1455
        amp=True,
        # ---------- 训练时长 ----------
        epochs=60,  # 已有 0.93 的底子，短程微调足够
        patience=20,
        # ---------- 优化器 ----------
        optimizer="AdamW",
        lr0=0.0005,  # 微调，步长比 0.001 再小一档
        cos_lr=True,
        # ---------- 增强 ----------
        degrees=15,  # 地面机器人不需要 180° 倒立
        scale=0.75,
        mosaic=0.8,
        close_mosaic=15,
        fliplr=0.5,
        hsv_v=0.5,
        # ---------- 输出 ----------
        name="train-10",
        exist_ok=True,
        plots=False,  # 关掉绘图，省一次内存峰值
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
