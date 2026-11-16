"""libpng_quiet —— 屏蔽 "libpng warning: iCCP: known incorrect sRGB profile" 刷屏
其余 stderr 内容（含 tqdm 进度条）原样输出，不影响报错排查。

用法：在脚本最开头（其它 import 之前）加一行
    import libpng_quiet   # noqa: F401
"""

import atexit
import os
import re
import sys
import threading

_NOISE = (b"libpng warning", b"iCCP")

_SEG = re.compile(rb"[^\r\n]*[\r\n]")
_installed = False
_OLD = None
_THREAD = None


def _pump(rfd: int, outfd: int) -> None:
    buf = b""
    try:
        with os.fdopen(rfd, "rb", 0) as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                buf += chunk
                while True:
                    m = _SEG.match(buf)
                    if not m:
                        break
                    seg, buf = buf[: m.end()], buf[m.end():]
                    if not any(n in seg for n in _NOISE):
                        os.write(outfd, seg)
        if buf and not any(n in buf for n in _NOISE):
            os.write(outfd, buf)
    except Exception:
        pass


def install() -> bool:
    """把 fd 2 接进过滤线程。重复调用无副作用。"""
    global _installed, _OLD, _THREAD
    if _installed or os.name != "nt":
        return False
    try:
        r, w = os.pipe()
        old = os.dup(2)
        os.dup2(w, 2)
        os.close(w)
        sys.stderr.flush()
        _OLD = old
        _THREAD = threading.Thread(target=_pump, args=(r, old), daemon=True)
        _THREAD.start()
        _installed = True
        atexit.register(_restore)
        return True
    except Exception:
        return False


def _restore() -> None:
    """退出前把 fd 2 还原并等管道排空，避免最后几行丢失。"""
    try:
        sys.stderr.flush()
        if _OLD is not None:
            os.dup2(_OLD, 2)
        if _THREAD is not None:
            _THREAD.join(2.0)
    except Exception:
        pass


install()
