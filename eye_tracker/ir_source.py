"""IR 摄像头 TCP 源 — 连接 C# 桥接程序读取红外帧"""

import socket
import struct
import time
import numpy as np


class IRSource:
    """模拟 cv2.VideoCapture 接口，从 TCP 读取 IR 帧。"""

    def __init__(self, host="127.0.0.1", port=9876):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect((host, port))
        print(f"[i] 已连接到 IR Bridge {host}:{port}")

        # 读取分辨率头
        header = self._recv_exact(8)
        self.width = struct.unpack("<i", header[:4])[0]
        self.height = struct.unpack("<i", header[4:])[0]
        print(f"[i] IR 分辨率: {self.width}x{self.height}")

        self._latest = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("IR Bridge 断开")
            buf.extend(chunk)
        return bytes(buf)

    def is_opened(self):
        return True

    def read(self):
        """返回 (True, bgr_frame)。C# 桥已过滤暗帧。"""
        try:
            size = struct.unpack("<i", self._recv_exact(4))[0]
            data = self._recv_exact(size)
            bgra = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
            bgr = bgra[:, :, :3]
            self._latest = bgr
            return True, bgr
        except Exception:
            return False, self._latest

    def release(self):
        self.sock.close()

    def get(self, prop):
        if prop == 3:   # CAP_PROP_FRAME_WIDTH
            return self.width
        if prop == 4:   # CAP_PROP_FRAME_HEIGHT
            return self.height
        return 0

    def set(self, prop, value):
        pass  # IR 源不支持设置


def start_ir_bridge():
    """启动 C# IR Bridge 子进程，返回进程对象。"""
    import subprocess
    import os
    exe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "ir_bridge", "bin", "Release",
                       "net9.0-windows10.0.19041.0", "ir_bridge.exe")
    if not os.path.exists(exe):
        raise FileNotFoundError(f"IR Bridge 未编译: {exe}\n请先: cd ir_bridge && dotnet build -c Release")
    proc = subprocess.Popen([exe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(1.5)
    return proc
