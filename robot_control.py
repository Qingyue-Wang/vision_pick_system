import json
import socket


class RobotControl:
    """
    机器人控制：
    - dry_run=True: 仅打印，不实际下发
    - dry_run=False: 通过 TCP 发送 JSON
    """

    def __init__(self, host="127.0.0.1", port=30003, dry_run=True, timeout=2.0):
        self.host = host
        self.port = port
        self.dry_run = dry_run
        self.timeout = timeout
        self.sock = None

    def connect(self):
        if self.dry_run:
            return
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_pose(self, pose_xyzrpy):
        x, y, z, rx, ry, rz = [float(v) for v in pose_xyzrpy]
        cmd = {"cmd": "movej_pose", "pose": [x, y, z, rx, ry, rz]}
        if self.dry_run:
            print("[DRY_RUN] send:", cmd)
        else:
            self.sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))

    def execute_pick(self, target_pose_xyzrpy, approach_mm=60.0):
        x, y, z, rx, ry, rz = [float(v) for v in target_pose_xyzrpy]
        pre_pose = [x, y, z + approach_mm, rx, ry, rz]
        pick_pose = [x, y, z, rx, ry, rz]

        print("执行抓取流程: pre-pick -> pick -> retreat")
        self.send_pose(pre_pose)
        self.send_pose(pick_pose)
        # 这里可插入夹爪闭合命令
        self.send_pose(pre_pose)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()