import cv2
import numpy as np
from camera_capture import CameraCapture
from pixel_to_camera import load_intrinsics, load_table_plane, pixel_to_camera_on_tilted_table
from handeye_calibration import _chessboard_inner_corners, load_transform


def cam_to_base(T_base_cam, p_cam):
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]
    return (R @ p_cam.reshape(3,)) + t


def main():
    K, dist = load_intrinsics("data/camera_intrinsics.npz")
    n, d = load_table_plane("data/table_plane_from_pnp.npz")
    T_base_cam = load_transform("data/T_base_cam.npy")

    errs = []
    with CameraCapture(camera_ip="192.168.1.88") as cam:
        cv2.namedWindow("test_handeye", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test_handeye", 1280, 720)

        while True:
            frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners_4, _ = _chessboard_inner_corners(gray, board_size=(7, 7))

            vis = frame.copy()
            if corners_4 is not None:
                labels = ["TL", "TR", "BL", "BR"]
                colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
                for p, lb, c in zip(corners_4, labels, colors):
                    u, v = float(p[0]), float(p[1])
                    cv2.circle(vis, (int(u), int(v)), 10, c, -1)
                    cv2.putText(vis, f"{lb}({u:.1f},{v:.1f})", (int(u)+10, int(v)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

            cv2.putText(vis, "c: capture&test  q: quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow("test_handeye", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key != ord("c"):
                continue
            if corners_4 is None:
                print("未检测到棋盘，跳过")
                continue
            
            print("\n--- 本次测试 ---")
            for lb, p in zip(["TL", "TR", "BL", "BR"], corners_4):
                u, v = float(p[0]), float(p[1])
                p_cam = pixel_to_camera_on_tilted_table(u, v, K, dist, n, d)
                p_pred = cam_to_base(T_base_cam, p_cam)
                print(f"{lb} 预测 robot(mm): {p_pred}")

                raw = input(f"输入 {lb} 实测 robot x y z（回车跳过）: ").strip()
                if not raw:
                    continue
                try:
                    x, y, z = [float(t) for t in raw.split()]
                    p_gt = np.array([x, y, z], dtype=np.float64)
                    e = np.linalg.norm(p_pred - p_gt)
                    errs.append(e)
                    print(f"{lb} 误差: {e:.3f} mm")
                except Exception:
                    print("输入格式错误，已跳过")

    cv2.destroyAllWindows()

    if errs:
        errs = np.asarray(errs)
        print("\n==== 测试统计 ====")
        print(f"count={len(errs)}, mean={errs.mean():.3f} mm, max={errs.max():.3f} mm, p95={np.percentile(errs,95):.3f} mm")
    else:
        print("无有效测试数据")


if __name__ == "__main__":
    main()