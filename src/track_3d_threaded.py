"""
==========================================
  MCE14 Vision — Threaded Ball Tracker v2.1
  แยก 3 Threads: Capture / Inference / Display
  เพื่อป้องกัน YOLO บล็อก Frame Capture ที่ 60 FPS
==========================================
"""
import cv2
import depthai as dai
import numpy as np
import torch
from ultralytics import YOLO
import socket
import time
import math
import os
import datetime
import threading
from collections import deque
from udp_sender import UDPSender
from ball_tracker_kf import BallTrackerKF

# ค้นหาว่ามี GPU ให้ใช้หรือไม่
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
USE_HALF = (DEVICE == 'cuda:0')

# ==========================================
# ⚙️ SYSTEM CONFIGURATIONS
# ==========================================
CAMERA_HEIGHT_CM = 121.4
CAMERA_PITCH_DEG = 0.0
ORIGIN_Y_DISTANCE_CM = 323.3
ORIGIN_X_OFFSET_CM = 0.0
ESP32_IPS = ["192.168.137.33"]
ESP32_PORT = 12345
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
CONFIDENCE_THRESHOLD = 0.5
DRAG_CORRECTION = 0.92
FRAME_SKIP = 1
MIN_PROCESS_COUNT = 2
LANDING_DEADLINE_SEC = 0.28
# ==========================================

# ==========================================
# 🧵 THREAD-SAFE SHARED STATE
# ==========================================
class SharedState:
    """ตัวแปรที่แชร์ระหว่าง 3 Threads อย่างปลอดภัย"""
    def __init__(self):
        self.lock = threading.Lock()
        # Capture → Inference
        self.latest_rgb = None
        self.latest_depth = None
        self.frame_ready = threading.Event()
        # Inference → Display
        self.display_frame = None
        self.display_ready = threading.Event()
        # Control
        self.running = True
        self.key_pressed = -1
        # Calibration (shared)
        self.is_calibrated = False
        self.latest_raw_depth = 0.0
        self.latest_raw_width = 0.0

shared = SharedState()

# ==========================================
# UDP + Log Setup
# ==========================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
esp32_sender = UDPSender(ESP32_IPS, ESP32_PORT)
LOG_FILE = f"landing_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("Timestamp,Clamped_X,Clamped_Y,Raw_X,Raw_Y,Vx,Vy,Vz,Elapsed_ms\n")
    print(f"📝 สร้างไฟล์ Log: {LOG_FILE}")

# ==========================================
# โหลดโมเดล YOLOv8
# ==========================================
model_path = r"C:\Users\punna\OneDrive\Documents\runs\detect\redball_model\weights\best.engine"
model = YOLO(model_path)

# ==========================================
# ฟังก์ชัน: หาระยะ Z จาก Depth Map
# ==========================================
def get_distance_from_center(depth_map, x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    roi_x1 = int(x1 + w * 0.35)
    roi_x2 = int(x2 - w * 0.35)
    roi_y1 = int(y1 + h * 0.35)
    roi_y2 = int(y2 - h * 0.35)
    if roi_x2 <= roi_x1: roi_x2 = roi_x1 + 1
    if roi_y2 <= roi_y1: roi_y2 = roi_y1 + 1
    roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
    roi_x2, roi_y2 = min(depth_map.shape[1], roi_x2), min(depth_map.shape[0], roi_y2)
    roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
    valid_depths = roi[roi > 0]
    if len(valid_depths) == 0:
        roi_x1, roi_x2 = int(x1 + w * 0.10), int(x2 - w * 0.10)
        roi_y1, roi_y2 = int(y1 + h * 0.10), int(y2 - h * 0.10)
        roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
        roi_x2, roi_y2 = min(depth_map.shape[1], roi_x2), min(depth_map.shape[0], roi_y2)
        roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
        valid_depths = roi[roi > 0]
    if len(valid_depths) > 0:
        sorted_depths = np.sort(valid_depths)
        half_idx = max(1, len(sorted_depths) // 2)
        return np.median(sorted_depths[:half_idx])
    return 0


# ==========================================
# 🧵 THREAD 1: CAPTURE (ดึงเฟรมจากกล้อง)
# ==========================================
def capture_thread(qRgb, qDepth):
    """ดึงเฟรมจาก OAK-D อย่างต่อเนื่อง ไม่ถูก YOLO บล็อก"""
    print("🧵 [CAPTURE] Thread เริ่มทำงาน")
    while shared.running:
        inRgb = qRgb.get()
        inDepth = qDepth.tryGet()
        if inRgb is not None:
            frame = inRgb.getCvFrame()
            depth = inDepth.getFrame() if inDepth is not None else None
            with shared.lock:
                shared.latest_rgb = frame
                shared.latest_depth = depth
            shared.frame_ready.set()
    print("🧵 [CAPTURE] Thread หยุดทำงาน")


# ==========================================
# 🧵 THREAD 2: INFERENCE (YOLO + KF + UDP)
# ==========================================
def inference_thread(camera_matrix, dist_coeffs, fx, fy, cx, cy):
    """รัน YOLO + Kalman Filter + Landing Prediction + UDP Send"""
    global ORIGIN_Y_DISTANCE_CM, ORIGIN_X_OFFSET_CM
    print("🧵 [INFERENCE] Thread เริ่มทำงาน")

    kf = BallTrackerKF(drag_correction=DRAG_CORRECTION)
    
    # สถานะทั้งหมดอยู่ใน Thread นี้ (ไม่ต้อง lock)
    landing_process_count = 0
    last_valid_landing = None
    landing_sent = False
    raw_frame_count = 0
    is_ball_thrown = False
    last_throw_time = 0
    throw_detect_time = 0
    missing_frames = 0
    
    # Pre-compute pitch rotation
    theta = math.radians(CAMERA_PITCH_DEG)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    while shared.running:
        # รอเฟรมใหม่จาก Capture Thread
        shared.frame_ready.wait(timeout=0.1)
        shared.frame_ready.clear()
        
        with shared.lock:
            frame = shared.latest_rgb
            depth_frame = shared.latest_depth
            is_calibrated = shared.is_calibrated
        
        if frame is None:
            continue
        
        # === YOLO Inference ===
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device=DEVICE, half=USE_HALF)
        annotated_frame = frame.copy()
        
        measured_xyz = None
        landing_pt = (None, None)
        
        # === Ball Detection (Class 0) ===
        ball_boxes = [box for box in results[0].boxes if int(box.cls[0].item()) == 0]
        
        if len(ball_boxes) > 0:
            missing_frames = 0
            best_box = sorted(ball_boxes, key=lambda x: x.conf[0].item(), reverse=True)[0]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            label_text = "N/A"
            if depth_frame is not None:
                Z_mm = get_distance_from_center(depth_frame, x1, y1, x2, y2)
                if Z_mm > 0:
                    pts = np.array([[[float(center_x), float(center_y)]]], dtype=np.float32)
                    undistorted_pts = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
                    u_cx, u_cy = undistorted_pts[0][0]
                    
                    X_c = ((u_cx - cx) * Z_mm / fx) / 10.0
                    Y_c = ((u_cy - cy) * Z_mm / fy) / 10.0
                    Z_c = Z_mm / 10.0
                    
                    Y_world_c = Y_c * cos_theta + Z_c * sin_theta
                    Z_world_c = -Y_c * sin_theta + Z_c * cos_theta
                    
                    with shared.lock:
                        shared.latest_raw_depth = Z_world_c
                        shared.latest_raw_width = X_c
                    
                    X_cm = ORIGIN_Y_DISTANCE_CM - Z_world_c
                    Y_cm = X_c - ORIGIN_X_OFFSET_CM
                    Z_cm = CAMERA_HEIGHT_CM - Y_world_c
                    
                    measured_xyz = [X_cm, Y_cm, Z_cm]
                    label_text = f"X: {X_cm:.1f}cm, Y: {Y_cm:.1f}cm, Z: {Z_cm:.1f}cm"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # === Robot Detection (Class 1) ===
            robot_boxes = [box for box in results[0].boxes if int(box.cls[0].item()) == 1]
            if len(robot_boxes) > 0:
                r_box = sorted(robot_boxes, key=lambda x: x.conf[0].item(), reverse=True)[0]
                rx1, ry1, rx2, ry2 = map(int, r_box.xyxy[0])
                r_label = "ROBOT N/A"
                if depth_frame is not None:
                    r_Z_mm = get_distance_from_center(depth_frame, rx1, ry1, rx2, ry2)
                    if r_Z_mm > 0:
                        r_pts = np.array([[[float((rx1+rx2)//2), float((ry1+ry2)//2)]]], dtype=np.float32)
                        r_und = cv2.undistortPoints(r_pts, camera_matrix, dist_coeffs, P=camera_matrix)
                        ru_cx, ru_cy = r_und[0][0]
                        r_X_c = ((ru_cx - cx) * r_Z_mm / fx) / 10.0
                        r_Y_c = ((ru_cy - cy) * r_Z_mm / fy) / 10.0
                        r_Z_c = r_Z_mm / 10.0
                        r_Z_world = -r_Y_c * sin_theta + r_Z_c * cos_theta
                        r_X_cm = ORIGIN_Y_DISTANCE_CM - r_Z_world
                        r_Y_cm = r_X_c - ORIGIN_X_OFFSET_CM
                        r_label = f"ROBOT X:{r_X_cm:.0f} Y:{r_Y_cm:.0f}"
                cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, r_label, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # === Kalman Filter Update ===
            raw_frame_count += 1
            if is_calibrated and raw_frame_count % FRAME_SKIP == 0:
                kf.process(measured_xyz, origin_distance_cm=ORIGIN_Y_DISTANCE_CM)
            
            # Hybrid Prediction
            if is_calibrated:
                if len(kf.history) >= 5:
                    landing_pt = kf.predict_landing()
                else:
                    landing_pt = kf.predict_landing_fast()
            
            # === UDP Send + Throw Detection ===
            if is_calibrated and measured_xyz is not None:
                vx, vy, vz = kf.state[3:].flatten()
                if landing_pt[0] is not None:
                    msg = f"{measured_xyz[0]:.2f},{measured_xyz[1]:.2f},{measured_xyz[2]:.2f},{landing_pt[0]:.2f},{landing_pt[1]:.2f},{vx:.2f},{vy:.2f},{vz:.2f}"
                else:
                    msg = f"{measured_xyz[0]:.2f},{measured_xyz[1]:.2f},{measured_xyz[2]:.2f},None,None,{vx:.2f},{vy:.2f},{vz:.2f}"
                try:
                    sock.sendto(msg.encode('utf-8'), (UDP_IP, UDP_PORT))
                    
                    # Throw Detection
                    if (vz > 15 or abs(vx) > 20) and (time.time() - last_throw_time > 1.5):
                        throw_detect_time = time.time()
                        print(f"🚀 [THROW DETECTED] (Vx:{vx:.1f}, Vz:{vz:.1f}) | เริ่มจับเวลา 0.3s")
                        kf.P[3:, 3:] = np.eye(3) * 1000.0
                        kf.reset_history()
                        is_ball_thrown = True
                        landing_sent = False
                        landing_process_count = 0
                        last_throw_time = time.time()
                    
                    is_falling = (vz < -2) and is_ball_thrown
                    time_since_throw = time.time() - throw_detect_time if throw_detect_time > 0 else 0
                    deadline_expired = (time_since_throw >= LANDING_DEADLINE_SEC) and is_ball_thrown and not landing_sent
                    
                    if landing_pt[0] is not None and (is_falling or deadline_expired):
                        clamped_x = max(-50.0, min(50.0, landing_pt[0]))
                        clamped_y = max(-50.0, min(50.0, landing_pt[1]))
                        last_valid_landing = (clamped_x, clamped_y)
                        
                        if raw_frame_count % FRAME_SKIP == 0:
                            landing_process_count += 1
                        
                        should_send = (landing_process_count >= MIN_PROCESS_COUNT) or deadline_expired
                        if should_send and not landing_sent:
                            elapsed_ms = time_since_throw * 1000
                            esp32_sender.send_data_binary(last_valid_landing[0], last_valid_landing[1], 0.0)
                            landing_sent = True
                            mode_str = "DEADLINE" if deadline_expired else f"KF x{landing_process_count}"
                            pred_mode = "SVD" if len(kf.history) >= 5 else "FAST"
                            print(f"\n🎯 [SENT in {elapsed_ms:.0f}ms] [{mode_str}] [{pred_mode}] -> ESP32: X={last_valid_landing[0]:.1f}, Y={last_valid_landing[1]:.1f}")
                            with open(LOG_FILE, "a", encoding="utf-8") as f:
                                ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                f.write(f"{ts},{last_valid_landing[0]:.2f},{last_valid_landing[1]:.2f},{landing_pt[0]:.2f},{landing_pt[1]:.2f},{vx:.2f},{vy:.2f},{vz:.2f},{elapsed_ms:.0f}\n")
                        
                        if not (-50 <= landing_pt[0] <= 50 and -50 <= landing_pt[1] <= 50):
                            cv2.putText(annotated_frame, "OUT OF BOUNDS (CLAMPED)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    print(f"[ERROR] UDP: {e}")
        else:
            missing_frames += 1
            if missing_frames > 15:
                landing_process_count = 0
                landing_sent = False
                last_valid_landing = None
                is_ball_thrown = False
            if kf.is_initialized and kf.state[2, 0] < 5.0 and is_ball_thrown:
                print("🏁 [บอลตกถึงพื้น] Z < 5cm -> รีเซ็ต")
                is_ball_thrown = False
        
        # === วาด HUD ===
        if not is_calibrated:
            cv2.putText(annotated_frame, "CALIBRATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(annotated_frame, "Press 'z' to SET ZERO", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif landing_pt[0] is not None:
            cv2.putText(annotated_frame, f"Pred Land X:{landing_pt[0]:.1f} Y:{landing_pt[1]:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            kx, ky, kz = kf.state[:3].flatten()
            cv2.putText(annotated_frame, f"KF X:{kx:.1f} Y:{ky:.1f} Z:{kz:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            _vx, _vy, _vz = kf.state[3:].flatten()
            cv2.putText(annotated_frame, f"VEL Vx:{_vx:.1f} Vy:{_vy:.1f} Vz:{_vz:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # ส่งเฟรมไป Display Thread
        with shared.lock:
            shared.display_frame = annotated_frame
        shared.display_ready.set()
    
    print("🧵 [INFERENCE] Thread หยุดทำงาน")


# ==========================================
# 🚀 MAIN — ตั้งค่า Pipeline + เริ่ม Threads
# ==========================================
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 3)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setFps(60)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.isp.link(xoutRgb.input)

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setSubpixel(True)

config = stereo.initialConfig.get()
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.spatialFilter.alpha = 0.5
config.postProcessing.spatialFilter.delta = 20
config.postProcessing.temporalFilter.enable = True
config.postProcessing.temporalFilter.alpha = 0.4
config.postProcessing.temporalFilter.delta = 20
config.postProcessing.temporalFilter.persistencyMode = dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_OUT_OF_8
stereo.initialConfig.set(config)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(camRgb.getIspWidth(), camRgb.getIspHeight())

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

# === รัน Device + Threads ===
with dai.Device(pipeline) as device:
    print("เริ่มเชื่อมต่อกล้อง กำลังดึงค่า Camera Intrinsics...")
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, camRgb.getIspWidth(), camRgb.getIspHeight())
    camera_matrix = np.array(intrinsics, dtype=np.float32)
    dist_coeffs = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), dtype=np.float32)
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    print(f"Lens: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    
    # เริ่ม Threads
    t_capture = threading.Thread(target=capture_thread, args=(qRgb, qDepth), daemon=True)
    t_inference = threading.Thread(target=inference_thread, args=(camera_matrix, dist_coeffs, fx, fy, cx, cy), daemon=True)
    
    t_capture.start()
    t_inference.start()
    print("🧵 Threads ทั้ง 3 เริ่มทำงานแล้ว (Capture / Inference / Display)")
    
    # === MAIN THREAD: Display + Key Handling ===
    # (OpenCV cv2.imshow ต้องอยู่ใน Main Thread เท่านั้น)
    while shared.running:
        if shared.display_ready.wait(timeout=0.1):
            shared.display_ready.clear()
            with shared.lock:
                display = shared.display_frame
            if display is not None:
                cv2.imshow("MCE14 Vision (Threaded)", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            shared.running = False
            break
        elif key == ord('z') and not shared.is_calibrated:
            with shared.lock:
                raw_d = shared.latest_raw_depth
                raw_w = shared.latest_raw_width
            if raw_d > 0:
                ORIGIN_Y_DISTANCE_CM = raw_d
                ORIGIN_X_OFFSET_CM = raw_w
                shared.is_calibrated = True
                print(f"\n✅ [SET ZERO] X=0: {ORIGIN_Y_DISTANCE_CM:.1f} cm, Y=0: {ORIGIN_X_OFFSET_CM:.1f} cm")
            else:
                print("\n⚠️ [SET ZERO FAILED] กรุณาวางลูกบอลให้กล้องเห็นชัดเจน")
    
    # Cleanup
    shared.running = False
    t_capture.join(timeout=2)
    t_inference.join(timeout=2)

cv2.destroyAllWindows()
esp32_sender.close()
print("🛑 ปิดโปรแกรมเรียบร้อย")
