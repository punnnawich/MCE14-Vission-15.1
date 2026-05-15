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
from udp_sender import UDPSender

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
USE_HALF = (DEVICE == 'cuda:0')

class BallTrackerKF:
    def __init__(self):
        self.state = np.zeros((6, 1))
        self.P = np.eye(6) * 1000.0
        self.R = np.diag([50.0, 10.0, 20.0])
        self.Q = np.eye(6) * 0.1
        self.Q[3:, 3:] = np.diag([150.0, 150.0, 150.0])
        self.F = np.eye(6)
        self.B = np.zeros((6, 1))
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0; self.H[1, 1] = 1.0; self.H[2, 2] = 1.0
        self.I = np.eye(6)
        self.last_time = time.time()
        self.is_initialized = False
        self.history = []

    def process(self, measurement=None):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        if dt > 1.0: self.is_initialized = False
        if not self.is_initialized:
            if measurement is not None:
                self.state[:3] = np.array(measurement).reshape((3, 1))
                self.state[3:] = 0
                self.is_initialized = True
            return self.state
            
        self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt
        self.B[2, 0] = -0.5 * 981 * dt**2 
        self.B[5, 0] = -981 * dt
        self.state = np.dot(self.F, self.state) + self.B
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        if measurement is not None:
            z = np.array(measurement).reshape((3, 1))
            distance_from_camera_cm = 323.3 - measurement[0]
            distance_m = abs(distance_from_camera_cm) / 100.0
            noise_x = 10.0 + (distance_m ** 2.0) * 15.0
            noise_y = 5.0 + (distance_m ** 1.5) * 3.0
            noise_z = 5.0 + (distance_m ** 1.5) * 5.0
            self.R = np.diag([noise_x, noise_y, noise_z])
            
            y = z - np.dot(self.H, self.state)
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            self.state = self.state + np.dot(K, y)
            self.P = np.dot(self.I - np.dot(K, self.H), self.P)
            
        self.history.append((self.last_time, self.state[0, 0], self.state[1, 0], self.state[2, 0]))
        if len(self.history) > 20: self.history.pop(0)
        return self.state
        
    def reset_history(self):
        self.history = []
        
    def predict_landing(self):
        if not self.is_initialized or len(self.history) < 5: return None, None
        t_arr = np.array([pt[0] for pt in self.history])
        t_arr = t_arr - t_arr[0]
        x_arr = np.array([pt[1] for pt in self.history])
        y_arr = np.array([pt[2] for pt in self.history])
        z_arr = np.array([pt[3] for pt in self.history])
        
        if z_arr[-1] <= 0: return float(x_arr[-1]), float(y_arr[-1])
            
        mean_x, mean_y = np.mean(x_arr), np.mean(y_arr)
        M = np.vstack((x_arr - mean_x, y_arr - mean_y)).T
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        dir_vector = Vt[0]
        projected = np.dot(M, dir_vector)
        clean_xy = np.outer(projected, dir_vector)
        clean_x = clean_xy[:, 0] + mean_x
        clean_y = clean_xy[:, 1] + mean_y
        
        px = np.polyfit(t_arr, clean_x, 1)
        py = np.polyfit(t_arr, clean_y, 1)
        vx, x0 = px[0], px[1]
        vy, y0 = py[0], py[1]
        
        g = 981.0
        z_adj = z_arr + 0.5 * g * (t_arr**2)
        pz = np.polyfit(t_arr, z_adj, 1)
        vz, z0 = pz[0], pz[1]
        
        t_current = t_arr[-1]
        a, b, c = -0.5 * g, vz, z0
        discriminant = (b**2) - (4 * a * c)
        if discriminant < 0: return None, None
            
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        t_land = max(t1, t2)
        
        if t_land < t_current or (t_land - t_current) > 2.5: return None, None
        return float(x0 + vx * t_land), float(y0 + vy * t_land)

# ==========================================
# ⚙️ SYSTEM CONFIGURATIONS
# ==========================================
WEBCAM_ID = 0
CAMERA_HEIGHT_CM = 121.4
CAMERA_PITCH_DEG = 5.0
ORIGIN_Y_DISTANCE_CM = 323.3
ORIGIN_X_OFFSET_CM = 0.0
ESP32_IPS = ["192.168.137.33"]
ESP32_PORT = 12345
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
CONFIDENCE_THRESHOLD = 0.5

# 1. โหลดข้อมูล Stereo Calibration (Extrinsics & Intrinsics)
if not os.path.exists("stereo_calib_data.npz"):
    print("❌ ไม่พบไฟล์ stereo_calib_data.npz กรุณารัน stereo_calibrate.py ก่อน!")
    exit()

calib_data = np.load("stereo_calib_data.npz")
mtx_oak = calib_data['mtx_oak']
mtx_web = calib_data['mtx_web']
R = calib_data['R']
T = calib_data['T']

# คำนวณ Projection Matrices
# P1 สำหรับกล้องหลัก (OAK-D) ถือว่าเป็นจุดศูนย์กลาง (Origin) ไม่มี Rotation/Translation
P1 = mtx_oak @ np.hstack((np.eye(3, 3), np.zeros((3, 1))))
# P2 สำหรับกล้องรอง (Webcam) ที่มี Rotation และ Translation สัมพันธ์กับกล้องหลัก
P2 = mtx_web @ np.hstack((R, T))

print("✅ โหลด Calibration Data สำเร็จ เตรียมทำ 3D Triangulation")

# 2. โหลดโมเดล YOLOv8
model_path = r"C:\Users\punna\OneDrive\Documents\runs\detect\redball_model\weights\best.engine"
model = YOLO(model_path)

# 3. เตรียมกล้อง Webcam
cap = cv2.VideoCapture(WEBCAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)

# 4. เตรียมกล้อง OAK-D (RGB Only, เลิกใช้ Stereo Depth เดิม)
pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 3) # 640x360
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setFps(30)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.isp.link(xoutRgb.input)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
esp32_sender = UDPSender(ESP32_IPS, ESP32_PORT)

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    kf = BallTrackerKF()
    
    landing_process_count = 0
    last_valid_landing = None
    landing_sent = False
    MIN_PROCESS_COUNT = 3
    is_calibrated = False
    latest_raw_depth = 0.0
    latest_raw_width = 0.0
    FRAME_SKIP = 2
    raw_frame_count = 0
    is_ball_thrown = False
    last_throw_time = 0
    missing_frames = 0
    
    latest_oak_frame = None
    print("🚀 เริ่มระบบ Dual Camera Triangulation (กด 'q' เพื่อออก)")
    
    while True:
        inRgb = qRgb.tryGet()
        if inRgb is not None:
            latest_oak_frame = inRgb.getCvFrame()
            
        ret, frame_web = cap.read()
        
        if latest_oak_frame is not None and ret and frame_web is not None:
            # ใช้ YOLO หากล่องทีละภาพ (เพราะ TensorRT Engine ที่แปลงไว้รองรับ Batch Size = 1 เท่านั้น)
            frame_web_resized = cv2.resize(frame_web, (640, 360))
            res_oak = model.predict(latest_oak_frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device=DEVICE, half=USE_HALF)[0]
            res_web = model.predict(frame_web_resized, conf=CONFIDENCE_THRESHOLD, verbose=False, device=DEVICE, half=USE_HALF)[0]
            
            ball_oak = [box for box in res_oak.boxes if int(box.cls[0].item()) == 0]
            ball_web = [box for box in res_web.boxes if int(box.cls[0].item()) == 0]
            
            measured_xyz = None
            annotated_oak = latest_oak_frame.copy()
            annotated_web = frame_web_resized.copy()
            
            if len(ball_oak) > 0 and len(ball_web) > 0:
                missing_frames = 0
                box_oak = sorted(ball_oak, key=lambda x: x.conf[0].item(), reverse=True)[0]
                box_web = sorted(ball_web, key=lambda x: x.conf[0].item(), reverse=True)[0]
                
                # หาจุดกึ่งกลางของกล่องทั้ง 2 กล้อง
                x1_o, y1_o, x2_o, y2_o = map(int, box_oak.xyxy[0])
                cx_o, cy_o = (x1_o + x2_o) / 2.0, (y1_o + y2_o) / 2.0
                
                x1_w, y1_w, x2_w, y2_w = map(int, box_web.xyxy[0])
                cx_w, cy_w = (x1_w + x2_w) / 2.0, (y1_w + y2_w) / 2.0
                
                cv2.rectangle(annotated_oak, (x1_o, y1_o), (x2_o, y2_o), (0, 0, 255), 2)
                cv2.rectangle(annotated_web, (x1_w, y1_w), (x2_w, y2_w), (0, 255, 0), 2)
                
                # +++ 3D Triangulation +++
                pt_oak = np.array([[cx_o, cy_o]], dtype=np.float32).T
                pt_web = np.array([[cx_w, cy_w]], dtype=np.float32).T
                
                # คืนค่าเป็นพิกัด 4 มิติ (X, Y, Z, W) แบบ Homogeneous
                pts4D = cv2.triangulatePoints(P1, P2, pt_oak, pt_web)
                
                # แปลงกลับเป็น 3 มิติปกติ โดยการหารด้วย W (จะได้พิกัดหน่วย มิลลิเมตร เทียบกับกล้อง OAK-D)
                X_mm = float(pts4D[0][0] / pts4D[3][0])
                Y_mm = float(pts4D[1][0] / pts4D[3][0])
                Z_mm = float(pts4D[2][0] / pts4D[3][0])
                
                # แปลงหน่วยเป็นเซนติเมตร
                X_c = X_mm / 10.0
                Y_c = Y_mm / 10.0
                Z_c = Z_mm / 10.0
                
                # หันแกนตามมุมก้มเงยของกล้องหลัก (OAK-D)
                theta = math.radians(CAMERA_PITCH_DEG)
                Y_world_c = Y_c * math.cos(theta) - Z_c * math.sin(theta)
                Z_world_c = Y_c * math.sin(theta) + Z_c * math.cos(theta)
                
                latest_raw_depth = Z_world_c
                latest_raw_width = X_c
                
                X_cm = ORIGIN_Y_DISTANCE_CM - Z_world_c
                Y_cm = X_c - ORIGIN_X_OFFSET_CM
                Z_cm = CAMERA_HEIGHT_CM - Y_world_c
                
                measured_xyz = [X_cm, Y_cm, Z_cm]
                label_text = f"X:{X_cm:.1f} Y:{Y_cm:.1f} Z:{Z_cm:.1f}"
                cv2.putText(annotated_oak, label_text, (x1_o, y1_o - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
            # --- อัพเดต Kalman Filter ---
            raw_frame_count += 1
            if is_calibrated and raw_frame_count % FRAME_SKIP == 0:
                kf.process(measured_xyz)
            landing_pt = kf.predict_landing() if is_calibrated else (None, None)
            
            if is_calibrated and measured_xyz is not None:
                vx, vy, vz = kf.state[3:].flatten()
                
                if (vz > 15 or abs(vx) > 20) and (time.time() - last_throw_time > 1.5):
                    kf.P[3:, 3:] = np.eye(3) * 1000.0
                    kf.reset_history()
                    is_ball_thrown = True
                    landing_sent = False
                    landing_process_count = 0
                    last_throw_time = time.time()
                    
                is_falling = (vz < -5) and is_ball_thrown
                
                if landing_pt[0] is not None and is_falling:
                    clamped_x = max(-50.0, min(50.0, landing_pt[0]))
                    clamped_y = max(-50.0, min(50.0, landing_pt[1]))
                    last_valid_landing = (clamped_x, clamped_y)
                    
                    if raw_frame_count % FRAME_SKIP == 0:
                        landing_process_count += 1
                    
                    if landing_process_count >= MIN_PROCESS_COUNT and not landing_sent:
                        esp32_sender.send_data_binary(last_valid_landing[0], last_valid_landing[1], 0.0)
                        landing_sent = True
                        print(f"🎯 [DUAL CAMERA] ส่งพิกัดตก: X={last_valid_landing[0]:.1f}, Y={last_valid_landing[1]:.1f}")
            else:
                missing_frames += 1
                if missing_frames > 30:
                    landing_process_count = 0
                    landing_sent = False
                    last_valid_landing = None
                    is_ball_thrown = False
                    
            if not is_calibrated:
                cv2.putText(annotated_oak, "PRESS 'z' TO SET ZERO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif landing_pt[0] is not None:
                cv2.putText(annotated_oak, f"Landing X:{landing_pt[0]:.1f} Y:{landing_pt[1]:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            combined = np.hstack((annotated_oak, annotated_web))
            cv2.imshow("Dual Camera Tracking", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('z') and not is_calibrated:
                if latest_raw_depth > 0:
                    ORIGIN_Y_DISTANCE_CM = latest_raw_depth
                    ORIGIN_X_OFFSET_CM = latest_raw_width
                    is_calibrated = True
                    print(f"✅ ตั้งศูนย์สำเร็จ! (X=0, Y=0) ห่างจากกล้อง {ORIGIN_Y_DISTANCE_CM:.1f} cm")

cap.release()
cv2.destroyAllWindows()
esp32_sender.close()
