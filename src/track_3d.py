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
from udp_sender import UDPSender # นำเข้าคลาสส่งข้อมูลไป ESP32
# +++ [H4] นำเข้า BallTrackerKF จากโมดูลกลาง (แชร์กับ track_dual_3d.py) +++
from ball_tracker_kf import BallTrackerKF

# ค้นหาว่ามี GPU ให้ใช้หรือไม่
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
USE_HALF = (DEVICE == 'cuda:0') # ใช้ FP16 เมื่อเป็น GPU เพื่อให้ไวที่สุด

class BallTrackerKF:
    def __init__(self):
        self.state = np.zeros((6, 1)) # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 1000.0
        # +++ ปรับเพิ่ม R (Measurement Noise) เพื่อไม่ให้ KF แกว่งตาม Noise ของกล้อง +++
        self.R = np.diag([50.0, 10.0, 20.0]) # สลับแกน: ตอนนี้แกน X (ความลึก) เป็นแกนที่แกว่งมากที่สุด
        self.Q = np.eye(6) * 0.1
        # +++ จูน Q ใหม่ให้ตามความเร็วได้ไวขึ้น (สำคัญมากตอนปาบอล) +++
        self.Q[3:, 3:] = np.diag([150.0, 150.0, 150.0]) # เพิ่ม Process Noise ของความเร็วให้สูงมากๆ เพื่อให้ไม่หน่วง
        
        # Pre-allocate matrices (Optimization for speed)
        self.F = np.eye(6)
        self.B = np.zeros((6, 1))
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0; self.H[1, 1] = 1.0; self.H[2, 2] = 1.0
        self.I = np.eye(6)
        
        self.last_time = time.time()
        self.is_initialized = False
        
        # +++ เพิ่มระบบเก็บประวัติพิกัดสำหรับ Curve Fitting +++
        self.history = [] # เก็บค่า (t, x, y, z) ย้อนหลัง

    def process(self, measurement=None):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt > 1.0:
            self.is_initialized = False
            
        if not self.is_initialized:
            if measurement is not None:
                self.state[:3] = np.array(measurement).reshape((3, 1))
                self.state[3:] = 0
                self.is_initialized = True
            return self.state
            
        # Predict
        self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt
        
        # เปลี่ยนให้แกน Z เป็นความสูง (ถูกแรงโน้มถ่วงดึงลง)
        self.B[2, 0] = -0.5 * 981 * dt**2 
        self.B[5, 0] = -981 * dt
        
        self.state = np.dot(self.F, self.state) + self.B
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # Update
        if measurement is not None:
            z = np.array(measurement).reshape((3, 1))
            
            # +++ Adaptive Kalman Filter (AKF) +++
            # กล้อง Stereo จะเพี้ยนหนักขึ้นแบบ "ก้าวกระโดด" ถ้าระยะทางไกลขึ้น (Error แปรผันตามระยะทางกำลังสอง)
            # [BUGFIX] ใช้ ORIGIN_Y_DISTANCE_CM แทน hardcoded 323.3 เพื่อให้ Adaptive R ทำงานถูกต้องหลัง Set Zero
            distance_from_camera_cm = ORIGIN_Y_DISTANCE_CM - measurement[0]
            distance_m = abs(distance_from_camera_cm) / 100.0  # ทำเป็นเมตร
            
            # ปรับค่า R อัตโนมัติ: ถ้าอยู่ไกล จะให้ KF สนใจฟิสิกส์มากกว่ากล้อง, ถ้าอยู่ใกล้ จะเชื่อกล้อง 100%
            noise_x = 10.0 + (distance_m ** 2.0) * 15.0 # แกน X (ความลึก) จะเพี้ยนรุนแรงที่สุดเมื่อไกล
            noise_y = 5.0 + (distance_m ** 1.5) * 3.0   # แกน Y (ซ้าย-ขวา)
            noise_z = 5.0 + (distance_m ** 1.5) * 5.0
            
            self.R = np.diag([noise_x, noise_y, noise_z])
            
            y = z - np.dot(self.H, self.state)
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            
            # +++ [H2] Innovation Gating: ปฏิเสธ Outlier ด้วย Mahalanobis Distance +++
            # ถ้าค่าที่วัดได้ห่างจากค่าที่ KF คาดการณ์มากผิดปกติ (chi-squared 3DOF, p=0.001)
            # จะข้ามไม่อัปเดต KF เพื่อป้องกัน KF กระโดดตาม YOLO ที่จับผิดวัตถุ
            mahal_sq = float(np.dot(np.dot(y.T, np.linalg.inv(S)), y))
            if mahal_sq > 16.27:  # chi-squared threshold: 3 DOF, p=0.001
                # Outlier! ข้ามการอัปเดต
                pass
            else:
                K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
                self.state = self.state + np.dot(K, y)
                self.P = np.dot(self.I - np.dot(K, self.H), self.P)
            
        # เก็บประวัติพิกัดเพื่อนำไปทำ Curve Fitting (เก็บ 20 เฟรมย้อนหลัง)
        self.history.append((self.last_time, self.state[0, 0], self.state[1, 0], self.state[2, 0]))
        if len(self.history) > 20:
            self.history.pop(0)
            
        return self.state
        
    def reset_history(self):
        self.history = []
        
    def predict_landing(self):
        # ต้องมีข้อมูลอย่างน้อย 5 เฟรมถึงจะลากเส้น Curve ได้แม่นยำ
        if not self.is_initialized or len(self.history) < 5:
            return None, None
            
        # ดึงประวัติพิกัดออกมาแยกแกน
        t_arr = np.array([pt[0] for pt in self.history])
        t_arr = t_arr - t_arr[0] # เริ่มนับเวลาจาก 0
        x_arr = np.array([pt[1] for pt in self.history])
        y_arr = np.array([pt[2] for pt in self.history])
        z_arr = np.array([pt[3] for pt in self.history])
        
        # ถ้าระดับความสูงล่าสุดติดลบ (อยู่ใต้พื้น) แปลว่าตกไปแล้ว
        if z_arr[-1] <= 0:
            return float(x_arr[-1]), float(y_arr[-1])
            
        # +++ 1. SVD (Singular Value Decomposition) ลด Noise แนวขวาง +++
        # ลูกบอลที่โยนจะเดินทางเป็น "เส้นตรง" เมื่อมองจากมุมท็อป แต่กล้องมักจะจับภาพแกว่งซ้ายขวา (Zigzag)
        mean_x = np.mean(x_arr)
        mean_y = np.mean(y_arr)
        
        # หาค่าความเบี่ยงเบนจากจุดศูนย์กลาง (Mean Centering)
        M = np.vstack((x_arr - mean_x, y_arr - mean_y)).T
        
        # ทำ SVD เพื่อดึง Principal Component (แกนหลักที่บอลเคลื่อนที่จริงๆ)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        dir_vector = Vt[0] # ทิศทางการเคลื่อนที่หลัก (เวกเตอร์ที่ขนานกับวิถีลูกบอล)
        
        # Project (ฉาย) พิกัดกลับลงไปบนเส้นตรงหลัก เพื่อลบ Noise ที่แตกแถวออกนอกเส้นทางทิ้ง 100%
        projected = np.dot(M, dir_vector)
        clean_xy = np.outer(projected, dir_vector)
        
        clean_x = clean_xy[:, 0] + mean_x
        clean_y = clean_xy[:, 1] + mean_y
        
        # +++ 2. Curve Fitting แกน X และ Y ด้วยพิกัดที่คลีนแล้ว +++
        px = np.polyfit(t_arr, clean_x, 1) # สมการ: X(t) = vx*t + x0
        py = np.polyfit(t_arr, clean_y, 1) # สมการ: Y(t) = vy*t + y0
        vx, x0 = px[0], px[1]
        vy, y0 = py[0], py[1]
        
        # +++ 3. Curve Fitting แกน Z (พาราโบลา + แรงโน้มถ่วงแบบฟิกซ์ค่า) +++
        g = 981.0
        # ปรับสมการเป็น Linear: Z(t) + 0.5*g*t^2 = vz*t + z0
        z_adj = z_arr + 0.5 * g * (t_arr**2)
        pz = np.polyfit(t_arr, z_adj, 1)
        vz, z0 = pz[0], pz[1]
        
        # เวลาปัจจุบัน (เฟรมล่าสุด)
        t_current = t_arr[-1]
        
        # หา t ที่ตกถึงพื้น Z(t) = 0 => -0.5*g*t^2 + vz*t + z0 = 0
        a = -0.5 * g
        b = vz
        c = z0
        discriminant = (b**2) - (4 * a * c)
        
        if discriminant < 0:
            return None, None
            
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        t_land = max(t1, t2)
        
        # ถ้าพยากรณ์ว่าตกก่อนเวลาปัจจุบัน หรือลอยนานเกินไป ถือว่า Noise ผิดปกติ
        if t_land < t_current or (t_land - t_current) > 2.5:
            return None, None
            
        # +++ [M3] ชดเชย Air Drag: ลูกบอลจริงตกใกล้กว่าที่คำนวณ ~8% +++
        sim_x = (x0 + vx * t_land) * DRAG_CORRECTION
        sim_y = (y0 + vy * t_land) * DRAG_CORRECTION
        
        return float(sim_x), float(sim_y)

    def predict_landing_fast(self):
        """⚡ คำนวณจุดตกแบบเร็ว ใช้ KF State โดยตรง ไม่ต้องรอสะสม History
        เหมาะสำหรับช่วง 0-170ms แรกหลังตรวจจับการโยน (ก่อนที่จะมี History พอสำหรับ SVD)
        """
        if not self.is_initialized:
            return None, None
        
        x, y, z = self.state[:3].flatten()
        vx, vy, vz = self.state[3:].flatten()
        
        # ถ้าอยู่ใต้พื้นแล้ว ให้คืนตำแหน่งปัจจุบัน
        if z <= 0:
            return float(x), float(y)
        
        # แก้สมการพาราโบลา: z + vz*t - 0.5*g*t² = 0
        g = 981.0
        a = -0.5 * g
        b = vz
        c = z
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None, None
        
        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        t2 = (-b - sqrt_disc) / (2.0 * a)
        t_land = max(t1, t2)
        
        if t_land <= 0 or t_land > 3.0:
            return None, None
        
        # +++ [M3] ชดเชย Air Drag +++
        pred_x = float(x + vx * t_land) * DRAG_CORRECTION
        pred_y = float(y + vy * t_land) * DRAG_CORRECTION
        return pred_x, pred_y

# ==========================================
# ⚙️ SYSTEM CONFIGURATIONS
# ==========================================
CAMERA_HEIGHT_CM = 121.4      # ความสูงของกล้องจากพื้น (เซนติเมตร)
CAMERA_PITCH_DEG = 0.0        # มุมก้มเงยของกล้อง (ก้มลงเป็นบวก, เงยขึ้นเป็นลบ)
ORIGIN_Y_DISTANCE_CM = 323.3  # +++ ระยะห่างจากกล้องถึงจุด (0,0) ของหุ่นยนต์ (ค่าเริ่มต้น จะถูกเขียนทับตอนกด z) +++
ORIGIN_X_OFFSET_CM = 0.0      # +++ จุดกึ่งกลางซ้ายขวา (ค่าเริ่มต้น จะถูกเขียนทับตอนกด z) +++
ESP32_IPS = ["192.168.137.33"]   # IPs ของ ESP32 (รองรับหลายตัว)
ESP32_PORT = 12345            # Port ของ ESP32
UDP_IP = "127.0.0.1"          # IP สำหรับส่งวาดกราฟ (Local)
UDP_PORT = 5005               # Port สำหรับส่งวาดกราฟ
CONFIDENCE_THRESHOLD = 0.5    # ความแม่นยำขั้นต่ำของ YOLO
DRAG_CORRECTION = 0.92        # +++ [M3] ค่าชดเชยแรงต้านอากาศ (ลูกบอล ~6cm, Cd≈0.47) ลดระยะทาง XY ~8% +++
# ==========================================

# ตั้งค่า UDP Socket สำหรับส่งข้อมูล 3D ไปวาดกราฟ
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ตั้งค่า UDP Sender สำหรับส่งไป ESP32
# อย่าลืมเปลี่ยน IP ตรงนี้ให้ตรงกับที่แสดงใน Serial Monitor ของ ESP32
esp32_sender = UDPSender(ESP32_IPS, ESP32_PORT)

# +++ สร้างไฟล์ Log สำหรับบันทึกจุดตก (ตั้งชื่อไฟล์ตามวันเวลาที่รัน) +++
LOG_FILE = f"landing_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("Timestamp,Clamped_X,Clamped_Y,Raw_X,Raw_Y,Vx,Vy,Vz,Elapsed_ms\n")
    print(f"📝 สร้างไฟล์ Log: {LOG_FILE}")

# 1. โหลดโมเดล YOLOv8 และบังคับใช้ GPU ถ้ารองรับ
model_path = r"C:\Users\punna\OneDrive\Documents\runs\detect\redball_model\weights\best.engine"
model = YOLO(model_path)
# model.to(DEVICE) # นำออกเพราะ TensorRT ไม่ต้องใช้คำสั่งนี้ (กำหนด device ตอน predict ไปแล้ว)

# 2. ตั้งค่า DepthAI (OAK-D Lite) Pipeline
pipeline = dai.Pipeline()

# 2.1 กล้องสี RGB
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 3)  # แปลงขนาดเป็น 1920//3 = 640, 1080//3 = 360
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setFps(60)         # +++ อัพเกรดเป็น 60 FPS เพื่อการตอบสนองขั้นสุด +++

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.isp.link(xoutRgb.input)

# 2.2 กล้องขาวดำซ้ายและขวา (สำหรับ Depth)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setFps(60)       # +++ อัพเกรดเป็น 60 FPS +++

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setFps(60)      # +++ อัพเกรดเป็น 60 FPS +++

# 2.3 ตัวรวมสัญญาณ 3 มิติ (Stereo Depth)
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY) # ใช้โหมดแม่นยำสูง

# +++ เพิ่มตัวกรอง (Filters) เพื่อความแม่นยำสูงสุด +++
stereo.setLeftRightCheck(True) # ตรวจสอบพิกเซลซ้ายขวาให้ตรงกัน (ลด Noise สัญญาณหลอก)
stereo.setSubpixel(True)       # คำนวณความลึกระดับทศนิยมพิกเซล (แม่นยำขึ้นมาก)

# +++ เปิดใช้งาน Spatial & Temporal Filters ของชิปกล้องเพื่อลด Noise ใหลื่นไหล +++
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
# เปิดใช้งาน Decimation Filter (ลดความละเอียด Depth เพื่อยุบรวม Pixel ที่เป็น Noise เข้าด้วยกัน)
# config.postProcessing.decimationFilter.decimationFactor = 2

stereo.initialConfig.set(config)

# เปิดใช้งาน Median Filter ระดับ 7x7 (เกลี่ยพิกเซลที่กระโดดผิดปกติทิ้ง)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

# จัดตำแหน่งภาพ 3 มิติให้ตรงกับกล้องสี และบังคับขนาดภาพให้ตรงกัน
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(camRgb.getIspWidth(), camRgb.getIspHeight())

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)


# ฟังก์ชัน: หาระยะ Z
def get_distance_from_center(depth_map, x1, y1, x2, y2):
    # ตีกรอบ (ROI) เป็น 30% ตรงกลางของกล่อง YOLO เพื่อเลี่ยงขอบลูกบอลและฉากหลัง
    w = x2 - x1
    h = y2 - y1
    roi_x1 = int(x1 + w * 0.35)
    roi_x2 = int(x2 - w * 0.35)
    roi_y1 = int(y1 + h * 0.35)
    roi_y2 = int(y2 - h * 0.35)
    
    # ป้องกันกรอบเล็กเกินไปจนเกิด Error
    if roi_x2 <= roi_x1: roi_x2 = roi_x1 + 1
    if roi_y2 <= roi_y1: roi_y2 = roi_y1 + 1
    
    # ป้องกันกรอบหลุดออกนอกจอ
    roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
    roi_x2, roi_y2 = min(depth_map.shape[1], roi_x2), min(depth_map.shape[0], roi_y2)
    
    roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
    valid_depths = roi[roi > 0]
    
    if len(valid_depths) == 0:
        # +++ Fallback: ถ้าหากลางกล่องไม่เจอพิกัดความลึก (เช่น ตรงกลางหุ่นเป็นช่องว่าง หรือสีดำเงา) ให้ขยายกรอบหาใหม่เป็น 80% +++
        roi_x1, roi_x2 = int(x1 + w * 0.10), int(x2 - w * 0.10)
        roi_y1, roi_y2 = int(y1 + h * 0.10), int(y2 - h * 0.10)
        roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
        roi_x2, roi_y2 = min(depth_map.shape[1], roi_x2), min(depth_map.shape[0], roi_y2)
        
        roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
        valid_depths = roi[roi > 0]

    if len(valid_depths) > 0:
        # ใช้เทคนิค เรียงลำดับพิกเซล และคัดเอาแค่ 50% แรกที่ "ใกล้กล้องที่สุด" 
        # (เพราะวัตถุจะอยู่ใกล้กว่าฉากหลังเสมอ)
        sorted_depths = np.sort(valid_depths)
        half_idx = max(1, len(sorted_depths) // 2)
        return np.median(sorted_depths[:half_idx]) # คืนค่า Z เป็น มิลลิเมตร
    return 0


# 3. รัน OAK-D Lite
with dai.Device(pipeline) as device:
    print("เริ่มเชื่อมต่อกล้อง กำลังดึงค่า Camera Intrinsics...")
    
    # อ่านค่า Calibration เพื่อดึงตัวแปรมาคำนวณแกน X และ Y
    calibData = device.readCalibration()
    
    # ขอค่า Intrinsics ของกล้องสี CAM_A ที่ขนาด 640x360
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, camRgb.getIspWidth(), camRgb.getIspHeight())
    camera_matrix = np.array(intrinsics, dtype=np.float32)
    dist_coeffs = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), dtype=np.float32)
    
    # ดึงค่าจุดโฟกัสและจุดกึ่งกลาง (Focal Length / Principal Point)
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    
    print(f"ค่าของเลนส์กล้อง: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print("เริ่มแสดงผล (กด 'q' เพื่อออก)")
    
    # maxSize=1 ดึงเฉพาะ Frame ล่าสุด ตัด Delay ที่เกิดจาก Buffer ของกล้อง
    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    
    kf = BallTrackerKF(drag_correction=DRAG_CORRECTION)  # +++ [H4] ใช้โมดูลกลาง +++
    
    # +++ ระบบ "ส่งหลังคำนวณครบตามจำนวนครั้งที่กำหนด" +++
    landing_process_count = 0   # นับจำนวนรอบที่ผ่าน KF แล้วคำนวณจุดตกได้สำเร็จ
    last_valid_landing = None   # เก็บจุดตกล่าสุดที่คำนวณได้
    landing_sent = False        # ส่งไป ESP32 แล้วหรือยัง (กันส่งซ้ำ)
    MIN_PROCESS_COUNT = 2       # +++ ลดเหลือ 2 รอบเพื่อส่งจุดตกภายใน 0.3 วินาที +++
    
    # +++ ระบบ Calibration (Set Zero) +++
    is_calibrated = False       # เริ่มต้นในโหมดตั้งศูนย์
    latest_raw_depth = 0.0      # เก็บค่าระยะกล้องเพื่อทำ Set Zero
    latest_raw_width = 0.0      # เก็บค่าความกว้างเพื่อทำ Set Zero
    
    # +++ ระบบข้ามเฟรม (Frame Skipping) +++
    FRAME_SKIP = 1              # +++ ลดเป็น 1 (ป้อน KF ทุกเฟรม) เพื่อส่งจุดตกเร็วขึ้น +++
    raw_frame_count = 0         # ตัวนับเฟรมดิบ
    
    # +++ ระบบป้องกันการแกว่งมือ (Throw Detection) +++
    is_ball_thrown = False      # ตรวจจับว่าลูกบอลหลุดจากมือหรือยัง
    last_throw_time = 0         # ตัวจับเวลาป้องกันการโยนซ้ำรัวๆ
    throw_detect_time = 0       # +++ เวลาที่ตรวจจับการโยน (สำหรับจับเวลา Deadline) +++
    LANDING_DEADLINE_SEC = 0.28 # +++ Deadline: ต้องส่งจุดตกภายใน 280ms หลังโยน +++
    
    missing_frames = 0          # จำนวนเฟรมที่หากล่อง YOLO ไม่เจอ (ป้องกันสถานะหลุดเวลาภาพเบลอ)
    
    # +++ ตั้งค่าระบบบันทึกวิดีโอ (Video Recording) +++
    is_recording = False
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # ใช้ MP4 Codec
    
    while True:
        inRgb = qRgb.get()
        inDepth = qDepth.tryGet()
        
        if inRgb is not None:
            frame = inRgb.getCvFrame()
            
            depth_frame = None
            if inDepth is not None:
                depth_frame = inDepth.getFrame()
            
            # YOLOv8 (Acceleration with FP16 + GPU Core)
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device=DEVICE, half=USE_HALF)
            annotated_frame = frame.copy()
            
            measured_xyz = None
            
            # กรองเฉพาะกล่องที่เป็น "ลูกบอลสีแดง" (Class 0) ไม่เอา "หุ่นยนต์" (Class 1) มาใช้คำนวณจุดตก
            ball_boxes = [box for box in results[0].boxes if int(box.cls[0].item()) == 0]
            
            if len(ball_boxes) > 0:
                missing_frames = 0
                # ดึงกล่องและจัดเรียงตามค่าความมั่นใจ (Confidence) จากมากไปน้อย
                boxes = sorted(ball_boxes, key=lambda x: x.conf[0].item(), reverse=True)
                best_box = boxes[0]
                
                # ใช้เฉพาะกล่องที่มั่นใจที่สุด
                for box in [best_box]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    label_text = "N/A"
                    if depth_frame is not None:
                        Z_mm = get_distance_from_center(depth_frame, x1, y1, x2, y2)
                        
                        if Z_mm > 0:
                            # +++ แก้ไข Distortion (เลนส์โค้ง) ด้วยการ Undistort เฉพาะจุด +++
                            pts = np.array([[[float(center_x), float(center_y)]]], dtype=np.float32)
                            undistorted_pts = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
                            u_cx, u_cy = undistorted_pts[0][0]

                            # คำนวณแกน X และ Y ด้วยพิกัดที่แก้ Distortion แล้ว
                            X_mm = (u_cx - cx) * Z_mm / fx
                            Y_mm = (u_cy - cy) * Z_mm / fy
                            
                            # ความสูงกล้องจากพื้นถูกย้ายไปตั้งค่าที่ SYSTEM CONFIGURATIONS ด้านบนแล้ว
                            pass
                            
                            # แปลงเป็นหน่วย เซนติเมตร (cm) ในมุมมองของเลนส์กล้อง
                            X_c = X_mm / 10.0
                            Y_c = Y_mm / 10.0
                            Z_c = Z_mm / 10.0
                            
                            # +++ ชดเชยมุมก้มเงยของกล้อง (Extrinsic Rotation Matrix) +++
                            # เพื่อแปลงแกนกล้องให้ขนานกับแกนโลก (World Coordinates) 100%
                            theta = math.radians(CAMERA_PITCH_DEG)
                            
                            # หมุนรอบแกน X (ชดเชยกล้องก้ม)
                            Y_world_c = Y_c * math.cos(theta) + Z_c * math.sin(theta)
                            Z_world_c = -Y_c * math.sin(theta) + Z_c * math.cos(theta)
                            
                            # เก็บค่าดิบไว้เพื่อทำ Set Zero ตอนกด z
                            latest_raw_depth = Z_world_c
                            latest_raw_width = X_c
                            
                            # +++ สลับแกนให้เข้ากับระบบของหุ่นยนต์ใหม่ +++
                            # ให้ X เป็นความลึก (Depth) สลับบวกลบ: ใกล้กล้องเป็นบวก, วิ่งไปหาหุ่นยนต์จะลดลงจนเป็น 0
                            X_cm = ORIGIN_Y_DISTANCE_CM - Z_world_c
                            
                            # ให้ Y เป็น ซ้าย-ขวา (Left-Right)
                            Y_cm = X_c - ORIGIN_X_OFFSET_CM
                            
                            # ให้ Z เป็นความสูง (Height) (ชี้ขึ้นฟ้าเป็นบวก)
                            # (ในโลกของกล้อง Y ชี้ลงพื้น เราจึงเอาไปลบออกจากความสูงของกล้อง)
                            Z_cm = CAMERA_HEIGHT_CM - Y_world_c
                            
                            measured_xyz = [X_cm, Y_cm, Z_cm]
                            
                            # (สามารถแสดงค่า KF แทนได้ถ้ารำคาญตัวเลขกระพริบ)
                            label_text = f"X: {X_cm:.1f}cm, Y: {Y_cm:.1f}cm, Z: {Z_cm:.1f}cm"
                    
                    # วาดสีกล่อง
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # วาดจุดกึ่งกลาง (Center Point)
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
                    
                    # ใส่ตัวหนังสือพิกัด 
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # +++ เริ่มต้นเพิ่มระบบแสดงหุ่นยนต์ (Class 1) +++
            robot_boxes = [box for box in results[0].boxes if int(box.cls[0].item()) == 1]
            if len(robot_boxes) > 0:
                # เลือกกล่องหุ่นยนต์ที่มั่นใจที่สุด
                r_box = sorted(robot_boxes, key=lambda x: x.conf[0].item(), reverse=True)[0]
                rx1, ry1, rx2, ry2 = map(int, r_box.xyxy[0])
                r_cx = (rx1 + rx2) // 2
                r_cy = (ry1 + ry2) // 2
                
                r_label = "ROBOT N/A"
                if depth_frame is not None:
                    r_Z_mm = get_distance_from_center(depth_frame, rx1, ry1, rx2, ry2)
                    if r_Z_mm > 0:
                        r_pts = np.array([[[float(r_cx), float(r_cy)]]], dtype=np.float32)
                        r_undistorted = cv2.undistortPoints(r_pts, camera_matrix, dist_coeffs, P=camera_matrix)
                        ru_cx, ru_cy = r_undistorted[0][0]
                        
                        r_X_c = ((ru_cx - cx) * r_Z_mm / fx) / 10.0
                        r_Y_c = ((ru_cy - cy) * r_Z_mm / fy) / 10.0
                        r_Z_c = r_Z_mm / 10.0
                        
                        theta = math.radians(CAMERA_PITCH_DEG)
                        r_Y_world = r_Y_c * math.cos(theta) + r_Z_c * math.sin(theta)
                        r_Z_world = -r_Y_c * math.sin(theta) + r_Z_c * math.cos(theta)
                        
                        r_X_cm = ORIGIN_Y_DISTANCE_CM - r_Z_world
                        r_Y_cm = r_X_c - ORIGIN_X_OFFSET_CM
                        r_label = f"ROBOT X:{r_X_cm:.0f} Y:{r_Y_cm:.0f}"
                
                # วาดกล่องหุ่นยนต์สีฟ้า (BGR: 255, 0, 0)
                cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, r_label, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # +++ สิ้นสุดการวาดหุ่นยนต์ +++
            
            # +++ อัพเดต Kalman Filter แบบข้ามเฟรม (เพื่อให้เห็นระยะทางชัดเจน ป้องกันการคำนวณความเร็วผิดพลาด) +++
            raw_frame_count += 1
            if is_calibrated and raw_frame_count % FRAME_SKIP == 0:
                kf.process(measured_xyz, origin_distance_cm=ORIGIN_Y_DISTANCE_CM)
            # +++ พยากรณ์จุดตกแบบ Hybrid: ใช้ Fast Mode ถ้า History ไม่พอ, ใช้ SVD Mode ถ้ามีพอ +++
            if is_calibrated:
                if len(kf.history) >= 5:
                    landing_pt = kf.predict_landing()       # SVD + Curve Fit (แม่นยำกว่า)
                else:
                    landing_pt = kf.predict_landing_fast()   # ⚡ KF State ตรง (เร็วกว่า)
            else:
                landing_pt = (None, None)
            
            # ส่งข้อมูลไปยัง plot_3d.py รวมพิกัดปัจจุบันและจุดตกที่พยากรณ์ได้ รวมถึงความเร็วจาก KF
            if is_calibrated and measured_xyz is not None:
                vx, vy, vz = kf.state[3:].flatten()
                if landing_pt[0] is not None:
                    msg = f"{measured_xyz[0]:.2f},{measured_xyz[1]:.2f},{measured_xyz[2]:.2f},{landing_pt[0]:.2f},{landing_pt[1]:.2f},{vx:.2f},{vy:.2f},{vz:.2f}"
                else:
                    msg = f"{measured_xyz[0]:.2f},{measured_xyz[1]:.2f},{measured_xyz[2]:.2f},None,None,{vx:.2f},{vy:.2f},{vz:.2f}"
                try:
                    sock.sendto(msg.encode('utf-8'), (UDP_IP, UDP_PORT))
                    
                    # 🎯 ระบบดักจับการโยน (Throw Detection)
                    # ถ้ายกบอลขึ้น (กำลังจะโยน) และผ่านไปแล้วอย่างน้อย 1.5 วินาทีจากการโยนครั้งก่อน
                    if (vz > 15 or abs(vx) > 20) and (time.time() - last_throw_time > 1.5):
                        throw_detect_time = time.time()  # +++ จับเวลาเริ่มต้น Deadline +++
                        print(f"🚀 [THROW DETECTED] ตรวจพบการโยนใหม่! (Vx:{vx:.1f}, Vz:{vz:.1f}) | เริ่มจับเวลา 0.3s")
                        # บังคับให้ KF ตื่นตัวทันทีตอนโยน
                        kf.P[3:, 3:] = np.eye(3) * 1000.0
                        # ล้างประวัติ Curve Fit เพื่อเริ่มวาดเส้นวิถีลูกโค้งเส้นใหม่
                        kf.reset_history()
                        
                        is_ball_thrown = True
                        landing_sent = False # ปลดล็อคการส่งข้อมูล
                        landing_process_count = 0
                        last_throw_time = time.time()
                        
                    # 🎯 จะนับว่ากำลังตก: ผ่อนเกณฑ์จาก vz<-5 เป็น vz<-2 เพื่อตรวจจับเร็วขึ้น
                    is_falling = (vz < -2) and is_ball_thrown
                    
                    # +++ ระบบ Deadline: ถ้าเกิน 280ms หลังโยน และมีจุดตกที่คำนวณได้ ส่งทันที! +++
                    time_since_throw = time.time() - throw_detect_time if throw_detect_time > 0 else 0
                    deadline_expired = (time_since_throw >= LANDING_DEADLINE_SEC) and is_ball_thrown and not landing_sent
                    
                    if landing_pt[0] is not None and (is_falling or deadline_expired):
                        # จำกัดค่า (Clamp) ให้อยู่ในพื้นที่ทำงาน +-50 cm
                        clamped_x = max(-50.0, min(50.0, landing_pt[0]))
                        clamped_y = max(-50.0, min(50.0, landing_pt[1]))
                        
                        last_valid_landing = (clamped_x, clamped_y)
                        
                        # เพิ่มตัวนับเฉพาะเฟรมที่ KF ได้คำนวณจริงๆ
                        if raw_frame_count % FRAME_SKIP == 0:
                            landing_process_count += 1
                        
                        # ส่งเมื่อ: (1) ครบรอบขั้นต่ำ หรือ (2) Deadline หมดเวลา
                        should_send = (landing_process_count >= MIN_PROCESS_COUNT) or deadline_expired
                        
                        if should_send and not landing_sent:
                            elapsed_ms = time_since_throw * 1000
                            esp32_sender.send_data_binary(last_valid_landing[0], last_valid_landing[1], 0.0)
                            landing_sent = True
                            mode_str = "DEADLINE" if deadline_expired else f"KF x{landing_process_count}"
                            pred_mode = "SVD" if len(kf.history) >= 5 else "FAST"
                            print(f"\n🎯 [SENT in {elapsed_ms:.0f}ms] [{mode_str}] [{pred_mode}] -> ESP32: X={last_valid_landing[0]:.1f}, Y={last_valid_landing[1]:.1f}")
                            print(f"   🔍 [ค่าดิบก่อน Clamp] X: {landing_pt[0]:.1f}, Y: {landing_pt[1]:.1f} | ความเร็ว Vx: {vx:.1f}, Vy: {vy:.1f}, Vz: {vz:.1f}")
                            
                            # +++ บันทึกข้อมูลการโยนรอบนี้ลงไฟล์ CSV +++
                            with open(LOG_FILE, "a", encoding="utf-8") as f:
                                timestamp_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                f.write(f"{timestamp_str},{last_valid_landing[0]:.2f},{last_valid_landing[1]:.2f},{landing_pt[0]:.2f},{landing_pt[1]:.2f},{vx:.2f},{vy:.2f},{vz:.2f},{elapsed_ms:.0f}\n")
                            
                        # แจ้งเตือนบนหน้าจอถ้าจุดตกจริงอยู่นอกพื้นที่ แต่ยังให้หุ่นยนต์วิ่งไปขอบสุด
                        if not (-50 <= landing_pt[0] <= 50 and -50 <= landing_pt[1] <= 50):
                            cv2.putText(annotated_frame, f"OUT OF BOUNDS (CLAMPED)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                except Exception as e:
                    cv2.putText(annotated_frame, f"NET ERROR: {str(e)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print(f"[ERROR] UDP Send Failed: {e}")
            else:
                missing_frames += 1
                
                # +++ [M2] ลดจาก 30 เหลือ 15 เฟรม (~250ms ที่ 60fps) เพื่อรีเซ็ตเร็วขึ้น +++
                if missing_frames > 15:
                    landing_process_count = 0
                    landing_sent = False
                    last_valid_landing = None
                    is_ball_thrown = False
                
                # +++ [M2] Z-based Termination: ถ้า KF บอกว่าบอลอยู่ต่ำกว่า 5cm แสดงว่าตกถึงพื้นแล้ว +++
                if kf.is_initialized and kf.state[2, 0] < 5.0 and is_ball_thrown:
                    print("🏁 [บอลตกถึงพื้น] KF Z < 5cm -> รีเซ็ตสถานะ")
                    is_ball_thrown = False
            
            if not is_calibrated:
                cv2.putText(annotated_frame, "CALIBRATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(annotated_frame, "Place red ball at Robot Center (0,0)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, "Press 'z' to SET ZERO", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif landing_pt[0] is not None:
                lx, ly = landing_pt
                cv2.putText(annotated_frame, f"Pred Land X:{lx:.1f} Y:{ly:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                kx, ky, kz = kf.state[:3].flatten()
                cv2.putText(annotated_frame, f"KF X:{kx:.1f} Y:{ky:.1f} Z:{kz:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                # +++ แสดงความเร็วบนหน้าจอให้เห็นชัดๆ +++
                _vx, _vy, _vz = kf.state[3:].flatten()
                cv2.putText(annotated_frame, f"VEL Vx:{_vx:.1f} Vy:{_vy:.1f} Vz:{_vz:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # +++ ระบบบันทึกวิดีโอ +++
            if is_recording:
                # วาดไฟ REC สีแดงกะพริบที่มุมจอ (แสดงแค่บนจอ ไม่บันทึกลงวิดีโอ)
                if int(time.time() * 2) % 2 == 0:
                    cv2.circle(annotated_frame, (30, 95), 8, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "REC", (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # เขียนรูปลงไฟล์วิดีโอ (ใช้ frame ต้นฉบับที่ไม่มีข้อความและกล่อง)
                if video_writer is not None:
                    video_writer.write(frame)

            cv2.imshow("Red Ball 3D Map (X, Y, Z)", annotated_frame)
            
            if depth_frame is not None:
                if depth_frame.shape[0] > 0 and depth_frame.shape[1] > 0:
                    depth_rendered = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depth_rendered = cv2.equalizeHist(depth_rendered)
                    depth_rendered = cv2.applyColorMap(depth_rendered, cv2.COLORMAP_JET)
                    cv2.imshow("Depth Heatmap", depth_rendered)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z') and not is_calibrated:
            # +++ ดำเนินการ Set Zero +++
            if latest_raw_depth > 0:
                ORIGIN_Y_DISTANCE_CM = latest_raw_depth
                ORIGIN_X_OFFSET_CM = latest_raw_width
                is_calibrated = True
                print(f"\n✅ [SET ZERO COMPLETE] ระบบตั้งศูนย์อัตโนมัติสำเร็จ!")
                print(f"   ระยะความลึกหุ่นยนต์ (X=0): {ORIGIN_Y_DISTANCE_CM:.1f} cm")
                print(f"   ระยะซ้ายขวาหุ่นยนต์ (Y=0): {ORIGIN_X_OFFSET_CM:.1f} cm")
            else:
                print("\n⚠️ [SET ZERO FAILED] ไม่สามารถตั้งศูนย์ได้ กรุณาวางลูกบอลให้กล้องเห็นชัดเจน")
        elif key == ord('r'):
            # กดตัว 'r' เพื่อสลับเริ่ม/หยุดการอัดภาพ
            is_recording = not is_recording
            if is_recording:
                h, w = frame.shape[:2]
                filename = f"track_record_{int(time.time())}.mp4"
                video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
                print(f"\n[RECORD] เริ่มบันทึกวิดีโอลงไฟล์: {filename}")
            else:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("\n[RECORD] หยุดบันทึกและเซฟไฟล์วิดีโอเรียบร้อยแล้ว")

if video_writer is not None:
    video_writer.release() # ป้องกันไฟล์เสียถ้ากำลังอัดอยู่แล้วกดปิดโปรแกรม

cv2.destroyAllWindows()
esp32_sender.close()
