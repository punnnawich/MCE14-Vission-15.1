"""
==========================================
  BallTrackerKF — Adaptive Kalman Filter + Landing Predictor
  โมดูลกลางสำหรับแชร์ระหว่าง track_3d.py และ track_dual_3d.py
==========================================
"""
import numpy as np
import time
import math


class BallTrackerKF:
    def __init__(self, drag_correction=0.92):
        self.state = np.zeros((6, 1)) # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 1000.0
        # ปรับเพิ่ม R (Measurement Noise) เพื่อไม่ให้ KF แกว่งตาม Noise ของกล้อง
        self.R = np.diag([50.0, 10.0, 20.0]) # แกน X (ความลึก) แกว่งมากที่สุด
        self.Q = np.eye(6) * 0.1
        # จูน Q ให้ตามความเร็วได้ไวขึ้น (สำคัญมากตอนปาบอล)
        self.Q[3:, 3:] = np.diag([150.0, 150.0, 150.0])
        
        # Pre-allocate matrices (Optimization for speed)
        self.F = np.eye(6)
        self.B = np.zeros((6, 1))
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0; self.H[1, 1] = 1.0; self.H[2, 2] = 1.0
        self.I = np.eye(6)
        
        self.last_time = time.time()
        self.is_initialized = False
        
        # ระบบเก็บประวัติพิกัดสำหรับ Curve Fitting
        self.history = [] # เก็บค่า (t, x, y, z) ย้อนหลัง
        
        # [M3] ค่าชดเชยแรงต้านอากาศ
        self.drag_correction = drag_correction

    def process(self, measurement=None, origin_distance_cm=323.3):
        """อัปเดต Kalman Filter ด้วยค่าวัดใหม่
        
        Args:
            measurement: [x, y, z] ในหน่วย cm หรือ None ถ้าไม่มีค่าวัด
            origin_distance_cm: ระยะกล้อง→จุด(0,0) สำหรับ Adaptive R
        """
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
        
        # แกน Z เป็นความสูง (ถูกแรงโน้มถ่วงดึงลง)
        self.B[2, 0] = -0.5 * 981 * dt**2 
        self.B[5, 0] = -981 * dt
        
        self.state = np.dot(self.F, self.state) + self.B
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # Update
        if measurement is not None:
            z = np.array(measurement).reshape((3, 1))
            
            # Adaptive Kalman Filter (AKF)
            # ใช้ origin_distance_cm แทน hardcoded 323.3
            distance_from_camera_cm = origin_distance_cm - measurement[0]
            distance_m = abs(distance_from_camera_cm) / 100.0
            
            noise_x = 10.0 + (distance_m ** 2.0) * 15.0
            noise_y = 5.0 + (distance_m ** 1.5) * 3.0
            noise_z = 5.0 + (distance_m ** 1.5) * 5.0
            
            self.R = np.diag([noise_x, noise_y, noise_z])
            
            y = z - np.dot(self.H, self.state)
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            
            # Innovation Gating: ปฏิเสธ Outlier ด้วย Mahalanobis Distance
            mahal_sq = float(np.dot(np.dot(y.T, np.linalg.inv(S)), y))
            if mahal_sq > 16.27:  # chi-squared threshold: 3 DOF, p=0.001
                pass  # Outlier — ข้ามการอัปเดต
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
        """คำนวณจุดตกด้วย SVD + Curve Fitting (แม่นยำ ต้องมี History >= 5 จุด)"""
        if not self.is_initialized or len(self.history) < 5:
            return None, None
            
        t_arr = np.array([pt[0] for pt in self.history])
        t_arr = t_arr - t_arr[0]
        x_arr = np.array([pt[1] for pt in self.history])
        y_arr = np.array([pt[2] for pt in self.history])
        z_arr = np.array([pt[3] for pt in self.history])
        
        if z_arr[-1] <= 0:
            return float(x_arr[-1]), float(y_arr[-1])
            
        # 1. SVD ลด Noise แนวขวาง
        mean_x = np.mean(x_arr)
        mean_y = np.mean(y_arr)
        M = np.vstack((x_arr - mean_x, y_arr - mean_y)).T
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        dir_vector = Vt[0]
        projected = np.dot(M, dir_vector)
        clean_xy = np.outer(projected, dir_vector)
        clean_x = clean_xy[:, 0] + mean_x
        clean_y = clean_xy[:, 1] + mean_y
        
        # 2. Curve Fitting แกน X และ Y
        px = np.polyfit(t_arr, clean_x, 1)
        py = np.polyfit(t_arr, clean_y, 1)
        vx, x0 = px[0], px[1]
        vy, y0 = py[0], py[1]
        
        # 3. Curve Fitting แกน Z (พาราโบลา + แรงโน้มถ่วงฟิกซ์ค่า)
        g = 981.0
        z_adj = z_arr + 0.5 * g * (t_arr**2)
        pz = np.polyfit(t_arr, z_adj, 1)
        vz, z0 = pz[0], pz[1]
        
        t_current = t_arr[-1]
        a = -0.5 * g
        b = vz
        c = z0
        discriminant = (b**2) - (4 * a * c)
        
        if discriminant < 0:
            return None, None
            
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        t_land = max(t1, t2)
        
        if t_land < t_current or (t_land - t_current) > 2.5:
            return None, None
            
        sim_x = (x0 + vx * t_land) * self.drag_correction
        sim_y = (y0 + vy * t_land) * self.drag_correction
        
        return float(sim_x), float(sim_y)

    def predict_landing_fast(self):
        """⚡ คำนวณจุดตกแบบเร็ว ใช้ KF State โดยตรง ไม่ต้องรอสะสม History"""
        if not self.is_initialized:
            return None, None
        
        x, y, z = self.state[:3].flatten()
        vx, vy, vz = self.state[3:].flatten()
        
        if z <= 0:
            return float(x), float(y)
        
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
        
        pred_x = float(x + vx * t_land) * self.drag_correction
        pred_y = float(y + vy * t_land) * self.drag_correction
        return pred_x, pred_y
