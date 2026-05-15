import cv2
import depthai as dai
import numpy as np
import os
import time

# ==========================================
# ⚙️ การตั้งค่ากระดานหมากรุก (Chessboard)
# ==========================================
# นับจำนวน "จุดตัดมุมด้านใน" (Inner Corners) ของตาราง
CHESSBOARD_SIZE = (8, 6) 
# ขนาดของช่องสี่เหลี่ยมจัตุรัส 1 ช่อง (หน่วยเป็น มิลลิเมตร)
SQUARE_SIZE_MM = 25.0 

WEBCAM_ID = 0

# เตรียมจุดพิกัด 3 มิติของกระดานหมากรุก (Z = 0 เสมอเพราะแผ่นกระดาษแบน)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM

# อาเรย์สำหรับเก็บจุด
objpoints = []   # จุด 3 มิติในโลกจริง
imgpoints_oak = [] # จุด 2 มิติในภาพ OAK-D
imgpoints_web = [] # จุด 2 มิติในภาพ Webcam

# ==========================================
# 📷 ตั้งค่ากล้อง
# ==========================================
cap = cv2.VideoCapture(WEBCAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)

pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 3) # 640x360
camRgb.setFps(30)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.isp.link(xoutRgb.input)

print("\n🚀 เริ่มระบบ Stereo Calibration...")
print(f"ตารางหมากรุกที่ตั้งค่าไว้: จุดตัดมุมใน {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} | ขนาดช่องละ {SQUARE_SIZE_MM} mm")
print("--------------------------------------------------")
print("[วิธีใช้งาน]")
print("1. ถือแผ่นหมากรุกให้กล้อง 2 ตัวมองเห็นพร้อมกัน")
print("2. กด 'c' เพื่อแคปเจอร์เก็บข้อมูล (เก็บสัก 20-30 มุมภาพ ในระยะใกล้-ไกล-เอียง)")
print("3. กด 's' เพื่อเริ่มคำนวณและบันทึกค่า Calibration")
print("4. กด 'q' เพื่อออก")
print("--------------------------------------------------\n")

captured_count = 0
last_capture_time = time.time()

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    
    while True:
        inRgb = qRgb.tryGet()
        frame_oak = None
        if inRgb is not None:
            frame_oak = inRgb.getCvFrame()
            
        ret, frame_web = cap.read()
        
        if frame_oak is not None and ret and frame_web is not None:
            # ย่อให้ขนาดเท่ากัน
            frame_web = cv2.resize(frame_web, (frame_oak.shape[1], frame_oak.shape[0]))
            
            # ทำภาพขวาซ้ายโชว์
            display_oak = frame_oak.copy()
            display_web = frame_web.copy()
            
            # แปลงเป็นขาวดำเพื่อหาจุด
            gray_oak = cv2.cvtColor(frame_oak, cv2.COLOR_BGR2GRAY)
            gray_web = cv2.cvtColor(frame_web, cv2.COLOR_BGR2GRAY)
            
            # ค้นหามุมหมากรุกแบบคร่าวๆ (เพื่อพรีวิวสีเส้น)
            ret_oak, corners_oak = cv2.findChessboardCorners(gray_oak, CHESSBOARD_SIZE, None)
            ret_web, corners_web = cv2.findChessboardCorners(gray_web, CHESSBOARD_SIZE, None)
            
            if ret_oak and ret_web:
                cv2.drawChessboardCorners(display_oak, CHESSBOARD_SIZE, corners_oak, ret_oak)
                cv2.drawChessboardCorners(display_web, CHESSBOARD_SIZE, corners_web, ret_web)
                cv2.putText(display_oak, "READY TO CAPTURE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_oak, "CHESSBOARD NOT FOUND", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.putText(display_oak, f"Captured: {captured_count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            combined = np.hstack((display_oak, display_web))
            cv2.imshow("Stereo Calibration", combined)
            
            key = cv2.waitKey(1) & 0xFF
            
            # --- ระบบถ่ายภาพอัตโนมัติ ---
            if ret_oak and ret_web:
                # ถ้าเจอหมากรุกทั้ง 2 กล้อง และผ่านไปแล้ว 1.5 วินาทีจากการถ่ายครั้งก่อน
                if time.time() - last_capture_time > 1.5 and captured_count < 30:
                    # ปรับแต่งความแม่นยำระดับ Sub-pixel
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_oak_sub = cv2.cornerSubPix(gray_oak, corners_oak, (11,11), (-1,-1), criteria)
                    corners_web_sub = cv2.cornerSubPix(gray_web, corners_web, (11,11), (-1,-1), criteria)
                    
                    objpoints.append(objp)
                    imgpoints_oak.append(corners_oak_sub)
                    imgpoints_web.append(corners_web_sub)
                    
                    captured_count += 1
                    last_capture_time = time.time()
                    print(f"📸 [AUTO] บันทึกมุมที่ {captured_count}/30 สำเร็จ! (ขยับกระดานหมากรุกเปลี่ยนมุมได้เลย)")
            
            # เมื่อถ่ายครบ 30 ภาพ ให้จำลองการกด 's' เพื่อเริ่มคำนวณอัตโนมัติ
            if captured_count >= 30:
                key = ord('s')
                
            # --- กด 'c' แคปเจอร์ (Manual แบบเดิมยังใช้ได้) ---
            if key == ord('c'):
                pass # นำโค้ดเก่าออก เพราะเราใช้ออโต้แล้ว

                    
            # --- กด 's' เริ่มคำนวณ ---
            elif key == ord('s'):
                if captured_count < 10:
                    print("⚠️ ควรมีภาพอย่างน้อย 10 ภาพก่อนกดคำนวณครับ!")
                    continue
                    
                print("\n⏳ กำลังคำนวณ Camera Intrinsics กล้องทีละตัว...")
                image_size = gray_oak.shape[::-1]
                
                # Calibrate เดี่ยว
                ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_oak, image_size, None, None)
                ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_web, image_size, None, None)
                
                print("⏳ กำลังคำนวณ Stereo Extrinsics (หาจุดเชื่อมกล้อง 2 ตัว)...")
                # Calibrate คู่
                flags = cv2.CALIB_FIX_INTRINSIC
                criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
                
                retS, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
                    objpoints, imgpoints_oak, imgpoints_web, 
                    mtx1, dist1, mtx2, dist2, 
                    image_size, criteria_stereo, flags)
                
                print("\n✅ คำนวณสำเร็จ!")
                print(f"ระยะห่างระหว่างกล้อง (Translation Vector X,Y,Z มิลลิเมตร):\n{T.flatten()}")
                
                # เซฟเก็บไว้ใช้
                np.savez("stereo_calib_data.npz", 
                         mtx_oak=mtx1, dist_oak=dist1, 
                         mtx_web=mtx2, dist_web=dist2, 
                         R=R, T=T)
                print("💾 บันทึกไฟล์ 'stereo_calib_data.npz' เรียบร้อยแล้ว!\n")
                break
                
            elif key == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
