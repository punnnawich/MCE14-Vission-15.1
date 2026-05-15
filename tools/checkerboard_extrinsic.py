import cv2
import depthai as dai
import numpy as np
import math

# ==========================================
# ⚙️ ตั้งค่าขนาดกระดานหมากรุก (Checkerboard)
# ==========================================
CHESSBOARD_SIZE = (8, 6)      # จำนวน "จุดตัด" ด้านใน (แนวนอน, แนวตั้ง) - ไม่ใช่จำนวนช่อง!
SQUARE_SIZE_CM = 2.5          # ขนาดของช่องสี่เหลี่ยม 1 ช่อง (เซนติเมตร)
# ==========================================

pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 3) # 640x360
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setFps(30)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.isp.link(xoutRgb.input)

# เตรียมพิกัด 3D ของมุมกระดานหมากรุกในโลกจริง (แผ่นหมากรุกวางเรียบอยู่บนพื้น Z=0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_CM

with dai.Device(pipeline) as device:
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, camRgb.getIspWidth(), camRgb.getIspHeight())
    camera_matrix = np.array(intrinsics, dtype=np.float32)
    dist_coeffs = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), dtype=np.float32)
    
    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    
    print("🚀 เริ่มระบบหา องศาและความสูงกล้อง ด้วย Checkerboard (SolvePnP)")
    print(f"-> 1. กรุณาพิมพ์กระดานหมากรุก หรือเปิดภาพหมากรุกใน iPad วางทาบที่พื้น")
    print(f"-> 2. เช็คว่าในรูปมี 'จุดตัด' {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]} จุด และช่องกว้าง {SQUARE_SIZE_CM} ซม.")
    print(f"      (ถ้าขนาดของคุณไม่ตรง ให้เข้าไปแก้ตัวเลขในไฟล์นี้ก่อนนะครับ)")
    print("-> 3. โปรแกรมจะคำนวณและแสดงค่าความสูงและองศาให้ดูแบบ Real-time")
    print("-> 4. ถ้ารู้สึกว่าตัวเลขบนหน้าจอนิ่งดีแล้ว ให้กด 'q' เพื่อออกและล็อกค่าล่าสุด")
    
    last_height = 0.0
    last_pitch = 0.0
    
    while True:
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ค้นหากระดานหมากรุก
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        display_frame = frame.copy()
        
        if ret:
            # เพิ่มความแม่นยำระดับทศนิยมพิกเซล
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners2, ret)
            
            # คำนวณหาตำแหน่งและองศากล้องด้วย SolvePnP ทันที (Auto)
            success, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
            if success:
                rmat, _ = cv2.Rodrigues(rvec)
                camera_pos = -np.matrix(rmat).T * np.matrix(tvec)
                height_cm = abs(camera_pos[2, 0])
                
                pitch_rad = math.atan2(-rmat[2, 2], -rmat[1, 2])
                pitch_deg = math.degrees(pitch_rad)
                
                last_height = height_cm
                last_pitch = pitch_deg
                
                cv2.putText(display_frame, "AUTO CALIBRATING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Height: {height_cm:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Pitch : {pitch_deg:.2f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Looking for Checkerboard...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("Checkerboard Extrinsic Calibration", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if last_height > 0:
                print("\n" + "="*50)
                print("🎯 [CALIBRATION SAVED] ค่าที่คำนวณได้ล่าสุดก่อนกดออก:")
                print(f"CAMERA_HEIGHT_CM = {last_height:.2f}")
                print(f"CAMERA_PITCH_DEG = {last_pitch:.2f}")
                print("="*50)
                print("นำค่าพวกนี้ไปแก้ใน track_3d.py ได้เลยครับ!")
            break
                
cv2.destroyAllWindows()
