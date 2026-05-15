import cv2
import depthai as dai
import numpy as np
import torch
from ultralytics import YOLO
import math

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
USE_HALF = (DEVICE == 'cuda:0')

# โหลดโมเดล
model_path = r"C:\Users\punna\OneDrive\Documents\runs\detect\redball_model\weights\best.engine"
model = YOLO(model_path)

# สร้าง Pipeline
pipeline = dai.Pipeline()

# กล้อง RGB
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 3)  # 640x360
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setFps(30)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.isp.link(xoutRgb.input)

# กล้อง Depth
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setFps(30)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setFps(30)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(camRgb.getIspWidth(), camRgb.getIspHeight())

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

def get_distance_from_center(depth_map, x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    roi_x1 = max(0, int(x1 + w * 0.35))
    roi_x2 = min(depth_map.shape[1], int(x2 - w * 0.35))
    roi_y1 = max(0, int(y1 + h * 0.35))
    roi_y2 = min(depth_map.shape[0], int(y2 - h * 0.35))
    
    if roi_x2 <= roi_x1: roi_x2 = roi_x1 + 1
    if roi_y2 <= roi_y1: roi_y2 = roi_y1 + 1
    
    roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
    valid_depths = roi[roi > 0]
    
    if len(valid_depths) == 0:
        roi_x1, roi_x2 = max(0, int(x1 + w * 0.1)), min(depth_map.shape[1], int(x2 - w * 0.1))
        roi_y1, roi_y2 = max(0, int(y1 + h * 0.1)), min(depth_map.shape[0], int(y2 - h * 0.1))
        roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
        valid_depths = roi[roi > 0]

    if len(valid_depths) > 0:
        sorted_depths = np.sort(valid_depths)
        half_idx = max(1, len(sorted_depths) // 2)
        return np.median(sorted_depths[:half_idx])
    return 0

pt1 = None
pt2 = None

with dai.Device(pipeline) as device:
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, camRgb.getIspWidth(), camRgb.getIspHeight())
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    
    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    
    print("🚀 เริ่มระบบ Calibrate องศากล้อง (Pitch)")
    print("1. ถือลูกบอลทาบกำแพง ระดับต่ำ (เช่น ระดับเอว) แล้วกดปุ่ม '1'")
    print("2. รูดลูกบอลขึ้นทาบกำแพง ระดับสูง (เช่น ระดับหน้าอก/หน้า) แล้วกดปุ่ม '2'")
    print("กด 'q' เพื่อออก")
    
    while True:
        inRgb = qRgb.get()
        inDepth = qDepth.tryGet()
        
        frame = inRgb.getCvFrame()
        depth_frame = inDepth.getFrame() if inDepth is not None else None
        
        results = model.predict(frame, conf=0.5, verbose=False, device=DEVICE, half=USE_HALF)
        annotated = frame.copy()
        
        ball_boxes = [box for box in results[0].boxes if int(box.cls[0].item()) == 0]
        
        current_Y_c = None
        current_Z_c = None
        
        if len(ball_boxes) > 0 and depth_frame is not None:
            box = sorted(ball_boxes, key=lambda x: x.conf[0].item(), reverse=True)[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            Z_mm = get_distance_from_center(depth_frame, x1, y1, x2, y2)
            if Z_mm > 0:
                Y_mm = (center_y - cy) * Z_mm / fy
                
                # แปลงเป็นหน่วย ซม. ตามกล้องดิบๆ (ยังไม่หมุนแกน)
                current_Y_c = Y_mm / 10.0
                current_Z_c = Z_mm / 10.0
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"Y_cam: {current_Y_c:.1f} cm", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated, f"Z_cam: {current_Z_c:.1f} cm", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # วาดสถานะ
        if pt1 is not None:
            cv2.putText(annotated, f"[PT1] Y:{pt1[0]:.1f}, Z:{pt1[1]:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if pt2 is not None:
            cv2.putText(annotated, f"[PT2] Y:{pt2[0]:.1f}, Z:{pt2[1]:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        cv2.imshow("Pitch Calibrator", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1') and current_Y_c is not None:
            pt1 = (current_Y_c, current_Z_c)
            print(f"✅ บันทึกจุดที่ 1 แล้ว: Y_cam = {pt1[0]:.1f}, Z_cam = {pt1[1]:.1f}")
        elif key == ord('2') and current_Y_c is not None:
            pt2 = (current_Y_c, current_Z_c)
            print(f"✅ บันทึกจุดที่ 2 แล้ว: Y_cam = {pt2[0]:.1f}, Z_cam = {pt2[1]:.1f}")
            
            if pt1 is not None:
                dy = pt1[0] - pt2[0]  # หาผลต่างแกนดิ่ง (ยิ่งลูกบอลอยู่สูง ค่า Y ในกล้องจะน้อยลง)
                dz = pt2[1] - pt1[1]  # หาผลต่างแกนลึก
                
                if abs(dy) < 5.0:
                    print("❌ เกิดข้อผิดพลาด: คุณไม่ได้ยกลูกบอลให้สูงพอ (กรุณายกให้ต่างกันอย่างน้อย 5 ซม.) ให้ทำใหม่")
                else:
                    theta_rad = math.atan2(dz, dy)
                    pitch_deg = math.degrees(theta_rad)
                    print("\n" + "="*60)
                    print(f"🎯 คำนวณเสร็จสิ้น! มุมก้มเงยของกล้องคุณคือ: {pitch_deg:.2f} องศา")
                    print("="*60)
                    print(f"ให้นำค่านี้ไปใส่บรรทัดที่ 176 ในไฟล์ track_3d.py นะครับ:")
                    print(f"CAMERA_PITCH_DEG = {pitch_deg:.2f}")
                    print("="*60 + "\n")

cv2.destroyAllWindows()
