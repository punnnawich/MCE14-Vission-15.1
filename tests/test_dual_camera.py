import cv2
import depthai as dai
import numpy as np

print("กำลังเตรียมระบบกล้อง...")

# ==========================================
# 1. ตั้งค่ากล้องมือถือ (Webcam)
# ==========================================
# หากใช้ Webcam ที่ติดมากับ Notebook มักจะเป็น ID = 0
WEBCAM_ID = 0
cap = cv2.VideoCapture(WEBCAM_ID)

# บังคับความละเอียดให้เป็น 640x360 และดันเฟรมเรต
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"❌ [ERROR] ไม่สามารถเปิดกล้องมือถือ (ID={WEBCAM_ID}) ได้!")
    print("กรุณาเช็คว่าเปิดแอปในมือถือและเชื่อมต่อ USB หรือยัง (หรือลองเข้าไปแก้รหัส WEBCAM_ID ในโค้ดเป็น 0 หรือ 2)")
    exit()

# ==========================================
# 2. ตั้งค่ากล้อง OAK-D (เปิดเฉพาะเลนส์สี RGB เพื่อลดโหลด)
# ==========================================
pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(1, 3) # ย่อจาก 1920x1080 เป็น 640x360
camRgb.setFps(30)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.isp.link(xoutRgb.input)

print("\n🚀 เริ่มสตรีมภาพ 2 กล้องพร้อมกัน...")
print("👉 [วิธีทดสอบความหน่วง]: ลองเอามือโบกผ่านหน้ากล้องทั้ง 2 ตัวพร้อมกัน แล้วดูว่าภาพไหนมาก่อน-หลัง")
print("กด 'q' เพื่อออกโปรแกรม\n")

# ==========================================
# 3. รันระบบดึงภาพพร้อมกัน
# ==========================================
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    
    # ตัวแปรเก็บภาพล่าสุดของ OAK-D ป้องกันการกระตุก
    latest_oak_frame = None 
    
    while True:
        # --- ดึงภาพ OAK-D ---
        inRgb = qRgb.tryGet()
        if inRgb is not None:
            latest_oak_frame = inRgb.getCvFrame()
            
        # --- ดึงภาพมือถือ ---
        ret, frame_phone = cap.read()
        
        # ถ้ามีภาพพร้อมทั้งคู่ ให้นำมาโชว์
        if latest_oak_frame is not None and ret and frame_phone is not None:
            # ย่อ/ขยายภาพจากมือถือให้สูงเท่ากับ OAK-D จะได้นำมาต่อกันได้
            h, w = latest_oak_frame.shape[:2]
            frame_phone_resized = cv2.resize(frame_phone, (w, h))
            
            # ใส่ชื่อกล้อง
            cv2.putText(latest_oak_frame, "OAK-D Lite", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_phone_resized, "Mobile Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # นำภาพมาต่อติดกัน (แนวนอน)
            combined_frame = np.hstack((latest_oak_frame, frame_phone_resized))
            
            # แสดงผล
            cv2.imshow("Multi-Camera Latency Test", combined_frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
