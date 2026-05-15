from ultralytics import YOLO
import os

def main():
    # โหลดโมเดล YOLOv8 ขนาด Nano (n) เบาสุดและเร็วสุด เหมาะสำหรับรันแบบ Real-time บนหุ่นยนต์
    model = YOLO('yolov8n.pt') 

    print("🚀 กำลังเริ่มต้นการเทรนโมเดล...")
    
    # Path ไปยังไฟล์ data.yaml ของ Dataset ที่ได้จาก Roboflow
    # *** ให้แก้ Path นี้ให้ตรงกับโฟลเดอร์ที่คุณแตกไฟล์ ZIP มานะครับ ***
    dataset_path = r'C:\Users\punna\OneDrive\Documents\MCE14 Vission\dataset\data.yaml'
    
    if not os.path.exists(dataset_path):
        print(f"❌ ไม่พบไฟล์ Dataset ที่: {dataset_path}")
        print("กรุณาแตกไฟล์ ZIP จาก Roboflow ไว้ในโฟลเดอร์ MCE14 Vission/dataset ก่อนครับ")
        return

    # สั่ง Train
    results = model.train(
        data=dataset_path, 
        epochs=100,      # จำนวนรอบการเรียนรู้ 
        imgsz=640,       # ขนาดภาพที่ใช้เทรน
        batch=8,         # ลดเหลือ 8 เพื่อป้องกันแรมการ์ดจอ (VRAM) เต็ม
        workers=0,       # +++ สำคัญ: ปิด Multiprocessing แก้บัค WinError 1455 ใน Windows +++
        device=0         # บังคับใช้การ์ดจอ (GPU 0)
    )
    
    print("✅ เทรนเสร็จสมบูรณ์! โมเดลที่เก่งที่สุดจะถูกเซฟไว้ที่: runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    main()
