from ultralytics import YOLO
import os

def main():
    # โฟลเดอร์ที่ได้จากการ Train (ปกติจะรันและสร้างโฟลเดอร์รันล่าสุด เช่น train, train-2, train-3)
    # อัปเดต Path ไปที่โฟลเดอร์ล่าสุดที่คุณเทรนสำเร็จ (train-3)
    model_path = r"runs\detect\train-3\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
        print("ลองตรวจสอบดูว่าคุณเทรนเสร็จแล้ว หรือมีการเปลี่ยนชื่อโฟลเดอร์ train เป็นอย่างอื่นหรือไม่")
        return

    model = YOLO(model_path) 

    print("🚀 กำลังแปลงโมเดลเป็น TensorRT (Engine)... (ขั้นตอนนี้อาจใช้เวลา 5-15 นาที ห้ามปิดโปรแกรม)")

    # แปลงเป็น TensorRT แบบ FP16
    model.export(
        format="engine",
        device=0,
        half=True,  # ใช้ FP16 ทำให้เร็วกว่าปกติ 2 เท่าบน GPU
        workspace=4 # ให้แรมทำงาน (GB)
    )

    print("✅ แปลงเสร็จเรียบร้อย! คุณจะได้ไฟล์ best.engine ในโฟลเดอร์ weights")

if __name__ == '__main__':
    main()
