import cv2
import os
import glob
from datetime import date

def main():
    # หาไฟล์วิดีโอ .mp4 ทั้งหมดที่ขึ้นต้นด้วย track_record_
    video_files = glob.glob('track_record_*.mp4')
    
    if not video_files:
        print("❌ ไม่พบไฟล์วิดีโอ track_record_...mp4 ในโฟลเดอร์นี้")
        return

    # คัดกรองเฉพาะไฟล์ที่สร้าง "วันนี้"
    today = date.today()
    todays_videos = []
    for f in video_files:
        file_date = date.fromtimestamp(os.path.getctime(f))
        if file_date == today:
            todays_videos.append(f)
            
    if not todays_videos:
        print("❌ ไม่พบไฟล์วิดีโอที่อัดใน 'วันนี้' เลยครับ")
        return

    print(f"🎬 พบไฟล์วิดีโอของวันนี้ทั้งหมด {len(todays_videos)} ไฟล์ กำลังเริ่มแตกรูปภาพ...")

    # สร้างโฟลเดอร์สำหรับเก็บรูป
    output_dir = "images_for_training"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_count = 0

    for video_file in todays_videos:
        print(f"\nกำลังประมวลผลไฟล์: {video_file}")
        cap = cv2.VideoCapture(video_file)
        
        # ดึงค่า FPS ของวิดีโอ (ถ้าไม่ได้กำหนดจะใช้ 30)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
        # เราต้องการประมาณ 2 รูปต่อวินาที
        frame_skip = int(fps / 2) 
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # บันทึกภาพทุกๆ frame_skip เฟรม
            if frame_count % frame_skip == 0:
                filename = os.path.join(output_dir, f"img_{saved_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
                
                # พิมพ์บอกสถานะทุกๆ 50 รูป จะได้ไม่รกจอ
                if saved_count % 50 == 0:
                    print(f"⏳ แตกไฟล์รูปไปแล้ว {saved_count} รูป...")
                    
            frame_count += 1

        cap.release()

    print(f"\n✅ เสร็จสิ้น! ได้รูปภาพทั้งหมด {saved_count} รูป จาก {len(todays_videos)} วิดีโอ")
    print(f"📂 รูปถูกเก็บไว้ในโฟลเดอร์: {os.path.abspath(output_dir)}")
    print("👉 ให้นำรูปภาพทั้งหมดในโฟลเดอร์นี้ ไปอัปโหลดเข้า Roboflow ได้เลยครับ")

if __name__ == '__main__':
    main()
