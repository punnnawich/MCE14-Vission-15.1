# 🚀 คู่มือยกระดับประสิทธิภาพระบบตรวจจับ (Ultimate Optimization Guide)
**Project:** MCE14 Vision
**Level:** Advanced / Production

คู่มือนี้รวบรวมขั้นตอนการทำอัปเกรดทั้ง 4 ส่วน เพื่อให้ระบบก้าวข้ามขีดจำกัดเรื่องความเร็วและความทนทานต่อสภาพแวดล้อม (Robustness)

---

## 🏎️ 1. การเร่งความเร็วด้วย TensorRT (เพิ่ม FPS บนการ์ดจอ NVIDIA)
TensorRT เป็นไลบรารีของ NVIDIA ที่จะรีดประสิทธิภาพการ์ดจอออกมาได้สูงสุด (เร็วกว่าไฟล์ `.pt` ของ PyTorch ประมาณ 2-3 เท่า)

**ขั้นตอนการทำ:**
1. เปิด Anaconda Prompt หรือ Terminal (ที่รันโค้ดปกติ)
2. พิมพ์คำสั่งเพื่อแปลงไฟล์ `best.pt` ของคุณให้เป็น `.engine`:
   ```bash
   yolo export model="C:\Users\punna\OneDrive\Documents\runs\detect\redball_model\weights\best.pt" format=engine device=0 half=true
   ```
3. รอจนเสร็จ คุณจะได้ไฟล์ `best.engine` โผล่ขึ้นมาในโฟลเดอร์เดียวกัน
4. **การนำไปใช้:** ในไฟล์ `track_3d.py` แค่เปลี่ยนพาธโมเดลบรรทัดที่ 105:
   ```python
   # ของเดิม
   # model_path = r"C:\...\best.pt" 
   
   # ของใหม่
   model_path = r"C:\...\best.engine" 
   model = YOLO(model_path)
   ```
*(ระบบจะโหลดช้าตอนเปิดโปรแกรมครั้งแรก แต่พอมันทำงาน FPS จะพุ่งทะลุขีดจำกัด)*

---

## 🤖 2. การย้าย AI ไปรันบนชิปกล้อง OAK-D (Edge AI Computing)
วิธีนี้คือขั้นสุดของการทำ Vission เพราะไม่ต้องง้อการ์ดจอคอมพิวเตอร์เลย กล้อง OAK-D จะทำหน้าที่หาลูกบอลและบอกพิกัด X, Y, Z สำเร็จรูปมาให้ทันที

**ขั้นตอนการทำ:**
1. **แปลงไฟล์เป็น Blob:** ต้องแปลงจาก `.pt` -> `OpenVINO` -> `.blob`
   - แนะนำให้ใช้เว็บไซต์ [Luxonis BlobConverter](https://blobconverter.luxonis.com/) หรือใช้โค้ดแปลงของเครื่องมือ `ultralytics` 
   - คำสั่งแปลงเบื้องต้น: `yolo export model=best.pt format=openvino` จากนั้นใช้เครื่องมือ `blobconverter` ของ DepthAI เปลี่ยนเป็นชิป `MyriadX`
2. **ปรับรื้อโครงสร้างโค้ด DepthAI:** (เรื่องใหญ่) 
   จะต้องเปลี่ยนจากที่เราส่งภาพมาให้คอมพิวเตอร์รัน YOLO เป็นการสร้าง Node ที่ชื่อว่า `SpatialDetectionNetwork` 
   **ตัวอย่างโครงสร้างโค้ดที่ต้องเปลี่ยน:**
   ```python
   spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
   spatialDetectionNetwork.setBlobPath("best.blob")
   spatialDetectionNetwork.setConfidenceThreshold(0.5)
   
   # เชื่อมภาพจากกล้องเข้า AI โดยตรง
   camRgb.preview.link(spatialDetectionNetwork.input)
   stereo.depth.link(spatialDetectionNetwork.inputDepth)
   
   # สิ่งที่ส่งกลับมาให้คอมฯ จะไม่ใช่ภาพ แต่เป็นพิกัด 3D เลย!
   ```
*(หากต้องการทำข้อนี้จริงๆ แจ้งผมได้ครับ ผมจะเขียนไฟล์เวอร์ชัน `track_3d_edge.py` แยกให้ต่างหาก เพราะต้องรื้อลอจิก DepthAI ใหม่ทั้งหมด)*

---

## 🎨 3. การทำ Hybrid Detection (YOLO + HSV Color Tracking)
ลูกบอลสีแดงเป็นสีที่มีเอกลักษณ์ (ถ้าแสงไม่เปลี่ยนมาก) เราสามารถใช้ OpenCV จับสีแดงมาช่วยในจังหวะที่ลูกบอลพุ่งเร็วจัดจน YOLO มองไม่เห็น (Motion Blur)

**ขั้นตอนการทำ:**
ในไฟล์ `track_3d.py` หลังจากเช็คกล่อง YOLO แล้ว ถ้าหาไม่เจอ ให้เราสลับไปใช้สี
```python
            if len(results[0].boxes) > 0:
                # ... <โค้ด YOLO เดิม> ...
            else:
                # 🛠️ HYBRID SYSTEM: ถ้า YOLO มองไม่เห็น ให้ใช้ HSV ค้นหาสีแดงช่วยชีวิต!
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # ช่วงสีแดง (แดงมี 2 ช่วงใน HSV)
                mask1 = cv2.inRange(hsv_frame, np.array([0, 120, 70]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv_frame, np.array([170, 120, 70]), np.array([180, 255, 255]))
                mask = mask1 | mask2
                
                # หาเส้นขอบวัตถุ
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # เลือกลูกที่ใหญ่ที่สุด
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 100: # กรองขยะ
                        x, y, w, h = cv2.boundingRect(c)
                        x1, y1, x2, y2 = x, y, x+w, y+h
                        center_x = x + w//2
                        center_y = y + h//2
                        
                        # วาดกล่องสีฟ้าเพื่อแยกให้รู้ว่านี่คือกล่องจากระบบ HSV ไม่ใช่ YOLO
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(annotated_frame, "HSV Fallback", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # นำ x1,y1,x2,y2 ไปหา Depth Z_mm ต่อได้เลยเหมือน YOLO เด๊ะๆ!
```
*(ระบบนี้ใช้ CPU ประมวลผลแค่ประมาณ 1-2 ms เท่านั้น เร็วมากและเป็นตัวสำรอง (Fallback) ชั้นเยี่ยม)*

---

## 📸 4. การจัดการ Dataset สำหรับเทรน AI (Robust Training)
ปัญหาหลักของ Computer Vision ตอนออกหน้างานคือ "สภาพแวดล้อมไม่เหมือนตอนถ่ายรูปซ้อม"

**แนวทางการปรับปรุง Dataset (ทำบน Roboflow หรือเขียน Script เอง):**
1. **Motion Blur (สำคัญที่สุด):** โยนลูกบอลจริงๆ มันจะเบลอเวลาพุ่งเร็ว ให้ใส่ Augmentation `Blur` ประมาณ 3px ถึง 7px เข้าไปในภาพตอนเทรน
2. **Brightness & Exposure:** ใส่รูปที่เพิ่มและลดแสง (±25%) เข้าไป เพราะตอนแข่งขันจริง แสงสนามหรือแสงพระอาทิตย์เปลี่ยนตลอด
3. **Occlusion (ถูกบัง):** หุ่นยนต์ตัวอื่น หรือแขนหุ่นยนต์อาจจะบังลูกบอลไปครึ่งนึง ให้ใช้เทคนิค `Cutout` เอาสี่เหลี่ยมดำๆ ไปแปะทับรูปลูกบอลบางส่วนตอนเทรน เพื่อให้ AI รู้จักลูกบอลแม้จะเห็นแค่ครึ่งลูก
4. **Background Variation:** ถ่ายรูปลูกบอลที่มีแบ็คกราวด์วุ่นวาย เช่น มีคนใส่เสื้อสีแดง มีเก้าอี้สีแดง เพื่อบังคับให้ AI ต้องดูทรงกลมด้วย ไม่ใช่ดูแค่สี

**คำแนะนำการเทรนใหม่:**
เมื่อเพิ่มรูปครบแล้ว ให้เทรนด้วย `YOLOv8n` (Nano) สัก 100-200 Epochs โมเดลที่ได้จะตัวเล็กมาก (ไม่เกิน 6MB) แต่ฉลาดเป็นกรด ทนทานต่อทุกสนามแข่งครับ
