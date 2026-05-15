# MCE14 Vission 15.1
### ระบบตรวจจับลูกบอลและคำนวณจุดตก 3 มิติ สำหรับหุ่นยนต์รับบอล

> **Version 15.1** — 15 พฤษภาคม 2569

## ภาพรวม

ระบบใช้กล้อง OAK-D Lite ตรวจจับลูกบอลสีแดงด้วย YOLOv8 TensorRT คำนวณตำแหน่ง 3D แบบ Real-time ผ่าน Adaptive Kalman Filter พยากรณ์จุดตกด้วย SVD + Curve Fitting และส่งพิกัดไป ESP32 ผ่าน UDP ภายใน **< 300ms**

## โครงสร้างโฟลเดอร์

```
MCE14 Vission 15.1/
├── src/                          # ซอร์สโค้ดหลัก
│   ├── track_3d.py               # ⭐ ไฟล์หลัก (Single-threaded)
│   ├── track_3d_threaded.py      # 🆕 เวอร์ชัน 3-Thread
│   ├── track_dual_3d.py          # Dual Camera mode
│   ├── ball_tracker_kf.py        # 🆕 โมดูลกลาง Kalman Filter
│   ├── udp_sender.py             # UDP Binary sender
│   ├── plot_3d.py                # กราฟ 3D Real-time
│   ├── local_udp_receiver.py     # จำลอง ESP32 ฝั่งรับ
│   └── config.yaml               # 🆕 ค่าคงที่ทั้งหมด
├── tools/                        # เครื่องมือ Calibration & Training
│   ├── pitch_calibrate.py
│   ├── checkerboard_extrinsic.py
│   ├── stereo_calibrate.py
│   ├── train.py
│   ├── export_engine.py
│   └── extract_frames.py
├── tests/                        # ชุดทดสอบ
│   ├── test_robot.py
│   ├── test_manual.py
│   └── test_dual_camera.py
├── docs/                         # เอกสาร
│   ├── Technical_Report.md       # 🆕 รายงานฉบับเต็ม
│   ├── README.md
│   ├── Optimization_Guide.md
│   ├── QA_QC_Report.md
│   └── Update_Log.md
├── firmware/                     # ESP32 Firmware
│   └── ROBOT_CONTROL.ino
├── models/                       # ไฟล์ Calibration
│   └── stereo_calib_data.npz
├── logs/                         # Landing Log CSV
└── .gitignore
```

## Quick Start

```bash
# 1. ติดตั้ง Dependencies
pip install opencv-python depthai ultralytics torch numpy matplotlib pyyaml

# 2. รัน (เลือกเวอร์ชัน)
cd src
python track_3d.py              # Single-thread (เสถียร)
python track_3d_threaded.py     # Multi-thread (เร็วกว่า)

# 3. กด 'z' เพื่อ Calibrate จุด Origin
# 4. โยนบอล → ระบบจะส่งพิกัดจุดตกไป ESP32 อัตโนมัติ
```

## Key Metrics

| Metric | ค่า |
|--------|-----|
| Throw → Send | ~50-100ms (Deadline 280ms) |
| Camera FPS | 60 FPS |
| YOLO Inference | ~5ms (TensorRT FP16) |
| UDP Packet | 16 bytes × 3 repeats |

## What's New in v15.1

- ⚡ `predict_landing_fast()` — คำนวณจุดตกเร็วขึ้น 5x
- 🛡️ Innovation Gating — ปฏิเสธ outlier จาก YOLO
- 🔧 Bugfix: Adaptive R ใช้ค่า calibrated แทน hardcoded
- 📡 UDP Redundant Send ×3 ลด packet loss
- 🧵 Threaded version (3 threads)
- 📄 Technical Report ฉบับเต็ม
- 📁 จัดโครงสร้างโฟลเดอร์ใหม่
