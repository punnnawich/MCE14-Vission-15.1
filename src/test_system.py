"""
==========================================
  MCE14 Vision — System Test (ไม่ต้องต่อกล้อง)
  ทดสอบ: KF, Landing Prediction, UDP, Innovation Gating
==========================================
"""
import sys
import os
import time
import numpy as np

# เพิ่ม path ไปหา source
sys.path.insert(0, os.path.dirname(__file__))
from ball_tracker_kf import BallTrackerKF

PASS = 0
FAIL = 0

def test(name, condition):
    global PASS, FAIL
    if condition:
        print(f"  ✅ {name}")
        PASS += 1
    else:
        print(f"  ❌ {name}")
        FAIL += 1

# ==========================================
# TEST 1: KF Initialization
# ==========================================
print("\n🧪 TEST 1: Kalman Filter Initialization")
kf = BallTrackerKF(drag_correction=0.92)
test("State vector เป็น 6x1 zeros", kf.state.shape == (6, 1) and np.allclose(kf.state, 0))
test("ยังไม่ initialized", kf.is_initialized == False)
test("History ว่าง", len(kf.history) == 0)
test("Drag correction = 0.92", kf.drag_correction == 0.92)

# ==========================================
# TEST 2: KF Process (First measurement)
# ==========================================
print("\n🧪 TEST 2: KF Process — First Measurement")
kf.process([100, 0, 80], origin_distance_cm=323.3)
test("Initialized หลังรับค่าแรก", kf.is_initialized == True)
test("Position = measurement", abs(kf.state[0,0] - 100) < 0.01)
test("Velocity = 0 ตอนเริ่มต้น", abs(kf.state[3,0]) < 0.01)

# ==========================================
# TEST 3: KF Process (Multiple measurements - simulate throw)
# ==========================================
print("\n🧪 TEST 3: KF Process — จำลองการโยนบอล")
kf2 = BallTrackerKF(drag_correction=0.92)
g = 981.0
dt = 0.017  # 60 FPS

# จำลองลูกบอล: เริ่มที่ (200, 0, 100) ความเร็ว vx=-150, vy=5, vz=50
x0, y0, z0 = 200.0, 0.0, 100.0
vx0, vy0, vz0 = -150.0, 5.0, 50.0

measurements = []
for i in range(15):
    t = i * dt
    x = x0 + vx0 * t
    y = y0 + vy0 * t
    z = z0 + vz0 * t - 0.5 * g * t**2
    if z > 0:
        # เพิ่ม noise เล็กน้อย
        noise = np.random.normal(0, 2, 3)
        measurements.append([x + noise[0], y + noise[1], z + noise[2]])

for m in measurements:
    time.sleep(0.001)  # ให้ dt ไม่เป็น 0
    kf2.process(m, origin_distance_cm=323.3)

test(f"History มี {len(kf2.history)} จุด (ควร >= 5)", len(kf2.history) >= 5)
test("KF velocity vx < 0 (บอลวิ่งเข้าหาหุ่น)", kf2.state[3,0] < 0)
test("KF position Z > 0 (ยังอยู่เหนือพื้น)", kf2.state[2,0] > 0)

# ==========================================
# TEST 4: predict_landing_fast
# ==========================================
print("\n🧪 TEST 4: predict_landing_fast()")
lx, ly = kf2.predict_landing_fast()
test("ได้ค่า landing X (ไม่ใช่ None)", lx is not None)
test("ได้ค่า landing Y (ไม่ใช่ None)", ly is not None)
if lx is not None:
    test(f"Landing X = {lx:.1f} (ควรอยู่ในช่วง -200 ถึง 200)", -200 < lx < 200)
    test(f"Landing Y = {ly:.1f} (ควรอยู่ในช่วง -50 ถึง 50)", -50 < ly < 50)

# ==========================================
# TEST 5: predict_landing (SVD)
# ==========================================
print("\n🧪 TEST 5: predict_landing() — SVD Mode")
sx, sy = kf2.predict_landing()
test("SVD ได้ค่า landing X (ไม่ใช่ None)", sx is not None)
if sx is not None and lx is not None:
    diff = abs(sx - lx)
    test(f"SVD vs FAST ต่างกัน {diff:.1f}cm (ควร < 30cm)", diff < 30)

# ==========================================
# TEST 6: Innovation Gating (Outlier Rejection)
# ==========================================
print("\n🧪 TEST 6: Innovation Gating — Outlier Rejection")
kf3 = BallTrackerKF(drag_correction=0.92)
# ให้ KF ติดตามจุดเดิม 10 รอบ
for _ in range(10):
    time.sleep(0.001)
    kf3.process([100, 0, 80], origin_distance_cm=323.3)

pos_before = kf3.state[0,0]
# ส่ง Outlier (กระโดดไป 500cm)
time.sleep(0.001)
kf3.process([500, 200, 80], origin_distance_cm=323.3)
pos_after = kf3.state[0,0]
jump = abs(pos_after - pos_before)
test(f"KF ไม่กระโดดตาม outlier (jump={jump:.1f}cm, ควร < 50)", jump < 50)

# ==========================================
# TEST 7: Drag Correction
# ==========================================
print("\n🧪 TEST 7: Drag Correction Factor")
kf_no_drag = BallTrackerKF(drag_correction=1.0)
kf_drag = BallTrackerKF(drag_correction=0.92)

for m in measurements:
    time.sleep(0.001)
    kf_no_drag.process(m, origin_distance_cm=323.3)
    kf_drag.process(m, origin_distance_cm=323.3)

lx_nd, _ = kf_no_drag.predict_landing_fast()
lx_d, _ = kf_drag.predict_landing_fast()

if lx_nd is not None and lx_d is not None:
    ratio = abs(lx_d / lx_nd) if lx_nd != 0 else 0
    test(f"Drag version สั้นกว่า ~8% (ratio={ratio:.3f})", 0.88 < ratio < 0.96)

# ==========================================
# TEST 8: Reset History
# ==========================================
print("\n🧪 TEST 8: Reset History")
kf2.reset_history()
test("History ว่างหลัง reset", len(kf2.history) == 0)
result = kf2.predict_landing()
test("predict_landing คืน None เมื่อไม่มี history", result == (None, None))
# predict_landing_fast ยังทำงานได้ (ใช้ KF state)
fast_result = kf2.predict_landing_fast()
test("predict_landing_fast ยังทำงานได้ (ใช้ KF state ไม่ต้องใช้ history)", fast_result[0] is not None)

# ==========================================
# TEST 9: UDP Sender
# ==========================================
print("\n🧪 TEST 9: UDP Sender")
from udp_sender import UDPSender
sender = UDPSender(["127.0.0.1"], 12345)
test("UDPSender สร้างได้สำเร็จ", sender is not None)
test("Sequence เริ่มที่ 0", sender.packet_seq == 0)
# ทดสอบส่ง (ส่งไป localhost — ไม่ต้องมี ESP32)
try:
    sender.send_data_binary(10.5, -5.3, 0.0, repeats=3)
    test("ส่ง UDP สำเร็จ (3 repeats)", True)
    test("Sequence เพิ่มเป็น 1", sender.packet_seq == 1)
except Exception as e:
    test(f"UDP send failed: {e}", False)
sender.close()

# ==========================================
# สรุปผล
# ==========================================
print(f"\n{'='*50}")
print(f"📊 ผลรวม: {PASS} ผ่าน / {FAIL} ไม่ผ่าน / {PASS+FAIL} ทั้งหมด")
if FAIL == 0:
    print("🎉 ผ่านทุกข้อ! ระบบพร้อมใช้งาน")
else:
    print(f"⚠️ มี {FAIL} ข้อไม่ผ่าน กรุณาตรวจสอบ")
print(f"{'='*50}")
