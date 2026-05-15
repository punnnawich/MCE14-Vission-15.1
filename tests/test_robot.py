"""
==========================================================
  🧪 ROBOT CONTROL TEST SUITE
  ระบบทดสอบการควบคุมหุ่นยนต์ ทดสอบได้โดยไม่ต้องใช้กล้อง
==========================================================
วิธีใช้: python test_robot.py
จากนั้นเลือกเมนูเบอร์ที่ต้องการทดสอบ
"""
import socket
import struct
import time
import sys

# ==========================================
# ⚙️ ตั้งค่า IP ของ ESP32 ที่ต้องการทดสอบ
# ==========================================
ESP32_IPS = [
    "192.168.137.123",
    # "192.168.137.243",  # เอา # ออกถ้ามีบอร์ดตัวที่ 2
]
ESP32_PORT = 12345

class UDPTester:
    def __init__(self, ips, port):
        self.ips = ips if isinstance(ips, list) else [ips]
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
    
    def send(self, x, y, z):
        data = struct.pack('<Ifff', self.seq, float(x), float(y), float(z))
        for ip in self.ips:
            self.sock.sendto(data, (ip, self.port))
        self.seq += 1
        return len(data)
    
    def close(self):
        self.sock.close()

# ==========================================
# 🧪 Test 1: ทดสอบ Ping (เช็คว่าคอมเห็นบอร์ดหรือไม่)
# ==========================================
def test_1_ping():
    import subprocess
    print("\n" + "="*50)
    print("  🧪 TEST 1: ทดสอบ Ping หาบอร์ด ESP32")
    print("="*50)
    for ip in ESP32_IPS:
        print(f"\n  📡 กำลัง Ping {ip} ...")
        result = subprocess.run(
            ["ping", "-n", "3", ip],
            capture_output=True, text=True, timeout=10
        )
        if "Reply from" in result.stdout or "bytes=" in result.stdout:
            print(f"  ✅ [{ip}] ตอบกลับ! คอมกับบอร์ดเห็นกัน")
        else:
            print(f"  ❌ [{ip}] ไม่ตอบกลับ! เช็ค WiFi หรือ IP Address")
            print(f"     💡 ลองดู Serial Monitor ว่า IP ถูกต้องหรือไม่")

# ==========================================
# 🧪 Test 2: ทดสอบส่ง UDP Packet (เช็ค Firewall)
# ==========================================
def test_2_udp_connection():
    print("\n" + "="*50)
    print("  🧪 TEST 2: ทดสอบส่ง UDP Packet")
    print("  📋 สิ่งที่ต้องดู: Serial Monitor ของ ESP32")
    print("     ถ้าเห็น '!!! Received Packet Size: 16' = สำเร็จ")
    print("     ถ้าไม่เห็นอะไรเลย = Firewall บล็อก")
    print("="*50)
    
    tester = UDPTester(ESP32_IPS, ESP32_PORT)
    print(f"\n  📤 ส่ง 10 Packets ไปที่ {ESP32_IPS}...")
    
    for i in range(10):
        size = tester.send(0.0, 0.0, 0.0)  # ส่งค่า 0 เพื่อไม่ให้หุ่นขยับ
        print(f"  ✉️  Packet #{i+1} ส่งแล้ว ({size} bytes)")
        time.sleep(0.5)
    
    tester.close()
    print("\n  ✅ ส่งครบ 10 Packets แล้ว! ไปดู Serial Monitor เลย")
    print("  💡 ถ้าไม่เห็นอะไร ให้ปิด Windows Firewall แล้วลองใหม่")

# ==========================================
# 🧪 Test 3: ทดสอบสั่งเคลื่อนที่ (ส่งพิกัดจริง)
# ==========================================
def test_3_move_command():
    print("\n" + "="*50)
    print("  🧪 TEST 3: ทดสอบสั่งหุ่นยนต์เคลื่อนที่")
    print("  ⚠️  หุ่นยนต์จะขยับจริง! วางบนพื้นก่อน!")
    print("="*50)
    
    x = input("\n  ป้อนค่า X (cm, เช่น 20): ").strip()
    y = input("  ป้อนค่า Y (cm, เช่น 30): ").strip()
    
    try:
        x = float(x)
        y = float(y)
    except ValueError:
        print("  ❌ กรุณาป้อนตัวเลข!")
        return
    
    print(f"\n  🎯 จะสั่งหุ่นวิ่งไปที่ X={x:.1f}cm, Y={y:.1f}cm")
    confirm = input("  กด Enter เพื่อส่งคำสั่ง (หรือพิมพ์ 'n' ยกเลิก): ")
    if confirm.lower() == 'n':
        print("  ยกเลิก.")
        return
    
    tester = UDPTester(ESP32_IPS, ESP32_PORT)
    # ส่ง 5 packets ซ้ำเพื่อให้แน่ใจว่าบอร์ดจับได้
    for i in range(5):
        tester.send(x, y, 0.0)
        time.sleep(0.05)
    
    tester.close()
    print(f"\n  ✅ ส่งคำสั่งเคลื่อนที่ไปแล้ว!")
    print(f"  📋 ดูที่ Serial Monitor ควรเห็น:")
    print(f"     >>> MISSION START <<<")
    print(f"     Target M: X:{x/100:.2f}, Y:{y/100:.2f}")

# ==========================================
# 🧪 Test 4: ทดสอบ Stall Detection (เบรกฉุกเฉิน)
# ==========================================
def test_4_stall():
    print("\n" + "="*50)
    print("  🧪 TEST 4: ทดสอบ Stall Detection")
    print("  📋 วิธีทดสอบ:")
    print("     1. ส่งคำสั่งเคลื่อนที่")
    print("     2. จับล้อไว้ไม่ให้หมุน (ใช้มือจับแน่นๆ)")
    print("     3. รอ ~0.5 วินาที")
    print("     4. Serial Monitor ควรขึ้น:")
    print("        '!!! FAULT: MOTOR STALLED !!! EMERGENCY STOP !!!'")
    print("="*50)
    
    tester = UDPTester(ESP32_IPS, ESP32_PORT)
    
    # ส่งค่าสูงๆ เพื่อให้มอเตอร์ออกแรงเยอะ
    print("\n  📤 ส่งคำสั่งเคลื่อนที่ X=50cm Y=50cm...")
    for i in range(5):
        tester.send(50.0, 50.0, 0.0)
        time.sleep(0.05)
    
    print("  ✅ ส่งแล้ว! จับล้อค้างไว้เลย ดู Serial Monitor")
    tester.close()

# ==========================================
# 🧪 Test 5: ทดสอบ Failsafe (ตัดไฟเมื่อ WiFi หลุด)
# ==========================================
def test_5_failsafe():
    print("\n" + "="*50)
    print("  🧪 TEST 5: ทดสอบ Communication Failsafe")
    print("  📋 วิธีทดสอบ:")
    print("     1. สั่งเคลื่อนที่ แล้วหยุดส่งข้อมูล")
    print("     2. หลังจาก 1.5 วินาที ระบบต้องตัดไฟเอง")
    print("     3. Serial Monitor ควรขึ้น:")
    print("        '!!! FAULT: CONNECTION LOST !!! EMERGENCY STOP !!!'")
    print("="*50)
    
    tester = UDPTester(ESP32_IPS, ESP32_PORT)
    
    print("\n  📤 ส่งคำสั่งเคลื่อนที่ X=100cm Y=100cm (ระยะไกล)...")
    for i in range(5):
        tester.send(100.0, 100.0, 0.0)
        time.sleep(0.05)
    
    print("  ⏳ หยุดส่งข้อมูล... รอ 3 วินาที ดู Serial Monitor")
    time.sleep(3)
    print("  ✅ เสร็จ! ถ้าระบบเซฟ หุ่นต้องหยุดเองภายใน 1.5 วินาที")
    tester.close()

# ==========================================
# 🧪 Test 6: ทดสอบ Soft Start (ไฟไม่กระชาก)
# ==========================================
def test_6_soft_start():
    print("\n" + "="*50)
    print("  🧪 TEST 6: ทดสอบ Soft Start / Slew Rate")
    print("  📋 วิธีดู:")
    print("     สังเกตมอเตอร์ต้องค่อยๆ เร่งความเร็วขึ้น")
    print("     ไม่ใช่กระชากเต็มแรงทันที")
    print("="*50)
    
    tester = UDPTester(ESP32_IPS, ESP32_PORT)
    print("\n  📤 ส่งคำสั่งเคลื่อนที่ X=80cm Y=80cm...")
    for i in range(5):
        tester.send(80.0, 80.0, 0.0)
        time.sleep(0.05)
    
    print("  ✅ ส่งแล้ว! สังเกตมอเตอร์ว่าออกตัวสมูทหรือไม่")
    tester.close()

# ==========================================
# 🧪 Test 7: ทดสอบครบวงจร (Full Integration Test)
# ==========================================
def test_7_full_integration():
    print("\n" + "="*50)
    print("  🧪 TEST 7: Full Integration Test")
    print("  📋 ทดสอบแบบจำลอง Sequence เต็มรูปแบบ:")
    print("     1. ส่งข้อมูลต่อเนื่อง (เหมือนกล้องจริง)")
    print("     2. หุ่นยนต์ต้อง: วิ่งไป -> หยุด -> วิ่งกลับ -> หยุด")
    print("  ⚠️  หุ่นจะขยับจริง!")
    print("="*50)
    
    confirm = input("\n  กด Enter เพื่อเริ่ม (หรือ 'n' ยกเลิก): ")
    if confirm.lower() == 'n':
        return
    
    tester = UDPTester(ESP32_IPS, ESP32_PORT)
    
    # Phase 1: ส่งพิกัดจุดตกซ้ำ 3 วินาที (เหมือนกล้องจับลูกบอลได้)
    target_x, target_y = 30.0, 40.0
    print(f"\n  [Phase 1] จำลองกล้องจับลูกบอล -> จุดตก X:{target_x}, Y:{target_y}")
    start = time.time()
    while time.time() - start < 3.0:
        tester.send(target_x, target_y, 0.0)
        time.sleep(1/30.0)  # 30 FPS
    
    # Phase 2: หยุดส่ง (เหมือนลูกบอลตกแล้ว)
    print("  [Phase 2] ลูกบอลตกแล้ว -> หยุดส่งข้อมูล")
    print("  ⏳ รอ 5 วินาที ให้หุ่นวิ่งไป -> หยุด -> วิ่งกลับ...")
    time.sleep(5)
    
    print("\n  ✅ Full Integration Test เสร็จสิ้น!")
    print("  📋 ผลลัพธ์ที่ควรเห็นใน Serial Monitor:")
    print("     1. '>>> MISSION START <<<'")
    print("     2. หุ่นวิ่งไปเป้าหมาย")
    print("     3. 'Returning Home...'")
    print("     4. 'Back to Start point.'")
    print("     5. (ถ้ารอนาน) '!!! FAULT: CONNECTION LOST !!!'")
    tester.close()

# ==========================================
# เมนูหลัก
# ==========================================
def main():
    print("\n" + "="*60)
    print("  🤖 ROBOT CONTROL TEST SUITE v1.0")
    print("  ระบบทดสอบหุ่นยนต์ MCE14 Vision")
    print(f"  เป้าหมาย: {', '.join(ESP32_IPS)} (Port {ESP32_PORT})")
    print("="*60)
    print("""
  📋 เลือกเมนูทดสอบ (แนะนำทดสอบตามลำดับ):
  
  [1] 📡 Ping Test       - เช็คว่าคอมเห็นบอร์ดหรือไม่
  [2] 📤 UDP Test         - เช็คว่า Packet เข้าบอร์ดไหม (Firewall)
  [3] 🏃 Move Test        - สั่งเคลื่อนที่ด้วยพิกัดที่กำหนด
  [4] 🛑 Stall Test       - ทดสอบระบบตัดไฟเมื่อล้อติด
  [5] 📡 Failsafe Test    - ทดสอบเบรกฉุกเฉินเมื่อ WiFi หลุด
  [6] 🔄 Soft Start Test  - ทดสอบออกตัวนุ่มนวล
  [7] 🎯 Full Test        - ทดสอบครบวงจร (จำลองระบบจริง)
  [0] ❌ ออก
    """)
    
    while True:
        choice = input("  เลือกเมนู [0-7]: ").strip()
        
        if choice == '0':
            print("  ลาก่อน! 👋")
            break
        elif choice == '1':
            test_1_ping()
        elif choice == '2':
            test_2_udp_connection()
        elif choice == '3':
            test_3_move_command()
        elif choice == '4':
            test_4_stall()
        elif choice == '5':
            test_5_failsafe()
        elif choice == '6':
            test_6_soft_start()
        elif choice == '7':
            test_7_full_integration()
        else:
            print("  ❌ กรุณาเลือก 0-7")
        
        print()  # บรรทัดว่าง

if __name__ == "__main__":
    main()
