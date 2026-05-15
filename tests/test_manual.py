"""
===========================================
  🎮 ระบบทดสอบหุ่นยนต์แบบ Manual
  กรอกค่า X, Y เอง แล้วหุ่นจะวิ่งไปตามที่สั่ง
===========================================
วิธีใช้: python test_manual.py
"""
import socket
import struct
import time

# ==========================================
# ⚙️ ตั้งค่า IP ของ ESP32
# ==========================================
ESP32_IPS = [
    "192.168.137.123",
    # "192.168.137.243",  # เอา # ออกถ้ามีบอร์ดตัวที่ 2
]
ESP32_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
seq = 0

def send(x, y, z=0.0):
    global seq
    data = struct.pack('<Ifff', seq, float(x), float(y), float(z))
    for ip in ESP32_IPS:
        sock.sendto(data, (ip, ESP32_PORT))
    seq += 1

print("=" * 50)
print("  🎮 ระบบทดสอบหุ่นยนต์แบบ Manual")
print(f"  📡 เป้าหมาย: {', '.join(ESP32_IPS)}")
print("=" * 50)
print("  พิมพ์ค่า X Y (หน่วย cm) แล้วกด Enter")
print("  ตัวอย่าง: 20 30")
print("  พิมพ์ 'q' เพื่อออก")
print("=" * 50)

while True:
    try:
        cmd = input("\n  🎯 ป้อน X Y (cm): ").strip()
        
        if cmd.lower() == 'q':
            print("  👋 ลาก่อน!")
            break
        
        parts = cmd.replace(",", " ").split()
        if len(parts) != 2:
            print("  ❌ กรุณาป้อน 2 ค่า เช่น: 20 30")
            continue
        
        x = float(parts[0])
        y = float(parts[1])
        
        print(f"  📤 ส่ง X={x:.1f}cm Y={y:.1f}cm ...")
        
        # ส่ง 5 ครั้งซ้ำเพื่อให้แน่ใจว่าบอร์ดจับได้
        for i in range(5):
            send(x, y)
            time.sleep(0.03)
        
        print(f"  ✅ ส่งแล้ว! ดู Serial Monitor")
        
    except ValueError:
        print("  ❌ กรุณาป้อนตัวเลข เช่น: 20 30")
    except KeyboardInterrupt:
        print("\n  👋 ลาก่อน!")
        break

sock.close()
