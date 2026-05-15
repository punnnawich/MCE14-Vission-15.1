import socket
import time
import struct

# ==========================================
# การตั้งค่า (Configuration)
# ==========================================
# ใส่ IP ของ ESP32 ทุกตัวที่ต้องการส่งไปหา คั่นด้วยลูกน้ำ
ESP32_IPS = [
    "192.168.137.123",
    "192.168.137.243", # ใส่ IP ตัวที่ 2, 3, 4 ตรงนี้ได้เลย
]
ESP32_PORT = 12345

class UDPSender:
    def __init__(self, ips, port):
        # รองรับทั้งแบบใส่ IP เดียว (String) และหลาย IP (List)
        self.ips = ips if isinstance(ips, list) else [ips]
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # ตัวแปรสำหรับนับลำดับ Packet (Sequence Number)
        self.packet_seq = 0 

    def send_data_binary(self, x, y, z, repeats=3):
        """
        ส่งพิกัดพร้อม Sequence Number (แพ็กเป็น Binary 16 bytes)
        '<Ifff' หมายถึง: 
        - < (Little Endian)
        - I (unsigned int, 4 bytes) สำหรับ Sequence Number
        - f (float, 4 bytes) สำหรับ x
        - f (float, 4 bytes) สำหรับ y
        - f (float, 4 bytes) สำหรับ z
        รวมเป็น 16 bytes
        
        repeats: จำนวนครั้งที่ส่งซ้ำ (ลด Packet Loss จาก ~1% เหลือ ~0.0001%)
        ESP32 จะ deduplicate ด้วย Sequence Number ที่เหมือนกัน
        """
        data = struct.pack('<Ifff', self.packet_seq, float(x), float(y), float(z))
        
        # +++ [H3] Redundant Send: ส่งซ้ำหลายรอบเพื่อลด Packet Loss +++
        for _ in range(repeats):
            for ip in self.ips:
                self.sock.sendto(data, (ip, self.port))
        
        # แสดง log บนหน้าจอ Terminal/Command Prompt
        print(f"[UDP_SENDER] ส่งไป {len(self.ips)} บอร์ด x{repeats} -> ลำดับ: {self.packet_seq} | X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")
        
        self.packet_seq += 1 # เพิ่มลำดับทุกครั้งที่ส่ง

    def close(self):
        self.sock.close()

if __name__ == "__main__":
    sender = UDPSender(ESP32_IPS, ESP32_PORT)
    print(f"🚀 เริ่มส่ง UDP ไปที่ {len(ESP32_IPS)} บอร์ด (Port: {ESP32_PORT})")
    
    try:
        x, y, z = 0.0, 0.0, 0.0
        while True:
            # จำลองข้อมูล
            x += 0.1; y += 0.2; z += 0.3
            
            # ส่งข้อมูลไป ESP32
            sender.send_data_binary(x, y, z)
            
            # **ข้อควรระวัง:** ไม่ควรส่งเร็วเกินไป (เช่น ไม่มี delay เลย) 
            # เพราะจะเกิด Data Flood ทำลาย Buffer ของ ESP32
            # ควรตั้งค่า Delay ให้ใกล้เคียงกับ Camera FPS (เช่น 30 FPS = ~0.033s)
            time.sleep(1/30.0) 
            
    except KeyboardInterrupt:
        print("\nหยุดการทำงาน")
    finally:
        sender.close()
