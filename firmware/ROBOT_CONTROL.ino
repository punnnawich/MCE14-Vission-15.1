#include <WiFi.h>
#include <WiFiUdp.h>
const char* ssid = "MCE14";      
const char* password = "12345678"; 
const int localUdpPort = 12345; 

WiFiUDP udp;

// ตัวแปรสำหรับเช็คข้อมูลที่มาหลงลำดับ (Out-of-order)
uint32_t lastSeqNum = 0;
bool isFirstPacket = true;
// ตัวแปรสำหรับเช็คสถานะ WiFi
unsigned long lastWiFiCheck = 0;
volatile unsigned long lastPacketTime = 0; // เช็ค Failsafe
// เพิ่ม volatile เพื่อความปลอดภัยของข้อมูล
volatile int Counter = 0; 
hw_timer_t * timer = NULL;  
volatile bool rmtloop = false;    

#define Upper1  4
#define Lower1  16
#define ENC1_A  32
#define ENC1_B  33

#define Upper2  17
#define Lower2  18
#define ENC2_A  25
#define ENC2_B  26

#define Upper3  19
#define Lower3  23
#define ENC3_A  34
#define ENC3_B  35

// --- Motor 1 (Front) ---
// ต้องใส่ volatile ทุกตัวที่ onTimer มีการอ่านหรือเขียนค่า
volatile float Kp1 = 1.50;
volatile float Kp2 = 1.50; 
volatile float Kp3 = 1.50;

volatile int pos1 = 0, target1 = 0;
volatile int pos2 = 0, target2 = 0;
volatile int pos3 = 0, target3 = 0;

volatile int controlSignal1 = 0;
volatile int controlSignal2 = 0;
volatile int controlSignal3 = 0;


volatile float PosREAL1 = 0 ;
volatile float PosREAL2 = 0 ;
volatile float PosREAL3 = 0 ;

// ตัวแปรที่ใช้คำนวณภายใน onTimer เท่านั้น ไม่ต้อง volatile ก็ได้ 
// แต่แนะนำให้ประกาศเป็น volatile ไว้ก่อนถ้าจะเอาไป Print ดูใน Loop ครับ
volatile float err1 = 0, err2 = 0, err3 = 0;
volatile int pwm1 = 0, pwm2 = 0, pwm3 = 0;
volatile float Ki1 = 0.01;
volatile float Ki2 = 0.01;
volatile float Ki3 = 0.01; // เริ่มจากค่าน้อยๆ ก่อนเสมอ (เช่น 0.01 - 0.05)
volatile float Kd1 = 0.05;
volatile float Kd2 = 0.05;
volatile float Kd3 = 0.05;
volatile float sumErr1 = 0, sumErr2 = 0, sumErr3 = 0;
volatile float lastErr1 = 0, lastErr2 = 0, lastErr3 = 0;
volatile int maxIntegral = 100;


// PID Variables
float errorValue = 0; 

int controlSignal = 0;
float targetPosition = 0;
volatile float velocurren = 0;
volatile int currentPos = 0;
volatile float Countelast = 0;
const float dt = 0.01; // แก้จาก int เป็น float
int Volte = 12;//แรงไฟ
bool smp = false;
int PWMvalue1 = 0;
float target =0;
volatile int motorDirection = 0;
float targetX = 0;
float targetY = 0;


// --- ข้อมูลล้อและ Encoder ---
const float WHEEL_DIAMETER = 0.082; // เมตร
const float PPR = 4.0;            // สมมติว่า 11 (ลองเช็คสเปคมอเตอร์อีกที)
const float GEAR_RATIO = 19.0;      // สมมติว่า 30 (ลองเช็คสเปคมอเตอร์อีกที)
const int DECODING_MODE = 4;        // เราใช้ 4x
const float PI_VALUE = 3.14159265;
// 1 รอบล้อจะได้กี่ Tick?
const float TICKS_PER_REV = PPR*GEAR_RATIO*DECODING_MODE; 
// 1 เมตรจะได้กี่ Tick?
const float TICKS_PER_METER = TICKS_PER_REV / (WHEEL_DIAMETER * PI_VALUE);
// Interrupt Service Routine สำหรับ Encoder
const float METER_PER_TICKS = 0.000847;
volatile int step = 0;               // 0: idle, 1: go, 2: pause, 3: back
volatile unsigned long pauseTimer = 0;


volatile int Counter1 = 0; 
volatile int Counter2 = 0; 
volatile int Counter3 = 0; 

// โครงสร้างข้อมูลสำหรับรับ UDP รวดเดียวจบ
struct __attribute__((packed)) UDPPacket {
    uint32_t seqNum;
    float x;
    float y;
    float z;
};


volatile bool readyToCompute = false;
volatile bool isMoving = false;
volatile float cycleCount = 0;        // ตัวนับรอบ
volatile float totalCycles = 65.0; // ต้องจบใน 65 รอบ (0.65 วิ)
volatile float finalTarget = 304.0;  // เป้าหมาย 304 Ticks

volatile float currentTarget1 = 0 ;
volatile float currentTarget2 = 0 ;
volatile float currentTarget3 = 0;
volatile float finalT1 =0;
volatile float finalT2 =0;
volatile float finalT3 =0;
volatile float progress = 0;

void setupWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  // ป้องกันการค้างลูปถาวร กรณีเราเตอร์ล่ม ให้รอแค่ 10 วินาทีต่อรอบ
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP()); 
    udp.begin(localUdpPort);
  } else {
    Serial.println("\nWiFi Failed to connect. Will retry later.");
  }
}

void IRAM_ATTR readEncoderISR1() {
  int a1 = digitalRead(ENC1_A);
  int b1 = digitalRead(ENC1_B);
  static int lastA1 = 0;
  static int lastB1 = 0;

  // --- ส่วนคำนวณ Counter (4x Logic) ---
  if (a1 != lastA1) {
    if (a1 != b1) Counter1++; else Counter1--;
  }
  if (b1 != lastB1) {
    if (b1 == a1) Counter1++; else Counter1--;
  }

  lastA1 = a1;
  lastB1 = b1;
}

void IRAM_ATTR readEncoderISR2() {
  int a2 = digitalRead(ENC2_A);
  int b2 = digitalRead(ENC2_B);
  static int lastA2 = 0;
  static int lastB2 = 0;

  // --- ส่วนคำนวณ Counter (4x Logic) ---
  if (a2 != lastA2) {
    if (a2 != b2) Counter2++; else Counter2--;
  }
  if (b2 != lastB2) {
    if (b2 == a2) Counter2++; else Counter2--;
  }

  lastA2 = a2;
  lastB2 = b2;
}

void IRAM_ATTR readEncoderISR3() {
  int a3 = digitalRead(ENC3_A);
  int b3 = digitalRead(ENC3_B);
  static int lastA3 = 0;
  static int lastB3 = 0;

  // --- ส่วนคำนวณ Counter (4x Logic) ---
  if (a3 != lastA3) {
    if (a3 != b3) Counter3++; else Counter3--;
  }
  if (b3 != lastB3) {
    if (b3 == a3) Counter3++; else Counter3--;
  }

  lastA3 = a3;
  lastB3 = b3;
}

TaskHandle_t MotorTask;

void IRAM_ATTR onTimer() {
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  vTaskNotifyGiveFromISR(MotorTask, &xHigherPriorityTaskWoken);
  if(xHigherPriorityTaskWoken) { portYIELD_FROM_ISR(); }
}

void MotorControlLoop(void * pvParameters) {
  for(;;) {
    // รอสัญญาณจาก Timer (Jitter = 0)
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    if (isMoving) {
            cycleCount++; 

            if (cycleCount <= totalCycles) {
                // 1. คำนวณ progress ช่วงที่กำลังสร้างเส้นทาง
                progress = cycleCount / totalCycles;
            } else {
                // ถ้าหมดเวลา 0.65 วิ แล้ว แต่ยังวิ่งไม่ถึง ให้ค้างเป้าหมายไว้ที่ 100% (รอจนกว่าจะถึง)
                progress = 1.0;
            }

            // --- ส่วนที่ปรับปรุง: การจัดการทิศทางตาม Step ---
            // ใช้ตัวแปร Global currentTarget1, currentTarget2, currentTarget3 แทนการประกาศใหม่
            
            if (step == 1) { // ขาไป: 0.0 -> 1.0 (0 ถึง 304 Ticks)
                currentTarget1 = finalT1 * progress;
                currentTarget2 = finalT2 * progress;
                currentTarget3 = finalT3 * progress;
            } else if (step == 3) { // ขากลับ: 1.0 -> 0.0 (304 ถึง 0 Ticks)
                currentTarget1 = finalT1 * (1.0 - progress);
                currentTarget2 = finalT2 * (1.0 - progress);
                currentTarget3 = finalT3 * (1.0 - progress);
            }

            // 2. คำนวณ PID (เพิ่ม Kd)
            err1 = currentTarget1 - Counter1;
            if (abs(err1) <= 2) { pwm1 = 0; sumErr1 = 0; lastErr1 = err1; }
            else {
                sumErr1 = constrain(sumErr1 + err1, -maxIntegral, maxIntegral);
                float dErr1 = (err1 - lastErr1) / dt;
                pwm1 = (Kp1 * err1) + (Ki1 * sumErr1) + (Kd1 * dErr1);
                lastErr1 = err1;
            }

            err2 = currentTarget2 - Counter2;
            if (abs(err2) <= 2) { pwm2 = 0; sumErr2 = 0; lastErr2 = err2; }
            else {
                sumErr2 = constrain(sumErr2 + err2, -maxIntegral, maxIntegral);
                float dErr2 = (err2 - lastErr2) / dt;
                pwm2 = (Kp2 * err2) + (Ki2 * sumErr2) + (Kd2 * dErr2);  
                lastErr2 = err2;
            }

            err3 = currentTarget3 - Counter3;
            if (abs(err3) <= 2) { pwm3 = 0; sumErr3 = 0; lastErr3 = err3; }
            else {
                sumErr3 = constrain(sumErr3 + err3, -maxIntegral, maxIntegral);
                float dErr3 = (err3 - lastErr3) / dt;
                pwm3 = (Kp3 * err3) + (Ki3 * sumErr3) + (Kd3 * dErr3); 
                lastErr3 = err3;
            }

            // --- 🛡️ Stall Detection (ระบบเช็คมอเตอร์ไหม้) ---
            static int stallCounter1 = 0, stallCounter2 = 0, stallCounter3 = 0;
            static int lastCnt1 = 0, lastCnt2 = 0, lastCnt3 = 0;
            
            if (abs(pwm1) > 100 && abs(Counter1 - lastCnt1) <= 1) stallCounter1++; else stallCounter1 = 0;
            if (abs(pwm2) > 100 && abs(Counter2 - lastCnt2) <= 1) stallCounter2++; else stallCounter2 = 0;
            if (abs(pwm3) > 100 && abs(Counter3 - lastCnt3) <= 1) stallCounter3++; else stallCounter3 = 0;
            
            if (stallCounter1 > 50 || stallCounter2 > 50 || stallCounter3 > 50) {
                Serial.println("!!! FAULT: MOTOR STALLED !!! EMERGENCY STOP !!!");
                isMoving = false;
                pwm1 = 0; pwm2 = 0; pwm3 = 0;
                stallCounter1 = 0; stallCounter2 = 0; stallCounter3 = 0;
            }
            lastCnt1 = Counter1; lastCnt2 = Counter2; lastCnt3 = Counter3;

            // เช็คว่าหุ่นยนต์เคลื่อนที่ถึงจุดหมายหรือยัง (Error ของทุกต้อต้องน้อยกว่า 5 ticks)
            bool isArrived = (abs(err1) <= 5 && abs(err2) <= 5 && abs(err3) <= 5);

            if (cycleCount > totalCycles && isArrived) {
                // เมื่อจ่ายเป้าหมายครบ 100% แล้ว และ ล้อหมุนถึงเป้าหมายจริงๆ ค่อยหยุด
                isMoving = false; 
                pwm1 = 0; pwm2 = 0; pwm3 = 0;
                sumErr1 = 0; sumErr2 = 0; sumErr3 = 0; // เคลียร์ Integral ทุกครั้งที่จบช่วง

                if (step == 1) {
                    step = 2; // จบขาไป ให้ไปสถานะ "พัก"
                    pauseTimer = millis(); 
                } else if (step == 3) {
                    step = 0; // จบขากลับ ให้หยุดทำงาน
                    Serial.println("Back to Start point.");
                }
            }

            // 3. สั่งงานมอเตอร์
            driveMotor1(constrain(pwm1, -255, 255));
            driveMotor2(constrain(pwm2, -255, 255));
            driveMotor3(constrain(pwm3, -255, 255));
    } else {
        // --- 🛡️ Graceful Stop: ปล่อยให้ระบบ Soft Start ทำงานจนล้อหยุดนิ่ง ---
        driveMotor1(0);
        driveMotor2(0);
        driveMotor3(0);
    }

        // --- ส่วนตรวจสอบการพัก (Delay โดยไม่หยุด CPU) ---
        if (step == 2) {
            if (millis() - pauseTimer > 2000) { // พัก 2 วินาที
                cycleCount = 0; // รีเซ็ตตัวนับเพื่อเริ่มนับ 1-65 ใหม่
                isMoving = true;
                step = 3; // เปลี่ยนเป็นขากลับ
                Serial.println("Returning Home...");
            }
        }
  }
}

void setup() {
  Serial.begin(115200);
  setupWiFi();
   xTaskCreatePinnedToCore(
    MotorControlLoop,   /* ฟังก์ชันที่จะรัน */
    "MotorTask",        /* ชื่อ Task */
    10000,              /* Stack size */
    NULL,               /* Parameter */
    1,                  /* Priority */
    &MotorTask,         /* Handle */
    0                   /* Core 0 */
  );
  
  pinMode(ENC1_A, INPUT_PULLUP);
  pinMode(ENC1_B, INPUT_PULLUP);
  
  pinMode(ENC2_A, INPUT_PULLUP);
  pinMode(ENC2_B, INPUT_PULLUP);
  
  pinMode(ENC3_A, INPUT_PULLUP);
  pinMode(ENC3_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC1_A), readEncoderISR1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC1_B), readEncoderISR1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_A), readEncoderISR2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC2_B), readEncoderISR2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC3_A), readEncoderISR3, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC3_B), readEncoderISR3, CHANGE);
  
  // ตั้งค่า PWM สำหรับ ESP32
  ledcSetup(0, 5000, 8); 
  ledcSetup(1, 5000, 8); 
  ledcAttachPin(Upper1, 0); 
  ledcAttachPin(Lower1, 1);

  ledcSetup(2, 5000, 8); 
  ledcSetup(3, 5000, 8); 
  ledcAttachPin(Upper2, 2); 
  ledcAttachPin(Lower2, 3);

  ledcSetup(4, 5000, 8); 
  ledcSetup(5, 5000, 8); 
  ledcAttachPin(Upper3, 4); 
  ledcAttachPin(Lower3, 5);
  
  timer = timerBegin(0, 80, true);
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 10000, true);
  timerAlarmEnable(timer);

  // 4. สร้าง Task ให้ไปรันที่ Core 0
 
}

void loop() {
  // --- 🛡️ Communication Failsafe (Heartbeat) ---
  if (isMoving && millis() - lastPacketTime > 1500) {
      Serial.println("!!! FAULT: CONNECTION LOST !!! EMERGENCY STOP !!!");
      isMoving = false; // ตัดไฟ
  }

  // ระบบเชื่อมต่ออัตโนมัติ
  if (millis() - lastWiFiCheck > 5000) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi connection lost. Reconnecting...");
      setupWiFi();
    }
    lastWiFiCheck = millis();
  }

  int packetSize = udp.parsePacket();
  if (packetSize) {
    // --- เพิ่มเพื่อเช็คว่าบอร์ดได้รับข้อมูลจริงๆ ---
    Serial.print("!!! Received Packet Size: ");
    Serial.println(packetSize);
    
    if (packetSize == sizeof(UDPPacket)) {
      UDPPacket pkt;
      udp.read((char*)&pkt, sizeof(UDPPacket));
      lastPacketTime = millis(); // อัพเดตเวลา Failsafe
      
      uint32_t seqNum = pkt.seqNum;
      float x = pkt.x;
      float y = pkt.y;
      float z = pkt.z;
      
      // ปรับปรุง: ยอมรับ Sequence ที่กระโดดลงมากๆ (สมมติว่า Python รีสตาร์ท) 
      // หรือแพ็กเกจแรก หรือแพ็กเกจที่ใหม่กว่าเดิม
      if (isFirstPacket || seqNum > lastSeqNum || (lastSeqNum > seqNum + 1000)) {
        lastSeqNum = seqNum;
        isFirstPacket = false;
        
        // รับงานใหม่ (เมื่อ step == 0 เท่านั้น และต้องมีค่าเป้าหมายที่ไม่ใช่ 0 ทั้งหมด)
        if (step == 0 && (abs(x) > 0.01 || abs(y) > 0.01)) {
            float x_m = x / 100.0;
            float y_m = y / 100.0;

            float tx = x_m * TICKS_PER_METER;
            float ty = y_m * TICKS_PER_METER;

            finalT1 = ty;
            finalT2 = -0.866 * tx - 0.5 * ty;
            finalT3 =  0.866 * tx - 0.5 * ty;

            cycleCount = 0;
            // รีเซ็ต Counter เป็น 0 เพื่อเริ่มนับระยะใหม่จากจุดปัจจุบัน
            Counter1 = 0; Counter2 = 0; Counter3 = 0; 
            
            step = 1;      
            isMoving = true;
            
            Serial.println("\n>>>> MISSION START <<<<");
            Serial.printf("Target M: X:%.2f, Y:%.2f\n", x_m, y_m);
            // ปริ้นท์ค่าที่รับมาเฉพาะตอนเริ่มงานเพื่อลดภาระ CPU
            Serial.printf("Seq:%d\tX:%.2f Y:%.2f Z:%.2f\n", seqNum, x, y, z);
        }
      }
    } else {
      udp.flush(); 
    }
  }
}
void driveMotor1(int controlSignal1) {
    static int currentPWM = 0;
    static int currentDir = 0;
    
    int targetDir = 0;
    if (controlSignal1 > 0) targetDir = 1;
    else if (controlSignal1 < 0) targetDir = -1;
    
    int targetPWM = 0;
    if (abs(controlSignal1) > 0) targetPWM = abs(controlSignal1) + 45; // Deadzone
    if (targetPWM > 255) targetPWM = 255;
    if (abs(err1) <= 2) targetPWM = 0; // Tolerance (ใช้ 2 แทน 1)

    // 🛡️ Shoot-through Protection
    if (targetDir != currentDir && targetDir != 0 && currentPWM > 0) {
        targetPWM = 0; // บังคับลดไฟเป็น 0 ก่อนกลับทิศ
    }
    
    // 🛡️ Slew Rate Limiting (Soft Start/Stop)
    int MAX_STEP = 20; 
    if (targetPWM > currentPWM) {
        currentPWM += MAX_STEP;
        if (currentPWM > targetPWM) currentPWM = targetPWM;
    } else if (targetPWM < currentPWM) {
        currentPWM -= MAX_STEP;
        if (currentPWM < targetPWM) currentPWM = targetPWM;
    }

    if (currentPWM == 0) currentDir = targetDir;

    if (currentDir == 1) {
        ledcWrite(0, 0); ledcWrite(1, currentPWM);
    } else if (currentDir == -1) {
        ledcWrite(0, currentPWM); ledcWrite(1, 0);
    } else {
        ledcWrite(0, 0); ledcWrite(1, 0);
    }
}

void driveMotor2(int controlSignal2) {
    static int currentPWM = 0;
    static int currentDir = 0;
    
    int targetDir = 0;
    if (controlSignal2 > 0) targetDir = 1;
    else if (controlSignal2 < 0) targetDir = -1;
    
    int targetPWM = 0;
    if (abs(controlSignal2) > 0) targetPWM = abs(controlSignal2) + 45; 
    if (targetPWM > 255) targetPWM = 255;
    if (abs(err2) <= 2) targetPWM = 0; 

    // 🛡️ Shoot-through Protection
    if (targetDir != currentDir && targetDir != 0 && currentPWM > 0) targetPWM = 0;
    
    // 🛡️ Slew Rate Limiting
    int MAX_STEP = 20; 
    if (targetPWM > currentPWM) {
        currentPWM += MAX_STEP;
        if (currentPWM > targetPWM) currentPWM = targetPWM;
    } else if (targetPWM < currentPWM) {
        currentPWM -= MAX_STEP;
        if (currentPWM < targetPWM) currentPWM = targetPWM;
    }

    if (currentPWM == 0) currentDir = targetDir;

    if (currentDir == 1) {
        ledcWrite(2, 0); ledcWrite(3, currentPWM);
    } else if (currentDir == -1) {
        ledcWrite(2, currentPWM); ledcWrite(3, 0);
    } else {
        ledcWrite(2, 0); ledcWrite(3, 0);
    }
}

void driveMotor3(int controlSignal3) {
    static int currentPWM = 0;
    static int currentDir = 0;
    
    int targetDir = 0;
    if (controlSignal3 > 0) targetDir = 1;
    else if (controlSignal3 < 0) targetDir = -1;
    
    int targetPWM = 0;
    if (abs(controlSignal3) > 0) targetPWM = abs(controlSignal3) + 45; 
    if (targetPWM > 255) targetPWM = 255;
    if (abs(err3) <= 2) targetPWM = 0; 

    // 🛡️ Shoot-through Protection
    if (targetDir != currentDir && targetDir != 0 && currentPWM > 0) targetPWM = 0;
    
    // 🛡️ Slew Rate Limiting
    int MAX_STEP = 20; 
    if (targetPWM > currentPWM) {
        currentPWM += MAX_STEP;
        if (currentPWM > targetPWM) currentPWM = targetPWM;
    } else if (targetPWM < currentPWM) {
        currentPWM -= MAX_STEP;
        if (currentPWM < targetPWM) currentPWM = targetPWM;
    }

    if (currentPWM == 0) currentDir = targetDir;

    if (currentDir == 1) {
        ledcWrite(4, 0); ledcWrite(5, currentPWM);
    } else if (currentDir == -1) {
        ledcWrite(4, currentPWM); ledcWrite(5, 0);
    } else {
        ledcWrite(4, 0); ledcWrite(5, 0);
    }
}