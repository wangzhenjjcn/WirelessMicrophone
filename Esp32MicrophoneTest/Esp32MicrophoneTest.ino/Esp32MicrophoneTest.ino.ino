#include <ESP_I2S.h>

I2SClass I2S;

// 配置参数
const int SAMPLE_RATE = 16000;
const int BUFFER_SIZE = 64;  // 批量发送的样本数
const int BAUD_RATE = 115200;

// 数据缓冲区
int16_t audioBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// 状态指示
bool i2sInitialized = false;
unsigned long lastDataTime = 0;
unsigned long dataCount = 0;

void setup() {
  // 初始化串口通信
  Serial.begin(BAUD_RATE);
  while (!Serial) {
    delay(10); // 等待串口连接
  }
  
  Serial.println("ESP32S3 音频采集器启动中...");
  
  // 设置PDM引脚 (42=时钟, 41=数据)
  I2S.setPinsPdmRx(42, 41);
  
  // 初始化I2S (PDM模式, 16kHz采样率, 16位深度, 单声道)
  if (!I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("错误: I2S初始化失败!");
    while (1) {
      delay(1000);
      Serial.println("I2S初始化失败，请检查硬件连接");
    }
  }
  
  i2sInitialized = true;
  Serial.println("I2S初始化成功");
  Serial.print("采样率: ");
  Serial.print(SAMPLE_RATE);
  Serial.println(" Hz");
  Serial.println("开始音频数据采集...");
  
  // 给系统一点时间稳定
  delay(500);
  
  // 清空初始的无效数据
  for (int i = 0; i < 100; i++) {
    I2S.read();
    delay(1);
  }
  
  lastDataTime = millis();
}

void loop() {
  if (!i2sInitialized) {
    delay(100);
    return;
  }
  
  // 读取音频样本
  int sample = I2S.read();
  
  // 数据有效性检查
  if (sample != 0 && sample != -1 && sample != 1) {
    // 将样本添加到缓冲区
    audioBuffer[bufferIndex] = (int16_t)sample;
    bufferIndex++;
    dataCount++;
    
    // 缓冲区满时批量发送
    if (bufferIndex >= BUFFER_SIZE) {
      sendAudioData();
      bufferIndex = 0;
    }
  }
  
  // 检查数据流状态
  unsigned long currentTime = millis();
  if (currentTime - lastDataTime > 5000) {  // 5秒无有效数据
    Serial.println("警告: 长时间无有效音频数据");
    lastDataTime = currentTime;
  }
  
  // 每隔一段时间发送状态信息
  static unsigned long lastStatusTime = 0;
  if (currentTime - lastStatusTime > 10000) {  // 每10秒
    Serial.print("状态: 已采集 ");
    Serial.print(dataCount);
    Serial.println(" 个样本");
    lastStatusTime = currentTime;
  }
  
  // 小延时以避免过度占用CPU
  delayMicroseconds(10);
}

void sendAudioData() {
  // 批量发送音频数据
  for (int i = 0; i < BUFFER_SIZE; i++) {
    Serial.println(audioBuffer[i]);
  }
  
  lastDataTime = millis();
  
  // 可选：添加数据包结束标记（如果需要）
  // Serial.println("END_PACKET");
}

// 错误恢复函数
void recoverI2S() {
  Serial.println("尝试恢复I2S连接...");
  
  // 重新初始化I2S
  I2S.end();
  delay(100);
  
  I2S.setPinsPdmRx(42, 41);
  if (I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("I2S恢复成功");
    i2sInitialized = true;
  } else {
    Serial.println("I2S恢复失败");
    i2sInitialized = false;
  }
}