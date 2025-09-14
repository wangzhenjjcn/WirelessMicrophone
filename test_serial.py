#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的串口测试工具
用于测试音频分析应用程序的数据接收功能
"""

import serial
import time
import numpy as np

def create_test_signal(duration=5.0, sample_rate=16000):
    """创建测试信号 - 440Hz基频 + 谐波"""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # 创建复合谐波信号
    fundamental = 440  # 基频440Hz
    amplitude = 8000
    
    signal = (amplitude * 0.7 * np.sin(2 * np.pi * fundamental * t) +        # 基频
             amplitude * 0.4 * np.sin(2 * np.pi * fundamental * 2 * t) +     # 2次谐波  
             amplitude * 0.2 * np.sin(2 * np.pi * fundamental * 3 * t) +     # 3次谐波
             amplitude * 0.1 * np.sin(2 * np.pi * fundamental * 4 * t))      # 4次谐波
    
    return signal.astype(np.int16)

def send_test_data(port_name, baudrate=115200):
    """发送测试数据"""
    try:
        # 连接串口
        ser = serial.Serial(port_name, baudrate, timeout=1)
        print(f"已连接到 {port_name}")
        
        # 生成测试信号
        print("生成测试信号...")
        test_data = create_test_signal(duration=10.0)
        
        print(f"开始发送 {len(test_data)} 个样本...")
        
        # 发送数据
        for i, sample in enumerate(test_data):
            ser.write(f"{sample}\n".encode('utf-8'))
            
            # 控制发送速率 (16kHz采样率)
            time.sleep(1/16000)
            
            # 进度显示
            if i % 1600 == 0:  # 每0.1秒显示一次进度
                progress = (i / len(test_data)) * 100
                print(f"发送进度: {progress:.1f}%")
        
        print("数据发送完成")
        ser.close()
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    print("串口测试工具")
    print("使用方法: python test_serial.py")
    print()
    
    # 简单交互界面
    port = input("请输入串口端口 (例如 COM3): ").strip()
    
    if port:
        send_test_data(port)
    else:
        print("未输入串口端口，程序退出") 