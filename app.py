#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无线麦克风音频分析应用程序
基于ESP32S3 Sense的实时音频数据分析
"""

import sys
import time
import numpy as np
import serial
import serial.tools.list_ports
from collections import deque
from threading import Thread, Lock
import librosa
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                              QHBoxLayout, QWidget, QPushButton, QComboBox, 
                              QLabel, QGridLayout, QGroupBox, QSpinBox,
                              QStatusBar, QMessageBox)
from PySide6.QtCore import QTimer, Signal, QThread, QObject
from PySide6.QtGui import QFont
import pyqtgraph as pg


class AudioAnalyzer:
    """音频分析器类，处理各种音频特征计算"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.window_size = 1024
        self.hop_length = 512
        
    def calculate_rms(self, data):
        """计算RMS (Root Mean Square)"""
        if len(data) == 0:
            return 0
        return np.sqrt(np.mean(data**2))
    
    def calculate_spl(self, rms_value, reference=1.0):
        """计算声压级 (Sound Pressure Level) in dB"""
        if rms_value <= 0:
            return -np.inf
        return 20 * np.log10(rms_value / reference)
    
    def calculate_spectrum(self, data):
        """计算频谱"""
        if len(data) < self.window_size:
            data = np.pad(data, (0, self.window_size - len(data)), 'constant')
        
        # 应用汉明窗
        windowed_data = data * signal.windows.hamming(len(data))
        
        # FFT计算
        spectrum = fft(windowed_data)
        freqs = fftfreq(len(windowed_data), 1/self.sample_rate)
        
        # 只取正频率部分
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(spectrum[:len(spectrum)//2])
        
        return positive_freqs, magnitude
    
    def find_peak_frequency(self, freqs, magnitude):
        """找到频谱峰值频率"""
        if len(magnitude) == 0:
            return 0, 0
        peak_idx = np.argmax(magnitude)
        return freqs[peak_idx], magnitude[peak_idx]
    
    def calculate_harmonics(self, freqs, magnitude, fundamental_freq, num_harmonics=5):
        """计算谐波特征"""
        harmonics = []
        if fundamental_freq <= 0:
            return harmonics
            
        for i in range(1, num_harmonics + 1):
            target_freq = fundamental_freq * i
            # 找到最接近的频率索引
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            if freq_idx < len(magnitude):
                harmonics.append(magnitude[freq_idx])
            else:
                harmonics.append(0)
        
        return harmonics
    
    def calculate_mfcc(self, data, n_mfcc=13):
        """计算MFCC (Mel-frequency cepstral coefficients)"""
        if len(data) < self.window_size:
            data = np.pad(data, (0, self.window_size - len(data)), 'constant')
        
        try:
            mfccs = librosa.feature.mfcc(y=data.astype(float), 
                                       sr=self.sample_rate, 
                                       n_mfcc=n_mfcc,
                                       hop_length=self.hop_length)
            return np.mean(mfccs, axis=1)
        except:
            return np.zeros(n_mfcc)
    
    def calculate_loudness(self, data):
        """计算响度 (基于A加权)"""
        if len(data) == 0:
            return 0
        
        # 简化的A加权响度计算
        rms = self.calculate_rms(data)
        # A加权近似
        loudness = rms * 1.0  # 简化版本
        return loudness
    
    def calculate_sharpness(self, freqs, magnitude):
        """计算尖锐度"""
        if len(magnitude) == 0:
            return 0
        
        # 高频能量与总能量的比值
        high_freq_threshold = 4000  # Hz
        high_freq_indices = freqs > high_freq_threshold
        
        total_energy = np.sum(magnitude**2)
        high_freq_energy = np.sum(magnitude[high_freq_indices]**2)
        
        if total_energy > 0:
            return high_freq_energy / total_energy
        return 0


class SerialWorker(QObject):
    """串口数据接收工作线程"""
    data_received = Signal(int)
    connection_status = Signal(bool, str)
    
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.is_running = False
        
    def connect_serial(self, port, baudrate=115200):
        """连接串口"""
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
            
            self.serial_port = serial.Serial(port, baudrate, timeout=1)
            self.is_running = True
            self.connection_status.emit(True, f"已连接到 {port}")
            return True
            
        except Exception as e:
            self.connection_status.emit(False, f"连接失败: {str(e)}")
            return False
    
    def disconnect_serial(self):
        """断开串口连接"""
        self.is_running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.connection_status.emit(False, "已断开连接")
    
    def read_data(self):
        """读取串口数据"""
        data_count = 0
        last_debug_time = time.time()
        
        while self.is_running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    
                    # 调试输出
                    current_time = time.time()
                    if current_time - last_debug_time > 2.0:  # 每2秒输出一次
                        print(f"串口调试: 已接收 {data_count} 个样本，最新数据: {line[:50]}")
                        last_debug_time = current_time
                    
                    # 更宽松的数据验证
                    if line:
                        try:
                            # 尝试解析为数字
                            sample = int(line)
                            # 检查数据范围（16位有符号整数）
                            if -32768 <= sample <= 32767:
                                self.data_received.emit(sample)
                                data_count += 1
                        except ValueError:
                            # 跳过非数字行（如状态信息）
                            if not line.startswith(('状态:', '警告:', 'ESP32S3')):
                                print(f"跳过无效数据: {line}")
                                
                time.sleep(0.001)  # 短暂延时避免CPU占用过高
                
            except Exception as e:
                print(f"读取数据错误: {e}")
                time.sleep(0.1)  # 错误后稍长延时


class AudioAnalysisGUI(QMainWindow):
    """主界面类"""
    
    def __init__(self):
        super().__init__()
        
        # 音频参数
        self.sample_rate = 16000
        self.buffer_size = 2048
        self.display_samples = 1000
        
        # 数据缓冲区
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.display_buffer = deque(maxlen=self.display_samples)
        
        # 音频分析器
        self.analyzer = AudioAnalyzer(self.sample_rate)
        
        # 串口工作线程
        self.serial_worker = SerialWorker()
        self.serial_thread = QThread()
        self.serial_worker.moveToThread(self.serial_thread)
        self.serial_worker.data_received.connect(self.on_data_received)
        self.serial_worker.connection_status.connect(self.on_connection_status)
        
        # 界面初始化
        self.setup_ui()
        
        # 定时器用于更新显示
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        self.update_timer.start(50)  # 20 FPS
        
        # 数据锁
        self.data_lock = Lock()
        
        # 启动串口线程
        self.serial_thread.start()
    
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("ESP32S3 音频分析器")
        self.setGeometry(100, 100, 1200, 800)
        
        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 顶部控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 图表区域
        charts_layout = QHBoxLayout()
        
        # 左侧：波形和频谱
        left_panel = self.create_left_panel()
        charts_layout.addWidget(left_panel, 2)
        
        # 右侧：参数显示
        right_panel = self.create_right_panel()
        charts_layout.addWidget(right_panel, 1)
        
        main_layout.addLayout(charts_layout)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("准备就绪")
        
        # 添加数据接收指示器
        from PySide6.QtWidgets import QProgressBar
        self.data_indicator = QProgressBar()
        self.data_indicator.setMaximum(1000)  # 最大显示样本数
        self.data_indicator.setValue(0)
        self.data_indicator.setTextVisible(True)
        self.data_indicator.setFormat("缓冲区: %v/%m")
        self.status_bar.addPermanentWidget(self.data_indicator)
    
    def create_control_panel(self):
        """创建控制面板"""
        group_box = QGroupBox("连接设置")
        layout = QHBoxLayout(group_box)
        
        # COM口选择
        layout.addWidget(QLabel("COM口:"))
        self.port_combo = QComboBox()
        self.refresh_ports()
        layout.addWidget(self.port_combo)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_ports)
        layout.addWidget(refresh_btn)
        
        # 连接按钮
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        layout.addWidget(self.connect_btn)
        
        # 波特率设置
        layout.addWidget(QLabel("波特率:"))
        self.baudrate_spin = QSpinBox()
        self.baudrate_spin.setRange(9600, 921600)
        self.baudrate_spin.setValue(115200)
        layout.addWidget(self.baudrate_spin)
        
        layout.addStretch()
        
        return group_box
    
    def create_left_panel(self):
        """创建左侧图表面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 波形显示
        waveform_group = QGroupBox("实时波形")
        waveform_layout = QVBoxLayout(waveform_group)
        
        self.waveform_plot = pg.PlotWidget(title="音频波形")
        self.waveform_plot.setLabel('left', '幅值')
        self.waveform_plot.setLabel('bottom', '样本')
        self.waveform_plot.setYRange(-32768, 32767)
        self.waveform_curve = self.waveform_plot.plot(pen='b')
        waveform_layout.addWidget(self.waveform_plot)
        
        layout.addWidget(waveform_group)
        
        # 频谱显示
        spectrum_group = QGroupBox("频谱分析")
        spectrum_layout = QVBoxLayout(spectrum_group)
        
        self.spectrum_plot = pg.PlotWidget(title="频率频谱")
        self.spectrum_plot.setLabel('left', '幅值 (dB)')
        self.spectrum_plot.setLabel('bottom', '频率 (Hz)')
        self.spectrum_plot.setLogMode(x=False, y=True)
        self.spectrum_curve = self.spectrum_plot.plot(pen='r')
        spectrum_layout.addWidget(self.spectrum_plot)
        
        layout.addWidget(spectrum_group)
        
        return widget
    
    def create_right_panel(self):
        """创建右侧参数显示面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 基础参数
        basic_group = QGroupBox("基础参数")
        basic_layout = QGridLayout(basic_group)
        
        # RMS值
        basic_layout.addWidget(QLabel("RMS:"), 0, 0)
        self.rms_label = QLabel("0.00")
        basic_layout.addWidget(self.rms_label, 0, 1)
        
        # 声压级
        basic_layout.addWidget(QLabel("声压级 (dB):"), 1, 0)
        self.spl_label = QLabel("0.00")
        basic_layout.addWidget(self.spl_label, 1, 1)
        
        # 峰值频率
        basic_layout.addWidget(QLabel("峰值频率 (Hz):"), 2, 0)
        self.peak_freq_label = QLabel("0.00")
        basic_layout.addWidget(self.peak_freq_label, 2, 1)
        
        # 峰值幅度
        basic_layout.addWidget(QLabel("峰值幅度:"), 3, 0)
        self.peak_mag_label = QLabel("0.00")
        basic_layout.addWidget(self.peak_mag_label, 3, 1)
        
        layout.addWidget(basic_group)
        
        # 高级参数
        advanced_group = QGroupBox("高级参数")
        advanced_layout = QGridLayout(advanced_group)
        
        # 响度
        advanced_layout.addWidget(QLabel("响度:"), 0, 0)
        self.loudness_label = QLabel("0.00")
        advanced_layout.addWidget(self.loudness_label, 0, 1)
        
        # 尖锐度
        advanced_layout.addWidget(QLabel("尖锐度:"), 1, 0)
        self.sharpness_label = QLabel("0.00")
        advanced_layout.addWidget(self.sharpness_label, 1, 1)
        
        layout.addWidget(advanced_group)
        
        # MFCC参数
        mfcc_group = QGroupBox("MFCC系数")
        mfcc_layout = QVBoxLayout(mfcc_group)
        
        self.mfcc_plot = pg.PlotWidget(title="MFCC")
        self.mfcc_plot.setLabel('left', '系数值')
        self.mfcc_plot.setLabel('bottom', 'MFCC索引')
        self.mfcc_curve = self.mfcc_plot.plot(pen='g', symbol='o')
        mfcc_layout.addWidget(self.mfcc_plot)
        
        layout.addWidget(mfcc_group)
        
        # 谐波分析
        harmonics_group = QGroupBox("谐波分析")
        harmonics_layout = QVBoxLayout(harmonics_group)
        
        self.harmonics_plot = pg.PlotWidget(title="谐波")
        self.harmonics_plot.setLabel('left', '幅值')
        self.harmonics_plot.setLabel('bottom', '谐波次数')
        self.harmonics_curve = self.harmonics_plot.plot(pen='m', symbol='s')
        harmonics_layout.addWidget(self.harmonics_plot)
        
        layout.addWidget(harmonics_group)
        
        layout.addStretch()
        
        return widget
    
    def refresh_ports(self):
        """刷新可用COM口"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}")
    
    def toggle_connection(self):
        """切换连接状态"""
        if self.connect_btn.text() == "连接":
            port_text = self.port_combo.currentText()
            if port_text:
                port = port_text.split(' - ')[0]
                baudrate = self.baudrate_spin.value()
                
                if self.serial_worker.connect_serial(port, baudrate):
                    # 在新线程中开始读取数据
                    thread = Thread(target=self.serial_worker.read_data, daemon=True)
                    thread.start()
                    
                    self.connect_btn.setText("断开")
        else:
            self.serial_worker.disconnect_serial()
            self.connect_btn.setText("连接")
    
    def on_data_received(self, sample):
        """处理接收到的数据"""
        with self.data_lock:
            self.audio_buffer.append(sample)
            self.display_buffer.append(sample)
            
            # 每1000个样本输出一次调试信息
            if len(self.audio_buffer) % 1000 == 0:
                print(f"缓冲区调试: audio_buffer={len(self.audio_buffer)}, display_buffer={len(self.display_buffer)}, sample={sample}")
    
    def on_connection_status(self, connected, message):
        """处理连接状态变化"""
        self.status_bar.showMessage(message)
        if not connected:
            self.connect_btn.setText("连接")
    
    def update_displays(self):
        """更新显示"""
        with self.data_lock:
            # 检查数据缓冲区状态
            display_len = len(self.display_buffer)
            audio_len = len(self.audio_buffer)
            
            if display_len < 10:
                # 显示缓冲区状态信息
                self.status_bar.showMessage(f"等待数据... (缓冲区: {display_len}/{self.display_samples})")
                self.data_indicator.setValue(display_len)
                return
            
            # 获取数据副本
            waveform_data = np.array(list(self.display_buffer))
            
            if audio_len >= 512:
                analysis_data = np.array(list(self.audio_buffer)[-1024:])
            else:
                analysis_data = waveform_data.copy()
            
            # 状态更新
            self.status_bar.showMessage(f"实时数据 - 波形点数: {len(waveform_data)}, 分析点数: {len(analysis_data)}")
            self.data_indicator.setValue(display_len)
        
        # 更新波形显示
        if len(waveform_data) > 0:
            # 添加时间轴
            time_axis = np.arange(len(waveform_data))
            self.waveform_curve.setData(time_axis, waveform_data)
            print(f"波形更新: {len(waveform_data)} 点, 范围: [{np.min(waveform_data)}, {np.max(waveform_data)}]")
        
        if len(analysis_data) > 0:
            # 计算各种参数
            rms = self.analyzer.calculate_rms(analysis_data)
            spl = self.analyzer.calculate_spl(rms)
            
            # 频谱分析
            freqs, magnitude = self.analyzer.calculate_spectrum(analysis_data)
            
            if len(magnitude) > 0:
                # 转换为dB
                magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
                self.spectrum_curve.setData(freqs, magnitude_db)
                
                # 峰值频率
                peak_freq, peak_mag = self.analyzer.find_peak_frequency(freqs, magnitude)
                
                # MFCC
                mfcc = self.analyzer.calculate_mfcc(analysis_data)
                self.mfcc_curve.setData(range(len(mfcc)), mfcc)
                
                # 谐波分析
                harmonics = self.analyzer.calculate_harmonics(freqs, magnitude, peak_freq)
                if harmonics and len(harmonics) > 0:
                    harmonic_indices = list(range(1, len(harmonics)+1))
                    self.harmonics_curve.setData(harmonic_indices, harmonics)
                    print(f"谐波更新: {len(harmonics)} 个谐波, 基频: {peak_freq:.1f} Hz")
                else:
                    # 显示空数据
                    self.harmonics_curve.setData([1], [0])
                    print(f"谐波分析: 无有效谐波数据, 基频: {peak_freq:.1f} Hz")
                
                # 响度和尖锐度
                loudness = self.analyzer.calculate_loudness(analysis_data)
                sharpness = self.analyzer.calculate_sharpness(freqs, magnitude)
                
                # 更新标签
                self.rms_label.setText(f"{rms:.2f}")
                self.spl_label.setText(f"{spl:.2f}")
                self.peak_freq_label.setText(f"{peak_freq:.2f}")
                self.peak_mag_label.setText(f"{peak_mag:.2f}")
                self.loudness_label.setText(f"{loudness:.3f}")
                self.sharpness_label.setText(f"{sharpness:.3f}")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        self.serial_worker.disconnect_serial()
        self.serial_thread.quit()
        self.serial_thread.wait()
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("ESP32S3 音频分析器")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口
    window = AudioAnalysisGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
