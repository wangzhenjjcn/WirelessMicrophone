import sys
import threading
import queue
import time
import socket
import struct
import wave
import subprocess
import importlib
import shutil
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

try:
    import serial
    import serial.tools.list_ports as list_ports
except Exception as exc:  # pragma: no cover
    serial = None
    list_ports = None

try:
    import esptool
except Exception:  # pragma: no cover
    esptool = None

# GUI
from PySide6.QtCore import QTimer, Qt, Signal, QObject
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
    QPlainTextEdit,
)
import pyqtgraph as pg

TSINGHUA_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

def ensure_dependency(package_spec: str, import_name: str):
    """确保模块可用；若缺失则使用清华源安装后再导入。
    返回导入后的模块对象，失败时返回 None。
    """
    try:
        return importlib.import_module(import_name)
    except Exception:
        pass
    try:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-i",
            TSINGHUA_MIRROR,
            package_spec,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None
    try:
        return importlib.import_module(import_name)
    except Exception:
        return None


@dataclass
class StreamConfig:
    sample_rate_hz: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    frame_size_samples: int = 2048  # UI 读取/分析窗口（更长便于稳定频谱）
    parser_mode: str = "Raw16LE"  # Raw16LE | LengthPrefixedLE | Auto
    calibration_offset_db: float = 0.0  # dBFS -> dB SPL 偏移
    # 频谱平滑系数（0-1，越大越灵敏）
    fft_average_alpha: float = 0.6


class AudioRingBuffer:
    def __init__(self, capacity_samples: int):
        self.capacity = int(capacity_samples)
        self.buffer = np.zeros(self.capacity, dtype=np.int16)
        self.lock = threading.Lock()
        self.write_index = 0
        self.size = 0

    def push_samples(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        with self.lock:
            n = int(samples.size)
            if n >= self.capacity:
                # 只保留最近的一段
                self.buffer[:] = samples[-self.capacity :]
                self.write_index = 0
                self.size = self.capacity
                return
            end_space = self.capacity - self.write_index
            if n <= end_space:
                self.buffer[self.write_index : self.write_index + n] = samples
                self.write_index = (self.write_index + n) % self.capacity
            else:
                self.buffer[self.write_index :] = samples[:end_space]
                self.buffer[: n - end_space] = samples[end_space:]
                self.write_index = n - end_space
            self.size = min(self.capacity, self.size + n)

    def read_latest(self, num_samples: int) -> np.ndarray:
        with self.lock:
            n = min(num_samples, self.size)
            if n == 0:
                return np.zeros(0, dtype=np.int16)
            start = (self.write_index - n) % self.capacity
            if start + n <= self.capacity:
                return self.buffer[start : start + n].copy()
            else:
                part1 = self.buffer[start:]
                part2 = self.buffer[: (start + n) % self.capacity]
                return np.concatenate([part1, part2]).copy()


class StopToken:
    def __init__(self):
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()


class SerialReceiverThread(threading.Thread):
    def __init__(
        self,
        port: str,
        baudrate: int,
        stream_config: StreamConfig,
        ring: AudioRingBuffer,
        stop_token: StopToken,
        status_queue: "queue.Queue[str]",
    ):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.stream_config = stream_config
        self.ring = ring
        self.stop_token = stop_token
        self.status_queue = status_queue
        self.ser: Optional[serial.Serial] = None if serial else None
        self._recv_buffer = bytearray()
        self._last_sample_time = time.time()

    def _open(self) -> bool:
        try:
            if self.ser and self.ser.is_open:
                return True
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            self.status_queue.put(f"串口已打开: {self.port} @{self.baudrate}")
            return True
        except Exception as exc:
            self.status_queue.put(f"打开串口失败: {exc}")
            return False

    def _close(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def run(self):
        if serial is None:
            self.status_queue.put("未安装 pyserial，无法使用串口")
            return
        bytes_per_sample = max(1, self.stream_config.bits_per_sample // 8)
        while not self.stop_token.stopped():
            if not self._open():
                time.sleep(0.5)
                continue
            try:
                chunk = self.ser.read(4096)
            except Exception as exc:
                self.status_queue.put(f"串口读取异常: {exc}")
                self._close()
                time.sleep(0.2)
                continue
            if not chunk:
                continue
            self._recv_buffer.extend(chunk)
            # 解析
            mode = self.stream_config.parser_mode
            if mode == "Auto":
                # 简单探测：尝试长度帧；否则按 Raw16LE
                parsed_any = False
                while len(self._recv_buffer) >= 4:
                    frame_len = struct.unpack_from("<I", self._recv_buffer, 0)[0]
                    if frame_len <= 16384 and len(self._recv_buffer) >= 4 + frame_len and frame_len % 2 == 0:
                        payload = self._recv_buffer[4 : 4 + frame_len]
                        del self._recv_buffer[: 4 + frame_len]
                        samples = np.frombuffer(payload, dtype=np.int16)
                        self.ring.push_samples(samples)
                        self._last_sample_time = time.time()
                        parsed_any = True
                    else:
                        break
                if not parsed_any:
                    avail = len(self._recv_buffer)
                    n = (avail // 2) * 2
                    if n >= 2:
                        raw = self._recv_buffer[:n]
                        del self._recv_buffer[:n]
                        samples = np.frombuffer(raw, dtype=np.int16)
                        self.ring.push_samples(samples)
                        self._last_sample_time = time.time()
            elif self.stream_config.parser_mode == "Raw16LE":
                avail = len(self._recv_buffer)
                n = (avail // 2) * 2
                if n >= 2:
                    raw = self._recv_buffer[:n]
                    del self._recv_buffer[:n]
                    samples = np.frombuffer(raw, dtype=np.int16)
                    self.ring.push_samples(samples)
                    self._last_sample_time = time.time()
            elif self.stream_config.parser_mode == "LengthPrefixedLE":
                while len(self._recv_buffer) >= 4:
                    frame_len = struct.unpack_from("<I", self._recv_buffer, 0)[0]
                    if len(self._recv_buffer) < 4 + frame_len:
                        break
                    payload = self._recv_buffer[4 : 4 + frame_len]
                    del self._recv_buffer[: 4 + frame_len]
                    if frame_len % 2 != 0:
                        continue
                    samples = np.frombuffer(payload, dtype=np.int16)
                    self.ring.push_samples(samples)
                    self._last_sample_time = time.time()
            else:
                avail = len(self._recv_buffer)
                n = (avail // 2) * 2
                if n >= 2:
                    raw = self._recv_buffer[:n]
                    del self._recv_buffer[:n]
                    samples = np.frombuffer(raw, dtype=np.int16)
                    self.ring.push_samples(samples)
                    self._last_sample_time = time.time()

            # 诊断：超过 1.5s 无新样本，提示检查
            if time.time() - self._last_sample_time > 1.5:
                try:
                    self.status_queue.put("警告: 超过1.5秒未解析到音频数据，请检查协议(建议Raw16LE)、波特率(建议921600)、采样率(建议16000)")
                except Exception:
                    pass
        self._close()
        self.status_queue.put("串口接收线程退出")


class TcpReceiverThread(threading.Thread):
    def __init__(
        self,
        host: str,
        port: int,
        stream_config: StreamConfig,
        ring: AudioRingBuffer,
        stop_token: StopToken,
        status_queue: "queue.Queue[str]",
    ):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.stream_config = stream_config
        self.ring = ring
        self.stop_token = stop_token
        self.status_queue = status_queue
        self.sock: Optional[socket.socket] = None
        self._recv_buffer = bytearray()

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(0.2)
            self.status_queue.put(f"TCP 已连接: {self.host}:{self.port}")
        except Exception as exc:
            self.status_queue.put(f"TCP 连接失败: {exc}")
            return
        try:
            while not self.stop_token.stopped():
                try:
                    chunk = self.sock.recv(4096)
                except socket.timeout:
                    continue
                except Exception as exc:
                    self.status_queue.put(f"TCP 接收异常: {exc}")
                    break
                if not chunk:
                    time.sleep(0.01)
                    continue
                self._recv_buffer.extend(chunk)
                if self.stream_config.parser_mode == "Raw16LE":
                    avail = len(self._recv_buffer)
                    n = (avail // 2) * 2
                    if n >= 2:
                        raw = self._recv_buffer[:n]
                        del self._recv_buffer[:n]
                        samples = np.frombuffer(raw, dtype=np.int16)
                        self.ring.push_samples(samples)
                elif self.stream_config.parser_mode == "LengthPrefixedLE":
                    while len(self._recv_buffer) >= 4:
                        frame_len = struct.unpack_from("<I", self._recv_buffer, 0)[0]
                        if len(self._recv_buffer) < 4 + frame_len:
                            break
                        payload = self._recv_buffer[4 : 4 + frame_len]
                        del self._recv_buffer[: 4 + frame_len]
                        if frame_len % 2 != 0:
                            continue
                        samples = np.frombuffer(payload, dtype=np.int16)
                        self.ring.push_samples(samples)
                else:
                    avail = len(self._recv_buffer)
                    n = (avail // 2) * 2
                    if n >= 2:
                        raw = self._recv_buffer[:n]
                        del self._recv_buffer[:n]
                        samples = np.frombuffer(raw, dtype=np.int16)
                        self.ring.push_samples(samples)
        finally:
            try:
                if self.sock:
                    self.sock.close()
            except Exception:
                pass
            self.status_queue.put("TCP 接收线程退出")


class Analyzer:
    def __init__(self, stream_config: StreamConfig):
        self.cfg = stream_config
        self.window = np.hanning(self.cfg.frame_size_samples)
        self.eps = 1e-12
        self.prev_mag_db: Optional[np.ndarray] = None

    def update_frame_size(self, n: int):
        self.cfg.frame_size_samples = int(n)
        self.window = np.hanning(self.cfg.frame_size_samples)
        self.prev_mag_db = None

    def compute_metrics(self, int16_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        if int16_samples.size == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                float("nan"),
                float("nan"),
            )
        # 转换到 float32 [-1,1)
        x = int16_samples.astype(np.float32) / 32768.0
        # 最新一帧
        frame_len = min(self.cfg.frame_size_samples, x.size)
        frame = x[-frame_len:]
        # RMS & dBFS
        rms = float(np.sqrt(np.mean(frame**2) + self.eps))
        dbfs = 20.0 * np.log10(rms + self.eps)
        spl = dbfs + self.cfg.calibration_offset_db
        # 频谱
        w = self.window[:frame_len]
        xw = frame * w
        fft = np.fft.rfft(xw)
        mag = np.abs(fft)
        mag_db = 20.0 * np.log10(mag + self.eps)
        # 简单指数平滑
        if self.prev_mag_db is not None and self.prev_mag_db.shape == mag_db.shape:
            a = float(self.cfg.fft_average_alpha)
            mag_db = a * mag_db + (1.0 - a) * self.prev_mag_db
        self.prev_mag_db = mag_db
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / self.cfg.sample_rate_hz)
        return frame, np.column_stack((freqs, mag_db)), dbfs, spl


class Recorder:
    def __init__(self, sample_rate_hz: int, channels: int = 1):
        self.sample_rate_hz = int(sample_rate_hz)
        self.channels = int(channels)
        self.file: Optional[wave.Wave_write] = None
        self.total_frames = 0
        self.lock = threading.Lock()

    def start(self, wav_path: str):
        with self.lock:
            self.stop()
            wf = wave.open(wav_path, "wb")
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.sample_rate_hz)
            self.file = wf
            self.total_frames = 0

    def write(self, samples: np.ndarray):
        if self.file is None:
            return
        # 仅支持 mono int16
        if samples.dtype != np.int16:
            samples = samples.astype(np.int16)
        with self.lock:
            try:
                self.file.writeframes(samples.tobytes())
                self.total_frames += samples.size // self.channels
            except Exception:
                pass

    def stop(self):
        with self.lock:
            if self.file is not None:
                try:
                    self.file.close()
                except Exception:
                    pass
                self.file = None


class UiSignals(QObject):
    status = Signal(str)
    dbfs = Signal(float)
    spl = Signal(float)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wireless Microphone - XIAO ESP32S3 Sense")
        self.resize(1100, 720)

        # 状态 & 数据
        self.cfg = StreamConfig()
        # 10 秒缓冲
        self.ring = AudioRingBuffer(capacity_samples=self.cfg.sample_rate_hz * 10)
        self.stop_token = StopToken()
        self.rx_thread: Optional[threading.Thread] = None
        self.status_queue: "queue.Queue[str]" = queue.Queue()
        self.analyzer = Analyzer(self.cfg)
        self.recorder = Recorder(self.cfg.sample_rate_hz, self.cfg.channels)
        self.last_build_dir: Optional[str] = None
        self.is_connecting = False
        self.is_connected = False

        # 信号
        self.signals = UiSignals()
        self.signals.status.connect(self._on_status)

        # UI
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)

        # 连接区域
        conn_row = QHBoxLayout()
        vbox.addLayout(conn_row)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Serial", "WiFi (TCP)", "Demo"])
        conn_row.addWidget(QLabel("模式"))
        conn_row.addWidget(self.mode_combo)

        self.port_combo = QComboBox()
        self._refresh_ports()
        self.port_combo.setEditable(True)
        self.port_combo.setCurrentText("COM100")
        conn_row.addWidget(QLabel("串口/地址"))
        conn_row.addWidget(self.port_combo)

        self.baud_spin = QSpinBox()
        self.baud_spin.setRange(9600, 921600)
        self.baud_spin.setValue(921600)
        conn_row.addWidget(QLabel("波特率/端口"))
        conn_row.addWidget(self.baud_spin)

        self.connect_btn = QPushButton("连接")
        self.disconnect_btn = QPushButton("断开")
        conn_row.addWidget(self.connect_btn)
        conn_row.addWidget(self.disconnect_btn)

        self.connect_btn.clicked.connect(self.on_connect)
        self.disconnect_btn.clicked.connect(self.on_disconnect)

        # 配置区域
        cfg_row = QHBoxLayout()
        vbox.addLayout(cfg_row)

        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 96000)
        self.sr_spin.setValue(self.cfg.sample_rate_hz)
        cfg_row.addWidget(QLabel("采样率"))
        cfg_row.addWidget(self.sr_spin)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(256, 16384)
        self.frame_spin.setValue(self.cfg.frame_size_samples)
        cfg_row.addWidget(QLabel("帧长"))
        cfg_row.addWidget(self.frame_spin)

        self.parser_combo = QComboBox()
        self.parser_combo.addItems(["Raw16LE", "LengthPrefixedLE", "Auto"])
        cfg_row.addWidget(QLabel("协议"))
        cfg_row.addWidget(self.parser_combo)

        self.calib_spin = QDoubleSpinBox()
        self.calib_spin.setDecimals(1)
        self.calib_spin.setRange(-200.0, 200.0)
        self.calib_spin.setValue(self.cfg.calibration_offset_db)
        cfg_row.addWidget(QLabel("校准(dB)"))
        cfg_row.addWidget(self.calib_spin)
 
        # 仅保留核心可视化与连接
 
        # 日志输出改为命令行，仅保留绘图区域
        # 数值显示
        meters_row = QHBoxLayout()
        vbox.addLayout(meters_row)
        self.dbfs_label = QLabel("dBFS: --")
        self.spl_label = QLabel("SPL: -- dB")
        meters_row.addWidget(self.dbfs_label)
        meters_row.addWidget(self.spl_label)
 
        # 绘图
        self.plot_widget = pg.GraphicsLayoutWidget()
        vbox.addWidget(self.plot_widget)

        pg.setConfigOptions(antialias=True)
        self.wave_plot = self.plot_widget.addPlot(row=0, col=0, title="波形 (示波器风格)")
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen('g', width=2))
        self.wave_plot.showGrid(x=True, y=True, alpha=0.5)
        # 波形图示波器风格设置
        self.wave_plot.getAxis('left').setPen('g')  # 绿色Y轴
        self.wave_plot.getAxis('bottom').setPen('g')  # 绿色X轴
        self.wave_plot.getAxis('left').setTextPen('g')  # 绿色Y轴文字
        self.wave_plot.getAxis('bottom').setTextPen('g')  # 绿色X轴文字
        self.wave_plot.setLabel('bottom', '时间', units='s')
        self.wave_plot.setLabel('left', '幅度', units='')

        self.fft_plot = self.plot_widget.addPlot(row=1, col=0, title="频谱分析仪 (示波器风格)")
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen('g', width=2))
        self.fft_plot.showGrid(x=True, y=True, alpha=0.5)
        self.fft_plot.setLogMode(x=True, y=False)
        # 示波器风格设置 - 使用正确的PyQtGraph API
        self.plot_widget.setBackground('k')  # 设置整体背景为黑色
        self.fft_plot.getAxis('left').setPen('g')  # 绿色Y轴
        self.fft_plot.getAxis('bottom').setPen('g')  # 绿色X轴
        self.fft_plot.getAxis('left').setTextPen('g')  # 绿色Y轴文字
        self.fft_plot.getAxis('bottom').setTextPen('g')  # 绿色X轴文字
        self.fft_plot.setLabel('bottom', '频率', units='Hz')
        self.fft_plot.setLabel('left', '幅度', units='dB')
        # 初始范围
        self.fft_plot.setXRange(1, self.cfg.sample_rate_hz / 2, padding=0)
        self.fft_plot.setYRange(-80, 20)
        # 定时刷新
        self.timer = QTimer()
        self.timer.setInterval(50)  # 20 FPS
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

        # 状态消息轮询
        self.status_timer = QTimer()
        self.status_timer.setInterval(100)
        self.status_timer.timeout.connect(self._drain_status)
        self.status_timer.start()

        # 事件绑定
        self.sr_spin.valueChanged.connect(self._on_cfg_changed)
        self.frame_spin.valueChanged.connect(self._on_cfg_changed)
        self.parser_combo.currentTextChanged.connect(self._on_cfg_changed)
        self.calib_spin.valueChanged.connect(self._on_cfg_changed)

        # 启动后自动连接 COM100（若存在）
        QTimer.singleShot(300, self._auto_connect_default)

    def closeEvent(self, event):
        self.on_disconnect()
        self.recorder.stop()
        event.accept()

    # UI 事件
    def _refresh_ports(self):
        self.port_combo.clear()
        items: List[str] = []
        if list_ports is not None:
            try:
                for p in list_ports.comports():
                    items.append(p.device)
            except Exception:
                pass
        # WiFi 地址示例
        items.extend(["COM100", "192.168.4.1"])  # 占位
        self.port_combo.addItems(sorted(set(items)))

    def on_connect(self):
        if self.is_connecting or self.is_connected:
            return
        self.is_connecting = True
        mode = self.mode_combo.currentText()
        addr = self.port_combo.currentText().strip()
        if mode == "Serial":
            # 确保 pyserial 可用（清华源）
            global serial, list_ports
            if serial is None:
                self.signals.status.emit("正在安装 pyserial（清华源）…")
                mod = ensure_dependency("pyserial>=3.5", "serial")
                if mod is None:
                    QMessageBox.critical(self, "依赖安装失败", "无法安装 pyserial，请检查网络或权限")
                    self.is_connecting = False
                    return
                serial = mod
                try:
                    list_ports = importlib.import_module("serial.tools.list_ports")
                except Exception:
                    list_ports = None
            try:
                baud = int(self.baud_spin.value())
            except Exception:
                baud = 921600
            self.stop_token = StopToken()
            self.rx_thread = SerialReceiverThread(
                port=addr,
                baudrate=baud,
                stream_config=self.cfg,
                ring=self.ring,
                stop_token=self.stop_token,
                status_queue=self.status_queue,
            )
            self.rx_thread.start()
            self.is_connected = True
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
        elif mode == "WiFi (TCP)":
            # 端口在 baud_spin 中复用
            try:
                tcp_port = int(self.baud_spin.value())
            except Exception:
                tcp_port = 12345
            self.stop_token = StopToken()
            self.rx_thread = TcpReceiverThread(
                host=addr,
                port=tcp_port,
                stream_config=self.cfg,
                ring=self.ring,
                stop_token=self.stop_token,
                status_queue=self.status_queue,
            )
            self.rx_thread.start()
            self.is_connected = True
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
        else:
            # Demo: 启动模拟线程
            self.stop_token = StopToken()
            self.rx_thread = threading.Thread(target=self._demo_feeder, daemon=True)
            self.rx_thread.start()
            self.is_connected = True
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
        self.signals.status.emit(f"连接模式: {mode} -> {addr}")
        self.is_connecting = False

    def on_disconnect(self):
        if self.rx_thread is not None:
            self.stop_token.stop()
            self.rx_thread.join(timeout=1.0)
            self.rx_thread = None
            self.signals.status.emit("已断开")
        self.is_connected = False
        self.is_connecting = False
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)

    def _demo_feeder(self):
        t = 0.0
        sr = float(self.cfg.sample_rate_hz)
        while not self.stop_token.stopped():
            dur = 0.03
            n = int(sr * dur)
            tt = (t + np.arange(n) / sr)
            sig = 0.2 * np.sin(2 * np.pi * 1000.0 * tt)  # 1 kHz
            t += dur
            samples = np.clip(sig, -0.999, 0.999)
            i16 = np.asarray(np.round(samples * 32767.0), dtype=np.int16)
            self.ring.push_samples(i16)
            self.recorder.write(i16)
            time.sleep(dur * 0.8)

    def _on_cfg_changed(self):
        self.cfg.sample_rate_hz = int(self.sr_spin.value())
        self.analyzer.update_frame_size(int(self.frame_spin.value()))
        self.cfg.parser_mode = self.parser_combo.currentText()
        self.cfg.calibration_offset_db = float(self.calib_spin.value())
        # 录音采样率更新
        self.recorder.sample_rate_hz = self.cfg.sample_rate_hz
        # 更新频谱坐标范围
        try:
            self.fft_plot.setXRange(1, self.cfg.sample_rate_hz / 2, padding=0)
        except Exception:
            pass

    def _auto_connect_default(self):
        try:
            if self.is_connected or self.is_connecting:
                return
            # 若列表中包含 COM100 则选取它
            for i in range(self.port_combo.count()):
                if self.port_combo.itemText(i).upper() == "COM100":
                    self.port_combo.setCurrentIndex(i)
                    break
            # 预设参数
            self.baud_spin.setValue(921600)
            self.sr_spin.setValue(16000)
            self.parser_combo.setCurrentText("Raw16LE")
            self.mode_combo.setCurrentText("Serial")
            self.on_connect()
        except Exception:
            pass

    def _on_tick(self):
        # 为了更稳定的频谱，读取更长窗口（至少帧长，最多 1/4 秒）
        desired = max(self.cfg.frame_size_samples, max(512, self.cfg.sample_rate_hz // 4))
        latest = self.ring.read_latest(desired)
        frame, spec, dbfs, spl = self.analyzer.compute_metrics(latest)
        # 波形
        if frame.size > 0:
            t_axis = np.arange(frame.size) / float(self.cfg.sample_rate_hz)
            self.wave_curve.setData(t_axis, frame)
        # 频谱
        if spec.size > 0:
            self.fft_curve.setData(spec[:, 0], spec[:, 1])
            # 保持合理显示范围
            try:
                self.fft_plot.setXRange(1, self.cfg.sample_rate_hz / 2, padding=0)
            except Exception:
                pass
        # 数值
        if not np.isnan(dbfs):
            self.dbfs_label.setText(f"dBFS: {dbfs:6.2f}")
        if not np.isnan(spl):
            self.spl_label.setText(f"SPL: {spl:6.2f} dB")

    def _drain_status(self):
        while True:
            try:
                msg = self.status_queue.get_nowait()
            except queue.Empty:
                break
            self._on_status(msg)

    def _on_status(self, msg: str):
        self.statusBar().showMessage(msg, 2000)
        try:
            print(msg, flush=True)
        except Exception:
            pass

    def on_start_record(self):
        pass
 
    def on_stop_record(self):
        pass

    def on_pick_bin(self):
        pass
 
    def on_flash(self):
        pass

    def on_pick_project(self):
        pass
 
    def on_build_firmware(self):
        pass
 
    def on_scan_and_merge(self):
        pass

    # 辅助：子进程流式日志
    def _run_streaming(self, cmd: List[str], cwd: Optional[str] = None, tag: str = "CMD"):
        try:
            self.signals.status.emit(f"[{tag}] {' '.join(cmd)} (cwd={cwd or os.getcwd()})")
            with subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            ) as proc:
                for line in proc.stdout:  # type: ignore[attr-defined]
                    self.signals.status.emit(line.rstrip("\n"))
            rc = proc.wait() if proc else 1
            self.signals.status.emit(f"[{tag}] exit {rc}")
        except Exception as exc:
            self.signals.status.emit(f"[{tag}] 异常: {exc}")

    def _find_platformio_firmware(self, project_dir: str) -> Optional[str]:
        # 典型路径 .pio/build/*/firmware.bin
        pio_dir = project_dir
        # 若传入的是构建子目录，也兼容
        for root, dirs, files in os.walk(pio_dir):
            if os.path.basename(root) == "build" and ".pio" in root:
                for f in files:
                    if f.lower() == "firmware.bin":
                        return os.path.join(root, f)
        # 广搜
        for root, dirs, files in os.walk(pio_dir):
            for f in files:
                if f.lower() == "firmware.bin" and ".pio" in root:
                    return os.path.join(root, f)
        return None

    def _find_idf_bins(self, build_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        bl = os.path.join(build_dir, "bootloader", "bootloader.bin")
        pt = os.path.join(build_dir, "partition_table", "partition-table.bin")
        app_candidate: Optional[str] = None
        # app 通常在 build 根目录，形如 <proj>.bin 或 app.bin
        if os.path.isdir(build_dir):
            candidates: List[str] = []
            for f in os.listdir(build_dir):
                if f.endswith(".bin") and not f.startswith("bootloader") and not f.startswith("partition"):
                    candidates.append(os.path.join(build_dir, f))
            if candidates:
                # 选择最大的 bin 作为 app
                app_candidate = max(candidates, key=lambda p: os.path.getsize(p))
        if not (os.path.isfile(bl) and os.path.isfile(pt) and app_candidate and os.path.isfile(app_candidate)):
            return None, None, None
        return bl, pt, app_candidate

    def _merge_bins(self, out_path: str, items: List[Tuple[int, str]]):
        pass

    def on_one_click_gen_build_flash(self):
        port = self.port_combo.currentText().strip()
        if not port:
            QMessageBox.warning(self, "缺少串口", "请先在‘串口/地址’中选择 COM 口，如 COM100")
            return
        try:
            baud = int(self.baud_spin.value())
        except Exception:
            baud = 460800

        def job():
            # 1) 生成工程
            proj_dir = self._write_autoproj()
            if not proj_dir:
                return
            self.project_edit.setText(proj_dir)
            # 2) 编译（PlatformIO，清华源安装）
            mod = ensure_dependency("platformio>=6.1", "platformio")
            if mod is None:
                QMessageBox.critical(self, "依赖安装失败", "无法安装 PlatformIO，请检查网络或权限")
                return
            cmd = [sys.executable, "-m", "platformio", "run"]
            self._run_streaming(cmd, cwd=proj_dir, tag="PIO")
            # 3) 定位 3 个分段并烧录
            env_name = "seeed_xiao_esp32s3"
            build_dir = os.path.join(proj_dir, ".pio", "build", env_name)
            bl = os.path.join(build_dir, "bootloader.bin")
            pt = os.path.join(build_dir, "partitions.bin")
            app = os.path.join(build_dir, "firmware.bin")
            if not (os.path.isfile(bl) and os.path.isfile(pt) and os.path.isfile(app)):
                self.signals.status.emit("未找到构建产物，请查看日志")
                return
            # 4) esptool 分段烧录，地址：0x1000/0x8000/0x10000
            mod2 = ensure_dependency("esptool>=4.7", "esptool")
            if mod2 is None:
                QMessageBox.critical(self, "依赖安装失败", "无法安装 esptool，请检查网络或权限")
                return
            cmd2 = [
                sys.executable,
                "-m",
                "esptool",
                "--chip",
                "esp32s3",
                "--baud",
                str(baud),
                "--port",
                port,
                "write_flash",
                "0x1000",
                bl,
                "0x8000",
                pt,
                "0x10000",
                app,
            ]
            self._run_streaming(cmd2, tag="FLASH")
            self.signals.status.emit("一键生成并烧录完成。如需重启数据流，请切换到 Serial 模式并连接。")

        threading.Thread(target=job, daemon=True).start()

    def _write_autoproj(self) -> Optional[str]:
        try:
            base = os.path.join(tempfile.gettempdir(), "xiaos3_mic_autofw")
            os.makedirs(os.path.join(base, "src"), exist_ok=True)
            # platformio.ini
            pio_ini = """
; Auto-generated by WirelessMicrophone app
[env:seeed_xiao_esp32s3]
platform = espressif32@6.2.0
board = seeed_xiao_esp32s3
framework = arduino
monitor_speed = 921600
build_flags = 
  -DARDUINO_USB_MODE=1
  -DARDUINO_USB_CDC_ON_BOOT=1
  -DBOARD_HAS_PSRAM
""".strip()
            with open(os.path.join(base, "platformio.ini"), "w", encoding="utf-8") as f:
                f.write(pio_ini + "\n")
            # src/main.cpp（采集 PDM Mic，经串口输出 Raw16LE）
            main_cpp = r"""
#include <Arduino.h>

#if __has_include(<I2S.h>)
#  include <I2S.h>
#  define HAS_ARDUINO_I2S 1
#endif

#if __has_include(<ESP_I2S.h>)
#  include <ESP_I2S.h>
#  define HAS_ESP_I2S 1
#endif

extern "C" {
#include "driver/i2s.h"
}

#define SAMPLE_RATE   16000
#define SAMPLE_BITS   16
#define BUF_SAMPLES   1024

static int16_t sample_buf[BUF_SAMPLES];

void setup() {
  Serial.begin(921600);
  delay(500);

#ifdef HAS_ARDUINO_I2S
  I2S.setAllPins(-1, 42, 41, -1, -1); // BCLK(unused), CLK=42, DATA=41
  if (!I2S.begin(PDM_MONO_MODE, SAMPLE_RATE, SAMPLE_BITS)) {
    while (1) { delay(1000); }
  }
#elif defined(HAS_ESP_I2S)
  static I2SClass I2S;
  I2S.setPinsPdmRx(42, 41);
  if (!I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    while (1) { delay(1000); }
  }
#else
#error "No I2S header found. Please ensure Arduino-ESP32 core is installed."
#endif
}

void loop() {
  size_t bytes_read = 0;
  i2s_read((i2s_port_t)0, (void*)sample_buf, sizeof(sample_buf), &bytes_read, portMAX_DELAY);
  if (bytes_read > 0) {
    // 直接以 Raw16LE 透传
    Serial.write((uint8_t*)sample_buf, bytes_read);
  }
}
""";
            with open(os.path.join(base, "src", "main.cpp"), "w", encoding="utf-8") as f:
                f.write(main_cpp)
            self.signals.status.emit(f"已生成自动工程: {base}")
            return base
        except Exception as exc:
            self.signals.status.emit(f"生成工程失败: {exc}")
            return None


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
