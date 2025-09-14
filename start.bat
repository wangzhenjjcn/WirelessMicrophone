@echo off
echo ESP32S3 音频分析器启动中...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo Python环境检查通过

REM 检查依赖是否已安装
echo 检查依赖库...
pip show PySide6 >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖库...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误: 依赖安装失败
        pause
        exit /b 1
    )
)

echo 依赖库检查通过
echo.
echo 启动音频分析器...
echo.

REM 启动主程序
python app.py

pause 