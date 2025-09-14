#!/bin/bash

echo "ESP32S3 音频分析器启动中..."
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

echo "Python环境检查通过"

# 检查依赖是否已安装
echo "检查依赖库..."
if ! python3 -c "import PySide6" &> /dev/null; then
    echo "正在安装依赖库..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "错误: 依赖安装失败"
        exit 1
    fi
fi

echo "依赖库检查通过"
echo
echo "启动音频分析器..."
echo

# 启动主程序
python3 app.py 