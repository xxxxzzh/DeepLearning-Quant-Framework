# 使用轻量级 Python 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制依赖清单
COPY requirements.txt .

# 安装依赖 (加上 --no-cache-dir 可以减小镜像体积)
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有的核心脚本和数据
COPY engine.py .
COPY utils.py .
COPY main.py .
COPY final_performance_results.csv .

# 启动命令
CMD ["python", "main.py"]