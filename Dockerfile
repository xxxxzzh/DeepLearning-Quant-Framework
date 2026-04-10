# 1. 使用官方轻量级 Python 镜像
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 先复制依赖清单并安装 (利用 Docker 缓存机制加速)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 复制当前目录下的所有代码到容器
COPY . .

# 5. 容器启动时默认运行的命令 (假设你想运行 09 号文件的重构逻辑)
# 注意：生产环境通常运行 .py 脚本，如果是跑 Notebook 建议手动启动
CMD ["python"]