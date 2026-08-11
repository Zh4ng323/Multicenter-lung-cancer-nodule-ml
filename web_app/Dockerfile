FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# 国内服务器：apt 也走阿里云镜像，避免官方源慢/超时
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# 国内服务器使用阿里云 PyPI 镜像，避免官方源超时/失败
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/

# Copy application code
COPY . .

# Verify model files exist
RUN python -c "import joblib; print('Dependencies OK')"

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Environment variables for cloud deployment
ENV SERVER_NAME=0.0.0.0
ENV SERVER_PORT=7860
ENV AUTH_ENABLED=true

# 健康检查：容器编排/云平台据此判断服务是否存活
# /config 需要登录会 401，健康检查改用无需鉴权的静态资源
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/manifest.json', timeout=5)"

CMD ["python", "app.py"]
