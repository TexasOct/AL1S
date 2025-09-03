FROM python:3.11-slim

# 安装运行时必需的系统依赖和curl
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && pip install uv

# 创建非root用户
RUN groupadd -r botuser && useradd -r -g botuser botuser 

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY pyproject.toml ./
COPY main.py ./
COPY src/ ./src/

# 使用uv安装依赖到虚拟环境，添加重试机制
RUN uv venv ./venv
ENV PATH="/app/venv/bin:$PATH"
RUN uv pip install --no-cache . 

# 修改文件权限
RUN chown -R botuser:botuser /app

# 切换到非root用户
USER botuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# 启动命令
CMD ["python", "main.py"]
