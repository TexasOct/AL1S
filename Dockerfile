FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 使用uv安装依赖到虚拟环境，添加重试机制
ENV PATH="/app/venv/bin:$PATH"

# 复制项目文件
COPY pyproject.toml ./
COPY main.py ./
COPY src/ ./src/

# 合并系统依赖、Node.js、uv安装、版本验证和用户创建为单层
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && pip install uv \
    && node --version && npm --version \
    && groupadd -r botuser && useradd -r -g botuser -d /app -s /bin/bash botuser \
    && rm -rf /var/lib/apt/lists/* \
    && uv venv ./venv \
    && uv pip install --no-cache . \
    && mkdir -p /app/.npm /app/.npm-global /app/.cache \
    && chown -R botuser:botuser /app

# 配置npm环境变量
ENV HOME=/app
ENV NPM_CONFIG_CACHE=/app/.npm
ENV NPM_CONFIG_PREFIX=/app/.npm-global
ENV NPM_CONFIG_USERCONFIG=/app/.npmrc
ENV PATH="/app/.npm-global/bin:$PATH"

# 切换到非root用户
USER botuser

# 健康检查 - 验证Python和Node.js环境
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" && node --version

# 启动命令
CMD ["python", "main.py"]
