FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 固定 uv 版本，确保容器依赖安装方式和本地 uv 环境一致。
RUN pip install --no-cache-dir uv==0.11.21

# 先复制依赖文件，让代码修改时可以复用 Docker 的依赖缓存层。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# 只复制运行应用需要的源代码和项目资料；.dockerignore 会排除本地环境与真实密钥。
COPY app ./app
COPY docs ./docs
COPY eval_data ./eval_data
COPY sample_docs ./sample_docs
COPY scripts ./scripts
COPY README.md README.en.md ./

RUN mkdir -p /app/data

EXPOSE 8000

# 单 worker 运行，避免 PostgreSQL checkpoint、Chroma 和进程内写入锁被多进程放大。
# 构建阶段使用 uv --frozen 安装依赖；运行阶段直接调用锁定环境中的 Python，
# 避免 uv run 在部分 Docker Desktop 环境下退出码 135。
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
