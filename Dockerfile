FROM python:3.12-slim
WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# 拷贝并安装应用
COPY pyproject.toml setup.cfg* ./
COPY deep-searcher/deepsearcher ./deepsearcher
COPY deep_searcher_chat/prod_chat_handler.py ./deep_searcher_chat/prod_chat_handler.py
COPY load_azure_private_data.py .
COPY handlers.py query_stream.py app.py ./

# 构建并安装 wheel
RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir dist . \
    && pip install --no-cache-dir dist/*.whl

# 设置 PYTHONPATH
ENV PYTHONPATH=/app:/app/deepsearcher

# 仅保留运行时文件
RUN rm -rf dist build *.egg-info

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]