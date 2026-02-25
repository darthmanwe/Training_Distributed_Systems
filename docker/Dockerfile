FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

RUN uv pip install --system -e "."

EXPOSE 8000

ENTRYPOINT ["python", "scripts/run_local.py"]
CMD ["--config", "configs/base.yaml"]
