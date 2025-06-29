FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-dev.txt pyproject.toml ./
COPY scripts/setup_env.sh ./scripts/setup_env.sh
RUN bash ./scripts/setup_env.sh
COPY . .
CMD ["pytest", "-vv"]
