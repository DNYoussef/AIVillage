FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt ./
RUN pip install --no-cache-dir --target=/opt/python -r requirements.txt

COPY services/gateway/ /app/
WORKDIR /app
RUN python -m compileall -q .

FROM gcr.io/distroless/python3-debian12
COPY --from=builder /opt/python /opt/python
COPY --from=builder /app /app

WORKDIR /app
USER 1000:1000
ENV PYTHONPATH=/opt/python

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
