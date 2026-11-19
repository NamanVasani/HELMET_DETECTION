# Helmet detector + dashboard for Linux servers and edge boxes (amd64 / arm64, CPU inference).
# Build:  docker build -t helmet .        (uses the committed best.pt)
# Run:    docker compose up -d            (see docker-compose.yml)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HELMET_HOST=0.0.0.0 HELMET_AUTH=1 HELMET_STORAGE=/data \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU wheels keep the image small; GPU hosts can swap the index URL.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0 \
 && pip install --no-cache-dir -r requirements.txt

COPY app.py dashboard.py event_store.py auth.py alerts.py cameras.py reports_pdf.py backup.py manage.py \
     model-manifest.json thresholds.json best.pt ./
COPY static ./static
COPY models ./models

RUN useradd --uid 10001 --create-home helmet && mkdir -p /data && chown helmet /data
USER helmet
VOLUME ["/data"]
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/healthz', timeout=3)"
CMD ["python", "-u", "app.py"]
