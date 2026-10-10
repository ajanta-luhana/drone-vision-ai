FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: real-footage YOLO detector.  docker compose build --build-arg INSTALL_YOLO=1
# (adds ~1.5 GB; weights are downloaded once at build time so the container works offline)
ARG INSTALL_YOLO=0
RUN if [ "$INSTALL_YOLO" = "1" ]; then \
      pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir ultralytics && \
      pip uninstall -y opencv-python && \
      pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==5.0.0.93 && \
      python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" ; \
    fi

COPY . .
RUN python scripts/make_demo_video.py
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request as u;u.urlopen('http://127.0.0.1:8000/api/v1/system/health',timeout=4)"
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
