FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: real-footage YOLO detector.  docker compose build --build-arg INSTALL_YOLO=1
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

# Bake a pre-processed sample mission into the image so free/ephemeral hosts (which reset their disk on every
# restart) always start with data. Users are NOT baked: they are created at startup from SEED_PASSWORD.
ARG BAKE_DEMO=1
RUN if [ "$BAKE_DEMO" = "1" ]; then cd /app/backend && python -m app.services.demo; fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os,urllib.request as u;u.urlopen('http://127.0.0.1:%s/api/v1/system/health'%os.environ.get('PORT','8000'),timeout=4)"
# Cloud hosts (Render, Railway, Koyeb, ...) inject $PORT; locally it stays 8000.
CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
