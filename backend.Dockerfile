# --- Backend (FastAPI) ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Tokyo

# tzdata만 깔면 충분 (슬림 이미지)
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 루트의 requirements.txt 하나로 통일해서 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 백엔드 소스 복사
COPY backend/ /app/backend/

EXPOSE 8000
# FastAPI 진입점: backend/app/main.py 안에 app = FastAPI(...) 가 있으므로
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

