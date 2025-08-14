# --- Frontend (Streamlit) ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Tokyo \
    # 컨테이너끼리 통신은 서비스명으로! (compose에서 backend라는 서비스명 사용)
    BACKEND_URL=http://backend:8000

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 프론트 소스 복사
COPY frontend/ /app/frontend/

EXPOSE 8501
# Streamlit 실행
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
