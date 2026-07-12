FROM python:3.11-slim

WORKDIR /app

# MediaPipe가 Linux에서 필요한 OpenGL/EGL 라이브러리 전체 설치
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgles2 \
    libegl1 \
    libgbm1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# GPU 없는 환경에서 소프트웨어 렌더링 사용 (MediaPipe headless 실행)
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads

# MediaPipe 포즈 모델 미리 다운로드
RUN wget -q -O pose_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

CMD uvicorn main:app --host 0.0.0.0 --port 8000
