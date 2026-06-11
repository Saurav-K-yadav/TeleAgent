FROM python:3.11-slim

WORKDIR /app

# System dependencies (CPU-only, no CUDA)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install dependencies using binary wheels where available to reduce build RAM.
ENV PIP_NO_BUILD_ISOLATION=1
ENV PIP_DEFAULT_TIMEOUT=100
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --prefer-binary -r requirements-spaces.txt

EXPOSE 7860

CMD ["python3", "app.py"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    pkg-config \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*