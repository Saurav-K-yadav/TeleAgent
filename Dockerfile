FROM python:3.12-slim

WORKDIR /app

# System dependencies (CPU-only, no CUDA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
# Install remaining requirements (llama-cpp-python will be installed via requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python3", "app.py"]
