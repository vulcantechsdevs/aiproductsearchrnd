FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies required by psycopg2
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt /app/requirements.txt

# ✅ Install CPU-only PyTorch first (from CPU wheel index)
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ✅ Then install everything else
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . /app

EXPOSE 8000

CMD ["python", "backend.py"]
