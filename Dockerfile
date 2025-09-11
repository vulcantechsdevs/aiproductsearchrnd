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

COPY requirements.txt /app/requirements.txt

# ✅ Install requirements (CPU-only PyTorch first)
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . /app

EXPOSE 8000

CMD ["python", "backend.py"]
