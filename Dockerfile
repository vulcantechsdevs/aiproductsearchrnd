FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies including Python and pip
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    build-essential \
    cuda-toolkit-12-1 \
    libcudnn8 \
    libnccl2 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt /app/requirements.txt

# Install CUDA-compatible PyTorch first
RUN pip3 install --no-cache-dir torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Then install everything else
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]