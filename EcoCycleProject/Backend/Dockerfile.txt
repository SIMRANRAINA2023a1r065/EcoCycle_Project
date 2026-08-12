FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces port 7860 use karta hai
EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]
