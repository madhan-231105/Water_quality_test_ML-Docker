FROM python:3.11-slim

WORKDIR /app

# 🔧 Install system dependencies (important for numpy, sklearn, xgboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy only required files
COPY app.py .
COPY index.html .
COPY models ./models

# Expose correct port (your Flask runs on 5001)
EXPOSE 5001

# Run app
CMD ["python", "app.py"]