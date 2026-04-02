FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Create data and logs directories
RUN mkdir -p data logs

# Default command: run the trading agent in paper mode
CMD ["python", "-m", "src.main", "--mode", "paper"]
