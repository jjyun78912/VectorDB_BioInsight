FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Create output directory
RUN mkdir -p output/history

# Set environment variables
ENV PYTHONPATH=/app
ENV TZ=Asia/Seoul

# Run scheduler
CMD ["python", "-m", "src.scheduler", "--daemon"]
