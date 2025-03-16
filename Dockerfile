FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \  # For numpy/pandas operations
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Create cache directory for embedding and LLM results
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV CACHE_DIR=/app/.cache

# Create volumes for persisting data and cache
VOLUME /app/data
VOLUME /app/.cache

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]