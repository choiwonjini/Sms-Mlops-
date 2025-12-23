FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Cloud Run defaults to 8080 usually, but we can configure expected port)
# cmd should listen on $PORT env var if provided, uvicorn defaults to 8000
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}
