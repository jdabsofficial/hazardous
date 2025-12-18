# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY wastehub_api.py .
COPY hazard_type_classes.json .
COPY wastehub_hazard_model_best.h5 .
COPY wastehub_hazard_type_model_best.h5 .

# Fallback models (if best models don't exist)
COPY wastehub_hazard_model.h5* ./
COPY wastehub_hazard_type_model.h5* ./

# Expose port (Render uses PORT env var)
EXPOSE 8000

# Use PORT environment variable if available, otherwise default to 8000
ENV PORT=8000

# Run the application
CMD uvicorn wastehub_api:app --host 0.0.0.0 --port ${PORT:-8000}

