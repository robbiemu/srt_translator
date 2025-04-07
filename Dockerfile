# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY .env* ./
COPY example.srt ./
COPY src/ .

# Create persistent directories with proper permissions
RUN mkdir -p /app/persistent_temp /app/.gradio && \
    chmod -R 777 /app/persistent_temp && \
    chmod -R 777 /app/.gradio

# Default Environment variables
ENV OPENAI_MODEL="gpt-4"
ENV TARGET_LANGUAGE="es"
ENV STYLE_GUIDELINE="Natural speech patterns"
ENV TECHNICAL_GUIDELINE="Use standard terminology"
ENV GRADIO_TEMP_DIR="/app/persistent_temp"
ENV GRADIO_CACHING_EXAMPLES="True"

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
