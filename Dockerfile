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


# Create directory for temporary files
RUN mkdir -p /tmp/srt_translator

# Default Environment variables
ENV OPENAI_MODEL="gpt-4"
ENV TARGET_LANGUAGE="es"
ENV STYLE_GUIDELINE="Natural speech patterns"
ENV TECHNICAL_GUIDELINE="Use standard terminology"

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]