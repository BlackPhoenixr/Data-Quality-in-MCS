# Use official Python image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables to avoid Python buffering and to ensure UTF-8 encoding
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# By default, run main.py (override in docker-compose for other scripts)
CMD ["python", "main.py"]