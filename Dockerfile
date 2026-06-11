FROM python:3.10-slim

# Install system dependencies some ML packages need
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy the rest of the project
COPY . .

# Create weights directory
RUN mkdir -p weights

EXPOSE 5000

CMD ["python", "api.py"]