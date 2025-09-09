FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scanner.py config.json app.py ./
COPY static ./static
COPY templates ./templates

# Create log & sqlite dir with proper permissions
RUN mkdir -p /app/logs /app/data && chown -R root:root /app

# Expose ports for scanner web UI
EXPOSE 5000

# Run both scanner and web server
CMD ["sh", "-c", "python scanner.py --config config.json & python app.py"]