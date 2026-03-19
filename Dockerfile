FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY refexplorer.py .

# Non-root user
RUN groupadd -r refex && useradd -r -g refex refex
USER refex

ENTRYPOINT ["python", "refexplorer.py"]
