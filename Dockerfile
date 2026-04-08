FROM python:3.11-slim

WORKDIR /app

# install deps
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# copy project
COPY . .

# launch FastAPI server (port 8004, matching OpenEnv convention)
EXPOSE 8004
ENV PYTHONPATH="/app"

CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8004"]
