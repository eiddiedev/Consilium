FROM python:3.12-slim

WORKDIR /app

# Install uv for fast package installation
RUN pip install --no-cache-dir uv

# Install dependencies using uv (much faster than pip)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source
COPY . .

ENV PORT=8080
ENV AGENT_MODULE=orchestrator.app:a2a_app

CMD ["sh", "-c", "exec uvicorn ${AGENT_MODULE} --host 0.0.0.0 --port ${PORT}"]
