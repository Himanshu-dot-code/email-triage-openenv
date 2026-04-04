FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir numpy \
 && pip install --no-cache-dir pydantic fastapi uvicorn \
 && pip install --no-cache-dir openenv-core

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]