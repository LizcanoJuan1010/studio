FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The port the container will listen on. Uvicorn will run on this port.
EXPOSE 8000
