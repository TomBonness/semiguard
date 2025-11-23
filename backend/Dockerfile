FROM python:3.11-slim

WORKDIR /app/backend

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY models/ /app/models/
COPY data/ /app/data/

EXPOSE 5050

CMD ["gunicorn", "--bind", "0.0.0.0:5050", "app:app"]
