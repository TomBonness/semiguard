FROM python:3.11-slim

WORKDIR /app/backend

# install deps first so this layer gets cached
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy backend source code
COPY backend/ .

# copy trained model, scaler, metadata, and baseline stats
COPY models/ /app/models/
COPY data/ /app/data/

EXPOSE 5050

CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "2", "app:app"]
