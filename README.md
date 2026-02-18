# SemiGuard

ML-powered semiconductor defect prediction dashboard. Uses the UCI SECOM dataset (590 sensor features from real wafer fabrication) to predict pass/fail on wafers, with drift detection to catch when the model starts degrading.

**Live demo:** [chipguardai.com](https://chipguardai.com)

**Stack:** PyTorch, Flask, Angular, SQLite, AWS (Amplify + EC2 + ALB)

## Architecture

```
+----------------+     +----------------+     +----------------+
|    Angular     |---->|   Flask API    |---->|    SQLite      |
|   Dashboard    |<----|   + PyTorch    |<----|   Database     |
+----------------+     +----------------+     +----------------+
                              |
                        +-----+-----+
                        |  Trained  |
                        |   Model   |
                        | (.pt file)|
                        +-----------+
```

- **Frontend** (Angular) - dashboard with stats cards, charts, drift monitoring, prediction form, and feedback UI
- **Backend** (Flask) - REST API serving predictions, logging to SQLite, drift detection via KS-test
- **Model** (PyTorch) - feedforward neural net trained on 562 features (after dropping high-missing columns)

## Setup

### 1. Get the data

Download the [UCI SECOM dataset from Kaggle](https://www.kaggle.com/datasets/paresh2047/uci-semcom) and put the CSV in `data/`:

```
data/uci-secom.csv
```

### 2. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the model

```bash
cd backend
python train.py
```

This saves the model, scaler, and metadata to `models/`.

### 4. Run the API

```bash
cd backend
source venv/bin/activate
python app.py
```

API runs on `http://localhost:5050`.

### 5. Frontend

```bash
cd frontend
npm install
npx ng serve
```

Dashboard at `http://localhost:4200`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | API status and model info |
| POST | /predict | Run prediction on sensor features |
| GET | /predictions | Recent prediction history |
| GET | /metrics | Prediction stats (pass/fail rates, counts) |
| POST | /feedback | Submit ground truth for a prediction |
| GET | /drift | Run drift detection on recent predictions |
| GET | /sample | Get a random data row for testing |

## Running Tests

```bash
cd backend
source venv/bin/activate
python -m pytest tests/ -v
```

## Docker

```bash
docker-compose up --build
```

Backend on `:5050`, frontend on `:4200`.

## What's going on with the model

- Feedforward net: 562 -> 128 -> 64 -> 1
- BCEWithLogitsLoss with pos_weight to handle the ~14:1 class imbalance
- Dropout (0.3) on hidden layers
- Trained for 50 epochs with Adam
- Drift detection uses Kolmogorov-Smirnov test per feature against the training baseline

## Deployment

The app is deployed on AWS. Frontend and backend are separate.

### Diagram

```
+----------------+     push to main     +-------------------+
|    GitHub      |--------------------->|  GitHub Actions   |
|   Repository   |                      |  (CI/CD)          |
+----------------+                      +-------------------+
                                           |            |
                               tests pass  |            | auto-trigger
                               + deploy    |            |
                                           v            v
                                    +----------+  +-----------+
                                    |   EC2    |  |  Amplify  |
                                    |  Docker  |  |  Angular  |
                                    |  Flask   |  |  Frontend |
                                    +----+-----+  +-----------+
                                         |              |
                                    +----+-----+        |
                                    |   ALB    |<-------+
                                    |  HTTPS   |  API calls
                                    +----------+
```

- **Frontend** is on AWS Amplify. Pushes to main auto-deploy.
- **Backend** runs in Docker on an EC2 t3.small instance. An Application Load Balancer handles HTTPS termination in front of it.
- **Domain** is chipguardai.com via Route 53. ACM provides the SSL certs.
- **CI/CD** is a GitHub Actions workflow that runs pytest, then SSHes into EC2 to pull and rebuild the container.

