from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import numpy as np
import joblib
import json
import logging
from datetime import datetime

import hashlib
from model import DefectClassifier
from database import log_prediction, get_metrics, get_recent_predictions, update_actual_label, get_recent_features
from drift import detect_drift

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# global model state
model = None
scaler = None
metadata = None
train_features = None


def load_model():
    global model, scaler, metadata, train_features
    try:
        with open('../models/metadata.json') as f:
            metadata = json.load(f)

        model = DefectClassifier(metadata['n_features'])
        model.load_state_dict(torch.load('../models/defect_model.pt', weights_only=True))
        model.eval()

        scaler = joblib.load('../models/scaler.pkl')

        train_features = np.load('../models/train_features.npy')

        logging.info(f"Model loaded ({metadata['n_features']} features)")
    except Exception as e:
        logging.warning(f"Could not load model: {e}")
        model = None
        scaler = None
        metadata = None
        train_features = None


# load on startup
load_model()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'n_features': metadata['n_features'] if metadata else None
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'model not loaded'}), 503

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'request must include "features" array'}), 400

    features = data['features']
    expected = metadata['n_features']

    if len(features) != expected:
        return jsonify({'error': f'expected {expected} features, got {len(features)}'}), 400

    if any(f is None for f in features):
        return jsonify({'error': 'features cannot contain null values'}), 400

    try:
        features_array = np.array(features, dtype=np.float64).reshape(1, -1)
        scaled = scaler.transform(features_array)
        tensor = torch.FloatTensor(scaled)

        with torch.no_grad():
            output = model(tensor).squeeze()
            confidence = torch.sigmoid(output).item()

        prediction = 'fail' if confidence >= 0.5 else 'pass'
        conf = round(confidence, 4)

        # hash the input for deduplication tracking
        input_hash = hashlib.md5(str(features).encode()).hexdigest()[:12]
        log_prediction(input_hash, prediction, conf, scaled[0].tolist())

        return jsonify({
            'prediction': prediction,
            'confidence': conf,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({'error': 'prediction failed'}), 500


@app.route('/sample', methods=['GET'])
def sample():
    """Return a random row from the dataset for testing."""
    try:
        from preprocessing import load_and_clean
        df = load_and_clean('../data/uci-secom.csv')
        row = df.sample(1).drop(columns=['Pass/Fail']).iloc[0]
        # replace any remaining NaN with 0
        features = row.fillna(0).tolist()
        return jsonify({'features': features, 'n_features': len(features)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predictions', methods=['GET'])
def predictions():
    n = request.args.get('n', 50, type=int)
    rows = get_recent_predictions(n)
    # strip out the scaled_features blob, frontend doesn't need it
    for row in rows:
        row.pop('scaled_features', None)
    return jsonify(rows)


@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(get_metrics())


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not data or 'id' not in data or 'actual_label' not in data:
        return jsonify({'error': 'request must include "id" and "actual_label"'}), 400

    label = data['actual_label']
    if label not in ('pass', 'fail'):
        return jsonify({'error': 'actual_label must be "pass" or "fail"'}), 400

    updated = update_actual_label(data['id'], label)
    if not updated:
        return jsonify({'error': 'prediction not found'}), 404

    return jsonify({'status': 'updated'})


@app.route('/drift', methods=['GET'])
def drift():
    if train_features is None:
        return jsonify({'error': 'training baseline not loaded'}), 503

    n = request.args.get('n', 100, type=int)
    recent = get_recent_features(n)

    if len(recent) < 10:
        return jsonify({'error': 'need at least 10 predictions to check drift'}), 400

    recent_array = np.array(recent)
    result = detect_drift(recent_array, train_features)
    result['samples_compared'] = len(recent)
    result['timestamp'] = datetime.now().isoformat()

    return jsonify(result)


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5050)
