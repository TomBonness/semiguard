from flask import Flask, jsonify
from flask_cors import CORS
import torch
import joblib
import json
import logging

from model import DefectClassifier

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# global model state
model = None
scaler = None
metadata = None


def load_model():
    global model, scaler, metadata
    try:
        with open('../models/metadata.json') as f:
            metadata = json.load(f)

        model = DefectClassifier(metadata['n_features'])
        model.load_state_dict(torch.load('../models/defect_model.pt', weights_only=True))
        model.eval()

        scaler = joblib.load('../models/scaler.pkl')

        logging.info(f"Model loaded ({metadata['n_features']} features)")
    except Exception as e:
        logging.warning(f"Could not load model: {e}")
        model = None
        scaler = None
        metadata = None


# load on startup
load_model()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'n_features': metadata['n_features'] if metadata else None
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
