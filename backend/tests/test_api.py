import sys
import os
import json

# make sure we can import from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_returns_200(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'ok'
    assert 'model_loaded' in data


def test_health_has_feature_count(client):
    resp = client.get('/health')
    data = resp.get_json()
    # model might or might not be loaded depending on test env
    if data['model_loaded']:
        assert data['n_features'] > 0
    else:
        assert data['n_features'] is None


def test_predict_missing_body(client):
    resp = client.post('/predict', content_type='application/json')
    assert resp.status_code in (400, 503)


def test_predict_bad_feature_count(client):
    resp = client.post('/predict',
        data=json.dumps({'features': [1, 2, 3]}),
        content_type='application/json')
    # 400 if model is loaded (wrong count), 503 if model not loaded
    assert resp.status_code in (400, 503)


def test_predict_no_features_key(client):
    resp = client.post('/predict',
        data=json.dumps({'data': [1, 2, 3]}),
        content_type='application/json')
    assert resp.status_code in (400, 503)


def test_metrics_returns_json(client):
    resp = client.get('/metrics')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'total_predictions' in data
    assert 'pass_rate' in data
    assert 'fail_rate' in data


def test_feedback_missing_fields(client):
    resp = client.post('/feedback',
        data=json.dumps({'id': 1}),
        content_type='application/json')
    assert resp.status_code == 400


def test_feedback_invalid_label(client):
    resp = client.post('/feedback',
        data=json.dumps({'id': 1, 'actual_label': 'maybe'}),
        content_type='application/json')
    assert resp.status_code == 400
