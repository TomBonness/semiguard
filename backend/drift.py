import numpy as np
from scipy import stats


def detect_drift(new_features, train_features, threshold=0.05, drift_pct_threshold=0.2):

    n_features = new_features.shape[1]
    drifted_features = []

    for i in range(n_features):
        stat, p_value = stats.ks_2samp(train_features[:, i], new_features[:, i])
        if p_value < threshold:
            drifted_features.append({
                'feature_index': i,
                'ks_statistic': round(stat, 4),
                'p_value': round(p_value, 6)
            })

    drift_score = len(drifted_features) / n_features
    drift_detected = drift_score > drift_pct_threshold

    return {
        'drift_detected': drift_detected,
        'drift_score': round(drift_score, 4),
        'features_tested': n_features,
        'features_drifted': len(drifted_features),
        # only return top 10 worst offenders, sorted by p-value
        'drifted_features': sorted(drifted_features, key=lambda x: x['p_value'])[:10]
    }
