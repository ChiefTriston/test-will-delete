# modules/fingerprint/fingerprint.py
import json
import os
from collections import Counter
import numpy as np
import portalocker

def compute_entropy(labels):
    from math import log
    counts = Counter(labels)
    total = sum(counts.values())
    return -sum((c/total) * log(c/total) for c in counts.values() if c > 0)

def run(context):
    output_dir = context['output_dir']
    for sp in context['speaker_ids']:
        base = os.path.join(output_dir, 'emotion_tags', sp)

        # load Tier 2 tags
        with open(os.path.join(base, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier2 = json.load(f)
            portalocker.unlock(f)

        # load drift_vector for Δ values
        with open(os.path.join(base, 'drift_vector.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)

        # load drift_log for slope
        with open(os.path.join(base, 'drift_log.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift_log = json.load(f)
            portalocker.unlock(f)

        # compute stats
        labels       = [t['label'] for t in tier2]
        confs        = [t['confidence'] for t in tier2]
        avg_conf     = float(np.mean(confs)) if confs else 0.0
        entropy      = compute_entropy(labels)       if labels else 0.0
        deltas       = drift.get('deltas', [])
        avg_drift    = float(np.mean(np.abs(deltas))) if deltas else 0.0
        drift_slope  = drift_log.get('confidence_drift_slope', 0.0)

        fingerprint = {
            'dominant_tags'  : Counter(labels).most_common(),
            'avg_confidence' : avg_conf,
            'entropy'        : entropy,
            'avg_drift'      : avg_drift,
            'drift_slope'    : drift_slope
        }

        # write fingerprint.json
        with open(os.path.join(base, 'fingerprint.json'), 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(fingerprint, f, indent=2)
            portalocker.unlock(f)

    return {'fingerprint': 'fingerprint.json'}
