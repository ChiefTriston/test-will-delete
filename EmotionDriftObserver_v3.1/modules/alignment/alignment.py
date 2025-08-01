# modules/alignment/alignment.py
"""
Alignment and composite scoring of slices.
Outputs alignment.json per speaker with ranked_slices and scores.
"""

import json
import numpy as np
import portalocker
import os

def run(context):
    cfg      = context['config']['alignment']
    out_base = context['output_dir']
    results  = {}

    for sp in context['speaker_ids']:
        base = os.path.join(out_base, 'emotion_tags', sp)

        # load transcript, prosody frame-series and time, drift deltas
        with open(os.path.join(base, 'transcript.json'), 'r') as f:
            slices = json.load(f)['slices']
        with open(os.path.join(base, 'prosody_trend.json'), 'r') as f:
            prosody_data = json.load(f)
        f0_z  = np.array(prosody_data['frame_series']['f0_z'])
        times = np.array(prosody_data['frame_series']['time'])
        with open(os.path.join(base, 'drift_vector.json'), 'r') as f:
            drift = json.load(f)['deltas']

        step = times[1] - times[0] if len(times) > 1 else 0.02

        scores = []
        for seg in slices:
            start = seg['start']
            end   = seg['end']

            # 1) silence_score
            silence_score = 1 - ((end - start) / cfg['max_slice_len'])

            # 2) prosody_score
            si = int(start / step)
            ei = int(end   / step)
            prosody_score = float(np.mean(f0_z[si:ei])) if ei > si else 0.0

            # 3) polarity_score
            deltas = np.array(drift)
            polarity_score = float(np.sign(np.mean(deltas[si:ei]))) if ei > si else 0.0

            # 4) vad_score
            vad_score = seg.get('score', 1.0)

            # composite
            comp = (
                cfg['weights']['silence'] * silence_score +
                cfg['weights']['prosody'] * prosody_score +
                cfg['weights']['polarity'] * polarity_score +
                cfg['weights']['vad']     * vad_score
            )
            scores.append(float(comp))

        ranked = list(np.argsort(scores)[::-1])
        alignment = {'ranked_slices': ranked, 'scores': scores}

        out_path = os.path.join(base, 'alignment.json')
        with open(out_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(alignment, f, indent=2)
            portalocker.unlock(f)

        results[sp] = out_path

    return {'alignment': results}

