# modules/anomaly/anomaly.py
"""
Anomaly flagging for hallucinations and VADER outliers.
Injects anomalies into drift_vector.json and updates drift_log.json.
"""
import json
import numpy as np
import portalocker
import os
from scipy.stats import entropy as shannon_entropy
from collections import Counter


def run(context):
    config = context['config']['anomaly']
    output_dir = context['output_dir']
    speaker_ids = context['speaker_ids']

    VALIDATION_SET_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "validation_set.json")
    calibration_file = os.path.join(os.path.dirname(__file__), "..", "..", "calibration.json")

    # Load or compute calibration thresholds
    try:
        with open(calibration_file, 'r') as cf:
            calib = json.load(cf)
    except FileNotFoundError:
        # Load validation set
        try:
            with open(VALIDATION_SET_FILE, 'r') as vf:
                validation = json.load(vf)
        except FileNotFoundError:
            validation = []

        if validation:
            # Text length threshold
            lengths = [len(v.get('text', '')) for v in validation]
            mean_len = float(np.mean(lengths)) if lengths else 5.0
            std_len = float(np.std(lengths)) if lengths else 1.0
            halluc_min_len = mean_len + 2 * std_len

            # Repetition threshold
            reps = []
            for v in validation:
                text = v.get('text', '')
                rep = max(Counter(text).values()) / len(text) if text else 0.0
                reps.append(rep)
            avg_rep = float(np.mean(reps)) if reps else 0.0
            std_rep = float(np.std(reps)) if reps else 0.0
            rep_thresh = avg_rep + std_rep

            # VADER outlier multiplier
            compounds_val = [v.get('compound', 0.0) for v in validation]
            std_comp = float(np.std(compounds_val)) if compounds_val else 0.1
            outlier_mult = std_comp * 2

            calib = {
                'halluc_min_len': halluc_min_len,
                'rep_thresh': rep_thresh,
                'outlier_mult': outlier_mult
            }
            # Save calibration atomically
            tmp = calibration_file + ".tmp"
            with open(tmp, 'w') as cf_tmp:
                json.dump(calib, cf_tmp)
            os.replace(tmp, calibration_file)
        else:
            calib = {
                'halluc_min_len': config['hallucination_min_len'],
                'rep_thresh': config['repetition_thresh'],
                'outlier_mult': config['outlier_std_mult']
            }

    # Process each speaker
    for speaker_id in speaker_ids:
        speaker_out = os.path.join(output_dir, 'emotion_tags', speaker_id)

        # Load data
        with open(os.path.join(speaker_out, 'transcript.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'tier1_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier1 = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'tier2_tags.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier2 = json.load(f)
            portalocker.unlock(f)
        with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)

        time = np.array(prosody['frame_series']['time'])
        time_delta = time[1] - time[0] if len(time) > 1 else 0.02
        energy_z = np.array(prosody['frame_series']['energy_z'])

        anomalies = []
        compounds = [t.get('compound', 0.0) for t in tier1]

        for idx, slice_data in enumerate(transcript.get('slices', [])):
            text = slice_data.get('text', '')
            start_time = slice_data.get('start', 0.0)
            end_time = slice_data.get('end', 0.0)

            # Calculate energy indices
            start_idx = min(int(start_time / time_delta), len(energy_z)-1)
            end_idx = min(int(end_time / time_delta), len(energy_z))
            slice_energy = energy_z[start_idx:end_idx] if end_idx > start_idx else np.array([])

            # Whisper hallucination checks
            if len(text) < calib['halluc_min_len']:
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx, 'reason': 'short_text'})
            if text and max(Counter(text).values()) / len(text) > calib['rep_thresh']:
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx, 'reason': 'repetitive'})
            silence_ratio = np.sum(slice_energy < -1.5) / len(slice_energy) if len(slice_energy) else 0.0
            if text and silence_ratio > 0.7:
                anomalies.append({'type': 'whisper_hallucination', 'slice': idx, 'reason': 'silent_with_words'})

            # VADER anomaly checks
            window_size = config.get('vader_window', 3)
            if idx >= window_size - 1:
                window = compounds[idx-window_size+1:idx+1]
                swing = max(window) - min(window)
                sigma = np.std(compounds) if compounds else 1.0
                if swing > calib['outlier_mult'] * sigma:
                    anomalies.append({'type': 'vader_anomaly', 'slice': idx, 'reason': 'swing'})
                elif abs(compounds[idx]) > np.mean(compounds) + calib['outlier_mult'] * sigma:
                    anomalies.append({'type': 'vader_anomaly', 'slice': idx, 'reason': 'outlier'})

        # Inject anomalies into drift_vector.json
        dv_path = os.path.join(speaker_out, 'drift_vector.json')
        with open(dv_path, 'r+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            drift = json.load(f)
            f.seek(0)
            drift['anomalies'] = anomalies
            json.dump(drift, f)
            f.truncate()
            portalocker.unlock(f)

        # Compute emotion entropy
        labels = [t.get('label') for t in tier2]
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum() if counts.sum() else []
        emotion_entropy = float(shannon_entropy(probs)) if probs else 0.0

        # Compute confidence drift slope
        confidences = [t.get('confidence', 0.0) for t in tier2]
        times = [ (s.get('start',0)+s.get('end',0))/2 for s in transcript.get('slices',[]) ]
        slope = float(np.polyfit(times, confidences, 1)[0]) if len(confidences) > 1 else 0.0

        # Update drift_log.json
        dl_path = os.path.join(speaker_out, 'drift_log.json')
        with open(dl_path, 'r+') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            log = json.load(f)
            f.seek(0)
            log['emotion_entropy'] = emotion_entropy
            log['confidence_drift_slope'] = slope
            json.dump(log, f)
            f.truncate()
            portalocker.unlock(f)

    return {'updated_drift': True}
