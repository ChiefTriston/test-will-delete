"""
Drift computation based on prosody deltas.
Outputs drift_vector.json, drift_log.json, and per‑slice plot maps.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import portalocker

from modules.utils.plot_utils import generate_segment_plot_map


def derive_segment_boundaries(duration_sec, drift_events):
    # simple boundaries: start at 0, each drift event, then end
    return [0.0] + drift_events + [duration_sec]


def run(context):
    cfg       = context['config']['drift']
    gcfg      = context['config']['global']
    out_base  = context['output_dir']
    speakers  = context['speaker_ids']
    plot_root = os.path.join(out_base, 'emotion_tags', gcfg['plot_map_dir'])

    vec_path = None
    log_path = None

    for spk in speakers:
        speaker_out = os.path.join(out_base, 'emotion_tags', spk)

        # load prosody
        with open(os.path.join(speaker_out, 'prosody_trend.json'), 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)

        time       = np.array(prosody['frame_series']['time'])
        f0_z       = np.array(prosody['frame_series']['f0_z'])
        energy_z   = np.array(prosody['frame_series']['energy_z'])
        if len(time) < 2:
            continue
        duration   = float(time[-1])

        # compute deltas and rolling std thresholds
        delta_f0     = np.diff(f0_z)
        delta_energy = np.diff(energy_z)
        window       = cfg.get('rolling_window', 50)
        rs_f0        = pd.Series(delta_f0).rolling(window, min_periods=1).std().values
        rs_en        = pd.Series(delta_energy).rolling(window, min_periods=1).std().values
        thresh_f0    = cfg['thresh_pitch']  * rs_f0
        thresh_energy= cfg['thresh_energy'] * rs_en

        # raw drift points
        drift_pts = np.where((np.abs(delta_f0) > thresh_f0) |
                              (np.abs(delta_energy) > thresh_energy))[0].tolist()
        boundaries = derive_segment_boundaries(duration, drift_pts)

        # buffer-zone merging
        frame_delta   = time[1] - time[0]
        buf_samps     = int(cfg['buffer_zone'] / frame_delta)
        merged_drifts = []
        if drift_pts:
            cur = drift_pts[0]
            for pt in drift_pts[1:]:
                if pt - cur > buf_samps:
                    merged_drifts.append(cur)
                    cur = pt
            merged_drifts.append(cur)

        # polarity grouping
        combined   = (delta_f0 + delta_energy) / 2
        pol_merged = []
        if merged_drifts:
            polarity = np.sign(combined[merged_drifts])
            cp = polarity[0]
            grp = [merged_drifts[0]]
            for idx, pt in enumerate(merged_drifts[1:], start=1):
                if polarity[idx] == cp:
                    grp.append(pt)
                else:
                    pol_merged.append(int(np.mean(grp)))
                    cp = polarity[idx]
                    grp = [pt]
            pol_merged.append(int(np.mean(grp)))

        # whiplash filter
        small_thresh    = min(np.mean(thresh_f0), np.mean(thresh_energy)) * 0.5
        filtered_drifts = []
        for pt in pol_merged:
            if not filtered_drifts or abs(combined[pt] - combined[filtered_drifts[-1]]) >= small_thresh:
                filtered_drifts.append(pt)

        # smoothing
        all_deltas      = np.concatenate([delta_f0, delta_energy])
        smoothed_deltas = savgol_filter(
            all_deltas,
            window_length=cfg['smoothing_window'],
            polyorder=cfg['smoothing_order']
        )

        # compute drift confidence for each event
        std_delta = np.std(combined) if combined.size > 0 else 0.0
        drift_confidences = []
        for pt in filtered_drifts:
            delta_val = abs(combined[pt])
            threshold = small_thresh
            if std_delta > 0:
                confidence = (delta_val - threshold) / std_delta
            else:
                confidence = 0.0
            drift_confidences.append(confidence)

        # slice boundaries in time
        slice_starts = [0] + filtered_drifts
        slice_ends   = filtered_drifts + [len(time) - 1]
        slice_bounds = [
            (float(time[s]), float(time[e]))
            for s, e in zip(slice_starts, slice_ends)
        ]

        # write JSONs
        drift_vector = {
            'deltas':           smoothed_deltas.tolist(),
            'slices':           filtered_drifts,
            'slice_boundaries': slice_bounds,
            'boundaries':       boundaries
        }
        drift_log = {
            'thresholds': {
                'f0':    float(np.mean(thresh_f0)),
                'energy': float(np.mean(thresh_energy))
            },
            'num_drifts':        len(filtered_drifts),
            'drift_confidences': drift_confidences
        }

        vec_path = os.path.join(speaker_out, 'drift_vector.json')
        log_path = os.path.join(speaker_out, 'drift_log.json')

        with open(vec_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(drift_vector, f, indent=2)
            portalocker.unlock(f)
        with open(log_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(drift_log,    f, indent=2)
            portalocker.unlock(f)

        # per-slice plot maps
        full_scores = [
            {'time': float(t), 'vader_compound': float(d)}
            for t, d in zip(time[1:], smoothed_deltas)
        ]
        pm_dir = os.path.join(speaker_out, gcfg['plot_map_dir'])
        os.makedirs(pm_dir, exist_ok=True)
        for idx, start_idx in enumerate(filtered_drifts):
            end_idx = filtered_drifts[idx + 1] if idx + 1 < len(filtered_drifts) else len(time) - 1
            generate_segment_plot_map(
                full_scores      = full_scores,
                segment_start    = float(time[start_idx]),
                segment_end      = float(time[end_idx]),
                clip_id          = f"{spk}_{idx}",
                tier1_transition = "",
                drift_reason     = "",
                save_dir         = pm_dir
            )

        # Immediate console feedback
        try:
            print(f"[Drift] Speaker: {spk}")
            print(f"[Drift] Vector path: {vec_path}")
            print(f"[Drift] Log path:    {log_path}")
            print(f"[Drift] Drift vector data: {json.dumps(drift_vector)}")
        except Exception as e:
            print(f"[Drift] Warning: could not print drift data ({e})")

    return {'drift_vector': vec_path, 'drift_log': log_path}

