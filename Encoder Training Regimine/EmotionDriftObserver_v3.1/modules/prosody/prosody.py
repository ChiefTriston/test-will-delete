# modules/prosody/prosody.py
"""
Prosody extraction using prosody3 at 50Hz.
Outputs prosody_trend.json with frame series, trendlines, summaries,
and full-session plot map using absolute timestamps.
"""
import sys
import os
import json
import numpy as np
import torch
import torchaudio
import portalocker
from scipy.signal import savgol_filter
from multiprocessing import Pool

# prosody3 dependency: ensure prosody3 package is installed or install via:
# pip install prosody3 or add path to your prosody3 repo
has_prosody3 = True
try:
    from prosody_predictor import ProsodyPredictorV15
except ImportError:
    has_prosody3 = False
    # fallback using librosa and parselmouth
    import librosa
    import parselmouth
    def extract_prosody_fallback(wav_path, hop_length, sr):
        # load audio
        y, _ = librosa.load(wav_path, sr=sr)
        # pitch via parselmouth
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=hop_length/sr)
        f0 = pitch.selected_array['frequency']
        # energy via RMS
        energy = librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length).flatten()
        # pitch variance
        pv = np.var(f0[np.isfinite(f0)])
        # approximate speech_rate and pause_dur as zeros
        sr_rate = 0.0
        pause_dur = 0.0
        return f0, energy, pv, sr_rate, pause_dur, []

from modules.utils.plot_utils import generate_segment_plot_map


def process_speaker(args):
    spk, context = args
    cfg = context['config']['prosody']
    gcfg = context['config']['global']
    out_base = context['output_dir']
    plot_map_root = os.path.join(out_base, 'emotion_tags', gcfg['plot_map_dir'])

    device = 'cuda' if gcfg.get('use_gpu', False) and torch.cuda.is_available() else 'cpu'
    sr = gcfg['sample_rate']
    hop_length = int(sr / cfg['extract_freq'])

    if has_prosody3:
        model = ProsodyPredictorV15(hop_length=hop_length, sample_rate=sr, **cfg).to(device)
        model.eval()

    # determine this speaker's absolute time offset
    mapping_file = os.path.join(out_base, 'emotion_tags', spk, 'speaker_mapping.json')
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as mf:
            mapping = json.load(mf)
        offsets = [t[0] for t in mapping.get(spk, {}).get('timestamps', [])]
        global_offset = float(min(offsets)) if offsets else 0.0
    else:
        global_offset = 0.0

    # load speaker-only WAV
    wav_path = os.path.join(out_base, 'emotion_tags', spk, f'{spk}.wav')
    if has_prosody3:
        waveform, _ = torchaudio.load(wav_path)
        mel = model.mel_spec(waveform.to(device))
        prosody = model(mel)
        f0 = prosody['f0'].squeeze(0).cpu().numpy()
        eng = prosody['energy'].squeeze(0).cpu().numpy()
        pv = prosody['pitch_var'].squeeze(0).cpu().numpy()
        sr_rate = prosody['speech_rate'].item()
        pause_dur = prosody['pause_dur'].item()
        mfcc = prosody.get('mfcc', torch.zeros(0)).squeeze(0).cpu().numpy()
    else:
        f0, eng, pv, sr_rate, pause_dur, mfcc = extract_prosody_fallback(wav_path, hop_length, sr)

    # build absolute time axis
    T = len(f0)
    local_time = np.arange(0, T * (hop_length / sr), hop_length / sr)
    abs_time = local_time + global_offset
    duration = float(abs_time[-1]) if T > 0 else 0.0

    # smooth the series
    window = min(len(f0), 5)
    f0_sm = savgol_filter(f0, window, polyorder=2) if T >= 5 else f0
    eng_sm = savgol_filter(eng, window, polyorder=2) if T >= 5 else eng
    pv_sm = savgol_filter(pv, window, polyorder=2) if T >= 5 else pv

    # z-normalize
    f0_z = (f0_sm - f0_sm.mean()) / f0_sm.std() if f0_sm.std() > 0 else f0_sm
    eng_z = (eng_sm - eng_sm.mean()) / eng_sm.std() if eng_sm.std() > 0 else eng_sm
    pv_z = (pv_sm - pv_sm.mean()) / pv_sm.std() if pv_sm.std() > 0 else pv_sm

    # trendlines
    trend_f0 = np.polyfit(abs_time, f0_z, 1).tolist() if T > 1 else [0.0, 0.0]
    trend_eng = np.polyfit(abs_time, eng_z, 1).tolist() if T > 1 else [0.0, 0.0]
    trend_pv = np.polyfit(abs_time, pv_z, 1).tolist() if T > 1 else [0.0, 0.0]

    # global summaries
    summaries = {
        'f0':    {'min': float(f0_sm.min()),  'max': float(f0_sm.max()),  'mean': float(f0_sm.mean()),  'std': float(f0_sm.std())},
        'energy':{'min': float(eng_sm.min()), 'max': float(eng_sm.max()), 'mean': float(eng_sm.mean()), 'std': float(eng_sm.std())},
        'pitch_var': {'min': float(pv_sm.min()), 'max': float(pv_sm.max()), 'mean': float(pv_sm.mean()), 'std': float(pv_sm.std())}
    }
    pause_ratio = pause_dur / duration if duration > 0 else 0.0

    # assemble prosody_data
    prosody_data = {
        'frame_series': {
            'time': abs_time.tolist(),
            'f0_z': f0_z.tolist(),
            'energy_z': eng_z.tolist(),
            'pitch_var_z': pv_z.tolist()
        },
        'globals': {
            'speech_rate': float(sr_rate),
            'pause_dur': float(pause_dur),
            'pause_ratio': pause_ratio,
            'mfcc': mfcc.tolist() if hasattr(mfcc, 'tolist') else []
        },
        'trendlines': {
            'f0': trend_f0,
            'energy': trend_eng,
            'pitch_var': trend_pv
        },
        'summaries': summaries
    }

    # write prosody_trend.json
    speaker_out = os.path.join(out_base, 'emotion_tags', spk)
    json_path = os.path.join(speaker_out, 'prosody_trend.json')
    with open(json_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(prosody_data, f, indent=2)
        portalocker.unlock(f)

    # full-session plot map
    os.makedirs(plot_map_root, exist_ok=True)
    generate_segment_plot_map(
        full_scores=[{'time': float(t), 'vader_compound': float(d)} for t, d in zip(abs_time, eng_z)],
        segment_start=global_offset,
        segment_end=duration,
        clip_id=spk,
        tier1_transition="",
        drift_reason="",
        save_dir=plot_map_root
    )

    return spk, json_path


def run(context):
    speakers = context['speaker_ids']
    args = [(spk, context) for spk in speakers]
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_speaker, args)
    return {'prosody_trend': {spk: path for spk, path in results}}


