"""
Module for dynamic frame slicing using hybrid VAD with probabilistic boundaries and speaker-awareness.
"""

import numpy as np
import torch
from scipy.signal import medfilt
import logging

# Constants (overridden at runtime via DiarizerController parameters)
SAMPLE_RATE    = 16000
MIN_SLICE_DUR  = 1.5    # seconds
MAX_SLICE_DUR  = 6.0    # seconds
MIN_MERGE_GAP  = 0.2    # seconds
PADDING        = 0.1    # seconds padding around each slice
SNR_THRESH     = 5.0    # dB threshold for noise filtering

# Load Silero VAD
silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
get_speech_timestamps, _, _, _, _ = utils

# WebRTC VAD
import webrtcvad
webrtc_vad = webrtcvad.Vad(3)

def dynamic_slice(audio, device='cpu', embs=None, labels=None):
    try:
        audio_t = torch.from_numpy(audio).float().to(device)
        
        # Silero VAD
        speech_timestamps = get_speech_timestamps(audio_t, silero_model, sampling_rate=SAMPLE_RATE, return_seconds=True)
        
        # WebRTC batch processing
        frame_size = int(SAMPLE_RATE * 0.03)
        frames = np.array_split(audio, np.arange(frame_size, len(audio), frame_size))
        webrtc_probs = []
        for frame in frames:
            if len(frame) == frame_size:
                frame_bytes = (frame * 32767).astype(np.int16).tobytes()
                webrtc_probs.append(webrtc_vad.is_speech(frame_bytes, SAMPLE_RATE))
            else:
                webrtc_probs.append(0)
        
        # Hybrid fusion: average probabilities
        hybrid_probs = []
        for ts in speech_timestamps:
            start_frame = int(ts['start'] / 0.03)
            end_frame = int(ts['end'] / 0.03)
            avg_webrtc = np.mean(webrtc_probs[start_frame:end_frame]) if end_frame > start_frame else 0
            hybrid_prob = (ts.get('probability', 1.0) + avg_webrtc) / 2
            hybrid_probs.append(hybrid_prob)
            ts['prob'] = hybrid_prob
        
        # Noise amplitude estimation
        non_speech_frames = []
        prev_end = 0
        for ts in speech_timestamps:
            non_speech_frames.append(audio[prev_end:int(ts['start'] * SAMPLE_RATE)])
            prev_end = int(ts['end'] * SAMPLE_RATE)
        non_speech_frames.append(audio[prev_end:])
        noise_amp = np.mean([np.max(np.abs(f)) for f in non_speech_frames if len(f) > 0]) if non_speech_frames else 0.01
        
        # Filter slices by duration and probability
        slices = [(ts['start'], ts['end'], ts['prob']) for ts in speech_timestamps
                  if MIN_SLICE_DUR <= (ts['end'] - ts['start']) <= MAX_SLICE_DUR and ts['prob'] > 0.5]
        
        # Merge slices (speaker-aware if embeddings provided)
        merged = []
        if slices:
            current_start, current_end, current_prob = slices[0]
            current_label = labels[0] if labels is not None else None
            for i, (start, end, prob) in enumerate(slices[1:], start=1):
                if labels is not None and embs is not None:
                    sim = np.dot(embs[i-1], embs[i]) / (np.linalg.norm(embs[i-1]) * np.linalg.norm(embs[i]))
                    if start - current_end < MIN_MERGE_GAP and sim > 0.7 and labels[i] == current_label:
                        current_end = end
                        current_prob = max(current_prob, prob)
                    else:
                        merged.append((current_start - PADDING, current_end + PADDING, current_prob))
                        current_start, current_end, current_prob = start, end, prob
                        current_label = labels[i]
                else:
                    if start - current_end < MIN_MERGE_GAP:
                        current_end = end
                        current_prob = max(current_prob, prob)
                    else:
                        merged.append((current_start - PADDING, current_end + PADDING, current_prob))
                        current_start, current_end, current_prob = start, end, prob
            merged.append((current_start - PADDING, current_end + PADDING, current_prob))
        
        audio_dur = len(audio) / SAMPLE_RATE
        merged = [(max(0, s), min(audio_dur, e), p) for s, e, p in merged]
        
        stats = {'noise_amp': float(noise_amp), 'slice_count': len(merged), 'boundary_probs': [p for _, _, p in merged]}
        
        return merged, stats
    except Exception as e:
        logging.error(f"Error in dynamic_slice: {e}")
        return [], {'noise_amp': 0, 'slice_count': 0, 'boundary_probs': []}
