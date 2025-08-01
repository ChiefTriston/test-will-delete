"""
Module for overlap detection with multi-feature checks and intra-slice analysis.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import torch
import torch.nn as nn
import logging

# Constants (overridden at runtime if needed)
SAMPLE_RATE = 16000
OVERLAP_ENERGY_THRESH = 0.3  # dynamic energy threshold multiplier
SIM_OVERLAP_THRESH   = 0.5  # cosine similarity threshold


class OverlapClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        # assumes input length ~ SAMPLE_RATE//10
        self.fc = nn.Linear(32 * ((SAMPLE_RATE // 10) // 2), 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

# Initialize overlap detection model (to be loaded with trained weights externally)
overlap_model = OverlapClassifier()

def detect_overlaps(audio, slices, labels, embs):
    """
    Detect overlapping speech segments or high-energy gaps between slices.
    Returns a list of tuples: (start, end, speaker1, speaker2, confidence).
    """
    try:
        # Calculate energy per speaker for thresholding
        energies = {}
        for label in np.unique(labels):
            segs = [audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
                    for idx, (s, e, _) in enumerate(slices) if labels[idx] == label]
            if segs:
                energies[label] = np.mean([np.mean(seg**2) for seg in segs])
            else:
                energies[label] = 0
        median_energy = np.median(list(energies.values()))
        energy_thresh = OVERLAP_ENERGY_THRESH * median_energy
        overlaps = []

        # Intra-slice analysis
        for i, (s, e, _) in enumerate(slices):
            seg_audio = audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
            if len(seg_audio) < SAMPLE_RATE * 0.1:
                continue
            mfcc = librosa.feature.mfcc(y=seg_audio, sr=SAMPLE_RATE, n_mfcc=13)
            delta = librosa.feature.delta(mfcc)
            spectral_flux = librosa.onset.onset_strength(y=seg_audio, sr=SAMPLE_RATE)
            if np.mean(np.abs(delta)) > 0.5 or np.mean(spectral_flux) > 0.5:
                x = torch.from_numpy(seg_audio).float().unsqueeze(0).unsqueeze(0)
                prob = overlap_model(x).item()
                if prob > 0.5:
                    overlaps.append((s, e, labels[i], labels[i], prob))

            # Inter-slice gap analysis
            if i < len(slices) - 1:
                gap_start, gap_end = e, slices[i+1][0]
                gap_audio = audio[int(gap_start * SAMPLE_RATE):int(gap_end * SAMPLE_RATE)]
                gap_energy = np.mean(gap_audio**2)
                norm_energy = gap_energy / max(energies[labels[i]], energies[labels[i+1]], 1e-6)
                if norm_energy > energy_thresh:
                    sim = cosine_similarity([embs[i]], [embs[i+1]])[0][0]
                    if sim < SIM_OVERLAP_THRESH:
                        conf = norm_energy * (1 - sim)
                        overlaps.append((gap_start, gap_end, labels[i], labels[i+1], conf))

        return overlaps
    except Exception as e:
        logging.error(f"Error in detect_overlaps: {e}")
        return []
