"""
Module for dual speaker embedding extraction and fusion with parallel processing and learned weights.
"""

import numpy as np
import torch
from speechbrain.inference import EncoderClassifier
from resemblyzer import VoiceEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear
from concurrent.futures import ThreadPoolExecutor
import logging

SAMPLE_RATE = 16000

# Determine device
device = 'cuda' if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 4e9 else 'cpu'

# Load pre-trained speaker embedding models
ecapa = EncoderClassifier.from_hparams(
    source=r"C:\Users\trist\OneDrive\Documents\Remastered TTS Final Version\TTS Core Remastered\models\ecapa-voxceleb",
    run_opts={"device": device}
)
res_encoder = VoiceEncoder(device=device)

# Fusion layer placeholder (to be loaded with trained weights later)
fusion_linear = Linear(2, 2).to(device)

# Dynamically infer embedding dimensions
try:
    with torch.no_grad():
        dummy_audio = torch.zeros((1, 1, SAMPLE_RATE), device=device)
        dummy_emb = ecapa.encode_batch(dummy_audio).squeeze().cpu().numpy()
        ecapa_dim = dummy_emb.shape[-1]
except Exception as e:
    logging.warning(f"Cannot infer ECAPA embedding dim, defaulting to 192: {e}")
    ecapa_dim = 192

# Resemblyzer outputs 256-dimensional embeddings
res_dim = 256

# Build transformer encoder for contextualization
d_model = ecapa_dim + res_dim
transformer_layer = TransformerEncoderLayer(d_model=d_model, nhead=4)
transformer_encoder = TransformerEncoder(transformer_layer, num_layers=2).to(device)

def extract_emb(audio, slices, noise_amp=None):
    """
    Extract and fuse speaker embeddings for each audio slice.
    Returns a (num_slices, d_model) NumPy array of contextual embeddings.
    """
    try:
        # Define single-slice extraction functions
        def extract_ecapa(slice_audio):
            slice_t = torch.tensor(slice_audio).unsqueeze(0).unsqueeze(0).float().to(device)
            emb = ecapa.encode_batch(slice_t).squeeze().cpu().numpy()
            return emb / (np.linalg.norm(emb) + 1e-6)

        def extract_res(slice_audio):
            try:
                emb = res_encoder.embed_utterance(slice_audio)
                return emb / (np.linalg.norm(emb) + 1e-6)
            except Exception as e:
                logging.error(f"Resemblyzer failed: {e}")
                return None

        # Parallel extraction
        with ThreadPoolExecutor() as executor:
            ecapa_futs = [executor.submit(extract_ecapa, audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]) for s, e, _ in slices]
            res_futs = [executor.submit(extract_res, audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]) for s, e, _ in slices]

            ecapa_embs = [f.result() for f in ecapa_futs]
            res_embs = []
            for idx, fut in enumerate(res_futs):
                r = fut.result()
                res_embs.append(r if r is not None else ecapa_embs[idx])

        # Fuse embeddings with learned weights
        fused = []
        for ec, r in zip(ecapa_embs, res_embs):
            conf_ec = np.linalg.norm(ec)
            conf_r = np.linalg.norm(r)
            weights = torch.softmax(fusion_linear(torch.tensor([conf_ec, conf_r], device=device)), dim=0).cpu().numpy()
            fused.append(np.concatenate([ec * weights[0], r * weights[1]]))

        embs = np.stack(fused, axis=0)
        embs_t = torch.from_numpy(embs).float().to(device).unsqueeze(1)
        contextual = transformer_encoder(embs_t).squeeze(1).cpu().numpy()
        return contextual

    except Exception as e:
        logging.error(f"Error in extract_emb: {e}")
        return np.zeros((len(slices), d_model))

