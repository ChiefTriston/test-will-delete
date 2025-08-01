"""
Tier 2 emotion refinement with NLP, heuristics, ESR, and dynamic thresholds.
Outputs tier2_tags.json per speaker.
"""
import json
import spacy
from negspacy.negation import Negex
import stanza
import portalocker
import os
import numpy as np
import torch
import torchaudio
from modules.utils.emotion_utils import (
    emotion_rules, GROUP_MAP, T2_AUTO, T2_MIN, SENTIMENT_STD_THRESHOLD
)
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLP pipelines once
nlp_spacy = spacy.load("en_core_web_sm")
nlp_spacy.add_pipe("negex")
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

def run(context):
    """
    Process Tier 1 tags and transcripts to produce Tier 2 emotion tags.
    Args:
        context (dict): pipeline context with keys:
            - 'output_dir': str base output directory
            - 'speaker_ids': list of speaker IDs
            - 'config': dict with 'tier2' and 'global' settings
    Returns:
        dict: mapping speaker ID to path of generated tier2_tags.json
    """
    cfg = context['config']['tier2']
    gcfg = context['config']['global']
    neg_w = cfg.get('negation_weight', 1.0)
    sr = gcfg['sample_rate']
    use_gpu = gcfg.get('use_gpu', False) and torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    # Load speaker embedding model once
    model = torch.hub.load('RF5/simple-speaker-embedding', 'convgru_embedder')
    model.to(device).eval()
    resampler = torchaudio.transforms.Resample(sr, 16000) if sr != 16000 else None

    results = {}
    for spk in context['speaker_ids']:
        spk_dir = os.path.join(context['output_dir'], 'emotion_tags', spk)
        
        # Load Tier 1 tags
        tier1_path = os.path.join(spk_dir, 'tier1_tags.json')
        with open(tier1_path, 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            tier1 = json.load(f)
            portalocker.unlock(f)

        # Load transcript
        transcript_path = os.path.join(spk_dir, 'transcript.json')
        with open(transcript_path, 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            transcript = json.load(f)
            portalocker.unlock(f)

        # Load prosody trends
        prosody_path = os.path.join(spk_dir, 'prosody_trend.json')
        with open(prosody_path, 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            prosody = json.load(f)
            portalocker.unlock(f)

        # Load drift data
        drift_path = os.path.join(spk_dir, 'drift_vector.json')
        with open(drift_path, 'r') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            drift = json.load(f)
            portalocker.unlock(f)

        # Prepare prosody features
        time_arr = np.array(prosody['frame_series']['time'])
        f0_z = np.array(prosody['frame_series']['f0_z'])
        energy_z = np.array(prosody['frame_series']['energy_z'])
        pros_comb = (f0_z + energy_z) / 2

        # Sentence-level stats
        compounds = [abs(t['compound']) for t in tier1]
        sent_amp = np.ptp(compounds) if compounds else 0.0
        sent_std = np.std(compounds) if compounds else 0.0
        drift_score = np.mean(np.abs(drift.get('deltas', []))) if drift.get('deltas') else 0.0

        emb_cache = {}
        tier2_tags = []

        # Process each slice
        for idx, seg in enumerate(transcript.get('slices', [])):
            text = seg.get('text', '')
            base_tag = tier1[idx]['tag']
            conf = abs(tier1[idx]['compound'])
            rule_id = 'base'

            # Negation inversion
            doc_sp = nlp_spacy(text)
            if any(ent._.negex for ent in doc_sp.ents):
                rule_id = 'negation_invert'
                base_tag = 'negative' if base_tag == 'positive' else 'positive'
                conf *= neg_w

            # Contradiction heuristic
            doc_st = nlp_stanza(text)
            for sent in doc_st.sentences:
                words = [w.text.lower() for w in sent.words]
                if 'should' in words and 'happy' in words:
                    rule_id = 'contradiction'
                    base_tag = 'despair'
                    conf *= 0.8
                    break

            # Prosody slice score
            s_time, e_time = seg.get('start', 0.0), seg.get('end', 0.0)
            si = np.searchsorted(time_arr, s_time)
            ei = np.searchsorted(time_arr, e_time)
            pros_score = float(np.mean(pros_comb[si:ei])) if ei > si else 0.0

            # Apply emotion_rules heuristics
            label = base_tag
            for emo, (fn, _) in emotion_rules.items():
                features = {
                    'pos': conf,
                    'neg': 1 - conf,
                    'neu': 1 - abs(conf),
                    'pitch_mean': pros_score,
                    'energy_mean': float(np.mean(energy_z[si:ei])) if ei > si else 0.0,
                    'keywords': text.lower().split()
                }
                if fn(features):
                    label = emo
                    rule_id = f"rule_{emo}"
                    break

            # Speaker embedding & ESR
            wav_path = os.path.join(spk_dir, f"{spk}.wav")
            waveform, _ = torchaudio.load(wav_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            start_s, end_s = int(s_time * sr), int(e_time * sr)
            slice_wave = waveform[:, start_s:end_s]
            if resampler:
                slice_wave = resampler(slice_wave)
            slice_wave = slice_wave.to(device)
            with torch.no_grad():
                emb = model(slice_wave).cpu().numpy().squeeze(0)
            stats = emb_cache.setdefault(spk, {'mean': None, 'n': 0})
            if stats['mean'] is None:
                stats['mean'], stats['n'] = emb, 1
            else:
                stats['mean'] = (stats['mean'] * stats['n'] + emb) / (stats['n'] + 1)
                stats['n'] += 1
            cos_sim = float(
                cosine_similarity([emb], [stats['mean']])[0][0]
            ) if stats['n'] > 1 else 0.0
            esr_score = max(conf, cos_sim)

            # Compute final Tier-2 confidence
            t2_conf = conf * (1 + min(0.3, drift_score) + min(0.2, sent_amp))

            # Determine status
            if t2_conf >= T2_AUTO:
                status = 'auto-accepted'
            elif t2_conf >= T2_MIN:
                status = 'needs-review'
            else:
                status = 'auto-reject'
            if status == 'auto-accepted' and sent_std > SENTIMENT_STD_THRESHOLD:
                status = 'needs-review'

            tier2_tags.append({
                'label': label,
                'confidence': t2_conf,
                'rule_id': rule_id,
                'esr_score': esr_score,
                'status': status
            })

        # Write Tier-2 tags
        out_path = os.path.join(spk_dir, 'tier2_tags.json')
        with open(out_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(tier2_tags, f, indent=2)
            portalocker.unlock(f)
        results[spk] = out_path

    return {'tier2_tags': results}