# hyper_diarizer.py
"""
Orchestrator for HyperDiarizer modules with CLI, logging, and feedback loops.
"""

import os
import json
import argparse
import torchaudio
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm
import time
from multiprocessing import Pool, cpu_count
from .slicer import dynamic_slice
from .embedding import extract_emb
from .cluster import time_aware_sim, temporal_cluster
from .reid import ReIDMemory as ReidMemory
from .audio_rebuilder import reconstruct_audio
from .overlap import detect_overlaps
import logging
from logging.handlers import RotatingFileHandler
from pyannote.core import Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity

SAMPLE_RATE = 16000

logging.basicConfig(handlers=[RotatingFileHandler('diarizer.log', maxBytes=10**6, backupCount=5)], level=logging.INFO)

def visualize_results(sim, labels, output_dir, slices):
    plt.imshow(sim, cmap='viridis')
    plt.colorbar()
    plt.title('Similarity Matrix')
    plt.savefig(os.path.join(output_dir, 'sim_matrix.png'))
    
    fig = go.Figure()
    for s, e, _ in slices:
        fig.add_vrect(x0=s, x1=e, fillcolor="blue", opacity=0.2)
    fig.write_html(os.path.join(output_dir, 'timeline.html'))

class DiarizerController:
    def __init__(self, args):
        self.args = args
        self.reid = ReIDMemory(thresh=args.voiceprint_thresh)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buffer = []  # For streaming

    def process_chunk(self, audio_chunk):
        slices, stats = dynamic_slice(audio_chunk, self.device)
        embs = extract_emb(audio_chunk, slices, stats['noise_amp'])
        sim = time_aware_sim(embs, slices)
        labels = temporal_cluster(sim, slices)
        labels, _, certainties = self.reid.re_id(embs, labels)
        # Append to buffer, etc.
        return labels, certainties

    def diarize_single(self, audio_path, output_dir, visualize=False, streaming=False):
        try:
            waveform, orig_sr = torchaudio.load(audio_path)
            if orig_sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)
                waveform = resampler(waveform)
            audio = waveform.squeeze().numpy()

            if streaming:
                chunk_size = SAMPLE_RATE * 10
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i+chunk_size]
                    labels, certainties = self.process_chunk(chunk)
                    # Process incrementally
                return {}

            times = {}
            t0 = time.time()
            slices, slicer_stats = dynamic_slice(audio, self.device)
            times['slicing'] = time.time() - t0

            t0 = time.time()
            embs = extract_emb(audio, slices, slicer_stats['noise_amp'])
            times['embedding'] = time.time() - t0

            t0 = time.time()
            sim = time_aware_sim(embs, slices)
            times['sim'] = time.time() - t0

            t0 = time.time()
            labels = temporal_cluster(sim, slices)
            times['clustering'] = time.time() - t0

            t0 = time.time()
            labels, label_map, certainties = self.reid.re_id(embs, labels, use_transformer=not self.args.hmm_off)
            times['reid'] = time.time() - t0

            if np.mean(certainties) < 0.7:
                logging.warning("Low certainty; reslicing with adjusted params")
                slices, slicer_stats = dynamic_slice(audio, self.device, min_slice_dur=MIN_SLICE_DUR * 0.8)

            t0 = time.time()
            overlaps = detect_overlaps(audio, slices, labels, embs)
            times['overlap'] = time.time() - t0

            t0 = time.time()
            speaker_paths = reconstruct_audio(audio_path, slices, labels, overlaps, output_dir, certainties)
            times['rebuild'] = time.time() - t0

            with open(os.path.join(output_dir, 'speaker_mapping.json'), 'w') as f:
                json.dump(speaker_paths, f, indent=2)
            np.save(os.path.join(output_dir, 'similarity_matrix.npy'), sim)
            np.save(os.path.join(output_dir, 'certainties.npy'), certainties)
            with open(os.path.join(output_dir, 'diarization_log.json'), 'w') as f:
                json.dump({'overlaps': overlaps, 'label_map': label_map}, f, indent=2)
            with open(os.path.join(output_dir, 'step_times.json'), 'w') as f:
                json.dump(times, f, indent=2)
            with open(os.path.join(output_dir, 'slicer_stats.json'), 'w') as f:
                json.dump(slicer_stats, f, indent=2)

            if visualize:
                visualize_results(sim, labels, output_dir, slices)

            return {'speaker_paths': speaker_paths}
        except RuntimeError as e:
            logging.error(f"Audio load error: {e}")
            return {}
        except Exception as e:
            logging.error(f"Diarization error: {e}", exc_info=True)
            return {}

def diarize_wrapper(tup):
    controller, audio_path, sub_out_dir, visualize = tup
    os.makedirs(sub_out_dir, exist_ok=True)
    controller.diarize_single(audio_path, sub_out_dir, visualize)

def calculate_der(gt_path, hyp_path):
    try:
        from pyannote.core import Annotation
        from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity
        
        # Assume gt_path and hyp_path are RTTM files
        reference = Annotation.from_rttm(gt_path)
        hypothesis = Annotation.from_rttm(hyp_path)
        
        der = DiarizationErrorRate()
        purity = DiarizationPurity()
        
        der_score = der(reference, hypothesis)
        purity_score = purity(reference, hypothesis)
        
        return {'der': der_score, 'purity': purity_score}
    except Exception as e:
        logging.error(f"DER calculation error: {e}")
        return {'der': 0.0, 'purity': 0.0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperDiarizer CLI")
    parser.add_argument('input', type=str, help="Input audio file or directory")
    parser.add_argument('output_dir', type=str, help="Output directory")
    parser.add_argument('--visualize', action='store_true', help="Generate visualizations")
    parser.add_argument('--batch', action='store_true', help="Batch mode for folder input")
    parser.add_argument('--eval', type=str, help="Path to GT RTTM for DER evaluation")
    parser.add_argument('--hmm_off', action='store_true', help="Disable transformer smoothing")
    parser.add_argument('--voiceprint_thresh', type=float, default=0.6, help="Voiceprint similarity threshold")
    parser.add_argument('--streaming', action='store_true', help="Real-time streaming mode")
    args = parser.parse_args()

    controller = DiarizerController(args)

    if args.batch:
        audio_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('.wav', '.mp3'))]
        pool_size = min(cpu_count(), len(audio_files))
        tasks = [(controller, audio_path, os.path.join(args.output_dir, os.path.basename(audio_path).rsplit('.', 1)[0]), args.visualize) for audio_path in audio_files]
        with Pool(processes=pool_size) as pool:
            pool.map(diarize_wrapper, tasks)
    else:
        controller.diarize_single(args.input, args.output_dir, args.visualize, args.streaming)

    if args.eval:
        der_results = calculate_der(args.eval, os.path.join(args.output_dir, 'diarization.rttm'))  # Assume hyp RTTM
        print(f"DER: {der_results['der']}, Purity: {der_results['purity']}")