import argparse
import yaml
import os
import uuid
import threading
import subprocess
import sys
import torch
import whisper
import whisperx  # for alignment model loading

from modules.trigger.trigger import job_queue, run_trigger_watcher
from modules.diarization.diarization import run as diarization_run
from modules.prosody.prosody import run as prosody_run
from modules.drift.drift import run as drift_run
from modules.transcription.transcription import run as transcription_run
from modules.alignment.alignment import run as alignment_run
from modules.tier1.tier1 import run as tier1_run
from modules.tier2.tier2 import run as tier2_run
from modules.anomaly.anomaly import run as anomaly_run
from modules.fingerprint.fingerprint import run as fingerprint_run
from modules.arc.arc import run as arc_run
from modules.plot_map.plot_map import run as plot_map_run
from modules.observer.observer import run as observer_run
from modules.git_sync.git_sync import run as git_sync_run
from modules.utils.dynamic_learning import (
    load_tagged_data,
    update_validation_set,
    check_accuracy_drop
)

# In-memory status tracker
descriptions = {}


def log_gpu_status():
    """
    Log NVIDIA GPU status using nvidia-smi.
    """
    print("\n=== GPU status (nvidia-smi) ===")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except FileNotFoundError:
        print("  nvidia-smi not found: no NVIDIA GPU or driver missing.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  nvidia-smi failed (code {e.returncode})", file=sys.stderr)
    print("=" * 32 + "\n")


def pipeline(context):
    # 1. Speaker splitting (diarization)
    diarization_run(context)
    context['speaker_ids'] = [
        d for d in os.listdir(os.path.join(context['output_dir'], 'emotion_tags'))
        if os.path.isdir(os.path.join(context['output_dir'], 'emotion_tags', d))
    ]

    # 2. Prosody, drift, transcription, alignment
    prosody_run(context)
    drift_run(context)
    transcription_run(context)
    alignment_run(context)

    # 3. Tier tagging & anomaly detection
    tier1_run(context)
    tier2_run(context)
    anomaly_run(context)

    # 4. Downstream analytics: fingerprinting, arc analysis, and plot mapping
    fingerprint_run(context)
    arc_run(context)
    plot_map_run(context)
    observer_run(context)

    # 5. Dynamic learning updates
    data_root = os.path.join(context['output_dir'], 'emotion_tags')
    validation_path = os.path.join(context['output_dir'], 'validation_set.json')
    load_tagged_data(data_root)
    update_validation_set(data_root, validation_path, sample_frac=0.05, max_samples=500)
    old_acc = context.get('old_accuracy', 1.0)
    new_acc = context.get('new_accuracy', 1.0)
    check_accuracy_drop(old_acc, new_acc)

    # 6. Git synchronization
    git_sync_run(context)


def enqueue_job(config, job_id, input_wav):
    output_base = config['global']['output_base']
    output_dir = os.path.join(output_base, job_id)
    os.makedirs(os.path.join(output_dir, 'emotion_tags'), exist_ok=True)

    context = {
        'job_id': job_id,
        'input_wav': input_wav,
        'output_dir': output_dir,
        'speaker_ids': [],
        'config': config
    }

    # Load models
    whisper_model_name = config.get('diarization', {}).get('model_name', 'small')
    context['models'] = {}
    context['models']['whisper'] = whisper.load_model(whisper_model_name)

    lang = config.get('diarization', {}).get('language', 'en')
    device = 'cuda' if config['global'].get('use_gpu', False) and torch.cuda.is_available() else 'cpu'
    align_model, metadata = whisperx.load_align_model(
        language_code=lang,
        device=device
    )
    context['models']['align'] = (align_model, metadata)

    pipeline(context)


def worker():
    while True:
        config, job_id, input_wav = job_queue.get()
        descriptions[job_id] = 'processing'
        try:
            enqueue_job(config, job_id, input_wav)
            descriptions[job_id] = 'done'
        except Exception as e:
            descriptions[job_id] = 'failed'
            print(f"Job {job_id} failed: {e}")
        finally:
            job_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description='Emotion Drift & Observer v3.1')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--job', help='Path to input WAV file')
    args = parser.parse_args()

    log_gpu_status()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.job:
        job_id = str(uuid.uuid4())
        descriptions[job_id] = 'queued'
        enqueue_job(config, job_id, args.job)
    elif args.watch:
        run_trigger_watcher(config['global'])
        threading.Thread(target=worker, daemon=True).start()
        threading.Event().wait()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
