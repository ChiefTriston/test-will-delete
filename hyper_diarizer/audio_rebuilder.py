"""
Module for reconstructing per-speaker audio with source separation and ASR fallback.
"""

import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# Constants (overridden at runtime if needed)
SAMPLE_RATE   = 16000
SILENCE_PAD_MS = 100  # milliseconds padding between segments

# Initialize ASR model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(
    "large-v3",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)
# Initialize speaker separation pipeline
separation_pipeline = Pipeline.from_pretrained("pyannote/speaker-separation")


def reconstruct_audio(audio_path, slices, labels, overlaps, output_dir, certainties):
    """
    Reconstruct per-speaker WAV files with intervals and statistics.
    Returns a dict of speaker_id -> wav_path.
    """
    try:
        waveform, _ = torchaudio.load(audio_path)
        unique_labels = np.unique(labels)
        speaker_paths = {}
        emotion_tags_dir = os.path.join(output_dir, 'emotion_tags')
        os.makedirs(emotion_tags_dir, exist_ok=True)
        stats = {}

        def process_speaker(label):
            try:
                speaker_id = f"speaker_{label:02d}"
                speaker_dir = os.path.join(emotion_tags_dir, speaker_id)
                os.makedirs(speaker_dir, exist_ok=True)
                wav_path = os.path.join(speaker_dir, f"{speaker_id}.wav")
                intervals_path = os.path.join(speaker_dir, 'intervals.json')
                csv_path = os.path.join(speaker_dir, 'summary.csv')
                rttm_path = os.path.join(speaker_dir, 'diarization.rttm')

                intervals = []
                speaker_wave = torch.tensor([])
                total_dur = 0
                word_count = 0
                turn_count = 0
                silence = torch.zeros(int(SILENCE_PAD_MS / 1000 * SAMPLE_RATE))

                # Write RTTM and build intervals
                with open(rttm_path, 'w') as rttm_f:
                    for idx, (start, end, _) in enumerate(slices):
                        if labels[idx] == label:
                            segment = waveform[0, int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
                            try:
                                segments, info = model.transcribe(
                                    segment.numpy(), beam_size=5
                                )
                                text = ' '.join([s.text for s in segments])
                                conf = info.language_probability
                            except Exception as e:
                                logging.error(f"ASR failed: {e}")
                                text = ''
                                conf = 0
                            intervals.append({
                                'start': start,
                                'end': end,
                                'text': text,
                                'conf': conf,
                                'certainty': certainties[idx]
                            })
                            speaker_wave = torch.cat([speaker_wave, segment, silence])
                            total_dur += (end - start)
                            turn_count += 1
                            word_count += len(text.split())
                            rttm_f.write(
                                f"SPEAKER {speaker_id} 1 {start:.3f} {(end - start):.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
                            )

                    # Handle overlaps with separation
                    for gap_start, gap_end, l1, l2, conf in overlaps:
                        if label in (l1, l2):
                            segment = waveform[0, int(gap_start * SAMPLE_RATE):int(gap_end * SAMPLE_RATE)]
                            try:
                                separated = separation_pipeline({
                                    'waveform': segment.unsqueeze(0),
                                    'sample_rate': SAMPLE_RATE
                                })
                                speaker_seg = separated.get(label, segment)
                            except Exception as e:
                                logging.error(f"Separation failed: {e}")
                                speaker_seg = segment
                            speaker_wave = torch.cat([speaker_wave, silence, speaker_seg])
                            intervals.append({
                                'start': gap_start,
                                'end': gap_end,
                                'text': '[overlap]',
                                'overlap': True,
                                'conf': conf
                            })
                            total_dur += (gap_end - gap_start)
                            rttm_f.write(
                                f"SPEAKER {speaker_id} 1 {gap_start:.3f} {(gap_end - gap_start):.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
                            )

                if speaker_wave.numel() == 0:
                    return None

                # Normalize waveform
                max_amp = torch.max(torch.abs(speaker_wave)) + 1e-6
                speaker_wave /= max_amp
                speaker_wave *= 0.99

                torchaudio.save(wav_path, speaker_wave.unsqueeze(0), SAMPLE_RATE)
                # Write intervals JSON
                with open(intervals_path, 'w') as f:
                    json.dump(intervals, f, indent=2)
                # Write CSV stats
                with open(csv_path, 'w') as f:
                    f.write(f"Duration (s),{total_dur}\nWord Count,{word_count}\nTurn Count,{turn_count}\n")

                return speaker_id, wav_path, {
                    'duration': total_dur,
                    'word_count': word_count,
                    'turn_count': turn_count
                }
            except Exception as e:
                logging.error(f"Error processing speaker {label}: {e}")
                return None

        # Parallel processing of speakers
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_speaker, label) for label in unique_labels]
            for future in futures:
                result = future.result()
                if result:
                    speaker_id, wav_path, speaker_stats = result
                    speaker_paths[speaker_id] = wav_path
                    stats[speaker_id] = speaker_stats

        # Save overall stats
        with open(os.path.join(emotion_tags_dir, 'speaker_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        return speaker_paths

    except Exception as e:
        logging.error(f"Error in reconstruct_audio: {e}")
        return {}
