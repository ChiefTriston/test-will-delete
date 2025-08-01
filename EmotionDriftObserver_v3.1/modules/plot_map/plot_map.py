
import os
import json
import numpy as np
from transformers import pipeline
import torch
import portalocker
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict
from modules.utils.plot_utils import plotly_trends

# Configure logging to match HyperDiazer/EmotionDriftObserver style
logging.basicConfig(
    handlers=[RotatingFileHandler('plot_map.log', maxBytes=10**6, backupCount=5)],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run(context):
    """
    Generate a plot map by summarizing transcripts into narrative beats, incorporating arc classification
    and speaker fingerprints, and tagging slices with beat IDs. Outputs plot_map.json at job level with
    plot paths and updates drift_vector.json per speaker with beat IDs.

    Args:
        context (dict): Pipeline context with keys:
            - 'output_dir': str base output directory
            - 'speaker_ids': list of speaker IDs
            - 'config': dict with 'plot_map' and 'global' settings

    Returns:
        dict: {'plot_map': path to plot_map.json}
    """
    try:
        # Extract config and setup
        cfg = context['config'].get('plot_map', {})
        gcfg = context['config']['global']
        out_dir = context['output_dir']
        speakers = context['speaker_ids']
        num_beats = cfg.get('num_beats', 8)
        beats_per_arc = cfg.get('beats_per_arc', 8)  # New: micro-beats per arc segment
        max_summary_len = cfg.get('max_summary_length', 60)
        min_summary_len = cfg.get('min_summary_length', 20)
        arc_confidence_threshold = cfg.get('arc_confidence_threshold', 0.7)
        use_gpu = gcfg.get('use_gpu', False) and torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
        plot_dir = os.path.join(out_dir, gcfg.get('plot_map_dir', 'plot_maps'))
        os.makedirs(plot_dir, exist_ok=True)

        logging.info(f"Starting plot_map for job {context.get('job_id', 'unknown')} on {device} with {len(speakers)} speakers")

        # Initialize summarizer with half-precision on GPU for memory savings
        try:
            if use_gpu:
                summarizer = pipeline(
                    "summarization",
                    model=cfg.get("summarizer_model", "facebook/bart-large-cnn"),
                    device=0,
                    framework="pt",
                    torch_dtype=torch.float16
                )
            else:
                summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1,
                    framework="pt"
                )
            logging.info(f"Summarizer loaded on {device} with {'FP16' if use_gpu else 'FP32'}")
        except Exception as e:
            logging.error(f"Failed to load summarizer: {e}")
            return {'plot_map': None}

        # Load arc classification
        arc_path = os.path.join(out_dir, 'arc_classification.json')
        try:
            with open(arc_path, 'r') as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                arc_data = json.load(f)
                portalocker.unlock(f)
            arc_segments = arc_data.get('arc_segments', [])
            named_arc = arc_data.get('named_arc', 'custom')
            pivot_times = arc_data.get('pivots', [])
            logging.info(f"Loaded arc_classification.json: {named_arc}, {len(arc_segments)} segments")
        except FileNotFoundError:
            logging.warning("arc_classification.json not found; using default arc")
            arc_segments = []
            named_arc = 'neutral'
            pivot_times = []
        except Exception as e:
            logging.error(f"Error loading arc_classification.json: {e}")
            arc_segments = []
            named_arc = 'neutral'
            pivot_times = []

        # Load speaker fingerprints
        fingerprints = {}
        for spk in speakers:
            spk_dir = os.path.join(out_dir, 'emotion_tags', spk)
            try:
                with open(os.path.join(spk_dir, 'fingerprint.json'), 'r') as f:
                    portalocker.lock(f, portalocker.LOCK_SH)
                    fingerprints[spk] = json.load(f)
                    portalocker.unlock(f)
            except FileNotFoundError:
                logging.warning(f"fingerprint.json not found for {spk}; using defaults")
                fingerprints[spk] = {
                    'dominant_tags': [('neutral', 1)],
                    'avg_confidence': 0.0,
                    'entropy': 0.0,
                    'avg_drift': 0.0,
                    'drift_slope': 0.0
                }
            except Exception as e:
                logging.error(f"Error loading fingerprint.json for {spk}: {e}")
                fingerprints[spk] = {
                    'dominant_tags': [('neutral', 1)],
                    'avg_confidence': 0.0,
                    'entropy': 0.0,
                    'avg_drift': 0.0,
                    'drift_slope': 0.0
                }

        # 1. Gather all transcripts with timestamps and metadata
        slices = []
        for spk in speakers:
            spk_dir = os.path.join(out_dir, 'emotion_tags', spk)
            try:
                with open(os.path.join(spk_dir, 'transcript.json'), 'r') as f:
                    portalocker.lock(f, portalocker.LOCK_SH)
                    txt = json.load(f)
                    portalocker.unlock(f)
                with open(os.path.join(spk_dir, 'drift_vector.json'), 'r') as f:
                    portalocker.lock(f, portalocker.LOCK_SH)
                    drift = json.load(f)
                    portalocker.unlock(f)
                with open(os.path.join(spk_dir, 'tier2_tags.json'), 'r') as f:
                    portalocker.lock(f, portalocker.LOCK_SH)
                    tier2 = json.load(f)
                    portalocker.unlock(f)
                with open(os.path.join(spk_dir, 'prosody_trend.json'), 'r') as f:
                    portalocker.lock(f, portalocker.LOCK_SH)
                    prosody = json.load(f)
                    portalocker.unlock(f)

                transcript_slices = txt.get('slices', [])
                boundaries = drift.get('slice_boundaries', [])
                if len(transcript_slices) != len(boundaries):
                    logging.warning(
                        f"Mismatch in {spk}: {len(transcript_slices)} transcript slices vs {len(boundaries)} drift boundaries"
                    )
                    continue

                time_arr = np.array(prosody['frame_series']['time'])
                f0_z = np.array(prosody['frame_series']['f0_z'])
                energy_z = np.array(prosody['frame_series']['energy_z'])

                for idx, (start, end) in enumerate(boundaries):
                    slice_text = transcript_slices[idx].get('text', '') if idx < len(transcript_slices) else ''
                    emotion = tier2[idx].get('label', 'neutral') if idx < len(tier2) else 'neutral'
                    confidence = tier2[idx].get('confidence', 0.0) if idx < len(tier2) else 0.0
                    si = np.searchsorted(time_arr, start)
                    ei = np.searchsorted(time_arr, end)
                    prosody_score = (
                        float(np.mean((f0_z[si:ei] + energy_z[si:ei]) / 2)) if ei > si else 0.0
                    )
                    slices.append({
                        'speaker': spk,
                        'start': start,
                        'end': end,
                        'text': slice_text,
                        'emotion': emotion,
                        'confidence': confidence,
                        'prosody_score': prosody_score
                    })

            except FileNotFoundError as e:
                logging.error(f"Missing file for {spk}: {e}")
                continue
            except Exception as e:
                logging.error(f"Error processing {spk}: {e}")
                continue

        if not slices:
            logging.error("No valid slices found; aborting plot_map")
            return {'plot_map': None}

        # 2. Sort slices chronologically
        slices.sort(key=lambda x: x['start'])
        total_duration = max(s['end'] for s in slices) if slices else 1.0

        # 3. Time-based chunking for beats, respecting arc pivots
        beats = []
        if arc_segments:
            for seg_idx, seg in enumerate(arc_segments):
                seg_start = seg['start']
                seg_end = seg['end']
                seg_emotion = seg['segment']
                # Sort slices within this arc segment
                seg_slices = sorted(
                    [s for s in slices if seg_start <= s['start'] < seg_end],
                    key=lambda x: x['start']
                )
                seg_confidence = float(np.mean([s['confidence'] for s in seg_slices])) if seg_slices else 0.0

                if not seg_slices:
                    continue

                # Subdivide into micro-beats
                micro_count = max(1, min(beats_per_arc, len(seg_slices)))  # Ensure at least 1 micro-beat
                dur = (seg_end - seg_start) / micro_count if micro_count > 0 else seg_end - seg_start
                for i in range(micro_count):
                    micro_start = seg_start + i * dur
                    micro_end = seg_start + (i + 1) * dur
                    micro_slices = [s for s in seg_slices if micro_start <= s['start'] < micro_end]
                    if not micro_slices:
                        continue

                    chunk_text = " ".join(s['text'] for s in micro_slices if s['text'])
                    safe_chunk = chunk_text[:1024]  # Memory optimization
                    if len(safe_chunk.strip()) < 50:
                        summary = "Too short to summarize"
                    else:
                        try:
                            summary = summarizer(
                                safe_chunk,
                                max_length=max_summary_len,
                                min_length=min_summary_len,
                                do_sample=False
                            )[0]['summary_text']
                            if seg_confidence < arc_confidence_threshold:
                                summary = f"[Low confidence {seg_confidence:.2f}] {summary}"
                        except Exception as e:
                            logging.warning(f"Summary failed for micro-beat {seg_idx}.{i}: {e}")
                            summary = safe_chunk[:100] + "..." if len(safe_chunk) > 100 else safe_chunk

                    dominant_emotion = max(
                        set(s['emotion'] for s in micro_slices),
                        key=lambda e: sum(1 for s in micro_slices if s['emotion'] == e),
                        default=seg_emotion
                    )
                    dominant_speaker = max(
                        set(s['speaker'] for s in micro_slices),
                        key=lambda spk: sum(1 for s in micro_slices if s['speaker'] == spk),
                        default=speakers[0] if speakers else 'unknown'
                    )
                    speaker_fingerprint = fingerprints.get(dominant_speaker, {})
                    dominant_tag = speaker_fingerprint['dominant_tags'][0][0] if speaker_fingerprint['dominant_tags'] else 'neutral'
                    entropy = speaker_fingerprint.get('entropy', 0.0)
                    title = (
                        f"{'High' if seg_confidence >= arc_confidence_threshold else 'Low'}-confidence "
                        f"{named_arc} ({dominant_emotion}, {dominant_speaker} {dominant_tag}): "
                        f"{summary.split('.')[0][:50]}"
                    )

                    speaker_insights = {
                        spk: {
                            'dominant_tag': fingerprints[spk]['dominant_tags'][0][0] if fingerprints[spk]['dominant_tags'] else 'neutral',
                            'avg_confidence': fingerprints[spk].get('avg_confidence', 0.0),
                            'entropy': fingerprints[spk].get('entropy', 0.0),
                            'avg_drift': fingerprints[spk].get('avg_drift', 0.0),
                            'drift_slope': fingerprints[spk].get('drift_slope', 0.0)
                        } for spk in set(s['speaker'] for s in micro_slices)
                    }

                    # Generate micro-beat visualization
                    beat_plot_path = os.path.join(plot_dir, f"beat_{seg_idx}_{i}.html")
                    try:
                        seg_times = [s['start'] for s in micro_slices] + [micro_slices[-1]['end']] if micro_slices else [micro_start, micro_end]
                        seg_f0_z = np.mean([s['prosody_score'] for s in micro_slices]) if micro_slices else 0.0
                        seg_energy_z = seg_confidence
                        fig = plotly_trends(seg_times, [seg_f0_z] * len(seg_times), [seg_energy_z] * len(seg_times))
                        fig.add_vrect(
                            x0=micro_start, x1=micro_end,
                            fillcolor="blue", opacity=0.2,
                            annotation_text=f"{dominant_emotion} ({seg_confidence:.2f})",
                            annotation_position="top left"
                        )
                        fig.write_html(beat_plot_path)
                    except Exception as e:
                        logging.warning(f"Plot failed for micro-beat {seg_idx}.{i}: {e}")
                        beat_plot_path = None

                    beats.append({
                        'beat_id': f"{seg_idx}.{i}",
                        'title': title,
                        'start': micro_start,
                        'end': micro_end,
                        'summary': summary,
                        'speakers': list({s['speaker'] for s in micro_slices}),
                        'dominant_emotion': dominant_emotion,
                        'avg_prosody_score': float(np.mean([s['prosody_score'] for s in micro_slices])) if micro_slices else 0.0,
                        'arc_segment': seg_emotion,
                        'arc_confidence': seg_confidence,
                        'named_arc': named_arc,
                        'speaker_insights': speaker_insights,
                        'plot_path': beat_plot_path
                    })
        else:
            # Fallback to time-based chunking
            beat_duration = total_duration / max(num_beats, 1)
            current_slices = []
            current_start = slices[0]['start']
            beat_id = 0
            for slice_data in slices:
                current_slices.append(slice_data)
                if slice_data['end'] >= current_start + beat_duration or slice_data is slices[-1]:
                    chunk_text = " ".join(s['text'] for s in current_slices if s['text'])
                    safe_chunk = chunk_text[:1024]
                    if len(safe_chunk.strip()) < 50:
                        summary = "Too short to summarize"
                    else:
                        try:
                            summary = summarizer(
                                safe_chunk,
                                max_length=max_summary_len,
                                min_length=min_summary_len,
                                do_sample=False
                            )[0]['summary_text']
                        except Exception as e:
                            logging.warning(f"Summary failed for beat {beat_id}: {e}")
                            summary = safe_chunk[:100] + "..." if len(safe_chunk) > 100 else safe_chunk

                    dominant_emotion = max(
                        set(s['emotion'] for s in current_slices),
                        key=lambda e: sum(1 for s in current_slices if s['emotion'] == e),
                        default='neutral'
                    )
                    dominant_speaker = max(
                        set(s['speaker'] for s in current_slices),
                        key=lambda spk: sum(1 for s in current_slices if s['speaker'] == spk),
                        default=speakers[0] if speakers else 'unknown'
                    )
                    speaker_fingerprint = fingerprints.get(dominant_speaker, {})
                    dominant_tag = speaker_fingerprint['dominant_tags'][0][0] if speaker_fingerprint['dominant_tags'] else 'neutral'
                    entropy = speaker_fingerprint.get('entropy', 0.0)
                    title = (
                        f"Low-confidence {named_arc} ({dominant_emotion}, {dominant_speaker} {dominant_tag}): "
                        f"{summary.split('.')[0][:50]}"
                    )

                    speaker_insights = {
                        spk: {
                            'dominant_tag': fingerprints[spk]['dominant_tags'][0][0] if fingerprints[spk]['dominant_tags'] else 'neutral',
                            'avg_confidence': fingerprints[spk].get('avg_confidence', 0.0),
                            'entropy': fingerprints[spk].get('entropy', 0.0),
                            'avg_drift': fingerprints[spk].get('avg_drift', 0.0),
                            'drift_slope': fingerprints[spk].get('drift_slope', 0.0)
                        } for spk in set(s['speaker'] for s in current_slices)
                    }

                    # Generate beat visualization
                    beat_plot_path = os.path.join(plot_dir, f"beat_{beat_id}.html")
                    try:
                        seg_times = [s['start'] for s in current_slices] + [current_slices[-1]['end']] if current_slices else [current_start, current_slices[-1]['end']]
                        seg_f0_z = np.mean([s['prosody_score'] for s in current_slices]) if current_slices else 0.0
                        seg_energy_z = 0.0
                        fig = plotly_trends(seg_times, [seg_f0_z] * len(seg_times), [seg_energy_z] * len(seg_times))
                        fig.add_vrect(
                            x0=current_slices[0]['start'], x1=current_slices[-1]['end'],
                            fillcolor="blue", opacity=0.2,
                            annotation_text=f"{dominant_emotion}",
                            annotation_position="top left"
                        )
                        fig.write_html(beat_plot_path)
                    except Exception as e:
                        logging.warning(f"Plot failed for beat {beat_id}: {e}")
                        beat_plot_path = None

                    beats.append({
                        'beat_id': str(beat_id),
                        'title': title,
                        'start': current_slices[0]['start'],
                        'end': current_slices[-1]['end'],
                        'summary': summary,
                        'speakers': list({s['speaker'] for s in current_slices}),
                        'dominant_emotion': dominant_emotion,
                        'avg_prosody_score': float(np.mean([s['prosody_score'] for s in current_slices])) if current_slices else 0.0,
                        'arc_segment': 'neutral',
                        'arc_confidence': 0.0,
                        'named_arc': named_arc,
                        'speaker_insights': speaker_insights,
                        'plot_path': beat_plot_path
                    })

                    beat_id += 1
                    current_slices = []
                    current_start = slice_data['end']

        # Ensure at least one beat
        if not beats:
            logging.warning("No beats generated; creating default beat")
            beat_plot_path = os.path.join(plot_dir, "beat_0.html")
            try:
                fig = plotly_trends([0.0, total_duration], [0.0, 0.0], [0.0, 0.0])
                fig.add_vrect(
                    x0=0.0, x1=total_duration,
                    fillcolor="blue", opacity=0.2,
                    annotation_text="No content",
                    annotation_position="top left"
                )
                fig.write_html(beat_plot_path)
            except Exception as e:
                logging.warning(f"Failed to generate default beat plot: {e}")
                beat_plot_path = None

            beats = [{
                'beat_id': "0",
                'title': f"{named_arc}: No content",
                'start': 0.0,
                'end': total_duration,
                'summary': "No spoken content available",
                'speakers': speakers,
                'dominant_emotion': 'neutral',
                'avg_prosody_score': 0.0,
                'arc_segment': 'neutral',
                'arc_confidence': 0.0,
                'named_arc': named_arc,
                'speaker_insights': {
                    spk: {
                        'dominant_tag': fingerprints[spk]['dominant_tags'][0][0] if fingerprints[spk]['dominant_tags'] else 'neutral',
                        'avg_confidence': fingerprints[spk].get('avg_confidence', 0.0),
                        'entropy': fingerprints[spk].get('entropy', 0.0),
                        'avg_drift': fingerprints[spk].get('avg_drift', 0.0),
                        'drift_slope': fingerprints[spk].get('drift_slope', 0.0)
                    } for spk in speakers
                },
                'plot_path': beat_plot_path
            }]

        # Adjust beat boundaries to align with pivot points if available
        if pivot_times:
            adjusted_beats = []
            pivot_idx = 0
            for beat in beats:
                while pivot_idx < len(pivot_times) and pivot_times[pivot_idx] < float(beat['start']):
                    pivot_idx += 1
                if pivot_idx < len(pivot_times) and float(beat['start']) < pivot_times[pivot_idx] < float(beat['end']):
                    # Split beat at pivot
                    adjusted_beats.append({
                        **beat,
                        'end': pivot_times[pivot_idx],
                        'title': f"{beat['title']} (Pre-Pivot)",
                        'plot_path': beat['plot_path'].replace('.html', '_pre.html') if beat['plot_path'] else None
                    })
                    adjusted_beats.append({
                        **beat,
                        'beat_id': f"{beat['beat_id']}.5",
                        'start': pivot_times[pivot_idx],
                        'title': f"{beat['title']} (Post-Pivot)",
                        'plot_path': beat['plot_path'].replace('.html', '_post.html') if beat['plot_path'] else None
                    })
                else:
                    adjusted_beats.append(beat)
            beats = adjusted_beats

        # 4. Write plot_map.json
        plot_map_path = os.path.join(out_dir, 'plot_map.json')
        try:
            with open(plot_map_path, 'w') as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                json.dump(beats, f, indent=2)
                portalocker.unlock(f)
            logging.info(f"Wrote plot_map.json to {plot_map_path}")
        except Exception as e:
            logging.error(f"Failed to write plot_map.json: {e}")
            return {'plot_map': None}

        # 5. Update drift_vector.json with beat_ids
        for spk in speakers:
            spk_dir = os.path.join(out_dir, 'emotion_tags', spk)
            drift_path = os.path.join(spk_dir, 'drift_vector.json')
            try:
                with open(drift_path, 'r+') as f:
                    portalocker.lock(f, portalocker.LOCK_EX)
                    drift = json.load(f)
                    beat_ids = []
                    for start, end in drift.get('slice_boundaries', []):
                        try:
                            bid = next(
                                b['beat_id'] for b in beats
                                if float(b['start']) <= start < float(b['end']) or
                                (b == beats[-1] and start >= float(b['start']))
                            )
                        except StopIteration:
                            bid = beats[-1]['beat_id']  # Fallback to last beat
                        beat_ids.append(bid)
                    drift['beat_ids'] = beat_ids
                    f.seek(0)
                    json.dump(drift, f, indent=2)
                    f.truncate()
                    portalocker.unlock(f)
                logging.info(f"Updated drift_vector.json for {spk} with beat_ids")
            except FileNotFoundError as e:
                logging.error(f"Missing drift_vector.json for {spk}: {e}")
                continue
            except Exception as e:
                logging.error(f"Error updating drift_vector.json for {spk}: {e}")
                continue

        return {'plot_map': plot_map_path}

    except Exception as e:
        logging.error(f"Plot map generation failed: {e}", exc_info=True)
        return {'plot_map': None}
