# modules/arc/arc.py
"""
Global arc classification with pivot detection.
Outputs arc_classification.json at job level.
"""
import json
import os
import numpy as np
import math
from sklearn.cluster import KMeans
import portalocker
from collections import Counter


def infer_named_arc(sequence):
    """
    Map a sequence of dominant emotion labels to a named narrative arc.
    Extend <patterns> as needed.
    Sequence matching is case-insensitive.
    """
    # lower-case patterns for case-insensitive matching
    patterns = {
        ('hope', 'betrayal', 'resignation'): 'hope→betrayal→resignation',
        ('surprise', 'sadness'): 'excitement→despair',
        ('fear', 'anxiety', 'relief'): 'tension release',
        ('happiness', 'frustration', 'hope'): 'overcoming adversity',
        ('calm', 'surprise', 'amusement'): 'serene discovery',
        ('sadness', 'anger', 'resolution'): 'grief to empowerment',
        ('enthusiasm', 'disappointment', 'determination'): 'setback to comeback',
        # additional common arcs
        ('anger', 'negotiation', 'resolution'): 'conflict resolution',
        ('boredom', 'curiosity', 'engagement'): 'awakening interest',
        ('despair', 'hope', 'action'): 'from despair to action',
    }
    seq_lower = tuple(label.lower() for label in sequence)
    return patterns.get(seq_lower, 'custom')


def run(context):
    cfg = context['config']['arc']
    out_base = context['output_dir']
    speaker_ids = context['speaker_ids']

    all_labels = []
    all_confs = []
    all_times = []

    # Gather every slice's label, confidence, and timestamp across speakers
    for spk in speaker_ids:
        spk_dir = os.path.join(out_base, 'emotion_tags', spk)
        tier2_path = os.path.join(spk_dir, 'tier2_tags.json')
        transcript_path = os.path.join(spk_dir, 'transcript.json')

        try:
            with open(tier2_path, 'r') as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                tier2 = json.load(f)
                portalocker.unlock(f)
        except FileNotFoundError:
            tier2 = []

        try:
            with open(transcript_path, 'r') as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                transcript = json.load(f)
                portalocker.unlock(f)
        except FileNotFoundError:
            transcript = {'slices': []}

        slices = transcript.get('slices', [])
        for idx, tag in enumerate(tier2):
            all_labels.append(tag.get('label', 'neutral'))
            all_confs.append(tag.get('confidence', 0.0))
            if idx < len(slices):
                all_times.append((slices[idx]['start'], slices[idx]['end']))
            else:
                all_times.append((0.0, 0.0))

    # If no data, output neutral arc
    if not all_confs:
        classification = {
            'pivots': [],
            'arc_segments': [{
                'segment': 'neutral',
                'start': all_times[0][0] if all_times else 0.0,
                'end': all_times[-1][1] if all_times else 0.0
            }],
            'named_arc': 'neutral'
        }
        out_path = os.path.join(out_base, 'arc_classification.json')
        with open(out_path, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            json.dump(classification, f, indent=2)
            portalocker.unlock(f)
        return {'arc_classification': out_path}

    # Compute audio duration
    duration = max(end for start, end in all_times)

    # Determine dynamic number of clusters based on duration (one cluster per 5min, capped at 3)
    k = max(1, min(3, math.ceil(duration / 300)))
    # Ensure k <= number of slices
    if k > len(all_confs):
        k = 1

    # Pivot detection via KMeans
    conf_array = np.array(all_confs).reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(conf_array)
    labels = kmeans.labels_

    # Find indices where cluster label changes
    changes = np.where(np.diff(labels) != 0)[0] + 1
    pivots = [all_times[i][0] for i in changes]

    # Build per-segment arcs
    arc_segs = []
    start_idx = 0
    for end_idx in list(changes) + [len(all_labels)]:
        seg_labels = all_labels[start_idx:end_idx]
        seg_times = all_times[start_idx:end_idx]
        dominant = Counter(seg_labels).most_common(1)[0][0] if seg_labels else 'neutral'
        seg_start = seg_times[0][0] if seg_times else 0.0
        seg_end = seg_times[-1][1] if seg_times else 0.0
        arc_segs.append({'segment': dominant, 'start': seg_start, 'end': seg_end})
        start_idx = end_idx

    # Name the arc
    seq = [seg['segment'] for seg in arc_segs]
    named_arc = infer_named_arc(seq)

    classification = {
        'pivots': pivots,
        'arc_segments': arc_segs,
        'named_arc': named_arc
    }

    # Write to JSON
    out_path = os.path.join(out_base, 'arc_classification.json')
    with open(out_path, 'w') as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(classification, f, indent=2)
        portalocker.unlock(f)

    return {'arc_classification': out_path}

