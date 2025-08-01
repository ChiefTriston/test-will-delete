# dynamic_learning.py

import json
import os
import random
import portalocker
from collections import defaultdict
import numpy as np
import logging

from modules.utils.emotion_utils import emotion_rules
from sklearn.metrics.pairwise import cosine_similarity  # not used

VALIDATION_SET_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "validation_set.json")
LEARNED_CONF_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "learned_confidences.json")
REVIEW_QUEUE = os.path.join(os.path.dirname(__file__), "..", "..", "review_queue")
TIER1_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "output", "emotion_tags")

def load_validation_set():
    try:
        with open(VALIDATION_SET_FILE, "r") as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            data = json.load(f)
            portalocker.unlock(f)
            return data
    except FileNotFoundError:
        return []

def save_validation_set(vset):
    tmp = VALIDATION_SET_FILE + ".tmp"
    with open(tmp, "w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(vset, f, indent=2)
        portalocker.unlock(f)
    os.replace(tmp, VALIDATION_SET_FILE)

def load_learned_confidences():
    try:
        with open(LEARNED_CONF_FILE, "r") as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            data = json.load(f)
            portalocker.unlock(f)
            return data
    except FileNotFoundError:
        return {}

def save_learned_confidences(conf):
    tmp = LEARNED_CONF_FILE + ".tmp"
    with open(tmp, "w") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(conf, f, indent=2)
        portalocker.unlock(f)
    os.replace(tmp, LEARNED_CONF_FILE)

def load_tagged_data():
    """Scan Tier-1 tags across all jobs and speakers to aggregate accept/reject counts per emotion."""
    tally = defaultdict(lambda: {"accept": 0, "reject": 0})
    for root, dirs, files in os.walk(TIER1_ROOT):
        if "tier1_tags.json" in files:
            path = os.path.join(root, "tier1_tags.json")
            with open(path, "r") as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                tags = json.load(f)
                portalocker.unlock(f)
            for tag in tags:
                emo = tag.get("tag")
                status = tag.get("status")
                if not emo:
                    continue
                if status == "auto-accept":
                    tally[emo]["accept"] += 1
                else:
                    tally[emo]["reject"] += 1
    return tally

def update_validation_set():
    """Add a stratified 5% sample per emotion (cap 500) to the validation set atomically and return number of new samples."""
    validation = load_validation_set()
    emotion_samples = defaultdict(list)
    for root, dirs, files in os.walk(TIER1_ROOT):
        if "tier1_tags.json" in files and "transcript.json" in files:
            tier1_path = os.path.join(root, "tier1_tags.json")
            trans_path = os.path.join(root, "transcript.json")
            with open(tier1_path, "r") as f1, open(trans_path, "r") as f2:
                portalocker.lock(f1, portalocker.LOCK_SH)
                portalocker.lock(f2, portalocker.LOCK_SH)
                tags = json.load(f1)
                trans = json.load(f2)
                portalocker.unlock(f1)
                portalocker.unlock(f2)
            for tag, slice_data in zip(tags, trans.get("slices", [])):
                emo = tag.get("tag")
                if not emo:
                    continue
                emotion_samples[emo].append({
                    "text": slice_data.get("text", ""),
                    "true_emotion": emo,
                    "compound": tag.get("compound", 0.0)
                })
    new_samples = []
    for emo, samples in emotion_samples.items():
        num_sample = max(1, int(len(samples) * 0.05))
        sample_count = min(num_sample, len(samples))
        new_samples.extend(random.sample(samples, sample_count))
    validation.extend(new_samples)
    if len(validation) > 500:
        validation = random.sample(validation, 500)
    save_validation_set(validation)
    return len(new_samples)

def check_accuracy_drop(old_acc, new_acc):
    """Print alert if accuracy drops by more than 5%."""
    if new_acc < old_acc * 0.95:
        print(f"Accuracy drop alert: from {old_acc:.2f} to {new_acc:.2f}")

def compute_accuracy(rules, validation):
    correct = 0
    for val in validation:
        x = {"pos": max(0, val["compound"]), "neg": max(0, -val["compound"]), "neu": 1 - abs(val["compound"])}
        predicted = next((emo for emo, (fn, conf) in rules.items() if fn(x)), "neutral")
        if predicted == val["true_emotion"]:
            correct += 1
    return correct / len(validation) if validation else 1.0

def update_emotion_rules(rule_updates, validation):
    learned = load_learned_confidences()
    old_rules = emotion_rules.copy()
    alpha = 0.9
    for emo, updates in rule_updates.items():
        new_conf = alpha * learned.get(emo, 0.5) + (1 - alpha) * updates.get("conf_adjust", 0.0)
        learned[emo] = new_conf
        fn, _ = emotion_rules.get(emo, (None, 0.0))
        if fn:
            emotion_rules[emo] = (fn, new_conf)
    old_acc = compute_accuracy(old_rules, validation)
    new_acc = compute_accuracy(emotion_rules, validation)
    check_accuracy_drop(old_acc, new_acc)
    save_learned_confidences(learned)
