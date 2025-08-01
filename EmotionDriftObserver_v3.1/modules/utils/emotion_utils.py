# emotion_utils.py
# ─── Emotion-rule definitions ─────────────────────────────────────────────────
# Each rule returns True/False and has an initial confidence score.

emotion_rules = {
    "Anger": (lambda x: x["neg"] > 0.7 and x["pitch_mean"] > 180, 0.5),
    "Anxiety": (lambda x: x["neg"] > 0.6 and x["pitch_std"] > 40 and (
        "worried" in x.get("keywords", []) or "nervous" in x.get("keywords", [])
    ), 0.5),
    "Contempt": (lambda x: x["neg"] > 0.5 and x["energy_mean"] < -0.5 and (
        "disdain" in x.get("keywords", []) or "scorn" in x.get("keywords", [])
    ), 0.5),
    "Despair": (lambda x: x["neg"] > 0.8 and x["pitch_mean"] < 100 and (
        "hopeless" in x.get("keywords", []) or "desperate" in x.get("keywords", [])
    ), 0.5),
    "Disgust": (lambda x: x["neg"] > 0.7 and x["energy_std"] > 30 and (
        "revolting" in x.get("keywords", []) or "gross" in x.get("keywords", [])
    ), 0.5),
    "Fear": (lambda x: x["neg"] > 0.75 and x["pitch_std"] > 50 and (
        "scared" in x.get("keywords", []) or "terrified" in x.get("keywords", [])
    ), 0.5),
    "Frustration": (lambda x: x["neg"] > 0.65 and x.get("speech_rate", 0) > 150 and (
        "annoyed" in x.get("keywords", []) or "frustrated" in x.get("keywords", [])
    ), 0.5),
    "Guilt": (lambda x: x["neg"] > 0.6 and x.get("pause_ratio", 0) > 0.3 and (
        "sorry" in x.get("keywords", []) or "regret" in x.get("keywords", [])
    ), 0.5),
    "Irritation": (lambda x: x["neg"] > 0.55 and x.get("pitch_var", 0) > 25 and (
        "irritated" in x.get("keywords", []) or "bothered" in x.get("keywords", [])
    ), 0.5),
    "Jealousy": (lambda x: x["neg"] > 0.6 and x["energy_mean"] > 0.5 and (
        "envy" in x.get("keywords", []) or "jealous" in x.get("keywords", [])
    ), 0.5),
    "Loneliness": (lambda x: x["neg"] > 0.7 and x.get("speech_rate", 0) < 100 and (
        "alone" in x.get("keywords", []) or "isolated" in x.get("keywords", [])
    ), 0.5),
    "Negative Surprise": (lambda x: x["neg"] > 0.5 and x["pitch_std"] > 60 and (
        "shock" in x.get("keywords", []) or "unexpected bad" in x.get("keywords", [])
    ), 0.5),
    "Sadness": (lambda x: x["neg"] > 0.8 and x["pitch_mean"] < 120, 0.5),
    "Boredom": (lambda x: x["neu"] > 0.8 and x["energy_mean"] < -1.0 and (
        "bored" in x.get("keywords", []) or "uninterested" in x.get("keywords", [])
    ), 0.5),
    "Calm": (lambda x: x["neu"] > 0.7 and x["pitch_std"] < 20 and (
        "peaceful" in x.get("keywords", []) or "relaxed" in x.get("keywords", [])
    ), 0.5),
    "Concentration": (lambda x: x["neu"] > 0.6 and x.get("speech_rate", 0) > 120 and (
        "focused" in x.get("keywords", []) or "attentive" in x.get("keywords", [])
    ), 0.5),
    "Flat narration": (lambda x: x["neu"] > 0.9 and x.get("pitch_var", 0) < 10 and (
        "monotone" in x.get("keywords", []) or "flat" in x.get("keywords", [])
    ), 0.5),
    "Hesitant": (lambda x: x["neu"] > 0.7 and x.get("pause_ratio", 0) > 0.4 and (
        "unsure" in x.get("keywords", []) or "hesitant" in x.get("keywords", [])
    ), 0.5),
    "Matter-of-fact Informational tone": (lambda x: x["neu"] > 0.8 and x["energy_std"] < 20 and (
        "factual" in x.get("keywords", []) or "informative" in x.get("keywords", [])
    ), 0.5),
    "Neutral": (lambda x: x["neu"] > 0.7 and x["pitch_std"] < 20, 0.5),
    "Tired": (lambda x: x["neu"] > 0.6 and x["energy_mean"] < -1.5 and (
        "exhausted" in x.get("keywords", []) or "weary" in x.get("keywords", [])
    ), 0.5),
    "Amusement": (lambda x: x["pos"] > 0.7 and x["energy_std"] > 40 and (
        "funny" in x.get("keywords", []) or "amused" in x.get("keywords", [])
    ), 0.5),
    "Enthusiasm": (lambda x: x["pos"] > 0.8 and x["pitch_mean"] > 160 and (
        "excited" in x.get("keywords", []) or "enthusiastic" in x.get("keywords", [])
    ), 0.5),
    "Gratitude": (lambda x: x["pos"] > 0.75 and x.get("speech_rate", 0) < 130 and (
        "thankful" in x.get("keywords", []) or "grateful" in x.get("keywords", [])
    ), 0.5),
    "Happiness": (lambda x: x["pos"] > 0.8 and x["energy_mean"] > 1.0 and (
        "joyful" in x.get("keywords", []) or "happy" in x.get("keywords", [])
    ), 0.5),
    "Hope": (lambda x: x["pos"] > 0.6 and x["pitch_std"] > 30 and (
        "hopeful" in x.get("keywords", []) or "optimistic" in x.get("keywords", [])
    ), 0.5),
    "Inspiration": (lambda x: x["pos"] > 0.85 and x["energy_mean"] > 0.8 and (
        "inspired" in x.get("keywords", []) or "motivated" in x.get("keywords", [])
    ), 0.5),
    "Love": (lambda x: x["pos"] > 0.9 and x.get("pause_ratio", 0) < 0.2 and (
        "love" in x.get("keywords", []) or "affection" in x.get("keywords", [])
    ), 0.5),
    "Pleasant": (lambda x: x["pos"] > 0.7 and x["pitch_mean"] > 140 and (
        "pleasant" in x.get("keywords", []) or "nice" in x.get("keywords", [])
    ), 0.5),
    "Relief": (lambda x: x["pos"] > 0.65 and x["energy_std"] < 25 and (
        "relieved" in x.get("keywords", []) or "eased" in x.get("keywords", [])
    ), 0.5),
    "Surprise": (lambda x: x["pos"] > 0.7 and x["pitch_std"] > 50, 0.5),
}

# ─── Emotion-group mapping ───────────────────────────────────────────────────
# Used downstream to route auto-accepted JSON into Tier1 folders.
GROUP_MAP = {
    "Anger": "Negative", "Anxiety": "Negative", "Contempt": "Negative",
    "Despair": "Negative", "Disgust": "Negative", "Fear": "Negative",
    "Frustration": "Negative", "Guilt": "Negative", "Irritation": "Negative",
    "Jealousy": "Negative", "Loneliness": "Negative", "Negative Surprise": "Negative",
    "Sadness": "Negative", "Boredom": "Neutral", "Calm": "Neutral",
    "Concentration": "Neutral", "Flat narration": "Neutral", "Hesitant": "Neutral",
    "Matter-of-fact Informational tone": "Neutral", "Neutral": "Neutral",
    "Tired": "Neutral", "Amusement": "Positive", "Enthusiasm": "Positive",
    "Gratitude": "Positive", "Happiness": "Positive", "Hope": "Positive",
    "Inspiration": "Positive", "Love": "Positive", "Pleasant": "Positive",
    "Relief": "Positive", "Surprise": "Positive",
}

# ─── Tier thresholds ─────────────────────────────────────────────────────────
# These mirror T1_AUTO, T1_MIN, T2_AUTO, T2_MIN, SENTIMENT_STD_THRESHOLD
T1_AUTO = 0.90  # Tier-1 auto-accept
T1_MIN = 0.80   # Tier-1 minimum pass

T2_AUTO = 0.90  # Tier-2 auto-accept
T2_MIN = 0.65   # Tier-2 review threshold

# If sentiment_std above this, even auto-accepted segments get flagged for review
SENTIMENT_STD_THRESHOLD = 0.30
