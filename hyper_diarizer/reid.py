# reid.py

"""
Speaker re-identification module.
Uses OverlapClassifier from overlap.py for overlap detection.
"""

from overlap import OverlapClassifier, detect_overlaps
from cluster import ReIDMemory
import numpy as np
import logging

class ReIDSystem:
    """
    Wraps clustering-based re-identification memory and overlap detection.
    """

    def __init__(self, thresh: float = 0.6, config_path: str = "HyperDiarizer config.yaml"):
        # memory-based speaker re-ID
        self.memory = ReIDMemory(config_path=config_path, thresh=thresh)
        # overlap detector
        self.overlap_clf = OverlapClassifier()

    def re_id(self, embs: np.ndarray, labels: np.ndarray, audio=None, slices=None):
        """
        1) Update labels via memory-based matching
        2) Detect overlaps if audio & slices provided
        Returns: (new_labels, overlaps, certainties)
        """
        # 1) memory-based re-identification
        try:
            new_labels, certainties = self.memory.update(labels.tolist(), embs)
        except Exception as e:
            logging.warning(f"ReID memory update failed: {e}")
            new_labels, certainties = labels, np.ones_like(labels, dtype=float)

        # 2) optional overlap detection
        overlaps = []
        if audio is not None and slices is not None:
            try:
                overlaps = detect_overlaps(audio, slices, new_labels, embs)
            except Exception as e:
                logging.warning(f"Overlap detection in re-ID failed: {e}")

        return np.array(new_labels), overlaps, np.array(certainties)


