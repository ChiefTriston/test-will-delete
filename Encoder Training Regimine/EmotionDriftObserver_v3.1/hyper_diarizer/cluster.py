"""
Module for advanced speaker clustering and re-identification with time-aware similarity, temporal graph clustering, and adaptive anomaly detection.
Supports pluggable components, FAISS indexing, callbacks, online learning, deep metric refinement, learned similarity, GNN clustering, temporal context, adaptive thresholding, OOD detection, end-to-end fine-tuning, massive-scale indexing, voiceprint fusion, and continuous evaluation.
"""

import numpy as np
import uuid
import logging
from typing import List, Tuple, Dict, Optional, Union, Callable
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from bidict import bidict
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from collections import deque
import faiss
import pickle
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import yaml  # For hyperparameter management
from random import choice

# Defaults for parameters (overridden at runtime via DiarizerController or config)
EMB_DIM: int = 448               # embedding dimension of fused embeddings
VOICEPRINT_THRESH: float = 0.6   # cosine similarity threshold for re-id
MEMORY_SIZE: int = 10            # max memory entries per speaker
TEMPORAL_DECAY: float = 2.0      # seconds for exponential temporal decay
ANOMALY_CONTAMINATION: float = 0.1 # contamination ratio for IsolationForest
CLUSTER_MIN_SIM: float = 0.5     # minimum similarity for edge in graph clustering
MIN_ANOMALY_SAMPLES: int = 3     # minimum samples for anomaly detection
MOMENTUM: float = 0.99           # Momentum for prototypical updates
LEARNING_RATE: float = 1e-3      # Learning rate for trainable components
HNSW_M: int = 32                 # HNSW parameter
IVFPQ_NLIST: int = 100           # IVFPQ nlist
IVFPQ_NPROBE: int = 10           # IVFPQ nprobe
GNN_HIDDEN_DIM: int = 128        # GNN hidden dim
GNN_HEADS: int = 4               # GNN attention heads
CONTRASTIVE_BATCH_SIZE: int = 32 # Batch size for contrastive training

class ReIDCallback(ABC):
    """Base class for re-identification callbacks."""
    def on_anomaly_filtered(self, orig_label: int, num_before: int, num_after: int) -> None:
        pass

    def on_avg_computed(self, orig_label: int, avg_emb: np.ndarray) -> None:
        pass

    def on_match(self, orig_label: int, speaker_id: str, score: float, avg_emb: np.ndarray, instance: 'ReIDMemory') -> None:
        pass

    def on_new_speaker(self, orig_label: int, new_id: str, avg_emb: np.ndarray, instance: 'ReIDMemory') -> None:
        pass

    def on_memory_updated(self, speaker_id: str, updated_size: int) -> None:
        pass

    def on_smoothing(self, raw_embs: np.ndarray, smoothed_embs: np.ndarray) -> None:
        pass

    def on_snapshot_saved(self, timestamp: float, size: int) -> None:
        pass

class WandBCallback(ReIDCallback):
    """Callback for logging metrics to Weights & Biases."""
    def __init__(self):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            logging.warning("WandB not installed; callback will be silent")
            self.wandb = None

    def on_match(self, orig_label: int, speaker_id: str, score: float, avg_emb: np.ndarray, instance: 'ReIDMemory') -> None:
        if self.wandb:
            self.wandb.log({"reid/match_score": score})

    def on_new_speaker(self, orig_label: int, new_id: str, avg_emb: np.ndarray, instance: 'ReIDMemory') -> None:
        if self.wandb:
            self.wandb.log({"reid/new_speakers": 1})

class PairCollector(ReIDCallback):
    """Callback to collect positive/negative pairs for contrastive learning."""
    def __init__(self):
        self.positives: List[Tuple[np.ndarray, np.ndarray]] = []  # (anchor, positive)
        self.negatives: List[Tuple[np.ndarray, np.ndarray]] = []  # (anchor, negative)

    def on_match(self, orig_label: int, speaker_id: str, score: float, avg_emb: np.ndarray, instance: 'ReIDMemory') -> None:
        prototype = instance.prototypes.get(speaker_id, avg_emb)
        self.positives.append((prototype, avg_emb))

    def on_new_speaker(self, orig_label: int, new_id: str, avg_emb: np.ndarray, instance: 'ReIDMemory') -> None:
        if instance.prototypes:
            random_proto = choice(list(instance.prototypes.values()))
            self.negatives.append((random_proto, avg_emb))

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module for transformer inputs."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        idx = positions.long().clamp(0, self.pe.size(1) - 1).unsqueeze(0)
        return x + self.pe[:, idx].squeeze(1)

class Clusterer(ABC):
    """Base class for clustering algorithms."""
    @abstractmethod
    def cluster(self, sim_matrix: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Cluster similarity matrix into speaker labels using features."""
        pass

class GreedyModularityClusterer(Clusterer):
    """Graph-based clustering using greedy modularity communities."""
    def cluster(self, sim_matrix: np.ndarray, features: np.ndarray) -> np.ndarray:
        try:
            if not isinstance(sim_matrix, np.ndarray) or sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
                raise ValueError(f"Expected square sim matrix, got {sim_matrix.shape if isinstance(sim_matrix, np.ndarray) else type(sim_matrix)}")

            n = sim_matrix.shape[0]
            rows, cols = np.triu_indices(n, k=1)
            mask = sim_matrix[rows, cols] >= CLUSTER_MIN_SIM
            edges = [(i, j, sim_matrix[i, j]) for i, j in zip(rows[mask], cols[mask])]
            
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_weighted_edges_from(edges)
            
            communities = greedy_modularity_communities(G, weight='weight')
            
            labels = np.full(n, -1, dtype=int)
            for label, community in enumerate(communities):
                for node in community:
                    labels[node] = label
            logging.debug(f"Clustered into {len(communities)} communities")
            
            return labels
        
        except ValueError as e:
            logging.error(f"Input validation error in cluster: {e}")
            raise
        except RuntimeError as e:
            logging.error(f"Runtime error in cluster: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in cluster: {e}")
            raise

class GATLayer(nn.Module):
    """Simple Graph Attention Layer."""
    def __init__(self, in_dim: int, out_dim: int, heads: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * heads)
        self.attn = nn.Linear(out_dim * heads * 2, heads)
        self.heads = heads
        self.out_dim = out_dim

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, n, d = features.shape
        h = self.fc(features).view(b, n, self.heads, self.out_dim)
        attn_src = h.unsqueeze(1).repeat(1, n, 1, 1, 1)
        attn_dst = h.unsqueeze(2).repeat(1, 1, n, 1, 1)
        attn = torch.cat([attn_src, attn_dst], dim=-1).view(b, n, n, -1)
        attn = self.attn(attn).mean(dim=-1, keepdim=True)
        attn = torch.softmax(attn + (1 - adj.unsqueeze(-1)) * -1e9, dim=2)
        out = torch.einsum('bijk, bjkl -> bijl', attn, h).mean(dim=2)
        return out

class GNNClusterer(Clusterer):
    """GNN-based clustering using GAT and K-means."""
    def __init__(self, hidden_dim: int = GNN_HIDDEN_DIM, heads: int = GNN_HEADS, k: Optional[int] = None):
        self.gat1 = GATLayer(EMB_DIM, hidden_dim, heads)
        self.gat2 = GATLayer(hidden_dim, EMB_DIM, heads)
        self.optimizer = optim.Adam(list(self.gat1.parameters()) + list(self.gat2.parameters()), lr=LEARNING_RATE)
        self.k = k

    def cluster(self, sim_matrix: np.ndarray, features: np.ndarray) -> np.ndarray:
        adj = torch.from_numpy(sim_matrix > CLUSTER_MIN_SIM).float().unsqueeze(0)
        features_t = torch.from_numpy(features).float().unsqueeze(0)
        features_t = self.gat1(features_t, adj)
        features_t = self.gat2(features_t, adj).squeeze(0).detach().numpy()
        if self.k is None:
            self.k = int(np.sqrt(features_t.shape[0]))
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.k)
        labels = kmeans.fit_predict(features_t)
        return labels

    def train(self, triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self.optimizer.zero_grad()
        loss = 0
        for anchor, pos, neg in triplets:
            anchor = torch.from_numpy(anchor).float().unsqueeze(0).unsqueeze(0)
            pos = torch.from_numpy(pos).float().unsqueeze(0).unsqueeze(0)
            neg = torch.from_numpy(neg).float().unsqueeze(0).unsqueeze(0)
            adj = torch.eye(1)
            anchor_out = self.gat2(self.gat1(anchor, adj))
            pos_out = self.gat2(self.gat1(pos, adj))
            neg_out = self.gat2(self.gat1(neg, adj))
            dist_pos = (anchor_out - pos_out).pow(2).sum()
            dist_neg = (anchor_out - neg_out).pow(2).sum()
            loss += torch.max(dist_pos - dist_neg + 1.0, torch.zeros(1))
        loss.backward()
        self.optimizer.step()

class ContrastiveHead(nn.Module):
    """Projection head for contrastive learning."""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(EMB_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    """NT-Xent contrastive loss."""
    N = z_i.shape[0]
    z = torch.cat((z_i, z_j), dim=0)
    sim = torch.mm(z, z.T) / temp
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    positive = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * N, 1)
    mask = ~torch.eye(2 * N, device=z.device, dtype=bool)
    negative = sim[mask].reshape(2 * N, -1)
    labels = torch.zeros(2 * N).long().to(z.device)
    logits = torch.cat((positive, negative), dim=1)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss / (2 * N)

class TemporalContext(nn.Module):
    """Simple TCN for temporal context."""
    def __init__(self):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x.transpose(1, 2)).transpose(1, 2)

class GatingNetwork(nn.Module):
    """Learnable gating for voiceprint fusion."""
    def __init__(self, num_sources: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(EMB_DIM * num_sources, num_sources),
            nn.Softmax(dim=-1)
        )

    def forward(self, embs: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.cat(embs, dim=-1)
        weights = self.gate(stacked)
        fused = sum(w.unsqueeze(-1) * e for w, e in zip(weights, embs))
        return fused

class ReIDMemory(nn.Module):
    def __init__(
        self,
        config_path: str = "HyperDiarizer config.yaml",
        thresh: float = VOICEPRINT_THRESH,
        memory_size: int = MEMORY_SIZE,
        nhead: int = 4,
        num_layers: int = 2,
        anomaly_contamination: float = ANOMALY_CONTAMINATION,
        min_anomaly_samples: int = MIN_ANOMALY_SAMPLES,
        device: str = 'cpu',
        random_state: Optional[int] = 42,
        use_faiss: bool = True,
        faiss_index_type: str = 'Flat'
    ):
        super().__init__()
        self._load_config(config_path)
        self.thresh = self.config.get('voiceprint_thresh', thresh)
        self.memory_size = self.config.get('memory_size', memory_size)
        self.anomaly_contamination = self.config.get('anomaly_contamination', anomaly_contamination)
        self.min_anomaly_samples = self.config.get('min_anomaly_samples', min_anomaly_samples)
        self.random_state = random_state
        self.use_faiss = use_faiss
        self.faiss_index_type = faiss_index_type
        self.memory: Dict[str, deque] = {}
        self.label_map = bidict()
        self.prototypes: Dict[str, np.ndarray] = {}
        self.callbacks: List[ReIDCallback] = []
        self.similarity_fn: Callable[[np.ndarray, np.ndarray], float] = self._learned_similarity
        self.anomaly_detector_factory: Callable[[], IsolationForest] = self._mahalanobis_anomaly
        self.refine_fn: Callable[[np.ndarray], np.ndarray] = self._contrastive_refine
        self.on_threshold_suggest: Optional[Callable[[List[float]], float]] = self._gmm_threshold
        self.clusterer: Clusterer = GNNClusterer()
        self.contrastive_head = ContrastiveHead().to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.get('learning_rate', LEARNING_RATE))
        self.W = nn.Parameter(torch.randn(EMB_DIM, EMB_DIM)).to(device)  # Learned similarity kernel
        self.temporal_context = TemporalContext().to(device)
        self.gating = GatingNetwork(num_sources=2).to(device)  # Assume 2 sources
        transformer_layer = nn.TransformerEncoderLayer(d_model=EMB_DIM, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers).to(device)
        self.pos_enc = PositionalEncoding(EMB_DIM).to(device)
        self.device = torch.device(device)
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._faiss_index: Optional[faiss.Index] = None
        self._index_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)  # Single worker for reindexing
        self._reindexing = False  # Flag to prevent multiple reindex tasks

    def close(self):
        self._executor.shutdown(wait=False)

    def __del__(self):
        self.close()

    def _load_config(self, path: str) -> None:
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f).get('hyperdiarizer', {})

    def _trigger(self, event: str, **kwargs) -> None:
        """Trigger callbacks for a given event."""
        for cb in self.callbacks:
            fn = getattr(cb, f"on_{event}", None)
            if fn:
                kwargs['instance'] = self
                fn(**kwargs)

    def _mahalanobis_anomaly(self) -> IsolationForest:
        # Placeholder for Mahalanobis; use IsolationForest for now
        return IsolationForest(contamination=self.anomaly_contamination, random_state=self.random_state)

    def _update_faiss_index(self, M: int = HNSW_M, nlist: int = IVFPQ_NLIST, nprobe: int = IVFPQ_NPROBE) -> None:
        with self._index_lock:
            if not self.memory:
                self._faiss_index = None
                self._reindexing = False
                return
            mem_avgs = [self.prototypes.get(mid, np.mean(self.memory[mid], axis=0)) for mid in self.memory.keys()]
            mem_matrix = np.stack(mem_avgs).astype(np.float32)
            faiss.normalize_L2(mem_matrix)
            
            if self.faiss_index_type == 'HNSW':
                index = faiss.IndexHNSWFlat(EMB_DIM, M)
                index.hnsw.efConstruction = 200  # Example param
            elif self.faiss_index_type == 'IVFPQ':
                quantizer = faiss.IndexFlatL2(EMB_DIM)
                index = faiss.IndexIVFPQ(quantizer, EMB_DIM, nlist, 8, 8)
                index.nprobe = nprobe
                index.train(mem_matrix)
            else:
                index = faiss.IndexFlatIP(EMB_DIM)
            
            index.add(mem_matrix)
            self._faiss_index = index
            if 'cuda' in self.device.type:
                res = faiss.StandardGpuResources()
                cfg = faiss.GpuIndexFlatConfig() if self.faiss_index_type == 'Flat' else faiss.GpuIndexIVFPQConfig()
                cfg.useFloat16 = True
                self._faiss_index = faiss.index_cpu_to_gpu(res, 0, self._faiss_index, cfg)
            self._reindexing = False

    def _schedule_reindex(self):
        if not self._reindexing:
            self._reindexing = True
            self._executor.submit(self._update_faiss_index)

    def re_id(
        self,
        embs: np.ndarray,
        labels: Union[np.ndarray, List[int]],
        slices: Optional[List[Tuple[float, float, float]]] = None,
        use_transformer: bool = True
    ) -> Tuple[np.ndarray, Dict[int, str], np.ndarray, Optional[np.ndarray]]:
        if embs.shape[1] != EMB_DIM:
            raise ValueError(f"Expected embs dim {EMB_DIM}, got {embs.shape[1]}")
        if len(labels) != embs.shape[0]:
            raise ValueError("Labels and embs mismatch")
        if slices and len(slices) != embs.shape[0]:
            raise ValueError("Slices and embs mismatch")

        labels = np.asarray(labels, dtype=int)
        embs_t = torch.from_numpy(embs).float().to(self.device).unsqueeze(0)
        if embs_t.device != self.device:
            raise RuntimeError("Device mismatch")
        embs = self.temporal_context(embs_t).squeeze(0).cpu().numpy()

        # Fusion example with two sources (duplicate for demo; replace with actual embs1, embs2 from embedding.py)
        embs1 = embs_t
        embs2 = embs_t + torch.randn_like(embs_t) * 0.1  # Simulated second source
        embs = self.gating([embs1, embs2]).squeeze(0).cpu().numpy()

        new_labels = labels.copy()
        certainties = np.zeros(len(labels), dtype=float)
        smoothed_embs = None
        best_scores = []

        self._schedule_reindex()

        for orig_label in np.unique(labels):
            mask = (labels == orig_label)
            slice_embs = embs[mask]
            num_before = slice_embs.shape[0]
            slice_embs = self._filter_anomalies(slice_embs)
            self._trigger("anomaly_filtered", orig_label=orig_label, num_before=num_before, num_after=slice_embs.shape[0])
            avg_emb = self._compute_avg_emb(slice_embs, slices, mask)
            self._trigger("avg_computed", orig_label=orig_label, avg_emb=avg_emb)
            avg_emb = self.refine_fn(avg_emb)
            best_id, best_score = self._find_best_match(avg_emb)
            best_scores.append(best_score)
            if best_id and best_score >= self.thresh:
                new_uuid = self.label_map.inverse[best_id]
                new_labels[mask] = new_uuid
                self._update_memory(best_id, slice_embs)
                certainties[mask] = best_score
                self._trigger("match", orig_label=orig_label, speaker_id=new_uuid, score=best_score, avg_emb=avg_emb)
            else:
                new_uuid = str(uuid.uuid4())
                self.memory[new_uuid] = deque(slice_embs.tolist(), maxlen=self.memory_size)
                self.prototypes[new_uuid] = avg_emb
                self.label_map[orig_label] = new_uuid
                certainties[mask] = 1.0
                self._trigger("new_speaker", orig_label=orig_label, new_id=new_uuid, avg_emb=avg_emb)

        if self.on_threshold_suggest and best_scores:
            self.thresh = self.on_threshold_suggest(best_scores)
            logging.debug(f"Updated threshold to {self.thresh:.3f}")

        if use_transformer:
            embs_t = torch.from_numpy(embs).float().to(self.device).unsqueeze(0)
            positions = torch.tensor([s for s, _, _ in slices] if slices else np.arange(len(embs)), device=self.device).float()
            embs_t = self.pos_enc(embs_t, positions)
            smoothed_embs = self.transformer(embs_t).squeeze(0).cpu().numpy()
            self._trigger("smoothing", raw_embs=embs, smoothed_embs=smoothed_embs)

        # Batch train contrastive head
        for cb in self.callbacks:
            if isinstance(cb, PairCollector) and (len(cb.positives) + len(cb.negatives)) > 0:
                # Train on mini-batches
                pairs = cb.positives + cb.negatives  # Combine for NT-Xent
                for i in range(0, len(pairs), CONTRASTIVE_BATCH_SIZE):
                    batch = pairs[i:i+CONTRASTIVE_BATCH_SIZE]
                    anchor_batch = torch.stack([torch.from_numpy(p[0]) for p in batch]).float().to(self.device)
                    pos_batch = torch.stack([torch.from_numpy(p[1]) for p in batch]).float().to(self.device)
                    z_i = self.contrastive_head(anchor_batch)
                    z_j = self.contrastive_head(pos_batch)
                    loss = nt_xent_loss(z_i, z_j)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                cb.positives.clear()
                cb.negatives.clear()

        return new_labels, dict(self.label_map), certainties, smoothed_embs

    def _compute_avg_emb(
        self,
        slice_embs: np.ndarray,
        slices: Optional[List[Tuple[float, float, float]]],
        mask: np.ndarray
    ) -> np.ndarray:
        if slices is not None:
            probs = np.array([p for s, e, p in slices])[mask]
            probs /= probs.sum() + 1e-6
            return np.average(slice_embs, axis=0, weights=probs)
        return slice_embs.mean(axis=0) if slice_embs.size else np.zeros(EMB_DIM)

    def _find_best_match(self, avg_emb: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.memory:
            return None, -1.0
        mem_ids = list(self.memory.keys())
        if self.use_faiss and self._faiss_index is not None:
            query = avg_emb.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query)
            scores, indices = self._faiss_index.search(query, 1)
            best_idx = indices[0][0]
            best_score = scores[0][0]
            return mem_ids[best_idx], best_score
        else:
            best_score = -1.0
            best_id = None
            for mid in mem_ids:
                mem_avg = self.prototypes.get(mid, self._memory_cache.get(mid, np.mean(self.memory[mid], axis=0)))
                score = self.similarity_fn(avg_emb, mem_avg)
                if score > best_score:
                    best_score = score
                    best_id = mid
            return best_id, best_score

    def _update_memory(self, mem_id: str, slice_embs: np.ndarray) -> None:
        self.memory[mem_id].extend(slice_embs.tolist())
        avg = np.mean(self.memory[mem_id], axis=0)
        if mem_id in self.prototypes:
            self.prototypes[mem_id] = MOMENTUM * self.prototypes[mem_id] + (1 - MOMENTUM) * avg
        else:
            self.prototypes[mem_id] = avg
        self._memory_cache[mem_id] = self.prototypes[mem_id]
        self._trigger("memory_updated", speaker_id=mem_id, updated_size=len(self.memory[mem_id]))
        self._schedule_reindex()

    def snapshot(self) -> bytes:
        state = {
            'memory': {k: list(v) for k, v in self.memory.items()},
            'label_map': dict(self.label_map),
            'prototypes': self.prototypes,
            'memory_cache': self._memory_cache
        }
        data = pickle.dumps(state)
        self._trigger("snapshot_saved", timestamp=time.time(), size=len(self.memory))
        return data

    def load_snapshot(self, data: bytes) -> None:
        try:
            state = pickle.loads(data)
            self.memory = {k: deque(v, maxlen=self.memory_size) for k, v in state['memory'].items()}
            self.label_map = bidict(state['label_map'])
            self.prototypes = state['prototypes']
            self._memory_cache = state['memory_cache']
            self._schedule_reindex()
        except Exception as e:
            logging.error(f"Error loading snapshot: {e}")
            raise

    def smooth_embeddings(
        self,
        embs: np.ndarray,
        slices: Optional[List[Tuple[float, float, float]]] = None
    ) -> np.ndarray:
        if embs.shape[1] != EMB_DIM:
            raise ValueError(f"Expected embs dim {EMB_DIM}, got {embs.shape[1]}")
        embs_t = torch.from_numpy(embs).float().to(self.device).unsqueeze(0)
        positions = torch.tensor([s for s, _, _ in slices] if slices else np.arange(len(embs)), device=self.device).float()
        embs_t = self.pos_enc(embs_t, positions)
        smoothed = self.transformer(embs_t).squeeze(0).cpu().numpy()
        self._trigger("smoothing", raw_embs=embs, smoothed_embs=smoothed)
        return smoothed

# Alias for legacy import
ReidMemory = ReIDMemory

def time_aware_sim(
    embs: np.ndarray,
    slices: List[Tuple[float, float, float]],
    temporal_weight: float = 0.5,
    decay: float = TEMPORAL_DECAY
) -> np.ndarray:
    try:
        # Input validation
        if not isinstance(embs, np.ndarray) or embs.ndim != 2 or embs.shape[1] != EMB_DIM:
            raise ValueError(f"Expected embs shape (n, {EMB_DIM}), got {embs.shape if isinstance(embs, np.ndarray) else type(embs)}")
        if len(slices) != embs.shape[0]:
            raise ValueError(f"Slices length {len(slices)} does not match embs rows {embs.shape[0]}")

        # Cosine similarity
        sim = cosine_similarity(embs)
        
        # Vectorized temporal proximity
        n_slices = len(slices)
        mids = np.array([(s + e) / 2 for s, e, _ in slices])
        probs = np.array([p for _, _, p in slices])
        gaps = np.abs(mids[:, np.newaxis] - mids[np.newaxis, :])
        mod = (probs[:, np.newaxis] + probs[np.newaxis, :]) / 2
        temporal_sim = mod * np.exp(-gaps / decay)
        np.fill_diagonal(temporal_sim, 1.0)
        
        # Combine and normalize
        combined_sim = (1 - temporal_weight) * sim + temporal_weight * temporal_sim
        min_val, max_val = combined_sim.min(), combined_sim.max()
        combined_sim = (combined_sim - min_val) / (max_val - min_val + 1e-6)
        
        return combined_sim
    
    except ValueError as e:
        logging.error(f"Input validation error in time_aware_sim: {e}")
        raise
    except RuntimeError as e:
        logging.error(f"Runtime error in time_aware_sim: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in time_aware_sim: {e}")
        raise

def temporal_cluster(
    sim: np.ndarray,
    slices: List[Tuple[float, float, float]],
    min_sim: float = CLUSTER_MIN_SIM,
    features: np.ndarray = None
) -> np.ndarray:
    if features is None:
        # Use zero features if not provided; in practice, pass embs
        features = np.zeros((sim.shape[0], EMB_DIM))
    return ReIDMemory().clusterer.cluster(sim, features)