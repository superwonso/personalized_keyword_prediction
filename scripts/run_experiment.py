#!/usr/bin/env python3
"""Controlled scholar-holdout evaluation for topic-adoption prediction.

The script is designed to make the evaluation protocol auditable:

* The topic vocabulary and GKN use observation-window data only.
* Scholar-holdout and known-scholar pair-stratified splits are both supported.
* GNN and feature-engineered baselines consume the same saved pair manifest.
* Every model selects its decision threshold on the same validation partition.
* Test uncertainty is estimated by paired scholar-cluster bootstrap.

The implementation deliberately favors auditable artifacts over notebook state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms.community import greedy_modularity_communities, modularity
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import degree, to_undirected
from tqdm import tqdm

FEATURE_NAMES = [f"x{i}" for i in range(1, 17)]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs-window", type=int, choices=[1, 3, 5], required=True)
    parser.add_argument("--lkn-scale", type=int, default=3000)
    parser.add_argument("--lkn-obs-dir", required=True)
    parser.add_argument("--lkn-pred-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--excluded-topics-file",
        default=None,
        help="Optional text file containing one topic ID per line to exclude.",
    )
    parser.add_argument(
        "--topic-id-regex",
        default=None,
        help="Optional regular expression used to validate topic identifiers.",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument(
        "--split-unit", choices=["scholar", "pair"], default="scholar"
    )
    parser.add_argument(
        "--negative-sampling",
        choices=["random", "degree-matched"],
        default="random",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "candidate-attention"],
        default="mean",
    )
    parser.add_argument(
        "--fusion",
        choices=["product", "rich"],
        default="product",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--scholar-batch-size", type=int, default=16)
    parser.add_argument("--eval-scholar-batch-size", type=int, default=32)
    parser.add_argument("--min-scholar-edge-count", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-incomplete-cohort",
        action="store_true",
        help="Allow sampled scholars without usable positive pairs (audit runs only).",
    )
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--feature-source", default=None)
    parser.add_argument(
        "--partial-feature-source",
        default=None,
        help=(
            "Reuse scholar-topic feature rows from another protocol with the same "
            "window, and compute only candidates absent from that source."
        ),
    )
    parser.add_argument("--resume-model", action="store_true")
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    obs_dir = Path(args.lkn_obs_dir).resolve()
    pred_dir = Path(args.lkn_pred_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (args.repo_root / "outputs" / f"obs{args.obs_window}y_lkn{args.lkn_scale}").resolve()
    )
    if not obs_dir.is_dir():
        raise FileNotFoundError(obs_dir)
    if not pred_dir.is_dir():
        raise FileNotFoundError(pred_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return obs_dir, pred_dir, output_dir


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, message: str) -> None:
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(stamped, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(stamped + "\n")


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU execution was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_excluded_topics(path_value: str | None) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.hidden_dim % args.heads:
        raise ValueError("--hidden-dim must be divisible by --heads.")
    return {
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "heads": args.heads,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "fusion": args.fusion,
    }


def read_topic_edges(
    path: Path, excluded_topics: set[str]
) -> tuple[list[tuple[str, str]], set[str]]:
    edges: list[tuple[str, str]] = []
    topics: set[str] = set()
    if not path.exists():
        return edges, topics
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().replace("'", "").split()
            if len(parts) < 2:
                continue
            t1, t2 = parts[0], parts[1]
            if t1 in excluded_topics or t2 in excluded_topics:
                continue
            edges.append((t1, t2))
            topics.update((t1, t2))
    return edges, topics


def stable_pair_id(scholar_id: str, topic_id: str, label: int) -> str:
    payload = f"{scholar_id}\t{topic_id}\t{label}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def split_contributing_scholars(
    scholar_ids: list[str], seed: int
) -> tuple[set[str], set[str], set[str]]:
    ids = sorted(scholar_ids)
    random.Random(seed + 1009).shuffle(ids)
    n = len(ids)
    if n < 5:
        raise RuntimeError(f"At least five contributing scholars are required; found {n}.")
    n_test = max(1, int(round(n * 0.20)))
    n_val = max(1, int(round(n * 0.20)))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    train_ids = set(ids[n_test + n_val :])
    return train_ids, val_ids, test_ids


def split_pairs(
    scholars: dict[str, dict[str, Any]], seed: int
) -> dict[str, str]:
    pair_ids_by_label: dict[int, list[str]] = {0: [], 1: []}
    for record in scholars.values():
        for pair in record["pairs"]:
            pair_ids_by_label[int(pair["label"])].append(pair["pair_id"])

    split_by_pair: dict[str, str] = {}
    for label, pair_ids in pair_ids_by_label.items():
        ids = sorted(pair_ids)
        random.Random(seed + 2003 + label).shuffle(ids)
        n_test = int(round(len(ids) * 0.20))
        n_validation = int(round(len(ids) * 0.20))
        for pair_id in ids[:n_test]:
            split_by_pair[pair_id] = "test"
        for pair_id in ids[n_test : n_test + n_validation]:
            split_by_pair[pair_id] = "validation"
        for pair_id in ids[n_test + n_validation :]:
            split_by_pair[pair_id] = "train"
    return split_by_pair


def build_controlled_dataset(
    obs_dir: Path,
    pred_dir: Path,
    lkn_scale: int,
    seed: int,
    min_scholar_edge_count: int,
    split_unit: str,
    negative_sampling: str,
    excluded_topics: set[str],
    topic_id_regex: str | None,
    logger: RunLogger,
    require_full_cohort: bool = False,
) -> dict[str, Any]:
    topic_id_pattern = re.compile(topic_id_regex) if topic_id_regex else None
    obs_files = sorted(obs_dir.glob("*.txt"), key=lambda p: p.name)
    pred_files = sorted(pred_dir.glob("*.txt"), key=lambda p: p.name)
    observation_scholar_ids = {path.stem for path in obs_files}
    prediction_scholar_ids = {path.stem for path in pred_files}
    eligible_scholar_ids = sorted(
        observation_scholar_ids & prediction_scholar_ids
    )
    if len(eligible_scholar_ids) < lkn_scale:
        raise RuntimeError(
            f"Requested {lkn_scale:,} scholars, but only "
            f"{len(eligible_scholar_ids):,} occur in both observation and "
            "prediction directories."
        )
    random.Random(seed).shuffle(eligible_scholar_ids)
    sampled_scholar_ids = eligible_scholar_ids[:lkn_scale]
    logger(
        f"Sampled {len(sampled_scholar_ids):,} eligible LKNs from "
        f"{len(eligible_scholar_ids):,} observation/prediction intersections "
        f"({len(observation_scholar_ids):,} observation, "
        f"{len(prediction_scholar_ids):,} prediction)."
    )

    obs_edges_by_scholar: dict[str, list[tuple[str, str]]] = {}
    obs_topics_by_scholar: dict[str, set[str]] = {}
    active_topics: set[str] = set()
    edge_counter: Counter[tuple[str, str]] = Counter()
    lkn_densities: list[float] = []
    complete_lkn_count = 0

    for scholar_id in tqdm(sampled_scholar_ids, desc="Observation graphs"):
        edges, topics = read_topic_edges(
            obs_dir / f"{scholar_id}.txt", excluded_topics
        )
        if not edges:
            continue
        obs_edges_by_scholar[scholar_id] = edges
        obs_topics_by_scholar[scholar_id] = topics
        active_topics.update(topics)
        undirected_edges = {
            tuple(sorted((t1, t2))) for t1, t2 in edges if t1 != t2
        }
        possible_edges = len(topics) * (len(topics) - 1) / 2
        if possible_edges > 0:
            density_value = len(undirected_edges) / possible_edges
            lkn_densities.append(density_value)
            complete_lkn_count += int(math.isclose(density_value, 1.0))
        for t1, t2 in edges:
            edge_counter[tuple(sorted((t1, t2)))] += 1

    invalid_observation_topics = sorted(
        topic
        for topic in active_topics
        if topic_id_pattern and not topic_id_pattern.fullmatch(topic)
    )
    if invalid_observation_topics:
        raise RuntimeError(
            "Observation topic IDs do not match --topic-id-regex. Examples: "
            + ", ".join(invalid_observation_topics[:5])
        )

    sorted_topics = sorted(active_topics)
    topic_to_idx = {topic: idx for idx, topic in enumerate(sorted_topics)}
    idx_to_topic = {idx: topic for topic, idx in topic_to_idx.items()}

    gkn_edges = [
        (topic_to_idx[t1], topic_to_idx[t2])
        for (t1, t2), count in edge_counter.items()
        if count >= min_scholar_edge_count and t1 in topic_to_idx and t2 in topic_to_idx
    ]
    if gkn_edges:
        gkn_edge_index = torch.tensor(gkn_edges, dtype=torch.long).t().contiguous()
        gkn_edge_index = to_undirected(gkn_edge_index, num_nodes=len(topic_to_idx))
    else:
        gkn_edge_index = torch.empty((2, 0), dtype=torch.long)

    raw_degree = degree(gkn_edge_index[0], num_nodes=len(topic_to_idx))
    degree_bucket = torch.floor(torch.log2(raw_degree + 1)).to(torch.int64).tolist()
    log_degree = torch.log(raw_degree + 1)
    norm_degree = (
        (log_degree - log_degree.min())
        / (log_degree.max() - log_degree.min() + 1e-6)
    ).unsqueeze(1)
    gkn_data = Data(
        edge_index=gkn_edge_index,
        num_nodes=len(topic_to_idx),
        node_indices=torch.arange(len(topic_to_idx)),
    )

    scholars: dict[str, dict[str, Any]] = {}
    dropped_future_only_positive_topics = 0
    total_new_prediction_topics = 0
    prediction_topics_seen: set[str] = set()
    all_topic_indices = set(range(len(topic_to_idx)))

    for scholar_id in tqdm(sampled_scholar_ids, desc="Labels and LKNs"):
        obs_edges = obs_edges_by_scholar.get(scholar_id)
        obs_topics = obs_topics_by_scholar.get(scholar_id)
        if not obs_edges or not obs_topics:
            continue

        pred_edges, pred_topics = read_topic_edges(
            pred_dir / f"{scholar_id}.txt", excluded_topics
        )
        if not pred_edges:
            continue
        prediction_topics_seen.update(pred_topics)

        new_prediction_topics = pred_topics - obs_topics
        total_new_prediction_topics += len(new_prediction_topics)
        positive_topics = sorted(topic for topic in new_prediction_topics if topic in topic_to_idx)
        dropped_future_only_positive_topics += len(new_prediction_topics) - len(positive_topics)
        if not positive_topics:
            continue

        obs_topic_indices = {topic_to_idx[topic] for topic in obs_topics if topic in topic_to_idx}
        pred_topic_indices = {topic_to_idx[topic] for topic in pred_topics if topic in topic_to_idx}
        negative_pool = sorted(all_topic_indices - obs_topic_indices - pred_topic_indices)
        if not negative_pool:
            continue

        scholar_rng = random.Random(f"{seed}:{scholar_id}:negative")
        if negative_sampling == "degree-matched":
            available_by_bucket: dict[int, list[int]] = defaultdict(list)
            for candidate_idx in negative_pool:
                available_by_bucket[int(degree_bucket[candidate_idx])].append(
                    candidate_idx
                )
            for candidates in available_by_bucket.values():
                scholar_rng.shuffle(candidates)
            negative_indices = []
            for topic in positive_topics:
                positive_idx = topic_to_idx[topic]
                positive_bucket = int(degree_bucket[positive_idx])
                nonempty_buckets = [
                    bucket
                    for bucket, candidates in available_by_bucket.items()
                    if candidates
                ]
                if nonempty_buckets:
                    selected_bucket = min(
                        nonempty_buckets,
                        key=lambda bucket: (abs(bucket - positive_bucket), bucket),
                    )
                    negative_indices.append(
                        available_by_bucket[selected_bucket].pop()
                    )
                else:
                    negative_indices.append(scholar_rng.choice(negative_pool))
        elif len(negative_pool) >= len(positive_topics):
            negative_indices = scholar_rng.sample(
                negative_pool, len(positive_topics)
            )
        else:
            negative_indices = scholar_rng.choices(
                negative_pool, k=len(positive_topics)
            )

        global_nodes = sorted(obs_topic_indices)
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(global_nodes)}
        local_edges = []
        for t1, t2 in obs_edges:
            if t1 not in topic_to_idx or t2 not in topic_to_idx:
                continue
            g1, g2 = topic_to_idx[t1], topic_to_idx[t2]
            if g1 in global_to_local and g2 in global_to_local:
                local_edges.append((global_to_local[g1], global_to_local[g2]))
        if not local_edges:
            continue
        local_edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
        local_edge_index = to_undirected(local_edge_index, num_nodes=len(global_nodes))
        lkn_data = Data(
            edge_index=local_edge_index,
            num_nodes=len(global_nodes),
            global_indices=torch.tensor(global_nodes, dtype=torch.long),
        )

        pairs = []
        for topic, negative_idx in zip(positive_topics, negative_indices):
            positive_idx = topic_to_idx[topic]
            pairs.append(
                {
                    "pair_id": stable_pair_id(scholar_id, topic, 1),
                    "topic_idx": positive_idx,
                    "topic_id": topic,
                    "label": 1,
                }
            )
            negative_topic = idx_to_topic[int(negative_idx)]
            pairs.append(
                {
                    "pair_id": stable_pair_id(scholar_id, negative_topic, 0),
                    "topic_idx": int(negative_idx),
                    "topic_id": negative_topic,
                    "label": 0,
                }
            )
        pairs.sort(key=lambda row: (row["topic_id"], row["label"]))
        scholars[scholar_id] = {
            "scholar_id": scholar_id,
            "lkn_data": lkn_data,
            "pairs": pairs,
        }

    invalid_prediction_topics = sorted(
        topic
        for topic in prediction_topics_seen
        if topic_id_pattern and not topic_id_pattern.fullmatch(topic)
    )
    if invalid_prediction_topics:
        raise RuntimeError(
            "Prediction topic IDs do not match --topic-id-regex. Examples: "
            + ", ".join(invalid_prediction_topics[:5])
        )
    namespace_intersection = active_topics & prediction_topics_seen
    if not namespace_intersection:
        raise RuntimeError(
            "Observation and prediction topic vocabularies have no overlap."
        )
    if require_full_cohort and len(scholars) != len(sampled_scholar_ids):
        raise RuntimeError(
            "The corrected protocol requires every sampled scholar to contribute "
            f"usable pairs, but only {len(scholars):,} of "
            f"{len(sampled_scholar_ids):,} did."
        )

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    manifest_rows = []
    if split_unit == "scholar":
        train_ids, val_ids, test_ids = split_contributing_scholars(
            list(scholars), seed
        )
        split_by_scholar = {
            scholar_id: (
                "train"
                if scholar_id in train_ids
                else "validation"
                if scholar_id in val_ids
                else "test"
            )
            for scholar_id in scholars
        }
        split_by_pair = {
            pair["pair_id"]: split_by_scholar[scholar_id]
            for scholar_id, record in scholars.items()
            for pair in record["pairs"]
        }
    else:
        split_by_pair = split_pairs(scholars, seed)

    for scholar_id in sorted(scholars):
        source_record = scholars[scholar_id]
        pairs_by_split: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for pair in source_record["pairs"]:
            split = split_by_pair[pair["pair_id"]]
            pairs_by_split[split].append(pair)
            manifest_rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "scholar_id": scholar_id,
                    "candidate_topic": pair["topic_id"],
                    "candidate_idx": pair["topic_idx"],
                    "label": pair["label"],
                    "split": split,
                }
            )
        for split, pairs in pairs_by_split.items():
            if not pairs:
                continue
            records_by_split[split].append(
                {
                    "scholar_id": scholar_id,
                    "lkn_data": source_record["lkn_data"],
                    "pairs": pairs,
                }
            )

    logger(
        "Controlled data ready: "
        f"topics={len(topic_to_idx):,}, GKN edges={gkn_edge_index.size(1) // 2:,}, "
        f"contributors={len(scholars):,}, pairs={len(manifest_rows):,}."
    )
    logger(
        f"{split_unit.capitalize()} split: "
        + ", ".join(
            f"{split}={len(records_by_split[split]):,} scholars/"
            f"{sum(len(record['pairs']) for record in records_by_split[split]):,} pairs"
            for split in ("train", "validation", "test")
        )
    )
    positive_degrees = [
        float(raw_degree[int(row["candidate_idx"])])
        for row in manifest_rows
        if int(row["label"]) == 1
    ]
    negative_degrees = [
        float(raw_degree[int(row["candidate_idx"])])
        for row in manifest_rows
        if int(row["label"]) == 0
    ]

    return {
        "sampled_scholar_ids": sampled_scholar_ids,
        "topic_to_idx": topic_to_idx,
        "idx_to_topic": idx_to_topic,
        "gkn_data": gkn_data,
        "norm_degree": norm_degree,
        "scholars": scholars,
        "records_by_split": records_by_split,
        "manifest_rows": manifest_rows,
        "stats": {
            "available_observation_lkns": len(observation_scholar_ids),
            "available_prediction_lkns": len(prediction_scholar_ids),
            "eligible_observation_prediction_lkns": len(eligible_scholar_ids),
            "sampled_observation_lkns": len(sampled_scholar_ids),
            "valid_observation_lkns": len(obs_edges_by_scholar),
            "contributing_scholars": len(scholars),
            "num_topics": len(topic_to_idx),
            "num_gkn_undirected_edges": int(gkn_edge_index.size(1) // 2),
            "total_pairs": len(manifest_rows),
            "total_positive_pairs": sum(row["label"] for row in manifest_rows),
            "total_negative_pairs": sum(1 - row["label"] for row in manifest_rows),
            "total_new_prediction_topics_before_vocab_filter": total_new_prediction_topics,
            "dropped_prediction_only_positive_topics": dropped_future_only_positive_topics,
            "vocabulary_source": "observation_window_only",
            "split_unit": split_unit,
            "negative_sampling": negative_sampling,
            "mean_positive_candidate_degree": float(np.mean(positive_degrees)),
            "mean_negative_candidate_degree": float(np.mean(negative_degrees)),
            "topic_id_regex": topic_id_regex or "",
            "observation_prediction_topic_intersection": len(
                namespace_intersection
            ),
            "mean_observation_lkn_density": float(np.mean(lkn_densities)),
            "median_observation_lkn_density": float(np.median(lkn_densities)),
            "complete_observation_lkn_fraction": (
                complete_lkn_count / len(lkn_densities)
                if lkn_densities
                else 0.0
            ),
        },
    }


def write_dataset_artifacts(
    dataset: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    obs_dir: Path,
    pred_dir: Path,
    excluded_topics: set[str],
) -> None:
    with (output_dir / "sampled_scholar_ids.txt").open("w", encoding="utf-8") as f:
        for scholar_id in dataset["sampled_scholar_ids"]:
            f.write(scholar_id + "\n")

    (output_dir / "topic_to_idx.json").write_text(
        json.dumps(dataset["topic_to_idx"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path = output_dir / "pair_manifest.csv"
    pd.DataFrame(dataset["manifest_rows"]).to_csv(manifest_path, index=False)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    split_summary = {}
    for split, records in dataset["records_by_split"].items():
        pairs = [pair for record in records for pair in record["pairs"]]
        split_summary[split] = {
            "scholars": len(records),
            "pairs": len(pairs),
            "positive_pairs": sum(int(pair["label"]) for pair in pairs),
            "negative_pairs": sum(1 - int(pair["label"]) for pair in pairs),
        }

    config = {
        "observation_window_years": args.obs_window,
        "lkn_scale": args.lkn_scale,
        "lkn_obs_dir": str(obs_dir),
        "lkn_pred_dir": str(pred_dir),
        "data_seed": args.seed,
        "model_seed": args.model_seed if args.model_seed is not None else args.seed,
        "min_scholar_edge_count": args.min_scholar_edge_count,
        "negative_sampling_ratio": "1:1",
        "negative_sampling": args.negative_sampling,
        "excluded_topics": sorted(excluded_topics),
        "topic_id_regex": args.topic_id_regex or "",
        "cohort_sampling": "observation_prediction_intersection_before_sampling",
        "require_full_contributing_cohort": not args.allow_incomplete_cohort,
        "split_unit": args.split_unit,
        "split_ratio": f"60:20:20 by {args.split_unit}",
        "pair_manifest_sha256": manifest_sha256,
        "model_config": model_config_from_args(args),
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "software": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "torch_geometric": __import__("torch_geometric").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output_dir / "controlled_run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                **dataset["stats"],
                "splits": split_summary,
                "pair_manifest_sha256": manifest_sha256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_ids = {
        record["scholar_id"] for record in dataset["records_by_split"]["train"]
    }
    val_ids = {
        record["scholar_id"] for record in dataset["records_by_split"]["validation"]
    }
    test_ids = {
        record["scholar_id"] for record in dataset["records_by_split"]["test"]
    }
    integrity = {
        "pair_ids_unique": len({row["pair_id"] for row in dataset["manifest_rows"]})
        == len(dataset["manifest_rows"]),
        "train_validation_scholar_overlap": len(train_ids & val_ids),
        "train_test_scholar_overlap": len(train_ids & test_ids),
        "validation_test_scholar_overlap": len(val_ids & test_ids),
        "prediction_only_topics_in_vocabulary": 0,
        "same_pair_manifest_for_all_models": True,
        "sampled_from_observation_prediction_intersection": (
            dataset["stats"]["eligible_observation_prediction_lkns"]
            >= len(dataset["sampled_scholar_ids"])
        ),
        "full_contributing_cohort": (
            dataset["stats"]["contributing_scholars"]
            == len(dataset["sampled_scholar_ids"])
        ),
        "topic_id_validation_enabled": bool(args.topic_id_regex),
    }
    scholar_overlap_is_valid = args.split_unit == "pair" or all(
        integrity[key] == 0
        for key in (
            "train_validation_scholar_overlap",
            "train_test_scholar_overlap",
            "validation_test_scholar_overlap",
        )
    )
    integrity["scholar_overlap_matches_protocol"] = scholar_overlap_is_valid
    (output_dir / "integrity_checks.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    if not all(
        [
            integrity["pair_ids_unique"],
            integrity["scholar_overlap_matches_protocol"],
            integrity["sampled_from_observation_prediction_intersection"],
            integrity["full_contributing_cohort"]
            or args.allow_incomplete_cohort,
        ]
    ):
        raise RuntimeError(f"Controlled split integrity check failed: {integrity}")


class ScholarDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class ScholarBatchCollator:
    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        lkn_batch = Batch.from_data_list([record["lkn_data"] for record in records])
        topic_indices: list[int] = []
        labels: list[float] = []
        scholar_positions: list[int] = []
        metadata: list[dict[str, Any]] = []
        for scholar_position, record in enumerate(records):
            for pair in record["pairs"]:
                topic_indices.append(int(pair["topic_idx"]))
                labels.append(float(pair["label"]))
                scholar_positions.append(scholar_position)
                metadata.append(
                    {
                        "pair_id": pair["pair_id"],
                        "scholar_id": record["scholar_id"],
                        "topic_id": pair["topic_id"],
                        "label": int(pair["label"]),
                    }
                )
        return {
            "lkn_batch": lkn_batch,
            "topic_indices": torch.tensor(topic_indices, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
            "scholar_positions": torch.tensor(scholar_positions, dtype=torch.long),
            "metadata": metadata,
        }


class NTSPredictor(nn.Module):
    def __init__(
        self,
        num_topics: int,
        embedding_dim: int,
        hidden_dim: int,
        out_dim: int,
        node_degrees: torch.Tensor,
        heads: int,
        dropout: float,
        pooling: str = "mean",
        fusion: str = "product",
    ):
        super().__init__()
        self.pooling = pooling
        self.fusion = fusion
        self.register_buffer("node_degrees", node_degrees)
        self.shared_embedding = nn.Embedding(num_topics, embedding_dim)
        input_dim = embedding_dim + 1
        gat_hidden = hidden_dim // heads
        self.gkn_gnn1 = GATConv(input_dim, gat_hidden, heads=heads, dropout=dropout)
        self.gkn_gnn2 = GATConv(
            hidden_dim, out_dim, heads=1, concat=False, dropout=dropout
        )
        self.lkn_gnn1 = GATConv(input_dim, gat_hidden, heads=heads, dropout=dropout)
        self.lkn_gnn2 = GATConv(
            hidden_dim, out_dim, heads=1, concat=False, dropout=dropout
        )
        predictor_input_dim = out_dim if fusion == "product" else out_dim * 4
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def node_features(self, node_indices: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.shared_embedding(node_indices), self.node_degrees[node_indices]],
            dim=-1,
        )

    def encode_gkn(self, gkn_data: Data) -> torch.Tensor:
        x = self.node_features(gkn_data.node_indices)
        x = F.elu(self.gkn_gnn1(x, gkn_data.edge_index))
        return self.gkn_gnn2(x, gkn_data.edge_index)

    def encode_scholar_nodes(self, lkn_batch: Batch) -> torch.Tensor:
        x = self.node_features(lkn_batch.global_indices)
        x = F.elu(self.lkn_gnn1(x, lkn_batch.edge_index))
        return self.lkn_gnn2(x, lkn_batch.edge_index)

    def encode_scholars(self, lkn_batch: Batch) -> torch.Tensor:
        return global_mean_pool(
            self.encode_scholar_nodes(lkn_batch), lkn_batch.batch
        )

    def encode_pairs(
        self,
        lkn_batch: Batch,
        scholar_positions: torch.Tensor,
        topic_indices: torch.Tensor,
        all_topic_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_topic_embeddings = all_topic_embeddings[topic_indices]
        if self.pooling == "mean":
            per_scholar = self.encode_scholars(lkn_batch)
            return per_scholar[scholar_positions], pair_topic_embeddings

        node_embeddings = self.encode_scholar_nodes(lkn_batch)
        pair_scholar_embeddings = torch.zeros_like(pair_topic_embeddings)
        scale = math.sqrt(max(node_embeddings.size(1), 1))
        for scholar_position in range(int(lkn_batch.num_graphs)):
            pair_indices = torch.nonzero(
                scholar_positions == scholar_position, as_tuple=False
            ).view(-1)
            if pair_indices.numel() == 0:
                continue
            scholar_nodes = node_embeddings[
                lkn_batch.batch == scholar_position
            ]
            candidate_queries = pair_topic_embeddings[pair_indices]
            attention = torch.softmax(
                candidate_queries @ scholar_nodes.t() / scale,
                dim=1,
            )
            pooled = attention @ scholar_nodes
            pair_scholar_embeddings = pair_scholar_embeddings.index_copy(
                0, pair_indices, pooled
            )
        return pair_scholar_embeddings, pair_topic_embeddings

    def forward(
        self, scholar_embeddings: torch.Tensor, topic_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if self.fusion == "product":
            features = scholar_embeddings * topic_embeddings
        else:
            features = torch.cat(
                [
                    scholar_embeddings,
                    topic_embeddings,
                    scholar_embeddings * topic_embeddings,
                    torch.abs(scholar_embeddings - topic_embeddings),
                ],
                dim=-1,
            )
        return self.predictor(features).view(-1)


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities > threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels) if len(labels) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_validation_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, dict[str, float]]:
    candidates = np.arange(0.05, 0.951, 0.01)
    scored = [
        (float(threshold), binary_metrics(labels, probabilities, float(threshold)))
        for threshold in candidates
    ]
    threshold, metrics = max(
        scored,
        key=lambda item: (
            item[1]["f1"],
            item[1]["accuracy"],
            -abs(item[0] - 0.5),
        ),
    )
    return threshold, metrics


def make_scholar_loader(
    records: list[dict[str, Any]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ScholarDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ScholarBatchCollator(),
        num_workers=0,
        generator=generator if shuffle else None,
    )


def score_gnn(
    model: NTSPredictor,
    gkn_data: Data,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    model.eval()
    loader = make_scholar_loader(records, batch_size, False, 0)
    rows = []
    with torch.no_grad():
        topic_embeddings = model.encode_gkn(gkn_data)
        for batch in loader:
            lkn_batch = batch["lkn_batch"].to(device)
            topic_indices = batch["topic_indices"].to(device)
            scholar_positions = batch["scholar_positions"].to(device)
            scholar_embeddings, pair_topic_embeddings = model.encode_pairs(
                lkn_batch,
                scholar_positions,
                topic_indices,
                topic_embeddings,
            )
            logits = model(
                scholar_embeddings,
                pair_topic_embeddings,
            )
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            for meta, probability in zip(batch["metadata"], probabilities):
                rows.append(
                    {
                        **meta,
                        "probability": float(probability),
                    }
                )
    frame = pd.DataFrame(rows)
    metrics = None
    if threshold is not None and not frame.empty:
        metrics = binary_metrics(
            frame["label"].to_numpy(),
            frame["probability"].to_numpy(),
            threshold,
        )
    return frame, metrics


def train_gnn(
    dataset: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    logger: RunLogger,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = model_config_from_args(args)
    model_seed = args.model_seed if args.model_seed is not None else args.seed
    seed_everything(model_seed)
    gkn_data = dataset["gkn_data"].to(device)
    model = NTSPredictor(
        num_topics=len(dataset["topic_to_idx"]),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        out_dim=config["out_dim"],
        node_degrees=dataset["norm_degree"],
        heads=config["heads"],
        dropout=config["dropout"],
        pooling=config["pooling"],
        fusion=config["fusion"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    train_loader = make_scholar_loader(
        dataset["records_by_split"]["train"],
        args.scholar_batch_size,
        True,
        model_seed,
    )

    checkpoint_path = output_dir / "controlled_gnn.pth"
    history_path = output_dir / "gnn_training_history.csv"
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0

    if args.resume_model and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        best_f1 = float(payload["validation_metrics"]["f1"])
        best_epoch = int(payload["epoch"])
        logger(f"Loaded existing controlled checkpoint from epoch {best_epoch}.")
    else:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            pair_count = 0
            for batch in train_loader:
                lkn_batch = batch["lkn_batch"].to(device)
                topic_indices = batch["topic_indices"].to(device)
                labels = batch["labels"].to(device)
                scholar_positions = batch["scholar_positions"].to(device)

                optimizer.zero_grad(set_to_none=True)
                topic_embeddings = model.encode_gkn(gkn_data)
                scholar_embeddings, pair_topic_embeddings = model.encode_pairs(
                    lkn_batch,
                    scholar_positions,
                    topic_indices,
                    topic_embeddings,
                )
                logits = model(
                    scholar_embeddings,
                    pair_topic_embeddings,
                )
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                optimizer.step()

                n_pairs = int(labels.numel())
                epoch_loss += float(loss.item()) * n_pairs
                pair_count += n_pairs

            val_frame, _ = score_gnn(
                model,
                gkn_data,
                dataset["records_by_split"]["validation"],
                device,
                args.eval_scholar_batch_size,
            )
            threshold, val_metrics = best_validation_threshold(
                val_frame["label"].to_numpy(),
                val_frame["probability"].to_numpy(),
            )
            scheduler.step(val_metrics["f1"])
            row = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(pair_count, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)

            improved = val_metrics["f1"] > best_f1 + 1e-6
            if improved:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "threshold": threshold,
                        "validation_metrics": val_metrics,
                        "model_config": config,
                        "num_topics": len(dataset["topic_to_idx"]),
                        "pair_manifest_sha256": json.loads(
                            (output_dir / "controlled_run_config.json").read_text(
                                encoding="utf-8"
                            )
                        )["pair_manifest_sha256"],
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1

            logger(
                f"GNN epoch {epoch:03d}: loss={row['train_loss']:.4f}, "
                f"val F1={val_metrics['f1']:.4f}, "
                f"P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}, "
                f"threshold={threshold:.2f}, stale={stale_epochs}/{args.patience}."
            )
            if stale_epochs >= args.patience:
                logger(f"GNN early stopping at epoch {epoch}; best epoch={best_epoch}.")
                break

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    threshold = float(payload["threshold"])
    val_frame, val_metrics = score_gnn(
        model,
        gkn_data,
        dataset["records_by_split"]["validation"],
        device,
        args.eval_scholar_batch_size,
        threshold,
    )
    test_frame, test_metrics = score_gnn(
        model,
        gkn_data,
        dataset["records_by_split"]["test"],
        device,
        args.eval_scholar_batch_size,
        threshold,
    )
    val_frame["model"] = "Ours"
    test_frame["model"] = "Ours"
    result = {
        "model": "Ours",
        "best_epoch": int(payload["epoch"]),
        "validation": val_metrics,
        "test": test_metrics,
        "threshold_selection": "validation F1",
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "gnn_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    val_frame.to_csv(output_dir / "gnn_validation_predictions.csv", index=False)
    test_frame.to_csv(output_dir / "gnn_test_predictions.csv", index=False)
    logger(
        "GNN test: "
        f"Acc={test_metrics['accuracy']:.4f}, P={test_metrics['precision']:.4f}, "
        f"R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}."
    )
    return result, val_frame, test_frame


def graph_average_shortest_path(graph: nx.Graph) -> float:
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0
    if nx.is_connected(graph):
        return float(nx.average_shortest_path_length(graph))
    largest_nodes = max(nx.connected_components(graph), key=len)
    largest = graph.subgraph(largest_nodes)
    return (
        float(nx.average_shortest_path_length(largest))
        if largest.number_of_nodes() > 1
        else 0.0
    )


def nx_graphs_from_dataset(
    dataset: dict[str, Any],
) -> tuple[nx.Graph, dict[str, nx.Graph]]:
    gkn = nx.Graph()
    gkn.add_nodes_from(range(len(dataset["topic_to_idx"])))
    gkn.add_edges_from(
        (int(src), int(dst))
        for src, dst in dataset["gkn_data"].edge_index.t().tolist()
        if int(src) < int(dst)
    )
    scholar_graphs = {}
    for scholar_id, record in dataset["scholars"].items():
        lkn_data = record["lkn_data"]
        global_indices = [int(value) for value in lkn_data.global_indices.tolist()]
        graph = nx.Graph()
        graph.add_nodes_from(global_indices)
        for src, dst in lkn_data.edge_index.t().tolist():
            g_src, g_dst = global_indices[int(src)], global_indices[int(dst)]
            if g_src != g_dst:
                graph.add_edge(g_src, g_dst)
        scholar_graphs[scholar_id] = graph
    return gkn, scholar_graphs


def scholar_feature_prefix(
    lkn: nx.Graph,
    gkn: nx.Graph,
    gkn_degrees: dict[int, int],
    n_gkn: int,
) -> tuple[list[float], list[int]]:
    n_j = lkn.number_of_nodes()
    e_j = lkn.number_of_edges()
    x1 = 2 * e_j / n_j if n_j else 0.0
    x2 = nx.density(lkn)
    x3 = graph_average_shortest_path(lkn)
    x4 = nx.average_clustering(lkn) if n_j else 0.0
    x5 = 0.0
    if e_j:
        try:
            communities = list(greedy_modularity_communities(lkn))
            x5 = float(modularity(lkn, communities))
        except Exception:
            x5 = 0.0

    valid_nodes = [int(node) for node in lkn.nodes if gkn.has_node(node)]
    sub_lkn = gkn.subgraph(valid_nodes)
    n_s = sub_lkn.number_of_nodes()
    e_s = sub_lkn.number_of_edges()
    x6 = 2 * e_s / n_s if n_s else 0.0
    x7 = nx.density(sub_lkn)
    x8 = graph_average_shortest_path(nx.Graph(sub_lkn))
    x9 = nx.average_clustering(sub_lkn) if n_s else 0.0
    if n_s:
        sum_degree = sum(gkn_degrees.get(node, 0) for node in valid_nodes)
        x10 = (sum_degree / max(n_gkn - 1, 1)) / n_s
        external_neighbors: set[int] = set()
        for node in valid_nodes:
            external_neighbors.update(gkn.neighbors(node))
        external_neighbors.difference_update(valid_nodes)
        x11 = len(external_neighbors) / n_s
    else:
        x10, x11 = 0.0, 0.0
    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], valid_nodes


def candidate_feature_suffix(
    topic: int,
    valid_nodes: list[int],
    gkn: nx.Graph,
    gkn_degrees: dict[int, int],
    clustering: dict[int, float],
    neighbor_cache: dict[int, set[int]],
    mean_distance_by_topic: dict[int, float],
    n_gkn: int,
) -> list[float]:
    n_s = len(valid_nodes)
    if gkn.has_node(topic) and n_s:
        topic_neighbors = neighbor_cache[topic]
        x12 = len(topic_neighbors.intersection(valid_nodes)) / n_s
        common_neighbor_sum = sum(
            len(topic_neighbors.intersection(neighbor_cache[node]))
            for node in valid_nodes
        )
        x14 = common_neighbor_sum / n_s
        x13 = mean_distance_by_topic.get(topic, 10.0)
    else:
        x12, x13, x14 = 0.0, 10.0, 0.0
    x15 = gkn_degrees.get(topic, 0) / max(n_gkn - 1, 1)
    x16 = clustering.get(topic, 0.0)
    return [x12, x13, x14, x15, x16]


def candidate_mean_distances(
    topics: list[int],
    valid_nodes: list[int],
    gkn_adjacency: csr_matrix,
) -> dict[int, float]:
    unique_topics = np.asarray(sorted(set(topics)), dtype=np.int32)
    if unique_topics.size == 0 or not valid_nodes:
        return {}
    unique_valid_nodes = np.asarray(sorted(set(valid_nodes)), dtype=np.int32)
    if unique_valid_nodes.size < unique_topics.size:
        distances = dijkstra(
            gkn_adjacency,
            directed=False,
            unweighted=True,
            indices=unique_valid_nodes,
            return_predecessors=False,
            limit=5,
        )
        distances = np.atleast_2d(distances)[:, unique_topics].T
    else:
        distances = dijkstra(
            gkn_adjacency,
            directed=False,
            unweighted=True,
            indices=unique_topics,
            return_predecessors=False,
            limit=5,
        )
        distances = np.atleast_2d(distances)[:, unique_valid_nodes]
    usable = np.isfinite(distances) & (distances <= 5)
    counts = usable.sum(axis=1)
    sums = np.where(usable, distances, 0.0).sum(axis=1)
    means = np.divide(
        sums,
        counts,
        out=np.full(unique_topics.size, 10.0, dtype=np.float64),
        where=counts > 0,
    )
    return {
        int(topic): float(mean)
        for topic, mean in zip(unique_topics.tolist(), means.tolist())
    }


def extract_one_scholar_features(
    record: dict[str, Any],
    split_by_pair: dict[str, str],
    lkn: nx.Graph,
    gkn: nx.Graph,
    gkn_degrees: dict[int, int],
    clustering: dict[int, float],
    neighbor_cache: dict[int, set[int]],
    gkn_adjacency: csr_matrix,
    n_gkn: int,
    prefix_override: list[float] | None = None,
) -> list[dict[str, Any]]:
    if prefix_override is None:
        prefix, valid_nodes = scholar_feature_prefix(
            lkn, gkn, gkn_degrees, n_gkn
        )
    else:
        prefix = prefix_override
        valid_nodes = [
            int(node) for node in lkn.nodes if gkn.has_node(int(node))
        ]
    mean_distance_by_topic = candidate_mean_distances(
        [int(pair["topic_idx"]) for pair in record["pairs"]],
        valid_nodes,
        gkn_adjacency,
    )
    rows = []
    for pair in record["pairs"]:
        suffix = candidate_feature_suffix(
            int(pair["topic_idx"]),
            valid_nodes,
            gkn,
            gkn_degrees,
            clustering,
            neighbor_cache,
            mean_distance_by_topic,
            n_gkn,
        )
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "scholar_id": record["scholar_id"],
                "candidate_topic": pair["topic_id"],
                "candidate_idx": int(pair["topic_idx"]),
                "label": int(pair["label"]),
                "split": split_by_pair[pair["pair_id"]],
                **dict(zip(FEATURE_NAMES, prefix + suffix)),
            }
        )
    return rows


def extract_baseline_features(
    dataset: dict[str, Any],
    output_dir: Path,
    n_jobs: int,
    reuse: bool,
    feature_source: str | None,
    partial_feature_source: str | None,
    logger: RunLogger,
) -> pd.DataFrame:
    feature_path = output_dir / "common_pair_features.csv.gz"
    if reuse and feature_path.exists():
        logger(f"Reusing baseline features from {feature_path}.")
        return pd.read_csv(feature_path)
    if feature_source:
        source_path = Path(feature_source).resolve()
        logger(f"Reusing structural feature values from {source_path}.")
        frame = pd.read_csv(source_path)
        split_by_pair = {
            row["pair_id"]: row["split"] for row in dataset["manifest_rows"]
        }
        expected = set(split_by_pair)
        actual = set(frame["pair_id"])
        if expected != actual or len(frame) != len(dataset["manifest_rows"]):
            raise RuntimeError(
                "Feature source does not match the controlled pair manifest."
            )
        frame["split"] = frame["pair_id"].map(split_by_pair)
        frame.to_csv(feature_path, index=False, compression="gzip")
        return frame

    logger("Constructing NetworkX graphs for the common-pair baselines.")
    gkn, scholar_graphs = nx_graphs_from_dataset(dataset)
    gkn_degrees = dict(gkn.degree())
    clustering = nx.clustering(gkn)
    neighbor_cache = {int(node): set(gkn.neighbors(node)) for node in gkn.nodes}
    n_gkn = gkn.number_of_nodes()
    gkn_adjacency = csr_matrix(
        nx.to_scipy_sparse_array(
            gkn,
            nodelist=range(n_gkn),
            dtype=np.int8,
            format="csr",
        )
    )
    gkn_adjacency.indices = gkn_adjacency.indices.astype(np.int32, copy=False)
    gkn_adjacency.indptr = gkn_adjacency.indptr.astype(np.int32, copy=False)

    tasks = [dataset["scholars"][scholar_id] for scholar_id in sorted(dataset["scholars"])]
    split_by_pair = {
        row["pair_id"]: row["split"] for row in dataset["manifest_rows"]
    }
    reused_frame = pd.DataFrame()
    prefix_by_scholar: dict[str, list[float]] = {}
    if partial_feature_source:
        source_path = Path(partial_feature_source).resolve()
        logger(f"Loading partial structural-feature cache from {source_path}.")
        source = pd.read_csv(source_path)
        required_columns = {"scholar_id", "candidate_idx", *FEATURE_NAMES}
        missing_columns = required_columns.difference(source.columns)
        if missing_columns:
            raise RuntimeError(
                "Partial feature source is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        source["scholar_id"] = source["scholar_id"].astype(str)
        source["candidate_idx"] = source["candidate_idx"].astype(int)
        source = source.drop_duplicates(
            subset=["scholar_id", "candidate_idx"], keep="first"
        )
        prefix_by_scholar = {
            str(row["scholar_id"]): [
                float(row[name]) for name in FEATURE_NAMES[:11]
            ]
            for _, row in source.drop_duplicates(
                subset=["scholar_id"], keep="first"
            ).iterrows()
        }

        manifest = pd.DataFrame(dataset["manifest_rows"])
        manifest["scholar_id"] = manifest["scholar_id"].astype(str)
        manifest["candidate_idx"] = manifest["candidate_idx"].astype(int)
        manifest["_manifest_order"] = np.arange(len(manifest))
        reusable = source[
            ["scholar_id", "candidate_idx", *FEATURE_NAMES]
        ]
        merged = manifest.merge(
            reusable,
            on=["scholar_id", "candidate_idx"],
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        reused_frame = merged[merged["_merge"] == "both"].drop(
            columns=["_merge"]
        )
        missing_pair_ids = set(
            merged.loc[merged["_merge"] == "left_only", "pair_id"].astype(str)
        )
        tasks = [
            {
                **record,
                "pairs": [
                    pair
                    for pair in record["pairs"]
                    if str(pair["pair_id"]) in missing_pair_ids
                ],
            }
            for record in tasks
        ]
        tasks = [record for record in tasks if record["pairs"]]
        logger(
            f"Reused {len(reused_frame):,} of {len(manifest):,} feature rows; "
            f"computing {len(missing_pair_ids):,} missing rows for "
            f"{len(tasks):,} scholars."
        )

    logger(
        f"Extracting 16 structural features for {len(tasks):,} scholars and "
        f"{sum(len(record['pairs']) for record in tasks):,} pairs."
    )
    results = joblib.Parallel(n_jobs=max(1, n_jobs), prefer="threads")(
        joblib.delayed(extract_one_scholar_features)(
            record,
            split_by_pair,
            scholar_graphs[record["scholar_id"]],
            gkn,
            gkn_degrees,
            clustering,
            neighbor_cache,
            gkn_adjacency,
            n_gkn,
            prefix_by_scholar.get(str(record["scholar_id"])),
        )
        for record in tqdm(tasks, desc="Baseline features")
    )
    rows = [row for group in results for row in group]
    computed_frame = pd.DataFrame(rows)
    if reused_frame.empty:
        frame = computed_frame
    elif computed_frame.empty:
        frame = (
            reused_frame.sort_values("_manifest_order")
            .drop(columns=["_manifest_order"])
            .reset_index(drop=True)
        )
    else:
        computed_frame["_manifest_order"] = computed_frame["pair_id"].map(
            {
                str(row["pair_id"]): index
                for index, row in enumerate(dataset["manifest_rows"])
            }
        )
        frame = (
            pd.concat([reused_frame, computed_frame], ignore_index=True)
            .sort_values("_manifest_order")
            .drop(columns=["_manifest_order"])
            .reset_index(drop=True)
        )
    frame.to_csv(feature_path, index=False, compression="gzip")

    expected = set(row["pair_id"] for row in dataset["manifest_rows"])
    actual = set(frame["pair_id"])
    if expected != actual or len(frame) != len(dataset["manifest_rows"]):
        raise RuntimeError(
            "Baseline feature rows do not exactly match the controlled pair manifest."
        )
    logger(f"Saved common-pair features to {feature_path}.")
    return frame


def fit_baselines(
    features: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    logger: RunLogger,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    train = features[features["split"] == "train"].reset_index(drop=True)
    validation = features[features["split"] == "validation"].reset_index(drop=True)
    test = features[features["split"] == "test"].reset_index(drop=True)
    x_train = np.nan_to_num(
        train[FEATURE_NAMES].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    x_validation = np.nan_to_num(
        validation[FEATURE_NAMES].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    x_test = np.nan_to_num(
        test[FEATURE_NAMES].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y_train = train["label"].to_numpy(dtype=np.int64)
    y_validation = validation["label"].to_numpy(dtype=np.int64)
    y_test = test["label"].to_numpy(dtype=np.int64)

    models = {
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=args.seed,
            verbose=-1,
            n_jobs=max(1, args.n_jobs),
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=max(1, args.n_jobs),
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            random_seed=args.seed,
            verbose=0,
            thread_count=max(1, args.n_jobs),
            allow_writing_files=False,
        ),
        "DegreeOnly": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=args.seed,
            ),
        ),
    }

    all_results: dict[str, Any] = {}
    validation_predictions = []
    test_predictions = []
    model_dir = output_dir / "baseline_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        logger(f"Training controlled baseline: {name}.")
        if name == "DegreeOnly":
            train_x = x_train[:, [14]]
            validation_x = x_validation[:, [14]]
            test_x = x_test[:, [14]]
        else:
            train_x, validation_x, test_x = x_train, x_validation, x_test
        model.fit(train_x, y_train)
        validation_probabilities = model.predict_proba(validation_x)[:, 1]
        threshold, validation_metrics = best_validation_threshold(
            y_validation, validation_probabilities
        )
        test_probabilities = model.predict_proba(test_x)[:, 1]
        test_metrics = binary_metrics(y_test, test_probabilities, threshold)
        all_results[name] = {
            "model": name,
            "validation": validation_metrics,
            "test": test_metrics,
            "threshold_selection": "validation F1",
        }
        joblib.dump(model, model_dir / f"{name.lower()}.joblib")

        for base, probability in zip(
            validation[
                ["pair_id", "scholar_id", "candidate_topic", "label"]
            ].to_dict("records"),
            validation_probabilities,
        ):
            validation_predictions.append(
                {**base, "model": name, "probability": float(probability)}
            )
        for base, probability in zip(
            test[["pair_id", "scholar_id", "candidate_topic", "label"]].to_dict(
                "records"
            ),
            test_probabilities,
        ):
            test_predictions.append(
                {**base, "model": name, "probability": float(probability)}
            )
        logger(
            f"{name} test: Acc={test_metrics['accuracy']:.4f}, "
            f"P={test_metrics['precision']:.4f}, R={test_metrics['recall']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, threshold={threshold:.2f}."
        )

    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    validation_frame = pd.DataFrame(validation_predictions)
    test_frame = pd.DataFrame(test_predictions)
    validation_frame.to_csv(output_dir / "baseline_validation_predictions.csv", index=False)
    test_frame.to_csv(output_dir / "baseline_test_predictions.csv", index=False)
    return all_results, validation_frame, test_frame


def cluster_bootstrap_summary(
    test_predictions: pd.DataFrame,
    thresholds: dict[str, float],
    best_baseline: str,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scholars = sorted(test_predictions["scholar_id"].astype(str).unique())
    model_frames = {
        model: frame.reset_index(drop=True).assign(
            scholar_id=lambda values: values["scholar_id"].astype(str)
        )
        for model, frame in test_predictions.groupby("model")
    }
    scholar_to_index = {
        scholar: index for index, scholar in enumerate(scholars)
    }
    rng = np.random.default_rng(seed + 919)
    sampled_indices = rng.integers(
        0,
        len(scholars),
        size=(replicates, len(scholars)),
        dtype=np.int32,
    )

    metrics_by_model: dict[str, dict[str, np.ndarray]] = {}
    for model, frame in model_frames.items():
        labels = frame["label"].to_numpy(dtype=np.int64)
        predictions = (
            frame["probability"].to_numpy() > thresholds[model]
        ).astype(np.int64)
        scholar_indices = frame["scholar_id"].map(scholar_to_index).to_numpy()
        per_scholar = np.zeros((len(scholars), 4), dtype=np.int64)
        categories = np.select(
            [
                (labels == 1) & (predictions == 1),
                (labels == 0) & (predictions == 0),
                (labels == 0) & (predictions == 1),
            ],
            [0, 1, 2],
            default=3,
        )
        np.add.at(per_scholar, (scholar_indices, categories), 1)

        totals = np.empty((replicates, 4), dtype=np.int64)
        for start in range(0, replicates, 100):
            end = min(start + 100, replicates)
            totals[start:end] = per_scholar[
                sampled_indices[start:end]
            ].sum(axis=1)
        tp, tn, fp, fn = totals.T
        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros(replicates, dtype=float),
            where=(tp + fp) > 0,
        )
        recall = np.divide(
            tp,
            tp + fn,
            out=np.zeros(replicates, dtype=float),
            where=(tp + fn) > 0,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(replicates, dtype=float),
            where=(precision + recall) > 0,
        )
        accuracy = (tp + tn) / np.maximum(totals.sum(axis=1), 1)
        metrics_by_model[model] = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "accuracy": accuracy,
        }

    baseline_metrics = metrics_by_model[best_baseline]
    frames = []
    for model, metrics in metrics_by_model.items():
        frames.append(
            pd.DataFrame(
                {
                    "replicate": np.arange(replicates),
                    "model": model,
                    **metrics,
                    "delta_f1_vs_best_baseline": (
                        metrics["f1"] - baseline_metrics["f1"]
                    ),
                    "delta_recall_vs_best_baseline": (
                        metrics["recall"] - baseline_metrics["recall"]
                    ),
                }
            )
        )
    bootstrap = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["replicate", "model"])
        .reset_index(drop=True)
    )
    summary: dict[str, Any] = {
        "replicates": replicates,
        "resampling_unit": "test scholar",
        "best_baseline_selected_by": "validation F1",
        "best_baseline": best_baseline,
        "models": {},
    }
    for model, frame in bootstrap.groupby("model"):
        model_summary = {}
        for metric in (
            "f1",
            "recall",
            "precision",
            "accuracy",
            "delta_f1_vs_best_baseline",
            "delta_recall_vs_best_baseline",
        ):
            values = frame[metric].to_numpy()
            model_summary[metric] = {
                "mean": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
            }
        if model != best_baseline:
            model_summary["probability_f1_above_best_baseline"] = float(
                np.mean(frame["delta_f1_vs_best_baseline"].to_numpy() > 0)
            )
        summary["models"][model] = model_summary
    return bootstrap, summary


def consolidate_results(
    output_dir: Path,
    gnn_result: dict[str, Any] | None,
    gnn_validation: pd.DataFrame | None,
    gnn_test: pd.DataFrame | None,
    baseline_results: dict[str, Any] | None,
    baseline_validation: pd.DataFrame | None,
    baseline_test: pd.DataFrame | None,
    args: argparse.Namespace,
    logger: RunLogger,
) -> None:
    results: dict[str, dict[str, Any]] = {}
    validation_frames = []
    test_frames = []
    if gnn_result is not None:
        results["Ours"] = gnn_result
        validation_frames.append(gnn_validation)
        test_frames.append(gnn_test)
    if baseline_results:
        results.update(baseline_results)
        validation_frames.append(baseline_validation)
        test_frames.append(baseline_test)
    if not results:
        return

    rows = []
    for model, result in results.items():
        rows.append(
            {
                "model": model,
                **{
                    f"validation_{key}": value
                    for key, value in result["validation"].items()
                    if key in {"accuracy", "precision", "recall", "f1", "threshold"}
                },
                **{
                    f"test_{key}": value
                    for key, value in result["test"].items()
                    if key in {"accuracy", "precision", "recall", "f1", "threshold"}
                },
            }
        )
    comparison = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    comparison.to_csv(output_dir / "controlled_model_comparison.csv", index=False)

    if gnn_result is None or not baseline_results:
        return
    eligible_baselines = [
        name for name in baseline_results if name in {"LightGBM", "XGBoost", "CatBoost"}
    ]
    best_baseline = max(
        eligible_baselines,
        key=lambda name: baseline_results[name]["validation"]["f1"],
    )
    validation_predictions = pd.concat(validation_frames, ignore_index=True)
    test_predictions = pd.concat(test_frames, ignore_index=True)
    thresholds = {
        model: float(result["validation"]["threshold"])
        for model, result in results.items()
    }
    bootstrap, bootstrap_summary = cluster_bootstrap_summary(
        test_predictions,
        thresholds,
        best_baseline,
        args.bootstrap_replicates,
        args.seed,
    )
    bootstrap.to_csv(output_dir / "scholar_cluster_bootstrap.csv.gz", index=False)
    (output_dir / "scholar_cluster_bootstrap_summary.json").write_text(
        json.dumps(bootstrap_summary, indent=2), encoding="utf-8"
    )

    ours_test = results["Ours"]["test"]
    baseline_test_metrics = results[best_baseline]["test"]
    headline = {
        "best_baseline_by_validation_f1": best_baseline,
        "ours_test": ours_test,
        "best_baseline_test": baseline_test_metrics,
        "f1_difference_percentage_points": 100
        * (ours_test["f1"] - baseline_test_metrics["f1"]),
        "recall_difference_percentage_points": 100
        * (ours_test["recall"] - baseline_test_metrics["recall"]),
        "pair_manifest": "pair_manifest.csv",
        "split_unit": args.split_unit,
        "vocabulary_source": "observation_window_only",
        "bootstrap": bootstrap_summary["models"]["Ours"],
    }
    (output_dir / "headline_result.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    logger(
        f"Controlled comparison complete. Best validation-selected baseline={best_baseline}; "
        f"test F1 difference={headline['f1_difference_percentage_points']:+.2f} pp."
    )


def load_existing_model_outputs(
    output_dir: Path,
) -> tuple[dict[str, Any] | None, pd.DataFrame | None, pd.DataFrame | None]:
    metrics_path = output_dir / "gnn_metrics.json"
    val_path = output_dir / "gnn_validation_predictions.csv"
    test_path = output_dir / "gnn_test_predictions.csv"
    if not all(path.exists() for path in (metrics_path, val_path, test_path)):
        return None, None, None
    return (
        json.loads(metrics_path.read_text(encoding="utf-8")),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )


def load_existing_baseline_outputs(
    output_dir: Path,
) -> tuple[dict[str, Any] | None, pd.DataFrame | None, pd.DataFrame | None]:
    metrics_path = output_dir / "baseline_metrics.json"
    val_path = output_dir / "baseline_validation_predictions.csv"
    test_path = output_dir / "baseline_test_predictions.csv"
    if not all(path.exists() for path in (metrics_path, val_path, test_path)):
        return None, None, None
    return (
        json.loads(metrics_path.read_text(encoding="utf-8")),
        pd.read_csv(val_path),
        pd.read_csv(test_path),
    )


def main() -> None:
    args = parse_args()
    obs_dir, pred_dir, output_dir = resolve_paths(args)
    logger = RunLogger(output_dir / "controlled_run.log")
    seed_everything(args.seed)
    excluded_topics = load_excluded_topics(args.excluded_topics_file)
    logger(
        f"Starting controlled {args.split_unit}-split run: obs={args.obs_window}y, "
        f"LKN scale={args.lkn_scale}, device request={args.device}."
    )
    dataset = build_controlled_dataset(
        obs_dir,
        pred_dir,
        args.lkn_scale,
        args.seed,
        args.min_scholar_edge_count,
        args.split_unit,
        args.negative_sampling,
        excluded_topics,
        args.topic_id_regex,
        logger,
        require_full_cohort=not args.allow_incomplete_cohort,
    )
    write_dataset_artifacts(
        dataset, output_dir, args, obs_dir, pred_dir, excluded_topics
    )
    if args.prepare_only:
        logger("Preparation-only run completed.")
        return

    device = choose_device(args.device)
    logger(f"Using device={device}.")
    gnn_result = gnn_validation = gnn_test = None
    baseline_results = baseline_validation = baseline_test = None

    if args.skip_gnn:
        gnn_result, gnn_validation, gnn_test = load_existing_model_outputs(output_dir)
    else:
        gnn_result, gnn_validation, gnn_test = train_gnn(
            dataset, output_dir, args, device, logger
        )

    if args.skip_baselines:
        (
            baseline_results,
            baseline_validation,
            baseline_test,
        ) = load_existing_baseline_outputs(output_dir)
    else:
        features = extract_baseline_features(
            dataset,
            output_dir,
            args.n_jobs,
            args.reuse_features,
            args.feature_source,
            args.partial_feature_source,
            logger,
        )
        (
            baseline_results,
            baseline_validation,
            baseline_test,
        ) = fit_baselines(features, output_dir, args, logger)

    consolidate_results(
        output_dir,
        gnn_result,
        gnn_validation,
        gnn_test,
        baseline_results,
        baseline_validation,
        baseline_test,
        args,
        logger,
    )
    logger(f"Controlled {args.split_unit}-split evaluation completed.")


if __name__ == "__main__":
    main()
