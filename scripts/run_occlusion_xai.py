#!/usr/bin/env python3
"""Checkpoint-faithful node-occlusion explanations for controlled NTS runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from textwrap import wrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_experiment as controlled  # noqa: E402


COLORS = {
    "ink": "#202124",
    "muted": "#667085",
    "supportive": "#2878B5",
    "suppressive": "#C94C4C",
    "neutral": "#D9DEE5",
    "edge": "#A7AFBA",
    "candidate": "#E3A638",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        default=str(repo_root / "outputs"),
    )
    parser.add_argument("--windows", default="1,3,5")
    parser.add_argument("--lkn-scale", type=int, default=3000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--plot-nodes", type=int, default=15)
    parser.add_argument("--graph-nodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--topic-map",
        default=None,
        help="Optional CSV used to replace topic IDs with display names.",
    )
    parser.add_argument("--topic-id-column", default="topic_id")
    parser.add_argument("--topic-name-column", default="topic_name")
    return parser.parse_args()


def load_topic_names(
    path_value: str | None, topic_id_column: str, topic_name_column: str
) -> dict[str, str]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path,
        usecols=[topic_id_column, topic_name_column],
        dtype=str,
    ).dropna()
    frame = frame.drop_duplicates(topic_id_column, keep="first")
    return dict(zip(frame[topic_id_column], frame[topic_name_column]))


def load_dataset_and_model(
    run_dir: Path,
    lkn_scale: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], controlled.NTSPredictor, dict[str, Any]]:
    run_config = json.loads(
        (run_dir / "controlled_run_config.json").read_text(encoding="utf-8")
    )
    dataset = controlled.build_controlled_dataset(
        Path(run_config["lkn_obs_dir"]),
        Path(run_config["lkn_pred_dir"]),
        lkn_scale,
        seed,
        int(run_config["min_scholar_edge_count"]),
        run_config.get("split_unit", "pair"),
        run_config.get("negative_sampling", "random"),
        set(run_config.get("excluded_topics", [])),
        run_config.get("topic_id_regex") or None,
        lambda _message: None,
        require_full_cohort=run_config.get(
            "require_full_contributing_cohort", False
        ),
    )
    checkpoint = torch.load(
        run_dir / "controlled_gnn.pth",
        map_location=device,
        weights_only=False,
    )
    model_config = checkpoint["model_config"]
    model = controlled.NTSPredictor(
        len(dataset["topic_to_idx"]),
        model_config["embedding_dim"],
        model_config["hidden_dim"],
        model_config["out_dim"],
        dataset["norm_degree"],
        model_config["heads"],
        model_config["dropout"],
        model_config.get("pooling", "mean"),
        model_config.get("fusion", "product"),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return dataset, model, checkpoint


def scholar_scores(
    predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for scholar_id, group in predictions.groupby("scholar_id", sort=True):
        metrics = controlled.binary_metrics(
            group["label"].to_numpy(),
            group["probability"].to_numpy(),
            threshold,
        )
        labels = group["label"].to_numpy(dtype=float)
        probabilities = group["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "scholar_id": str(scholar_id),
                "test_pairs": int(len(group)),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mean_absolute_error": float(
                    np.mean(np.abs(labels - probabilities))
                ),
            }
        )
    return pd.DataFrame(rows)


def select_representative_pair(
    predictions: pd.DataFrame,
    scholar_id: str,
    threshold: float,
    cohort: str,
) -> pd.Series:
    group = predictions[
        predictions["scholar_id"].astype(str) == str(scholar_id)
    ].copy()
    group["prediction"] = (group["probability"] > threshold).astype(int)
    group["error"] = np.abs(group["label"] - group["probability"])
    group["correct"] = group["prediction"] == group["label"]
    if cohort == "top":
        candidates = group[(group["label"] == 1) & group["correct"]]
        if candidates.empty:
            candidates = group[group["correct"]]
        if candidates.empty:
            candidates = group
        return candidates.sort_values(
            ["probability", "error", "topic_id"],
            ascending=[False, True, True],
        ).iloc[0]

    candidates = group[~group["correct"]]
    if candidates.empty:
        candidates = group
    return candidates.sort_values(
        ["error", "probability", "topic_id"],
        ascending=[False, False, True],
    ).iloc[0]


def remove_lkn_node(data: Data, removed_node: int) -> Data:
    keep = torch.ones(data.num_nodes, dtype=torch.bool)
    keep[removed_node] = False
    old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long)
    old_to_new[keep] = torch.arange(int(keep.sum()))
    edge_keep = keep[data.edge_index[0]] & keep[data.edge_index[1]]
    edge_index = data.edge_index[:, edge_keep]
    if edge_index.numel():
        edge_index = old_to_new[edge_index]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(
        edge_index=edge_index,
        num_nodes=int(keep.sum()),
        global_indices=data.global_indices[keep].clone(),
    )


def score_occlusions(
    model: controlled.NTSPredictor,
    all_topic_embeddings: torch.Tensor,
    lkn_data: Data,
    candidate_idx: int,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        original_batch = Batch.from_data_list([lkn_data]).to(device)
        original_scholar, original_topic = model.encode_pairs(
            original_batch,
            torch.zeros(1, dtype=torch.long, device=device),
            torch.tensor([candidate_idx], dtype=torch.long, device=device),
            all_topic_embeddings,
        )
        base_probability = float(
            torch.sigmoid(model(original_scholar, original_topic)).item()
        )

        occluded_graphs = [
            remove_lkn_node(lkn_data, node_index)
            for node_index in range(lkn_data.num_nodes)
        ]
        occluded_batch = Batch.from_data_list(occluded_graphs).to(device)
        scholar_positions = torch.arange(
            len(occluded_graphs), dtype=torch.long, device=device
        )
        topic_indices = torch.full(
            (len(occluded_graphs),),
            candidate_idx,
            dtype=torch.long,
            device=device,
        )
        occluded_scholar, occluded_topic = model.encode_pairs(
            occluded_batch,
            scholar_positions,
            topic_indices,
            all_topic_embeddings,
        )
        occluded_probabilities = torch.sigmoid(
            model(occluded_scholar, occluded_topic)
        ).cpu().numpy()
    return base_probability, occluded_probabilities


def write_case_bar_plot(
    frame: pd.DataFrame,
    output_path: Path,
    max_nodes: int,
) -> None:
    selected = frame.nlargest(max_nodes, "absolute_effect").sort_values(
        "normalized_effect_percent"
    )
    colors = np.where(
        selected["normalized_effect_percent"] >= 0,
        COLORS["supportive"],
        COLORS["suppressive"],
    )
    height = max(4.8, 0.39 * len(selected) + 1.6)
    fig, ax = plt.subplots(figsize=(9.2, height))
    y = np.arange(len(selected))
    ax.barh(
        y,
        selected["normalized_effect_percent"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    labels = [
        "\n".join(wrap(str(value), width=32))
        for value in selected["topic_name"]
    ]
    ax.set_yticks(y, labels)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xlabel("Occlusion effect relative to the largest absolute effect (%)")
    ax.grid(axis="x", color="#E4E7EC", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [
        Line2D([0], [0], color=COLORS["supportive"], lw=7, label="Supportive"),
        Line2D([0], [0], color=COLORS["suppressive"], lw=7, label="Suppressive"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def connected_explanation_nodes(
    lkn_graph: nx.Graph,
    importance: dict[int, float],
    top_n: int,
    max_total: int = 20,
) -> tuple[list[int], set[int]]:
    ranked = sorted(
        importance,
        key=lambda node: abs(importance[node]),
        reverse=True,
    )[:top_n]
    included = set(ranked)
    connectors: set[int] = set()
    if not ranked:
        return [], connectors
    anchor = ranked[0]
    for node in ranked[1:]:
        if len(included) >= max_total:
            break
        try:
            path = nx.shortest_path(lkn_graph, anchor, node)
        except nx.NetworkXNoPath:
            continue
        for connector in path[1:-1]:
            if len(included) >= max_total:
                break
            if connector not in included:
                included.add(connector)
                connectors.add(connector)
    return sorted(included), connectors


def write_case_graph(
    frame: pd.DataFrame,
    lkn_data: Data,
    output_path: Path,
    max_nodes: int,
    seed: int,
) -> int:
    lkn_graph = nx.Graph()
    lkn_graph.add_nodes_from(range(lkn_data.num_nodes))
    lkn_graph.add_edges_from(lkn_data.edge_index.t().tolist())
    importance = dict(
        zip(frame["local_node_index"].astype(int), frame["effect"].astype(float))
    )
    included, connectors = connected_explanation_nodes(
        lkn_graph,
        importance,
        max_nodes,
        max_total=max_nodes + 4,
    )
    subgraph = lkn_graph.subgraph(included).copy()
    if not included:
        return 0

    if nx.is_connected(subgraph) and subgraph.number_of_nodes() > 2:
        positions = nx.kamada_kawai_layout(subgraph)
    else:
        positions = nx.spring_layout(
            subgraph,
            seed=seed,
            k=max(1.35 / math.sqrt(max(len(included), 1)), 0.42),
            iterations=300,
        )
    effects = np.asarray([importance[node] for node in included])
    max_abs = max(float(np.max(np.abs(effects))), 1e-9)
    explanation_nodes = [node for node in included if node not in connectors]
    connector_nodes = [node for node in included if node in connectors]
    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    nx.draw_networkx_edges(
        subgraph,
        positions,
        ax=ax,
        width=1.0,
        alpha=0.55,
        edge_color=COLORS["edge"],
    )
    collection = nx.draw_networkx_nodes(
        subgraph,
        positions,
        nodelist=explanation_nodes,
        ax=ax,
        node_color=[importance[node] for node in explanation_nodes],
        node_size=[
            850 + 1900 * abs(importance[node]) / max_abs
            for node in explanation_nodes
        ],
        cmap="coolwarm_r",
        vmin=-max_abs,
        vmax=max_abs,
        edgecolors="white",
        linewidths=1.0,
    )
    if connector_nodes:
        nx.draw_networkx_nodes(
            subgraph,
            positions,
            nodelist=connector_nodes,
            ax=ax,
            node_color=COLORS["neutral"],
            node_size=500,
            edgecolors="white",
            linewidths=1.0,
        )
    names = dict(
        zip(frame["local_node_index"].astype(int), frame["topic_name"].astype(str))
    )
    labels = {
        node: "\n".join(wrap(names[node], width=16))
        for node in included
        if node not in connectors
    }
    nx.draw_networkx_labels(
        subgraph,
        positions,
        labels=labels,
        ax=ax,
        font_size=9.3,
        font_color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
    )
    colorbar = fig.colorbar(collection, ax=ax, fraction=0.044, pad=0.025)
    colorbar.set_label("Probability change after node occlusion")
    ax.set_aspect("equal")
    ax.margins(0.20)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return int(subgraph.number_of_edges())


def run_window(
    args: argparse.Namespace,
    obs_window: int,
    device: torch.device,
    mesh_names: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = (
        Path(args.evaluation_root)
        / f"obs{obs_window}y_lkn{args.lkn_scale}"
    ).resolve()
    output_dir = (
        Path(args.evaluation_root)
        / "controlled_xai"
        / f"obs{obs_window}y_lkn{args.lkn_scale}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, model, checkpoint = load_dataset_and_model(
        run_dir, args.lkn_scale, args.seed, device
    )
    predictions = pd.read_csv(run_dir / "gnn_test_predictions.csv")
    predictions["scholar_id"] = predictions["scholar_id"].astype(str)
    threshold = float(checkpoint["threshold"])
    score_frame = scholar_scores(predictions, threshold)

    top = score_frame.sort_values(
        ["f1", "accuracy", "mean_absolute_error", "scholar_id"],
        ascending=[False, False, True, True],
    ).head(args.top_k)
    worst = score_frame.sort_values(
        ["f1", "accuracy", "mean_absolute_error", "scholar_id"],
        ascending=[True, True, False, True],
    ).head(args.top_k)
    selected = pd.concat(
        [
            top.assign(cohort="top"),
            worst.assign(cohort="worst"),
        ],
        ignore_index=True,
    )

    summary_rows = []
    importance_frames = []
    exemplar_candidates = []
    scholar_records = dataset["scholars"]
    idx_to_topic = dataset["idx_to_topic"]
    gkn_data = dataset["gkn_data"]
    with torch.no_grad():
        all_topic_embeddings = model.encode_gkn(gkn_data.to(device))

    for cohort in ("top", "worst"):
        cohort_rows = selected[selected["cohort"] == cohort].reset_index(drop=True)
        for rank, score_row in cohort_rows.iterrows():
            scholar_id = str(score_row["scholar_id"])
            pair = select_representative_pair(
                predictions, scholar_id, threshold, cohort
            )
            scholar_record = scholar_records[scholar_id]
            lkn_data = scholar_record["lkn_data"]
            candidate_idx = int(dataset["topic_to_idx"][str(pair["topic_id"])])
            base_probability, occluded_probabilities = score_occlusions(
                model,
                all_topic_embeddings,
                lkn_data,
                candidate_idx,
                device,
            )
            archived_difference = abs(base_probability - float(pair["probability"]))
            if archived_difference > 1e-5:
                raise RuntimeError(
                    "Checkpoint probability differs from the archived score by "
                    f"{archived_difference:.3g} for {pair['pair_id']}."
                )

            effects = base_probability - occluded_probabilities
            max_abs = max(float(np.max(np.abs(effects))), 1e-12)
            topic_ids = [
                idx_to_topic[int(index)]
                for index in lkn_data.global_indices.tolist()
            ]
            case_id = f"{cohort}_{rank + 1:02d}_{scholar_id}"
            case_frame = pd.DataFrame(
                {
                    "observation_window_years": obs_window,
                    "case_id": case_id,
                    "cohort": cohort,
                    "cohort_rank": rank + 1,
                    "scholar_id": scholar_id,
                    "candidate_topic_id": str(pair["topic_id"]),
                    "candidate_topic_name": mesh_names.get(
                        str(pair["topic_id"]), str(pair["topic_id"])
                    ),
                    "candidate_label": int(pair["label"]),
                    "base_probability": base_probability,
                    "local_node_index": np.arange(lkn_data.num_nodes),
                    "topic_id": topic_ids,
                    "topic_name": [
                        mesh_names.get(topic_id, topic_id)
                        for topic_id in topic_ids
                    ],
                    "occluded_probability": occluded_probabilities,
                    "effect": effects,
                    "absolute_effect": np.abs(effects),
                    "normalized_effect_percent": effects / max_abs * 100.0,
                }
            )
            case_path = output_dir / case_id
            case_path.mkdir(parents=True, exist_ok=True)
            case_frame.sort_values(
                "absolute_effect", ascending=False
            ).to_csv(case_path / "node_occlusion_importance.csv", index=False)
            write_case_bar_plot(
                case_frame,
                case_path / "node_occlusion_top_effects.png",
                args.plot_nodes,
            )
            graph_edges = write_case_graph(
                case_frame,
                lkn_data,
                case_path / "lkn_explanation_subgraph.png",
                args.graph_nodes,
                args.seed + obs_window * 100 + rank,
            )

            supportive = int(np.sum(effects > 0))
            suppressive = int(np.sum(effects < 0))
            summary_rows.append(
                {
                    "observation_window_years": obs_window,
                    "case_id": case_id,
                    "cohort": cohort,
                    "cohort_rank": rank + 1,
                    "scholar_id": scholar_id,
                    "scholar_test_pairs": int(score_row["test_pairs"]),
                    "scholar_accuracy": float(score_row["accuracy"]),
                    "scholar_precision": float(score_row["precision"]),
                    "scholar_recall": float(score_row["recall"]),
                    "scholar_f1": float(score_row["f1"]),
                    "candidate_topic_id": str(pair["topic_id"]),
                    "candidate_topic_name": mesh_names.get(
                        str(pair["topic_id"]), str(pair["topic_id"])
                    ),
                    "candidate_label": int(pair["label"]),
                    "predicted_label": int(pair["probability"] > threshold),
                    "base_probability": base_probability,
                    "threshold": threshold,
                    "lkn_nodes": int(lkn_data.num_nodes),
                    "lkn_edges": int(lkn_data.edge_index.size(1) // 2),
                    "supportive_nodes": supportive,
                    "suppressive_nodes": suppressive,
                    "largest_absolute_effect": max_abs,
                    "top5_absolute_effect_share": float(
                        case_frame.nlargest(5, "absolute_effect")[
                            "absolute_effect"
                        ].sum()
                        / max(case_frame["absolute_effect"].sum(), 1e-12)
                    ),
                    "display_subgraph_edges": graph_edges,
                }
            )
            importance_frames.append(case_frame)
            exemplar_candidates.append(
                {
                    "case_id": case_id,
                    "case_frame": case_frame,
                    "lkn_data": lkn_data,
                    "graph_edges": graph_edges,
                    "sign_balance": min(supportive, suppressive),
                    "effect": max_abs,
                }
            )

    summary = pd.DataFrame(summary_rows)
    importance = pd.concat(importance_frames, ignore_index=True)
    summary.to_csv(output_dir / "xai_case_summary.csv", index=False)
    importance.to_csv(output_dir / "xai_node_importance.csv", index=False)
    score_frame.to_csv(output_dir / "scholar_test_metrics.csv", index=False)

    exemplar = max(
        exemplar_candidates,
        key=lambda row: (
            row["graph_edges"] > 0,
            row["sign_balance"] > 0,
            row["graph_edges"],
            row["effect"],
        ),
    )
    write_case_bar_plot(
        exemplar["case_frame"],
        output_dir / "representative_node_effects.png",
        args.plot_nodes,
    )
    write_case_graph(
        exemplar["case_frame"],
        exemplar["lkn_data"],
        output_dir / "representative_lkn_subgraph.png",
        args.graph_nodes,
        args.seed + obs_window * 1000,
    )
    (output_dir / "representative_case.json").write_text(
        json.dumps(
            {
                "case_id": exemplar["case_id"],
                "selection_rule": (
                    "Prefer a connected display subgraph with both supportive and "
                    "suppressive effects, then more displayed edges and a larger effect."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary, importance


def main() -> None:
    args = parse_args()
    device = controlled.choose_device(args.device)
    mesh_names = load_topic_names(
        args.topic_map, args.topic_id_column, args.topic_name_column
    )
    windows = [int(value) for value in args.windows.split(",") if value.strip()]
    all_summaries = []
    all_importance = []
    for obs_window in windows:
        print(f"[RUN] controlled XAI: {obs_window}y", flush=True)
        summary, importance = run_window(
            args, obs_window, device, mesh_names
        )
        all_summaries.append(summary)
        all_importance.append(importance)

    output_root = Path(args.evaluation_root) / "controlled_xai"
    pd.concat(all_summaries, ignore_index=True).to_csv(
        output_root / "xai_case_summary_all_windows.csv", index=False
    )
    pd.concat(all_importance, ignore_index=True).to_csv(
        output_root / "xai_node_importance_all_windows.csv", index=False
    )
    print(f"[DONE] {output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
