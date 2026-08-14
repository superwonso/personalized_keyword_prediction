from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_experiment as experiment  # noqa: E402


class SilentLogger:
    def __call__(self, _message: str) -> None:
        pass


class PublicPipelineSmokeTest(unittest.TestCase):
    def test_generic_topic_ids_build_and_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            observation_dir = root / "observation"
            prediction_dir = root / "prediction"
            observation_dir.mkdir()
            prediction_dir.mkdir()

            for index in range(10):
                scholar_id = f"scholar_{index:02d}"
                if index % 2 == 0:
                    observation = "T1 T2\n"
                    prediction = "T1 T3\n"
                else:
                    observation = "T3 T4\n"
                    prediction = "T3 T1\n"
                (observation_dir / f"{scholar_id}.txt").write_text(observation)
                (prediction_dir / f"{scholar_id}.txt").write_text(prediction)

            dataset = experiment.build_controlled_dataset(
                observation_dir,
                prediction_dir,
                lkn_scale=10,
                seed=7,
                min_scholar_edge_count=1,
                split_unit="scholar",
                negative_sampling="random",
                excluded_topics=set(),
                topic_id_regex=None,
                logger=SilentLogger(),
                require_full_cohort=True,
            )

            records = dataset["records_by_split"]["train"][:2]
            batch = experiment.ScholarBatchCollator()(records)
            model = experiment.NTSPredictor(
                num_topics=len(dataset["topic_to_idx"]),
                embedding_dim=8,
                hidden_dim=8,
                out_dim=8,
                node_degrees=dataset["norm_degree"],
                heads=2,
                dropout=0.0,
                pooling="candidate-attention",
                fusion="rich",
            )
            model.eval()
            with torch.no_grad():
                topic_embeddings = model.encode_gkn(dataset["gkn_data"])
                scholar_embeddings, pair_embeddings = model.encode_pairs(
                    batch["lkn_batch"],
                    batch["scholar_positions"],
                    batch["topic_indices"],
                    topic_embeddings,
                )
                logits = model(scholar_embeddings, pair_embeddings)

            self.assertEqual(logits.shape, batch["labels"].shape)
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
