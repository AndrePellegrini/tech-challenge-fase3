"""Testes sintéticos da validação cruzada agrupada.

Nenhum teste lê o parquet real nem acessa o S3: o dataset é construído em memória,
com municípios suficientes para o GroupKFold.
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.modeling.cross_validation import (cross_validate_candidate, development_frame,
                                           finalist_specs, separation_verdict, summarize)

RANDOM_STATE = 42
FEATURES = ["taxa_alfabetizacao_municipio_2023", "idhm", "rede", "sigla_uf"]


def build_frame(municipalities: int = 30, per_municipality: int = 20) -> pd.DataFrame:
    """Alunos agrupados em municípios, com sinal contextual fraco mas presente."""
    generator = np.random.default_rng(RANDOM_STATE)
    rows = []
    for index in range(municipalities):
        rate = generator.uniform(.25, .85)
        for _ in range(per_municipality):
            rows.append({
                "id_municipio": f"{3500000 + index}",
                "taxa_alfabetizacao_municipio_2023": rate * 100,
                "idhm": .55 + rate * .3,
                "rede": "Municipal" if index % 4 else "Estadual",
                "sigla_uf": ["SP", "BA", "MG"][index % 3],
                "alfabetizado": int(generator.random() < rate),
            })
    return pd.DataFrame(rows)


def tree_spec(name: str = "arvore") -> dict:
    return {
        "name": name,
        "model": DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                        random_state=RANDOM_STATE),
        "scale": False,
        "note": "sintético",
    }


class FinalistSelectionTests(unittest.TestCase):
    def test_returns_specs_in_requested_order(self):
        names = ("decision_tree_d5_l100", "logistic_baseline")
        specs = finalist_specs(names)
        self.assertEqual([spec["name"] for spec in specs], list(names))

    def test_unknown_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            finalist_specs(("modelo_inexistente",))

    def test_default_finalists_all_exist(self):
        specs = finalist_specs()
        self.assertEqual(len(specs), 4)
        self.assertIn("random_forest_controlled", [spec["name"] for spec in specs])


class DevelopmentFrameTests(unittest.TestCase):
    def setUp(self):
        self.data = build_frame()

    def test_test_rows_are_excluded(self):
        splits = {
            "train": np.arange(0, 300),
            "validation": np.arange(300, 450),
            "test": np.arange(450, 600),
        }
        frame = development_frame(self.data, splits)
        self.assertEqual(len(frame), 450)
        self.assertTrue((frame.index < 450).all())

    def test_overlap_with_test_is_rejected(self):
        splits = {
            "train": np.arange(0, 300),
            "validation": np.arange(290, 450),   # invade o teste
            "test": np.arange(295, 600),
        }
        with self.assertRaises(ValueError):
            development_frame(self.data, splits)

    def test_rows_come_back_in_original_order(self):
        splits = {
            "train": np.arange(100, 200),
            "validation": np.arange(0, 100),
            "test": np.arange(200, 600),
        }
        frame = development_frame(self.data, splits)
        self.assertListEqual(list(frame.index), list(range(200)))


class CrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.data = build_frame()
        self.rows = cross_validate_candidate(tree_spec(), FEATURES, self.data, n_splits=5)

    def test_one_row_per_fold(self):
        self.assertEqual(len(self.rows), 5)
        self.assertListEqual([r["fold"] for r in self.rows], [1, 2, 3, 4, 5])

    def test_every_fold_produces_a_valid_auc(self):
        for row in self.rows:
            self.assertGreaterEqual(row["roc_auc"], 0.0)
            self.assertLessEqual(row["roc_auc"], 1.0)

    def test_every_student_is_validated_exactly_once(self):
        total = sum(row["validation_rows"] for row in self.rows)
        self.assertEqual(total, len(self.data))

    def test_folds_never_share_a_municipality(self):
        # cross_validate_candidate levanta ValueError se houver sobreposição;
        # aqui confirmamos que a soma de municípios validados fecha o total.
        validated = sum(row["validation_municipalities"] for row in self.rows)
        self.assertEqual(validated, self.data["id_municipio"].nunique())

    def test_result_is_reproducible(self):
        again = cross_validate_candidate(tree_spec(), FEATURES, self.data, n_splits=5)
        self.assertListEqual([r["roc_auc"] for r in self.rows],
                             [r["roc_auc"] for r in again])


class SummaryTests(unittest.TestCase):
    def test_mean_and_spread_are_computed_per_model(self):
        results = pd.DataFrame({
            "model": ["a", "a", "b", "b"],
            "fold": [1, 2, 1, 2],
            "roc_auc": [.60, .70, .50, .52],
            "training_seconds": [1.0, 1.0, 2.0, 2.0],
        })
        summary = summarize(results)
        self.assertListEqual(list(summary["model"]), ["a", "b"])
        self.assertAlmostEqual(summary.iloc[0]["roc_auc_mean"], .65)
        self.assertAlmostEqual(summary.iloc[0]["roc_auc_min"], .60)
        self.assertAlmostEqual(summary.iloc[0]["roc_auc_max"], .70)
        self.assertAlmostEqual(summary.iloc[1]["seconds_total"], 4.0)

    def test_verdict_detects_a_consistent_advantage(self):
        summary = pd.DataFrame({
            "model": ["a", "b"],
            "roc_auc_mean": [.70, .60],
            "roc_auc_std": [.001, .001],
        })
        self.assertIn("supera", separation_verdict(summary))

    def test_verdict_detects_an_inconclusive_difference(self):
        summary = pd.DataFrame({
            "model": ["a", "b"],
            "roc_auc_mean": [.6409, .6407],
            "roc_auc_std": [.02, .02],
        })
        verdict = separation_verdict(summary)
        self.assertIn("não supera", verdict)
        self.assertIn("indistinguíveis", verdict)

    def test_single_candidate_is_not_compared(self):
        summary = pd.DataFrame({"model": ["a"], "roc_auc_mean": [.7], "roc_auc_std": [.01]})
        self.assertIn("indisponível", separation_verdict(summary))


if __name__ == "__main__":
    unittest.main()
