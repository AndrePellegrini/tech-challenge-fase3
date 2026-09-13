"""Testes das primitivas de visualização e dos módulos que passaram a usá-las.

Os gráficos são gravados em diretório temporário: nenhum teste escreve em `images/`.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.visualization.plots import (grouped_bar, histogram_with_marker,
                                     horizontal_bar, overlapping_histograms,
                                     precision_recall_curves, roc_curves, save_figure)

RANDOM_STATE = 42


def synthetic_scores(size: int = 200):
    """Rótulos e scores correlacionados, para as curvas terem forma definida."""
    generator = np.random.default_rng(RANDOM_STATE)
    y = generator.integers(0, 2, size)
    score = np.clip(y * .35 + generator.random(size) * .5, 0, 1)
    return y, score


class PlotPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def test_roc_curve_is_written(self):
        y, score = synthetic_scores()
        path = roc_curves({"modelo": (y, score)}, self.directory / "roc.png", title="ROC")
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_roc_creates_missing_directories(self):
        y, score = synthetic_scores()
        path = roc_curves({"m": (y, score)}, self.directory / "a" / "b" / "roc.png", title="ROC")
        self.assertTrue(path.exists())

    def test_roc_accepts_several_curves(self):
        y, score = synthetic_scores()
        path = roc_curves(
            {"a": (y, score), "b": (y, 1 - score)}, self.directory / "roc2.png", title="ROC",
        )
        self.assertTrue(path.exists())

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            roc_curves({}, self.directory / "x.png", title="vazio")
        with self.assertRaises(ValueError):
            precision_recall_curves({}, self.directory / "y.png", title="vazio")
        with self.assertRaises(ValueError):
            overlapping_histograms({}, self.directory / "z.png", title="vazio", xlabel="x")
        with self.assertRaises(ValueError):
            horizontal_bar([], [], self.directory / "w.png", title="vazio", xlabel="x")

    def test_precision_recall_is_written(self):
        y, score = synthetic_scores()
        path = precision_recall_curves(
            {"classe 1": (y, score)}, self.directory / "pr.png", title="PR",
        )
        self.assertTrue(path.exists())

    def test_overlapping_histograms_is_written(self):
        generator = np.random.default_rng(RANDOM_STATE)
        path = overlapping_histograms(
            {"a": generator.random(300), "b": generator.random(300)},
            self.directory / "hist.png", title="Distribuições", xlabel="valor",
        )
        self.assertTrue(path.exists())

    def test_histogram_with_marker_is_written(self):
        generator = np.random.default_rng(RANDOM_STATE)
        path = histogram_with_marker(
            generator.normal(size=300), self.directory / "marker.png",
            title="Gap", xlabel="pontos percentuais", marker=0,
        )
        self.assertTrue(path.exists())

    def test_horizontal_bar_is_written(self):
        path = horizontal_bar(
            ["feature_a", "feature_b"], [.3, .7], self.directory / "bar.png",
            title="Importância", xlabel="peso",
        )
        self.assertTrue(path.exists())

    def test_grouped_bar_is_written(self):
        frame = pd.DataFrame({"f1": [.5, .6], "roc_auc": [.61, .63]}, index=["arvore", "floresta"])
        path = grouped_bar(
            frame, ["f1", "roc_auc"], self.directory / "grouped.png",
            title="Comparação", ylabel="Métrica",
        )
        self.assertTrue(path.exists())

    def test_save_figure_closes_the_current_figure(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([0, 1], [0, 1])
        save_figure(self.directory / "plain.png")
        self.assertEqual(plt.get_fignums(), [])


class RefactoredCallerTests(unittest.TestCase):
    """Garante que os módulos de modelagem continuam produzindo seus gráficos."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

    def test_final_test_evaluation_still_writes_every_figure(self):
        from src.modeling import final_test_evaluation as module
        y_test, p_test = synthetic_scores()
        y_validation, p_validation = synthetic_scores(150)
        with patch.object(module, "IMAGES", self.directory):
            module.plot_final(y_test, p_test, y_validation, p_validation)
        produced = {path.name for path in self.directory.glob("*.png")}
        self.assertSetEqual(produced, {
            "roc_final_test.png",
            "precision_recall_literate_test.png",
            "precision_recall_risk_test.png",
            "risk_distribution_validation_vs_test.png",
        })

    def test_final_validation_plots_curves_from_fitted_pipelines(self):
        from src.modeling import final_validation as module
        generator = np.random.default_rng(RANDOM_STATE)
        X = pd.DataFrame({"a": generator.random(120)})
        y = pd.Series(generator.integers(0, 2, 120))
        model = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE).fit(X, y)
        with patch.object(module, "IMAGES", self.directory):
            module.plot_curves({"dummy": model}, X, y)
        produced = {path.name for path in self.directory.glob("*.png")}
        self.assertSetEqual(produced, {"roc_validation.png", "precision_recall_validation.png"})

    def test_train_baselines_still_writes_comparison_and_importance(self):
        from src.modeling import train_baselines as module
        comparison = pd.DataFrame({
            "model": ["dummy", "tree"], "f1": [.4, .6],
            "recall": [.5, .7], "roc_auc": [.5, .63],
        })
        importance = pd.DataFrame({"feature": ["a", "b"], "importance": [.6, .4]})
        with patch.object(module, "IMAGE_DIR", self.directory):
            module._plot_results(comparison, importance)
        produced = {path.name for path in self.directory.glob("*.png")}
        self.assertSetEqual(produced, {
            "06_baseline_comparison.png", "07_feature_importance.png",
        })

    def test_train_baselines_skips_importance_when_empty(self):
        from src.modeling import train_baselines as module
        comparison = pd.DataFrame({
            "model": ["dummy"], "f1": [.4], "recall": [.5], "roc_auc": [.5],
        })
        empty = pd.DataFrame({"feature": [], "importance": []})
        with patch.object(module, "IMAGE_DIR", self.directory):
            module._plot_results(comparison, empty)
        produced = {path.name for path in self.directory.glob("*.png")}
        self.assertSetEqual(produced, {"06_baseline_comparison.png"})


if __name__ == "__main__":
    unittest.main()
