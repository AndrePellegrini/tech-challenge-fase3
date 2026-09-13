"""Testes sintéticos do módulo de interpretabilidade por SHAP.

Nenhum teste carrega o modelo congelado nem o parquet real: os formatos devolvidos pelo
TreeExplainer são simulados, porque é exatamente aí que o módulo pode quebrar entre
versões da biblioteca.
"""
import unittest

import numpy as np
import pandas as pd

from src.modeling.shap_analysis import (aggregate_to_source, compare_with_native,
                                        mean_absolute_shap)

ENCODED = [
    "numeric__idhm",
    "numeric__taxa_alfabetizacao_municipio_2023",
    "categorical__rede_Municipal",
    "categorical__rede_Estadual",
]
SOURCE = ["idhm", "taxa_alfabetizacao_municipio_2023", "rede"]


class MeanAbsoluteShapTests(unittest.TestCase):
    def test_two_dimensional_output(self):
        values = np.array([[1.0, -2.0, .5, -.5], [-3.0, 4.0, .5, -.5]])
        frame = mean_absolute_shap(values, ENCODED)
        self.assertListEqual(list(frame["encoded_feature"]), ENCODED)
        self.assertAlmostEqual(frame.iloc[0]["mean_abs_shap"], 2.0)
        self.assertAlmostEqual(frame.iloc[1]["mean_abs_shap"], 3.0)

    def test_three_dimensional_output_uses_the_positive_class(self):
        """Versões recentes devolvem (n, features, classes); interessa a última."""
        values = np.zeros((2, 4, 2))
        values[:, :, 0] = 99.0          # classe negativa, deve ser ignorada
        values[:, :, 1] = [[1.0, -2.0, .5, -.5], [-3.0, 4.0, .5, -.5]]
        frame = mean_absolute_shap(values, ENCODED)
        self.assertAlmostEqual(frame.iloc[0]["mean_abs_shap"], 2.0)
        self.assertAlmostEqual(frame.iloc[1]["mean_abs_shap"], 3.0)

    def test_column_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            mean_absolute_shap(np.zeros((2, 3)), ENCODED)

    def test_unexpected_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            mean_absolute_shap(np.zeros(4), ENCODED)


class AggregationTests(unittest.TestCase):
    def test_one_hot_columns_are_summed_back_into_the_source_feature(self):
        encoded = pd.DataFrame({
            "encoded_feature": ENCODED,
            "mean_abs_shap": [.10, .40, .15, .05],
        })
        frame = aggregate_to_source(encoded, SOURCE)
        values = dict(zip(frame["feature"], frame["mean_abs_shap"]))
        self.assertAlmostEqual(values["rede"], .20)          # 0,15 + 0,05
        self.assertAlmostEqual(values["taxa_alfabetizacao_municipio_2023"], .40)
        self.assertAlmostEqual(values["idhm"], .10)

    def test_result_is_ordered_by_contribution(self):
        encoded = pd.DataFrame({
            "encoded_feature": ENCODED,
            "mean_abs_shap": [.10, .40, .15, .05],
        })
        frame = aggregate_to_source(encoded, SOURCE)
        self.assertListEqual(
            list(frame["feature"]),
            ["taxa_alfabetizacao_municipio_2023", "rede", "idhm"],
        )


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.shap_frame = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "mean_abs_shap": [.5, .3, .2],
        })
        self.native = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "importance": [.2, .3, .5],   # ordem exatamente invertida
        })

    def test_ranks_and_displacement_are_computed(self):
        merged = compare_with_native(self.shap_frame, self.native)
        row = merged.loc[merged["feature"].eq("a")].iloc[0]
        self.assertEqual(row["rank_shap"], 1)
        self.assertEqual(row["rank_nativa"], 3)
        self.assertEqual(row["deslocamento"], 2)

    def test_output_is_ordered_by_shap_rank(self):
        merged = compare_with_native(self.shap_frame, self.native)
        self.assertListEqual(list(merged["feature"]), ["a", "b", "c"])

    def test_inverted_rankings_give_negative_spearman(self):
        merged = compare_with_native(self.shap_frame, self.native)
        spearman = merged[["rank_shap", "rank_nativa"]].corr(method="spearman").iloc[0, 1]
        self.assertAlmostEqual(spearman, -1.0)


if __name__ == "__main__":
    unittest.main()
