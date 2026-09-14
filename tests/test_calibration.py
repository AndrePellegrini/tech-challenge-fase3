"""Testes sinteticos da avaliacao de calibracao municipal.

Nenhum teste le o artefato real nem escreve em reports/.
"""
import unittest

import numpy as np
import pandas as pd

from src.evaluation.calibration import (calibration_metrics, calibration_table,
                                        load_municipal_table, verdict)

OBSERVED = "taxa_observada_alfabetizacao"
PREDICTED = "probabilidade_media_alfabetizacao"
WEIGHT = "alunos"


def build_frame(offset: float = 0.0, rows: int = 200) -> pd.DataFrame:
    """Municipios com previsao espalhada e observado deslocado por `offset`."""
    generator = np.random.default_rng(42)
    predicted = np.linspace(.2, .9, rows)
    observed = np.clip(predicted + offset, 0, 1)
    return pd.DataFrame({
        PREDICTED: predicted,
        OBSERVED: observed,
        WEIGHT: generator.integers(20, 500, rows),
    })


class MetricTests(unittest.TestCase):
    def test_perfect_calibration_has_no_error(self):
        metrics = calibration_metrics(build_frame())
        self.assertAlmostEqual(metrics["vies"], 0.0, places=6)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=6)
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=6)
        self.assertAlmostEqual(metrics["correlacao_pearson"], 1.0, places=6)

    def test_constant_offset_shows_up_as_bias(self):
        metrics = calibration_metrics(build_frame(offset=-.05))
        self.assertAlmostEqual(metrics["vies"], .05, places=4)
        self.assertAlmostEqual(metrics["mae"], .05, places=4)

    def test_bias_sign_follows_the_convention(self):
        """Vies positivo significa que o modelo preve acima do observado."""
        acima = calibration_metrics(build_frame(offset=-.10))["vies"]
        abaixo = calibration_metrics(build_frame(offset=.10))["vies"]
        self.assertGreater(acima, 0)
        self.assertLess(abaixo, 0)

    def test_shrinkage_ratio_detects_compressed_predictions(self):
        frame = build_frame()
        # Comprime as previsoes em direcao a media, mantendo o observado.
        media = frame[PREDICTED].mean()
        frame[PREDICTED] = media + (frame[PREDICTED] - media) * .5
        metrics = calibration_metrics(frame)
        self.assertAlmostEqual(metrics["razao_de_encolhimento"], .5, places=4)

    def test_weighted_mae_follows_the_weights(self):
        frame = build_frame()
        frame[OBSERVED] = frame[PREDICTED]
        frame.loc[0, OBSERVED] = frame.loc[0, PREDICTED] - .5   # municipio errado
        frame.loc[0, WEIGHT] = 1                                 # e minusculo
        metrics = calibration_metrics(frame)
        self.assertLess(metrics["mae_ponderado_por_aluno"], metrics["mae"])

    def test_missing_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            calibration_metrics(build_frame().drop(columns=[OBSERVED]))


class TableTests(unittest.TestCase):
    def test_one_row_per_bin_and_bins_are_numbered_from_one(self):
        table = calibration_table(build_frame(), bins=10)
        self.assertEqual(len(table), 10)
        self.assertListEqual(list(table["faixa"]), list(range(1, 11)))

    def test_every_municipality_falls_into_exactly_one_bin(self):
        frame = build_frame()
        table = calibration_table(frame, bins=10)
        self.assertEqual(int(table["municipios"].sum()), len(frame))

    def test_predictions_increase_across_bins(self):
        table = calibration_table(build_frame(), bins=10)
        self.assertTrue(table["previsto"].is_monotonic_increasing)

    def test_deviation_is_zero_under_perfect_calibration(self):
        table = calibration_table(build_frame(), bins=10)
        self.assertLess(table["desvio"].abs().max(), 1e-9)


class VerdictTests(unittest.TestCase):
    def test_small_deviation_reads_as_well_calibrated(self):
        frame = build_frame()
        text = verdict(calibration_metrics(frame), calibration_table(frame))
        self.assertIn("bem calibrada", text)

    def test_large_deviation_reads_as_ordering_only(self):
        frame = build_frame(offset=-.20)
        text = verdict(calibration_metrics(frame), calibration_table(frame))
        self.assertIn("ordenamento", text)


class LoadingTests(unittest.TestCase):
    def test_missing_source_raises_a_clear_error(self):
        from pathlib import Path
        with self.assertRaises(FileNotFoundError):
            load_municipal_table(Path("nao_existe_calibracao.csv"))


if __name__ == "__main__":
    unittest.main()
