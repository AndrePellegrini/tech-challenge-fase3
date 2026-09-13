"""Testes sintéticos da projeção de risco de não atingimento de meta."""
import unittest

import numpy as np
import pandas as pd

from src.modeling.goal_risk_analysis import (
    alert_metrics, concentration_note, project_goal_attainment,
)


def build_frame() -> pd.DataFrame:
    """Quatro municípios com meta e um sem meta, que deve ser descartado."""
    return pd.DataFrame({
        "id_municipio": [1, 2, 3, 4, 5],
        "id_municipio_nome": ["Um", "Dois", "Tres", "Quatro", "SemMeta"],
        "sigla_uf": ["AA", "AA", "BB", "BB", "CC"],
        "alunos": [100, 200, 50, 80, 10],
        # Projeta 40%, 90%, 55% e 70%.
        "probabilidade_media_alfabetizacao": [.40, .90, .55, .70, .60],
        # Observado: 30%, 95%, 80% e 50%.
        "taxa_observada_alfabetizacao": [.30, .95, .80, .50, .60],
        "historico_2023": [35.0, 88.0, np.nan, 65.0, 60.0],
        "meta_2024": [60.0, 80.0, 70.0, 60.0, np.nan],
        "probabilidade_media_risco": [.60, .10, .45, .30, .40],
    })


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.result = project_goal_attainment(build_frame())

    def test_municipality_without_goal_is_dropped(self):
        self.assertEqual(len(self.result), 4)
        self.assertNotIn("SemMeta", set(self.result["id_municipio_nome"]))

    def test_scales_are_reconciled_to_percentage_points(self):
        row = self.result.loc[self.result["id_municipio"].eq(1)].iloc[0]
        self.assertAlmostEqual(row["taxa_prevista_2024"], 40.0)
        self.assertAlmostEqual(row["taxa_observada_2024"], 30.0)
        self.assertAlmostEqual(row["gap_previsto"], 20.0)
        self.assertAlmostEqual(row["gap_observado"], 30.0)

    def test_alert_is_raised_when_projection_falls_short(self):
        alerts = dict(zip(self.result["id_municipio"], self.result["alerta_nao_atingir"]))
        self.assertTrue(alerts[1])    # projeta 40 contra meta 60
        self.assertFalse(alerts[2])   # projeta 90 contra meta 80
        self.assertTrue(alerts[3])    # projeta 55 contra meta 70
        self.assertFalse(alerts[4])   # projeta 70 contra meta 60

    def test_observed_outcome_is_independent_of_the_alert(self):
        observed = dict(zip(self.result["id_municipio"], self.result["nao_atingiu_observado"]))
        self.assertTrue(observed[1])    # observou 30 contra meta 60
        self.assertFalse(observed[2])   # observou 95 contra meta 80
        self.assertFalse(observed[3])   # observou 80 contra meta 70, alerta errou
        self.assertTrue(observed[4])    # observou 50 contra meta 60, alerta errou

    def test_naive_baseline_is_null_without_history(self):
        naive = dict(zip(self.result["id_municipio"], self.result["alerta_naive_2023"]))
        self.assertTrue(bool(naive[1]))       # 2023 em 35 contra meta 60
        self.assertFalse(bool(naive[2]))      # 2023 em 88 contra meta 80
        self.assertTrue(pd.isna(naive[3]))    # sem histórico Gold
        self.assertFalse(bool(naive[4]))      # 2023 em 65 contra meta 60

    def test_ranking_orders_alerts_first_and_by_severity(self):
        head = self.result.head(2)
        self.assertTrue(head["alerta_nao_atingir"].all())
        self.assertGreaterEqual(head.iloc[0]["gap_previsto"], head.iloc[1]["gap_previsto"])
        self.assertListEqual(list(self.result["ranking_risco_meta"]), [1, 2, 3, 4])

    def test_missing_columns_are_rejected(self):
        with self.assertRaises(ValueError):
            project_goal_attainment(build_frame().drop(columns=["meta_2024"]))


class AlertMetricTests(unittest.TestCase):
    def test_confusion_counts_and_derived_rates(self):
        observed = pd.Series([True, True, False, False])
        alert = pd.Series([True, False, True, False])
        metrics = alert_metrics(observed, alert)
        self.assertEqual(metrics["verdadeiro_positivo"], 1)
        self.assertEqual(metrics["falso_negativo"], 1)
        self.assertEqual(metrics["falso_positivo"], 1)
        self.assertEqual(metrics["verdadeiro_negativo"], 1)
        self.assertAlmostEqual(metrics["acuracia"], .5)
        self.assertAlmostEqual(metrics["precision"], .5)
        self.assertAlmostEqual(metrics["recall"], .5)

    def test_no_alert_raised_does_not_divide_by_zero(self):
        metrics = alert_metrics(pd.Series([True, False]), pd.Series([False, False]))
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)


class ConcentrationNoteTests(unittest.TestCase):
    def test_note_appears_when_one_uf_dominates(self):
        frame = project_goal_attainment(build_frame())
        top = frame.loc[frame["sigla_uf"].eq("AA")]
        note = concentration_note(top, frame)
        self.assertIn("AA", note)

    def test_note_is_empty_when_spread_across_ufs(self):
        frame = project_goal_attainment(build_frame())
        self.assertEqual(concentration_note(frame, frame), "")


if __name__ == "__main__":
    unittest.main()
