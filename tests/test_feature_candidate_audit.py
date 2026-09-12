import unittest

import numpy as np
import pandas as pd

from src.modeling.feature_candidate_audit import (
    BUSINESS_COLUMNS, COLUMN_CLASSIFICATION, build_summary, candidate_matrix,
    detect_constant, official_X_unchanged, proficiency_statistics,
)


def sample() -> pd.DataFrame:
    rows = []
    for index, (target, proficiency) in enumerate([("Não", 700), ("Sim", 743), ("Sim", 800)]):
        rows.append({
            "ano": 2024, "id_municipio": "1", "id_municipio_nome": "A",
            "id_escola": "e", "id_aluno": str(index), "caderno": "A", "serie": "2º ano",
            "rede": "Municipal", "presenca": "Presente",
            "preenchimento_caderno": "Prova preenchida", "alfabetizado": target,
            "proficiencia": proficiency, "peso_aluno": 1.0,
        })
    return pd.DataFrame(rows, columns=BUSINESS_COLUMNS)


class FeatureCandidateAuditTests(unittest.TestCase):
    def test_column_classification_complete(self):
        self.assertEqual(set(BUSINESS_COLUMNS + ["_ingestion_ts", "_source", "_table", "_processed_ts", "_layer"]), set(COLUMN_CLASSIFICATION))

    def test_constant_detection(self):
        self.assertTrue(detect_constant(pd.Series(["x", "x", "x"])))
        self.assertFalse(detect_constant(pd.Series(["x", "y"])))

    def test_proficiency_threshold_reconstructs_target(self):
        stats = proficiency_statistics(pd.Series([700, 743, 800, np.nan]), pd.Series([0, 1, 1, 0]))
        self.assertEqual(stats["threshold_mismatches"], 0)
        self.assertEqual(stats["complete_rows"], 3)
        self.assertEqual(stats["threshold_accuracy"], 1.0)

    def test_candidate_matrix_has_required_contract(self):
        summary = build_summary(sample())
        matrix = candidate_matrix(summary)
        self.assertEqual(set(matrix["recomendacao"]), {"NÃO USAR", "TESTAR", "TESTAR como sample_weight", "INVESTIGAR MAIS"})
        self.assertIn("proficiencia", matrix["feature"].tolist())

    def test_official_X_is_unchanged_and_test_not_accessed(self):
        before = official_X_unchanged()
        summary = build_summary(sample())
        self.assertEqual(summary["official_X_before"], before)
        self.assertFalse(summary["official_X_changed"])
        self.assertFalse(summary["test_accessed"])


if __name__ == "__main__":
    unittest.main()
