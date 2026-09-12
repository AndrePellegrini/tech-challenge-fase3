import unittest

import pandas as pd

from src.modeling.censo_escolar_poc import (
    SELECTED, normalize_school_id, official_X_guard, profile_summary,
    safe_school_join,
)


def students():
    data = pd.DataFrame({
        "id_aluno": ["a", "b", "c"], "id_escola": [123, 123, 456],
        "id_municipio": ["1", "1", "2"], "rede": ["Municipal"] * 3,
        "alfabetizado": [0, 1, 1],
    })
    for column in official_X_guard():
        if column not in data:
            data[column] = "x" if column in {"sigla_uf", "atingiu_meta_municipio_2024"} else 1.0
    return data


def schools(duplicate=False):
    ids = ["00000123", "00000456"] + (["00000123"] if duplicate else [])
    data = pd.DataFrame({"id_escola_join": ids})
    for column in SELECTED:
        data[column] = range(len(data))
    return data


class CensoEscolarPocTests(unittest.TestCase):
    def test_key_normalization_is_explicit(self):
        values, audit = normalize_school_id(pd.Series([123, "00000456", None]))
        self.assertEqual(values.iloc[:2].tolist(), ["00000123", "00000456"])
        self.assertEqual(audit["invalid_or_null"], 1)

    def test_many_to_one_join_preserves_students_and_coverage(self):
        joined, audit = safe_school_join(students(), schools())
        self.assertEqual(len(joined), 3)
        self.assertEqual(audit["row_expansion"], 0)
        self.assertEqual(audit["student_id_duplicates_after"], 0)
        self.assertEqual(audit["student_row_coverage"], 1.0)

    def test_expansion_risk_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expansão"):
            safe_school_join(students(), schools(duplicate=True))

    def test_profile_count_and_missing_are_consistent(self):
        data = students()
        data.loc[0, "idhm"] = None
        data.loc[1, "idhm"] = None
        summary = profile_summary(data, official_X_guard())
        self.assertGreaterEqual(summary["profiles"], 2)

    def test_official_X_and_test_remain_out_of_scope(self):
        self.assertTrue(set(official_X_guard()).isdisjoint(SELECTED))


if __name__ == "__main__":
    unittest.main()
