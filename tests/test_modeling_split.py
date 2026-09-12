"""Testes sintéticos; nenhum acesso à AWS/GCP ou treinamento."""
import unittest

import numpy as np
import pandas as pd

from src.modeling.split import (
    get_modeling_columns, get_modeling_data, split_by_municipality, validate_split,
)
from tests.test_dataset_contract import build_synthetic_dataset


def fixture():
    data = build_synthetic_dataset(80)
    data["id_municipio"] = [str(3500000 + i // 4) for i in range(len(data))]
    data["alfabetizado"] = np.tile([0, 1], 40).astype("int8")
    return data


class ModelingSplitTests(unittest.TestCase):
    def test_zero_overlap_and_all_rows_once(self):
        data = fixture()
        splits = split_by_municipality(data)
        report = validate_split(data, splits)
        self.assertTrue(report["valid"], report["errors"])
        self.assertTrue(report["rows_preserved"])
        self.assertEqual(sum(report["group_overlap"].values()), 0)
        self.assertEqual(sum(report["student_overlap"].values()), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(data)))
        self.assertEqual([v["municipalities"] for v in report["summary"].values()], [14, 3, 3])

    def test_reproducibility_and_row_order_independence(self):
        data = fixture()
        first = split_by_municipality(data)
        second = split_by_municipality(data)
        shuffled = data.sample(frac=1, random_state=7)
        third = split_by_municipality(shuffled)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(set(data.iloc[first[name]].id_municipio), set(shuffled.iloc[third[name]].id_municipio))

    def test_target_does_not_select_split(self):
        data = fixture()
        first = split_by_municipality(data)
        data["alfabetizado"] = 1 - data.alfabetizado
        second = split_by_municipality(data)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_X_y_groups_weight_separated_aligned_and_unmodified(self):
        data = fixture()
        data.index = np.arange(1000, 1080)
        before = data.copy(deep=True)
        parts = get_modeling_data(data)
        self.assertEqual(parts["X"].shape, (80, 16))
        self.assertEqual(parts["X"].columns.tolist(), get_modeling_columns()["X"])
        self.assertFalse(set(parts["X"].columns) & set(get_modeling_columns()["excluded_from_X"]))
        self.assertEqual(set(get_modeling_columns()["X"]) | set(get_modeling_columns()["excluded_from_X"]), set(data.columns))
        for name, column in [("y", "alfabetizado"), ("groups", "id_municipio"), ("sample_weight", "peso_aluno")]:
            pd.testing.assert_series_equal(parts[name], data[column])
        self.assertTrue(parts["X"].isna().any().any())
        pd.testing.assert_frame_equal(data, before)

    def test_positions_work_with_duplicate_pandas_index(self):
        data = fixture()
        data.index = [0] * len(data)
        self.assertTrue(validate_split(data, split_by_municipality(data))["valid"])

    def test_overlap_rejected(self):
        data = fixture()
        splits = split_by_municipality(data)
        a, b = splits["train"][0], splits["test"][0]
        splits["train"][0], splits["test"][0] = b, a
        report = validate_split(data, splits)
        self.assertFalse(report["valid"])
        self.assertTrue(report["rows_preserved"])
        self.assertGreater(report["group_overlap"]["train/test"], 0)

    def test_missing_or_repeated_rows_rejected(self):
        data = fixture()
        for repeat in [False, True]:
            splits = split_by_municipality(data)
            if repeat:
                splits["train"][0] = splits["train"][1]
            else:
                splits["train"] = splits["train"][1:]
            self.assertFalse(validate_split(data, splits)["valid"])

    def test_invalid_positions_or_names_rejected(self):
        data = fixture()
        for invalid in [np.array([-1]), np.array([80]), np.array([1.5]), np.array([[1]])]:
            splits = split_by_municipality(data)
            splits["train"] = invalid
            self.assertFalse(validate_split(data, splits)["valid"])
        self.assertFalse(validate_split(data, {})["valid"])

    def test_invalid_group_or_duplicate_student_rejected(self):
        for column, value in [("id_municipio", None), ("id_aluno", "aluno_0000001")]:
            data = fixture()
            data.loc[0, column] = value
            with self.assertRaises(ValueError):
                split_by_municipality(data)
        with self.assertRaises(ValueError):
            split_by_municipality(fixture().iloc[:8])

    def test_unbalanced_class_reported(self):
        data = fixture()
        splits = split_by_municipality(data)
        data.loc[splits["test"], "alfabetizado"] = 1
        report = validate_split(data, splits)
        self.assertFalse(report["valid"])
        self.assertTrue(any("classe 1" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
