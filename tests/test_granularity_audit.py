import unittest

import numpy as np
import pandas as pd

from src.modeling.granularity_audit import (
    context_stats,
    empirical_profile_bound,
    identify_profiles,
    official_feature_columns,
    profile_target_stats,
    profiles_per_context,
)


def synthetic_dataset() -> pd.DataFrame:
    rows = 6
    data = {}
    for column in official_feature_columns():
        if column == "rede":
            data[column] = ["Municipal"] * 4 + ["Estadual"] * 2
        elif column == "sigla_uf":
            data[column] = ["SP"] * rows
        elif column == "atingiu_meta_municipio_2024":
            data[column] = [True] * rows
        elif column == "gold_historico_disponivel":
            data[column] = [1] * rows
        else:
            data[column] = [1.0, 1.0, np.nan, np.nan, 2.0, 3.0]
    frame = pd.DataFrame(data)
    frame["id_municipio"] = ["1"] * 4 + ["2"] * 2
    frame["alfabetizado"] = [0, 1, 1, 1, 0, 0]
    return frame


class GranularityAuditTests(unittest.TestCase):
    def test_official_X_excludes_target(self):
        self.assertNotIn("alfabetizado", official_feature_columns())

    def test_equal_profiles_and_nan_are_consistent(self):
        data = synthetic_dataset()
        ids = identify_profiles(data[official_feature_columns()])
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[2], ids[3])
        self.assertNotEqual(ids[0], ids[2])

    def test_mixed_profile_and_majority_bound(self):
        data = synthetic_dataset()
        ids = identify_profiles(data[official_feature_columns()])
        stats = profile_target_stats(ids, data["alfabetizado"])
        mixed = stats.loc[stats["type"].eq("mixed")]
        self.assertEqual(len(mixed), 1)
        bound = empirical_profile_bound(stats)
        self.assertEqual(bound["minimum_conflict_errors_within_partition"], 1)
        self.assertAlmostEqual(bound["majority_class_accuracy_within_partition"], 5 / 6)

    def test_municipality_network_stats(self):
        data = synthetic_dataset()
        _, summary = context_stats(data, ["id_municipio", "rede"])
        self.assertEqual(summary["groups"], 2)
        self.assertEqual(summary["mixed_groups"], 1)
        self.assertAlmostEqual(summary["mixed_group_student_share"], 4 / 6)

    def test_multiple_profiles_inside_context(self):
        data = synthetic_dataset()
        ids = identify_profiles(data[official_feature_columns()])
        _, summary = profiles_per_context(data, ids)
        self.assertEqual(summary["groups_with_multiple_profiles"], 2)
        self.assertEqual(summary["max_profiles_in_context"], 2)
        self.assertEqual(summary["profiles_shared_by_multiple_contexts"], 0)


if __name__ == "__main__":
    unittest.main()
