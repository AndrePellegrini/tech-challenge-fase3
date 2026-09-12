import unittest
from unittest.mock import patch

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.modeling.final_validation import (assert_feature_contract, choose_candidate,
                                           fit_candidate, model_candidates, threshold_metrics)
from src.modeling.split import get_modeling_columns


class FinalValidationTests(unittest.TestCase):
    def test_feature_contract_blocks_target_ids_and_proficiency(self):
        for forbidden in ("alfabetizado", "id_aluno", "id_escola", "id_municipio", "proficiencia"):
            with self.assertRaises(ValueError):
                assert_feature_contract([*get_modeling_columns()["X"], forbidden])

    def test_official_features_are_accepted(self):
        assert_feature_contract(list(get_modeling_columns()["X"]))

    def test_threshold_metrics_and_confusion_matrix(self):
        result = threshold_metrics(np.array([0, 0, 1, 1]), np.array([.1, .6, .4, .9]), .5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["precision"], .5)
        with self.assertRaises(ValueError):
            threshold_metrics([0], [.2], 1.0)

    def test_candidates_have_bounded_random_forest_and_valid_parameters(self):
        specs = model_candidates()
        forest = next(x["model"] for x in specs if x["name"] == "random_forest_controlled")
        self.assertLessEqual(forest.n_estimators, 50)
        self.assertIsNotNone(forest.max_depth)
        self.assertGreater(forest.min_samples_leaf, 1)
        for spec in specs:
            if spec["name"] != "dummy_prior":
                self.assertEqual(spec["model"].get_params()["random_state"], 42)

    def test_selection_prefers_stability_within_auc_tolerance(self):
        import pandas as pd
        frame = pd.DataFrame([
            {"model":"a", "validation_roc_auc":.620, "auc_gap":.01, "validation_f1":.70, "training_seconds":2},
            {"model":"b", "validation_roc_auc":.621, "auc_gap":.03, "validation_f1":.72, "training_seconds":1},
        ])
        self.assertEqual(choose_candidate(frame), "a")

    def test_flow_has_no_test_reference(self):
        import inspect
        from src.modeling import final_validation
        source = inspect.getsource(final_validation.main)
        self.assertNotIn('iloc[splits["test"]]', source)
        self.assertNotIn("predict(X_test", source)


if __name__ == "__main__":
    unittest.main()
