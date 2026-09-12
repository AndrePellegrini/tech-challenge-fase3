import inspect
import unittest

import numpy as np

from src.modeling.threshold_semantics_audit import (corrected_municipal_analysis,
                                                     risk_probability,
                                                     semantic_threshold_metrics)


class ThresholdSemanticsAuditTests(unittest.TestCase):
    def test_class_zero_metrics_and_risk_errors(self):
        result = semantic_threshold_metrics([0, 0, 1, 1], [.1, .6, .4, .9], .5)
        self.assertEqual(result["confusion_matrix_true_rows_predicted_columns_0_1"], [[1, 1], [1, 1]])
        self.assertEqual(result["precision_class_0_risk"], .5)
        self.assertEqual(result["recall_class_0_risk"], .5)
        self.assertEqual(result["true_risk_predicted_literate"], 1)

    def test_probability_semantics(self):
        prob_alfabetizado = np.array([.2, .8])
        np.testing.assert_allclose(risk_probability(prob_alfabetizado), [.8, .2])
        with self.assertRaises(ValueError):
            risk_probability([1.1])

    def test_higher_literate_threshold_predicts_more_risk(self):
        y, score = [0, 0, 1, 1], [.2, .4, .6, .8]
        low = semantic_threshold_metrics(y, score, .3)
        high = semantic_threshold_metrics(y, score, .7)
        self.assertGreater(high["recall_class_0_risk"], low["recall_class_0_risk"])
        self.assertGreater(high["false_risk_alerts_literate_predicted_risk"], low["false_risk_alerts_literate_predicted_risk"])

    def test_main_never_materializes_test(self):
        from src.modeling import threshold_semantics_audit
        source = inspect.getsource(threshold_semantics_audit.main)
        self.assertNotIn('iloc[splits["test"]]', source)
        self.assertNotIn("X_test", source)


if __name__ == "__main__":
    unittest.main()
