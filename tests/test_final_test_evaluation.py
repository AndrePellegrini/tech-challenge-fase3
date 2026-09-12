import inspect
import unittest

from src.modeling.final_test_evaluation import final_metrics, territorial_analysis


class FinalTestEvaluationTests(unittest.TestCase):
    def test_metrics_report_both_classes(self):
        result=final_metrics([0,0,1,1],[.1,.7,.4,.9])
        self.assertEqual(result["confusion_matrix_true_rows_predicted_columns_0_1"],[[1,1],[1,1]])
        self.assertEqual(result["recall_class_0_risk"],.5)
        self.assertEqual(result["recall_class_1_literate"],.5)

    def test_single_opening_guard_precedes_test_materialization(self):
        from src.modeling import final_test_evaluation
        source=inspect.getsource(final_test_evaluation.main)
        self.assertLess(source.index("METRICS_PATH.exists()"),source.index('iloc[splits["test"]]'))
        self.assertIn("segunda abertura recusada",source)


if __name__=="__main__":
    unittest.main()
