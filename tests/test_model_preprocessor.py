import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import classification_metrics
from src.modeling.split import get_modeling_columns
from src.preprocessing.model_preprocessor import build_model_preprocessor


def synthetic_X() -> pd.DataFrame:
    columns = get_modeling_columns()
    data = {}
    for column in columns["numeric"]:
        data[column] = [1.0, np.nan, 3.0, 4.0, 5.0, 6.0]
    data["gold_historico_disponivel"] = [1, 0, 1, 1, 0, 1]
    data["rede"] = ["Municipal", "Estadual"] * 3
    data["sigla_uf"] = ["SP", "RJ", "SP", "MG", "RJ", "MG"]
    data["atingiu_meta_municipio_2024"] = [True, False, None, True, False, True]
    return pd.DataFrame(data).loc[:, columns["X"]]


class ModelPreprocessorTests(unittest.TestCase):
    def test_contract_does_not_include_target_or_forbidden_columns(self):
        columns = get_modeling_columns()
        forbidden = {"alfabetizado", "ano", "id_aluno", "id_municipio", "id_escola", "peso_aluno"}
        self.assertTrue(forbidden.isdisjoint(columns["X"]))

    def test_imputation_and_sparse_encoding(self):
        transformed = build_model_preprocessor().fit_transform(synthetic_X())
        self.assertEqual(transformed.shape[0], 6)
        self.assertFalse(np.isnan(transformed.data).any())

    def test_unseen_categories_are_accepted(self):
        X = synthetic_X()
        transformer = build_model_preprocessor().fit(X.iloc[:5])
        unseen = X.iloc[[5]].copy()
        unseen.loc[:, "rede"] = "Outra"
        unseen.loc[:, "sigla_uf"] = "ZZ"
        self.assertEqual(transformer.transform(unseen).shape[0], 1)

    def test_baseline_pipeline_and_metrics(self):
        X = synthetic_X()
        y = pd.Series([0, 1, 0, 1, 0, 1])
        pipeline = Pipeline([
            ("preprocessing", build_model_preprocessor()),
            ("model", DummyClassifier(strategy="prior")),
        ]).fit(X, y)
        metrics = classification_metrics(pipeline, X, y)
        self.assertEqual(
            set(metrics), {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"},
        )
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
