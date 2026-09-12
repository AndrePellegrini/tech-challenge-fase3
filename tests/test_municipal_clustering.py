import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.modeling.municipal_clustering import (FEATURES, build_municipal_table,
                                               build_pipeline, validate_features)


class MunicipalClusteringTests(unittest.TestCase):
    def sample(self):
        rows=[]
        extras={"gap_para_meta_municipio_2024","idhm_educacao","idhm_renda",
                "idhm_longevidade","total_alunos_municipio_2023"}
        for municipality in range(6):
            for network,value in (("Municipal",0),("Estadual",10)):
                for student in range(2+value//10):
                    row={"id_municipio":str(municipality),"id_municipio_nome":f"M{municipality}",
                         "sigla_uf":"SP","rede":network,
                         "id_aluno":f"{municipality}-{network}-{student}"}
                    for j,column in enumerate(set(FEATURES)|extras):
                        row[column]=municipality*10+j+value
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_one_row_per_municipality_and_weighted_networks(self):
        table,_=build_municipal_table(self.sample())
        self.assertEqual(len(table),6)
        self.assertFalse(table.id_municipio.duplicated().any())

    def test_forbidden_features_rejected(self):
        for forbidden in ("alfabetizado","id_municipio","rede","probabilidade_media_risco"):
            with self.assertRaises(ValueError):
                validate_features([*FEATURES,forbidden])

    def test_imputation_scaling_kmeans_silhouette_and_clusters(self):
        X=self.sample().groupby("id_municipio")[FEATURES].mean()
        X.iloc[0,0]=np.nan
        pipeline=build_pipeline(2)
        labels=pipeline.fit_predict(X)
        transformed=pipeline[:-1].transform(X)
        self.assertFalse(np.isnan(transformed).any())
        self.assertTrue(np.allclose(transformed.mean(axis=0),0,atol=1e-7))
        self.assertEqual(len(set(labels)),2)
        self.assertGreater(silhouette_score(transformed,labels),0)

    def test_reproducible_assignments(self):
        X=self.sample().groupby("id_municipio")[FEATURES].mean()
        np.testing.assert_array_equal(
            build_pipeline(2,42).fit_predict(X),build_pipeline(2,42).fit_predict(X))


if __name__=="__main__":
    unittest.main()
