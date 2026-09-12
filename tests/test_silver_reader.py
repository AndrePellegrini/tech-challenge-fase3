"""Testes offline da população Silver e sua integração com o builder Gold."""
import unittest
from unittest.mock import patch

import pandas as pd

from src.preprocessing.silver_reader import (
    STUDENT_COLUMNS, SILVER_COLUMNS, SILVER_ALUNOS_PATH,
    build_modeling_population, read_silver_students,
)
from src.preprocessing.build_modeling_dataset import build_modeling_dataset_from_gold
from src.preprocessing.dataset_contract import GOLD_FEATURE_COLUMNS
from src.preprocessing.validate_dataset import validate_dataset
from tests.test_dataset_contract import build_synthetic_dataset


def sample():
    base = dict(ano=2024, id_municipio="3550308", id_escola="e1",
                rede="Municipal", peso_aluno=1.0, alfabetizado="Sim",
                presenca="Presente", preenchimento_caderno="Prova preenchida")
    rows = [dict(base, id_aluno=str(i)) for i in range(8)]
    rows[1].update(rede="Estadual", alfabetizado="Não")
    rows[2]["ano"] = 2023
    rows[3]["presenca"] = "Ausente"
    rows[4]["preenchimento_caderno"] = "Prova não preenchida"
    rows[5]["rede"] = "Privada"
    rows[6]["presenca"] = None
    rows[7]["rede"] = None
    return pd.DataFrame(rows)


class SilverTests(unittest.TestCase):
    def test_filters_and_seven_columns_without_mutation(self):
        data = sample()
        data["proficiencia"] = 750.0
        before = data.copy(deep=True)
        population = build_modeling_population(data)
        self.assertEqual(population.columns.tolist(), STUDENT_COLUMNS)
        self.assertEqual(population["id_aluno"].tolist(), ["0", "1"])
        self.assertEqual(population["alfabetizado"].tolist(), ["Sim", "Não"])
        pd.testing.assert_frame_equal(data, before)

    def test_duplicate_and_null_id_rejected(self):
        for value in ["0", None]:
            data = sample()
            data.loc[1, "id_aluno"] = value
            with self.assertRaises(ValueError):
                build_modeling_population(data)

    def test_excluded_year_does_not_cause_duplicate(self):
        data = sample()
        data.loc[2, "id_aluno"] = "0"
        self.assertEqual(len(build_modeling_population(data)), 2)

    def test_required_filter_columns(self):
        for column in SILVER_COLUMNS:
            with self.assertRaisesRegex(ValueError, "Colunas ausentes"):
                build_modeling_population(sample().drop(columns=column))

    def test_reader_projects_columns_and_fixed_partition(self):
        with patch("src.preprocessing.silver_reader.load_dotenv"), patch(
            "src.preprocessing.silver_reader.pd.read_parquet", return_value=sample()
        ) as read:
            self.assertEqual(len(read_silver_students()), 2)
            read.assert_called_once_with(SILVER_ALUNOS_PATH, columns=SILVER_COLUMNS)

    def test_target_conversion_and_contract_compatibility(self):
        population = build_modeling_population(sample())
        gold = build_synthetic_dataset(20).iloc[[-2]][GOLD_FEATURE_COLUMNS].copy()
        gold["id_municipio"] = "3550308"
        gold["rede"] = "Municipal"
        dataset = build_modeling_dataset_from_gold(population, gold)
        self.assertEqual(dataset["alfabetizado"].tolist(), [1, 0])
        self.assertEqual(dataset["gold_historico_disponivel"].tolist(), [1, 0])
        report = validate_dataset(dataset, strict_counts=False)
        self.assertTrue(report["valido"], report["errors"])

    def test_invalid_target_still_rejected_by_existing_builder(self):
        population = build_modeling_population(sample())
        population.loc[0, "alfabetizado"] = "inválido"
        with self.assertRaisesRegex(ValueError, "alfabetizado"):
            build_modeling_dataset_from_gold(population, pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
