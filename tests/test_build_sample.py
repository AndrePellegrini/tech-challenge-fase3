"""Testes sintéticos do gerador da amostra anonimizada.

Nenhum teste lê o parquet real nem escreve em `data/`.
"""
import unittest

import numpy as np
import pandas as pd

from src.preprocessing.build_sample import (NO_GOLD_STRATUM, build_sample,
                                            representativeness, select_municipalities,
                                            surrogate)

RANDOM_STATE = 42


def build_universe(municipalities: int = 120, per_municipality: int = 40) -> pd.DataFrame:
    """Universo sintético com UFs, redes e um bloco sem histórico Gold.

    Os últimos dez municípios simulam ausência de correspondência na camada Gold:
    `sigla_uf` nulo e flag zerada, como acontece no dataset real.
    """
    generator = np.random.default_rng(RANDOM_STATE)
    ufs = ["SP", "BA", "MG", "RS"]
    rows = []
    for index in range(municipalities):
        no_gold = index >= municipalities - 10
        rate = generator.uniform(.3, .85)
        for student in range(per_municipality):
            rows.append({
                "id_aluno": f"{index:04d}{student:04d}",
                "id_escola": f"esc{index:05d}",
                "id_municipio": f"{3500000 + index}",
                "sigla_uf": None if no_gold else ufs[index % len(ufs)],
                "rede": "Municipal" if index % 5 else "Estadual",
                "idhm": np.nan if no_gold else .55 + rate * .3,
                "gold_historico_disponivel": 0 if no_gold else 1,
                "alfabetizado": int(generator.random() < rate),
            })
    return pd.DataFrame(rows)


class SurrogateTests(unittest.TestCase):
    def test_identifiers_are_replaced_by_sequential_codes(self):
        series = pd.Series(["11020232", "11020233"])
        result = surrogate(series, "a")
        self.assertListEqual(list(result), ["a0001", "a0002"])

    def test_no_original_value_survives(self):
        series = pd.Series(["11020232", "11020233"])
        result = surrogate(series, "a")
        self.assertFalse(set(result) & set(series))

    def test_repeated_values_share_the_surrogate(self):
        """Preserva a relacao aluno-escola: a mesma escola recebe o mesmo codigo."""
        series = pd.Series(["esc1", "esc1", "esc2", "esc1"])
        result = surrogate(series, "e")
        self.assertEqual(result.iloc[0], result.iloc[1])
        self.assertEqual(result.iloc[0], result.iloc[3])
        self.assertNotEqual(result.iloc[0], result.iloc[2])
        self.assertEqual(result.nunique(), 2)

    def test_distinct_inputs_give_distinct_surrogates(self):
        result = surrogate(pd.Series([str(i) for i in range(500)]), "a")
        self.assertEqual(result.nunique(), 500)

    def test_width_grows_with_the_number_of_values(self):
        result = surrogate(pd.Series([str(i) for i in range(12_000)]), "a")
        self.assertEqual(result.iloc[0], "a00001")

    def test_nulls_are_preserved(self):
        result = surrogate(pd.Series(["123", None]), "a")
        self.assertTrue(pd.isna(result.iloc[1]))

    def test_result_is_reproducible(self):
        series = pd.Series(["x", "y", "x", "z"])
        self.assertListEqual(list(surrogate(series, "a")), list(surrogate(series, "a")))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = build_universe()

    def test_municipalities_without_gold_form_their_own_stratum(self):
        """Sem estrato próprio, o groupby descartaria as linhas de sigla_uf nula."""
        chosen = select_municipalities(self.data, target_rows=1_200,
                                       random_state=RANDOM_STATE)
        subset = self.data.loc[self.data["id_municipio"].isin(chosen)]
        self.assertGreater(int((subset["gold_historico_disponivel"] == 0).sum()), 0)

    def test_every_stratum_is_represented(self):
        chosen = select_municipalities(self.data, target_rows=1_200,
                                       random_state=RANDOM_STATE)
        subset = self.data.loc[self.data["id_municipio"].isin(chosen)].copy()
        subset["estrato"] = subset["sigla_uf"].fillna(NO_GOLD_STRATUM)
        expected = set(self.data["sigla_uf"].fillna(NO_GOLD_STRATUM))
        self.assertSetEqual(set(subset["estrato"]), expected)

    def test_bigger_target_selects_more_municipalities(self):
        small = select_municipalities(self.data, target_rows=800, random_state=RANDOM_STATE)
        large = select_municipalities(self.data, target_rows=3_200, random_state=RANDOM_STATE)
        self.assertGreater(len(large), len(small))

    def test_selection_is_reproducible(self):
        first = select_municipalities(self.data, target_rows=1_200, random_state=RANDOM_STATE)
        second = select_municipalities(self.data, target_rows=1_200, random_state=RANDOM_STATE)
        self.assertListEqual(first, second)


class BuildSampleTests(unittest.TestCase):
    def setUp(self):
        self.data = build_universe()
        self.sample = build_sample(self.data, target_rows=1_200, random_state=RANDOM_STATE)

    def test_municipalities_come_whole(self):
        """O split é agrupado por município; um município partido o inviabilizaria."""
        for municipality, group in self.sample.groupby("id_municipio"):
            original = int(self.data["id_municipio"].eq(municipality).sum())
            self.assertEqual(len(group), original)

    def test_identifiers_are_replaced_but_municipality_is_preserved(self):
        self.assertFalse(set(self.sample["id_aluno"]) & set(self.data["id_aluno"]))
        self.assertFalse(set(self.sample["id_escola"]) & set(self.data["id_escola"]))
        self.assertTrue(set(self.sample["id_municipio"]) <= set(self.data["id_municipio"]))

    def test_school_cardinality_survives_the_substitution(self):
        """O surrogate nao pode fundir nem multiplicar escolas."""
        origem = self.data.loc[
            self.data["id_municipio"].isin(self.sample["id_municipio"].unique())
        ]
        self.assertEqual(self.sample["id_escola"].nunique(),
                         origem["id_escola"].nunique())

    def test_student_identifier_stays_unique(self):
        self.assertTrue(self.sample["id_aluno"].is_unique)

    def test_schema_is_preserved(self):
        self.assertListEqual(list(self.sample.columns), list(self.data.columns))

    def test_sample_is_smaller_than_the_universe(self):
        self.assertLess(len(self.sample), len(self.data))


class RepresentativenessTests(unittest.TestCase):
    def test_differences_are_reported_for_the_key_dimensions(self):
        data = build_universe()
        sample = build_sample(data, target_rows=1_600, random_state=RANDOM_STATE)
        report = representativeness(data, sample)
        self.assertIn("universo", report)
        self.assertIn("amostra", report)
        self.assertSetEqual(
            set(report["diferencas"]),
            {"taxa_alfabetizacao", "share_municipal", "cobertura_gold", "idhm_medio"},
        )
        self.assertLess(abs(report["diferencas"]["taxa_alfabetizacao"]), .25)


if __name__ == "__main__":
    unittest.main()
