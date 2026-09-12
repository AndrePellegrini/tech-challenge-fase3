"""
Testes do contrato e da validação do dataset de modelagem.

Utilizam dados sintéticos no mesmo schema do dataset real, sem dependência
de BigQuery ou AWS. Podem ser executados com:

    python -m tests.test_dataset_contract
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.dataset_contract import (
    GOLD_FEATURE_COLUMNS,
    HISTORY_FEATURE_COLUMNS,
    COLUMN_GROUPS,
    get_expected_columns,
)
from src.preprocessing.validate_dataset import (
    validate_dataset,
)


def build_synthetic_dataset(rows: int = 1_000) -> pd.DataFrame:
    """
    Constrói um dataset sintético válido no schema do contrato.

    Os primeiros 5% dos registros simulam alunos sem histórico
    municipal de 2023 (flag 0 e features históricas nulas).
    """

    rng = np.random.default_rng(42)

    missing_rows = rows // 20

    flag = np.ones(rows, dtype="int8")
    flag[:missing_rows] = 0

    taxa = rng.uniform(20, 95, rows).astype("float64")
    media = rng.uniform(650, 800, rows).astype("float64")

    taxa[:missing_rows] = np.nan
    media[:missing_rows] = np.nan

    dataset = pd.DataFrame(
        {
            "ano": np.full(rows, 2024),
            "id_aluno": [
                f"aluno_{i:07d}" for i in range(rows)
            ],
            "id_municipio": rng.choice(
                ["3550308", "3304557", "2927408"],
                rows,
            ),
            "id_escola": rng.choice(
                ["11000001", "22000002", "33000003"],
                rows,
            ),
            "rede": rng.choice(
                ["Municipal", "Estadual"],
                rows,
                p=[0.87, 0.13],
            ),
            "peso_aluno": rng.uniform(0.5, 2.0, rows),
            "alfabetizado": rng.choice(
                [0, 1],
                rows,
                p=[0.4, 0.6],
            ).astype("int8"),
            "taxa_alfabetizacao_municipio_2023": taxa,
            "media_portugues_municipio_2023": media,
            "gold_historico_disponivel": flag,
        }
    )

    for column in GOLD_FEATURE_COLUMNS:
        if column not in dataset:
            dataset[column] = 0.75
    dataset["id_municipio_nome"] = "Município sintético"
    dataset["sigla_uf"] = "SP"
    dataset["sigla_uf_nome"] = "São Paulo"
    dataset["meta_alfabetizacao_municipio_2024"] = 70.0
    dataset["gap_para_meta_municipio_2024"] = 70.0 - taxa
    dataset["atingiu_meta_municipio_2024"] = pd.Series(taxa >= 70, dtype=object)
    dataset.loc[dataset["gold_historico_disponivel"].eq(0), GOLD_FEATURE_COLUMNS] = np.nan

    # Missingness parcial legítima em alunos com correspondência Gold.
    optional = [c for c in GOLD_FEATURE_COLUMNS if c not in HISTORY_FEATURE_COLUMNS]
    dataset.loc[dataset.index[-1:], optional] = np.nan
    return dataset[get_expected_columns()]


def test_dataset_valido_passa_sem_strict():
    dataset = build_synthetic_dataset()

    report = validate_dataset(
        dataset,
        strict_counts=False,
    )

    assert report["valido"], report["errors"]


def test_contagens_exatas_exigidas_no_modo_strict():
    dataset = build_synthetic_dataset()

    report = validate_dataset(
        dataset,
        strict_counts=True,
    )

    # O sintético não possui 1.851.828 linhas,
    # portanto o modo estrito deve falhar.
    assert not report["valido"]


def test_duplicidade_de_id_aluno_reprovada():
    dataset = build_synthetic_dataset()

    dataset.loc[1, "id_aluno"] = dataset.loc[0, "id_aluno"]

    report = validate_dataset(
        dataset,
        strict_counts=False,
    )

    assert not report["valido"]
    assert report["checks"]["unicidade_id_aluno"]


def test_target_nulo_reprovado():
    dataset = build_synthetic_dataset()

    dataset["alfabetizado"] = (
        dataset["alfabetizado"].astype("float64")
    )
    dataset.loc[0, "alfabetizado"] = np.nan

    report = validate_dataset(
        dataset,
        strict_counts=False,
    )

    assert not report["valido"]
    assert report["checks"]["target"]


def test_target_fora_do_dominio_reprovado():
    dataset = build_synthetic_dataset()

    dataset.loc[0, "alfabetizado"] = 2

    report = validate_dataset(
        dataset,
        strict_counts=False,
    )

    assert not report["valido"]
    assert report["checks"]["target"]


def test_incoerencia_flag_e_nulos_reprovada():
    dataset = build_synthetic_dataset()

    # Registro com histórico disponível, porém feature nula.
    dataset.loc[
        dataset.index[-1],
        "taxa_alfabetizacao_municipio_2023",
    ] = np.nan

    report = validate_dataset(
        dataset,
        strict_counts=False,
    )

    assert not report["valido"]
    assert report["checks"]["gold_historico"]


def test_schema_incompleto_reprovado():
    dataset = build_synthetic_dataset()

    dataset = dataset.drop(
        columns=["peso_aluno"]
    )

    report = validate_dataset(
        dataset,
        strict_counts=False,
    )

    assert not report["valido"]
    assert report["checks"]["schema"]


def test_coluna_inesperada_reprovada():
    dataset = build_synthetic_dataset()
    dataset["extra"] = 0
    assert validate_dataset(dataset, False)["checks"]["schema"]


def test_coluna_com_nome_duplicado_reprovada():
    dataset = build_synthetic_dataset()
    dataset = pd.concat([dataset, dataset[["alfabetizado"]]], axis=1)
    assert validate_dataset(dataset, False)["checks"]["schema"]


def test_cada_feature_gold_obrigatoria_no_schema():
    dataset = build_synthetic_dataset()
    for column in GOLD_FEATURE_COLUMNS:
        assert not validate_dataset(dataset.drop(columns=column), False)["valido"]


def test_flag_invalida_reprovada_sem_excecao_ou_coercao():
    for value in [None, pd.NA, np.nan, 2, -1, 0.5, "1", "inválida"]:
        dataset = build_synthetic_dataset()
        dataset["gold_historico_disponivel"] = dataset["gold_historico_disponivel"].astype(object)
        dataset.loc[0, "gold_historico_disponivel"] = value
        assert validate_dataset(dataset, False)["checks"]["gold_historico"]


def test_sem_gold_nao_pode_ter_valores_do_join():
    for column in GOLD_FEATURE_COLUMNS:
        dataset = build_synthetic_dataset()
        dataset[column] = dataset[column].astype(object)
        dataset.loc[0, column] = 1
        assert validate_dataset(dataset, False)["checks"]["gold_historico"], column


def test_com_gold_exige_nucleo_completo():
    for column in HISTORY_FEATURE_COLUMNS:
        dataset = build_synthetic_dataset()
        dataset.loc[dataset.index[-1], column] = None
        assert validate_dataset(dataset, False)["checks"]["gold_historico"], column


def test_nulos_parciais_com_gold_permitidos_sem_modificar_dados():
    dataset = build_synthetic_dataset()
    before = dataset.copy(deep=True)
    assert dataset.iloc[-1]["gold_historico_disponivel"] == 1
    assert dataset.iloc[-1][COLUMN_GROUPS["metas_contexto"]["colunas"]].isna().all()
    assert validate_dataset(dataset, False)["valido"]
    pd.testing.assert_frame_equal(dataset, before)


def test_id_aluno_nulo_reprovado():
    dataset = build_synthetic_dataset()
    dataset.loc[0, "id_aluno"] = None
    assert validate_dataset(dataset, False)["checks"]["unicidade_id_aluno"]


def test_schema_independe_da_ordem():
    dataset = build_synthetic_dataset()
    assert validate_dataset(dataset[dataset.columns[::-1]], False)["valido"]


def test_contagens_strict_isoladas_com_referencias_sinteticas():
    from unittest.mock import patch
    from src.preprocessing import validate_dataset as validator

    dataset = build_synthetic_dataset(20)
    with patch.multiple(validator, EXPECTED_ROWS=20,
                        EXPECTED_HISTORY_AVAILABLE=19, EXPECTED_HISTORY_MISSING=1):
        assert validator.validate_dataset(dataset)["valido"]
        # Mesmo total de linhas e missingness coerente, mas cobertura divergente.
        dataset.loc[1, "gold_historico_disponivel"] = 0
        dataset.loc[1, GOLD_FEATURE_COLUMNS] = np.nan
        report = validator.validate_dataset(dataset)
        assert not report["checks"]["shape"]
        assert len(report["checks"]["gold_historico"]) == 2
        assert validator.validate_dataset(dataset, False)["valido"]


def test_referencias_e_destino_gold():
    from src.preprocessing import dataset_contract as contract
    from src.preprocessing.validate_dataset import DATASET_PATH

    assert contract.EXPECTED_ROWS == 1_851_828
    assert contract.EXPECTED_HISTORY_AVAILABLE == 1_816_270
    assert contract.EXPECTED_HISTORY_MISSING == 35_558
    assert contract.EXPECTED_ROWS == contract.EXPECTED_HISTORY_AVAILABLE + contract.EXPECTED_HISTORY_MISSING
    assert len(set(get_expected_columns())) == contract.EXPECTED_COLUMNS == 24
    assert DATASET_PATH.name == "modeling_dataset_2024_gold.parquet"


def test_builders_gold_produzem_schema_do_contrato():
    from src.preprocessing.gold_features import build_consolidated_gold_features
    from src.preprocessing.build_modeling_dataset import build_modeling_dataset_from_gold

    indicador = pd.DataFrame({
        "ano": [2023], "id_municipio": ["3550308"],
        "id_municipio_nome": ["São Paulo"], "rede": ["Municipal"],
        "taxa_alfabetizacao": [60.0], "media_portugues": [750.0],
        "percentual_participacao": [90.0], "meta_alfabetizacao_2024": [70.0],
    })
    desempenho = pd.DataFrame({
        "ano": [2023], "id_municipio": ["3550308"],
        "id_municipio_nome": ["São Paulo"], "rede": ["Municipal"],
        "total_alunos": [100], "pct_alfabetizados": [60.0],
        "taxa_alfabetizacao_municipio": [60.0], "proficiencia_media_ponderada": [750.0],
    })
    metas = pd.DataFrame({
        "nivel_geografico": ["municipio", "uf"], "ano": [2023, 2023],
        "ano_meta": [2024, 2024], "rede": ["Municipal", "Municipal"],
        "id_municipio": ["3550308", None], "id_municipio_nome": ["São Paulo", None],
        "taxa_alfabetizacao": [60.0, 60.0], "meta_alfabetizacao": [70.0, 70.0],
        "gap_para_meta": [10.0, 10.0], "atingiu_meta": [False, False],
        "sigla_uf": ["SP", "SP"], "sigla_uf_nome": ["São Paulo", "São Paulo"],
        "idhm": [0.8, 0.8], "idhm_educacao": [0.8, 0.8],
        "idhm_renda": [0.8, 0.8], "idhm_longevidade": [0.8, 0.8],
    })
    alunos = pd.DataFrame({
        "ano": [2024, 2024], "id_aluno": ["a", "b"],
        "id_municipio": ["3550308", "3304557"], "id_escola": ["e1", "e2"],
        "rede": ["3", "2"], "peso_aluno": [1.0, 1.0], "alfabetizado": [1, 0],
    })
    gold = build_consolidated_gold_features(indicador, desempenho, metas)
    dataset = build_modeling_dataset_from_gold(alunos, gold)
    assert dataset["gold_historico_disponivel"].tolist() == [1, 0]
    report = validate_dataset(dataset, False)
    assert report["valido"], report["errors"]


def _run_all() -> None:
    tests = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]

    for test in tests:
        test()
        print(f"OK - {test.__name__}")

    print(f"\n{len(tests)}/{len(tests)} testes passaram.")


if __name__ == "__main__":
    _run_all()
