"""Valida o parquet Gold local sem acessar AWS/BigQuery ou alterar dados.

Uso: python -m src.preprocessing.validate_dataset
Schema exato (ordem livre), contagens, unicidade, target e missingness Gold.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.dataset_contract import (
    DATASET_FILENAME,
    GOLD_FEATURE_COLUMNS,
    EXPECTED_COLUMNS,
    EXPECTED_HISTORY_AVAILABLE,
    EXPECTED_HISTORY_MISSING,
    EXPECTED_ROWS,
    EXPECTED_TARGET_VALUES,
    HISTORY_FEATURE_COLUMNS,
    HISTORY_FLAG_COLUMN,
    TARGET_COLUMN,
    UNIQUE_KEY_COLUMN,
    describe_contract,
    get_expected_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / DATASET_FILENAME
)


def check_schema(dataset: pd.DataFrame) -> list[str]:
    """
    Verifica se o dataset possui exatamente
    as colunas definidas no contrato.
    """

    errors: list[str] = []

    if dataset.columns.has_duplicates:
        errors.append("Nomes de colunas duplicados no schema.")

    expected = set(get_expected_columns())
    found = set(dataset.columns)

    missing = sorted(expected - found)
    unexpected = sorted(found - expected)

    if missing:
        errors.append(
            f"Colunas ausentes: {missing}"
        )

    if unexpected:
        errors.append(
            f"Colunas inesperadas: {unexpected}"
        )

    return errors


def check_shape(
    dataset: pd.DataFrame,
    strict_counts: bool = True,
) -> list[str]:
    """
    Verifica o shape esperado do dataset.

    Parameters
    ----------
    strict_counts : bool
        Quando True, exige as contagens exatas do
        resultado validado. Quando False, valida apenas
        a estrutura (uso em testes sintéticos).
    """

    errors: list[str] = []

    if dataset.shape[1] != EXPECTED_COLUMNS:
        errors.append(
            "Quantidade inesperada de colunas. "
            f"Esperado: {EXPECTED_COLUMNS}. "
            f"Obtido: {dataset.shape[1]}."
        )

    if strict_counts and dataset.shape[0] != EXPECTED_ROWS:
        errors.append(
            "Quantidade inesperada de registros. "
            f"Esperado: {EXPECTED_ROWS:,}. "
            f"Obtido: {dataset.shape[0]:,}."
        )

    return errors


def check_uniqueness(dataset: pd.DataFrame) -> list[str]:
    """
    Verifica ausência de duplicidades de `id_aluno`.
    """

    errors: list[str] = []

    if dataset[UNIQUE_KEY_COLUMN].isna().any():
        errors.append(f"Valores nulos em {UNIQUE_KEY_COLUMN}.")

    duplicated = int(
        dataset[UNIQUE_KEY_COLUMN]
        .duplicated()
        .sum()
    )

    if duplicated > 0:
        errors.append(
            f"Foram encontradas {duplicated} "
            f"duplicidades em {UNIQUE_KEY_COLUMN}."
        )

    return errors


def check_target(dataset: pd.DataFrame) -> list[str]:
    """
    Verifica ausência de nulos e domínio {0, 1} do target.
    """

    errors: list[str] = []

    null_count = int(
        dataset[TARGET_COLUMN]
        .isna()
        .sum()
    )

    if null_count > 0:
        errors.append(
            f"Target com {null_count} valores nulos."
        )

    observed_values = set(
        dataset[TARGET_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    if not observed_values.issubset(EXPECTED_TARGET_VALUES):
        errors.append(
            "Target contém valores fora do domínio "
            f"{EXPECTED_TARGET_VALUES}: {observed_values}"
        )

    return errors


def check_history(
    dataset: pd.DataFrame,
    strict_counts: bool = True,
) -> list[str]:
    """
    Verifica a disponibilidade do histórico de 2023 e a
    coerência entre a flag e os nulos das features históricas.
    """

    errors: list[str] = []

    flag = dataset[HISTORY_FLAG_COLUMN]
    if flag.isna().any() or not flag.isin([0, 1]).all():
        return [f"{HISTORY_FLAG_COLUMN} deve ser não nula e conter apenas 0 ou 1."]

    available = int((flag == 1).sum())
    missing = int((flag == 0).sum())

    if strict_counts:
        if available != EXPECTED_HISTORY_AVAILABLE:
            errors.append(
                "Registros com histórico divergentes. "
                f"Esperado: {EXPECTED_HISTORY_AVAILABLE:,}. "
                f"Obtido: {available:,}."
            )

        if missing != EXPECTED_HISTORY_MISSING:
            errors.append(
                "Registros sem histórico divergentes. "
                f"Esperado: {EXPECTED_HISTORY_MISSING:,}. "
                f"Obtido: {missing:,}."
            )

    for column in GOLD_FEATURE_COLUMNS:
        invalid = (flag.eq(0) & dataset[column].notna()).sum()
        if invalid:
            errors.append(
                f"{invalid:,} registros sem Gold com {column} preenchida."
            )

    for column in HISTORY_FEATURE_COLUMNS:
        invalid = (flag.eq(1) & dataset[column].isna()).sum()
        if invalid:
            errors.append(
                f"{invalid:,} registros com Gold e núcleo {column} nulo."
            )

    return errors


def validate_dataset(
    dataset: pd.DataFrame,
    strict_counts: bool = True,
) -> dict:
    """
    Executa todas as validações do contrato e retorna
    um relatório estruturado.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset de modelagem a validar.

    strict_counts : bool
        Quando True, exige as contagens exatas do
        resultado final validado pelo grupo.

    Returns
    -------
    dict
        Relatório com o resultado de cada verificação.
    """

    schema_errors = check_schema(dataset)

    report: dict = {
        "shape": tuple(dataset.shape),
        "checks": {},
        "errors": [],
    }

    report["checks"]["schema"] = schema_errors
    report["errors"].extend(schema_errors)

    # Sem o schema correto, as demais checagens
    # não podem ser executadas com segurança.
    if schema_errors:
        report["valido"] = False
        return report

    checks = {
        "shape": check_shape(dataset, strict_counts),
        "unicidade_id_aluno": check_uniqueness(dataset),
        "target": check_target(dataset),
        "gold_historico": check_history(
            dataset,
            strict_counts,
        ),
    }

    for name, errors in checks.items():
        report["checks"][name] = errors
        report["errors"].extend(errors)

    report["valido"] = len(report["errors"]) == 0

    return report


def main() -> None:
    """
    Valida o artefato local do dataset de modelagem.
    """

    print(describe_contract())

    print(
        "\nLendo dataset de: "
        f"{DATASET_PATH}"
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset Gold não encontrado: {DATASET_PATH}. "
            "Disponibilize o parquet local ou reproduza a integração descrita em "
            "python -m src.preprocessing.gold_pipeline (requer leitura AWS/S3)."
        )

    dataset = pd.read_parquet(DATASET_PATH)

    print(
        f"Shape carregado: {dataset.shape[0]:,} "
        f"x {dataset.shape[1]}"
    )

    report = validate_dataset(dataset)

    print("\nResultado das verificações:")

    for check_name, errors in report["checks"].items():
        status = "OK" if not errors else "FALHOU"
        print(f"- {check_name}: {status}")

        for error in errors:
            print(f"    {error}")

    if not report["valido"]:
        raise ValueError(
            "Dataset de modelagem não atende ao contrato: "
            f"{len(report['errors'])} erro(s)."
        )

    print(
        "\nDataset de modelagem validado com sucesso "
        "contra o contrato da Fase 3."
    )


if __name__ == "__main__":
    main()
