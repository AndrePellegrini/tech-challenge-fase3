from __future__ import annotations

from pathlib import Path

from src.preprocessing.bigquery_reader import run_query
from src.preprocessing.build_modeling_dataset import build_modeling_dataset
from src.preprocessing.queries import (
    QUERY_ALUNOS_2024,
    QUERY_MUNICIPIO_2023,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_2024.parquet"
)


def validate_modeling_dataset(dataset):
    """
    Executa validações estruturais do dataset de modelagem.
    """

    expected_rows = 1_851_852
    expected_history_rows = 1_816_270

    if dataset.shape[0] != expected_rows:
        raise ValueError(
            "Quantidade inesperada de registros. "
            f"Esperado: {expected_rows}. "
            f"Obtido: {dataset.shape[0]}."
        )

    if dataset["id_aluno"].duplicated().any():
        duplicate_count = (
            dataset["id_aluno"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Foram encontrados {duplicate_count} "
            "alunos duplicados."
        )

    if dataset["alfabetizado"].isna().any():
        raise ValueError(
            "Foram encontrados valores nulos no target."
        )

    target_values = set(
        dataset["alfabetizado"].unique()
    )

    if target_values != {0, 1}:
        raise ValueError(
            "Target contém valores inesperados: "
            f"{target_values}"
        )

    history_count = int(
        dataset[
            "historico_2023_disponivel"
        ].sum()
    )

    if history_count != expected_history_rows:
        raise ValueError(
            "Quantidade inesperada de registros "
            "com histórico de 2023. "
            f"Esperado: {expected_history_rows}. "
            f"Obtido: {history_count}."
        )

    historical_missing_match = (
        dataset["taxa_alfabetizacao_2023"].isna()
        ==
        dataset["media_portugues_2023"].isna()
    ).all()

    if not historical_missing_match:
        raise ValueError(
            "As features históricas apresentam "
            "padrões de ausência diferentes."
        )

    print("Validações concluídas com sucesso.")


def main():
    """
    Executa o pipeline completo de construção
    do dataset de modelagem.
    """

    print("Iniciando pipeline da Fase 3.")

    print("\n1. Consultando alunos de 2024...")
    alunos_2024 = run_query(
        query=QUERY_ALUNOS_2024,
    )

    print(
        f"Alunos carregados: "
        f"{alunos_2024.shape[0]:,}"
    )

    print("\n2. Consultando histórico municipal de 2023...")
    municipio_2023 = run_query(
        query=QUERY_MUNICIPIO_2023,
    )

    print(
        f"Registros municipais carregados: "
        f"{municipio_2023.shape[0]:,}"
    )

    print("\n3. Construindo dataset de modelagem...")
    dataset = build_modeling_dataset(
        alunos_df=alunos_2024,
        municipio_df=municipio_2023,
    )

    print(
        f"Dataset construído: "
        f"{dataset.shape[0]:,} linhas "
        f"x {dataset.shape[1]} colunas"
    )

    print("\n4. Executando validações...")
    validate_modeling_dataset(dataset)

    print("\n5. Salvando dataset...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    file_size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"Arquivo salvo em:\n{OUTPUT_PATH}"
    )

    print(
        f"Tamanho: {file_size_mb:.2f} MB"
    )

    print("\nPipeline concluído com sucesso.")


if __name__ == "__main__":
    main()