from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / "env", override=False)


GOLD_BASE_PATH = (
    "s3://tc-fase-dois-alfabetizacao-bronze/gold"
)

PROCESSING_DATE = "2026-07-10"


GOLD_FILES = {
    "indicador_alfabetizacao_municipio": (
        f"{GOLD_BASE_PATH}/"
        "indicador_alfabetizacao_municipio/"
        f"processing_date={PROCESSING_DATE}/"
        "indicador_alfabetizacao_municipio.parquet"
    ),

    "comparativo_metas_resultados": (
        f"{GOLD_BASE_PATH}/"
        "comparativo_metas_resultados/"
        f"processing_date={PROCESSING_DATE}/"
        "comparativo_metas_resultados.parquet"
    ),

    "evolucao_temporal_indicador": (
        f"{GOLD_BASE_PATH}/"
        "evolucao_temporal_indicador/"
        f"processing_date={PROCESSING_DATE}/"
        "evolucao_temporal_indicador.parquet"
    ),

    "desempenho_alunos_municipio": (
        f"{GOLD_BASE_PATH}/"
        "desempenho_alunos_municipio/"
        f"processing_date={PROCESSING_DATE}/"
        "desempenho_alunos_municipio.parquet"
    ),
}


def read_gold_table(
    table_name: str,
) -> pd.DataFrame:
    """
    Lê uma tabela da camada Gold produzida
    no Tech Challenge da Fase 2.

    Parameters
    ----------
    table_name : str
        Nome lógico da tabela Gold.

    Returns
    -------
    pd.DataFrame
        Tabela Gold carregada em memória.
    """

    if table_name not in GOLD_FILES:
        available = ", ".join(GOLD_FILES.keys())

        raise ValueError(
            f"Tabela Gold desconhecida: {table_name}. "
            f"Disponíveis: {available}"
        )

    return pd.read_parquet(
        GOLD_FILES[table_name]
    )