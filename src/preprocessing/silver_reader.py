"""População individual da Silver da Fase 2, sem dependência de GCP."""
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.preprocessing.build_modeling_dataset import TARGET_YEAR, VALID_MODELING_NETWORKS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_ALUNOS_PATH = (
    "s3://tc-fase-dois-alfabetizacao-bronze/silver/alunos/"
    "processing_date=2026-07-09/alunos.parquet"
)
STUDENT_COLUMNS = [
    "ano", "id_aluno", "id_municipio", "id_escola",
    "rede", "peso_aluno", "alfabetizado",
]
SILVER_COLUMNS = STUDENT_COLUMNS + ["presenca", "preenchimento_caderno"]


def build_modeling_population(silver: pd.DataFrame) -> pd.DataFrame:
    """Filtra elegibilidade antes de selecionar os sete atributos individuais.

    Preserva o target original; sua conversão ocorre no builder Gold existente.
    Não deduplica silenciosamente nem imputa valores.
    """
    missing = sorted(set(SILVER_COLUMNS) - set(silver.columns))
    if missing:
        raise ValueError(f"Colunas ausentes na Silver alunos: {missing}")
    population = silver.loc[
        silver["ano"].eq(TARGET_YEAR)
        & silver["presenca"].eq("Presente")
        & silver["preenchimento_caderno"].eq("Prova preenchida")
        & silver["rede"].isin(VALID_MODELING_NETWORKS),
        STUDENT_COLUMNS,
    ].copy()
    if population["id_aluno"].isna().any():
        raise ValueError("ID de aluno nulo na população Silver.")
    if population["id_aluno"].duplicated().any():
        raise ValueError("IDs de aluno duplicados na população Silver.")
    return population.reset_index(drop=True)


def read_silver_students() -> pd.DataFrame:
    """Lê somente as nove colunas necessárias da partição auditada no S3.

    Usa Parquet/S3 como na Fase 2, com projeção via pandas/s3fs já usados na
    Fase 3. A data é fixa para reprodução, sem seleção automática da última.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / "env", override=False)
    silver = pd.read_parquet(SILVER_ALUNOS_PATH, columns=SILVER_COLUMNS)
    return build_modeling_population(silver)
