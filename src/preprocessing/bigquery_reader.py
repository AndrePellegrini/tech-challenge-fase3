from __future__ import annotations

import os
from pathlib import Path

import basedosdados as bd
import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_query(
    query: str,
    project_id: str | None = None,
) -> pd.DataFrame:
    """Executa SQL pela Base dos Dados, seguindo o acesso usado na Fase 2.

    O project_id explícito continua aceito por compatibilidade. Quando
    omitido, usa BILLING_PROJECT_ID do ambiente, .env ou env da raiz.
    Credenciais e autenticação são gerenciadas pela biblioteca basedosdados.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / "env", override=False)
    billing_project_id = project_id if project_id is not None else os.getenv("BILLING_PROJECT_ID")
    if not billing_project_id or not billing_project_id.strip():
        raise ValueError("Configure BILLING_PROJECT_ID no ambiente ou no arquivo env da raiz.")

    return bd.read_sql(
        query=query,
        billing_project_id=billing_project_id,
    )
