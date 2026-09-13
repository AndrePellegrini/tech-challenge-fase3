"""Amostra anonimizada e versionável do dataset de modelagem.

O dataset completo tem 1.851.828 linhas e não é versionado, de modo que a pipeline só
era executável por quem tivesse credencial de leitura no S3 privado da equipe. Isso
contrariava o entregável "pipeline reproduzível". Esta amostra resolve o caso: com ela
versionada, qualquer avaliador executa split, pré-processamento, treino e avaliação a
partir de um clone limpo.

Duas decisões de desenho merecem registro.

**A amostragem é por município, não por aluno.** Amostrar alunos individualmente
destruiria a estrutura de grupo de que o projeto depende: o split é agrupado por
`id_municipio` e as features são contextuais, compartilhadas por todos os alunos do
mesmo município e rede. Uma amostra de alunos soltos tornaria a reprodução do split
impossível e daria uma falsa impressão de variação individual.

**A estratificação é por UF.** `sigla_uf` é feature do modelo e a distribuição
territorial é o principal eixo de variação do dataset. Amostrar municípios sem
estratificar deixaria UFs pequenas de fora e mudaria o comportamento do encoder.

Os identificadores `id_aluno` e `id_escola` são substituídos por hash. `id_municipio` e
`id_municipio_nome` são preservados: são códigos e nomes públicos do IBGE, e o split
agrupado depende deles.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.validate_dataset import DATASET_PATH

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SAMPLE_PATH = ROOT / "data" / "processed" / "sample_modeling_dataset.parquet"
SUMMARY_PATH = REPORTS / "sample_dataset_summary.json"

TARGET_ROWS = 50_000
RANDOM_STATE = 42
HASHED_COLUMNS = ("id_aluno", "id_escola")
# Sal fixo: a amostra precisa ser reproduzível bit a bit por quem clonar o repositório.
# A troca é consciente — o sal é público, então o hash protege contra republicação
# direta dos identificadores, não contra um ataque de força bruta sobre o espaço de
# códigos do INEP, que já são pseudônimos e não trazem dado pessoal.
SALT = "tech-challenge-fase3"
# Estrato dos municípios sem histórico Gold, cujo `sigla_uf` é nulo.
NO_GOLD_STRATUM = "SEM_GOLD"
HASH_LENGTH = 12
# Passos da busca binária que calibra a fração sorteada até o volume alvo.
CALIBRATION_STEPS = 18


def anonymize(series: pd.Series) -> pd.Series:
    """Substitui cada identificador por um hash curto e estável."""
    def digest(value):
        if pd.isna(value):
            return value
        raw = f"{SALT}:{value}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:HASH_LENGTH]
    return series.map(digest)


def _draw(by_municipality: pd.DataFrame, factor: float, random_state: int) -> list:
    """Sorteia uma fração `factor` dos municípios de cada estrato."""
    generator = np.random.default_rng(random_state)
    chosen = []
    for _, group in by_municipality.groupby("estrato", observed=True):
        quota = min(len(group), max(1, int(round(len(group) * factor))))
        picked = generator.choice(group["id_municipio"].to_numpy(), size=quota, replace=False)
        chosen.extend(picked.tolist())
    return chosen


def select_municipalities(data: pd.DataFrame, target_rows: int = TARGET_ROWS,
                          random_state: int = RANDOM_STATE) -> list:
    """Sorteia municípios por estrato até se aproximar do volume de linhas desejado.

    A cota de cada estrato é proporcional ao seu **número de municípios**, e não ao seu
    número de alunos. A escolha é deliberada: como o tamanho dos municípios é muito
    assimétrico — mediana de 110 alunos contra um máximo acima de 94 mil —, uma cota por
    alunos faz um único município grande dominar o estrato e distorce a composição por
    rede. A cota por contagem preserva a representatividade, e o volume final é ajustado
    pela calibração abaixo.

    Municípios sem correspondência no histórico Gold têm `sigla_uf` nulo e formam um
    estrato próprio. Sem isso o `groupby` os descartaria em silêncio, e a amostra
    deixaria de exercitar o caminho de imputação e a flag `gold_historico_disponivel`.

    A calibração é uma busca binária determinística sobre a fração sorteada. Com semente
    fixa, o resultado é reproduzível bit a bit por quem clonar o repositório.
    """
    by_municipality = data.groupby("id_municipio", observed=True).agg(
        sigla_uf=("sigla_uf", "first"), alunos=("id_aluno", "size"),
    ).reset_index()
    by_municipality["estrato"] = by_municipality["sigla_uf"].fillna(NO_GOLD_STRATUM)
    sizes = dict(zip(by_municipality["id_municipio"], by_municipality["alunos"]))

    low, high = 0.0, 1.0
    best, best_error = None, None
    for _ in range(CALIBRATION_STEPS):
        factor = (low + high) / 2
        chosen = _draw(by_municipality, factor, random_state)
        rows = sum(sizes[key] for key in chosen)
        error = abs(rows - target_rows)
        if best is None or error < best_error:
            best, best_error = chosen, error
        if rows < target_rows:
            low = factor
        else:
            high = factor
    return sorted(best)


def build_sample(data: pd.DataFrame, target_rows: int = TARGET_ROWS,
                 random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Recorta os municípios sorteados e anonimiza os identificadores individuais."""
    municipalities = select_municipalities(data, target_rows, random_state)
    sample = data.loc[data["id_municipio"].isin(municipalities)].copy()
    for column in HASHED_COLUMNS:
        sample[column] = anonymize(sample[column])
    if sample["id_aluno"].duplicated().any():
        raise ValueError("Colisão de hash em id_aluno; aumente HASH_LENGTH.")
    return sample.sort_values(["id_municipio", "id_aluno"], ignore_index=True)


def representativeness(full: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Compara amostra e universo nas dimensões que importam para a modelagem."""
    def profile(frame):
        return {
            "linhas": int(len(frame)),
            "municipios": int(frame["id_municipio"].nunique()),
            "ufs": int(frame["sigla_uf"].nunique()),
            "taxa_alfabetizacao": float(frame["alfabetizado"].mean()),
            "share_municipal": float(frame["rede"].eq("Municipal").mean()),
            "cobertura_gold": float(frame["gold_historico_disponivel"].mean()),
            "idhm_medio": float(frame["idhm"].mean()),
        }
    universe, subset = profile(full), profile(sample)
    return {
        "universo": universe,
        "amostra": subset,
        "diferencas": {
            key: round(subset[key] - universe[key], 6)
            for key in ("taxa_alfabetizacao", "share_municipal", "cobertura_gold", "idhm_medio")
        },
    }


def main() -> None:
    data = pd.read_parquet(DATASET_PATH)
    sample = build_sample(data)
    comparison = representativeness(data, sample)

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(SAMPLE_PATH, index=False, compression="snappy")

    summary = {
        "proposito": ("amostra para verificar a execução da pipeline; os resultados "
                      "oficiais do projeto vêm do dataset completo, não versionado"),
        "origem": str(DATASET_PATH.name),
        "arquivo": str(SAMPLE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "bytes": int(SAMPLE_PATH.stat().st_size),
        "estrategia": "amostragem de municípios estratificada por UF, com estrato próprio para os municípios sem histórico Gold",
        "random_state": RANDOM_STATE,
        "colunas_anonimizadas": list(HASHED_COLUMNS),
        "anonimizacao": f"sha256 com sal fixo, truncado em {HASH_LENGTH} caracteres",
        **comparison,
    }
    REPORTS.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "linhas": summary["amostra"]["linhas"],
        "municipios": summary["amostra"]["municipios"],
        "ufs": summary["amostra"]["ufs"],
        "megabytes": round(summary["bytes"] / 1024 / 1024, 2),
        "diferencas": summary["diferencas"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
