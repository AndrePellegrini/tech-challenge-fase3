"""POC descritiva de atributos escolares de 2023, sem alterar dataset/X/modelos."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling.granularity_audit import (
    identify_profiles, official_feature_columns, profile_target_stats,
    variance_decomposition,
)
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = Path(
    r"C:\Users\pedro\Documents\ChatGPT\PÓS FIAP - Implemantação\tmp\censo_poc\microdados_censo_escolar_2023.zip"
)
REPORT_PATH = ROOT / "reports" / "censo_escolar_feasibility.md"
INVENTORY_PATH = ROOT / "reports" / "censo_escolar_feature_inventory.csv"
JOIN_PATH = ROOT / "reports" / "censo_escolar_join_audit.json"
GRANULARITY_PATH = ROOT / "reports" / "censo_escolar_granularity_comparison.json"
IMAGE_DIR = ROOT / "images" / "modeling" / "censo_escolar_poc"

KEY = "CO_ENTIDADE"
STATUS = "TP_SITUACAO_FUNCIONAMENTO"
SELECTED = [
    "TP_DEPENDENCIA", "TP_LOCALIZACAO", "QT_MAT_FUND_AI_2",
    "IN_AGUA_POTAVEL", "IN_ENERGIA_REDE_PUBLICA", "IN_ESGOTO_REDE_PUBLICA",
    "IN_INTERNET", "IN_BANDA_LARGA", "IN_BIBLIOTECA_SALA_LEITURA",
    "QT_DOC_FUND_AI",
]
ALTERNATIVES = [
    "TP_LOCALIZACAO_DIFERENCIADA", "IN_LIXO_SERVICO_COLETA",
    "IN_LABORATORIO_CIENCIAS", "IN_LABORATORIO_INFORMATICA",
    "IN_QUADRA_ESPORTES", "IN_ACESSIBILIDADE_RAMPAS", "IN_ALIMENTACAO",
    "QT_SALAS_UTILIZADAS", "QT_DESKTOP_ALUNO", "QT_TUR_FUND_AI",
]
DESCRIPTIONS = {
    "TP_DEPENDENCIA": "Dependência administrativa",
    "TP_LOCALIZACAO": "Localização urbana/rural",
    "QT_MAT_FUND_AI_2": "Matrículas do ensino fundamental - anos iniciais - 2º ano",
    "IN_AGUA_POTAVEL": "Fornece água potável para consumo humano",
    "IN_ENERGIA_REDE_PUBLICA": "Abastecimento de energia elétrica por rede pública",
    "IN_ESGOTO_REDE_PUBLICA": "Esgoto sanitário por rede pública",
    "IN_INTERNET": "Acesso à internet",
    "IN_BANDA_LARGA": "Internet banda larga",
    "IN_BIBLIOTECA_SALA_LEITURA": "Biblioteca e/ou sala de leitura",
    "QT_DOC_FUND_AI": "Docentes do ensino fundamental - anos iniciais",
    "TP_LOCALIZACAO_DIFERENCIADA": "Localização diferenciada da escola",
    "IN_LIXO_SERVICO_COLETA": "Destinação do lixo por serviço de coleta",
    "IN_LABORATORIO_CIENCIAS": "Laboratório de ciências",
    "IN_LABORATORIO_INFORMATICA": "Laboratório de informática",
    "IN_QUADRA_ESPORTES": "Quadra de esportes",
    "IN_ACESSIBILIDADE_RAMPAS": "Acessibilidade por rampas",
    "IN_ALIMENTACAO": "Alimentação escolar para os alunos",
    "QT_SALAS_UTILIZADAS": "Salas de aula utilizadas",
    "QT_DESKTOP_ALUNO": "Computadores de mesa para uso dos alunos",
    "QT_TUR_FUND_AI": "Turmas do ensino fundamental - anos iniciais",
}
BINARY = {column for column in SELECTED + ALTERNATIVES if column.startswith("IN_")}


def json_ready(value):
    """Converte escalares e valores não finitos para JSON estrito."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def official_X_guard() -> list[str]:
    before = official_feature_columns()
    if set(before).intersection(SELECTED):
        raise ValueError("Features da POC já aparecem no X oficial.")
    return before


def normalize_school_id(series: pd.Series) -> tuple[pd.Series, dict]:
    """Normaliza código numérico para texto de 8 dígitos, com auditoria explícita."""
    original_dtype = str(series.dtype)
    text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    valid = text.str.fullmatch(r"\d{1,8}", na=False)
    normalized = text.where(valid).str.zfill(8)
    return normalized, {
        "original_dtype": original_dtype,
        "normalization": "texto numérico com zero à esquerda até 8 posições",
        "invalid_or_null": int((~valid).sum()),
    }


def read_censo_zip(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(f"ZIP do Censo Escolar 2023 não encontrado: {path}")
    usecols = [KEY, STATUS, *SELECTED, *ALTERNATIVES]
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("microdados_ed_basica_2023.csv")]
        if len(names) != 1:
            raise ValueError("CSV escolar oficial não identificado unicamente no ZIP.")
        with archive.open(names[0]) as raw:
            chunks = pd.read_csv(
                raw,
                sep=";",
                encoding="cp1252",
                usecols=usecols,
                low_memory=False,
                chunksize=25_000,
            )
            active_chunks = []
            rows_all_statuses = 0
            for chunk in chunks:
                rows_all_statuses += len(chunk)
                active_chunks.append(chunk.loc[chunk[STATUS].eq(1)])
            data = pd.concat(active_chunks, ignore_index=True)
    normalized, key_audit = normalize_school_id(data[KEY])
    data["id_escola_join"] = normalized
    active = data.copy()
    if active["id_escola_join"].duplicated().any():
        raise ValueError("Censo ativo não é único por escola; join many-to-one inseguro.")
    return active, {
        "source": "Inep - Microdados do Censo Escolar da Educação Básica 2023",
        "url": "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2023.zip",
        "zip_bytes": int(path.stat().st_size), "csv_uncompressed_bytes": 210_035_219,
        "format": "ZIP contendo CSV separado por ponto e vírgula e dicionário XLSX",
        "year": 2023, "granularity": "uma linha por escola",
        "source_key": KEY, "access": "download HTTP oficial; sem GCP",
        "rows_all_statuses": int(rows_all_statuses), "active_school_rows": int(len(active)),
        "key_audit": key_audit,
    }


def safe_school_join(students: pd.DataFrame, schools: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if students["id_aluno"].duplicated().any():
        raise ValueError("Alunos duplicados antes do join.")
    student_key, student_key_audit = normalize_school_id(students["id_escola"])
    left = students.copy()
    left["id_escola_join"] = student_key
    school_columns = ["id_escola_join", *SELECTED]
    if schools["id_escola_join"].duplicated().any():
        raise ValueError("Expansão potencial: mais de uma linha escolar por id_escola.")
    student_schools = set(left["id_escola_join"].dropna())
    census_schools = set(schools["id_escola_join"].dropna())
    if student_schools.isdisjoint(census_schools):
        joined = left
        for column in SELECTED:
            joined[column] = pd.Series(pd.NA, index=joined.index, dtype="Float32")
        joined["_merge"] = "left_only"
    else:
        joined = left.merge(
            schools[school_columns], on="id_escola_join", how="left",
            validate="many_to_one", indicator=True,
        )
    if len(joined) != len(students) or joined["id_aluno"].duplicated().any():
        raise ValueError("Join expandiu linhas ou duplicou id_aluno.")
    matched = joined["_merge"].eq("both")
    missing_ids = sorted(student_schools - census_schools)
    audit = {
        "rows_before": int(len(students)), "rows_after": int(len(joined)),
        "unique_students_after": int(joined["id_aluno"].nunique()),
        "row_expansion": int(len(joined) - len(students)),
        "student_id_duplicates_after": int(joined["id_aluno"].duplicated().sum()),
        "student_key": student_key_audit,
        "student_unique_schools": int(len(student_schools)),
        "census_active_unique_schools": int(len(census_schools)),
        "matched_student_schools": int(len(student_schools & census_schools)),
        "school_key_coverage": float(len(student_schools & census_schools) / len(student_schools)),
        "unmatched_student_school_share": float(len(missing_ids) / len(student_schools)),
        "student_row_coverage": float(matched.mean()),
        "students_without_match": int((~matched).sum()),
        "unmatched_school_id_examples": missing_ids[:20],
        "missing_after_join": {column: float(joined[column].isna().mean()) for column in SELECTED},
        "cardinality_after_join": {column: int(joined[column].nunique(dropna=False)) for column in SELECTED},
    }
    return joined.drop(columns="_merge"), audit


def profile_summary(data: pd.DataFrame, columns: list[str]) -> dict:
    ids = data.groupby(columns, dropna=False, observed=True, sort=False).ngroup().to_numpy()
    stats = profile_target_stats(ids, data["alfabetizado"])
    sizes = stats["students"]
    mixed = stats["type"].eq("mixed")
    return {
        "profiles": int(len(stats)), "students_per_profile_mean": float(sizes.mean()),
        "students_per_profile_median": float(sizes.median()),
        "students_per_profile_p90": float(sizes.quantile(.90)),
        "students_per_profile_p95": float(sizes.quantile(.95)),
        "students_per_profile_max": int(sizes.max()),
        "mixed_profiles": int(mixed.sum()), "mixed_profile_share": float(mixed.mean()),
        "students_in_mixed_profiles": int(stats.loc[mixed, "students"].sum()),
        "mixed_profile_student_share": float(stats.loc[mixed, "students"].sum() / len(data)),
    }


def school_group_summary(data: pd.DataFrame, keys: list[str]) -> dict:
    groups = data.groupby(keys, dropna=False, observed=True)["alfabetizado"].agg(
        students="size", class_1="sum", target_rate="mean",
    )
    groups["class_0"] = groups["students"] - groups["class_1"]
    mixed = groups["class_0"].gt(0) & groups["class_1"].gt(0)
    variance = variance_decomposition(data, keys)
    return {
        "groups": int(len(groups)), "mixed_groups": int(mixed.sum()),
        "mixed_group_share": float(mixed.mean()),
        "students_in_mixed_groups": int(groups.loc[mixed, "students"].sum()),
        "mixed_group_student_share": float(groups.loc[mixed, "students"].sum() / len(data)),
        "students_per_group": {
            "mean": float(groups["students"].mean()), "median": float(groups["students"].median()),
            "p90": float(groups["students"].quantile(.90)), "p95": float(groups["students"].quantile(.95)),
            "max": int(groups["students"].max()),
        },
        "target_rate": groups["target_rate"].describe(percentiles=[.1, .25, .5, .75, .9]).to_dict(),
        "variance": variance,
    }


def feature_associations(data: pd.DataFrame) -> dict:
    output = {}
    for column in SELECTED:
        complete = data[[column, "alfabetizado"]].dropna()
        if column.startswith("QT_") and complete[column].nunique() > 10:
            grouped_column = pd.qcut(complete[column], q=5, duplicates="drop").astype(str)
            grouping = "quintis"
        else:
            grouped_column = complete[column].astype("string")
            grouping = "categorias"
        table = complete.assign(_group=grouped_column).groupby("_group", observed=True)["alfabetizado"].agg(
            students="size", target_rate="mean",
        ).reset_index()
        rates = table["target_rate"]
        output[column] = {
            "coverage": float(len(complete) / len(data)), "grouping": grouping,
            "target_rate_range": float(rates.max() - rates.min()),
            "groups": table.rename(columns={"_group": "value"}).to_dict(orient="records"),
        }
    return output


def feature_inventory(schools: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in SELECTED + ALTERNATIVES:
        if column in {"TP_DEPENDENCIA", "TP_LOCALIZACAO", "TP_LOCALIZACAO_DIFERENCIADA"}:
            category = "caracterização"
        elif column.startswith(("QT_MAT_", "QT_TUR_")):
            category = "porte e matrículas"
        elif column.startswith("QT_DOC_"):
            category = "corpo docente"
        else:
            category = "infraestrutura e serviços"
        rows.append({
            "original_name": column, "meaning": DESCRIPTIONS[column],
            "category": category,
            "dtype": str(schools[column].dtype),
            "cardinality": int(schools[column].nunique(dropna=True)),
            "missing_rate_active_schools": float(schools[column].isna().mean()),
            "granularity": "escola", "year": 2023,
            "temporality": "pré-target 2024", "leakage_risk": "A - baixo risco",
            "selected_for_poc": column in SELECTED,
            "potential_utility": "diferenciar escolas dentro do mesmo município+rede",
        })
    return pd.DataFrame(rows)


def create_figures(join: dict, comparison: dict, data: pd.DataFrame, associations: dict) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    coverage = [join["student_row_coverage"], 1 - join["student_row_coverage"]]
    plt.figure(figsize=(6, 4)); plt.bar(["Com match", "Sem match"], coverage, color=["#55a868", "#c44e52"])
    plt.ylim(0, 1); plt.ylabel("Proporção de alunos"); plt.title("Cobertura do Censo Escolar 2023")
    plt.tight_layout(); plt.savefig(IMAGE_DIR / "01_join_coverage.png", dpi=150); plt.close()

    school_sizes = data.groupby("id_escola_join", dropna=False).size()
    plt.figure(figsize=(8, 5)); plt.hist(school_sizes, bins=50, log=True, color="#4c72b0")
    plt.xlabel("Alunos por escola"); plt.ylabel("Escolas (escala log)"); plt.title("Distribuição da população por escola")
    plt.tight_layout(); plt.savefig(IMAGE_DIR / "02_students_per_school.png", dpi=150); plt.close()

    profile = pd.DataFrame({
        "versão": ["X oficial", "X + escola"],
        "perfis": [comparison["before"]["profiles"], comparison["after"]["profiles"]],
    })
    plt.figure(figsize=(6, 4)); plt.bar(profile["versão"], profile["perfis"], color=["#8172b3", "#4c72b0"])
    plt.ylabel("Perfis únicos"); plt.title("Quebra de perfis repetidos")
    plt.tight_layout(); plt.savefig(IMAGE_DIR / "03_profiles_before_after.png", dpi=150); plt.close()

    for number, column in enumerate(("TP_LOCALIZACAO", "QT_MAT_FUND_AI_2"), start=4):
        table = pd.DataFrame(associations[column]["groups"])
        plt.figure(figsize=(8, 4))
        if table.empty:
            plt.text(.5, .5, "Sem observações após o join", ha="center", va="center")
            plt.xticks([]); plt.yticks([])
        else:
            plt.bar(table["value"].astype(str), table["target_rate"], color="#55a868")
            plt.xticks(rotation=30, ha="right"); plt.ylabel("Taxa observada de alfabetização")
        plt.title(f"Associação descritiva: {DESCRIPTIONS[column]}")
        plt.tight_layout(); plt.savefig(IMAGE_DIR / f"0{number}_{column.lower()}.png", dpi=150); plt.close()


def render_report(source: dict, join: dict, comparison: dict, groups: dict, associations: dict) -> str:
    before, after = comparison["before"], comparison["after"]
    selected_rows = "\n".join(
        f"- `{column}` — {DESCRIPTIONS[column]}; cobertura no join {1-join['missing_after_join'][column]:.2%}; risco A (baixo)."
        for column in SELECTED
    )
    return f"""# POC de viabilidade — Censo Escolar 2023

## Classificação da evidência

Usar a Gold e tratar leakage são requisitos do Tech Challenge. O PDF permite
Censo Escolar como fonte externa, mas não o torna obrigatório. A seleção das dez
features, normalização da chave e esta POC são decisões metodológicas do grupo e
extensão com conhecimento externo.

## Fonte

- fonte: Inep, Microdados do Censo Escolar da Educação Básica 2023;
- arquivo: `microdados_censo_escolar_2023.zip` ({source['zip_bytes']/1024/1024:.2f} MiB);
- CSV interno: {source['csv_uncompressed_bytes']/1024/1024:.2f} MiB;
- granularidade: uma linha por escola;
- chave: `CO_ENTIDADE`;
- escolas ativas: {source['active_school_rows']:,};
- formato: ZIP + CSV `;` + dicionário XLSX;
- acesso: download HTTP oficial, sem GCP ou cloud de processamento.

A busca no repositório atual, na Fase 2 e nos prefixos Silver/Gold do S3 não
encontrou uma tabela de atributos escolares. A Base dos Dados exigiria um projeto
de cobrança GCP, indisponível neste ambiente. Por isso, a POC usou o arquivo
oficial do Inep, que é pequeno o bastante para leitura local em blocos. Materiais
complementares da FIAP não forneceram uma tabela de correspondência de escolas.

## Compatibilidade e join

`id_escola` nos alunos tem dtype `{join['student_key']['original_dtype']}` e
`CO_ENTIDADE` no Censo tem dtype `{source['key_audit']['original_dtype']}`. A POC
normaliza explicitamente ambos como texto numérico de oito posições, sem alterar
as colunas originais.

- escolas únicas nos alunos: {join['student_unique_schools']:,};
- escolas ativas no Censo: {join['census_active_unique_schools']:,};
- escolas dos alunos encontradas: {join['matched_student_schools']:,}
  ({join['school_key_coverage']:.2%});
- alunos cobertos: {join['student_row_coverage']:.2%};
- alunos sem match: {join['students_without_match']:,};
- linhas antes/depois: {join['rows_before']:,}/{join['rows_after']:,};
- expansão e duplicidade: {join['row_expansion']}/{join['student_id_duplicates_after']}.

Possíveis causas de não match: escola não ativa no Censo 2023, mudança de código,
ausência/invalidade da chave ou participação na avaliação em contexto não coberto
pela edição escolar selecionada. Esses casos precisam de auditoria antes da
integração oficial.

Neste dataset, os exemplos de `id_escola` começam em códigos sequenciais `60...`,
sem qualquer interseção com `CO_ENTIDADE`. Isso evidencia anonimização ou
recodificação da chave na fonte de alunos. Sem a tabela segura de correspondência
entre o identificador anonimizado e o código Inep, o enriquecimento não é viável.

## Features escolhidas

{selected_rows}

Todas são administrativas de 2023, anteriores ao target 2024 e não usam
alfabetização, proficiência ou resultado de prova. Quantidades recebem imputação
somente em eventual pipeline futuro; nenhuma imputação foi feita nesta POC.

## Quebra dos perfis

O X oficial tem {before['profiles']:,} perfis. O X experimental tem
{after['profiles']:,}, aumento de {comparison['absolute_profile_increase']:,}
({comparison['relative_profile_increase']:.2%}). A média cai de
{before['students_per_profile_mean']:.2f} para {after['students_per_profile_mean']:.2f}
alunos por perfil; a mediana muda de {before['students_per_profile_median']:.0f}
para {after['students_per_profile_median']:.0f}.

Alunos em perfis mistos mudam de {before['mixed_profile_student_share']:.2%}
para {after['mixed_profile_student_share']:.2%}; perfis mistos, de
{before['mixed_profile_share']:.2%} para {after['mixed_profile_share']:.2%}.
Essa quebra mede granularidade, não ganho preditivo.

## Variabilidade por escola

- escolas na população: {groups['school']['groups']:,};
- escolas com ambas as classes: {groups['school']['mixed_group_share']:.2%};
- variação entre escolas: {groups['school']['variance']['between_share']:.2%};
- variação dentro das escolas: {groups['school']['variance']['within_share']:.2%};
- comparação município+rede: {groups['municipality_network']['variance']['between_share']:.2%}
  entre contextos e {groups['municipality_network']['variance']['within_share']:.2%} dentro.

Descer para escola explica parcela adicional da heterogeneidade, mas a maior
parte ainda permanece dentro da escola. As associações das features do Censo
ficaram indisponíveis porque a cobertura do join foi zero. Os gráficos registram
explicitamente essa ausência; não há estimativa ou interpretação causal.

## Custo e decisão da POC

Esforço estimado: **médio**. O download é pequeno, o CSV tem cerca de 200 MiB,
o join é many-to-one e não requer GCP. O trabalho restante envolve armazenar a
fonte de modo reproduzível, parametrizar o reader, mapear categorias, definir
missing, validar cobertura, adicionar testes e versionar documentação.

Viabilidade: **{comparison['viability']}**.

Recomendação técnica: não incorporar ao modelo oficial com as chaves atuais.
Somente retomar uma versão experimental se o responsável pela anonimização
fornecer uma tabela de correspondência auditável entre `id_escola` e
`CO_ENTIDADE`. Não foram treinados modelos, portanto não há evidência de ganho em
ROC AUC. A obtenção desse mapa e a autorização de uso ficam **REVISAR COM O GRUPO**.
"""


def main() -> None:
    official_X = official_X_guard()
    dataset = pd.read_parquet(DATASET_PATH)
    validation = validate_dataset(dataset, strict_counts=True)
    if not validation["valido"]:
        raise ValueError(validation["errors"])
    censo_path = Path(os.getenv("CENSO_ESCOLAR_2023_ZIP", str(DEFAULT_ZIP)))
    schools, source = read_censo_zip(censo_path)
    joined, join_audit = safe_school_join(dataset, schools)
    before = profile_summary(joined, official_X)
    after = profile_summary(joined, [*official_X, *SELECTED])
    comparison = {
        "before": before, "after": after,
        "absolute_profile_increase": after["profiles"] - before["profiles"],
        "relative_profile_increase": after["profiles"] / before["profiles"] - 1,
        "official_X_changed": False, "official_dataset_overwritten": False,
        "model_trained": False, "test_accessed": False,
    }
    # Regra diagnóstica transparente; decisão substantiva permanece com o grupo.
    if join_audit["student_row_coverage"] >= .90 and comparison["relative_profile_increase"] >= .25:
        comparison["viability"] = "ALTA"
    elif join_audit["student_row_coverage"] >= .70:
        comparison["viability"] = "MÉDIA"
    elif join_audit["student_row_coverage"] > 0:
        comparison["viability"] = "BAIXA"
    else:
        comparison["viability"] = "NÃO RECOMENDADO"
    groups = {
        "school": school_group_summary(joined, ["id_escola_join"]),
        "municipality_network_school": school_group_summary(joined, ["id_municipio", "rede", "id_escola_join"]),
        "municipality_network": school_group_summary(joined, ["id_municipio", "rede"]),
    }
    associations = feature_associations(joined)
    inventory = feature_inventory(schools)
    join_audit.update({"source": source, "selected_features": SELECTED, "leakage": {column: "A - baixo risco; administrativo de 2023" for column in SELECTED}})
    granularity = {"profiles": comparison, "groups": groups, "feature_associations": associations}
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(INVENTORY_PATH, index=False)
    JOIN_PATH.write_text(json.dumps(json_ready(join_audit), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    GRANULARITY_PATH.write_text(json.dumps(json_ready(granularity), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(source, join_audit, comparison, groups, associations), encoding="utf-8")
    create_figures(join_audit, comparison, joined, associations)
    print(json.dumps(json_ready({"source": source, "join": join_audit, "comparison": comparison, "groups": groups}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
