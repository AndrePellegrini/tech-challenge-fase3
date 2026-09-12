"""Auditoria descritiva de target e candidatas na Silver, sem alterar X/modelos."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.modeling.split import get_modeling_columns
from src.preprocessing.silver_reader import PROJECT_ROOT, SILVER_ALUNOS_PATH

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "reports" / "target_lineage_summary.json"
LINEAGE_PATH = ROOT / "reports" / "target_lineage_audit.md"
CANDIDATE_PATH = ROOT / "reports" / "feature_candidate_audit.md"
MATRIX_PATH = ROOT / "reports" / "feature_candidate_matrix.csv"
IMAGE_DIR = ROOT / "images" / "modeling" / "feature_audit"
THRESHOLD = 743.0

BUSINESS_COLUMNS = [
    "ano", "id_municipio", "id_municipio_nome", "id_escola", "id_aluno",
    "caderno", "serie", "rede", "presenca", "preenchimento_caderno",
    "alfabetizado", "proficiencia", "peso_aluno",
]
TECHNICAL_COLUMNS = ["_ingestion_ts", "_source", "_table", "_processed_ts", "_layer"]

COLUMN_CLASSIFICATION = {
    "ano": ("critério de elegibilidade", "Seleciona o ciclo 2024; constante após filtro."),
    "id_municipio": ("identificador/contexto", "Chave territorial e de split; não é feature individual."),
    "id_municipio_nome": ("identificador auxiliar", "Rótulo para apresentação; redundante com o código."),
    "id_escola": ("identificador", "Permite join escolar futuro, mas não deve entrar cru em X."),
    "id_aluno": ("identificador", "Chave individual; sem semântica preditiva legítima."),
    "caderno": ("possível artefato técnico", "Código de caderno; significado detalhado não documentado no projeto; contemporâneo à avaliação."),
    "serie": ("possível feature individual", "Etapa escolar declarada; utilidade depende da variação após filtro."),
    "rede": ("feature contextual", "Dependência/rede de ensino; já pertence ao X oficial."),
    "presenca": ("critério de elegibilidade", "Usado para selecionar participantes presentes."),
    "preenchimento_caderno": ("critério de elegibilidade", "Usado para selecionar provas preenchidas."),
    "alfabetizado": ("target", "Classificação do mesmo evento avaliativo."),
    "proficiencia": ("leakage direto", "Resultado contínuo do mesmo evento que define o target pelo corte 743."),
    "peso_aluno": ("peso amostral", "Reservado para eventual sample_weight; não usar como feature."),
    "_ingestion_ts": ("metadado técnico", "Momento de ingestão."),
    "_source": ("metadado técnico", "Origem técnica."),
    "_table": ("metadado técnico", "Nome técnico da tabela."),
    "_processed_ts": ("metadado técnico", "Momento de processamento Silver."),
    "_layer": ("metadado técnico", "Camada da arquitetura."),
}


def official_X_unchanged() -> list[str]:
    features = list(get_modeling_columns()["X"])
    forbidden = {"proficiencia", "caderno", "serie", "presenca", "preenchimento_caderno"}
    if forbidden.intersection(features):
        raise ValueError("A auditoria não pode alterar o X oficial.")
    return features


def load_silver_projection() -> pd.DataFrame:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / "env", override=False)
    return pd.read_parquet(SILVER_ALUNOS_PATH, columns=BUSINESS_COLUMNS)


def filter_official_population(data: pd.DataFrame) -> pd.DataFrame:
    required = set(BUSINESS_COLUMNS) - set(TECHNICAL_COLUMNS)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    mask = (
        data["ano"].eq(2024)
        & data["presenca"].eq("Presente")
        & data["preenchimento_caderno"].eq("Prova preenchida")
        & data["rede"].isin({"Estadual", "Municipal"})
    )
    return data.loc[mask].copy()


def detect_constant(series: pd.Series) -> bool:
    return series.nunique(dropna=False) <= 1


def _value_counts(series: pd.Series, limit: int = 50) -> list[dict]:
    counts = series.astype("string").fillna("<NA>").value_counts(dropna=False).head(limit)
    total = len(series)
    return [{"value": str(value), "count": int(count), "share": float(count / total)} for value, count in counts.items()]


def categorical_audit(data: pd.DataFrame, columns: list[str]) -> dict:
    return {
        column: {
            "cardinality_including_na": int(data[column].nunique(dropna=False)),
            "constant": detect_constant(data[column]),
            "values": _value_counts(data[column]),
            "target_by_value": [
                {"value": str(index), "students": int(row["size"]), "target_rate": float(row["mean"])}
                for index, row in data.groupby(column, dropna=False, observed=True)["target_binary"].agg(["size", "mean"]).iterrows()
            ],
        }
        for column in columns
    }


def normalize_target(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.casefold()
    mapped = normalized.map({"sim": 1, "não": 0, "nao": 0})
    if mapped.isna().any():
        raise ValueError(f"Target não reconhecido: {series[mapped.isna()].drop_duplicates().tolist()}")
    return mapped.astype("int8")


def proficiency_statistics(proficiency: pd.Series, target: pd.Series, threshold: float = THRESHOLD) -> dict:
    frame = pd.DataFrame({"proficiencia": pd.to_numeric(proficiency, errors="coerce"), "target": target})
    complete = frame.dropna()
    predicted = complete["proficiencia"].ge(threshold).astype("int8")
    mismatches = predicted.ne(complete["target"])
    by_target = {}
    for value, group in complete.groupby("target"):
        q = group["proficiencia"].quantile([0, .01, .25, .5, .75, .99, 1])
        by_target[str(int(value))] = {str(index): float(number) for index, number in q.items()}
    return {
        "threshold": threshold,
        "rows": int(len(frame)), "complete_rows": int(len(complete)),
        "missing": int(frame["proficiencia"].isna().sum()),
        "by_target_quantiles": by_target,
        "correlation_with_binary_target": float(complete["proficiencia"].corr(complete["target"])),
        "threshold_accuracy": float((~mismatches).mean()),
        "threshold_mismatches": int(mismatches.sum()),
        "threshold_reconstruction_share_all_rows": float((~mismatches).sum() / len(frame)),
        "target_1_below_threshold": int(((complete["target"] == 1) & (complete["proficiencia"] < threshold)).sum()),
        "target_0_at_or_above_threshold": int(((complete["target"] == 0) & (complete["proficiencia"] >= threshold)).sum()),
        "minimum_target_1": float(complete.loc[complete["target"].eq(1), "proficiencia"].min()),
        "maximum_target_0": float(complete.loc[complete["target"].eq(0), "proficiencia"].max()),
    }


def candidate_matrix(summary: dict) -> pd.DataFrame:
    rows = [
        ["proficiencia", "Silver alunos / avaliação 2024", "aluno", 2024, "resultado contínuo", "4 - derivada do próprio resultado", "não", "100%", "alto", "determinístico", "baixo", "NÃO USAR"],
        ["serie", "Silver alunos", "aluno", 2024, "etapa escolar", "2 - contemporânea à avaliação", "contemporânea", "100%", "baixo", "baixo", "baixo", "NÃO USAR" if summary["categoricals"]["serie"]["constant"] else "INVESTIGAR MAIS"],
        ["caderno", "Silver alunos", "aluno", 2024, "código de caderno sem definição local", "2 - contemporânea à avaliação", "não", "100%", "médio", "artefato técnico", "baixo", "NÃO USAR"],
        ["presenca", "Silver alunos", "aluno", 2024, "elegibilidade", "2 - contemporânea à avaliação", "contemporânea", "100%", "médio", "constante após filtro", "baixo", "NÃO USAR"],
        ["preenchimento_caderno", "Silver alunos", "aluno", 2024, "elegibilidade", "3 - pós-avaliação", "não", "100%", "alto", "constante após filtro", "baixo", "NÃO USAR"],
        ["peso_aluno", "Silver alunos", "aluno", 2024, "peso amostral", "5 - temporalidade incerta", "incerta", "100%", "baixo", "possível desenho amostral", "baixo", "TESTAR como sample_weight"],
        ["dependencia_administrativa_escola", "Censo Escolar 2023", "escola", 2023, "contexto escolar", "1 - pré-avaliação", "sim", "a medir por id_escola", "baixo", "baixo", "médio", "TESTAR"],
        ["localizacao_urbana_rural_escola", "Censo Escolar 2023", "escola", 2023, "territorial escolar", "1 - pré-avaliação", "sim", "a medir por id_escola", "baixo", "baixo", "médio", "TESTAR"],
        ["porte_matriculas_escola", "Censo Escolar 2023", "escola", 2023, "porte agregado", "1 - pré-avaliação", "sim", "a medir por id_escola", "baixo", "baixo", "médio", "TESTAR"],
        ["infraestrutura_escola", "Censo Escolar 2023", "escola", 2023, "recursos/infraestrutura", "1 - pré-avaliação", "sim", "a medir por id_escola", "baixo", "baixo", "alto", "TESTAR"],
        ["perfil_docentes_agregado", "Censo Escolar 2023", "escola", 2023, "docentes agregado", "1 - pré-avaliação", "sim", "a medir", "médio", "baixo", "alto", "INVESTIGAR MAIS"],
        ["INSE_escola_anterior", "Indicadores educacionais Inep", "escola", "<=2023", "nível socioeconômico", "1 - pré-avaliação", "sim se edição anterior", "a medir", "baixo", "baixo", "médio", "INVESTIGAR MAIS"],
        ["rendimento_fluxo_anterior", "Indicadores/Censo Escolar", "escola", "<=2023", "histórico escolar agregado", "1 - pré-avaliação", "sim", "a medir", "baixo", "baixo", "médio", "TESTAR"],
        ["caracteristicas_individuais_pre_avaliacao", "fonte não encontrada no projeto", "aluno", "<=2023", "individual", "5 - temporalidade incerta", "incerta", "desconhecida", "incerto", "incerto", "alto", "INVESTIGAR MAIS"],
    ]
    return pd.DataFrame(rows, columns=[
        "feature", "fonte", "granularidade", "ano", "tipo", "categoria_temporal", "disponivel_pre_target",
        "cobertura", "risco_leakage", "risco_proxy", "custo_integracao", "recomendacao",
    ])


def build_summary(data: pd.DataFrame) -> dict:
    filtered = filter_official_population(data)
    filtered["target_binary"] = normalize_target(filtered["alfabetizado"])
    categoricals = categorical_audit(filtered, ["serie", "caderno", "presenca", "preenchimento_caderno"])
    summary = {
        "source_rows": int(len(data)), "official_population_rows": int(len(filtered)),
        "official_X_before": official_X_unchanged(), "official_X_changed": False,
        "column_classification": {
            column: {"classification": values[0], "justification": values[1]}
            for column, values in COLUMN_CLASSIFICATION.items()
        },
        "categoricals": categoricals,
        "proficiency": proficiency_statistics(filtered["proficiencia"], filtered["target_binary"]),
        "school_sources_in_phase2": {
            "attribute_table_found": False,
            "available_key": "id_escola somente em alunos/alunos_integrado",
            "tables": ["alunos", "municipio", "uf", "meta_municipio", "meta_uf", "meta_brasil", "atlas_uf"],
        },
        "phase2_source_inventory": [
            {"source": "alunos", "location": "Silver/S3", "granularity": "ano + id_aluno", "keys": ["ano", "id_aluno"], "years": "2023-2024", "useful": ["serie", "caderno", "peso_aluno", "id_escola"], "leakage": "proficiencia/alfabetizado são resultados", "integration_cost": "baixo"},
            {"source": "alunos_integrado", "location": "Silver/S3", "granularity": "aluno", "keys": ["ano", "id_aluno"], "years": "2023-2024", "useful": ["contexto municipal agregado"], "leakage": "alto para agregados contemporâneos", "integration_cost": "baixo"},
            {"source": "municipio", "location": "Silver/S3", "granularity": "ano + município + rede", "keys": ["ano", "id_municipio", "rede"], "years": "2023-2024", "useful": ["taxa", "Português", "níveis"], "leakage": "usar somente histórico", "integration_cost": "baixo"},
            {"source": "uf/atlas_uf", "location": "Silver/S3", "granularity": "ano + UF + rede / UF", "keys": ["ano", "sigla_uf", "rede"], "years": "histórico disponível", "useful": ["indicadores educacionais", "IDHM"], "leakage": "baixo se histórico", "integration_cost": "baixo"},
            {"source": "metas", "location": "Silver/S3", "granularity": "município, UF ou Brasil", "keys": ["localidade", "ano"], "years": "2024-2030", "useful": ["metas", "participação"], "leakage": "avaliar temporalidade de campos realizados", "integration_cost": "baixo"},
            {"source": "Gold existente", "location": "Gold/S3", "granularity": "município+rede/UF/Brasil", "keys": ["localidade", "rede", "ano"], "years": "2023-2024", "useful": ["features oficiais atuais"], "leakage": "controlado pelo recorte histórico", "integration_cost": "já integrado"},
        ],
        "test_accessed": False, "model_retrained": False,
    }
    return summary


def create_figures(data: pd.DataFrame, summary: dict) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(data["proficiencia"].min(), data["proficiencia"].max(), 61)
    plt.figure(figsize=(9, 5))
    for target, label, color in [(0, "Não alfabetizado", "#c44e52"), (1, "Alfabetizado", "#4c72b0")]:
        values = data.loc[data["target_binary"].eq(target), "proficiencia"].dropna()
        plt.hist(values, bins=bins, alpha=.55, density=True, label=label, color=color)
    plt.axvline(THRESHOLD, color="black", linestyle="--", label="corte 743")
    plt.xlabel("Proficiência 2024"); plt.ylabel("Densidade"); plt.title("Proficiência do mesmo evento por target")
    plt.legend(); plt.tight_layout(); plt.savefig(IMAGE_DIR / "01_proficiency_by_target.png", dpi=150); plt.close()

    for number, column in enumerate(("serie", "caderno"), start=2):
        table = data.groupby(column, dropna=False, observed=True)["target_binary"].agg(students="size", target_rate="mean").reset_index()
        table = table.sort_values("students", ascending=False).head(30)
        plt.figure(figsize=(9, 5)); plt.bar(table[column].astype(str), table["target_rate"], color="#55a868")
        plt.xticks(rotation=45, ha="right"); plt.ylabel("Taxa observada de alfabetização")
        plt.title(f"Target por {column} (descritivo; não recomenda uso)")
        plt.tight_layout(); plt.savefig(IMAGE_DIR / f"0{number}_{column}_by_target.png", dpi=150); plt.close()


def render_lineage(summary: dict) -> str:
    p = summary["proficiency"]
    return f"""# Auditoria da linhagem do target `alfabetizado`

## Conclusão

No batch real da Fase 2, `alfabetizado` já existe na tabela
`basedosdados.br_inep_avaliacao_alfabetizacao.alunos`. A query Bronze seleciona
simultaneamente `dados.alfabetizado` e `dados.proficiencia`, decodificando apenas
o primeiro pelo `dicionario`. A Silver normaliza nomes/tipos e remove duplicatas;
não calcula o target. A Fase 3 converte `Sim`/`Não` para 1/0 sem redefinir a regra.

A documentação oficial do Inep define 743 pontos na escala de proficiência como
o corte a partir do qual a criança é considerada alfabetizada:
https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/avaliacao-da-alfabetizacao

O threshold 743 também aparece no gerador de streaming da Fase 2, mas esse código
simula eventos e não é a origem do batch oficial.

## Evidência empírica na Silver oficial

- população filtrada: {summary['official_population_rows']:,} alunos;
- proficiência completa: {p['complete_rows']:,};
- correlação com target binário: {p['correlation_with_binary_target']:.6f};
- accuracy ao reconstruir por `proficiencia >= 743`: {p['threshold_accuracy']:.6%};
- divergências: {p['threshold_mismatches']:,};
- menor proficiência na classe 1: {p['minimum_target_1']};
- maior proficiência na classe 0: {p['maximum_target_0']}.

Classificação final: **leakage direto**. A proficiência é produzida no mesmo
evento e operacionaliza a definição do target. Não deve entrar em X, mesmo que
eleve fortemente as métricas.

## Evidências de código

- Fase 2 `src/bronze/queries.py`: seleciona as duas colunas da fonte e faz o join
  do dicionário de `alfabetizado`;
- Fase 2 `src/silver/transform.py`: apenas normalização, deduplicação e tipos;
- Fase 2 `src/silver/catalog.py`: proficiência tem somente regra de faixa;
- Fase 3 `src/preprocessing/build_modeling_dataset.py`: converte os rótulos do
  target para binário;
- Fase 2 `src/streaming/events.py`: usa 743 apenas na simulação Kafka.
"""


def render_candidates(summary: dict, matrix: pd.DataFrame) -> str:
    c = summary["categoricals"]
    def line(column):
        values = ", ".join(f"{row['value']} ({row['count']:,})" for row in c[column]["values"])
        return f"- `{column}`: cardinalidade {c[column]['cardinality_including_na']}; constante={c[column]['constant']}; {values}."
    return f"""# Auditoria de candidatas a features

## Silver alunos

{line('serie')}
{line('caderno')}
{line('presenca')}
{line('preenchimento_caderno')}

`presenca` e `preenchimento_caderno` são critérios do filtro e ficam constantes.
`serie` deve ser descartada por ser constante. O projeto não documenta o
significado detalhado dos 22 códigos de `caderno`; a hipótese de versão/formulário
é plausível, mas permanece **evidência insuficiente**. Como o código nasce no
evento de aplicação e pode capturar artefatos técnicos, não é recomendado.

O inventário completo e sua justificativa estão no JSON. Metadados iniciados por
`_` são técnicos. IDs não entram crus em X. `peso_aluno` permanece candidato
apenas a `sample_weight`.

## Fontes existentes

Fase 2/S3 contém `alunos`, `alunos_integrado`, município, UF, metas municipais,
estaduais/nacionais, `atlas_uf` e as quatro tabelas Gold. Não foi encontrada
tabela com atributos de escola, turma ou docente. `id_escola` existe e permite
um join futuro, mas o projeto atual não fornece os atributos.

## Tech Challenge e fonte externa

O PDF oficial exige Gold da Fase 2 e um modelo supervisionado com variáveis
educacionais, territoriais e socioeconômicas. Ele **permite**, sem obrigar,
enriquecimento com IBGE, Censo Escolar, FUNDEB, Atlas, PNAD, Cadastro Único e
fontes socioeconômicas regionais.

Como extensão externa, o Censo Escolar 2023 é a opção mais compatível. O Inep
publica microdados anuais e coleta informações de estabelecimentos, gestores,
turmas, alunos e profissionais escolares:
https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar

Antes de integrar, devem ser confirmados no dicionário de 2023: compatibilidade
do código da escola, cobertura dos `id_escola`, granularidade e disponibilidade
temporal. Candidatas plausíveis são dependência administrativa, localização
urbana/rural, porte por matrículas e infraestrutura. Resultados de 2024,
proficiência, preenchimento da prova e qualquer variável derivada do resultado
são alto risco.

## Conteúdo curricular disponível

Não foram encontrados materiais curriculares separados no repositório. A única
fonte FIAP disponível é o PDF oficial do desafio, que exige tratamento de data
leakage, engenharia de atributos, transformação, separação entre treino,
validação e teste, replicabilidade e generalização. A matriz temporal e esta
auditoria de linhagem são metodologia do grupo/extensão, não conteúdo específico
atribuído à FIAP.

## Cenários

1. **Manter modelo contextual:** menor esforço e melhor alinhamento às perguntas
   territoriais; mantém a limitação individual e desempenho próximo do atual.
2. **Enriquecer com escola:** testar Censo Escolar 2023 por `id_escola`; esforço
   médio/alto, cobertura ainda a medir e potencial para diferenciar escolas do
   mesmo município.
3. **Features individuais legítimas:** nenhuma fonte pré-avaliação foi encontrada
   no projeto. Investigar somente se existir fonte autorizada e temporalmente
   anterior; não usar proficiência, caderno ou resultado da própria avaliação.

Nenhum cenário foi escolhido pelo grupo. Cenários 2 e 3 permanecem **REVISAR COM
O GRUPO**. A matriz operacional está em `feature_candidate_matrix.csv`.
"""


def main() -> None:
    raw = load_silver_projection()
    summary = build_summary(raw)
    filtered = filter_official_population(raw)
    filtered["target_binary"] = normalize_target(filtered["alfabetizado"])
    matrix = candidate_matrix(summary)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LINEAGE_PATH.write_text(render_lineage(summary), encoding="utf-8")
    CANDIDATE_PATH.write_text(render_candidates(summary, matrix), encoding="utf-8")
    matrix.to_csv(MATRIX_PATH, index=False)
    create_figures(filtered, summary)
    print(json.dumps({
        "rows": summary["official_population_rows"], "proficiency": summary["proficiency"],
        "categoricals": summary["categoricals"], "official_X_changed": False,
        "test_accessed": False, "model_retrained": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
