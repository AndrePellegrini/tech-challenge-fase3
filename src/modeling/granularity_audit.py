"""Auditoria da granularidade do X oficial, sem treino ou acesso ao teste.

Execute: python -m src.modeling.granularity_audit
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import positive_class_score
from src.modeling.split import get_modeling_columns, split_by_municipality
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "granularity_audit.md"
SUMMARY_PATH = ROOT / "reports" / "granularity_audit_summary.json"
IMAGE_DIR = ROOT / "images" / "modeling" / "granularity"
TARGET = "alfabetizado"
CONTEXT = ["id_municipio", "rede"]

FEATURE_GRANULARITY = {
    "rede": "rede",
    "sigla_uf": "UF",
    "taxa_alfabetizacao_municipio_2023": "município + rede",
    "media_portugues_municipio_2023": "município + rede",
    "percentual_participacao_municipio_2023": "município + rede",
    "total_alunos_municipio_2023": "município + rede",
    "pct_alfabetizados_municipio_2023": "município + rede",
    "proficiencia_media_ponderada_2023": "município + rede",
    "meta_alfabetizacao_municipio_2024": "município + rede",
    "gap_para_meta_municipio_2024": "município + rede (derivada)",
    "idhm": "UF",
    "idhm_educacao": "UF",
    "idhm_renda": "UF",
    "idhm_longevidade": "UF",
    "atingiu_meta_municipio_2024": "município + rede (derivada)",
    "gold_historico_disponivel": "disponibilidade do contexto município + rede",
}


def official_feature_columns() -> list[str]:
    columns = list(get_modeling_columns()["X"])
    forbidden = {
        TARGET, "ano", "id_aluno", "id_municipio", "id_escola",
        "id_municipio_nome", "sigla_uf_nome", "peso_aluno",
    }
    if forbidden.intersection(columns):
        raise ValueError("X contém target, identificador, nome auxiliar ou peso.")
    if set(columns) != set(FEATURE_GRANULARITY):
        raise ValueError("Mapa de granularidade não corresponde ao X oficial.")
    return columns


def identify_profiles(X: pd.DataFrame) -> np.ndarray:
    """IDs exatos de linhas iguais; NaN participa como uma categoria consistente."""
    expected = official_feature_columns()
    if list(X.columns) != expected:
        raise ValueError("A análise exige exatamente o X bruto oficial e na ordem oficial.")
    return X.groupby(expected, dropna=False, sort=False, observed=True).ngroup().to_numpy()


def profile_target_stats(profile_ids: np.ndarray, y: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"profile_id": profile_ids, TARGET: y.to_numpy(copy=False)})
    stats = frame.groupby("profile_id", sort=False)[TARGET].agg(students="size", class_1="sum")
    stats["class_0"] = stats["students"] - stats["class_1"]
    stats["target_rate"] = stats["class_1"] / stats["students"]
    stats["type"] = np.select(
        [stats["class_1"].eq(0), stats["class_0"].eq(0)],
        ["pure_0", "pure_1"], default="mixed",
    )
    return stats.reset_index()


def profile_size_summary(stats: pd.DataFrame) -> dict:
    sizes = stats["students"]
    duplicated_students = int(stats.loc[sizes.gt(1), "students"].sum())
    return {
        "unique_profiles": int(len(stats)),
        "students_in_duplicated_profiles": duplicated_students,
        "duplicated_profile_student_share": float(duplicated_students / sizes.sum()),
        "students_per_profile": {
            "mean": float(sizes.mean()), "median": float(sizes.median()),
            "p90": float(sizes.quantile(.90)), "p95": float(sizes.quantile(.95)),
            "p99": float(sizes.quantile(.99)), "max": int(sizes.max()),
        },
    }


def size_bands(stats: pd.DataFrame) -> list[dict]:
    labels = ["1", "2-5", "6-10", "11-50", "51-100", "101-500", ">500"]
    band = pd.cut(
        stats["students"], bins=[0, 1, 5, 10, 50, 100, 500, np.inf],
        labels=labels, include_lowest=True,
    )
    table = stats.assign(band=band).groupby("band", observed=False)["students"].agg(
        profiles="size", students="sum",
    ).reindex(labels, fill_value=0)
    total = stats["students"].sum()
    return [
        {"band": index, "profiles": int(row.profiles), "students": int(row.students),
         "student_share": float(row.students / total)}
        for index, row in table.iterrows()
    ]


def profile_type_summary(stats: pd.DataFrame) -> list[dict]:
    table = stats.groupby("type")["students"].agg(profiles="size", students="sum")
    total = stats["students"].sum()
    return [
        {"type": kind, "profiles": int(row.profiles), "students": int(row.students),
         "profile_share": float(row.profiles / len(stats)),
         "student_share": float(row.students / total)}
        for kind, row in table.iterrows()
    ]


def uncertainty_bands(stats: pd.DataFrame) -> list[dict]:
    mixed = stats.loc[stats["type"].eq("mixed")].copy()
    labels = ["0-10%", "10-30%", "30-45%", "45-55%", "55-70%", "70-90%", "90-100%"]
    # right=False evita dupla inclusão nas fronteiras; 100% não ocorre em mistos.
    mixed["band"] = pd.cut(
        mixed["target_rate"], bins=[0, .10, .30, .45, .55, .70, .90, 1.0000001],
        labels=labels, right=False, include_lowest=True,
    )
    table = mixed.groupby("band", observed=False)["students"].agg(profiles="size", students="sum")
    total = stats["students"].sum()
    return [
        {"band": index, "profiles": int(row.profiles), "students": int(row.students),
         "dataset_student_share": float(row.students / total)}
        for index, row in table.reindex(labels, fill_value=0).iterrows()
    ]


def context_stats(dataset: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, dict]:
    stats = dataset.groupby(keys, dropna=False, observed=True)[TARGET].agg(
        students="size", class_1="sum", target_rate="mean", target_variance="var",
    ).reset_index()
    stats["class_0"] = stats["students"] - stats["class_1"]
    stats["mixed"] = stats["class_0"].gt(0) & stats["class_1"].gt(0)
    return stats, {
        "groups": int(len(stats)),
        "mixed_groups": int(stats["mixed"].sum()),
        "mixed_group_share": float(stats["mixed"].mean()),
        "students_in_mixed_groups": int(stats.loc[stats["mixed"], "students"].sum()),
        "mixed_group_student_share": float(stats.loc[stats["mixed"], "students"].sum() / len(dataset)),
        "students_per_group": {
            "mean": float(stats["students"].mean()), "median": float(stats["students"].median()),
            "p90": float(stats["students"].quantile(.90)), "p95": float(stats["students"].quantile(.95)),
            "p99": float(stats["students"].quantile(.99)), "max": int(stats["students"].max()),
        },
        "target_rate": stats["target_rate"].describe(percentiles=[.1, .25, .5, .75, .9]).to_dict(),
        "within_group_variance": stats["target_variance"].describe(percentiles=[.1, .25, .5, .75, .9]).to_dict(),
    }


def profiles_per_context(dataset: pd.DataFrame, profile_ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    frame = dataset[CONTEXT].copy()
    frame["profile_id"] = profile_ids
    frame["context_id"] = frame.groupby(
        CONTEXT, dropna=False, observed=True, sort=False,
    ).ngroup()
    counts = frame.groupby(CONTEXT, dropna=False, observed=True)["profile_id"].nunique().rename("profiles").reset_index()
    one = counts["profiles"].eq(1)
    student_counts = frame.groupby(CONTEXT, dropna=False, observed=True).size().rename("students").reset_index()
    counts = counts.merge(student_counts, on=CONTEXT, how="left", validate="one_to_one")
    contexts_per_profile = frame.groupby("profile_id", sort=False)["context_id"].nunique()
    return counts, {
        "groups": int(len(counts)),
        "groups_with_one_profile": int(one.sum()),
        "one_profile_group_share": float(one.mean()),
        "groups_with_multiple_profiles": int((~one).sum()),
        "students_in_one_profile_groups": int(counts.loc[one, "students"].sum()),
        "one_profile_group_student_share": float(counts.loc[one, "students"].sum() / len(dataset)),
        "max_profiles_in_context": int(counts["profiles"].max()),
        "profiles_shared_by_multiple_contexts": int(contexts_per_profile.gt(1).sum()),
        "shared_profile_share": float(contexts_per_profile.gt(1).mean()),
        "max_contexts_sharing_profile": int(contexts_per_profile.max()),
    }


def prediction_identity(dataset: pd.DataFrame, probabilities: np.ndarray) -> dict:
    frame = dataset[CONTEXT].copy()
    frame["probability"] = probabilities
    grouped = frame.groupby(CONTEXT, dropna=False, observed=True)["probability"].agg(
        students="size", unique_probabilities="nunique",
    ).reset_index()
    identical = grouped["unique_probabilities"].eq(1)
    return {
        "groups": int(len(grouped)),
        "identical_probability_groups": int(identical.sum()),
        "identical_probability_group_share": float(identical.mean()),
        "students_in_identical_probability_groups": int(grouped.loc[identical, "students"].sum()),
        "identical_probability_student_share": float(grouped.loc[identical, "students"].sum() / len(dataset)),
        "max_unique_probabilities_in_group": int(grouped["unique_probabilities"].max()),
    }


def empirical_profile_bound(stats: pd.DataFrame) -> dict:
    correct = np.maximum(stats["class_0"], stats["class_1"]).sum()
    errors = np.minimum(stats["class_0"], stats["class_1"]).sum()
    total = stats["students"].sum()
    return {
        "majority_class_accuracy_within_partition": float(correct / total),
        "minimum_conflict_errors_within_partition": int(errors),
        "minimum_conflict_error_rate_within_partition": float(errors / total),
        "interpretation": "limite descritivo in-sample; não estima desempenho em municípios inéditos",
    }


def variance_decomposition(dataset: pd.DataFrame, keys: list[str]) -> dict:
    y = dataset[TARGET].astype(float)
    overall = float(y.mean())
    grouped = dataset.groupby(keys, dropna=False, observed=True)[TARGET].agg(["size", "mean"])
    total_variance = float(((y - overall) ** 2).mean())
    between = float((grouped["size"] * (grouped["mean"] - overall) ** 2).sum() / len(dataset))
    within = float((grouped["size"] * grouped["mean"] * (1 - grouped["mean"])).sum() / len(dataset))
    return {
        "groups": int(len(grouped)), "target_mean": overall,
        "total_variance": total_variance, "between_group_variance": between,
        "within_group_variance": within,
        "between_share": float(between / total_variance),
        "within_share": float(within / total_variance),
        "identity_residual": float(total_variance - between - within),
    }


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return None if pd.isna(value) else value.item()
    return None if isinstance(value, float) and pd.isna(value) else value


def create_figures(profile_stats: pd.DataFrame, context: pd.DataFrame, size_table: list[dict], uncertainty: list[dict]) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    size = pd.DataFrame(size_table)
    plt.figure(figsize=(8, 5)); plt.bar(size["band"], size["students"]); plt.yscale("log")
    plt.ylabel("Alunos (escala log)"); plt.title("Alunos por faixa de tamanho do perfil X")
    plt.tight_layout(); plt.savefig(IMAGE_DIR / "01_students_by_profile_size.png", dpi=150); plt.close()

    mixed = profile_stats.loc[profile_stats["type"].eq("mixed")]
    plt.figure(figsize=(8, 5)); plt.hist(mixed["target_rate"], bins=30, weights=mixed["students"], color="#4c72b0")
    plt.xlabel("Taxa de alfabetização do perfil misto"); plt.ylabel("Alunos")
    plt.title("Ambiguidade entre alunos com X idêntico")
    plt.tight_layout(); plt.savefig(IMAGE_DIR / "02_mixed_profile_target_rate.png", dpi=150); plt.close()

    unc = pd.DataFrame(uncertainty)
    plt.figure(figsize=(8, 5)); plt.bar(unc["band"], unc["students"], color="#c44e52")
    plt.xticks(rotation=30); plt.ylabel("Alunos"); plt.title("Alunos por taxa do target em perfis mistos")
    plt.tight_layout(); plt.savefig(IMAGE_DIR / "03_mixed_profile_uncertainty.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5)); plt.hist(context["students"], bins=40, log=True, color="#55a868")
    plt.xlabel("Alunos por município + rede"); plt.ylabel("Grupos (escala log)")
    plt.title("Tamanho dos contextos município + rede")
    plt.tight_layout(); plt.savefig(IMAGE_DIR / "04_context_group_size.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5)); plt.hist(context["target_rate"], bins=30, weights=context["students"], color="#8172b3")
    plt.xlabel("Taxa observada no município + rede"); plt.ylabel("Alunos")
    plt.title("Distribuição contextual da alfabetização")
    plt.tight_layout(); plt.savefig(IMAGE_DIR / "05_context_target_rate.png", dpi=150); plt.close()


def render_markdown(summary: dict) -> str:
    p = summary["profiles"]
    types = {row["type"]: row for row in summary["profile_types"]}
    c = summary["municipality_network"]
    relation = summary["profile_vs_municipality_network"]
    pred = summary["validation_prediction_identity"]
    bound = summary["train_empirical_profile_bound"]
    variance = summary["variance_decomposition"]
    feature_rows = "\n".join(f"| `{name}` | {granularity} |" for name, granularity in FEATURE_GRANULARITY.items())
    size_rows = "\n".join(
        f"| {row['band']} | {row['profiles']:,} | {row['students']:,} | {row['student_share']:.2%} |"
        for row in summary["profile_size_bands"]
    )
    type_rows = "\n".join(
        f"| {row['type']} | {row['profiles']:,} | {row['students']:,} | {row['student_share']:.2%} |"
        for row in summary["profile_types"]
    )
    return f"""# Auditoria de granularidade das features

## Escopo

O target individual é requisito do Tech Challenge. O uso da Gold contextual é
requisito/arquitetura atual. Esta auditoria é uma decisão metodológica do grupo
e uma extensão diagnóstica; não é atribuída como técnica específica ensinada
pela FIAP. Nenhum modelo foi treinado, o teste permaneceu fechado e o X não foi
alterado.

## X oficial bruto ({summary['feature_count']} features)

| Feature | Granularidade conceitual |
|---|---|
{feature_rows}

## Perfis idênticos

Há {p['unique_profiles']:,} perfis de X entre {summary['students']:,} alunos.
Perfis duplicados representam {p['duplicated_profile_student_share']:.2%} dos
alunos. O tamanho médio é {p['students_per_profile']['mean']:.2f}, mediana
{p['students_per_profile']['median']:.0f}, p90 {p['students_per_profile']['p90']:.0f},
p95 {p['students_per_profile']['p95']:.0f}, p99 {p['students_per_profile']['p99']:.0f}
e máximo {p['students_per_profile']['max']:,}.

| Alunos por perfil | Perfis | Alunos | % alunos |
|---|---:|---:|---:|
{size_rows}

## Conflito do target

| Tipo | Perfis | Alunos | % alunos |
|---|---:|---:|---:|
{type_rows}

Perfis mistos contêm {types.get('mixed', {}).get('student_share', 0):.2%} dos
alunos. Pessoas com X exatamente igual frequentemente possuem targets distintos;
um classificador determinístico baseado somente nesse X não consegue separá-las.

## Município + rede

Existem {c['groups']:,} grupos município+rede. {c['mixed_group_share']:.2%} dos
grupos e {c['mixed_group_student_share']:.2%} dos alunos estão em grupos com as
duas classes. {relation['one_profile_group_share']:.2%} desses grupos têm um
único perfil X; {relation['groups_with_multiple_profiles']:,} têm múltiplos
perfis. Portanto, município+rede determina X em 100% dos casos observados. A
relação inversa não é sempre única: {relation['profiles_shared_by_multiple_contexts']:,}
perfis são compartilhados por mais de um contexto, pois contextos diferentes
podem ter os mesmos valores agregados, inclusive o padrão sem histórico Gold.

## Previsões na validação

Sem retreino, os pipelines existentes produziram probabilidades idênticas para
todos os alunos do grupo em {pred['logistic_regression']['identical_probability_group_share']:.2%}
dos grupos na regressão e {pred['decision_tree']['identical_probability_group_share']:.2%}
na árvore. Isso afeta, respectivamente,
{pred['logistic_regression']['identical_probability_student_share']:.2%} e
{pred['decision_tree']['identical_probability_student_share']:.2%} dos alunos da validação.

## Limite empírico no treino

Escolher a classe majoritária dentro de cada perfil do próprio treino produz
accuracy descritiva de {bound['majority_class_accuracy_within_partition']:.4f}.
O erro conflitante mínimo é {bound['minimum_conflict_errors_within_partition']:,}
({bound['minimum_conflict_error_rate_within_partition']:.2%}). Esse valor é
in-sample e usa o target para descrever ambiguidade; não é estimativa de
generalização nem teto de AUC em municípios inéditos.

## Variabilidade do target

Por município, {variance['municipality']['between_share']:.2%} da variância é
entre grupos e {variance['municipality']['within_share']:.2%} permanece dentro
dos municípios. Por município+rede, são
{variance['municipality_network']['between_share']:.2%} entre grupos e
{variance['municipality_network']['within_share']:.2%} dentro dos grupos.

## Interpretação

A tarefa formal continua sendo prever `alfabetizado` por aluno, mas a informação
disponível é quase inteiramente territorial e de rede. A leitura mais adequada é
uma probabilidade individual condicionada ao contexto, útil sobretudo para
identificar territórios/redes de maior risco. Ela não distingue adequadamente
alunos do mesmo contexto e não substitui diagnóstico pedagógico individual.

A auditoria fornece explicação plausível, não prova exclusiva, para AUCs próximas
de 0,62, pequeno ganho da árvore e dificuldade de generalização municipal. A
formulação permanece válida para o desafio se essa limitação ficar explícita.
Recomenda-se manter target e fluxo atuais, ajustar a interpretação e marcar o
enriquecimento futuro com features individuais como **REVISAR COM O GRUPO**.
"""


def main() -> None:
    dataset = pd.read_parquet(DATASET_PATH)
    contract = validate_dataset(dataset, strict_counts=True)
    if not contract["valido"]:
        raise ValueError(contract["errors"])
    features = official_feature_columns()
    profile_ids = identify_profiles(dataset[features])
    profiles = profile_target_stats(profile_ids, dataset[TARGET])
    size_summary = profile_size_summary(profiles)
    size_table = size_bands(profiles)
    types = profile_type_summary(profiles)
    uncertainty = uncertainty_bands(profiles)
    context_table, context_summary = context_stats(dataset, CONTEXT)
    _, relation = profiles_per_context(dataset, profile_ids)

    splits = split_by_municipality(dataset)
    validation = dataset.iloc[splits["validation"]]
    predictions = {}
    for model_name in ("logistic_regression", "decision_tree"):
        model = joblib.load(ROOT / "models" / f"{model_name}.joblib")
        probability = positive_class_score(model, validation[features])
        predictions[model_name] = prediction_identity(validation, probability)

    train = dataset.iloc[splits["train"]]
    train_profile_ids = identify_profiles(train[features])
    train_profiles = profile_target_stats(train_profile_ids, train[TARGET])
    summary = {
        "scope": "auditoria descritiva; sem causalidade, retreino ou acesso ao teste",
        "students": int(len(dataset)), "feature_count": len(features), "features": features,
        "feature_granularity": FEATURE_GRANULARITY,
        "profiles": size_summary, "profile_size_bands": size_table,
        "profile_types": types, "mixed_profile_uncertainty": uncertainty,
        "municipality_network": context_summary,
        "profile_vs_municipality_network": relation,
        "validation_prediction_identity": predictions,
        "train_empirical_profile_bound": empirical_profile_bound(train_profiles),
        "variance_decomposition": {
            "municipality": variance_decomposition(dataset, ["id_municipio"]),
            "municipality_network": variance_decomposition(dataset, CONTEXT),
        },
        "test_status": "fechado; nenhuma leitura posicional, previsão ou métrica",
        "recommendation": "manter formulação, ajustar interpretação e revisar enriquecimento futuro com features individuais",
        "review_with_group": [
            "interpretação territorial como uso principal",
            "incorporação futura de features individuais",
            "eventual reformulação do produto analítico sem mudar o target nesta etapa",
        ],
    }
    summary = _json_value(summary)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_markdown(summary), encoding="utf-8")
    create_figures(profiles, context_table, size_table, uncertainty)
    print(json.dumps({
        "students": summary["students"], "unique_profiles": summary["profiles"]["unique_profiles"],
        "mixed_profiles": next((row for row in types if row["type"] == "mixed"), None),
        "municipality_network": context_summary,
        "prediction_identity": predictions,
        "train_bound": summary["train_empirical_profile_bound"],
        "variance": summary["variance_decomposition"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
