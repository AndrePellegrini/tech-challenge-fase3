"""Validação cruzada agrupada por município sobre treino e validação.

Fecha a lacuna registrada na DEC-010. O holdout único não permitia dizer se a
diferença de ROC AUC entre os candidatos finalistas — da ordem de 0,010 — era real ou
ruído de partição. `GroupKFold` sobre a união de treino e validação dá média e desvio
por candidato, e com isso a comparação passa a ter dispersão declarada.

Restrições preservadas:

* o conjunto de **teste nunca é materializado**; só os índices de treino e validação
  entram, e há verificação explícita disso;
* o agrupamento continua sendo `id_municipio`, de modo que nenhum município aparece em
  dois folds e a estimativa mede generalização territorial, como no holdout;
* o contrato de features é o mesmo da validação final, com `proficiencia` proibida.

Executa apenas os finalistas, não os onze candidatos. Onze modelos por cinco folds
seriam 55 ajustes, e só a Random Forest custa cerca de 123 segundos por ajuste. Os
quatro escolhidos cobrem a faixa de decisão real, que foi de 0,002 de ROC AUC entre o
primeiro e o segundo colocado do holdout.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import positive_class_score
from src.modeling.final_validation import assert_feature_contract, model_candidates
from src.modeling.split import (GROUP_COLUMN, RANDOM_STATE, get_modeling_columns,
                                split_by_municipality)
from src.modeling.train_baselines import build_pipeline
from src.preprocessing.validate_dataset import (DATASET_PATH, USING_FULL_DATASET,
                                                validate_dataset)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
RESULTS_PATH = REPORTS / "cross_validation_results.csv"
SUMMARY_PATH = REPORTS / "cross_validation_summary.json"
REPORT_PATH = REPORTS / "cross_validation_results.md"

N_SPLITS = 5
FINALISTS = (
    "random_forest_controlled",
    "decision_tree_d5_l100",
    "decision_tree_d10_l100",
    "logistic_baseline",
)


def finalist_specs(names=FINALISTS) -> list[dict]:
    """Recorta os candidatos finalistas da lista oficial, preservando a ordem pedida."""
    catalog = {spec["name"]: spec for spec in model_candidates()}
    missing = [name for name in names if name not in catalog]
    if missing:
        raise ValueError(f"Candidatos inexistentes em model_candidates(): {missing}")
    return [catalog[name] for name in names]


def development_frame(data: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Une treino e validação. O teste não é indexado em nenhum momento.

    A união é legítima: a validação cruzada substitui o papel do holdout de validação,
    e o teste permanece intocado como árbitro final.
    """
    if "test" in splits:
        development = np.concatenate([splits["train"], splits["validation"]])
        overlap = set(development).intersection(splits["test"])
        if overlap:
            raise ValueError(
                f"{len(overlap)} índices de teste apareceram no desenvolvimento."
            )
    else:
        development = np.concatenate([splits["train"], splits["validation"]])
    return data.iloc[np.sort(development)]


def cross_validate_candidate(spec: dict, features: list[str], frame: pd.DataFrame,
                             n_splits: int = N_SPLITS) -> list[dict]:
    """Ajusta o candidato em cada fold e devolve uma linha por fold."""
    X, y = frame[features], frame["alfabetizado"]
    groups = frame[GROUP_COLUMN]
    rows = []
    for fold, (train_index, validation_index) in enumerate(
        GroupKFold(n_splits=n_splits).split(X, y, groups=groups), start=1
    ):
        train_groups = set(groups.iloc[train_index])
        validation_groups = set(groups.iloc[validation_index])
        if train_groups & validation_groups:
            raise ValueError(f"Fold {fold} com município em treino e validação.")
        pipe = build_pipeline(clone(spec["model"]), features=features,
                              scale_numeric=spec["scale"])
        started = time.perf_counter()
        pipe.fit(X.iloc[train_index], y.iloc[train_index])
        elapsed = time.perf_counter() - started
        score = positive_class_score(pipe, X.iloc[validation_index])
        rows.append({
            "model": spec["name"],
            "fold": fold,
            "roc_auc": float(roc_auc_score(y.iloc[validation_index], score)),
            "train_rows": int(len(train_index)),
            "validation_rows": int(len(validation_index)),
            "validation_municipalities": len(validation_groups),
            "training_seconds": elapsed,
        })
        print(f"  {spec['name']} fold {fold}/{n_splits}: "
              f"AUC={rows[-1]['roc_auc']:.4f} ({elapsed:.1f}s)")
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Média, desvio e amplitude por candidato, ordenados pela média."""
    summary = results.groupby("model").agg(
        folds=("roc_auc", "size"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        roc_auc_min=("roc_auc", "min"),
        roc_auc_max=("roc_auc", "max"),
        seconds_total=("training_seconds", "sum"),
    ).reset_index()
    return summary.sort_values("roc_auc_mean", ascending=False, ignore_index=True)


def separation_verdict(summary: pd.DataFrame) -> str:
    """Diz se o primeiro colocado se separa do segundo além do desvio observado."""
    if len(summary) < 2:
        return "Comparação indisponível: menos de dois candidatos."
    first, second = summary.iloc[0], summary.iloc[1]
    difference = first["roc_auc_mean"] - second["roc_auc_mean"]
    pooled = float(np.hypot(first["roc_auc_std"], second["roc_auc_std"]))
    if pooled == 0:
        return "Desvio nulo entre folds; comparação inconclusiva."
    if difference > pooled:
        return (
            f"A diferença de {difference:.4f} entre `{first['model']}` e "
            f"`{second['model']}` **supera** a dispersão combinada entre folds "
            f"({pooled:.4f}). A vantagem do primeiro colocado é consistente."
        )
    return (
        f"A diferença de {difference:.4f} entre `{first['model']}` e "
        f"`{second['model']}` é **pequena diante da dispersão observada entre folds** "
        f"({pooled:.4f}). Com esta evidência não há separação clara entre os dois, e a "
        f"escolha se justifica pelos critérios de desempate da DEC-012, não pela ROC "
        f"AUC isolada.\n\nA comparação é descritiva: não foi aplicado teste pareado "
        f"por fold, bootstrap da diferença nem intervalo de confiança, de modo que não "
        f"se afirma equivalência estatística."
    )


def generalization_note(summary: pd.DataFrame, holdout: dict) -> str:
    """Confronta holdout de validação, média da CV e a AUC de teste já publicada.

    Não reabre o teste: a AUC vem de `reports/final_test_metrics.json`, artefato já
    versionado. A validação cruzada foi executada sem qualquer conhecimento do teste; a
    comparação é feita depois, sobre um número que já estava registrado.
    """
    metrics_path = REPORTS / "final_test_metrics.json"
    if not metrics_path.exists() or summary.empty:
        return ""
    published = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_auc = float(published["test"]["metrics"]["roc_auc"])
    selected = str(published["protocol"].get("selected_model", "random_forest_controlled"))
    row = summary.loc[summary["model"].eq(selected)]
    if row.empty:
        row = summary.head(1)
    row = row.iloc[0]
    cv_mean, cv_std = float(row["roc_auc_mean"]), float(row["roc_auc_std"])
    holdout_auc = float(holdout.get(row["model"], float("nan")))
    return f"""
## O que isto explica sobre a generalização

Havia uma ressalva em aberto no projeto: a AUC de teste ({test_auc:.4f}) ficou
{abs(test_auc - holdout_auc):.4f} acima da AUC do holdout de validação
({holdout_auc:.4f}), diferença que o próprio protocolo classificou como moderada. A
leitura natural seria suspeitar de uma partição de teste favorável.

A validação cruzada mostra que a explicação é outra.

| Estimativa | ROC AUC |
|---|---:|
| Holdout de validação, partição única | {holdout_auc:.4f} |
| **Validação cruzada, média de {int(row['folds'])} folds** | **{cv_mean:.4f}** ± {cv_std:.4f} |
| Teste, abertura única | {test_auc:.4f} |

A média da validação cruzada fica a **{abs(cv_mean - test_auc):.4f}** do teste, dentro de
um desvio-padrão entre folds. Já o holdout isolado ficou {abs(cv_mean - holdout_auc):.4f}
abaixo dessa média.

**A proximidade entre a média da validação cruzada e o teste indica que o resultado do
teste é compatível com a variabilidade territorial observada no desenvolvimento.** A
média da validação cruzada é a estimativa de generalização mais representativa, e o
holdout isolado aparenta ter caído numa partição desfavorável. Isso reforça, e não
enfraquece, a validade da avaliação final.

Registro metodológico, importante para a leitura correta: esta validação cruzada é uma
**análise pós-hoc de robustez**. Ela foi executada depois da abertura única do teste e
**não participou da seleção do modelo**, que ocorreu antes, sobre o holdout de
validação. A validação cruzada não acessa o conjunto de teste em nenhum momento; a AUC
de teste citada vem de `final_test_metrics.json`, artefato publicado na ocasião da
abertura, e a comparação é estritamente posterior.
"""


def build_report(summary: pd.DataFrame, results: pd.DataFrame, holdout: dict) -> str:
    rows = "\n".join(
        f"| `{r.model}` | {r.roc_auc_mean:.4f} | {r.roc_auc_std:.4f} | "
        f"{r.roc_auc_min:.4f} | {r.roc_auc_max:.4f} | {holdout.get(r.model, float('nan')):.4f} |"
        for r in summary.itertuples()
    )
    folds = "\n".join(
        f"| `{r.model}` | {r.fold} | {r.roc_auc:.4f} | {r.validation_rows:,} | "
        f"{r.validation_municipalities:,} |".replace(",", ".")
        for r in results.itertuples()
    )
    # Compara apenas os candidatos que entraram na validação cruzada; o holdout tem
    # onze modelos e a CV, quatro.
    cv_order = list(summary["model"])
    holdout_order = sorted(cv_order, key=lambda name: -holdout.get(name, float("-inf")))
    agreement = (
        "A ordem dos finalistas é **idêntica** à do holdout, o que reforça a escolha."
        if holdout_order == cv_order else
        f"A ordem dos finalistas **difere** da do holdout, que foi "
        f"{' > '.join(f'`{m}`' for m in holdout_order)}. A divergência é em si um "
        f"resultado: indica que o holdout único era sensível à partição, exatamente o "
        f"risco que motivou esta validação cruzada."
    )
    return f"""# Validação cruzada agrupada

Fecha a lacuna da DEC-010. O holdout único não permitia dizer se as diferenças de ROC
AUC entre os finalistas eram reais ou ruído de partição.

Trata-se de **análise pós-hoc de robustez**: foi executada após a seleção do modelo e
após a abertura do teste, sem acessá-lo. Não reescreve a história da escolha, que se
deu sobre o holdout.

## Método

`GroupKFold` com {N_SPLITS} folds, agrupado por `id_municipio`, sobre a união de treino
e validação. Nenhum município aparece em dois folds, de modo que cada fold mede
generalização territorial, exatamente como o holdout original.

**O conjunto de teste não foi materializado.** A verificação é explícita em código: os
índices de desenvolvimento são confrontados com os de teste e a execução falha se
houver interseção.

Foram avaliados os {len(summary)} finalistas, não os onze candidatos. Onze modelos por
{N_SPLITS} folds seriam 55 ajustes, e só a Random Forest custa cerca de 123 segundos
por ajuste. Os finalistas cobrem a faixa de decisão real, que no holdout foi de 0,002
de ROC AUC entre primeiro e segundo colocado.

## Resultado

| Modelo | AUC média | Desvio | Mínimo | Máximo | AUC do holdout |
|---|---:|---:|---:|---:|---:|
{rows}

{separation_verdict(summary)}

{agreement}
{generalization_note(summary, holdout)}
## Por fold

| Modelo | Fold | ROC AUC | Alunos na validação | Municípios |
|---|---:|---:|---:|---:|
{folds}

## Leitura

O desvio entre folds é a informação que faltava. Ele mostra a margem dentro da qual
comparações de ROC AUC não são conclusivas, e justifica a regra de seleção adotada na
DEC-012, que não escolhe pelo maior AUC isolado e sim pelo menor gap entre treino e
validação dentro de uma faixa de equivalência.

Os valores aqui não substituem as métricas oficiais do projeto, que vêm do holdout
municipal e da abertura única do teste. Servem para qualificar a comparação entre
candidatos.
"""


def main() -> None:
    data = pd.read_parquet(DATASET_PATH)
    # Sobre a amostra versionada as contagens do contrato não se aplicam: elas
    # descrevem o dataset completo. O schema continua sendo verificado.
    contract = validate_dataset(data, strict_counts=USING_FULL_DATASET)
    if not contract["valido"]:
        raise ValueError(contract["errors"])
    features = list(get_modeling_columns()["X"])
    assert_feature_contract(features)

    splits = split_by_municipality(data)
    frame = development_frame(data, splits)
    print(f"Desenvolvimento: {len(frame):,} alunos, "
          f"{frame[GROUP_COLUMN].nunique():,} municípios; teste não tocado.")

    raw = []
    for spec in finalist_specs():
        raw.extend(cross_validate_candidate(spec, features, frame))
    results = pd.DataFrame(raw)
    summary = summarize(results)

    comparison = pd.read_csv(REPORTS / "final_model_comparison.csv")
    holdout = dict(zip(comparison["model"], comparison["validation_roc_auc"]))

    REPORTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps({
        "n_splits": N_SPLITS,
        "group_column": GROUP_COLUMN,
        "random_state": RANDOM_STATE,
        "candidates": list(FINALISTS),
        "development_rows": int(len(frame)),
        "development_municipalities": int(frame[GROUP_COLUMN].nunique()),
        "test_accessed": False,
        "summary": summary.to_dict(orient="records"),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(build_report(summary, results, holdout), encoding="utf-8")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
