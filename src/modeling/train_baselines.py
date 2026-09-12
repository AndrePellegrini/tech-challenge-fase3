"""Treina baselines no split municipal e avalia somente treino/validação.

Execute: python -m src.modeling.train_baselines
O conjunto de teste permanece reservado e não recebe predict nesta etapa.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.metrics import classification_metrics, positive_class_score
from src.modeling.split import RANDOM_STATE, get_modeling_columns, split_by_municipality, validate_split
from src.preprocessing.model_preprocessor import build_model_preprocessor
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "baseline_results.json"
COMPARISON_PATH = ROOT / "reports" / "baseline_model_comparison.csv"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"
MUNICIPAL_PATH = ROOT / "reports" / "municipal_validation_analysis.csv"
MODEL_DIR = ROOT / "models"
IMAGE_DIR = ROOT / "images" / "modeling"


def model_specs() -> dict[str, tuple[object, bool]]:
    """Estimadores simples; limites das árvores reduzem custo e sobreajuste."""
    return {
        "dummy_prior": (DummyClassifier(strategy="prior"), False),
        "logistic_regression": (LogisticRegression(
            solver="saga", max_iter=150, tol=1e-3, random_state=RANDOM_STATE,
        ), True),
        "decision_tree": (DecisionTreeClassifier(
            max_depth=14, min_samples_leaf=100, random_state=RANDOM_STATE,
        ), False),
    }


def build_pipeline(estimator, *, features: list[str], scale_numeric: bool) -> Pipeline:
    return Pipeline([
        ("preprocessing", build_model_preprocessor(
            feature_columns=features, scale_numeric=scale_numeric,
        )),
        ("model", estimator),
    ])


def _overfit_note(train: dict, validation: dict) -> str:
    f1_gap = train["f1"] - validation["f1"]
    auc_gap = train["roc_auc"] - validation["roc_auc"]
    if f1_gap > .05 or auc_gap > .05:
        return f"sinal relevante: gap F1={f1_gap:.4f}, gap ROC AUC={auc_gap:.4f}"
    return f"sem gap > 0,05: gap F1={f1_gap:.4f}, gap ROC AUC={auc_gap:.4f}"


def _source_feature(encoded_name: str, source_columns: list[str]) -> str:
    clean = encoded_name.split("__", 1)[-1]
    matches = [column for column in source_columns if clean == column or clean.startswith(column + "_")]
    return max(matches, key=len) if matches else clean


def aggregate_tree_importance(pipeline: Pipeline, source_columns: list[str]) -> pd.DataFrame:
    names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    encoded = pd.DataFrame({"encoded_feature": names, "importance": values})
    encoded["feature"] = encoded["encoded_feature"].map(lambda name: _source_feature(name, source_columns))
    return (
        encoded.groupby("feature", as_index=False)["importance"].sum()
        .sort_values("importance", ascending=False, ignore_index=True)
    )


def municipal_analysis(dataset: pd.DataFrame, positions: np.ndarray, pipeline: Pipeline, features: list[str]) -> pd.DataFrame:
    part = dataset.iloc[positions]
    probability = positive_class_score(pipeline, part[features])
    frame = part[[
        "id_municipio", "id_municipio_nome", "sigla_uf", "alfabetizado",
        "taxa_alfabetizacao_municipio_2023", "meta_alfabetizacao_municipio_2024",
        "gap_para_meta_municipio_2024",
    ]].copy()
    frame["probabilidade_alfabetizacao"] = probability
    frame["erro_assinado"] = probability - frame["alfabetizado"]
    frame["erro_absoluto"] = frame["erro_assinado"].abs()
    frame["previsto_em_risco_limiar_0_5"] = probability < .5
    # Uma linha por município. Nome/UF podem estar ausentes em registros sem Gold,
    # portanto não podem compor a chave do agrupamento.
    grouped = frame.groupby("id_municipio", dropna=False)
    result = grouped.agg(
        id_municipio_nome=("id_municipio_nome", "first"),
        sigla_uf=("sigla_uf", "first"),
        alunos=("alfabetizado", "size"),
        probabilidade_media_prevista=("probabilidade_alfabetizacao", "mean"),
        taxa_observada=("alfabetizado", "mean"),
        erro_medio_assinado=("erro_assinado", "mean"),
        erro_absoluto_medio=("erro_absoluto", "mean"),
        percentual_previsto_em_risco=("previsto_em_risco_limiar_0_5", "mean"),
        taxa_historica_2023=("taxa_alfabetizacao_municipio_2023", "mean"),
        meta_2024=("meta_alfabetizacao_municipio_2024", "mean"),
        gap_para_meta=("gap_para_meta_municipio_2024", "mean"),
    ).reset_index()
    result["percentil_risco_previsto"] = (1 - result["probabilidade_media_prevista"]).rank(pct=True)
    result["percentil_volume"] = result["alunos"].rank(pct=True)
    result["percentil_gap_alto"] = result["gap_para_meta"].rank(pct=True)
    result["percentil_historico_baixo"] = (-result["taxa_historica_2023"]).rank(pct=True)
    dimensions = [
        "percentil_risco_previsto", "percentil_volume",
        "percentil_gap_alto", "percentil_historico_baixo",
    ]
    result["dimensoes_prioridade_disponiveis"] = result[dimensions].notna().sum(axis=1)
    result["prioridade_analitica"] = result[dimensions].mean(axis=1, skipna=True)
    return result.sort_values(
        ["dimensoes_prioridade_disponiveis", "prioridade_analitica"],
        ascending=False, ignore_index=True,
    )


def _plot_results(comparison: pd.DataFrame, importance: pd.DataFrame) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    comparison.set_index("model")[["f1", "recall", "roc_auc"]].plot.bar(
        figsize=(9, 5), ylim=(0, 1), rot=20, title="Baselines — validação municipal",
    )
    plt.ylabel("Métrica")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "06_baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    if len(importance):
        top = importance.head(15).sort_values("importance")
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"], top["importance"], color="#4c72b0")
        plt.title("Importância nativa agregada — melhor baseline de árvore")
        plt.xlabel("Importância (associativa, não causal)")
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "07_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()


def main() -> None:
    dataset = pd.read_parquet(DATASET_PATH)
    contract = validate_dataset(dataset, strict_counts=True)
    if not contract["valido"]:
        raise ValueError(contract["errors"])
    splits = split_by_municipality(dataset)
    split_report = validate_split(dataset, splits)
    if not split_report["valid"]:
        raise ValueError(split_report["errors"])

    features = list(get_modeling_columns()["X"])
    train = dataset.iloc[splits["train"]]
    validation = dataset.iloc[splits["validation"]]
    X_train, y_train = train[features], train["alfabetizado"]
    X_validation, y_validation = validation[features], validation["alfabetizado"]
    MODEL_DIR.mkdir(exist_ok=True)
    results, pipelines, warnings_by_model = {}, {}, {}

    for name, (estimator, scale_numeric) in model_specs().items():
        pipeline = build_pipeline(estimator, features=features, scale_numeric=scale_numeric)
        started = time.process_time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X_train, y_train)
        elapsed = time.process_time() - started
        train_metrics = classification_metrics(pipeline, X_train, y_train)
        validation_metrics = classification_metrics(pipeline, X_validation, y_validation)
        encoded_features = int(len(pipeline.named_steps["preprocessing"].get_feature_names_out()))
        warnings_by_model[name] = [str(item.message) for item in caught]
        results[name] = {
            "train": train_metrics, "validation": validation_metrics,
            "training_seconds": elapsed, "encoded_features": encoded_features,
            "overfitting": _overfit_note(train_metrics, validation_metrics),
            "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in caught),
        }
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib", compress=3)
        pipelines[name] = pipeline
        print(f"{name}: {elapsed:.1f}s; validation F1={validation_metrics['f1']:.4f}; AUC={validation_metrics['roc_auc']:.4f}")

    comparison = pd.DataFrame([
        {"model": name, **values["validation"], "training_seconds": values["training_seconds"],
         "encoded_features": values["encoded_features"], "overfitting": values["overfitting"]}
        for name, values in results.items()
    ]).drop(columns=["confusion_matrix"]).sort_values(["roc_auc", "f1", "recall"], ascending=False)
    best_name = comparison.iloc[0]["model"]

    reduced_features = [column for column in features if column not in {
        "gap_para_meta_municipio_2024", "atingiu_meta_municipio_2024",
    }]
    reduced = build_pipeline(
        LogisticRegression(solver="saga", max_iter=150, tol=1e-3, random_state=RANDOM_STATE),
        features=reduced_features, scale_numeric=True,
    )
    started = time.process_time()
    reduced.fit(train[reduced_features], y_train)
    redundancy = {
        "version_a_all_features": results["logistic_regression"]["validation"],
        "version_b_without_gap_and_reached": classification_metrics(
            reduced, validation[reduced_features], y_validation,
        ),
        "version_b_training_seconds": time.process_time() - started,
        "removed": ["gap_para_meta_municipio_2024", "atingiu_meta_municipio_2024"],
    }

    tree_candidates = [name for name in ("decision_tree", "random_forest") if name in results]
    importance_model = max(tree_candidates, key=lambda name: results[name]["validation"]["roc_auc"])
    importance = aggregate_tree_importance(pipelines[importance_model], features)
    importance.insert(0, "model", importance_model)
    municipal = municipal_analysis(dataset, splits["validation"], pipelines[best_name], features)

    report = {
        "random_state": RANDOM_STATE,
        "test_status": "reservado; nenhuma previsão ou métrica calculada",
        "selection_rule": "maior ROC AUC de validação; F1 e recall como desempate/contexto; limiar padrão 0,5",
        "best_validation_baseline": best_name,
        "results": results,
        "warnings": warnings_by_model,
        "redundancy_experiment": redundancy,
        "importance_model": importance_model,
        "shap": "não executado ainda por custo computacional e dependência adicional; importância nativa atende ao diagnóstico inicial",
        "skipped_models": {
            "random_forest": "interrompido após tempo desproporcional e cerca de 888 MB no ambiente local; REVISAR COM O GRUPO",
        },
        "municipal_analysis": "diagnóstico derivado apenas da validação; não é previsão oficial de meta municipal",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison.to_csv(COMPARISON_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    municipal.to_csv(MUNICIPAL_PATH, index=False)
    _plot_results(comparison, importance)
    print(json.dumps({
        "best_validation_baseline": best_name,
        "reports": [str(REPORT_PATH), str(COMPARISON_PATH), str(IMPORTANCE_PATH), str(MUNICIPAL_PATH)],
        "test_status": report["test_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
