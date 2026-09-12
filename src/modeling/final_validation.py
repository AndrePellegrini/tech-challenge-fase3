"""Seleção final usando exclusivamente treino e validação; teste permanece fechado."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.metrics import positive_class_score
from src.modeling.split import RANDOM_STATE, get_modeling_columns, split_by_municipality
from src.modeling.train_baselines import aggregate_tree_importance, build_pipeline
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images" / "modeling" / "final_validation"
MODELS = ROOT / "models"
FORBIDDEN = {"alfabetizado", "proficiencia", "id_aluno", "id_escola", "id_municipio"}
THRESHOLDS = (.30, .35, .40, .45, .50, .55, .60, .65, .70)


def assert_feature_contract(features: list[str]) -> None:
    forbidden = FORBIDDEN.intersection(features)
    if forbidden:
        raise ValueError(f"Colunas proibidas em X: {sorted(forbidden)}")
    if features != list(get_modeling_columns()["X"]):
        raise ValueError("A validação final exige exatamente as 16 features congeladas.")


def threshold_metrics(y, score, threshold: float) -> dict:
    if not 0 < threshold < 1:
        raise ValueError("Threshold deve estar entre zero e um.")
    prediction = np.asarray(score) >= threshold
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def evaluate(estimator, X, y, threshold: float = .5) -> dict:
    score = positive_class_score(estimator, X)
    return {**threshold_metrics(y, score, threshold), "roc_auc": float(roc_auc_score(y, score))}


def model_candidates() -> list[dict]:
    return [
        {"name": "dummy_prior", "model": DummyClassifier(strategy="prior"), "scale": False,
         "note": "referência sem discriminação"},
        {"name": "logistic_baseline", "model": LogisticRegression(solver="saga", C=1.0, max_iter=150,
         tol=1e-3, random_state=RANDOM_STATE), "scale": True, "note": "baseline linear"},
        {"name": "logistic_tuned_c_0_1", "model": LogisticRegression(solver="saga", C=.1, max_iter=150,
         tol=1e-3, random_state=RANDOM_STATE), "scale": True, "note": "regularização maior"},
        {"name": "logistic_tuned_c_0_3", "model": LogisticRegression(solver="saga", C=.3, max_iter=150,
         tol=1e-3, random_state=RANDOM_STATE), "scale": True, "note": "regularização moderada"},
        {"name": "decision_tree_baseline", "model": DecisionTreeClassifier(max_depth=14,
         min_samples_leaf=100, random_state=RANDOM_STATE), "scale": False, "note": "baseline vigente"},
        {"name": "decision_tree_d5_l100", "model": DecisionTreeClassifier(max_depth=5,
         min_samples_leaf=100, random_state=RANDOM_STATE), "scale": False, "note": "árvore simples"},
        {"name": "decision_tree_d7_l100", "model": DecisionTreeClassifier(max_depth=7,
         min_samples_leaf=100, random_state=RANDOM_STATE), "scale": False, "note": "árvore moderada"},
        {"name": "decision_tree_d10_l100", "model": DecisionTreeClassifier(max_depth=10,
         min_samples_leaf=100, random_state=RANDOM_STATE), "scale": False, "note": "profundidade intermediária"},
        {"name": "decision_tree_d10_l500", "model": DecisionTreeClassifier(max_depth=10,
         min_samples_leaf=500, random_state=RANDOM_STATE), "scale": False, "note": "folhas mais estáveis"},
        {"name": "decision_tree_d15_l500", "model": DecisionTreeClassifier(max_depth=15,
         min_samples_leaf=500, random_state=RANDOM_STATE), "scale": False, "note": "profunda com regularização"},
        {"name": "random_forest_controlled", "model": RandomForestClassifier(n_estimators=40,
         max_depth=10, min_samples_leaf=200, max_features="sqrt", n_jobs=2,
         random_state=RANDOM_STATE), "scale": False, "note": "tentativa única, memória limitada"},
    ]


def choose_candidate(results: pd.DataFrame) -> str:
    eligible = results.loc[~results["model"].eq("dummy_prior")].copy()
    best_auc = eligible["validation_roc_auc"].max()
    near = eligible.loc[eligible["validation_roc_auc"].ge(best_auc - .002)]
    near = near.sort_values(["auc_gap", "validation_f1", "training_seconds"],
                            ascending=[True, False, True])
    return str(near.iloc[0]["model"])


def fit_candidate(spec, features, X_train, y_train, X_validation, y_validation,
                  sample_weight=None) -> tuple[Pipeline, dict]:
    pipe = build_pipeline(clone(spec["model"]), features=features, scale_numeric=spec["scale"])
    started = time.perf_counter()
    fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    pipe.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.perf_counter() - started
    train_metrics = evaluate(pipe, X_train, y_train)
    validation_metrics = evaluate(pipe, X_validation, y_validation)
    return pipe, {
        "model": spec["name"], "parameters": spec["model"].get_params(deep=False),
        "training_seconds": elapsed,
        "encoded_features": len(pipe.named_steps["preprocessing"].get_feature_names_out()),
        "train": train_metrics, "validation": validation_metrics,
        "auc_gap": train_metrics["roc_auc"] - validation_metrics["roc_auc"],
        "observation": spec["note"],
    }


def plot_curves(pipelines, X_validation, y_validation) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    for name, pipe in pipelines.items():
        score = positive_class_score(pipe, X_validation)
        fpr, tpr, _ = roc_curve(y_validation, score)
        plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_validation, score):.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC — somente validação"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(IMAGES / "roc_validation.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 6))
    for name, pipe in pipelines.items():
        score = positive_class_score(pipe, X_validation)
        precision, recall, _ = precision_recall_curve(y_validation, score)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall — somente validação")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(IMAGES / "precision_recall_validation.png", dpi=150); plt.close()


def municipal_analysis(data, score, threshold):
    frame = data[["id_municipio", "rede", "alfabetizado", "taxa_alfabetizacao_municipio_2023",
                  "meta_alfabetizacao_municipio_2024", "gap_para_meta_municipio_2024"]].copy()
    frame["score"] = score; frame["risk"] = np.asarray(score) < threshold
    frame["absolute_error"] = (frame["score"] - frame["alfabetizado"]).abs()
    out = frame.groupby(["id_municipio", "rede"], dropna=False).agg(
        alunos=("alfabetizado", "size"), taxa_observada=("alfabetizado", "mean"),
        probabilidade_media_prevista=("score", "mean"), percentual_previsto_em_risco=("risk", "mean"),
        historico_2023=("taxa_alfabetizacao_municipio_2023", "mean"),
        meta_2024=("meta_alfabetizacao_municipio_2024", "mean"),
        gap=("gap_para_meta_municipio_2024", "mean"), erro_absoluto_medio=("absolute_error", "mean"),
    ).reset_index()
    out["prioridade_analitica"] = (1 - out["probabilidade_media_prevista"]).rank(pct=True)
    return out.sort_values("prioridade_analitica", ascending=False)


def main() -> None:
    data = pd.read_parquet(DATASET_PATH)
    contract = validate_dataset(data, strict_counts=True)
    if not contract["valido"]:
        raise ValueError(contract["errors"])
    features = list(get_modeling_columns()["X"]); assert_feature_contract(features)
    splits = split_by_municipality(data)
    # Proteção central: nenhuma seleção, transformação ou previsão referencia splits['test'].
    train = data.iloc[splits["train"]]
    validation = data.iloc[splits["validation"]]
    if set(train["id_municipio"]).intersection(validation["id_municipio"]):
        raise ValueError("Overlap municipal entre treino e validação.")
    X_train, y_train = train[features], train["alfabetizado"]
    X_validation, y_validation = validation[features], validation["alfabetizado"]

    raw, pipelines = [], {}
    for spec in model_candidates():
        pipe, result = fit_candidate(spec, features, X_train, y_train, X_validation, y_validation)
        raw.append(result); pipelines[spec["name"]] = pipe
        print(f"{spec['name']}: {result['training_seconds']:.1f}s AUC val={result['validation']['roc_auc']:.4f}")
    comparison = pd.json_normalize(raw, sep="_")
    selected = choose_candidate(comparison)
    selected_pipe = pipelines[selected]
    validation_score = positive_class_score(selected_pipe, X_validation)
    threshold = pd.DataFrame([threshold_metrics(y_validation, validation_score, t) for t in THRESHOLDS])
    # Compromisso explícito: menor limiar com recall >= 0,85; desempate por F1. Caso vazio, 0,5.
    viable = threshold.loc[threshold["recall"].ge(.85)]
    selected_threshold = float(viable.sort_values(["f1", "precision"], ascending=False).iloc[0]["threshold"]) if len(viable) else .5

    chosen_spec = next(spec for spec in model_candidates() if spec["name"] == selected)
    weighted_pipe, weighted = fit_candidate(chosen_spec, features, X_train, y_train,
                                             X_validation, y_validation,
                                             sample_weight=train["peso_aluno"].to_numpy())
    unweighted_auc = float(comparison.loc[comparison["model"].eq(selected), "validation_roc_auc"].iloc[0])
    use_weight = weighted["validation"]["roc_auc"] > unweighted_auc + .002
    if use_weight:
        selected_pipe = weighted_pipe
        validation_score = positive_class_score(selected_pipe, X_validation)

    tree_rows = comparison.loc[comparison["model"].str.startswith(("decision_tree", "random_forest"))]
    best_tree_name = str(tree_rows.sort_values("validation_roc_auc", ascending=False).iloc[0]["model"])
    importance = aggregate_tree_importance(pipelines[best_tree_name], features)
    importance.insert(0, "model", best_tree_name)
    plot_names = [selected, "logistic_baseline", best_tree_name]
    plot_curves({name: pipelines[name] for name in dict.fromkeys(plot_names)}, X_validation, y_validation)
    municipal = municipal_analysis(validation, validation_score, selected_threshold)

    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    comparison.to_csv(REPORTS / "final_model_comparison.csv", index=False)
    threshold.to_csv(REPORTS / "threshold_analysis.csv", index=False)
    importance.to_csv(REPORTS / "final_feature_importance.csv", index=False)
    municipal.to_csv(REPORTS / "final_municipal_validation_analysis.csv", index=False)
    joblib.dump(selected_pipe, MODELS / "final_candidate_validation.joblib", compress=3)
    shap = {"executed": False, "reason": "pacote shap ausente; não instalado para evitar dependência pesada",
            "test_accessed": False}
    (REPORTS / "shap_summary.json").write_text(json.dumps(shap, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    metadata = {
        "dataset": str(DATASET_PATH), "shape": list(data.shape), "random_state": RANDOM_STATE,
        "features": features, "preprocessing": "mediana numérica; moda categórica; one-hot sparse; escala somente logística",
        "selected_model": selected, "selected_parameters": chosen_spec["model"].get_params(deep=False),
        "selected_threshold": selected_threshold, "sample_weight_used": use_weight,
        "sample_weight_experiment": weighted, "test_accessed": False,
        "grouped_cv": "não executada: custo redundante após holdout municipal em 1,85M linhas",
        "shap": shap, "versions": {"python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    (REPORTS / "final_validation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    protocol = f"""# Protocolo congelado do modelo candidato\n\nDataset: `data/processed/modeling_dataset_2024_gold.parquet`, 1.851.828 × 24.\nVersão da decisão: 2026-09-09. Semente: {RANDOM_STATE}.\n\nFeatures candidatas finais ({len(features)}): {', '.join(f'`{x}`' for x in features)}.\nExcluídas: target, peso e todos os identificadores; `proficiencia` é proibida por leakage direto.\n\nPreprocessing: mediana numérica, moda categórica/booleana, OneHotEncoder sparse com categorias desconhecidas ignoradas; escala sparse somente para regressão logística.\n\nAlgoritmo candidato: `{selected}`. Parâmetros: `{chosen_spec['model'].get_params(deep=False)}`.\nThreshold candidato: {selected_threshold:.2f}. Sample weight oficial: {'sim' if use_weight else 'não'}.\nMétricas de seleção: ROC AUC principal; recall e F1; gap treino/validação; interpretabilidade, custo e simplicidade.\n\nA interpretação é territorial: o modelo estima probabilidade individual condicionada ao contexto municipal, rede e UF; alunos do mesmo município+rede recebem a mesma probabilidade. Não é diagnóstico individual nem previsão oficial de meta.\n\nLimitações: features contextuais repetidas, generalização territorial limitada, ausência de atributos individuais legítimos e incerteza sobre o significado operacional de `peso_aluno`. SHAP não foi executado porque o pacote não está instalado.\n\n**O conjunto de teste só será acessado após aprovação deste protocolo.**\n"""
    (REPORTS / "final_model_protocol.md").write_text(protocol, encoding="utf-8")
    report = f"""# Validação final controlada\n\nForam comparados Dummy, três regressões logísticas, seis árvores e uma Random Forest controlada usando apenas treino e validação. O candidato `{selected}` foi escolhido entre modelos a até 0,002 da melhor AUC, priorizando menor gap, F1 e custo.\n\nA análise de threshold selecionou {selected_threshold:.2f} entre {', '.join(map(str, THRESHOLDS))}, exigindo recall mínimo de 0,85 e usando F1/precision no desempate. O uso de peso foi {'adotado' if use_weight else 'rejeitado'}: exige ganho de AUC superior a 0,002 e seu significado segue **REVISAR COM O GRUPO**.\n\nA Random Forest foi limitada a 40 árvores, profundidade 10, folha mínima 200 e dois jobs. GroupKFold não foi executado pelo custo em 1,85 milhão de linhas. SHAP não foi executado porque a dependência está ausente; não houve instalação.\n\nFeature importance é associativa, agregada para as 16 colunas originais e não causal. A análise municipal/rede é ferramenta de priorização analítica.\n\n**TEST SET NÃO ACESSADO.** Nenhuma seleção, fit, transformação, predição, SHAP ou métrica usou o teste.\n"""
    (REPORTS / "final_validation_results.md").write_text(report, encoding="utf-8")
    print(json.dumps({"selected": selected, "threshold": selected_threshold,
                      "sample_weight": use_weight, "test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
