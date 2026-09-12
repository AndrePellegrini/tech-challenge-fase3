"""Audita a semântica do limiar no modelo salvo, usando somente validação."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score)

from src.evaluation.metrics import positive_class_score
from src.modeling.final_validation import IMAGES, REPORTS, THRESHOLDS
from src.modeling.split import get_modeling_columns, split_by_municipality
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "final_candidate_validation.joblib"


def risk_probability(prob_alfabetizado) -> np.ndarray:
    probability = np.asarray(prob_alfabetizado, dtype=float)
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilidade fora de [0, 1].")
    return 1.0 - probability


def semantic_threshold_metrics(y, prob_alfabetizado, threshold: float) -> dict:
    """Retorna métricas simétricas; linhas/colunas da matriz seguem [0, 1]."""
    if not 0 < threshold < 1:
        raise ValueError("Threshold deve estar entre zero e um.")
    y = np.asarray(y)
    prediction = np.asarray(prob_alfabetizado) >= threshold
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    true_risk_predicted_literate = int(matrix[0, 1])
    true_risk = int(matrix[0].sum())
    return {
        "threshold_prob_alfabetizado": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision_class_0_risk": float(precision_score(y, prediction, pos_label=0, zero_division=0)),
        "recall_class_0_risk": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "f1_class_0_risk": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "precision_class_1_literate": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        "recall_class_1_literate": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "f1_class_1_literate": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "specificity_for_class_1": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "true_risk_predicted_literate": true_risk_predicted_literate,
        "true_risk_predicted_literate_share_of_true_risk": true_risk_predicted_literate / true_risk,
        "true_risk_predicted_literate_share_of_validation": true_risk_predicted_literate / len(y),
        "false_risk_alerts_literate_predicted_risk": int(matrix[1, 0]),
        "confusion_matrix_true_rows_predicted_columns_0_1": matrix.tolist(),
    }


def corrected_municipal_analysis(validation, prob_alfabetizado, threshold=None):
    frame = validation[["id_municipio", "rede", "alfabetizado",
                        "taxa_alfabetizacao_municipio_2023",
                        "meta_alfabetizacao_municipio_2024",
                        "gap_para_meta_municipio_2024"]].copy()
    frame["prob_alfabetizado"] = prob_alfabetizado
    frame["prob_risco"] = risk_probability(prob_alfabetizado)
    frame["erro_absoluto_prob_alfabetizado"] = (frame["prob_alfabetizado"] - frame["alfabetizado"]).abs()
    if threshold is not None:
        frame["previsto_em_risco"] = frame["prob_alfabetizado"] < threshold
    aggregation = {
        "alunos": ("alfabetizado", "size"), "taxa_observada_alfabetizacao": ("alfabetizado", "mean"),
        "probabilidade_media_alfabetizacao": ("prob_alfabetizado", "mean"),
        "probabilidade_media_risco": ("prob_risco", "mean"),
        "historico_2023": ("taxa_alfabetizacao_municipio_2023", "mean"),
        "meta_2024": ("meta_alfabetizacao_municipio_2024", "mean"),
        "gap": ("gap_para_meta_municipio_2024", "mean"),
        "erro_absoluto_medio": ("erro_absoluto_prob_alfabetizado", "mean"),
    }
    if threshold is not None:
        aggregation["percentual_previsto_em_risco"] = ("previsto_em_risco", "mean")
    out = frame.groupby(["id_municipio", "rede"], dropna=False).agg(**aggregation).reset_index()
    out["ranking_prioridade_risco"] = out["probabilidade_media_risco"].rank(method="min", ascending=False)
    return out.sort_values(["probabilidade_media_risco", "alunos"], ascending=[False, False])


def main() -> None:
    data = pd.read_parquet(DATASET_PATH)
    contract = validate_dataset(data, strict_counts=True)
    if not contract["valido"]:
        raise ValueError(contract["errors"])
    splits = split_by_municipality(data)
    # Única partição materializada: validação. O teste não é indexado.
    validation = data.iloc[splits["validation"]]
    features = list(get_modeling_columns()["X"])
    model = joblib.load(MODEL_PATH)
    prob_alfabetizado = positive_class_score(model, validation[features])
    prob_risco = risk_probability(prob_alfabetizado)
    threshold = pd.DataFrame([
        semantic_threshold_metrics(validation["alfabetizado"], prob_alfabetizado, value)
        for value in THRESHOLDS
    ])
    threshold.to_csv(REPORTS / "threshold_analysis.csv", index=False)

    balanced_row = threshold.sort_values(["balanced_accuracy", "f1_class_0_risk"], ascending=False).iloc[0]
    high_risk = threshold.loc[threshold["recall_class_0_risk"].ge(.70)]
    risk_row = high_risk.sort_values(["precision_class_0_risk", "balanced_accuracy"], ascending=False).iloc[0]
    municipal = corrected_municipal_analysis(validation, prob_alfabetizado, threshold=None)
    municipal.to_csv(REPORTS / "final_municipal_validation_analysis.csv", index=False)

    IMAGES.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(
        (validation["alfabetizado"].to_numpy() == 0).astype(int), prob_risco,
    )
    plt.figure(figsize=(7, 6)); plt.plot(recall, precision, color="#c44e52")
    plt.xlabel("Recall — risco (classe 0)"); plt.ylabel("Precision — risco (classe 0)")
    plt.title("Precision–Recall da não alfabetização — validação")
    plt.tight_layout(); plt.savefig(IMAGES / "precision_recall_risk_validation.png", dpi=150); plt.close()

    audit = {
        "class_semantics": {"0": "não alfabetizado / risco", "1": "alfabetizado"},
        "predict_proba_column_1": "P(alfabetizado)", "prob_risco": "1 - P(alfabetizado)",
        "previous_threshold_interpretation_correct": False,
        "correction": "reduzir o threshold de P(alfabetizado) aumenta previsões de classe 1 e reduz detecção da classe 0",
        "balanced_objective_candidate": balanced_row.to_dict(),
        "high_risk_sensitivity_example": risk_row.to_dict(),
        "recommendation": "C - não congelar threshold binário; ordenar territórios por prob_risco contínua",
        "municipal_analysis": "ranking por probabilidade média de risco; sem classificação binária congelada",
        "test_accessed": False,
    }
    (REPORTS / "threshold_semantics_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata_path = REPORTS / "final_validation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "selected_threshold": None,
        "threshold_status": "não congelado após auditoria semântica",
        "risk_probability": "1 - predict_proba[:, 1]",
        "territorial_priority": "ranking contínuo por probabilidade média de risco",
        "threshold_semantics_corrected": True,
        "test_accessed": False,
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
