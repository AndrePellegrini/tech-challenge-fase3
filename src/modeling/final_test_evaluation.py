"""Avaliação final única do protocolo congelado no conjunto de teste."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.evaluation.metrics import positive_class_score
from src.modeling.final_validation import assert_feature_contract
from src.modeling.split import RANDOM_STATE, get_modeling_columns, split_by_municipality
from src.modeling.threshold_semantics_audit import risk_probability
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images" / "modeling" / "final_test"
MODEL_PATH = ROOT / "models" / "final_candidate_validation.joblib"
METRICS_PATH = REPORTS / "final_test_metrics.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_metrics(y, prob_alfabetizado) -> dict:
    """Métricas finais em limiar convencional 0,50, com ambas as classes."""
    y = np.asarray(y); probability = np.asarray(prob_alfabetizado)
    prediction = probability >= .5
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "accuracy_threshold_0_50": float(accuracy_score(y, prediction)),
        "balanced_accuracy_threshold_0_50": float(balanced_accuracy_score(y, prediction)),
        "precision_class_0_risk": float(precision_score(y, prediction, pos_label=0, zero_division=0)),
        "recall_class_0_risk": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "f1_class_0_risk": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "precision_class_1_literate": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        "recall_class_1_literate": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "f1_class_1_literate": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "confusion_matrix_true_rows_predicted_columns_0_1": matrix.tolist(),
    }


def territorial_analysis(test, prob_alfabetizado, keys):
    frame = test[[*keys, "id_municipio_nome", "sigla_uf", "alfabetizado",
                  "taxa_alfabetizacao_municipio_2023", "meta_alfabetizacao_municipio_2024",
                  "gap_para_meta_municipio_2024"]].copy()
    frame["prob_alfabetizado"] = prob_alfabetizado
    frame["prob_risco"] = risk_probability(prob_alfabetizado)
    frame["erro_absoluto"] = (frame["prob_alfabetizado"] - frame["alfabetizado"]).abs()
    out = frame.groupby(keys, dropna=False).agg(
        id_municipio_nome=("id_municipio_nome", "first"), sigla_uf=("sigla_uf", "first"),
        alunos=("alfabetizado", "size"), taxa_observada_alfabetizacao=("alfabetizado", "mean"),
        probabilidade_media_alfabetizacao=("prob_alfabetizado", "mean"),
        probabilidade_media_risco=("prob_risco", "mean"),
        historico_2023=("taxa_alfabetizacao_municipio_2023", "mean"),
        meta_2024=("meta_alfabetizacao_municipio_2024", "mean"),
        gap=("gap_para_meta_municipio_2024", "mean"), erro_absoluto_medio=("erro_absoluto", "mean"),
    ).reset_index()
    out["ranking_risco_relativo"] = out["probabilidade_media_risco"].rank(method="min", ascending=False)
    return out.sort_values(["probabilidade_media_risco", "alunos"], ascending=[False, False])


def plot_final(y_test, p_test, y_validation, p_validation):
    IMAGES.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, p_test)
    plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, label=f"teste AUC={roc_auc_score(y_test,p_test):.4f}")
    plt.plot([0,1],[0,1],"--",color="gray"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
    plt.title("ROC final — teste"); plt.tight_layout(); plt.savefig(IMAGES/"roc_final_test.png",dpi=150); plt.close()
    for suffix, event, score, label in (
        ("literate", y_test, p_test, "alfabetização (classe 1)"),
        ("risk", (np.asarray(y_test)==0).astype(int), 1-np.asarray(p_test), "risco (classe 0)"),
    ):
        precision, recall, _ = precision_recall_curve(event, score)
        plt.figure(figsize=(7,6)); plt.plot(recall,precision); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision–Recall — {label} — teste"); plt.tight_layout()
        plt.savefig(IMAGES/f"precision_recall_{suffix}_test.png",dpi=150); plt.close()
    plt.figure(figsize=(8,5)); plt.hist(1-np.asarray(p_validation),bins=40,alpha=.55,density=True,label="validação")
    plt.hist(1-np.asarray(p_test),bins=40,alpha=.55,density=True,label="teste")
    plt.xlabel("Probabilidade de risco"); plt.ylabel("Densidade"); plt.legend(); plt.title("Risco: validação vs teste")
    plt.tight_layout(); plt.savefig(IMAGES/"risk_distribution_validation_vs_test.png",dpi=150); plt.close()


def main() -> None:
    if METRICS_PATH.exists():
        raise RuntimeError("Avaliação final já registrada; segunda abertura recusada.")
    data = pd.read_parquet(DATASET_PATH)
    contract = validate_dataset(data, strict_counts=True)
    if not contract["valido"]:
        raise ValueError(contract["errors"])
    features = list(get_modeling_columns()["X"]); assert_feature_contract(features)
    splits = split_by_municipality(data)
    train_groups = set(data.iloc[splits["train"]]["id_municipio"])
    validation = data.iloc[splits["validation"]]
    validation_groups = set(validation["id_municipio"])
    test = data.iloc[splits["test"]]  # PRIMEIRA E ÚNICA materialização autorizada.
    test_groups = set(test["id_municipio"])
    overlaps = {
        "test_train_municipalities": len(test_groups & train_groups),
        "test_validation_municipalities": len(test_groups & validation_groups),
        "test_train_students": len(set(test["id_aluno"]) & set(data.iloc[splits["train"]]["id_aluno"])),
        "test_validation_students": len(set(test["id_aluno"]) & set(validation["id_aluno"])),
    }
    if any(overlaps.values()):
        raise ValueError(f"Sobreposição territorial/individual: {overlaps}")
    model = joblib.load(MODEL_PATH)  # Congelado e treinado somente em TRAIN.
    p_validation = positive_class_score(model, validation[features])
    p_test = positive_class_score(model, test[features])
    validation_metrics = final_metrics(validation["alfabetizado"], p_validation)
    test_metrics = final_metrics(test["alfabetizado"], p_test)
    delta = test_metrics["roc_auc"] - validation_metrics["roc_auc"]
    magnitude = "muito pequena" if abs(delta)<.005 else "pequena" if abs(delta)<.02 else "moderada" if abs(delta)<.05 else "grande"
    municipal = territorial_analysis(test, p_test, ["id_municipio"])
    contexts = territorial_analysis(test, p_test, ["id_municipio", "rede"])
    municipal.to_csv(REPORTS/"final_test_municipal_analysis.csv",index=False)
    contexts.to_csv(REPORTS/"final_test_context_ranking.csv",index=False)
    plot_final(test["alfabetizado"],p_test,validation["alfabetizado"],p_validation)
    result = {
        "opened_once_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"fit_strategy":"modelo congelado ajustado somente em TRAIN; sem refit",
                     "random_state":RANDOM_STATE,"operational_threshold":None,
                     "descriptive_threshold":.5,"model_sha256":sha256(MODEL_PATH),
                     "dataset_sha256":sha256(DATASET_PATH)},
        "test":{"rows":len(test),"municipalities":len(test_groups),
                "class_counts":{"0":int(test["alfabetizado"].eq(0).sum()),"1":int(test["alfabetizado"].eq(1).sum())},
                "class_shares":{"0":float(test["alfabetizado"].eq(0).mean()),"1":float(test["alfabetizado"].eq(1).mean())},
                "prob_risk_summary":pd.Series(1-p_test).describe(percentiles=[.1,.25,.5,.75,.9]).to_dict(),
                "metrics":test_metrics},
        "validation":{"rows":len(validation),"class_shares":{"0":float(validation["alfabetizado"].eq(0).mean()),"1":float(validation["alfabetizado"].eq(1).mean())},
                      "prob_risk_summary":pd.Series(1-p_validation).describe(percentiles=[.1,.25,.5,.75,.9]).to_dict(),"metrics":validation_metrics},
        "comparison":{"auc_test_minus_validation":delta,"absolute_auc_difference":abs(delta),"magnitude":magnitude},
        "overlap":overlaps,"retuning_after_test":False,
    }
    METRICS_PATH.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    top = contexts.head(10)
    rows = "\n".join(f"| {r.id_municipio_nome or r.id_municipio} | {r.sigla_uf} | {r.rede} | {r.alunos:,} | {r.probabilidade_media_risco:.4f} | {r.historico_2023:.4f} | {r.meta_2024:.4f} | {r.gap:.4f} |" for r in top.itertuples())
    report = f"""# Avaliação final única no teste\n\nAbertura registrada em `{result['opened_once_at_utc']}`. Foi aplicado o pipeline Random Forest congelado, treinado somente em TRAIN, sem refit. O threshold 0,50 é referência descritiva; não é decisão operacional.\n\nTeste: {len(test):,} alunos, {len(test_groups)} municípios; classe 0: {result['test']['class_counts']['0']:,} ({result['test']['class_shares']['0']:.2%}); classe 1: {result['test']['class_counts']['1']:,} ({result['test']['class_shares']['1']:.2%}).\n\nAUC teste={test_metrics['roc_auc']:.4f}; validação={validation_metrics['roc_auc']:.4f}; delta={delta:+.4f}, diferença {magnitude}. Em 0,50: accuracy={test_metrics['accuracy_threshold_0_50']:.4f}, balanced accuracy={test_metrics['balanced_accuracy_threshold_0_50']:.4f}; risco precision/recall/F1={test_metrics['precision_class_0_risk']:.4f}/{test_metrics['recall_class_0_risk']:.4f}/{test_metrics['f1_class_0_risk']:.4f}; alfabetizado precision/recall/F1={test_metrics['precision_class_1_literate']:.4f}/{test_metrics['recall_class_1_literate']:.4f}/{test_metrics['f1_class_1_literate']:.4f}. Matriz [linhas verdadeiras, colunas previstas 0/1]: `{test_metrics['confusion_matrix_true_rows_predicted_columns_0_1']}`.\n\nOverlaps de município teste/treino e teste/validação: 0/0. Overlaps de aluno: 0/0. O resultado mede generalização territorial para municípios inéditos.\n\n## Contextos com maior risco relativo\n\n| Município | UF | Rede | Alunos | Prob. risco | Histórico 2023 | Meta | Gap |\n|---|---|---|---:|---:|---:|---:|---:|\n{rows}\n\nRanking associativo para priorização analítica; não prevê oficialmente atingimento de meta e não é diagnóstico individual.\n\n## Limitações\n\nO target é individual, mas as features são contextuais; alunos do mesmo município+rede compartilham X. Cerca de 90% da variação observada estava dentro dos contextos. O enriquecimento escolar falhou pela anonimização de `id_escola`; `proficiencia` foi excluída por leakage direto. As associações não são causais. Feature importance permaneceu a do modelo congelado.\n\n**Nenhum retuning ocorreu após a abertura do teste.**\n"""
    (REPORTS/"final_test_results.md").write_text(report,encoding="utf-8")
    print(json.dumps({"test_metrics":test_metrics,"comparison":result["comparison"],"overlap":overlaps},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
