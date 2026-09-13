"""Risco de não atingimento da meta municipal de 2024.

Responde à pergunta de negócio "como prever municípios que podem não atingir metas
futuras" a partir do artefato territorial da avaliação final.

A projeção usa a probabilidade média de alfabetização prevista pelo modelo congelado
como estimativa da taxa municipal de 2024, e a compara com a meta do município. Como
o conjunto de teste carrega a taxa observada de 2024, a própria projeção é auditada:
o relatório reporta acerto, precisão e recall do alerta contra o desfecho real, além
de comparar com uma linha de base ingênua que assume a repetição da taxa de 2023.

Entrada: `reports/final_test_municipal_analysis.csv`, produzido pela avaliação final
em municípios inéditos. Não acessa o parquet de modelagem nem o modelo serializado,
portanto roda a partir de um clone limpo.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images" / "modeling" / "goal_risk"
SOURCE_PATH = REPORTS / "final_test_municipal_analysis.csv"
RANKING_PATH = REPORTS / "goal_risk_ranking.csv"
SUMMARY_PATH = REPORTS / "goal_risk_summary.json"
REPORT_PATH = REPORTS / "goal_risk_2024.md"

# As taxas históricas e a meta vêm em pontos percentuais; as probabilidades do
# modelo vêm em fração. Este é o fator que reconcilia as duas escalas.
PERCENT = 100.0


def load_municipal_table(path: Path = SOURCE_PATH) -> pd.DataFrame:
    """Carrega o ranking territorial da avaliação final."""
    if not path.exists():
        raise FileNotFoundError(
            f"Artefato da avaliação final não encontrado: {path}. "
            "Ele é versionado no repositório; verifique o clone."
        )
    return pd.read_csv(path)


def project_goal_attainment(frame: pd.DataFrame) -> pd.DataFrame:
    """Projeta o atingimento da meta de 2024 e confronta com o desfecho observado.

    Municípios sem meta publicada são descartados: sem meta não existe pergunta de
    atingimento. Municípios sem histórico Gold são mantidos, porque a projeção do
    modelo não depende do histórico.
    """
    required = {
        "meta_2024", "probabilidade_media_alfabetizacao",
        "taxa_observada_alfabetizacao", "alunos",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no artefato de origem: {sorted(missing)}")

    out = frame.loc[frame["meta_2024"].notna()].copy()

    out["taxa_prevista_2024"] = out["probabilidade_media_alfabetizacao"] * PERCENT
    out["taxa_observada_2024"] = out["taxa_observada_alfabetizacao"] * PERCENT
    out["gap_previsto"] = out["meta_2024"] - out["taxa_prevista_2024"]
    out["gap_observado"] = out["meta_2024"] - out["taxa_observada_2024"]

    # Alerta do modelo e desfecho real. Ambos na mesma convenção: True significa
    # "não atinge a meta".
    out["alerta_nao_atingir"] = out["gap_previsto"] > 0
    out["nao_atingiu_observado"] = out["gap_observado"] > 0

    # Linha de base ingênua: supor que 2024 repete 2023. Só existe onde há histórico.
    out["alerta_naive_2023"] = np.where(
        out["historico_2023"].notna(), out["meta_2024"] > out["historico_2023"], np.nan,
    )

    out["severidade"] = out["gap_previsto"].clip(lower=0)
    out = out.sort_values(
        ["alerta_nao_atingir", "severidade", "alunos"], ascending=[False, False, False],
    )
    out["ranking_risco_meta"] = np.arange(1, len(out) + 1)
    return out


def alert_metrics(observed: pd.Series, alert: pd.Series) -> dict:
    """Qualidade de um alerta binário contra o desfecho observado."""
    observed = pd.Series(observed).astype(bool).to_numpy()
    alert = pd.Series(alert).astype(bool).to_numpy()
    true_positive = int((alert & observed).sum())
    false_positive = int((alert & ~observed).sum())
    false_negative = int((~alert & observed).sum())
    true_negative = int((~alert & ~observed).sum())
    total = true_positive + false_positive + false_negative + true_negative
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "municipios": total,
        "acuracia": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "verdadeiro_positivo": true_positive,
        "falso_positivo": false_positive,
        "falso_negativo": false_negative,
        "verdadeiro_negativo": true_negative,
    }


def plot_risk_versus_gap(frame: pd.DataFrame) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for flag, color, label in (
        (True, "#c44e52", "alerta: não atinge a meta"),
        (False, "#4c72b0", "sem alerta"),
    ):
        subset = frame.loc[frame["alerta_nao_atingir"].eq(flag)]
        plt.scatter(
            subset["meta_2024"], subset["taxa_prevista_2024"],
            s=np.clip(subset["alunos"] / 25, 6, 180), alpha=.55, color=color,
            edgecolors="none", label=f"{label} (n={len(subset)})",
        )
    limits = [
        float(min(frame["meta_2024"].min(), frame["taxa_prevista_2024"].min())) - 2,
        float(max(frame["meta_2024"].max(), frame["taxa_prevista_2024"].max())) + 2,
    ]
    plt.plot(limits, limits, "--", color="gray", linewidth=1, label="meta atingida na projeção")
    plt.xlabel("Meta municipal para 2024 (%)")
    plt.ylabel("Taxa de alfabetização projetada para 2024 (%)")
    plt.title("Projeção do modelo contra a meta municipal\nabaixo da diagonal = risco de não atingimento")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(IMAGES / "01_projecao_vs_meta.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(frame["gap_previsto"], bins=40, color="#4c72b0", alpha=.8)
    plt.axvline(0, color="#c44e52", linestyle="--", linewidth=1.2)
    plt.xlabel("Gap projetado em pontos percentuais (meta menos projeção)")
    plt.ylabel("Municípios")
    plt.title("Distribuição do gap projetado para a meta de 2024\nà direita de zero = risco de não atingimento")
    plt.tight_layout()
    plt.savefig(IMAGES / "02_distribuicao_gap_projetado.png", dpi=150)
    plt.close()


def concentration_note(top: pd.DataFrame, frame: pd.DataFrame) -> str:
    """Aponta concentração do topo do ranking numa única UF, quando houver.

    O gap é medido em pontos percentuais absolutos, então UFs que fixaram metas mais
    ambiciosas tendem a dominar o topo. Sem essa ressalva o ranking parece enviesado.
    """
    if top.empty or top["sigla_uf"].isna().all():
        return ""
    counts = top["sigla_uf"].value_counts()
    uf, hits = counts.index[0], int(counts.iloc[0])
    if hits < len(top) * .6:
        return ""
    uf_goal = frame.loc[frame["sigla_uf"].eq(uf), "meta_2024"].mean()
    overall_goal = frame["meta_2024"].mean()
    return (
        f"\n**Leitura obrigatória do ranking.** {hits} dos {len(top)} municípios do topo "
        f"são de {uf}. Isso não indica erro: a meta média em {uf} é de {uf_goal:.1f}%, "
        f"contra {overall_goal:.1f}% no conjunto avaliado. Como o gap é medido em pontos "
        f"percentuais absolutos, UFs que fixaram metas mais ambiciosas concentram os "
        f"maiores gaps mesmo quando partem de patamares melhores. Para priorização "
        f"orçamentária, combine este ranking com o de risco absoluto em "
        f"`final_test_municipal_analysis.csv`, que ordena por probabilidade de risco e "
        f"não depende da ambição da meta estadual.\n"
    )


def build_report(frame: pd.DataFrame, summary: dict) -> str:
    top = frame.loc[frame["alerta_nao_atingir"]].head(15)
    rows = "\n".join(
        f"| {int(r.ranking_risco_meta)} | {r.id_municipio_nome} | {r.sigla_uf} | "
        f"{int(r.alunos):,} | {r.meta_2024:.2f} | {r.taxa_prevista_2024:.2f} | "
        f"{r.gap_previsto:+.2f} | {r.taxa_observada_2024:.2f} |".replace(",", ".")
        for r in top.itertuples()
    )
    model = summary["alerta_modelo"]
    naive = summary["alerta_naive_2023"]
    return f"""# Risco de não atingimento da meta municipal de 2024

Responde à pergunta de negócio *"como prever municípios que podem não atingir metas
futuras"*.

## Método

A probabilidade média de alfabetização prevista pelo modelo congelado é usada como
estimativa da taxa municipal de 2024 e comparada com a meta do município. Quando a
projeção fica abaixo da meta, o município recebe alerta de risco de não atingimento.

A base é `reports/final_test_municipal_analysis.csv`, restrita aos
{summary['municipios_avaliados']} municípios do conjunto de teste que possuem meta
publicada. São municípios **inéditos** para o modelo: não aparecem no treino nem na
validação. Dos {summary['municipios_no_teste']} municípios do teste,
{summary['municipios_sem_meta']} foram descartados por não terem meta.

## Resultado da projeção

{summary['municipios_em_risco']} municípios ({summary['percentual_em_risco']:.1%})
receberam alerta de não atingimento. O gap projetado mediano entre meta e projeção é
de {summary['gap_previsto_mediano']:+.2f} pontos percentuais.

Na realidade observada em 2024, {summary['municipios_nao_atingiram']} municípios
({summary['percentual_nao_atingiu']:.1%}) de fato não atingiram a meta.

## Qualidade do alerta

Como o conjunto de teste carrega a taxa efetivamente observada em 2024, o alerta pode
ser auditado contra o desfecho real. A linha de base ingênua supõe que 2024 repete a
taxa de 2023.

| Estratégia | Acurácia | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| Projeção do modelo | {model['acuracia']:.4f} | {model['precision']:.4f} | {model['recall']:.4f} | {model['f1']:.4f} |
| Linha de base: repetir 2023 | {naive['acuracia']:.4f} | {naive['precision']:.4f} | {naive['recall']:.4f} | {naive['f1']:.4f} |

Matriz do alerta do modelo: {model['verdadeiro_positivo']} alertas corretos,
{model['falso_positivo']} alarmes falsos, {model['falso_negativo']} municípios em
risco que passaram despercebidos e {model['verdadeiro_negativo']} corretamente
liberados.

## Municípios prioritários

Os quinze com maior gap projetado para a meta. A última coluna mostra a taxa
efetivamente observada, para conferência.

| # | Município | UF | Alunos | Meta 2024 | Projeção | Gap projetado | Observado |
|---:|---|---|---:|---:|---:|---:|---:|
{rows}

{concentration_note(top, frame)}
Ranking completo em `goal_risk_ranking.csv`; números consolidados em
`goal_risk_summary.json`.

## Limitações

O alerta é uma projeção associativa, não uma previsão oficial de cumprimento de meta.
A taxa projetada é a média das probabilidades individuais previstas dentro do
município, e o modelo é predominantemente territorial: alunos do mesmo município e
rede recebem a mesma probabilidade. A projeção herda, portanto, todas as limitações
registradas em `modeling_limitations.md`.

A comparação com a linha de base ingênua mede apenas a capacidade de ordenar e
sinalizar municípios, não de estimar a magnitude exata da taxa futura. Municípios sem
meta publicada ficam fora da análise por construção.
"""


def main() -> None:
    frame = project_goal_attainment(load_municipal_table())

    model_metrics = alert_metrics(frame["nao_atingiu_observado"], frame["alerta_nao_atingir"])
    with_naive = frame.loc[frame["alerta_naive_2023"].notna()]
    naive_metrics = alert_metrics(
        with_naive["nao_atingiu_observado"], with_naive["alerta_naive_2023"],
    )

    source_rows = len(load_municipal_table())
    summary = {
        "fonte": str(SOURCE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "municipios_no_teste": source_rows,
        "municipios_sem_meta": source_rows - len(frame),
        "municipios_avaliados": len(frame),
        "municipios_em_risco": int(frame["alerta_nao_atingir"].sum()),
        "percentual_em_risco": float(frame["alerta_nao_atingir"].mean()),
        "municipios_nao_atingiram": int(frame["nao_atingiu_observado"].sum()),
        "percentual_nao_atingiu": float(frame["nao_atingiu_observado"].mean()),
        "gap_previsto_mediano": float(frame["gap_previsto"].median()),
        "alerta_modelo": model_metrics,
        "alerta_naive_2023": naive_metrics,
    }

    columns = [
        "ranking_risco_meta", "id_municipio", "id_municipio_nome", "sigla_uf", "alunos",
        "historico_2023", "meta_2024", "taxa_prevista_2024", "gap_previsto",
        "alerta_nao_atingir", "taxa_observada_2024", "gap_observado",
        "nao_atingiu_observado", "probabilidade_media_risco",
    ]
    REPORTS.mkdir(exist_ok=True)
    frame[columns].to_csv(RANKING_PATH, index=False)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    plot_risk_versus_gap(frame)
    REPORT_PATH.write_text(build_report(frame, summary), encoding="utf-8")

    print(json.dumps({
        "municipios_avaliados": summary["municipios_avaliados"],
        "municipios_em_risco": summary["municipios_em_risco"],
        "acuracia_modelo": round(model_metrics["acuracia"], 4),
        "acuracia_naive": round(naive_metrics["acuracia"], 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
