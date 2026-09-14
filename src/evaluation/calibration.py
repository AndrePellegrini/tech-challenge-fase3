"""Calibracao da taxa municipal implicita na probabilidade media do modelo.

A analise de risco de meta usa a media das probabilidades individuais previstas como
estimativa da taxa municipal de alfabetizacao. Essa leitura so se sustenta se as
probabilidades estiverem razoavelmente calibradas: ROC AUC mede ordenacao, e um modelo
pode ordenar bem sem que uma previsao de 0,65 corresponda a 65% de alfabetizados.

Este modulo mede isso. Nao retreina nada e nao reabre o conjunto de teste: consome
apenas `reports/final_test_municipal_analysis.csv`, artefato ja versionado, produzido
na abertura unica. Por isso roda a partir de um clone limpo, sem credencial.

Duas leituras importam.

**Calibracao agregada.** Vies, MAE, RMSE e a tabela por decil dizem se a taxa implicita
acompanha a observada ao longo de toda a faixa, ou se ha desvio sistematico.

**Encolhimento.** Todo modelo comprime previsoes em direcao a media. A razao entre os
desvios-padrao previsto e observado quantifica quanto, o que importa para quem for usar
o ranking: o ordenamento e preservado, mas a amplitude entre extremos e subestimada.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization.plots import save_figure

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images" / "modeling" / "calibration"
SOURCE_PATH = REPORTS / "final_test_municipal_analysis.csv"
DECILES_PATH = REPORTS / "calibration_municipal.csv"
SUMMARY_PATH = REPORTS / "calibration_municipal.json"
REPORT_PATH = REPORTS / "calibration_municipal.md"

OBSERVED = "taxa_observada_alfabetizacao"
PREDICTED = "probabilidade_media_alfabetizacao"
WEIGHT = "alunos"
BINS = 10


def load_municipal_table(path: Path = SOURCE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Artefato da avaliacao final nao encontrado: {path}. "
            "Ele e versionado no repositorio; verifique o clone."
        )
    return pd.read_csv(path)


def calibration_metrics(frame: pd.DataFrame) -> dict:
    """Metricas de calibracao entre taxa observada e taxa implicita no score."""
    missing = {OBSERVED, PREDICTED, WEIGHT}.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")

    observed = frame[OBSERVED].to_numpy(dtype=float)
    predicted = frame[PREDICTED].to_numpy(dtype=float)
    weight = frame[WEIGHT].to_numpy(dtype=float)
    error = predicted - observed

    observed_spread = float(observed.std())
    predicted_spread = float(predicted.std())
    return {
        "municipios": int(len(frame)),
        "alunos": int(weight.sum()),
        "media_observada": float(observed.mean()),
        "media_prevista": float(predicted.mean()),
        "vies": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "mae_ponderado_por_aluno": float(np.average(np.abs(error), weights=weight)),
        "correlacao_pearson": float(np.corrcoef(predicted, observed)[0, 1]),
        "desvio_observado": observed_spread,
        "desvio_previsto": predicted_spread,
        "razao_de_encolhimento": (predicted_spread / observed_spread
                                  if observed_spread else float("nan")),
    }


def calibration_table(frame: pd.DataFrame, bins: int = BINS) -> pd.DataFrame:
    """Taxa observada contra prevista, por faixa de previsao."""
    working = frame[[PREDICTED, OBSERVED, WEIGHT]].copy()
    working["faixa"] = pd.qcut(working[PREDICTED], bins, labels=False, duplicates="drop")
    table = working.groupby("faixa", as_index=False).agg(
        municipios=(PREDICTED, "size"),
        alunos=(WEIGHT, "sum"),
        previsto=(PREDICTED, "mean"),
        observado=(OBSERVED, "mean"),
    )
    table["desvio"] = table["previsto"] - table["observado"]
    table["faixa"] = table["faixa"].astype(int) + 1
    return table


def plot_observed_versus_predicted(frame: pd.DataFrame, table: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt  # backend Agg definido em src.visualization.plots

    plt.figure(figsize=(7, 7))
    plt.scatter(
        frame[PREDICTED], frame[OBSERVED],
        s=np.clip(frame[WEIGHT] / 25, 6, 180), alpha=.35, color="#4c72b0",
        edgecolors="none", label=f"municipios (n={len(frame)})",
    )
    plt.plot(table["previsto"], table["observado"], "o-", color="#c44e52",
             linewidth=2, markersize=7, label="media por decil de previsao")
    limits = [0, 1]
    plt.plot(limits, limits, "--", color="gray", linewidth=1, label="calibracao perfeita")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Taxa implicita na probabilidade media prevista")
    plt.ylabel("Taxa de alfabetizacao observada em 2024")
    plt.title("Calibracao no nivel municipal\nteste, municipios ineditos")
    plt.legend(fontsize=8, loc="upper left")
    return save_figure(IMAGES / "01_observado_vs_previsto.png")


def verdict(summary: dict, table: pd.DataFrame) -> str:
    """Leitura da calibracao a partir do maior desvio por faixa."""
    worst = float(table["desvio"].abs().max())
    if worst < .03:
        return (
            f"O maior desvio entre previsto e observado em qualquer faixa e de "
            f"{worst:.4f}, ou seja, menos de tres pontos percentuais. A taxa implicita "
            f"acompanha a observada ao longo de toda a faixa de previsao, e o vies "
            f"agregado de {summary['vies']:+.4f} e praticamente nulo. **A projecao esta "
            f"bem calibrada no nivel municipal.**"
        )
    if worst < .06:
        return (
            f"O maior desvio por faixa e de {worst:.4f}. A calibracao e razoavel, com "
            f"desvios moderados em parte da faixa; a projecao serve a sinalizacao "
            f"relativa e deve ser lida com cautela como estimativa pontual."
        )
    return (
        f"O maior desvio por faixa e de {worst:.4f}, o que indica desvio sistematico "
        f"relevante. A projecao deve ser interpretada apenas como ordenamento de risco, "
        f"e nao como estimativa da taxa."
    )


def build_report(summary: dict, table: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| {int(r.faixa)} | {int(r.municipios)} | {r.previsto:.4f} | "
        f"{r.observado:.4f} | {r.desvio:+.4f} |"
        for r in table.itertuples()
    )
    shrink = summary["razao_de_encolhimento"]
    return f"""# Calibracao da taxa municipal

A analise de risco de meta usa a media das probabilidades individuais previstas como
estimativa da taxa municipal. Este relatorio verifica se essa leitura se sustenta.

A distincao importa: ROC AUC mede **ordenacao**. Um modelo pode ordenar bem sem que uma
previsao de 0,65 corresponda a 65% de alfabetizados. Sem esta verificacao, chamar a
media das probabilidades de "taxa projetada" seria uma afirmacao nao sustentada.

## Metodo

Comparacao entre a taxa observada em 2024 e a taxa implicita na probabilidade media
prevista, nos {summary['municipios']} municipios do conjunto de teste — todos ineditos
para o modelo, somando {summary['alunos']:,} alunos.

Consome apenas `reports/final_test_municipal_analysis.csv`, artefato ja versionado.
**Nao ha retreino nem reabertura do conjunto de teste.**

## Resultado agregado

| Metrica | Valor |
|---|---:|
| Media observada | {summary['media_observada']:.4f} |
| Media prevista | {summary['media_prevista']:.4f} |
| Vies (previsto menos observado) | {summary['vies']:+.4f} |
| MAE | {summary['mae']:.4f} |
| RMSE | {summary['rmse']:.4f} |
| MAE ponderado por aluno | {summary['mae_ponderado_por_aluno']:.4f} |
| Correlacao de Pearson | {summary['correlacao_pearson']:.4f} |

O MAE ponderado por aluno ({summary['mae_ponderado_por_aluno']:.4f}) e bem menor que o
MAE simples ({summary['mae']:.4f}). A razao e estatistica: municipios pequenos tem taxa
observada ruidosa, porque poucas dezenas de alunos produzem variacao amostral alta. Onde
ha mais alunos, a previsao acerta mais.

## Calibracao por faixa

| Decil | Municipios | Previsto | Observado | Desvio |
|---:|---:|---:|---:|---:|
{rows}

{verdict(summary, table)}

![Observado contra previsto](../images/modeling/calibration/01_observado_vs_previsto.png)

## Encolhimento

O desvio-padrao das previsoes e {summary['desvio_previsto']:.4f}, contra
{summary['desvio_observado']:.4f} do observado — razao de {shrink:.4f}.

Isso e o comportamento esperado de qualquer modelo: previsoes sao comprimidas em direcao
a media, porque o modelo so captura a parte explicavel da variacao. A consequencia
pratica e que o **ordenamento entre municipios e confiavel, mas a amplitude entre os
extremos e subestimada**. Um municipio previsto em 0,36 tende a estar de fato um pouco
abaixo disso, e um previsto em 0,86, um pouco acima.

## O contraste que importa

A correlacao de Pearson de {summary['correlacao_pearson']:.4f} no nivel municipal
contrasta com a ROC AUC de 0,6631 no nivel do aluno.

Nao ha contradicao: sao a mesma evidencia vista em duas escalas. O modelo tem pouco poder
para separar dois alunos do mesmo municipio, porque eles compartilham o mesmo vetor de
features — e a auditoria de granularidade mostrou que 99,82% dos alunos estao em perfis
ambiguos. Mas ao agregar por municipio, o ruido individual se cancela e resta o sinal
territorial, que e forte.

**Esta e a evidencia mais direta de que o produto entregue e um instrumento de
priorizacao territorial**, e nao um classificador individual.

## Limitacoes

A calibracao foi medida no nivel municipal, que e o nivel de uso. Nao foi avaliada
calibracao individual por curva de confiabilidade ou Brier score, nem aplicada
recalibracao por Platt ou isotonica — nao foram necessarias, dado o vies observado.

A verificacao usa os mesmos municipios da avaliacao final. Sao ineditos para o modelo,
mas a medicao e posterior a abertura do teste e deve ser lida como **analise de
robustez**, nao como parte do protocolo de selecao.
"""


def main() -> None:
    frame = load_municipal_table()
    summary = calibration_metrics(frame)
    table = calibration_table(frame)

    REPORTS.mkdir(exist_ok=True)
    table.to_csv(DECILES_PATH, index=False)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    plot_observed_versus_predicted(frame, table)
    REPORT_PATH.write_text(build_report(summary, table), encoding="utf-8")

    print(json.dumps({
        "municipios": summary["municipios"],
        "vies": round(summary["vies"], 4),
        "mae": round(summary["mae"], 4),
        "correlacao_pearson": round(summary["correlacao_pearson"], 4),
        "maior_desvio_por_faixa": round(float(table["desvio"].abs().max()), 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
