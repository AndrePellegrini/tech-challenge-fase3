"""Interpretabilidade por SHAP sobre o modelo congelado.

Fecha a lacuna da DEC-013. Até aqui o projeto tinha apenas a importância nativa da
Random Forest, e `reports/shap_summary.json` era um literal fixo declarando que o SHAP
não havia sido executado.

O valor de acrescentar SHAP não é ter mais um ranking, e sim ter um ranking construído
por outro mecanismo. A importância nativa de árvores mede redução de impureza e é
enviesada quando os preditores são correlacionados — exatamente o caso aqui, já que as
seis features educacionais de 2023 medem facetas do mesmo fenômeno municipal. SHAP
atribui contribuição marginal por predição e não sofre desse viés da mesma forma. Onde
os dois divergem, há sinal.

Restrições preservadas:

* usa **apenas a partição de validação**; o teste nunca é materializado;
* o modelo é carregado congelado do `.joblib`, sem refit;
* as contribuições das 43 colunas codificadas são reagregadas para as 16 features de
  origem, com a mesma regra da importância nativa, para que `rede` e `sigla_uf` não
  apareçam fragmentadas em categorias.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.modeling.final_validation import assert_feature_contract
from src.modeling.split import RANDOM_STATE, get_modeling_columns, split_by_municipality
from src.modeling.train_baselines import _source_feature, aggregate_tree_importance
from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset
from src.visualization.plots import horizontal_bar

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images" / "modeling" / "shap"
MODEL_PATH = ROOT / "models" / "final_candidate_validation.joblib"
SUMMARY_PATH = REPORTS / "shap_summary.json"
IMPORTANCE_PATH = REPORTS / "shap_importance.csv"
REPORT_PATH = REPORTS / "shap_results.md"

# Amostra da validação. Com 40 árvores de profundidade 10, o TreeExplainer é rápido;
# 8.000 linhas dão estimativa estável sem custo desproporcional.
SAMPLE_SIZE = 8_000


def sample_validation(data: pd.DataFrame, features: list[str],
                      size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Amostra aleatória da validação. O teste não é indexado."""
    splits = split_by_municipality(data)
    validation = data.iloc[splits["validation"]]
    if len(validation) <= size:
        return validation[features]
    return validation[features].sample(n=size, random_state=RANDOM_STATE)


def mean_absolute_shap(values: np.ndarray, encoded_names) -> pd.DataFrame:
    """Média do valor absoluto por coluna codificada.

    O TreeExplainer de um classificador binário pode devolver formatos diferentes
    conforme a versão: `(n, f)`, `(n, f, 2)` ou uma lista com uma matriz por classe.
    Em todos os casos interessa a contribuição para a classe positiva.
    """
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    elif isinstance(values, list):
        array = np.asarray(values[-1])
    if array.ndim != 2:
        raise ValueError(f"Formato inesperado de valores SHAP: {array.shape}")
    if array.shape[1] != len(encoded_names):
        raise ValueError(
            f"SHAP devolveu {array.shape[1]} colunas para "
            f"{len(encoded_names)} features codificadas."
        )
    return pd.DataFrame({
        "encoded_feature": list(encoded_names),
        "mean_abs_shap": np.abs(array).mean(axis=0),
    })


def aggregate_to_source(encoded: pd.DataFrame, source_columns: list[str]) -> pd.DataFrame:
    """Soma as contribuições das colunas one-hot de volta para a feature de origem."""
    frame = encoded.copy()
    frame["feature"] = frame["encoded_feature"].map(
        lambda name: _source_feature(name, source_columns)
    )
    return (
        frame.groupby("feature", as_index=False)["mean_abs_shap"].sum()
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )


def compare_with_native(shap_frame: pd.DataFrame, native: pd.DataFrame) -> pd.DataFrame:
    """Junta os dois rankings e mede o deslocamento de posição de cada feature."""
    merged = shap_frame.merge(native, on="feature", how="outer")
    merged["rank_shap"] = merged["mean_abs_shap"].rank(ascending=False, method="min")
    merged["rank_nativa"] = merged["importance"].rank(ascending=False, method="min")
    merged["deslocamento"] = merged["rank_nativa"] - merged["rank_shap"]
    return merged.sort_values("rank_shap", ignore_index=True)


def territorial_note(comparison: pd.DataFrame) -> str:
    """Destaca o caso de `sigla_uf`, se ele subir no ranking do SHAP.

    É a divergência mais informativa do projeto: a variável puramente territorial
    ganhar posições confirma, por um segundo método independente, a conclusão da
    auditoria de granularidade.
    """
    row = comparison.loc[comparison["feature"].eq("sigla_uf")]
    if row.empty:
        return ""
    row = row.iloc[0]
    if row["deslocamento"] < 2:
        return ""
    return (
        f"\n### O caso de `sigla_uf`\n\n"
        f"A divergência mais informativa é a da única feature puramente territorial: "
        f"`sigla_uf` aparece em {int(row['rank_nativa'])}º lugar pela importância nativa "
        f"e em {int(row['rank_shap'])}º pelo SHAP, um salto de "
        f"{int(row['deslocamento'])} posições.\n\n"
        f"A explicação é mecânica. A importância nativa é calculada sobre as colunas "
        f"codificadas: cada UF vira uma coluna one-hot que isoladamente reduz pouca "
        f"impureza, e a soma dessas parcelas subestima o quanto o território importa. "
        f"O SHAP mede contribuição marginal por predição, então captura o efeito "
        f"conjunto de saber em que UF o aluno está.\n\n"
        f"A consequência analítica é relevante: **um segundo método, independente, "
        f"confirma a conclusão da auditoria de granularidade**. O modelo se apoia mais "
        f"no território do que a importância nativa sugeria, o que reforça a leitura de "
        f"que o produto é um instrumento de priorização territorial e não um "
        f"diagnóstico individual.\n"
    )


def build_report(comparison: pd.DataFrame, summary: dict) -> str:
    rows = "\n".join(
        f"| {int(r.rank_shap)} | `{r.feature}` | {r.mean_abs_shap:.4f} | "
        f"{int(r.rank_nativa)} | {r.importance:.4f} | {int(r.deslocamento):+d} |"
        for r in comparison.itertuples()
    )
    sample_label = f"{summary['sample_rows']:,}".replace(",", ".")
    moved = comparison.loc[comparison["deslocamento"].abs() >= 2]
    if len(moved):
        divergence = "\n".join(
            f"- `{r.feature}`: {int(r.rank_nativa)}º pela importância nativa contra "
            f"{int(r.rank_shap)}º pelo SHAP"
            for r in moved.itertuples()
        )
        divergence_note = (
            f"**Onde os dois métodos divergem** em duas ou mais posições:\n\n{divergence}\n\n"
            "Divergência é esperada e informativa. A importância nativa distribui o "
            "crédito entre preditores correlacionados de forma sensível à ordem em que "
            "as árvores os utilizam; o SHAP atribui contribuição marginal por predição. "
            "Como as seis features educacionais de 2023 são fortemente correlacionadas "
            "entre si, é justamente entre elas que a diferença aparece."
        )
    else:
        divergence_note = (
            "**Os dois métodos concordam** em todas as posições relevantes, sem "
            "deslocamento de duas ou mais posições. Isso reforça a leitura de que o "
            "histórico educacional municipal domina a predição, e indica que a "
            "correlação entre as features de 2023 não distorceu materialmente a "
            "importância nativa."
        )
    return f"""# Interpretabilidade por SHAP

Fecha a lacuna da DEC-013.

## Método

`shap.TreeExplainer` aplicado ao modelo congelado `random_forest_controlled`, carregado
do `.joblib` sem refit, sobre amostra de {sample_label} linhas da partição de
**validação**. O conjunto de teste não foi materializado.

As {summary['encoded_features']} colunas codificadas foram reagregadas para as
{summary['source_features']} features de origem pela mesma regra da importância nativa,
somando as categorias one-hot de `rede` e `sigla_uf` de volta à sua feature.

## Ranking comparado

| # SHAP | Feature | SHAP médio absoluto | # Nativa | Importância nativa | Deslocamento |
|---:|---|---:|---:|---:|---:|
{rows}

{divergence_note}
{territorial_note(comparison)}
## Correlação entre os rankings

A correlação de Spearman entre as duas ordenações é de
**{summary['spearman_ranks']:.4f}**.

## Leitura

Nenhum dos dois rankings é causal. Ambos descrevem como o modelo usa as features, não
como a alfabetização é produzida no mundo. Uma feature com contribuição alta indica que
o modelo se apoia nela, e não que intervir sobre ela mude o desfecho.

Vale lembrar a auditoria de granularidade: alunos do mesmo município e rede recebem o
mesmo vetor de entrada, então as contribuições aqui descrevem o que separa **contextos
territoriais**, não o que separa crianças dentro de um mesmo contexto.
"""


def main() -> None:
    import shap  # importado aqui para manter a dependência opcional no restante do projeto

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo congelado não encontrado: {MODEL_PATH}. "
            "Execute `python -m src.modeling.final_validation` antes."
        )
    data = pd.read_parquet(DATASET_PATH)
    contract = validate_dataset(data, strict_counts=True)
    if not contract["valido"]:
        raise ValueError(contract["errors"])
    features = list(get_modeling_columns()["X"])
    assert_feature_contract(features)

    pipeline = joblib.load(MODEL_PATH)
    sample = sample_validation(data, features)
    print(f"Amostra de validação: {len(sample):,} linhas; teste não tocado.")

    preprocessing = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]
    encoded_names = preprocessing.get_feature_names_out()
    matrix = preprocessing.transform(sample)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(matrix, check_additivity=False)
    encoded = mean_absolute_shap(values, encoded_names)
    shap_frame = aggregate_to_source(encoded, features)

    native = aggregate_tree_importance(pipeline, features)
    comparison = compare_with_native(shap_frame, native)
    spearman = float(comparison[["rank_shap", "rank_nativa"]].corr(method="spearman").iloc[0, 1])

    summary = {
        "executed": True,
        "explainer": "shap.TreeExplainer",
        "model": "random_forest_controlled",
        "model_path": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "partition": "validation",
        "test_accessed": False,
        "sample_rows": int(len(sample)),
        "encoded_features": int(len(encoded_names)),
        "source_features": int(len(shap_frame)),
        "random_state": RANDOM_STATE,
        "spearman_ranks": spearman,
        "top_features": shap_frame.head(5).to_dict(orient="records"),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                     "shap": shap.__version__},
    }

    REPORTS.mkdir(exist_ok=True)
    comparison.to_csv(IMPORTANCE_PATH, index=False)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    top = shap_frame.head(15).sort_values("mean_abs_shap")
    horizontal_bar(
        top["feature"], top["mean_abs_shap"], IMAGES / "01_shap_importance.png",
        title="Contribuição SHAP média absoluta — modelo final",
        xlabel="SHAP médio absoluto (associativo, não causal)",
    )
    REPORT_PATH.write_text(build_report(comparison, summary), encoding="utf-8")
    print(json.dumps({
        "sample_rows": summary["sample_rows"],
        "spearman_ranks": round(spearman, 4),
        "top_shap": [r["feature"] for r in summary["top_features"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
