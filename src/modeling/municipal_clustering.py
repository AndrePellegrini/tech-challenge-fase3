"""Clustering complementar de municípios; não altera ou avalia o classificador."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing.validate_dataset import DATASET_PATH, validate_dataset

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
IMAGES = ROOT / "images" / "modeling" / "clustering"
RANDOM_STATE = 42
K_VALUES = (2, 3, 4, 5, 6)
CANDIDATES = [
    "taxa_alfabetizacao_municipio_2023", "media_portugues_municipio_2023",
    "percentual_participacao_municipio_2023", "meta_alfabetizacao_municipio_2024",
    "gap_para_meta_municipio_2024", "idhm", "idhm_educacao", "idhm_renda",
    "idhm_longevidade", "total_alunos_municipio_2023",
]
FEATURES = [
    "taxa_alfabetizacao_municipio_2023", "media_portugues_municipio_2023",
    "percentual_participacao_municipio_2023", "meta_alfabetizacao_municipio_2024", "idhm",
]
FORBIDDEN = {"alfabetizado", "proficiencia", "id_aluno", "id_escola", "id_municipio",
             "rede", "sigla_uf", "probabilidade_media_risco"}


def validate_features(features):
    if not 5 <= len(features) <= 8:
        raise ValueError("Use de 5 a 8 features no clustering municipal.")
    invalid = FORBIDDEN.intersection(features)
    if invalid:
        raise ValueError(f"Features proibidas: {sorted(invalid)}")


def build_municipal_table(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Média sobre alunos: pondera redes pelo número de alunos sem escolher uma."""
    validate_features(FEATURES)
    variation = {}
    grouped = data.groupby("id_municipio", sort=True, dropna=False)
    for column in CANDIDATES:
        variation[column] = {
            "municipalities_with_more_than_one_value": int(grouped[column].nunique(dropna=True).gt(1).sum()),
            "aggregation": "média ponderada pelo número de alunos (média das linhas individuais)",
        }
    municipal = grouped.agg(
        id_municipio_nome=("id_municipio_nome", "first"), sigla_uf=("sigla_uf", "first"),
        alunos=("id_municipio", "size"), **{column: (column, "mean") for column in CANDIDATES},
    ).reset_index()
    if municipal["id_municipio"].duplicated().any():
        raise ValueError("Tabela municipal não é única por município.")
    return municipal, variation


def build_pipeline(k: int, random_state: int = RANDOM_STATE) -> Pipeline:
    if k not in K_VALUES:
        raise ValueError(f"k deve pertencer a {K_VALUES}.")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=random_state)),
    ])


def evaluate_k(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, Pipeline]]:
    rows, fitted = [], {}
    for k in K_VALUES:
        pipe = build_pipeline(k); labels = pipe.fit_predict(X); fitted[k] = pipe
        sizes = pd.Series(labels).value_counts()
        transformed = pipe[:-1].transform(X)
        silhouette_sample = min(3_000, len(X))
        rows.append({"k": k, "silhouette": float(silhouette_score(
                         transformed, labels, sample_size=silhouette_sample,
                         random_state=RANDOM_STATE)),
                     "silhouette_sample_size": silhouette_sample,
                     "inertia": float(pipe.named_steps["kmeans"].inertia_),
                     "min_cluster_size": int(sizes.min()),
                     "min_cluster_share": float(sizes.min()/len(X))})
    return pd.DataFrame(rows), fitted


def choose_k(metrics: pd.DataFrame) -> int:
    """Prefere solução interpretável sem cluster <3%, perto do melhor silhouette."""
    eligible = metrics.loc[metrics["min_cluster_share"].ge(.03)]
    best = eligible["silhouette"].max()
    near = eligible.loc[eligible["silhouette"].ge(best-.02)]
    return int(near.sort_values(["k", "silhouette"], ascending=[True, False]).iloc[0]["k"])


def label_profiles(profile: pd.DataFrame) -> dict[int, str]:
    z = profile.set_index("cluster")[[f"z_{c}" for c in FEATURES]]
    educational = z[[f"z_{FEATURES[0]}", f"z_{FEATURES[1]}"]].mean(axis=1)
    participation = z[f"z_{FEATURES[2]}"]
    socioeconomic = z[f"z_{FEATURES[4]}"]
    labels = {}
    for cluster in z.index:
        if educational[cluster] == educational.min():
            label = "maior vulnerabilidade educacional relativa"
        elif educational[cluster] == educational.max() and socioeconomic[cluster] >= 0:
            label = "melhor contexto educacional e socioeconômico relativo"
        elif participation[cluster] == participation.min():
            label = "participação relativa mais baixa"
        else:
            label = "contexto intermediário"
        labels[int(cluster)] = label
    return labels


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main():
    columns=["id_municipio","id_municipio_nome","sigla_uf",*CANDIDATES]
    parquet=pq.ParquetFile(DATASET_PATH)
    available=set(parquet.schema_arrow.names)
    missing_columns=set(columns)-available
    if parquet.metadata.num_rows!=1_851_828 or missing_columns:
        raise ValueError(f"Parquet inesperado: rows={parquet.metadata.num_rows}; missing={missing_columns}")
    data=pd.read_parquet(DATASET_PATH,columns=columns)
    municipal, variation=build_municipal_table(data)
    missing={column:int(municipal[column].isna().sum()) for column in FEATURES}
    metrics,pipelines=evaluate_k(municipal[FEATURES]); selected_k=choose_k(metrics)
    pipe=pipelines[selected_k]; labels=pipe.predict(municipal[FEATURES])
    municipal["cluster"]=labels
    transformed=pipe[:-1].transform(municipal[FEATURES])
    centers=pipe.named_steps["kmeans"].cluster_centers_
    distances=np.linalg.norm(transformed-centers[labels],axis=1)
    municipal["distancia_centroide"]=distances

    original_centers=pipe.named_steps["scaler"].inverse_transform(centers)
    profiles=[]
    for cluster in range(selected_k):
        part=municipal.loc[municipal["cluster"].eq(cluster)]
        row={"cluster":cluster,"municipios":len(part),"percentual_municipios":len(part)/len(municipal),
             "principais_ufs":"; ".join(part["sigla_uf"].value_counts().head(5).index.astype(str))}
        for index,column in enumerate(FEATURES):
            row[f"media_{column}"]=float(part[column].mean())
            row[f"mediana_{column}"]=float(part[column].median())
            row[f"centroide_{column}"]=float(original_centers[cluster,index])
            row[f"z_{column}"]=float(centers[cluster,index])
        profiles.append(row)
    profiles=pd.DataFrame(profiles); names=label_profiles(profiles)
    profiles["perfil_interpretativo"]=profiles["cluster"].map(names)
    municipal["perfil_interpretativo"]=municipal["cluster"].map(names)

    risk_path=REPORTS/"final_test_municipal_analysis.csv"
    risk=pd.read_csv(risk_path,dtype={"id_municipio":"string"})[["id_municipio","alunos","taxa_observada_alfabetizacao","probabilidade_media_risco"]]
    municipal["id_municipio"]=municipal["id_municipio"].astype("string")
    relation=municipal[["id_municipio","cluster"]].merge(risk,on="id_municipio",how="inner")
    relation_summary=relation.groupby("cluster").apply(lambda p:pd.Series({
        "municipios_com_resultado_final":len(p),"alunos":int(p["alunos"].sum()),
        "probabilidade_media_risco_ponderada":float(np.average(p["probabilidade_media_risco"],weights=p["alunos"])),
        "taxa_observada_alfabetizacao_ponderada":float(np.average(p["taxa_observada_alfabetizacao"],weights=p["alunos"])),
    }),include_groups=False).reset_index()
    profiles=profiles.merge(relation_summary,on="cluster",how="left")

    representatives=(municipal.sort_values(["cluster","distancia_centroide"])
                     .groupby("cluster",as_index=False).head(5))
    representative_map={int(c):g[["id_municipio_nome","sigla_uf","distancia_centroide"]].to_dict("records")
                        for c,g in representatives.groupby("cluster")}

    stability=[]
    for seed in (7,21,42,84,123):
        candidate=build_pipeline(selected_k,seed).fit_predict(municipal[FEATURES])
        stability.append({"random_state":seed,"adjusted_rand_vs_seed_42":float(adjusted_rand_score(labels,candidate)),
                          "cluster_sizes":pd.Series(candidate).value_counts().sort_index().to_dict()})
    pca=PCA(n_components=2,random_state=RANDOM_STATE).fit(transformed)
    coords=pca.transform(transformed)

    REPORTS.mkdir(exist_ok=True); IMAGES.mkdir(parents=True,exist_ok=True)
    municipal.to_csv(REPORTS/"municipal_cluster_assignments.csv",index=False)
    profiles.to_csv(REPORTS/"municipal_cluster_profiles.csv",index=False)
    pca_report={"use":"visualização apenas; clusters definidos no espaço padronizado completo",
                "explained_variance_ratio":pca.explained_variance_ratio_.tolist(),
                "explained_variance_total":float(pca.explained_variance_ratio_.sum())}
    (REPORTS/"municipal_clustering_pca.json").write_text(json.dumps(pca_report,indent=2)+"\n",encoding="utf-8")
    report={"unit":"município","municipalities_before_imputation":len(municipal),
            "municipalities_after_imputation":len(municipal),"candidates":CANDIDATES,"features":FEATURES,
            "removed":{"gap_para_meta_municipio_2024":"derivada de meta - taxa",
                       "idhm_educacao; idhm_renda; idhm_longevidade":"redundância com IDHM agregado",
                       "total_alunos_municipio_2023":"assimetria forte e efeito de porte",
                       "pct_alfabetizados_municipio_2023":"equivalente à taxa histórica",
                       "proficiencia_media_ponderada_2023":"equivalente à média de Português"},
            "missing_before_imputation":missing,"imputation":"mediana por feature","scaling":"StandardScaler",
            "aggregation":variation,"k_metrics":metrics.to_dict("records"),"selected_k":selected_k,
            "selection_rule":"silhouette amostral principal (até 3.000, seed 42); solução dentro de 0,02 do melhor, sem cluster <3%, preferindo menor k",
            "profiles":profiles.to_dict("records"),"representatives":representative_map,
            "stability":stability,"pca":pca_report,"supervised_model_changed":False,
            "second_test_evaluation":False,"dataset_sha256":sha256(DATASET_PATH),
            "model_sha256":sha256(ROOT/"models"/"final_candidate_validation.joblib")}
    (REPORTS/"municipal_clustering_metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    plt.figure(figsize=(7,4)); plt.plot(metrics["k"],metrics["silhouette"],marker="o")
    plt.xlabel("k");plt.ylabel("Silhouette");plt.title("Silhouette por k");plt.xticks(K_VALUES);plt.tight_layout();plt.savefig(IMAGES/"01_silhouette_by_k.png",dpi=150);plt.close()
    sizes=municipal["cluster"].value_counts().sort_index();plt.figure(figsize=(7,4));plt.bar(sizes.index.astype(str),sizes.values);plt.xlabel("Cluster");plt.ylabel("Municípios");plt.title("Tamanho dos clusters");plt.tight_layout();plt.savefig(IMAGES/"02_cluster_sizes.png",dpi=150);plt.close()
    z=profiles.set_index("cluster")[[f"z_{c}" for c in FEATURES]];plt.figure(figsize=(10,4));plt.imshow(z,aspect="auto",cmap="RdYlBu",vmin=-2,vmax=2);plt.colorbar(label="Centroide padronizado");plt.yticks(range(len(z)),z.index);plt.xticks(range(len(FEATURES)),FEATURES,rotation=35,ha="right");plt.tight_layout();plt.savefig(IMAGES/"03_standardized_centroids.png",dpi=150);plt.close()
    plt.figure(figsize=(8,6));
    for c in range(selected_k):
        mask=labels==c;plt.scatter(coords[mask,0],coords[mask,1],s=9,alpha=.55,label=f"Cluster {c}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})");plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})");plt.title("PCA 2D apenas para visualização");plt.legend();plt.tight_layout();plt.savefig(IMAGES/"04_pca_clusters.png",dpi=150);plt.close()

    profile_lines="\n".join(f"- Cluster {r.cluster} — **{r.perfil_interpretativo}**: {r.municipios} municípios ({r.percentual_municipios:.1%}); risco médio pós-hoc {r.probabilidade_media_risco_ponderada:.4f}." for r in profiles.itertuples())
    rep_lines="\n".join(f"- Cluster {c}: "+", ".join(f"{x['id_municipio_nome']} ({x['sigla_uf']})" for x in rows) for c,rows in representative_map.items())
    md=f"""# Perfis territoriais por K-means\n\n## Classificação curricular\n\nK-means, clustering não supervisionado e Silhouette Score são conteúdos curriculares explícitos FIAP. Features, faixa de k, regras de agregação/seleção e interpretação são decisões metodológicas do grupo. O uso territorial é aplicação analítica do grupo; PCA, estabilidade e relação pós-hoc com risco são extensões.\n\n## Método\n\nUma linha por município ({len(municipal):,}). Redes foram agregadas por média ponderada pelo número de alunos. Missing foi imputado por mediana e as cinco features foram padronizadas com StandardScaler antes do K-means: {', '.join(f'`{c}`' for c in FEATURES)}. Target, IDs, UF, rede e probabilidades supervisionadas ficaram fora do treinamento.\n\nForam avaliados k=2..6. Escolheu-se k={selected_k} com Silhouette amostral {metrics.set_index('k').loc[selected_k,'silhouette']:.4f} (3.000 municípios, seed 42), conciliando score, interpretabilidade e tamanho mínimo. O K-means foi ajustado nos 5.517 municípios.\n\n## Perfis encontrados\n\n{profile_lines}\n\nMunicípios próximos aos centroides:\n{rep_lines}\n\nOs clusters diferem principalmente em desempenho histórico, Português, participação, meta e IDHM. A relação com risco foi calculada somente depois dos clusters usando as previsões finais já registradas; não entrou no K-means.\n\n## Resposta ao Tech Challenge\n\nA análise identificou {selected_k} perfis territoriais entre os municípios. Eles separam contextos relativos de maior vulnerabilidade educacional, melhor desempenho/contexto socioeconômico, participação mais baixa e posições intermediárias, conforme os centroides reais. Municípios do mesmo cluster compartilham padrões contextuais semelhantes; isso não implica causalidade ou equivalência completa.\n\nComo aplicação analítica do grupo, perfis de maior vulnerabilidade podem apoiar priorização de acompanhamento e recursos; intermediários, monitoramento direcionado; e perfis mais favoráveis, benchmarking cuidadoso de práticas.\n\nPCA explica {pca_report['explained_variance_total']:.1%} e foi usada apenas para visualização. A estabilidade por sementes está no JSON. O classificador, dataset e protocolo final não foram alterados; o teste não foi reavaliado.\n"""
    (REPORTS/"municipal_clustering_results.md").write_text(md,encoding="utf-8")
    print(json.dumps({"municipalities":len(municipal),"selected_k":selected_k,"metrics":metrics.to_dict("records"),"sizes":sizes.to_dict(),"stability":stability},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
