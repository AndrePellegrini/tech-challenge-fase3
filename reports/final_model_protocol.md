# Protocolo congelado do modelo candidato

Dataset: `data/processed/modeling_dataset_2024_gold.parquet`, 1.851.828 × 24.
Versão da decisão: 2026-09-09. Semente: 42.

Features candidatas finais (16): `rede`, `sigla_uf`, `taxa_alfabetizacao_municipio_2023`, `media_portugues_municipio_2023`, `percentual_participacao_municipio_2023`, `total_alunos_municipio_2023`, `pct_alfabetizados_municipio_2023`, `proficiencia_media_ponderada_2023`, `meta_alfabetizacao_municipio_2024`, `gap_para_meta_municipio_2024`, `idhm`, `idhm_educacao`, `idhm_renda`, `idhm_longevidade`, `atingiu_meta_municipio_2024`, `gold_historico_disponivel`.
Excluídas: target, peso e todos os identificadores; `proficiencia` é proibida por leakage direto.

Preprocessing: mediana numérica, moda categórica/booleana, OneHotEncoder sparse com categorias desconhecidas ignoradas; escala sparse somente para regressão logística.

Algoritmo candidato: `random_forest_controlled`. Parâmetros: `{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 200, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 40, 'n_jobs': 2, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}`.
Threshold binário: **não congelado após auditoria semântica**. A proposta 0,35
foi retirada porque as métricas padrão tratavam `1 = alfabetizado` como classe
positiva; reduzir o limiar aumentava previsões de alfabetizado e reduzia a
detecção da classe 0. Para priorização territorial, usar
`prob_risco = 1 - predict_proba[:, 1]` como ranking contínuo. Se uma decisão
binária vier a ser operacionalmente necessária, 0,60 em P(alfabetizado) é o
ponto de equilíbrio observado na validação e exige aprovação específica.
Sample weight oficial: não.
Métricas de seleção: ROC AUC principal; recall e F1; gap treino/validação; interpretabilidade, custo e simplicidade.

A interpretação é territorial: o modelo estima probabilidade individual condicionada ao contexto municipal, rede e UF; alunos do mesmo município+rede recebem a mesma probabilidade. Não é diagnóstico individual nem previsão oficial de meta.

Limitações: features contextuais repetidas, generalização territorial limitada, ausência de atributos individuais legítimos e incerteza sobre o significado operacional de `peso_aluno`. SHAP não foi executado porque o pacote não está instalado.

**O conjunto de teste só será acessado após aprovação deste protocolo corrigido.**
