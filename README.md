# Tech Challenge - Fase 3

## Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto desenvolvido durante a **Fase 3 da Pós-Tech Data & AI Scientist**, com foco na aplicação de técnicas de Machine Learning para predição da alfabetização individual a partir de dados educacionais, territoriais e socioeconômicos.

> **Status:** fluxo de dados, modelagem supervisionada, avaliação territorial, auditorias metodológicas e clustering municipal concluídos.

---

## Sobre o projeto

Este projeto dá continuidade ao pipeline de Engenharia de Dados desenvolvido durante o **Tech Challenge da Fase 2**.

Na etapa anterior foram construídas pipelines para ingestão, tratamento, integração e disponibilização dos dados em diferentes camadas, incluindo a **camada Gold**, utilizada agora como principal fonte de features analíticas para a modelagem.

O objetivo da Fase 3 é transformar os dados tratados anteriormente em uma solução de **Machine Learning supervisionado**, capaz de estimar a probabilidade de um aluno ser classificado como alfabetizado.

Além da capacidade preditiva, o projeto busca gerar inteligência analítica sobre fatores educacionais, territoriais e socioeconômicos associados à alfabetização.

---

## Objetivo analítico

O problema foi estruturado como uma tarefa de **classificação binária supervisionada**.

A unidade de análise corresponde ao **aluno avaliado em 2024**.

A variável-alvo utilizada é:

- `0`: não alfabetizado;
- `1`: alfabetizado.

O objetivo do modelo é estimar a condição de alfabetização individual utilizando informações históricas e contextuais disponíveis sem utilizar o resultado contemporâneo da avaliação do próprio aluno.

A hipótese central é que a combinação entre:

- histórico educacional do município e da rede;
- contexto territorial;
- indicadores socioeconômicos;
- metas educacionais;

contém informação relevante para estimar a probabilidade de alfabetização individual.

---

## Arquitetura e origem dos dados

A Fase 3 reutiliza os dados tratados e integrados durante o **Tech Challenge da Fase 2**.

Na Fase 2, os dados originais foram obtidos principalmente por meio da **Base dos Dados / Google BigQuery**, processados pelo pipeline de Engenharia de Dados e posteriormente disponibilizados em diferentes camadas.

Para a Fase 3, as **features educacionais, territoriais e socioeconômicas utilizadas na modelagem são provenientes da camada Gold desenvolvida na Fase 2**, armazenada no Amazon S3.

As principais tabelas Gold utilizadas são:

- `indicador_alfabetizacao_municipio`;
- `comparativo_metas_resultados`;
- `evolucao_temporal_indicador`;
- `desempenho_alunos_municipio`.

Essas tabelas foram auditadas quanto à:

- granularidade;
- cobertura;
- temporalidade;
- consistência das chaves;
- valores ausentes;
- redundância entre indicadores;
- disponibilidade de informações territoriais e socioeconômicas.

A integração dessas fontes resultou em uma camada consolidada de features históricas utilizada na construção do dataset de modelagem.

---

## Estratégia temporal

O target corresponde à classificação individual observada em **2024**.

Para reduzir o risco de **data leakage**, os principais indicadores educacionais utilizados como preditores representam informações históricas referentes a **2023**.

Entre eles estão:

- taxa de alfabetização do município/rede em 2023;
- média de Português em 2023;
- percentual de participação em 2023;
- total de alunos no contexto município/rede;
- percentual histórico de alfabetizados;
- proficiência média ponderada;
- indicadores socioeconômicos;
- metas educacionais disponíveis para 2024.

Variáveis contemporâneas derivadas diretamente do desempenho observado dos alunos em 2024 não são utilizadas como preditores.

---

## Dataset de modelagem

O dataset final possui granularidade individual e contém:

- **1.851.828 alunos**;
- **24 colunas**;
- ausência de duplicidades em `id_aluno`;
- ausência de valores nulos na variável-alvo.

A distribuição da variável-alvo é aproximadamente:

- **59,8% alfabetizados**;
- **40,2% não alfabetizados**.

A população analisada está distribuída entre:

- **86,98% rede Municipal**;
- **13,02% rede Estadual**.

A integração com o contexto histórico da camada Gold apresentou cobertura de aproximadamente **98,08% dos alunos**.

O dataset materializado localmente é armazenado em:

```text
data/processed/modeling_dataset_2024_gold.parquet
```

O arquivo não é versionado no GitHub por representar um artefato derivado de grande volume que pode ser reproduzido a partir do pipeline do projeto.

O contrato e as regras de validação estão em
[Definição do dataset Gold](reports/modeling_dataset_definition.md).
O fluxo oficial combina a população individual da Silver da Fase 2
(partição `2026-07-09`) com as features Gold (partição `2026-07-10`).
Usa somente AWS/S3, com credenciais em `.env` ou `env` na raiz:

```bash
python -m src.preprocessing.gold_pipeline
```

A população é filtrada para 2024, presença, prova preenchida e redes
Estadual/Municipal. O arquivo `modeling_dataset_2024.parquet` e o pipeline
BigQuery são legados opcionais e não são necessários nesse fluxo.

Para validar o parquet já disponível localmente, sem acessar as fontes:

```bash
python -m src.preprocessing.validate_dataset
```

Os testes sintéticos independem de credenciais e do parquet:

```bash
python -m tests.test_dataset_contract
```


---

## Features disponíveis

As features candidatas à modelagem incluem diferentes dimensões.

### Contexto educacional histórico

- `taxa_alfabetizacao_municipio_2023`;
- `media_portugues_municipio_2023`;
- `percentual_participacao_municipio_2023`;
- `total_alunos_municipio_2023`;
- `pct_alfabetizados_municipio_2023`;
- `proficiencia_media_ponderada_2023`.

### Metas educacionais

- `meta_alfabetizacao_municipio_2024`;
- `gap_para_meta_municipio_2024`;
- `atingiu_meta_municipio_2024`.

### Contexto territorial

- `rede`;
- `sigla_uf`.

### Contexto socioeconômico

- `idhm`;
- `idhm_educacao`;
- `idhm_renda`;
- `idhm_longevidade`.

### Variáveis auxiliares

Identificadores como `id_aluno`, `id_escola`, `id_municipio` e nomes geográficos são preservados para rastreabilidade e análises, mas não entram como features preditivas.

A variável `peso_aluno`, relacionada à ponderação amostral, também não entra como feature.

---

## Análise Exploratória de Dados

A EDA foi realizada no notebook:

```text
notebooks/01_eda.ipynb
```

A análise contemplou:

1. visão geral e granularidade do dataset;
2. qualidade dos dados;
3. análise da variável-alvo;
4. distribuições das variáveis numéricas;
5. análise das variáveis categóricas;
6. investigação de outliers;
7. relações entre features e target;
8. análise territorial;
9. correlações e redundância;
10. auditoria de data leakage;
11. definição das hipóteses para modelagem.

---

## Principais resultados da EDA

### Histórico educacional

Os indicadores educacionais históricos apresentaram as associações mais relevantes com a classificação individual observada em 2024.

A taxa histórica de alfabetização do município/rede apresentou diferença média de aproximadamente **7,8 pontos percentuais** entre os contextos associados a alunos alfabetizados e não alfabetizados.

Também foram observadas diferenças relevantes nas medidas históricas de proficiência e percentual de alfabetizados.

### Relação temporal

A segmentação dos alunos em quintis de acordo com a taxa de alfabetização do município/rede em 2023 revelou uma relação monotônica com o resultado individual observado em 2024.

A proporção de alunos alfabetizados evoluiu aproximadamente de:

- **42,6%** no menor quintil histórico;
- **54,7%** no segundo quintil;
- **59,8%** no terceiro quintil;
- **65,1%** no quarto quintil;
- **77,1%** no maior quintil histórico.

A diferença entre os extremos é de aproximadamente **34,5 pontos percentuais**.

Esse resultado reforça a hipótese de que o contexto educacional histórico contém sinal preditivo relevante.

### Contexto territorial

Foram observadas diferenças expressivas na proporção de alunos classificados como alfabetizados entre as UFs presentes no dataset.

Essas diferenças representam exclusivamente a população analisada e **não devem ser interpretadas como indicadores oficiais das taxas estaduais de alfabetização**.

Os resultados sugerem que o contexto territorial pode contribuir para a capacidade preditiva quando combinado com fatores educacionais e socioeconômicos.

### Contexto socioeconômico

Os indicadores de IDHM apresentaram associações individuais mais moderadas com o target.

Como essas informações possuem granularidade estadual, os mesmos valores são compartilhados por grande número de alunos. Sua contribuição foi avaliada em combinação com as demais características do dataset.

---

## Valores ausentes

Os valores ausentes foram preservados durante a construção do dataset.

As principais taxas observadas foram:

- aproximadamente **1,92%** de ausência no contexto histórico principal;
- aproximadamente **4,70%** em algumas informações relacionadas à participação e metas;
- aproximadamente **23,18%** em algumas features provenientes do desempenho agregado.

Nenhuma imputação foi realizada durante a EDA.

O tratamento ocorre diretamente no **pipeline de Machine Learning**, permitindo que os parâmetros de imputação sejam aprendidos exclusivamente a partir do conjunto de treinamento e evitando vazamento de informações entre treino, validação e teste.

---

## Outliers

A EDA identificou forte assimetria principalmente em:

- `peso_aluno`;
- `total_alunos_municipio_2023`.

A investigação mostrou que os maiores valores de `total_alunos_municipio_2023` estão associados a grandes redes de ensino, representando diferenças reais de escala e não erros nos dados.

Os valores extremos de `peso_aluno` também foram preservados.

Nenhuma remoção automática de outliers foi realizada.

---

## Correlação e redundância

A análise identificou forte correlação entre algumas features.

Os principais casos encontrados foram:

- `media_portugues_municipio_2023` × `proficiencia_media_ponderada_2023`: aproximadamente **0,999**;
- `taxa_alfabetizacao_municipio_2023` × `pct_alfabetizados_municipio_2023`: aproximadamente **0,971**;
- `idhm` × `idhm_renda`: aproximadamente **0,957**;
- `idhm` × `idhm_educacao`: aproximadamente **0,926**.

A seleção final considerou o algoritmo, desempenho, capacidade de generalização e interpretabilidade.

---

## Prevenção de data leakage

A temporalidade das features foi auditada antes da modelagem.

Foi confirmado que as principais features educacionais utilizadas representam informações históricas de 2023 ou informações contextuais disponíveis independentemente do resultado individual observado em 2024.

Também foi validado que:

```text
gap_para_meta_municipio_2024
=
meta_alfabetizacao_municipio_2024
-
taxa_alfabetizacao_municipio_2023
```

e que:

```text
atingiu_meta_municipio_2024
=
taxa_alfabetizacao_municipio_2023
>=
meta_alfabetizacao_municipio_2024
```

A validação apresentou correspondência de **100%** para a condição de atingimento da meta.

Dessa forma, essas variáveis não utilizam o resultado individual observado em 2024. A seleção considerou sua redundância com outras features.

---

## Metodologia

O desenvolvimento do projeto está organizado nas seguintes etapas:

1. auditoria e entendimento dos dados;
2. validação da camada Gold da Fase 2;
3. consolidação das features históricas;
4. construção do dataset individual de modelagem;
5. Análise Exploratória de Dados;
6. análise e prevenção de data leakage;
7. definição das features candidatas;
8. separação dos conjuntos de treino, validação e teste;
9. construção do pipeline de pré-processamento;
10. desenvolvimento do modelo baseline;
11. treinamento e comparação de modelos supervisionados;
12. validação e otimização de hiperparâmetros;
13. avaliação de generalização e overfitting;
14. seleção do modelo final;
15. interpretabilidade e explicabilidade;
16. geração de insights analíticos;
17. discussão de aplicações em políticas públicas.

---

## Status do projeto

### Concluído

- [x] Estruturação inicial do repositório
- [x] Auditoria das fontes da Fase 2
- [x] Recuperação das tabelas Gold
- [x] Validação de granularidade e cobertura
- [x] Validação da temporalidade dos dados
- [x] Consolidação das features Gold
- [x] Integração com a população individual de 2024
- [x] Construção do dataset de modelagem
- [x] Análise Exploratória de Dados
- [x] Análise de valores ausentes
- [x] Análise de outliers
- [x] Análise territorial
- [x] Análise socioeconômica
- [x] Análise das relações entre features e target
- [x] Análise de correlações e redundância
- [x] Auditoria de data leakage
- [x] Definição das hipóteses iniciais de modelagem

### Etapas finais

- [x] Estratégia de separação treino/validação/teste por município
- [x] Pipeline de pré-processamento
- [x] Modelos baseline (Dummy, regressão logística e árvore)
- [x] Comparação inicial de modelos supervisionados
- [x] Seleção controlada de hiperparâmetros na validação
- [x] Avaliação inicial de generalização e overfitting
- [x] Seleção do modelo final
- [x] Interpretabilidade inicial por Feature Importance
- [ ] SHAP (adiado por custo computacional)
- [x] Análise municipal diagnóstica na validação
- [x] Documentação técnica final
- [ ] Apresentação e vídeo executivo

---

## Estrutura do projeto

```text
tech-challenge-fase3/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── gold/
│
├── notebooks/
│   ├── 00_test_build_modeling_dataset.ipynb
│   ├── 00_test_gold_integration.ipynb
│   └── 01_eda.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   ├── evaluation/
│   └── visualization/
│
├── reports/
├── images/
├── requirements.txt
├── README.md
└── .gitignore
```

### Diretórios

- `data/`: artefatos de dados utilizados durante o desenvolvimento;
- `notebooks/`: auditorias, análises exploratórias e experimentos;
- `src/preprocessing/`: leitura, integração, preparação e transformação dos dados;
- `src/modeling/`: treinamento, pipelines e otimização dos modelos;
- `src/evaluation/`: métricas e avaliação;
- `src/visualization/`: funções e recursos de visualização;
- `reports/`: documentação técnica, decisões analíticas e resultados;
- `images/`: gráficos e imagens utilizados na documentação.

---

## Modelagem

X, y, groups e a divisão municipal estão definidos em
[Definição do split](reports/modeling_split_definition.md). O comando
`python -m src.modeling.split` reproduz a divisão do parquet local,
sem treinamento ou pré-processamento aprendido.

O problema foi tratado como classificação binária supervisionada. Das 16
features brutas, o pré-processamento integrado ao `sklearn Pipeline` produz 43
colunas. Ele inclui:

- imputação de valores numéricos ausentes;
- tratamento das variáveis categóricas;
- encoding;
- transformações numéricas;
- treinamento do estimador.

Foram comparados Dummy, regressão logística, árvore de decisão e Random Forest.
O modelo final é uma Random Forest controlada com `n_estimators=40`,
`max_depth=10`, `min_samples_leaf=200`, `max_features="sqrt"`, `n_jobs=2` e
`random_state=42`. A seleção usou somente treino e validação; a ROC AUC de
validação foi 0,6409.

O protocolo e os resultados estão em
[Protocolo do modelo final](reports/final_model_protocol.md) e
[Resultados de validação](reports/final_validation_results.md).
As restrições de interpretação e generalização estão em
[Limitações da modelagem](reports/modeling_limitations.md).
O diagnóstico entre target individual e features contextuais está em
[Auditoria de granularidade](reports/granularity_audit.md).
A linhagem do target, o risco de `proficiencia` e as candidatas futuras estão em
[Auditoria da linhagem](reports/target_lineage_audit.md) e
[Auditoria de candidatas](reports/feature_candidate_audit.md).

```bash
python -m src.modeling.modeling_eda
python -m src.modeling.train_baselines
python -m src.modeling.final_validation
```

Esses comandos trabalham com treino e validação. O artefato `.joblib` não é
versionado e é regenerado pelo treinamento/validação. A avaliação final não deve
ser repetida, pois o teste já foi aberto uma única vez.

---

## Avaliação

O split agrupado por município contém 1.311.003 alunos no treino, 297.079 na
validação e 243.746 no teste, sem sobreposição de municípios. A seleção utilizou
ROC AUC, balanced accuracy, precision, recall e F1 por classe. Não existe limiar
operacional congelado; 0,50 aparece somente como referência descritiva.

---

## Interpretabilidade

Foi registrada a importância das features da Random Forest. SHAP não foi
executado devido ao custo computacional e não é requisito para reproduzir os
resultados. A auditoria mostrou que a proficiência individual reconstrói o target
com corte 743; ela foi excluída por leakage direto.

---

## Insights e aplicação em políticas públicas

O score `prob_risco = 1 - P(alfabetizado)` permite ordenar territórios para
diagnóstico e priorização de investigação. Ele pode apoiar alocação de suporte,
monitoramento e estudos locais, sem substituir avaliação pedagógica individual
nem sustentar conclusões causais.

---

## Limitações

Algumas limitações já identificadas incluem:

- parte das features possui granularidade municipal/rede ou estadual, enquanto o target possui granularidade individual;
- alguns indicadores apresentam valores ausentes;
- os indicadores socioeconômicos disponíveis atualmente possuem granularidade estadual;
- algumas features educacionais apresentam forte correlação entre si;
- associações encontradas não representam necessariamente relações causais;
- a capacidade de generalização dependerá da estratégia de validação e da representatividade dos dados disponíveis.

Alunos do mesmo município e rede compartilham predominantemente o mesmo vetor
de features, e cerca de 90% da variabilidade do target ocorre dentro desses
contextos. O modelo representa sobretudo risco contextual/territorial e não é
diagnóstico individual. A POC com Censo Escolar teve cobertura zero ao juntar
`id_escola` e `CO_ENTIDADE`, indicando chave anonimizada ou recodificada; por
isso, essa fonte externa permitida não foi incorporada ao modelo final.

---

## Avaliação final em municípios inéditos

Após o congelamento e a aprovação do protocolo, o conjunto de teste foi aberto
uma única vez. Aplicou-se a Random Forest já ajustada somente no treino, com 40
árvores, profundidade 10, folha mínima 200, `max_features="sqrt"` e semente 42.
Não houve refit, retuning ou alteração de features após a abertura.

O teste contém 243.746 alunos de 828 municípios ausentes do treino e da
validação. A ROC AUC foi 0,6631, contra 0,6409 na validação. No limiar 0,50,
usado apenas como referência descritiva, a balanced accuracy foi 0,5832; para a
classe de risco, precision/recall/F1 foram 0,5609/0,3426/0,4254.

Não foi congelado threshold operacional. A análise territorial ordena
municípios e redes por `prob_risco = 1 - P(alfabetizado)`. O ranking indica risco
relativo no conjunto de atributos contextuais e não constitui diagnóstico
individual nem previsão oficial de cumprimento de metas. Resultados completos:
`reports/final_test_results.md` e `reports/final_test_metrics.json`.

---

## Perfis territoriais por clustering

Uma análise complementar de K-means agrupou os 5.517 municípios usando taxa de
alfabetização 2023, média de Português, participação, meta 2024 e IDHM. Os dados
municipais foram imputados por mediana e padronizados; nenhum target, ID, rede,
UF ou score supervisionado entrou no treinamento.

Foram avaliados `k=2..6`. `k=2` foi escolhido com Silhouette amostral 0,3591 e
separou 3.371 municípios de contexto educacional e socioeconômico relativamente
mais favorável de 2.146 municípios com maior vulnerabilidade relativa. O segundo
perfil apresentou probabilidade média de risco pós-hoc de 0,4890, contra 0,2936
no primeiro. O resultado apoia priorização territorial e não tem interpretação
causal. Detalhes: `reports/municipal_clustering_results.md`.

---

## Reprodutibilidade

O projeto foi estruturado para permitir reprodução das principais etapas por meio dos scripts e notebooks versionados no repositório.

Os arquivos derivados de grande volume, incluindo o dataset final de modelagem, não são versionados diretamente no GitHub.

No Windows, crie o ambiente e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Crie o arquivo local `env` com `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION` e `S3_BUCKET_NAME`, sem versioná-lo. O
fluxo oficial usa a Silver e a Gold no S3 e não requer GCP:

```powershell
python -m src.preprocessing.gold_pipeline
python -m src.preprocessing.validate_dataset
python -m src.modeling.modeling_eda
python -m src.modeling.train_baselines
python -m src.modeling.final_validation
python -m src.modeling.municipal_clustering
```

O código de BigQuery/Base dos Dados permanece como caminho legado e opcional.
Não execute novamente `src.modeling.final_test_evaluation`.

---

## Evoluções futuras

Possíveis extensões incluem:

- enriquecimento com novas variáveis socioeconômicas e territoriais;
- incorporação de indicadores em granularidades mais detalhadas;
- avaliação temporal com novos ciclos da avaliação;
- monitoramento de drift;
- criação de pipelines automatizados de treinamento e inferência;
- disponibilização das previsões por meio de aplicações analíticas;
- expansão das análises voltadas à identificação de regiões e grupos prioritários.

---

## Tecnologias

O projeto utiliza principalmente:

- Python;
- Pandas;
- NumPy;
- Scikit-learn;
- Jupyter Notebook;
- Parquet;
- Amazon S3;
- Google BigQuery/Base dos Dados (legado opcional);
- Git;
- GitHub.

Os próximos passos reais são preparar a apresentação e o vídeo executivo,
avaliar novas fontes com chaves compatíveis e monitorar estabilidade em ciclos
futuros.
