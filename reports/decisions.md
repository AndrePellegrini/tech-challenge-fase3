# Registro de Decisões Técnicas e Analíticas

Este documento registra as principais decisões tomadas durante o desenvolvimento do Tech Challenge da Fase 3.

O objetivo é manter a rastreabilidade das escolhas relacionadas aos dados, preparação, modelagem, avaliação e interpretação dos resultados.

As decisões serão registradas conforme o projeto evoluir, evitando justificativas construídas apenas ao final do trabalho.

---

## DEC-001 — Separação do projeto da Fase 3

**Status:** Aprovada

**Contexto:**
O Tech Challenge da Fase 3 dá continuidade ao trabalho desenvolvido na Fase 2, porém possui foco diferente. Enquanto a etapa anterior concentrou-se na construção e organização da arquitetura de dados, a nova fase tem como foco análise exploratória, Machine Learning, avaliação, otimização e interpretação dos modelos.

**Decisão:**
Criar um novo repositório denominado `tech-challenge-fase3`, mantendo o repositório da Fase 2 preservado.

**Justificativa:**
A separação permite manter o histórico da solução de Engenharia de Dados da Fase 2 e organizar a nova etapa de Ciência de Dados e Machine Learning de maneira independente.

**Impacto:**
Os dados tratados e integrados na Fase 2 serão utilizados como fonte para o desenvolvimento da Fase 3, sem misturar as responsabilidades dos dois projetos.

---

## DEC-002 — Utilização dos dados da Fase 2 como ponto de partida

**Status:** Em análise

**Contexto:**
O Tech Challenge orienta a utilização dos dados tratados na etapa anterior como base para o desenvolvimento das análises e modelos da Fase 3.

**Decisão:**
A definição do dataset de modelagem será realizada somente após uma auditoria das bases produzidas na Fase 2.

**Justificativa:**
Antes de definir features, target ou algoritmos, é necessário confirmar:

* granularidade dos dados;
* unidade de análise;
* período disponível;
* qualidade dos registros;
* variáveis disponíveis;
* existência de informações individuais ou agregadas;
* possibilidade de construção da variável-alvo;
* possíveis riscos de data leakage.

**Impacto:**
Nenhum modelo será treinado antes da conclusão da auditoria dos dados.

---

## DEC-003 — Estratégia de desenvolvimento

**Status:** Aprovada

**Contexto:**
Projetos de Machine Learning podem concentrar preparação, exploração e modelagem em notebooks extensos, dificultando manutenção e reprodutibilidade.

**Decisão:**
Utilizar notebooks para exploração, análise e experimentação, mantendo código reutilizável e consolidado dentro de `src/`.

**Estrutura:**

* `notebooks/`: exploração e experimentos;
* `src/preprocessing/`: preparação e transformação dos dados;
* `src/modeling/`: treinamento e otimização;
* `src/evaluation/`: avaliação e métricas;
* `src/visualization/`: visualizações reutilizáveis;
* `reports/`: documentação técnica e decisões;
* `images/`: gráficos utilizados na documentação.

**Justificativa:**
Essa separação melhora organização, legibilidade, reutilização e reprodutibilidade.

---

## DEC-004 — Prevenção de data leakage

**Status:** Diretriz aprovada

**Contexto:**
O Tech Challenge exige atenção explícita ao risco de data leakage durante preparação, seleção de variáveis e modelagem.

**Decisão:**
Toda variável candidata à modelagem será avaliada quanto ao risco de utilizar informações que não estariam disponíveis no momento real da previsão ou que sejam derivadas direta ou indiretamente da variável-alvo.

As transformações aprendidas a partir dos dados deverão ser ajustadas apenas sobre os dados de treinamento.

**Justificativa:**
Data leakage pode produzir métricas artificialmente elevadas e comprometer a capacidade de generalização do modelo.

**Impacto:**
Durante a auditoria será criado um dicionário classificando as variáveis, quando aplicável, em:

* variável-alvo;
* identificador;
* feature candidata;
* feature suspeita;
* leakage;
* variável não utilizada.

---

## DEC-005 — Uso de Pipeline para preparação e modelagem

**Status:** Diretriz aprovada

**Contexto:**
O projeto poderá exigir diferentes tratamentos para variáveis numéricas e categóricas, além de imputação, encoding, scaling e outras transformações.

**Decisão:**
Priorizar `Pipeline` e `ColumnTransformer` do Scikit-learn para integrar preprocessing e modelagem.

**Justificativa:**
Essa estratégia:

* reduz risco de data leakage;
* garante aplicação consistente das transformações;
* facilita validação cruzada;
* facilita otimização de hiperparâmetros;
* melhora a reprodutibilidade.

**Impacto:**
As transformações definitivas não serão realizadas manualmente sobre todo o dataset antes da separação entre treino e teste.

---

# Decisões pendentes

As decisões abaixo serão preenchidas conforme avançarmos no projeto.

## DEC-006 — Definição da unidade de análise e variável-alvo

**Status:** Definida após integração Gold e EDA.

**Classificação:** decisão específica do grupo.

Uma linha por aluno avaliado em 2024 nas redes Estadual e Municipal,
com target binário `alfabetizado`. A definição vigente consta em
[Definição do dataset Gold](modeling_dataset_definition.md).

---

## DEC-007 — Estratégia de separação entre treino, validação e teste

**Status:** Definida e validada no parquet Gold em 2026-09-08.

**Classificação:** decisão metodológica do grupo, não prescrição da FIAP.

Usar `id_municipio` como group, com GroupShuffleSplit em dois estágios
(70/30 e 50/50 do temporário), semente 42 em ambos. Nenhum município pode
aparecer em mais de um conjunto. X inicial contém 16 candidatas;
identificadores e peso ficam separados. Não houve seleção por correlação.
A população real resultou em 1.311.003/297.079/243.746 alunos nos conjuntos
treino/validação/teste, com zero overlap e classes próximas ao global.
A decisão completa, classificação das colunas e limitações estão em
[Definição de X/y/groups e split](modeling_split_definition.md).

---

## DEC-008 — Métrica principal de avaliação

**Status:** Definida provisoriamente para baselines.

**Classificação:** decisão metodológica do grupo — REVISAR COM O GRUPO.

Comparar modelos primeiro por ROC AUC na validação, considerando F1 e recall no
contexto de identificação de risco. Manter limiar padrão 0,5 nesta rodada. A
decisão evita selecionar o Dummy pela accuracy/F1 de uma única classe e não é
atribuída como prescrição específica da FIAP.

---

## DEC-009 — Algoritmos candidatos

**Status:** Concluída. Onze candidatos comparados em validação.

**Classificação:** decisão metodológica do grupo.

A comparação progressiva partiu de Dummy, regressão logística e árvore de decisão.
Uma primeira tentativa de Random Forest foi interrompida por custo local, mas a
configuração foi posteriormente limitada (40 árvores, profundidade 10, folha mínima
200, `max_features="sqrt"`, dois jobs) e executada com sucesso.

O conjunto final avaliado em validação tem onze candidatos: `dummy_prior`, três
regressões logísticas (C = 1,0 / 0,3 / 0,1), seis árvores de decisão e a Random
Forest controlada. Não foi adicionada dependência de boosting. Resultados completos
em [Validação final](final_validation_results.md) e `final_model_comparison.csv`.

---

## DEC-010 — Estratégia de validação cruzada

**Status:** Executada. `GroupKFold` de 5 folds sobre treino e validação.

**Classificação:** decisão metodológica do grupo.

**Escopo:** `GroupKFold` agrupado por `id_municipio`, 5 folds, sobre a união de
treino e validação — 1.608.082 alunos em 4.689 municípios. O conjunto de teste não
foi materializado, e há verificação em código que falha se índices de teste
aparecerem no desenvolvimento.

Foram avaliados os quatro finalistas, e não os onze candidatos. Onze modelos por
cinco folds seriam 55 ajustes, e só a Random Forest custa cerca de 120 segundos por
ajuste. Os finalistas cobrem a faixa de decisão real do holdout.

**Resultado:**

| Modelo | AUC média | Desvio | AUC do holdout |
|---|---:|---:|---:|
| `random_forest_controlled` | 0,6608 | 0,0079 | 0,6409 |
| `logistic_baseline` | 0,6560 | 0,0146 | 0,6186 |
| `decision_tree_d5_l100` | 0,6512 | 0,0114 | 0,6307 |
| `decision_tree_d10_l100` | 0,6504 | 0,0083 | 0,6302 |

**Dois achados relevantes.**

Primeiro, a ordem dos finalistas **difere** da do holdout: a regressão logística era
a última entre eles no holdout e aparece em segundo na validação cruzada. O holdout
único era, portanto, sensível à partição — exatamente o risco que motivou esta
execução.

Segundo, a diferença de 0,0047 entre o primeiro e o segundo colocado **não supera** a
dispersão combinada entre folds (0,0166). Random Forest e regressão logística são
estatisticamente indistinguíveis nesta evidência. A escolha do modelo final se
sustenta pelos critérios de desempate da DEC-012 — menor gap entre treino e validação
e maior estabilidade das probabilidades agregadas —, e não pela ROC AUC isolada.

**Consequência para a avaliação final:** a média da validação cruzada (0,6608) fica a
0,0023 da AUC de teste (0,6631), dentro de um desvio-padrão entre folds, enquanto o
holdout de validação ficou 0,0199 abaixo. Isso esclarece a diferença de +0,0222 entre
teste e validação que o protocolo havia classificado como moderada: a partição de
validação era pessimista, e não a de teste favorável.

Resultados completos em [Validação cruzada](cross_validation_results.md).

---

## DEC-011 — Estratégia de otimização de hiperparâmetros

**Status:** Concluída por grade manual controlada.

**Classificação:** decisão metodológica do grupo.

A otimização foi feita por grade manual explícita, declarada em código, e não por
`GridSearchCV` ou `RandomizedSearchCV`. Variaram-se a regularização da regressão
logística (C = 1,0 / 0,3 / 0,1) e a capacidade das árvores (profundidade 5, 7, 10,
14 e 15; folha mínima 100 e 500), além da Random Forest com capacidade limitada.

**Justificativa:** a busca automatizada multiplicaria o número de ajustes sobre
1,85 milhão de linhas sem validação cruzada que sustentasse a comparação. A grade
manual mantém cada configuração rastreável e reprodutível.

**Limitação reconhecida:** a grade é pequena e definida a priori; não há garantia
de que o ótimo esteja dentro dela.

---

## DEC-012 — Seleção do modelo final

**Status:** Concluída. Modelo final selecionado, congelado e avaliado uma única vez.

**Classificação:** decisão metodológica do grupo.

**Modelo final:** `random_forest_controlled` — `RandomForestClassifier` com 40
estimadores, profundidade máxima 10, folha mínima 200, `max_features="sqrt"` e
`random_state=42`.

**Regra de seleção:** entre os candidatos a até 0,002 da melhor ROC AUC de
validação, prefere-se o de menor gap treino/validação; depois maior F1; depois
menor tempo de treino. A regra evita escolher por diferenças de AUC dentro do
ruído, coerente com a ausência de validação cruzada (DEC-010).

**Resultado em validação:** ROC AUC 0,6409, gap treino/validação 0,0343. Os
concorrentes mais próximos foram `decision_tree_d5_l100` (0,6307, gap 0,0310) e
`decision_tree_d10_l100` (0,6302, gap 0,0449).

**Resultado no teste:** ROC AUC 0,6631 em 828 municípios inéditos e 243.746
alunos, sem qualquer sobreposição territorial ou individual. A diferença em
relação à validação é de +0,0222, classificada como moderada pelo próprio
protocolo. O teste ficou **acima** da validação, o que afasta overfitting de
seleção e sugere partição de teste marginalmente mais favorável.

**Protocolo de abertura única:** o modelo foi congelado antes do teste, o
conjunto foi aberto uma só vez, não houve refit nem retuning posterior. O
bloqueio é garantido em código e coberto por testes automatizados.

Detalhes em [Protocolo congelado](final_model_protocol.md) e
[Avaliação final](final_test_results.md).

---

## DEC-013 — Estratégia de interpretabilidade

**Status:** Feature Importance do modelo final concluída; SHAP permanece pendente.

**Classificação:** curricular complementar/recomendado e decisão de execução do grupo.

Foi usada a importância nativa da Random Forest final, reagregada das 43 colunas
codificadas para as 16 features de origem, de modo que as categorias one-hot de
`rede` e `sigla_uf` não apareçam fragmentadas. Resultado em
`final_feature_importance.csv`.

SHAP não foi executado. A execução depende do modelo serializado, que não é
versionado, e do parquet de modelagem, que também não é. Com o ambiente montado o
custo é baixo para uma floresta de 40 árvores e profundidade 10. Permanece como
pendência explícita, e não como decisão de descarte.

Nenhuma importância recebe interpretação causal.

---

## DEC-014 — Análises complementares

**Status:** Parcialmente concluída. Agrupamento executado; análise temporal descartada.

**Classificação:** decisão metodológica do grupo.

* **Agrupamento de municípios — executado.** K-means sobre 5.517 municípios, com
  `Pipeline` de imputação por mediana, padronização e clusterização. Escolhido
  k = 2, silhouette 0,3591, estabilidade ARI entre 0,9971 e 1,0 em cinco sementes.
  PCA usada apenas para visualização, explicando 82,2% da variância. Resultados em
  [Clustering municipal](municipal_clustering_results.md).
* **Identificação de perfis semelhantes — executada** como subproduto do
  agrupamento e da auditoria de granularidade.
* **Análise temporal — descartada nesta fase.** A tabela Gold
  `evolucao_temporal_indicador` existe, mas o dataset de modelagem usa um único ano
  de corte (2023) para as features de contexto. Construir série histórica exigiria
  redesenhar o contrato do dataset, o que está fora do escopo desta entrega.
* **Risco de não atingimento de metas futuras — ver DEC-019.**

**Limitação do agrupamento:** com k = 2 a separação é essencialmente Sul/Sudeste
contra Norte/Nordeste. É interpretável, mas de granularidade baixa para priorização
fina de política pública. A regra de seleção privilegiou a métrica de silhouette;
k = 4 daria segmentação mais acionável ao custo de cerca de 0,08 de silhouette.

## DEC-015 — Estratégia de acesso e materialização dos dados

**Status:** Atualizada após integração Gold (PR #2) e alinhamento do contrato.

**Classificação:** decisão específica do grupo, apoiada no requisito oficial
que determina uso da camada Gold da Fase 2.

**Decisão vigente:** utilizar as features históricas, territoriais e
socioeconômicas da Gold da Fase 2 no S3, consolidadas por `gold_features.py`
e integradas aos alunos de 2024 por `build_modeling_dataset_from_gold`.
O artefato de modelagem é `data/processed/modeling_dataset_2024_gold.parquet`,
formalizado em `dataset_contract.py` e verificado por `validate_dataset.py`.

**Histórico:** a decisão inicial utilizava consultas diretas ao BigQuery e
materializava `modeling_dataset_2024.parquet`. Esse fluxo foi superado como
definição da modelagem. O arquivo é legado opcional. A entrada individual vigente vem diretamente
da Silver de alunos no S3, partição 2026-07-09, filtrada para alunos presentes
com prova preenchida em 2024 nas redes Estadual e Municipal.
`gold_pipeline.py` executa o fluxo Silver + Gold sem BigQuery ou GCP.

**Impactos:** o parquet Gold é um artefato derivado local não versionado.
Reproduzir a integração requer acesso às fontes; validar um parquet já
materializado e executar testes sintéticos não requer credenciais.
O fluxo de reprodução, o schema de 24 colunas e as regras de missingness
estão em [Definição do dataset Gold](modeling_dataset_definition.md).

---

## DEC-016 — Interpretação da granularidade do modelo

**Status:** auditoria concluída; interpretação a revisar com o grupo.

**Classificação:** decisão metodológica do grupo/extensão diagnóstica. O target
individual e o uso da Gold atendem ao Tech Challenge; esta auditoria específica
não é atribuída como prescrição da FIAP.

**Problema:** o target é individual, enquanto as 16 features são de rede,
município+rede, UF ou disponibilidade do contexto.

**Alternativas consideradas:** interpretar a saída como discriminação individual
plena; reformular imediatamente o target; ou preservar a tarefa formal e limitar
a interpretação ao risco condicionado ao contexto.

**Decisão:** preservar target, features e split nesta etapa e interpretar a
probabilidade como condicionada ao contexto territorial/rede. O enriquecimento
com variáveis individuais fica como extensão futura, marcada **REVISAR COM O
GRUPO**.

**Impacto:** alunos do mesmo município+rede recebem o mesmo vetor e a mesma
probabilidade nos pipelines atuais. O uso de negócio mais defensável é priorizar
territórios/redes, sem apresentar o resultado como diagnóstico pedagógico do
aluno. Evidências completas em [Auditoria de granularidade](granularity_audit.md).

---

## DEC-017 — Proficiência contemporânea e enriquecimento escolar

**Status:** auditoria concluída; cenários futuros pendentes do grupo.

**Classificação:** tratamento de data leakage é requisito explícito do Tech
Challenge. A auditoria de linhagem, a matriz temporal e a escolha de fontes são
decisões metodológicas do grupo. Censo Escolar é uma fonte externa permitida,
não obrigatória.

**Problema:** a Silver contém `proficiencia`, que poderia aparentar ser uma
feature individual forte, enquanto o projeto carece de atributos escolares.

**Decisão nesta etapa:** não alterar X. Classificar `proficiencia` 2024 como
leakage direto porque reconstrói `alfabetizado` pelo corte oficial de 743 sem
divergência nos 1.851.828 alunos elegíveis. Manter `serie`, `caderno`, `presenca`
e `preenchimento_caderno` fora de X. Avaliar Censo Escolar 2023 por `id_escola`
como cenário futuro.

**Impacto:** preserva a validade da avaliação atual e abre uma alternativa para
adicionar granularidade escolar pré-target. Integração, cobertura e seleção de
atributos permanecem **REVISAR COM O GRUPO**. Evidências em
[Linhagem do target](target_lineage_audit.md) e
[Matriz de candidatas](feature_candidate_audit.md).

---

## DEC-018 — Origem do target na camada Silver

**Status:** Formalizada.

**Classificação:** desvio consciente em relação ao enunciado, com justificativa técnica.

**Contexto:** o enunciado da Fase 3 determina que os dados de modelagem venham da
camada Gold construída na Fase 2, e ao mesmo tempo exige um modelo que preveja
**se um aluno** será alfabetizado.

**Problema:** as duas exigências são incompatíveis com a Gold existente. O catálogo
Gold da Fase 2 tem quatro tabelas — `indicador_alfabetizacao_municipio`,
`comparativo_metas_resultados`, `evolucao_temporal_indicador` e
`desempenho_alunos_municipio` — e **todas são agregadas** por município, UF ou
Brasil. Não existe tabela Gold em granularidade de aluno, logo o target individual
não pode ser obtido apenas da Gold.

**Decisão:** adotar origem híbrida e declará-la explicitamente.

* **População e target** vêm da Silver `alunos`, partição `processing_date=2026-07-09`.
  Isso inclui `id_aluno`, `id_municipio`, `id_escola`, `rede`, `peso_aluno`, os
  filtros de elegibilidade e a própria coluna `alfabetizado`.
* **Todas as features de contexto** vêm da camada Gold: histórico municipal de 2023,
  desempenho agregado, metas municipais para 2024 e os indicadores IDHM em nível de UF.

**Justificativa:** preservar a granularidade individual exigida pelo enunciado sem
inventar uma tabela Gold que a Fase 2 não produziu. A alternativa seria modelar em
granularidade municipal, o que contrariaria a definição do problema.

**Alternativa não adotada:** criar retroativamente uma Gold `alunos_modelagem` na
Fase 2. Descartada por exigir alteração no escopo de uma fase já entregue.

**Impacto:** o requisito de dados provenientes da camada Gold é atendido
integralmente no conjunto de features e parcialmente na origem da população. A
mistura está documentada no diagrama de
[Definição do dataset](modeling_dataset_definition.md). Evolução recomendada para
uma fase futura: promover a população elegível a uma tabela Gold própria.

---

## DEC-019 — Risco de não atingimento da meta municipal de 2024

**Status:** Concluída e auditada.

**Classificação:** pergunta de negócio exigida pelo Tech Challenge.

**Contexto:** o enunciado pede explicitamente "como prever municípios que podem não
atingir metas futuras". Até esta etapa a pergunta não tinha resposta, e os relatórios
declaravam apenas que o ranking de risco *não* previa atingimento de meta.

**Decisão:** projetar a taxa municipal de 2024 pela média das probabilidades de
alfabetização previstas pelo modelo congelado e compará-la com a meta do município.
Projeção abaixo da meta gera alerta de risco de não atingimento.

**Auditoria da projeção:** como o conjunto de teste carrega a taxa efetivamente
observada em 2024, o alerta é confrontado com o desfecho real e comparado a uma linha
de base ingênua que supõe a repetição da taxa de 2023.

| Estratégia | Acurácia | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| Projeção do modelo | 0,7026 | 0,6667 | 0,6610 | 0,6638 |
| Referência: classe majoritária | 0,5558 | 0,0000 | 0,0000 | 0,0000 |
| Referência: repetir 2023 | 0,4780 | 0,4506 | 0,7994 | 0,5764 |

**O ganho a ser citado é de +0,1468 sobre a classe majoritária**, e não a diferença
contra a baseline ingênua. Dos 797 municípios avaliados, 443 (55,6%) atingiram a meta,
de modo que apostar que todos atingem já acerta 55,6% sem modelo algum. Precisão e F1
dessa referência são zero por construção, porque ela nunca emite alerta.

A baseline ingênua é **pior que a classe majoritária**: com 0,4780 ela sobre-alerta,
tendo recall de 0,7994 e precisão de 0,4506. Sinaliza quase todo mundo e por isso quase
não informa. Citar apenas a comparação contra ela inflaria o ganho aparente, e foi
exatamente esse o erro corrigido nesta revisão.

**Escopo:** 797 dos 828 municípios do conjunto de teste; 31 foram descartados por não
terem meta publicada. Todos são municípios inéditos, ausentes do treino e da validação.

**Limitação:** a análise é associativa e não constitui previsão oficial de cumprimento
de meta. O ranking por gap absoluto concentra-se em UFs com metas mais ambiciosas, o
que está sinalizado no próprio relatório. Para priorização orçamentária, deve ser
combinado com o ranking de risco absoluto.

Resultados em [Risco de meta 2024](goal_risk_2024.md), `goal_risk_ranking.csv` e
`goal_risk_summary.json`.
