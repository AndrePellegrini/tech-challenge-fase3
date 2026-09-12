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

**Status:** Baselines executados.

**Classificação:** decisão metodológica do grupo.

Dummy, regressão logística e árvore de decisão formam a comparação progressiva.
Random Forest foi tentado e interrompido por custo local desproporcional. Não foi
adicionada dependência de boosting.

---

## DEC-010 — Estratégia de validação cruzada

**Status:** Pendente

A definir considerando estrutura e distribuição dos dados.

---

## DEC-011 — Estratégia de otimização de hiperparâmetros

**Status:** Pendente

A definir após avaliação inicial dos modelos candidatos.

---

## DEC-012 — Seleção do modelo final

**Status:** Melhor baseline provisório definido; modelo final pendente.

**Classificação:** decisão metodológica do grupo — REVISAR COM O GRUPO.

A árvore de decisão obteve a maior ROC AUC de validação (0,6214), mas mostrou
gap de 0,0603 em relação ao treino. Ela serve à análise inicial e não é declarada
modelo final antes de tuning controlado e avaliação única do teste.

---

## DEC-013 — Estratégia de interpretabilidade

**Status:** Feature Importance inicial concluída; SHAP adiado.

**Classificação:** curricular complementar/recomendado e decisão de execução do grupo.

Foi usada importância nativa da árvore, agregada às features de origem. SHAP não
foi executado por custo e dependência adicional. Nenhuma importância recebe
interpretação causal.

---

## DEC-014 — Análises complementares

**Status:** Pendente

Será avaliada a viabilidade e utilidade de análises complementares, como:

* agrupamento de municípios ou regiões;
* identificação de perfis semelhantes;
* análise temporal;
* avaliação de risco de não atingimento de metas futuras.

A utilização dessas técnicas dependerá da estrutura e qualidade dos dados disponíveis.

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
