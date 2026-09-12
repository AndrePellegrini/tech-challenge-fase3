# POC de viabilidade — Censo Escolar 2023

## Classificação da evidência

Usar a Gold e tratar leakage são requisitos do Tech Challenge. O PDF permite
Censo Escolar como fonte externa, mas não o torna obrigatório. A seleção das dez
features, normalização da chave e esta POC são decisões metodológicas do grupo e
extensão com conhecimento externo.

## Fonte

- fonte: Inep, Microdados do Censo Escolar da Educação Básica 2023;
- arquivo: `microdados_censo_escolar_2023.zip` (30.61 MiB);
- CSV interno: 200.31 MiB;
- granularidade: uma linha por escola;
- chave: `CO_ENTIDADE`;
- escolas ativas: 180,230;
- formato: ZIP + CSV `;` + dicionário XLSX;
- acesso: download HTTP oficial, sem GCP ou cloud de processamento.

A busca no repositório atual, na Fase 2 e nos prefixos Silver/Gold do S3 não
encontrou uma tabela de atributos escolares. A Base dos Dados exigiria um projeto
de cobrança GCP, indisponível neste ambiente. Por isso, a POC usou o arquivo
oficial do Inep, que é pequeno o bastante para leitura local em blocos. Materiais
complementares da FIAP não forneceram uma tabela de correspondência de escolas.

## Compatibilidade e join

`id_escola` nos alunos tem dtype `object` e
`CO_ENTIDADE` no Censo tem dtype `int64`. A POC
normaliza explicitamente ambos como texto numérico de oito posições, sem alterar
as colunas originais.

- escolas únicas nos alunos: 42,327;
- escolas ativas no Censo: 180,230;
- escolas dos alunos encontradas: 0
  (0.00%);
- alunos cobertos: 0.00%;
- alunos sem match: 1,851,828;
- linhas antes/depois: 1,851,828/1,851,828;
- expansão e duplicidade: 0/0.

Possíveis causas de não match: escola não ativa no Censo 2023, mudança de código,
ausência/invalidade da chave ou participação na avaliação em contexto não coberto
pela edição escolar selecionada. Esses casos precisam de auditoria antes da
integração oficial.

Neste dataset, os exemplos de `id_escola` começam em códigos sequenciais `60...`,
sem qualquer interseção com `CO_ENTIDADE`. Isso evidencia anonimização ou
recodificação da chave na fonte de alunos. Sem a tabela segura de correspondência
entre o identificador anonimizado e o código Inep, o enriquecimento não é viável.

## Features escolhidas

- `TP_DEPENDENCIA` — Dependência administrativa; cobertura no join 0.00%; risco A (baixo).
- `TP_LOCALIZACAO` — Localização urbana/rural; cobertura no join 0.00%; risco A (baixo).
- `QT_MAT_FUND_AI_2` — Matrículas do ensino fundamental - anos iniciais - 2º ano; cobertura no join 0.00%; risco A (baixo).
- `IN_AGUA_POTAVEL` — Fornece água potável para consumo humano; cobertura no join 0.00%; risco A (baixo).
- `IN_ENERGIA_REDE_PUBLICA` — Abastecimento de energia elétrica por rede pública; cobertura no join 0.00%; risco A (baixo).
- `IN_ESGOTO_REDE_PUBLICA` — Esgoto sanitário por rede pública; cobertura no join 0.00%; risco A (baixo).
- `IN_INTERNET` — Acesso à internet; cobertura no join 0.00%; risco A (baixo).
- `IN_BANDA_LARGA` — Internet banda larga; cobertura no join 0.00%; risco A (baixo).
- `IN_BIBLIOTECA_SALA_LEITURA` — Biblioteca e/ou sala de leitura; cobertura no join 0.00%; risco A (baixo).
- `QT_DOC_FUND_AI` — Docentes do ensino fundamental - anos iniciais; cobertura no join 0.00%; risco A (baixo).

Todas são administrativas de 2023, anteriores ao target 2024 e não usam
alfabetização, proficiência ou resultado de prova. Quantidades recebem imputação
somente em eventual pipeline futuro; nenhuma imputação foi feita nesta POC.

## Quebra dos perfis

O X oficial tem 6,430 perfis. O X experimental tem
6,430, aumento de 0
(0.00%). A média cai de
288.00 para 288.00
alunos por perfil; a mediana muda de 93
para 93.

Alunos em perfis mistos mudam de 99.82%
para 99.82%; perfis mistos, de
98.76% para 98.76%.
Essa quebra mede granularidade, não ganho preditivo.

## Variabilidade por escola

- escolas na população: 42,327;
- escolas com ambas as classes: 96.05%;
- variação entre escolas: 16.76%;
- variação dentro das escolas: 83.24%;
- comparação município+rede: 10.17%
  entre contextos e 89.83% dentro.

Descer para escola explica parcela adicional da heterogeneidade, mas a maior
parte ainda permanece dentro da escola. As associações das features do Censo
ficaram indisponíveis porque a cobertura do join foi zero. Os gráficos registram
explicitamente essa ausência; não há estimativa ou interpretação causal.

## Custo e decisão da POC

Esforço estimado: **médio**. O download é pequeno, o CSV tem cerca de 200 MiB,
o join é many-to-one e não requer GCP. O trabalho restante envolve armazenar a
fonte de modo reproduzível, parametrizar o reader, mapear categorias, definir
missing, validar cobertura, adicionar testes e versionar documentação.

Viabilidade: **NÃO RECOMENDADO**.

Recomendação técnica: não incorporar ao modelo oficial com as chaves atuais.
Somente retomar uma versão experimental se o responsável pela anonimização
fornecer uma tabela de correspondência auditável entre `id_escola` e
`CO_ENTIDADE`. Não foram treinados modelos, portanto não há evidência de ganho em
ROC AUC. A obtenção desse mapa e a autorização de uso ficam **REVISAR COM O GRUPO**.
