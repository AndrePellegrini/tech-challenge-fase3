# Auditoria de candidatas a features

## Silver alunos

- `serie`: cardinalidade 1; constante=True; 2° ano do Ensino Fundamental (1,851,828).
- `caderno`: cardinalidade 22; constante=False; 1 (97,138), 3 (93,366), 5 (93,351), 2 (93,301), 4 (93,242), 7 (93,064), 6 (93,049), 15 (86,462), 8 (86,290), 20 (85,437), 10 (85,391), 9 (85,313), 12 (85,278), 13 (85,242), 16 (85,207), 11 (85,200), 17 (85,194), 21 (85,160), 18 (85,106), 19 (85,054), 14 (84,971), 43 (12).
- `presenca`: cardinalidade 1; constante=True; Presente (1,851,828).
- `preenchimento_caderno`: cardinalidade 1; constante=True; Prova preenchida (1,851,828).

`presenca` e `preenchimento_caderno` são critérios do filtro e ficam constantes.
`serie` deve ser descartada por ser constante. O projeto não documenta o
significado detalhado dos 22 códigos de `caderno`; a hipótese de versão/formulário
é plausível, mas permanece **evidência insuficiente**. Como o código nasce no
evento de aplicação e pode capturar artefatos técnicos, não é recomendado.

O inventário completo e sua justificativa estão no JSON. Metadados iniciados por
`_` são técnicos. IDs não entram crus em X. `peso_aluno` permanece candidato
apenas a `sample_weight`.

## Fontes existentes

Fase 2/S3 contém `alunos`, `alunos_integrado`, município, UF, metas municipais,
estaduais/nacionais, `atlas_uf` e as quatro tabelas Gold. Não foi encontrada
tabela com atributos de escola, turma ou docente. `id_escola` existe e permite
um join futuro, mas o projeto atual não fornece os atributos.

## Tech Challenge e fonte externa

O PDF oficial exige Gold da Fase 2 e um modelo supervisionado com variáveis
educacionais, territoriais e socioeconômicas. Ele **permite**, sem obrigar,
enriquecimento com IBGE, Censo Escolar, FUNDEB, Atlas, PNAD, Cadastro Único e
fontes socioeconômicas regionais.

Como extensão externa, o Censo Escolar 2023 é a opção mais compatível. O Inep
publica microdados anuais e coleta informações de estabelecimentos, gestores,
turmas, alunos e profissionais escolares:
https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar

Antes de integrar, devem ser confirmados no dicionário de 2023: compatibilidade
do código da escola, cobertura dos `id_escola`, granularidade e disponibilidade
temporal. Candidatas plausíveis são dependência administrativa, localização
urbana/rural, porte por matrículas e infraestrutura. Resultados de 2024,
proficiência, preenchimento da prova e qualquer variável derivada do resultado
são alto risco.

## Conteúdo curricular disponível

Não foram encontrados materiais curriculares separados no repositório. A única
fonte FIAP disponível é o PDF oficial do desafio, que exige tratamento de data
leakage, engenharia de atributos, transformação, separação entre treino,
validação e teste, replicabilidade e generalização. A matriz temporal e esta
auditoria de linhagem são metodologia do grupo/extensão, não conteúdo específico
atribuído à FIAP.

## Cenários

1. **Manter modelo contextual:** menor esforço e melhor alinhamento às perguntas
   territoriais; mantém a limitação individual e desempenho próximo do atual.
2. **Enriquecer com escola:** testar Censo Escolar 2023 por `id_escola`; esforço
   médio/alto, cobertura ainda a medir e potencial para diferenciar escolas do
   mesmo município.
3. **Features individuais legítimas:** nenhuma fonte pré-avaliação foi encontrada
   no projeto. Investigar somente se existir fonte autorizada e temporalmente
   anterior; não usar proficiência, caderno ou resultado da própria avaliação.

Nenhum cenário foi escolhido pelo grupo. Cenários 2 e 3 permanecem **REVISAR COM
O GRUPO**. A matriz operacional está em `feature_candidate_matrix.csv`.
