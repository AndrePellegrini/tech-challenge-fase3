# Auditoria da linhagem do target `alfabetizado`

## Conclusão

No batch real da Fase 2, `alfabetizado` já existe na tabela
`basedosdados.br_inep_avaliacao_alfabetizacao.alunos`. A query Bronze seleciona
simultaneamente `dados.alfabetizado` e `dados.proficiencia`, decodificando apenas
o primeiro pelo `dicionario`. A Silver normaliza nomes/tipos e remove duplicatas;
não calcula o target. A Fase 3 converte `Sim`/`Não` para 1/0 sem redefinir a regra.

A documentação oficial do Inep define 743 pontos na escala de proficiência como
o corte a partir do qual a criança é considerada alfabetizada:
https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/avaliacao-da-alfabetizacao

O threshold 743 também aparece no gerador de streaming da Fase 2, mas esse código
simula eventos e não é a origem do batch oficial.

## Evidência empírica na Silver oficial

- população filtrada: 1,851,828 alunos;
- proficiência completa: 1,851,828;
- correlação com target binário: 0.798898;
- accuracy ao reconstruir por `proficiencia >= 743`: 100.000000%;
- divergências: 0;
- menor proficiência na classe 1: 743.0;
- maior proficiência na classe 0: 742.9996337890625.

Classificação final: **leakage direto**. A proficiência é produzida no mesmo
evento e operacionaliza a definição do target. Não deve entrar em X, mesmo que
eleve fortemente as métricas.

## Evidências de código

- Fase 2 `src/bronze/queries.py`: seleciona as duas colunas da fonte e faz o join
  do dicionário de `alfabetizado`;
- Fase 2 `src/silver/transform.py`: apenas normalização, deduplicação e tipos;
- Fase 2 `src/silver/catalog.py`: proficiência tem somente regra de faixa;
- Fase 3 `src/preprocessing/build_modeling_dataset.py`: converte os rótulos do
  target para binário;
- Fase 2 `src/streaming/events.py`: usa 743 apenas na simulação Kafka.
