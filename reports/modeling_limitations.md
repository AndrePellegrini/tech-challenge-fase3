# Limitações da modelagem

- As features educacionais são contextuais e compartilhadas por muitos alunos;
  o modelo tem pouco sinal individual além do target.
- IDHM está em granularidade de UF, embora a unidade de predição seja o aluno.
- Há ausências estruturais na Gold; imputação preserva linhas, mas não recupera
  informação inexistente.
- Os principais indicadores educacionais são de 2023 e o target é de 2024.
- As relações descritas e as importâncias representam associação e padrão
  preditivo, não causalidade.
- O modelo não substitui avaliação ou diagnóstico pedagógico individual.
- O split por município mede um cenário conservador de generalização para
  territórios não vistos e produz variação natural no volume por conjunto.
- A ausência de variáveis individuais mais ricas limita o poder preditivo e a
  interpretação das diferenças entre estudantes do mesmo contexto.
- `peso_aluno` foi reservado fora de X; seu uso futuro como `sample_weight`
  precisa ser avaliado pelo grupo.
- A análise municipal deriva previsões individuais de validação e não representa
  estimativa oficial de cumprimento de metas.
- A validação cruzada agrupada foi executada nos quatro finalistas, não nos onze
  candidatos, por custo computacional. A dispersão entre folds mostra que Random
  Forest e regressão logística são estatisticamente indistinguíveis, de modo que a
  escolha do modelo final se apoia nos critérios de desempate e não na ROC AUC.
- A otimização de hiperparâmetros usou grade manual pequena e definida a priori,
  sem busca automatizada; não há garantia de que o ótimo esteja contido nela.
- A diferença de ROC AUC entre teste (0,6631) e validação (0,6409) é de +0,0222.
  A validação cruzada esclareceu a origem: sua média é 0,6608, a 0,0023 do teste,
  o que indica partição de validação pessimista e não partição de teste favorável.
  A melhor estimativa de generalização é a média da validação cruzada.
- A interpretabilidade por SHAP usa amostra de 8.000 linhas da validação, não a
  partição inteira. O ranking é associativo e descreve como o modelo usa as
  features, não como a alfabetização é produzida.
- A reprodução do dataset completo exige credenciais de leitura no bucket S3 privado
  da equipe. Sem elas, é possível executar a suíte de testes e toda a pipeline sobre
  a amostra anonimizada versionada, mas não regenerar as 1.851.828 linhas.
