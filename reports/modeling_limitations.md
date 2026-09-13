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
- Não foi executada validação cruzada agrupada. Com holdout único não há
  estimativa de variância entre folds, e diferenças de ROC AUC na casa de 0,002
  entre candidatos próximos não são estatisticamente distinguíveis.
- A otimização de hiperparâmetros usou grade manual pequena e definida a priori,
  sem busca automatizada; não há garantia de que o ótimo esteja contido nela.
- A diferença de ROC AUC entre teste (0,6631) e validação (0,6409) é de +0,0222,
  classificada como moderada pelo próprio protocolo. O teste ficou acima da
  validação, o que afasta overfitting de seleção, mas indica que a partição de
  teste é marginalmente mais favorável — a estimativa de generalização deve ser
  lida com essa ressalva.
- SHAP não foi executado. Depende do modelo serializado e do parquet de
  modelagem, nenhum dos dois versionado. Permanece como pendência explícita.
- A reprodução completa da pipeline exige credenciais de leitura no bucket S3
  privado da equipe. Sem elas, um avaliador externo consegue executar a suíte de
  testes, mas não regenerar o dataset de modelagem.
