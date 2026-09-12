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
- A seleção do melhor baseline é provisória. Teste, tuning, calibração e escolha
  de limiar continuam reservados para etapas posteriores.
- Random Forest e SHAP não foram concluídos por custo local. Reavaliar ambos é
  uma decisão complementar, não condição para validar estes baselines.
