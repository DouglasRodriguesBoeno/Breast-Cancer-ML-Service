# Mammography Training

Este diretorio documenta o treinamento futuro da V3 de mamografia. O treinamento e separado da API: scripts experimentais deverao treinar e validar modelos offline, enquanto a API carregara artefatos ja treinados apenas para inferencia.

O baseline planejado sera uma arquitetura de classificacao por transfer learning em PyTorch, usando imagens 2D pre-processadas. A definicao final de resize, normalizacao, adaptacao de canais e metricas ainda sera feita em uma etapa posterior, com experimentos controlados.

Artefatos treinados deverao ser salvos em:

```text
artifacts/mammography/<versao>
```

A API V3 devera carregar uma versao explicita desse artefato quando houver um modelo disponivel. Nesta etapa inicial, nao existe modelo clinico ou experimental carregado para predicao, e nenhuma saida atual deve ser interpretada como diagnostico.

Limites atuais:

- nao ha inferencia de mamografia;
- nao ha score, BI-RADS, heatmap ou recomendacao clinica;
- o endpoint atual apenas valida arquivo, pixels e metadados tecnicos seguros;
- imagens reais de pacientes nao devem ser adicionadas ao repositorio.

