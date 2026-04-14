# Breast Cancer LLM

API educacional para inferencia de risco em cancer de mama usando o dataset Wisconsin Diagnostic Breast Cancer (WDBC), com pipeline de treino em Python e servico HTTP em FastAPI.

O projeto foi construido para estudo de Machine Learning aplicado a saude, com foco em:

- pipeline de treino reproduzivel
- comparacao entre modelos
- ajuste de threshold para reduzir falso negativo
- artefatos versionaveis para servir inferencia
- respostas explicativas na API

Importante: este projeto e estritamente educacional. Nao deve ser usado para diagnostico clinico.

## Visao Geral

Na V1, o fluxo principal faz:

1. carrega o dataset WDBC local ou fallback do `scikit-learn`
2. separa conjunto de desenvolvimento e teste final untouched
3. compara modelos base em cross-validation
4. faz tuning de hiperparametros para regressao logistica
5. combina regressao logistica + random forest por media de probabilidades
6. ajusta o `threshold_malignant` no conjunto de desenvolvimento
7. salva artefatos para inferencia na API

O servico exposto em FastAPI:

- valida a entrada
- ignora features extras
- imputa features faltantes com medias do treino
- retorna probabilidades, classificacao, faixa de risco e avisos de uso

## Estrutura

```text
.
|-- README.md
|-- requirements.txt
`-- ml-service
    |-- app
    |   `-- main.py
    |-- artifacts
    |   |-- cv_results.json
    |   |-- error_analysis_test.csv
    |   |-- feature_metadata.json
    |   |-- feature_stats.json
    |   |-- model.joblib
    |   |-- model_info.json
    |   `-- results_summary.json
    |-- data
    |   |-- wdbc.data
    |   `-- wdbc.names
    `-- training
        `-- train_model.py
```

## Stack

- Python
- FastAPI
- Pydantic v2
- scikit-learn
- pandas
- numpy
- joblib

## Requisitos

- Python 3.11+
- `pip`

## Instalacao

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se preferir manter o ambiente virtual dentro de `ml-service`, ajuste os comandos de ativacao de acordo com o local escolhido.

## Como Treinar

Treino usando o dataset local em `ml-service/data/wdbc.data`:

```powershell
python ml-service/training/train_model.py --data-path ml-service/data/wdbc.data --output-dir ml-service/artifacts
```

Treino usando o fallback do `scikit-learn`:

```powershell
python ml-service/training/train_model.py --output-dir ml-service/artifacts
```

Parametros uteis:

- `--output-dir`: pasta onde os artefatos serao salvos
- `--data-path`: caminho do dataset UCI/WDBC
- `--random-state`: semente para reprodutibilidade
- `--n-estimators`: numero de arvores da random forest
- `--n-splits`: numero de folds da cross-validation

## Artefatos Gerados

O treino gera principalmente:

- `model.joblib`: modelo oficial da V1
- `model_info.json`: metadados do modelo, threshold e metricas
- `feature_stats.json`: ordem das features e estatisticas do treino
- `cv_results.json`: comparacao de modelos em cross-validation
- `error_analysis_test.csv`: analise de erros no teste final

## Resultado Atual da V1

Com base no artefato atual `ml-service/artifacts/model_info.json`:

- modelo oficial: `ensemble_mean_logistic_random_forest`
- `accuracy_test`: `0.9912`
- `threshold_malignant`: `0.3433`
- `ROC-AUC(M)`: `0.9977`
- matriz de confusao no teste final: `[[72, 0], [1, 41]]`

Interpretacao rapida:

- 72 benignos classificados corretamente
- 41 malignos classificados corretamente
- 1 maligno ficou como falso negativo
- 0 falsos positivos no teste final atual

## Como Subir a API

Com os artefatos ja gerados em `ml-service/artifacts`:

```powershell
$env:ARTIFACTS_DIR="ml-service/artifacts"
uvicorn ml-service.app.main:app --reload
```

Se estiver dentro de `ml-service`, voce tambem pode usar:

```powershell
$env:ARTIFACTS_DIR="artifacts"
uvicorn app.main:app --reload
```

API padrao:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Endpoints

### `GET /v1/health`

Retorna status da API e se a inferencia esta pronta.

### `GET /v1/features`

Retorna as features esperadas pelo modelo com estatisticas do treino.

### `GET /v1/model-info`

Retorna tipo de modelo, data de treino, accuracy de teste, threshold e metadados extras.

### `POST /v1/predict`

Recebe um dicionario de features numericas e retorna:

- label prevista
- probabilidade de maligno e benigno
- threshold usado
- modelo utilizado
- faixa de risco
- resumo textual
- observacoes sobre confianca e qualidade do input

## Exemplo de Requisicao

```json
{
  "features": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }
}
```

Exemplo com `curl`:

```powershell
curl -X POST "http://127.0.0.1:8000/v1/predict" `
  -H "Content-Type: application/json" `
  -d "{\"features\":{\"radius_mean\":17.99,\"texture_mean\":10.38,\"perimeter_mean\":122.8,\"area_mean\":1001.0,\"smoothness_mean\":0.1184,\"compactness_mean\":0.2776,\"concavity_mean\":0.3001,\"concave_points_mean\":0.1471,\"symmetry_mean\":0.2419,\"fractal_dimension_mean\":0.07871,\"radius_se\":1.095,\"texture_se\":0.9053,\"perimeter_se\":8.589,\"area_se\":153.4,\"smoothness_se\":0.006399,\"compactness_se\":0.04904,\"concavity_se\":0.05373,\"concave_points_se\":0.01587,\"symmetry_se\":0.03003,\"fractal_dimension_se\":0.006193,\"radius_worst\":25.38,\"texture_worst\":17.33,\"perimeter_worst\":184.6,\"area_worst\":2019.0,\"smoothness_worst\":0.1622,\"compactness_worst\":0.6656,\"concavity_worst\":0.7119,\"concave_points_worst\":0.2654,\"symmetry_worst\":0.4601,\"fractal_dimension_worst\":0.1189}}"
```

## Observacoes de Engenharia

- O modelo servido atualmente e um ensemble simples por media de probabilidades.
- O threshold final e ajustado no conjunto de desenvolvimento, sem usar o teste final para decisao.
- O endpoint de predicao foi enriquecido com explicacoes textuais para tornar a resposta mais interpretavel na V1.
- O projeto mantem um aviso explicito de uso educacional em toda a camada de inferencia.

## Limites da V1

- dataset pequeno e academico
- sem autenticacao
- sem testes automatizados formais
- sem containerizacao
- sem pipeline CI/CD
- sem rastreamento de experimentos
- sem calibracao probabilistica dedicada

## Proximos Passos Sugeridos

- adicionar testes automatizados para treino e API
- criar `Dockerfile` e `docker-compose`
- separar dependencias de runtime e de treinamento
- versionar artefatos por experimento
- incluir metricas adicionais e dashboards simples
- adicionar validacao de schema para payloads de entrada de exemplos

## Aviso

Este repositorio tem objetivo de estudo e demonstracao tecnica. Nao deve ser usado para suporte a diagnostico, triagem clinica ou decisao medica real.
