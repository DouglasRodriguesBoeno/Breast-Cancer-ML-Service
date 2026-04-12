from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

EDUCATIONAL_WARNING = "Modelo treinado para fins educacionais. Nao usar para diagnostico clinico."
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "feature_stats.json"
MODEL_INFO_PATH = ARTIFACTS_DIR / "model_info.json"


class HealthResponse(BaseModel):
    status: str
    modelLoaded: bool
    predictReady: bool


class FeatureItem(BaseModel):
    name: str
    mean: float
    std: float
    min: float
    max: float


class FeaturesResponse(BaseModel):
    features: List[FeatureItem]


class ModelInfoResponse(BaseModel):
    model_type: str
    trained_at: str
    accuracy_test: float
    threshold_malignant: float = DEFAULT_CLASSIFICATION_THRESHOLD
    notes: str
    extra: Dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    features: Dict[str, float]

    @field_validator("features")
    @classmethod
    def non_empty_and_numeric(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("features nao pode ser vazio.")

        for k, val in v.items():
            fv = float(val)
            if not np.isfinite(fv):
                raise ValueError(f"Feature '{k}' e NaN/Inf.")

        return v


class PredictResponse(BaseModel):
    predicted_label: str
    predicted_label_name: str
    probability_malignant: float
    probability_benign: float
    used_threshold_malignant: float
    model_type: str
    risk_band: str
    summary: str
    confidence_note: str
    input_quality_note: str
    clinical_note: str
    used_features: List[str]
    ignored_features: List[str] = Field(default_factory=list)
    imputed_features: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


app = FastAPI(title="Breast Cancer Prediction API", version="1.1.0")

model: Optional[Any] = None
feature_meta: Optional[dict] = None
model_info: Optional[dict] = None


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path.as_posix())
    return json.loads(path.read_text(encoding="utf-8"))


def label_name(label: str) -> str:
    return {"M": "Maligno", "B": "Benigno"}.get(label, label)


def expected_features_order() -> List[str]:
    if not feature_meta or "features_order" not in feature_meta:
        raise RuntimeError("feature_stats.json sem 'features_order'.")
    return list(feature_meta["features_order"])


def means_map() -> Dict[str, float]:
    feats = (feature_meta or {}).get("features", [])
    return {f["name"]: f["mean"] for f in feats if "name" in f and "mean" in f}


def model_type_name() -> str:
    if isinstance(model, dict) and "logistic_model" in model and "random_forest_model" in model:
        return "ensemble_mean_logistic_random_forest"
    return str((model_info or {}).get("model_type", "unknown"))


def model_classes(m: Any) -> List[str]:
    classes = getattr(m, "classes_", None)
    if classes is not None:
        return [str(c) for c in classes]

    steps = getattr(m, "named_steps", {})
    if "classifier" in steps:
        return [str(c) for c in steps["classifier"].classes_]
    if "clf" in steps:
        return [str(c) for c in steps["clf"].classes_]

    raise RuntimeError("Nao foi possivel determinar classes do modelo.")


def _single_model_proba_for_classes(m: Any, x: np.ndarray) -> Dict[str, float]:
    if not hasattr(m, "predict_proba"):
        raise RuntimeError("Modelo nao suporta predict_proba.")

    proba = m.predict_proba(x)[0]
    classes = model_classes(m)
    return {str(c): float(p) for c, p in zip(classes, proba)}


def proba_for_classes(m: Any, x: np.ndarray) -> Dict[str, float]:
    if isinstance(m, dict) and "logistic_model" in m and "random_forest_model" in m:
        logistic_probs = _single_model_proba_for_classes(m["logistic_model"], x)
        rf_probs = _single_model_proba_for_classes(m["random_forest_model"], x)

        all_classes = sorted(set(logistic_probs.keys()) | set(rf_probs.keys()))
        return {
            cls: float((logistic_probs.get(cls, 0.0) + rf_probs.get(cls, 0.0)) / 2.0)
            for cls in all_classes
        }

    return _single_model_proba_for_classes(m, x)


def malignant_threshold() -> float:
    raw_threshold = (model_info or {}).get(
        "threshold_malignant",
        DEFAULT_CLASSIFICATION_THRESHOLD,
    )

    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        return DEFAULT_CLASSIFICATION_THRESHOLD

    if not np.isfinite(threshold):
        return DEFAULT_CLASSIFICATION_THRESHOLD
    if threshold < 0.0:
        return 0.0
    if threshold > 1.0:
        return 1.0

    return threshold


def is_predict_ready() -> bool:
    if model is None or feature_meta is None or model_info is None:
        return False

    try:
        expected = expected_features_order()
        _ = means_map()
        dummy_x = np.zeros((1, len(expected)), dtype=float)
        probs = proba_for_classes(model, dummy_x)
        return "M" in probs and "B" in probs
    except Exception:
        return False


def risk_band_from_probability(probability_malignant: float) -> str:
    if probability_malignant < 0.30:
        return "low"
    if probability_malignant < 0.70:
        return "medium"
    return "high"


def summary_from_result(predicted_label: str, probability_malignant: float) -> str:
    risk_band = risk_band_from_probability(probability_malignant)

    if predicted_label == "B":
        if risk_band == "low":
            return "Os dados informados apresentam maior compatibilidade com padrao benigno segundo o modelo treinado."
        return "Os dados informados sugerem classificacao benigna, mas com nivel de atencao moderado segundo o modelo."
    else:
        if risk_band == "high":
            return "Os dados informados apresentam maior compatibilidade com padrao maligno segundo o modelo treinado."
        return "Os dados informados sugerem classificacao maligna, mas em uma faixa de probabilidade intermediaria."


def confidence_note_from_result(probability_malignant: float, threshold: float) -> str:
    distance = abs(probability_malignant - threshold)

    if distance < 0.05:
        return "O resultado ficou muito proximo ao threshold configurado, indicando um caso de menor separacao entre as classes."
    if distance < 0.15:
        return "O resultado ficou acima da margem de decisao, com confianca moderada na classificacao preditiva."
    return "O resultado ficou bem distante do threshold configurado, indicando maior separacao probabilistica entre as classes."


def input_quality_note_from_result(
    imputed_features: List[str],
    ignored_features: List[str],
) -> str:
    imputed_count = len(imputed_features)
    ignored_count = len(ignored_features)

    if imputed_count == 0 and ignored_count == 0:
        return "Todos os campos utilizados foram fornecidos conforme esperado pelo modelo."
    if imputed_count <= 3 and ignored_count == 0:
        return "Poucos campos estavam ausentes e foram preenchidos automaticamente com medias do treino."
    if imputed_count > 3:
        return "Varios campos estavam ausentes e foram preenchidos automaticamente, o que reduz a confiabilidade operacional da resposta."
    return "Foram identificados campos extras ignorados e/ou campos ausentes preenchidos automaticamente."


def clinical_note() -> str:
    return "Resultado educacional e probabilistico. Nao substitui avaliacao medica, exames complementares ou diagnostico clinico."


@app.on_event("startup")
def startup() -> None:
    global model, feature_meta, model_info

    try:
        model = joblib.load(MODEL_PATH)
        feature_meta = read_json(FEATURES_PATH)
        model_info = read_json(MODEL_INFO_PATH)
        print(f"[startup] Model loaded from {MODEL_PATH.as_posix()}")
    except Exception as e:
        model = None
        feature_meta = None
        model_info = None
        print(f"[startup] Could not load artifacts: {e}")


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready = is_predict_ready()
    return HealthResponse(
        status="ok" if ready else "degraded",
        modelLoaded=model is not None,
        predictReady=ready,
    )


@app.get("/v1/features", response_model=FeaturesResponse)
def features() -> FeaturesResponse:
    if feature_meta is None:
        raise HTTPException(status_code=500, detail="Feature metadata nao carregado. Rode o treino.")

    items = [FeatureItem(**f) for f in feature_meta.get("features", [])]
    return FeaturesResponse(features=items)


@app.get("/v1/model-info", response_model=ModelInfoResponse)
def info() -> ModelInfoResponse:
    if model_info is None:
        raise HTTPException(status_code=500, detail="Model info nao carregado. Rode o treino.")

    return ModelInfoResponse(
        model_type=model_info.get("model_type", model_info.get("model_name", "unknown")),
        trained_at=model_info.get("trained_at", "unknown"),
        accuracy_test=float(model_info.get("accuracy_test", 0.0)),
        threshold_malignant=float(
            model_info.get("threshold_malignant", DEFAULT_CLASSIFICATION_THRESHOLD)
        ),
        notes=model_info.get("notes", ""),
        extra=model_info.get("extra", {}) or {},
    )


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not is_predict_ready():
        raise HTTPException(
            status_code=500,
            detail="Modelo/metadata nao estao prontos para inferencia. Rode o treino e gere artifacts/.",
        )

    expected = expected_features_order()
    means = means_map()

    provided = {k: float(v) for k, v in req.features.items()}
    warnings = [EDUCATIONAL_WARNING]

    ignored_features = [k for k in provided.keys() if k not in expected]
    if ignored_features:
        warnings.append(f"Features desconhecidas ignoradas: {ignored_features}")
        for k in ignored_features:
            provided.pop(k, None)

    imputed_features = [f for f in expected if f not in provided]
    if imputed_features:
        warnings.append(
            f"{len(imputed_features)} features faltantes preenchidas com mean do treino."
        )
        for f in imputed_features:
            if f not in means:
                raise HTTPException(
                    status_code=500,
                    detail=f"Sem mean para feature '{f}' no metadata.",
                )
            provided[f] = float(means[f])

    x = np.array([[provided[f] for f in expected]], dtype=float)

    probs = proba_for_classes(model, x)
    p_m = float(probs.get("M", 0.0))
    p_b = float(probs.get("B", 0.0))
    threshold_m = malignant_threshold()
    pred = "M" if p_m >= threshold_m else "B"

    risk_band = risk_band_from_probability(p_m)
    summary = summary_from_result(pred, p_m)
    confidence_note = confidence_note_from_result(p_m, threshold_m)
    input_quality_note = input_quality_note_from_result(
        imputed_features=imputed_features,
        ignored_features=ignored_features,
    )

    return PredictResponse(
        predicted_label=pred,
        predicted_label_name=label_name(pred),
        probability_malignant=p_m,
        probability_benign=p_b,
        used_threshold_malignant=threshold_m,
        model_type=model_type_name(),
        risk_band=risk_band,
        summary=summary,
        confidence_note=confidence_note,
        input_quality_note=input_quality_note,
        clinical_note=clinical_note(),
        used_features=expected,
        ignored_features=ignored_features,
        imputed_features=imputed_features,
        warnings=warnings,
    )