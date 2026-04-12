from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

WDBC_FEATURES: List[str] = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


@dataclass
class FeatureStats:
    name: str
    min: float
    max: float
    mean: float
    std: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_dataset_from_uci_wdbc(path: Path) -> Tuple[pd.DataFrame, pd.Series, str]:
    df = pd.read_csv(path, header=None)

    if df.shape[1] == 32:
        df.columns = ["id", "diagnosis"] + WDBC_FEATURES
        y = df["diagnosis"].astype(str).str.strip().str.upper().str[0]
        X = df[WDBC_FEATURES].apply(pd.to_numeric, errors="raise")
    elif df.shape[1] == 31:
        df.columns = ["diagnosis"] + WDBC_FEATURES
        y = df["diagnosis"].astype(str).str.strip().str.upper().str[0]
        X = df[WDBC_FEATURES].apply(pd.to_numeric, errors="raise")
    else:
        raise ValueError(f"Formato inesperado do wdbc.data: {df.shape[1]} colunas.")

    y = y.replace({"BENIGN": "B", "MALIGNANT": "M"})
    if not set(y.unique()).issubset({"B", "M"}):
        raise ValueError(f"Labels inesperadas em diagnosis: {set(y.unique())}")

    return X, y, f"uci_wdbc_file:{path.as_posix()}"


def _sklearn_name_to_wdbc(col: str) -> str:
    c = col.strip().lower()

    if c.startswith("mean "):
        base = c.replace("mean ", "", 1)
        suffix = "mean"
    elif c.startswith("worst "):
        base = c.replace("worst ", "", 1)
        suffix = "worst"
    elif c.endswith(" error"):
        base = c[: -len(" error")]
        suffix = "se"
    else:
        base = c
        suffix = ""

    base = base.replace(" ", "_")
    if suffix:
        return f"{base}_{suffix}"
    return base


def load_dataset_fallback_sklearn() -> Tuple[pd.DataFrame, pd.Series, str]:
    from sklearn.datasets import load_breast_cancer

    X_raw, y_raw = load_breast_cancer(as_frame=True, return_X_y=True)

    assert isinstance(X_raw, pd.DataFrame), f"Esperava DataFrame, veio {type(X_raw)}"

    X_raw = X_raw.copy()
    X_raw.columns = [_sklearn_name_to_wdbc(c) for c in X_raw.columns]

    missing = [f for f in WDBC_FEATURES if f not in X_raw.columns]
    if missing:
        raise RuntimeError(f"Missing expected features after rename: {missing}")

    X = X_raw[WDBC_FEATURES]
    y_arr = np.asarray(y_raw)
    y = pd.Series(np.where(y_arr == 0, "M", "B"))

    return X, y, "sklearn.datasets.load_breast_cancer"


def build_model(model_name: str, random_state: int, n_estimators: int, model_params: dict | None = None):
    
    model_params = model_params or {}

    if model_name == "dummy":
        return Pipeline(
            [
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=random_state,
                        class_weight=model_params.get("class_weight", "balanced"),
                        C=float(model_params.get("C", 1.0)),
                        solver=model_params.get("solver", "lbfgs"),
                        max_iter=int(model_params.get("max_iter", 1000)),
                    ),
                ),
            ]
        )

    raise ValueError(f"Modelo nao suportado: {model_name}")

def build_error_analysis_dataframe(
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tuned_threshold: float,
        default_threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
    ) -> pd.DataFrame: 
        malignant_scores = malignant_probability_scores(model, X_test)

        pred_default = labels_from_malignant_scores(malignant_scores, default_threshold)
        pred_tuned = labels_from_malignant_scores(malignant_scores, tuned_threshold)
        
        df = X_test.copy()
        df["y_true"] = y_test.values
        df["malignant_score"] = malignant_scores
        df["pred_default"] = pred_default
        df["pred_tuned"] = pred_tuned

        def case_type(y_true: str, y_pred: str) -> str:
            if y_true == "M" and y_pred == "M":
                return "TP"
            if y_true == "B" and y_pred == "B":
                return "TN"
            if y_true == "B" and y_pred == "M":
                return "FP"
            return "FN"
        df["case_default"] = [
            case_type(y_true, y_pred)
            for y_true, y_pred in zip(df["y_true"], df["pred_default"])
        ]

        df["case_tuned"] = [
            case_type(y_true, y_pred)
            for y_true, y_pred in zip(df["y_true"], df["pred_tuned"])
        ]

        df["distance_to_default_threshold"] = df["malignant_score"] - default_threshold
        df["distance_to_tuned_threshold"] = df["malignant_score"] - tuned_threshold

        return df.sort_values("malignant_score", ascending=False)

def compute_feature_stats(X_train: pd.DataFrame) -> dict:
    features: List[FeatureStats] = []
    for name in WDBC_FEATURES:
        s = pd.to_numeric(X_train[name], errors="raise")
        features.append(
            FeatureStats(
                name=name,
                min=float(s.min()),
                max=float(s.max()),
                mean=float(s.mean()),
                std=float(s.std(ddof=0)),
            )
        )

    return {
        "computed_at": utc_now_iso(),
        "computed_on": "train_split",
        "features_order": WDBC_FEATURES,
        "features": [asdict(f) for f in features],
    }


def malignant_probability_scores(model, X_data: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X_data)
    classes = list(getattr(model, "classes_", model.named_steps["classifier"].classes_))

    if "M" not in classes:
        raise RuntimeError(f"Classe 'M' ausente nas classes do modelo: {classes}")

    idx_m = classes.index("M")
    return np.asarray(proba[:, idx_m], dtype=float)


def labels_from_malignant_scores(malignant_scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(malignant_scores >= threshold, "M", "B")


def evaluate_predictions(y_true: pd.Series, malignant_scores: np.ndarray, threshold: float) -> dict:
    y_pred = labels_from_malignant_scores(malignant_scores, threshold)

    acc = float(accuracy_score(y_true, y_pred))
    report_text = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f2_m = float(fbeta_score(y_true, y_pred, beta=2, pos_label="M"))

    y_true_bin = (y_true == "M").astype(int)
    roc_auc = float(roc_auc_score(y_true_bin, malignant_scores))

    cm = confusion_matrix(y_true, y_pred, labels=["B", "M"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    precision_m = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_m = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_b = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "y_pred": y_pred,
        "accuracy": acc,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "precision_malignant": precision_m,
        "recall_malignant": recall_m,
        "fnr_malignant": fnr,
        "specificity_benign": specificity_b,
        "f2_malignant": f2_m,
        "roc_auc_malignant": roc_auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_labels": ["B", "M"],
    }


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> dict:
    malignant_scores = malignant_probability_scores(model, X_test)
    return evaluate_predictions(y_test, malignant_scores, threshold)

def train_evaluate_candidate(
    model_name: str,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_test_final: pd.DataFrame,
    y_test_final: pd.Series,
    random_state: int,
    n_estimators: int,
    n_splits: int,
    model_params: dict | None = None, 
) -> dict: 
    threshold_tuning = tune_decision_threshold(
        X=X_dev,
        y=y_dev,
        model_name=model_name,
        random_state=random_state,
        n_estimators=n_estimators,
        n_splits=n_splits,
        model_params=model_params,
    )
    selected_threshold = float(threshold_tuning["selected_threshold"])

    model = build_model(
        model_name,
        random_state=random_state,
        n_estimators=n_estimators,
        model_params=model_params,
    )
    model.fit(X_dev, y_dev)

    final_metrics_default = evaluate_model(
        model,
        X_test_final,
        y_test_final,
        threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
    )

    final_metrics_tuned = evaluate_model(
        model,
        X_test_final,
        y_test_final,
        threshold=selected_threshold,
    )

    error_analysis_df = build_error_analysis_dataframe(
        model=model,
        X_test=X_test_final,
        y_test=y_test_final,
        tuned_threshold=selected_threshold,
    )

    return {
        "model_name": model_name,
        "model_params": model_params or {},
        "model": model,
        "threshold_tuning": threshold_tuning,
        "selected_threshold": selected_threshold,
        "final_metrics_default": final_metrics_default,
        "final_metrics_tuned": final_metrics_tuned,
        "error_analysis_df": error_analysis_df,
    }

def compact_metrics(metrics: dict) -> dict:
    return {
        "threshold": float(metrics["threshold"]),
        "accuracy": float(metrics["accuracy"]),
        "precision_malignant": float(metrics["precision_malignant"]),
        "recall_malignant": float(metrics["recall_malignant"]),
        "fnr_malignant": float(metrics["fnr_malignant"]),
        "specificity_benign": float(metrics["specificity_benign"]),
        "f2_malignant": float(metrics["f2_malignant"]),
        "roc_auc_malignant": float(metrics["roc_auc_malignant"]),
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_labels": metrics["confusion_matrix_labels"],
        "classification_report_dict": metrics["classification_report_dict"],
    }

def candidate_thresholds(malignant_scores: np.ndarray) -> np.ndarray:
    return np.unique(
        np.concatenate(
            (
                np.asarray([0.0, DEFAULT_CLASSIFICATION_THRESHOLD, 1.0], dtype=float),
                np.asarray(malignant_scores, dtype=float),
            )
        )
    )

def tune_decision_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    random_state: int,
    n_estimators: int,
    n_splits: int,
    model_params: dict | None = None,
) -> dict:
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_malignant_scores = np.zeros(len(X), dtype=float)

    for train_idx, valid_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]

        model = build_model(
            model_name,
            random_state=random_state,
            n_estimators=n_estimators,
            model_params=model_params,
        )
        model.fit(X_train, y_train)
        oof_malignant_scores[valid_idx] = malignant_probability_scores(model, X_valid)

    default_metrics = evaluate_predictions(y, oof_malignant_scores, DEFAULT_CLASSIFICATION_THRESHOLD)
    best_metrics = default_metrics
    best_key = (
        default_metrics["f2_malignant"],
        default_metrics["recall_malignant"],
        default_metrics["accuracy"],
        default_metrics["precision_malignant"],
        default_metrics["specificity_benign"],
        default_metrics["threshold"],
    )

    for threshold in candidate_thresholds(oof_malignant_scores):
        candidate_metrics = evaluate_predictions(y, oof_malignant_scores, float(threshold))
        candidate_key = (
            candidate_metrics["f2_malignant"],
            candidate_metrics["recall_malignant"],
            candidate_metrics["accuracy"],
            candidate_metrics["precision_malignant"],
            candidate_metrics["specificity_benign"],
            candidate_metrics["threshold"],
        )
        if candidate_key > best_key:
            best_metrics = candidate_metrics
            best_key = candidate_key

    return {
        "model_name": model_name,
        "model_params": model_params or {},
        "strategy": "maximize_f2_on_dev_oof",
        "default_threshold": DEFAULT_CLASSIFICATION_THRESHOLD,
        "selected_threshold": float(best_metrics["threshold"]),
        "oof_default_metrics": compact_metrics(default_metrics),
        "oof_tuned_metrics": compact_metrics(best_metrics),
    }

def logistic_hyperparameter_grid() -> List[dict]:
    return [
        {"C": 0.01, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000},
        {"C": 0.1, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000},
        {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000},
        {"C": 10.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000},
        {"C": 100.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000},
        {"C": 0.1, "class_weight": None, "solver": "lbfgs", "max_iter": 1000},
        {"C": 1.0, "class_weight": None, "solver": "lbfgs", "max_iter": 1000},
        {"C": 10.0, "class_weight": None, "solver": "lbfgs", "max_iter": 1000},
    ]

def select_best_logistic_candidate(results: List[dict]) -> dict:
    return max(
        results,
        key=lambda r: (
            r["oof_tuned_metrics"]["recall_malignant"],
            r["oof_tuned_metrics"]["f2_malignant"],
            -r["oof_tuned_metrics"]["fnr_malignant"],
            r["oof_tuned_metrics"]["accuracy"],
            r["oof_tuned_metrics"]["precision_malignant"],
            r["oof_tuned_metrics"]["specificity_benign"],
        ),
    )

def tune_logistic_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_estimators: int,
    n_splits: int,
) -> dict:
    candidate_results = []

    for model_params in logistic_hyperparameter_grid():
        threshold_tuning = tune_decision_threshold(
            X=X,
            y=y,
            model_name="logistic_regression",
            random_state=random_state,
            n_estimators=n_estimators,
            n_splits=n_splits,
            model_params=model_params,
        )

        candidate_results.append(
            {
                "model_name": "logistic_regression",
                "model_params": model_params,
                "selected_threshold": threshold_tuning["selected_threshold"],
                "oof_default_metrics": threshold_tuning["oof_default_metrics"],
                "oof_tuned_metrics": threshold_tuning["oof_tuned_metrics"],
                "threshold_tuning": threshold_tuning,
            }
        )

    best_candidate = select_best_logistic_candidate(candidate_results)

    return {
        "strategy": "logistic_grid_plus_threshold_tuning_on_dev_oof",
        "candidates": candidate_results,
        "best_candidate": best_candidate,
    }

def select_best_cv_result(results: List[dict]) -> dict:
    return max(
        results,
        key=lambda r: (
            r["recall_malignant_mean"],
            r["f2_malignant_mean"],
            r["roc_auc_malignant_mean"],
            r["accuracy_mean"],
        ),
    )

def summarize_cv_results(model_name: str, fold_results: List[dict]) -> dict:
    metric_names = [
        "accuracy",
        "recall_malignant",
        "fnr_malignant",
        "f2_malignant",
        "roc_auc_malignant",
    ]

    summary = {
        "model_name": model_name,
        "n_folds": len(fold_results),
        "fold_results": fold_results,
    }

    for metric in metric_names:
        values = [fold[metric] for fold in fold_results]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=0))

    return summary

def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: List[str],
    random_state: int,
    n_estimators: int,
    n_splits: int = 5,
) -> List[dict]:
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    cv_results = []

    for model_name in model_names:
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model = build_model(
                model_name,
                random_state=random_state,
                n_estimators=n_estimators,
            )
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)
            fold_results.append(
                {
                    "fold": fold_idx,
                    "accuracy": metrics["accuracy"],
                    "recall_malignant": metrics["recall_malignant"],
                    "fnr_malignant": metrics["fnr_malignant"],
                    "f2_malignant": metrics["f2_malignant"],
                    "roc_auc_malignant": metrics["roc_auc_malignant"],
                }
            )

        summary = summarize_cv_results(model_name, fold_results)
        cv_results.append(summary)

    return cv_results

def save_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_cross_validation_summary(best_result: dict, n_splits: int) -> dict:
    return {
        "n_splits": n_splits,
        "cv_best_model_name": best_result["model_name"],
        "accuracy_mean": best_result["accuracy_mean"],
        "accuracy_std": best_result["accuracy_std"],
        "recall_malignant_mean": best_result["recall_malignant_mean"],
        "recall_malignant_std": best_result["recall_malignant_std"],
        "fnr_malignant_mean": best_result["fnr_malignant_mean"],
        "fnr_malignant_std": best_result["fnr_malignant_std"],
        "f2_malignant_mean": best_result["f2_malignant_mean"],
        "f2_malignant_std": best_result["f2_malignant_std"],
        "roc_auc_malignant_mean": best_result["roc_auc_malignant_mean"],
        "roc_auc_malignant_std": best_result["roc_auc_malignant_std"],
    }


def print_candidate_report(title: str, candidate: dict) -> None:
    oof_default = candidate["threshold_tuning"]["oof_default_metrics"]
    oof_tuned = candidate["threshold_tuning"]["oof_tuned_metrics"]
    test_default = candidate["final_metrics_default"]
    test_tuned = candidate["final_metrics_tuned"]

    print(f"\n=== {title} ===")
    print(
        f"Model: {candidate['model_name']} | "
        f"params: {candidate['model_params']} | "
        f"threshold: {candidate['selected_threshold']:.4f}"
    )
    print(
        f"Dev OOF | Recall(M): {oof_default['recall_malignant']:.4f} -> {oof_tuned['recall_malignant']:.4f} | "
        f"FNR(M): {oof_default['fnr_malignant']:.4f} -> {oof_tuned['fnr_malignant']:.4f} | "
        f"F2(M): {oof_default['f2_malignant']:.4f} -> {oof_tuned['f2_malignant']:.4f}"
    )
    print(
        f"Test     | Default Recall(M): {test_default['recall_malignant']:.4f} | "
        f"Tuned Recall(M): {test_tuned['recall_malignant']:.4f} | "
        f"Tuned FNR(M): {test_tuned['fnr_malignant']:.4f} | "
        f"Tuned F2(M): {test_tuned['f2_malignant']:.4f} | "
        f"Tuned Accuracy: {test_tuned['accuracy']:.4f} | "
        f"Tuned ROC-AUC(M): {test_tuned['roc_auc_malignant']:.4f} | "
        f"CM: {test_tuned['confusion_matrix']}"
    )

    fn_df = candidate["error_analysis_df"][
        candidate["error_analysis_df"]["case_tuned"] == "FN"
    ][
        [
            "y_true",
            "malignant_score",
            "pred_default",
            "pred_tuned",
            "distance_to_tuned_threshold",
        ]
    ]

    print("\nRemaining false negatives:")
    print(fn_df if not fn_df.empty else "None")


def print_model_comparison(logistic_candidate: dict, random_forest_candidate: dict) -> None:
    print("\n=== Logistic vs Random Forest (tuned threshold) ===")

    for candidate in [logistic_candidate, random_forest_candidate]:
        tuned = candidate["final_metrics_tuned"]
        print(
            f"{candidate['model_name']} | "
            f"threshold={candidate['selected_threshold']:.4f} | "
            f"Recall(M): {tuned['recall_malignant']:.4f} | "
            f"FNR(M): {tuned['fnr_malignant']:.4f} | "
            f"F2(M): {tuned['f2_malignant']:.4f} | "
            f"Accuracy: {tuned['accuracy']:.4f} | "
            f"ROC-AUC(M): {tuned['roc_auc_malignant']:.4f} | "
            f"CM: {tuned['confusion_matrix']}"
        )


def print_remaining_fn_comparison(reference_candidate: dict, comparison_candidate: dict) -> None:
    reference_df = reference_candidate["error_analysis_df"]
    comparison_df = comparison_candidate["error_analysis_df"]

    fn_indices = reference_df[reference_df["case_tuned"] == "FN"].index.tolist()

    print("\n=== Remaining FN from logistic vs random forest ===")
    if not fn_indices:
        print("Nenhum falso negativo restante na logistic tuned.")
        return

    comparison_cols = [
        "y_true",
        "malignant_score",
        "pred_default",
        "pred_tuned",
        "case_tuned",
        "distance_to_tuned_threshold",
    ]

    print("\nLogistic:")
    print(reference_df.loc[fn_indices, comparison_cols])

    print("\nRandom forest:")
    print(comparison_df.loc[fn_indices, comparison_cols])

def build_oof_malignant_scores(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    random_state: int,
    n_estimators: int,
    n_splits: int,
    model_params: dict | None = None,
) -> np.ndarray:
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_scores = np.zeros(len(X), dtype=float)

    for train_idx, valid_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]

        model = build_model(
            model_name,
            random_state=random_state,
            n_estimators=n_estimators,
            model_params=model_params,
        )
        model.fit(X_train, y_train)
        oof_scores[valid_idx] = malignant_probability_scores(model, X_valid)

    return oof_scores

def tune_ensemble_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_estimators: int,
    n_splits: int,
    logistic_params: dict,
) -> dict:
    logistic_oof = build_oof_malignant_scores(
        X=X,
        y=y,
        model_name="logistic_regression",
        random_state=random_state,
        n_estimators=n_estimators,
        n_splits=n_splits,
        model_params=logistic_params,
    )

    rf_oof = build_oof_malignant_scores(
        X=X,
        y=y,
        model_name="random_forest",
        random_state=random_state,
        n_estimators=n_estimators,
        n_splits=n_splits,
        model_params=None,
    )

    ensemble_oof = (logistic_oof + rf_oof) / 2.0

    default_metrics = evaluate_predictions(y, ensemble_oof, DEFAULT_CLASSIFICATION_THRESHOLD)
    best_metrics = default_metrics
    best_key = (
        default_metrics["f2_malignant"],
        default_metrics["recall_malignant"],
        default_metrics["accuracy"],
        default_metrics["precision_malignant"],
        default_metrics["specificity_benign"],
        default_metrics["threshold"],
    )

    for threshold in candidate_thresholds(ensemble_oof):
        candidate_metrics = evaluate_predictions(y, ensemble_oof, float(threshold))
        candidate_key = (
            candidate_metrics["f2_malignant"],
            candidate_metrics["recall_malignant"],
            candidate_metrics["accuracy"],
            candidate_metrics["precision_malignant"],
            candidate_metrics["specificity_benign"],
            candidate_metrics["threshold"],
        )
        if candidate_key > best_key:
            best_metrics = candidate_metrics
            best_key = candidate_key

    return {
        "model_name": "ensemble_mean_logistic_random_forest",
        "model_params": {
            "logistic_params": logistic_params,
            "ensemble_rule": "mean_probability",
        },
        "strategy": "maximize_f2_on_dev_oof",
        "default_threshold": DEFAULT_CLASSIFICATION_THRESHOLD,
        "selected_threshold": float(best_metrics["threshold"]),
        "oof_default_metrics": compact_metrics(default_metrics),
        "oof_tuned_metrics": compact_metrics(best_metrics),
    }

def train_evaluate_ensemble_candidate(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_test_final: pd.DataFrame,
    y_test_final: pd.Series,
    random_state: int,
    n_estimators: int,
    n_splits: int,
    logistic_params: dict,
) -> dict:
    threshold_tuning = tune_ensemble_threshold(
        X=X_dev,
        y=y_dev,
        random_state=random_state,
        n_estimators=n_estimators,
        n_splits=n_splits,
        logistic_params=logistic_params,
    )
    selected_threshold = float(threshold_tuning["selected_threshold"])

    logistic_model = build_model(
        "logistic_regression",
        random_state=random_state,
        n_estimators=n_estimators,
        model_params=logistic_params,
    )
    logistic_model.fit(X_dev, y_dev)

    rf_model = build_model(
        "random_forest",
        random_state=random_state,
        n_estimators=n_estimators,
        model_params=None,
    )
    rf_model.fit(X_dev, y_dev)

    logistic_scores = malignant_probability_scores(logistic_model, X_test_final)
    rf_scores = malignant_probability_scores(rf_model, X_test_final)
    ensemble_scores = (logistic_scores + rf_scores) / 2.0

    final_metrics_default = evaluate_predictions(
        y_test_final,
        ensemble_scores,
        DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    final_metrics_tuned = evaluate_predictions(
        y_test_final,
        ensemble_scores,
        selected_threshold,
    )

    error_analysis_df = X_test_final.copy()
    error_analysis_df["y_true"] = y_test_final.values
    error_analysis_df["logistic_score"] = logistic_scores
    error_analysis_df["random_forest_score"] = rf_scores
    error_analysis_df["malignant_score"] = ensemble_scores
    error_analysis_df["pred_default"] = labels_from_malignant_scores(
        ensemble_scores, DEFAULT_CLASSIFICATION_THRESHOLD
    )
    error_analysis_df["pred_tuned"] = labels_from_malignant_scores(
        ensemble_scores, selected_threshold
    )

    def case_type(y_true: str, y_pred: str) -> str:
        if y_true == "M" and y_pred == "M":
            return "TP"
        if y_true == "B" and y_pred == "B":
            return "TN"
        if y_true == "B" and y_pred == "M":
            return "FP"
        return "FN"

    error_analysis_df["case_default"] = [
        case_type(y_true, y_pred)
        for y_true, y_pred in zip(error_analysis_df["y_true"], error_analysis_df["pred_default"])
    ]
    error_analysis_df["case_tuned"] = [
        case_type(y_true, y_pred)
        for y_true, y_pred in zip(error_analysis_df["y_true"], error_analysis_df["pred_tuned"])
    ]
    error_analysis_df["distance_to_default_threshold"] = (
        error_analysis_df["malignant_score"] - DEFAULT_CLASSIFICATION_THRESHOLD
    )
    error_analysis_df["distance_to_tuned_threshold"] = (
        error_analysis_df["malignant_score"] - selected_threshold
    )
    error_analysis_df = error_analysis_df.sort_values("malignant_score", ascending=False)

    return {
        "model_name": "ensemble_mean_logistic_random_forest",
        "model_params": {
            "logistic_params": logistic_params,
            "ensemble_rule": "mean_probability",
        },
        "model": {
            "logistic_model": logistic_model,
            "random_forest_model": rf_model,
        },
        "threshold_tuning": threshold_tuning,
        "selected_threshold": selected_threshold,
        "final_metrics_default": final_metrics_default,
        "final_metrics_tuned": final_metrics_tuned,
        "error_analysis_df": error_analysis_df,
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Caminho para o wdbc.data (UCI). Se vazio, usa fallback do scikit-learn.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path) if args.data_path else None
    if data_path:
        if not data_path.exists():
            raise FileNotFoundError(f"Arquivo informado em --data-path nao existe: {data_path}")
        X, y, source = load_dataset_from_uci_wdbc(data_path)
    else:
        X, y, source = load_dataset_fallback_sklearn()

    X_dev, X_test_final, y_dev, y_test_final = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=args.random_state,
    )

    model_names = ["dummy", "random_forest", "logistic_regression"]
    cv_results = run_cross_validation(
        X=X_dev,
        y=y_dev,
        model_names=model_names,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        n_splits=args.n_splits,
    )
    best_result = select_best_cv_result(cv_results)

    hyperparameter_tuning = tune_logistic_hyperparameters(
        X=X_dev,
        y=y_dev,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        n_splits=args.n_splits,
    )
    best_logistic_candidate = hyperparameter_tuning["best_candidate"]
    selected_model_params = best_logistic_candidate["model_params"]

    official_candidate = train_evaluate_ensemble_candidate(
        X_dev=X_dev,
        y_dev=y_dev,
        X_test_final=X_test_final,
        y_test_final=y_test_final,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        n_splits=args.n_splits,
        logistic_params=selected_model_params,
    )

    joblib.dump(official_candidate["model"], out_dir / "model.joblib")
    official_candidate["error_analysis_df"].to_csv(
        out_dir / "error_analysis_test.csv",
        index=True,
    )

    feature_meta = compute_feature_stats(X_dev)
    feature_meta["dataset_source"] = source
    save_json(out_dir / "feature_stats.json", feature_meta)
    save_json(out_dir / "cv_results.json", cv_results)

    cross_validation_summary = build_cross_validation_summary(
        best_result=best_result,
        n_splits=args.n_splits,
    )

    model_info = {
        "model_type": official_candidate["model_name"],
        "model_name": official_candidate["model_name"],
        "trained_at": utc_now_iso(),
        "accuracy_test": official_candidate["final_metrics_tuned"]["accuracy"],
        "threshold_malignant": official_candidate["selected_threshold"],
        "notes": "Modelo educacional. Nao usar para diagnostico clinico.",
        "dataset_source": source,
        "cross_validation": cross_validation_summary,
        "selected_model_params": selected_model_params,
        "extra": {
            "dataset_source": source,
            "threshold_tuning": official_candidate["threshold_tuning"],
            "test_metrics_default_threshold": compact_metrics(
                official_candidate["final_metrics_default"]
            ),
            "test_metrics_tuned_threshold": compact_metrics(
                official_candidate["final_metrics_tuned"]
            ),
            "hyperparameter_tuning": hyperparameter_tuning,
            "ensemble_rule": "mean_probability",
            "base_models": {
                "logistic_regression": {
                    "params": selected_model_params,
                },
                "random_forest": {
                    "n_estimators": args.n_estimators,
                    "class_weight": "balanced",
                },
            },
        },
    }
    save_json(out_dir / "model_info.json", model_info)

    tuned = official_candidate["final_metrics_tuned"]
    print("Training finished.")
    print(f"Official model: {official_candidate['model_name']}")
    print(f"Threshold: {official_candidate['selected_threshold']:.4f}")
    print(
        f"Recall(M): {tuned['recall_malignant']:.4f} | "
        f"FNR(M): {tuned['fnr_malignant']:.4f} | "
        f"F2(M): {tuned['f2_malignant']:.4f} | "
        f"Accuracy: {tuned['accuracy']:.4f} | "
        f"ROC-AUC(M): {tuned['roc_auc_malignant']:.4f}"
    )
    print(f"Confusion matrix: {tuned['confusion_matrix']}")
    print(f"Artifacts saved in: {out_dir.resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
