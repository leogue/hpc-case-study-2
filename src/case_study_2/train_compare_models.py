"""Entraîne et compare plusieurs modèles ML classiques pour SCANIA.

Ce script utilise les fichiers générés par ``preprocessing.py`` dans
``preprocessed_data_v2``. Il compare :
- Logistic Regression ;
- Random Forest ;
- HistGradientBoosting ;
- LightGBM si la librairie ``lightgbm`` est installée.

Sorties générées
----------------
Dans ``model_outputs`` par défaut :
- ``model_metrics.csv`` : tableau comparatif des métriques ;
- ``validation_predictions.csv`` : probabilités prédites par modèle ;
- ``model_comparison.txt`` : résumé lisible avec classement et matrices de confusion.

Métriques
---------
Le dataset est déséquilibré, donc l'accuracy seule est insuffisante. Le script
calcule aussi precision, recall, F1-score, ROC-AUC et PR-AUC.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ID = "vehicle_id"
TIME = "time_step"
TRAIN_TARGET = "in_study_repair"
LABEL_TARGET = "class_label"
EXCLUDED_COLUMNS = {ID, TIME, TRAIN_TARGET, LABEL_TARGET, "length_of_study_time_step"}


def load_split(
    path: Path,
    target_column: str | None,
    nrows: int | None = None,
    binarize_target: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series]:
    """Charge un split prétraité, sépare X/y et conserve les vehicle_id."""
    frame = pd.read_csv(path, nrows=nrows)
    ids = frame[ID]
    y = None
    if target_column and target_column in frame.columns:
        y = frame[target_column].astype(int)
        if binarize_target:
            y = (y > 0).astype(int)
    feature_columns = [column for column in frame.columns if column not in EXCLUDED_COLUMNS]
    return frame[feature_columns], y, ids


def align_features(
    train_x: pd.DataFrame,
    validation_x: pd.DataFrame,
    test_x: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Aligne les colonnes des splits dans le même ordre."""
    validation_x = validation_x.reindex(columns=train_x.columns, fill_value=0)
    if test_x is not None:
        test_x = test_x.reindex(columns=train_x.columns, fill_value=0)
    return train_x, validation_x, test_x


def build_models(random_state: int) -> dict[str, object]:
    """Construit les modèles à comparer avec des paramètres raisonnables."""
    models: dict[str, object] = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2_000,
                random_state=random_state,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.06,
            max_iter=250,
            random_state=random_state,
        ),
    }

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return models

    models["lightgbm"] = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    return models


def predict_scores(model: object, x: pd.DataFrame) -> np.ndarray:
    """Retourne un score de classe positive utilisable pour ROC-AUC/PR-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def evaluate_predictions(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    """Calcule les métriques de classification à partir des scores prédits."""
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def train_and_evaluate(
    models: dict[str, object],
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    validation_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entraîne les modèles, mesure leurs performances et garde leurs prédictions."""
    metrics_rows: list[dict[str, object]] = []
    predictions = pd.DataFrame({ID: validation_ids, "y_true": validation_y})

    for name, model in models.items():
        started = perf_counter()
        model.fit(train_x, train_y)
        fit_seconds = perf_counter() - started

        y_score = predict_scores(model, validation_x)
        row = evaluate_predictions(validation_y, y_score)
        row.update({"model": name, "fit_seconds": fit_seconds, "status": "ok"})
        metrics_rows.append(row)
        predictions[f"{name}_score"] = y_score
        predictions[f"{name}_pred"] = (y_score >= 0.5).astype(int)

        print(
            f"{name}: F1={row['f1']:.4f}, recall={row['recall']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}, temps={fit_seconds:.1f}s"
        )

    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics.sort_values(["pr_auc", "f1", "recall"], ascending=False)
    return metrics, predictions


def write_report(metrics: pd.DataFrame, output_path: Path) -> None:
    """Écrit un rapport texte court pour lire rapidement la comparaison."""
    lines = ["Comparaison des modèles", "=" * 24, ""]
    if metrics.empty:
        lines.append("Aucun modèle n'a été évalué.")
    else:
        best = metrics.iloc[0]
        lines.append(f"Meilleur modèle selon PR-AUC : {best['model']}")
        lines.append("")
        for _, row in metrics.iterrows():
            if row.get("status") != "ok":
                lines.append(f"{row['model']} | {row['status']}")
                lines.append("")
                continue

            lines.append(
                f"{row['model']} | PR-AUC={row['pr_auc']:.4f} | "
                f"ROC-AUC={row['roc_auc']:.4f} | F1={row['f1']:.4f} | "
                f"precision={row['precision']:.4f} | recall={row['recall']:.4f}"
            )
            lines.append(
                "confusion matrix: "
                f"TN={int(row['true_negative'])}, FP={int(row['false_positive'])}, "
                f"FN={int(row['false_negative'])}, TP={int(row['true_positive'])}"
            )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Point d'entrée en ligne de commande."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=root / "preprocessed_data_v2")
    parser.add_argument("--output-dir", type=Path, default=root / "model_outputs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    args = parser.parse_args()

    train_x, train_y, _ = load_split(
        args.features_dir / "train_features.csv",
        TRAIN_TARGET,
        nrows=args.max_train_rows,
    )
    validation_x, validation_y, validation_ids = load_split(
        args.features_dir / "validation_features.csv",
        LABEL_TARGET,
        nrows=args.max_validation_rows,
        binarize_target=True,
    )

    if train_y is None:
        raise ValueError("La cible train 'in_study_repair' est absente.")
    if validation_y is None:
        raise ValueError("La cible validation 'class_label' est absente.")

    train_x, validation_x, _ = align_features(train_x, validation_x, None)
    models = build_models(args.random_state)

    if "lightgbm" not in models:
        print("LightGBM non installé : modèle ignoré. Installer avec `uv add lightgbm`.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions = train_and_evaluate(
        models,
        train_x,
        train_y,
        validation_x,
        validation_y,
        validation_ids,
    )

    if "lightgbm" not in models:
        metrics = pd.concat(
            [
                metrics,
                pd.DataFrame(
                    [
                        {
                            "model": "lightgbm",
                            "status": "skipped_not_installed",
                            "accuracy": np.nan,
                            "precision": np.nan,
                            "recall": np.nan,
                            "f1": np.nan,
                            "roc_auc": np.nan,
                            "pr_auc": np.nan,
                            "true_negative": np.nan,
                            "false_positive": np.nan,
                            "false_negative": np.nan,
                            "true_positive": np.nan,
                            "fit_seconds": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    write_report(metrics, args.output_dir / "model_comparison.txt")
    print(f"\nRésultats écrits dans : {args.output_dir}")


if __name__ == "__main__":
    main()
