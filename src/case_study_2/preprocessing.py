"""Prétraitement V2 pour les données SCANIA Component X.

Objectif général
----------------
Les fichiers opérationnels SCANIA sont des séries temporelles : un véhicule a
plusieurs relevés dans le temps. Les modèles ML classiques attendent plutôt une
table tabulaire avec une seule ligne par véhicule. Ce script transforme donc
les historiques bruts en fichiers directement exploitables par des modèles
comme Logistic Regression, Random Forest, HistGradientBoosting, LightGBM,
XGBoost ou CatBoost.

Prétraitement appliqué
----------------------
1. Lecture par morceaux des gros fichiers CSV pour limiter la mémoire.
2. Conservation d'une fenêtre récente par véhicule, par défaut 20 relevés.
3. Création d'une ligne de features par véhicule.
4. Ajout de features temporelles : nombre de relevés, durée d'observation,
   premier/dernier time_step, intervalle moyen, maximum et écart-type.
5. Ajout des valeurs brutes du dernier relevé : raw__...
6. Ajout de features d'évolution récente : variation totale, dernière
   variation, moyenne, maximum, écart-type et taux de variation.
7. Traitement spécifique des histogrammes : total, proportions par bin, bin
   dominant, part maximale, entropie et évolution des proportions.
8. Ajout d'indicateurs qualité : valeurs manquantes, deltas négatifs,
   deltas nuls.
9. Fusion avec les spécifications véhicule quand elles existent.
10. Fusion avec les labels/TTE quand ils existent.
11. Imputation médiane apprise uniquement sur train pour les numériques.
12. One-hot encoding appris uniquement sur train pour les catégorielles.

Les outliers ne sont pas supprimés ici : en maintenance prédictive, les valeurs
extrêmes peuvent être justement les observations les plus intéressantes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


ID = "vehicle_id"
TIME = "time_step"
HISTOGRAM_PREFIXES = ("167", "272", "291", "158", "459", "397")


def histogram_columns(columns: list[str], prefix: str) -> list[str]:
    """Retourne les colonnes d'un histogramme dans l'ordre numérique des bins."""
    selected = [column for column in columns if column.startswith(f"{prefix}_")]
    return sorted(selected, key=lambda column: int(column.split("_")[1]))


def _vehicle_time_stats(group: pd.DataFrame) -> dict[str, float]:
    """Calcule les statistiques temporelles d'un véhicule sur tout son historique."""
    time_values = group[TIME].to_numpy()
    delta_time = np.diff(time_values)
    return {
        ID: group[ID].iloc[0],
        "context__n_readouts": float(len(group)),
        "context__first_time": float(time_values[0]),
        "context__last_time": float(time_values[-1]),
        "context__observation_duration": float(time_values[-1] - time_values[0]),
        "context__mean_delta_time": float(np.mean(delta_time)) if len(delta_time) else np.nan,
        "context__max_delta_time": float(np.max(delta_time)) if len(delta_time) else np.nan,
        "context__std_delta_time": float(np.std(delta_time, ddof=0)) if len(delta_time) else np.nan,
    }


def read_recent_rows_with_context(
    path: Path,
    window_size: int = 20,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lit une fenêtre récente par véhicule et les stats temporelles globales."""
    if window_size < 2:
        raise ValueError("window_size doit être supérieur ou égal à 2")

    windows: list[pd.DataFrame] = []
    context_rows: list[dict[str, float]] = []
    carry = pd.DataFrame()

    for chunk in pd.read_csv(path, chunksize=chunksize):
        data = pd.concat([carry, chunk], ignore_index=True)
        last_vehicle = data[ID].iloc[-1]
        complete = data[data[ID] != last_vehicle]

        if not complete.empty:
            for _, group in complete.groupby(ID, sort=False):
                windows.append(group.tail(window_size))
                context_rows.append(_vehicle_time_stats(group))

        carry = data[data[ID] == last_vehicle]

    if carry.empty:
        raise ValueError(f"Fichier vide : {path}")

    for _, group in carry.groupby(ID, sort=False):
        windows.append(group.tail(window_size))
        context_rows.append(_vehicle_time_stats(group))

    return pd.concat(windows, ignore_index=True), pd.DataFrame(context_rows)


def make_histogram_features(history: pd.DataFrame, last_index: pd.Index) -> pd.DataFrame:
    """Crée les features de profil et d'évolution pour les histogrammes."""
    frames: list[pd.DataFrame] = []
    all_columns = list(history.columns)
    by_vehicle = history.groupby(ID, sort=False)

    for prefix in HISTOGRAM_PREFIXES:
        columns = histogram_columns(all_columns, prefix)
        if not columns:
            continue

        first = by_vehicle[columns].first().reset_index(drop=True)
        last = history.loc[last_index, columns].reset_index(drop=True)
        total = last.sum(axis=1, skipna=True)
        first_total = first.sum(axis=1, skipna=True)

        shares = last.div(total.replace(0, np.nan), axis=0)
        first_shares = first.div(first_total.replace(0, np.nan), axis=0)
        share_change = shares - first_shares

        probabilities = shares.to_numpy(dtype=float)
        safe_probabilities = np.where(probabilities > 0, probabilities, np.nan)
        entropy = -np.nansum(safe_probabilities * np.log(safe_probabilities), axis=1)
        dominant_source = np.where(np.isnan(probabilities), -np.inf, probabilities)
        has_valid_share = np.isfinite(dominant_source).any(axis=1)
        dominant_bin = np.where(has_valid_share, np.argmax(dominant_source, axis=1), np.nan)
        max_share = np.full(len(probabilities), np.nan)
        max_share[has_valid_share] = np.nanmax(probabilities[has_valid_share], axis=1)

        frames.append(
            pd.DataFrame(
                {
                    f"histo__{prefix}_total": total.to_numpy(),
                    f"histo__{prefix}_dominant_bin": dominant_bin,
                    f"histo__{prefix}_max_share": max_share,
                    f"histo__{prefix}_entropy": entropy,
                    f"histo_trend__{prefix}_total_change": total.to_numpy() - first_total.to_numpy(),
                }
            )
        )
        frames.append(shares.add_prefix(f"histo__{prefix}_share__"))
        frames.append(share_change.add_prefix(f"histo_trend__{prefix}_share_change__"))

    if not frames:
        return pd.DataFrame(index=range(history[ID].nunique()))

    return pd.concat(frames, axis=1)


def make_operational_features(
    history: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transforme une fenêtre temporelle en une ligne de features par véhicule."""
    required = {ID, TIME}
    if not required.issubset(history.columns):
        raise ValueError(f"Colonnes obligatoires absentes : {required - set(history.columns)}")

    history = history.sort_values([ID, TIME]).reset_index(drop=True)
    if history.duplicated([ID, TIME]).any():
        raise ValueError("Couple vehicle_id/time_step dupliqué")

    sensor_columns = [column for column in history.columns if column not in required]
    by_vehicle = history.groupby(ID, sort=False)
    delta_time = by_vehicle[TIME].diff()

    if (delta_time.dropna() <= 0).any():
        raise ValueError("time_step doit être strictement croissant par véhicule")

    raw_delta = by_vehicle[sensor_columns].diff()
    positive_delta = raw_delta.clip(lower=0)
    rate = positive_delta.div(delta_time, axis=0)

    last_index = by_vehicle.tail(1).index
    ids = history.loc[last_index, [ID, TIME]].reset_index(drop=True)
    raw = history.loc[last_index, sensor_columns].reset_index(drop=True).add_prefix("raw__")
    rate_last = rate.loc[last_index].reset_index(drop=True).add_prefix("rate__")

    first = by_vehicle[sensor_columns].first().reset_index(drop=True)
    last = history.loc[last_index, sensor_columns].reset_index(drop=True)
    trend_change = (last - first).add_prefix("trend__change__")

    grouped_delta = positive_delta.groupby(history[ID], sort=False)
    delta_features = pd.concat(
        [
            grouped_delta.mean().reset_index(drop=True).add_prefix("delta_mean__"),
            grouped_delta.std(ddof=0).reset_index(drop=True).add_prefix("delta_std__"),
            grouped_delta.max().reset_index(drop=True).add_prefix("delta_max__"),
            grouped_delta.sum().reset_index(drop=True).add_prefix("delta_sum__"),
            positive_delta.loc[last_index].reset_index(drop=True).add_prefix("delta_last__"),
        ],
        axis=1,
    )

    quality = pd.DataFrame(
        {
            "quality__raw_missing_count": history.loc[last_index, sensor_columns].isna().sum(axis=1).to_numpy(),
            "quality__window_missing_count": history[sensor_columns].isna().sum(axis=1).groupby(history[ID], sort=False).sum().to_numpy(),
            "quality__negative_delta_count": (raw_delta < 0).sum(axis=1).groupby(history[ID], sort=False).sum().to_numpy(),
            "quality__zero_delta_count": (raw_delta == 0).sum(axis=1).groupby(history[ID], sort=False).sum().to_numpy(),
        }
    )

    features = pd.concat(
        [
            raw,
            rate_last,
            trend_change,
            delta_features,
            make_histogram_features(history, last_index),
            context.sort_values(ID).drop(columns=[ID]).reset_index(drop=True),
            quality,
        ],
        axis=1,
    )
    return ids.sort_values(ID).reset_index(drop=True), features


def add_specifications(
    data_dir: Path,
    split: str,
    ids: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Ajoute les caractéristiques fixes du véhicule si le fichier existe."""
    ready = pd.concat([ids.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    path = data_dir / f"{split}_specifications.csv"
    if not path.exists():
        return ready

    specs = pd.read_csv(path)
    specs = specs.add_prefix("spec__").rename(columns={f"spec__{ID}": ID})
    return ready.merge(specs, on=ID, how="left")


def add_target(data_dir: Path, split: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la cible train/validation/test si le fichier correspondant existe."""
    candidates = []
    if split == "train":
        candidates.append(data_dir / "train_tte.csv")
    candidates.append(data_dir / f"{split}_labels.csv")

    for path in candidates:
        if path.exists():
            return frame.merge(pd.read_csv(path), on=ID, how="left")

    return frame


def split_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Sépare les colonnes numériques et catégorielles à transformer."""
    protected = {ID, TIME, "class_label", "in_study_repair", "length_of_study_time_step"}
    feature_columns = [column for column in frame.columns if column not in protected]
    numerical = [column for column in feature_columns if is_numeric_dtype(frame[column])]
    categorical = [column for column in feature_columns if column not in numerical]
    return numerical, categorical


def fit_transformers(
    train_frame: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> tuple[SimpleImputer, OneHotEncoder | None]:
    """Apprend l'imputation et l'encodage uniquement sur le train."""
    imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
        keep_empty_features=True,
    ).fit(train_frame[numerical_columns])

    encoder = None
    if categorical_columns:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
            train_frame[categorical_columns].fillna("unknown")
        )

    return imputer, encoder


def apply_transformers(
    frame: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
    imputer: SimpleImputer,
    encoder: OneHotEncoder | None,
) -> pd.DataFrame:
    """Applique les transformations apprises sur train à un split donné."""
    numeric_values = imputer.transform(frame[numerical_columns])
    numeric_frame = pd.DataFrame(
        numeric_values,
        columns=imputer.get_feature_names_out(numerical_columns),
        index=frame.index,
    )

    parts = [numeric_frame]
    if encoder is not None and categorical_columns:
        categorical_values = encoder.transform(frame[categorical_columns].fillna("unknown"))
        parts.append(
            pd.DataFrame(
                categorical_values,
                columns=encoder.get_feature_names_out(categorical_columns),
                index=frame.index,
            )
        )

    return pd.concat(parts, axis=1)


def preprocess(data_dir: Path, output_dir: Path, window_size: int = 20) -> None:
    """Prétraite train, validation et test puis exporte des CSV prêts pour le ML."""
    split_frames: dict[str, pd.DataFrame] = {}

    for split in ("train", "validation", "test"):
        history, context = read_recent_rows_with_context(
            data_dir / f"{split}_operational_readouts.csv",
            window_size=window_size,
        )
        ids, operational_features = make_operational_features(history, context)
        frame = add_specifications(data_dir, split, ids, operational_features)
        split_frames[split] = add_target(data_dir, split, frame)

    numerical_columns, categorical_columns = split_feature_columns(split_frames["train"])
    imputer, encoder = fit_transformers(split_frames["train"], numerical_columns, categorical_columns)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, frame in split_frames.items():
        target_columns = [
            column
            for column in ("class_label", "in_study_repair", "length_of_study_time_step")
            if column in frame.columns
        ]
        ids_and_target = frame[[ID, TIME] + target_columns].reset_index(drop=True)
        model_features = apply_transformers(
            frame,
            numerical_columns,
            categorical_columns,
            imputer,
            encoder,
        ).reset_index(drop=True)

        ready = pd.concat([ids_and_target, model_features], axis=1)
        path = output_dir / f"{split}_features.csv"
        ready.to_csv(path, index=False)
        n_features = ready.shape[1] - len(ids_and_target.columns)
        print(f"{split}: {ready.shape[0]} lignes, {n_features} features -> {path}")

    pd.DataFrame(
        {"feature": imputer.feature_names_in_, "train_median": imputer.statistics_}
    ).to_csv(output_dir / "train_medians.csv", index=False)

    if encoder is not None:
        pd.DataFrame({"feature": encoder.get_feature_names_out(categorical_columns)}).to_csv(
            output_dir / "encoded_categorical_features.csv",
            index=False,
        )


def main() -> None:
    """Point d'entrée en ligne de commande."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "preprocessed_data_v2")
    parser.add_argument("--window-size", type=int, default=20)
    args = parser.parse_args()
    preprocess(args.data_dir, args.output_dir, args.window_size)


if __name__ == "__main__":
    main()
