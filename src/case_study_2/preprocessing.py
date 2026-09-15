"""Prétraitement retenu pour les relevés SCANIA.

Choix appliqués :
- valeurs cumulées + taux entre les deux derniers relevés ;
- imputation par la médiane du train avec indicateurs de valeurs manquantes ;
- pas de suppression des outliers et pas de normalisation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer


ID = "vehicle_id"
TIME = "time_step"


def read_last_rows(path: Path, n: int = 2, chunksize: int = 100_000) -> pd.DataFrame:
    """Lit seulement les ``n`` dernières lignes de chaque véhicule.

    Les CSV sont triés par ``vehicle_id``. La lecture par morceaux évite de
    charger le fichier train de plus de 1 Go entièrement en mémoire.
    """
    parts: list[pd.DataFrame] = []
    carry = pd.DataFrame()

    for chunk in pd.read_csv(path, chunksize=chunksize):
        data = pd.concat([carry, chunk], ignore_index=True)
        last_vehicle = data[ID].iloc[-1]

        # Tous les véhicules sauf le dernier sont complets dans ce morceau.
        complete = data[data[ID] != last_vehicle]
        if not complete.empty:
            parts.append(complete.groupby(ID, sort=False).tail(n))

        # Le dernier véhicule peut continuer dans le morceau suivant.
        carry = data[data[ID] == last_vehicle].tail(n)

    if carry.empty:
        raise ValueError(f"Fichier vide : {path}")

    parts.append(carry)
    return pd.concat(parts, ignore_index=True)


def make_features(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crée une ligne de features par véhicule à son dernier relevé."""
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

    # Les mesures sont cumulatives. Les rares petites corrections négatives
    # sont ramenées à zéro avant de calculer le taux.
    delta = by_vehicle[sensor_columns].diff().clip(lower=0)
    rate = delta.div(delta_time, axis=0)

    last = by_vehicle.tail(1).index
    ids = history.loc[last, [ID, TIME]].reset_index(drop=True)
    raw = history.loc[last, sensor_columns].reset_index(drop=True).add_prefix("raw__")
    rate = rate.loc[last].reset_index(drop=True).add_prefix("rate__")

    context = pd.DataFrame(
        {
            "context__delta_time": delta_time.loc[last].to_numpy(),
            "quality__raw_missing_count": (
                history.loc[last, sensor_columns].isna().sum(axis=1).to_numpy()
            ),
        }
    )

    # context__has_previous et quality__negative_delta_count ne sont pas créées :
    # elles étaient constantes et n'apportaient rien au modèle.
    features = pd.concat([raw, rate, context], axis=1)
    return ids, features


def fit_imputer(train_features: pd.DataFrame) -> SimpleImputer:
    """Apprend les médianes uniquement sur le train pour éviter la fuite."""
    return SimpleImputer(
        strategy="median",
        add_indicator=True,
        keep_empty_features=True,
    ).fit(train_features)


def apply_imputer(imputer: SimpleImputer, features: pd.DataFrame) -> pd.DataFrame:
    """Applique les médianes du train et conserve des noms de colonnes clairs."""
    values = imputer.transform(features)
    columns = imputer.get_feature_names_out(features.columns)
    return pd.DataFrame(values, columns=columns, index=features.index)


def preprocess(data_dir: Path, output_dir: Path) -> None:
    """Prétraite train, validation et test puis les exporte en CSV."""
    snapshots: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for split in ("train", "validation", "test"):
        history = read_last_rows(data_dir / f"{split}_operational_readouts.csv")
        snapshots[split] = make_features(history)

    imputer = fit_imputer(snapshots["train"][1])
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, (ids, features) in snapshots.items():
        ready = pd.concat([ids, apply_imputer(imputer, features)], axis=1)
        path = output_dir / f"{split}_features.csv"
        ready.to_csv(path, index=False)
        print(f"{split}: {ready.shape[0]} lignes, {ready.shape[1] - 2} features -> {path}")

    pd.DataFrame(
        {"feature": snapshots["train"][1].columns, "train_median": imputer.statistics_}
    ).to_csv(output_dir / "train_medians.csv", index=False)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "preprocessed_data")
    args = parser.parse_args()
    preprocess(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
