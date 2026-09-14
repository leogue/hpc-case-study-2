from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# CONFIGURATION
# ============================================================


# eda_scania.py is located in:
# hpc-case-study-2/src/case_study_2/eda_scania.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "train_operational_readouts.csv"
OUTPUT_DIR = PROJECT_ROOT / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUT_DIR / "scania_eda_report.pdf"

print("PROJECT_ROOT :", PROJECT_ROOT)
print("DATA_PATH    :", DATA_PATH)
print("FILE EXISTS  :", DATA_PATH.exists())

# 8 cumulative numerical counters described in the dataset
COUNTER_FEATURES = [
    "171_0",
    "666_0",
    "427_0",
    "837_0",
    "309_0",
    "835_0",
    "370_0",
    "100_0",
]

# Histogram variables
HISTOGRAM_PREFIXES = [
    "167",
    "272",
    "291",
    "158",
    "459",
    "397",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def save_figure(fig, filename, pdf):
    """
    Save a figure both as PNG and inside the global PDF report.
    """
    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    pdf.savefig(
        fig,
        bbox_inches="tight"
    )

    print(f"Saved: {path}")

    plt.close(fig)


def get_histogram_columns(df, prefix):
    """
    Return histogram columns in numerical bin order.

    Example:
        397_0, 397_1, ..., 397_10
    instead of lexical ordering such as:
        397_0, 397_1, 397_10, 397_11, ...
    """
    columns = [
        col
        for col in df.columns
        if col.startswith(prefix + "_")
    ]

    return sorted(
        columns,
        key=lambda col: int(col.split("_")[1])
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading SCANIA operational data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Rows:     {len(df):,}")
print(f"Columns:  {df.shape[1]:,}")
print(f"Vehicles: {df['vehicle_id'].nunique():,}")

# Sort because several analyses require chronological order
df = df.sort_values(
    ["vehicle_id", "time_step"]
).reset_index(drop=True)


# ============================================================
# BASIC SUMMARY
# ============================================================

summary = {
    "n_rows": len(df),
    "n_columns": df.shape[1],
    "n_vehicles": df["vehicle_id"].nunique(),
    "min_time_step": df["time_step"].min(),
    "max_time_step": df["time_step"].max(),
    "mean_missing_percent": df.isna().mean().mean() * 100,
}

summary_df = pd.DataFrame(
    summary.items(),
    columns=["metric", "value"]
)

summary_df.to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False
)

print("\nDataset summary:")
print(summary_df)


# ============================================================
# VEHICLE LEVEL STATISTICS
# ============================================================

vehicle_stats = (
    df.groupby("vehicle_id")
    .agg(
        n_readouts=("time_step", "size"),
        first_time=("time_step", "min"),
        last_time=("time_step", "max"),
    )
    .reset_index()
)

vehicle_stats["observation_duration"] = (
    vehicle_stats["last_time"]
    - vehicle_stats["first_time"]
)

vehicle_stats.to_csv(
    OUTPUT_DIR / "vehicle_statistics.csv",
    index=False
)


# ============================================================
# PRECOMPUTE COUNTER DELTAS
# ============================================================

print("\nComputing counter increments...")

counter_deltas = (
    df.groupby("vehicle_id")[COUNTER_FEATURES]
    .diff()
)

counter_deltas.columns = [
    f"delta_{col}"
    for col in COUNTER_FEATURES
]


# ============================================================
# CREATE REPORT
# ============================================================

with PdfPages(PDF_PATH) as pdf:

    # ========================================================
    # 1. MISSING VALUES
    # ========================================================

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(20, 7))

    ax.bar(
        range(len(missing)),
        missing.values
    )

    ax.set_xticks(range(len(missing)))
    ax.set_xticklabels(
        missing.index,
        rotation=90,
        fontsize=6
    )

    ax.set_ylabel("Missing values (%)")
    ax.set_xlabel("Feature")
    ax.set_title(
        "Missing Values by Feature"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "01_missing_values.png",
        pdf
    )


    # ========================================================
    # 2. NUMBER OF READOUTS PER VEHICLE
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        vehicle_stats["n_readouts"],
        bins=60
    )

    ax.set_xlabel("Number of readouts")
    ax.set_ylabel("Number of vehicles")
    ax.set_title(
        "Number of Operational Readouts per Vehicle"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "02_readouts_per_vehicle.png",
        pdf
    )


    # ========================================================
    # 3. OBSERVATION DURATION
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        vehicle_stats["observation_duration"],
        bins=60
    )

    ax.set_xlabel(
        "Observation duration (time steps)"
    )

    ax.set_ylabel(
        "Number of vehicles"
    )

    ax.set_title(
        "Observation Duration per Vehicle"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "03_observation_duration.png",
        pdf
    )


    # ========================================================
    # 4. SAMPLING INTERVAL
    # ========================================================

    delta_time = (
        df.groupby("vehicle_id")["time_step"]
        .diff()
    )

    # Remove NaNs, zeros and extreme values
    valid_delta_time = delta_time[
        delta_time > 0
    ].dropna()

    upper_limit = valid_delta_time.quantile(0.99)

    plot_delta_time = valid_delta_time[
        valid_delta_time <= upper_limit
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        plot_delta_time,
        bins=60
    )

    ax.set_xlabel(
        "Time between consecutive readouts"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        "Sampling Interval Distribution (99% of observations)"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "04_sampling_interval.png",
        pdf
    )


    # ========================================================
    # 5. NORMALIZED COUNTER EVOLUTION
    # ========================================================

    # Select vehicle with most observations
    example_vehicle = (
        vehicle_stats
        .sort_values(
            "n_readouts",
            ascending=False
        )
        .iloc[0]["vehicle_id"]
    )

    vehicle_df = (
        df[
            df["vehicle_id"] == example_vehicle
        ][
            ["time_step"] + COUNTER_FEATURES
        ]
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for feature in COUNTER_FEATURES:

        values = vehicle_df[feature]

        min_value = values.min()
        max_value = values.max()

        if max_value > min_value:

            normalized = (
                values - min_value
            ) / (
                max_value - min_value
            )

            ax.plot(
                vehicle_df["time_step"],
                normalized,
                label=feature
            )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized value")

    ax.set_title(
        f"Normalized Counter Evolution — Vehicle {example_vehicle}"
    )

    ax.legend(
        title="Counter",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    save_figure(
        fig,
        "05_normalized_counter_evolution.png",
        pdf
    )


    # ========================================================
    # 6. RAW COUNTER CORRELATION
    # ========================================================

    raw_corr = (
        df[COUNTER_FEATURES]
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    im = ax.imshow(
        raw_corr,
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(
        range(len(COUNTER_FEATURES))
    )

    ax.set_yticks(
        range(len(COUNTER_FEATURES))
    )

    ax.set_xticklabels(
        COUNTER_FEATURES,
        rotation=45
    )

    ax.set_yticklabels(
        COUNTER_FEATURES
    )

    for i in range(len(COUNTER_FEATURES)):
        for j in range(len(COUNTER_FEATURES)):

            ax.text(
                j,
                i,
                f"{raw_corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(
        im,
        ax=ax,
        label="Correlation"
    )

    ax.set_title(
        "Correlation between Cumulative Counters"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "06_raw_counter_correlation.png",
        pdf
    )


    # ========================================================
    # 7. COUNTER INCREMENT CORRELATION
    # ========================================================

    delta_corr = (
        counter_deltas
        .corr()
    )

    delta_corr.index = COUNTER_FEATURES
    delta_corr.columns = COUNTER_FEATURES

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    im = ax.imshow(
        delta_corr,
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(
        range(len(COUNTER_FEATURES))
    )

    ax.set_yticks(
        range(len(COUNTER_FEATURES))
    )

    ax.set_xticklabels(
        COUNTER_FEATURES,
        rotation=45
    )

    ax.set_yticklabels(
        COUNTER_FEATURES
    )

    for i in range(len(COUNTER_FEATURES)):
        for j in range(len(COUNTER_FEATURES)):

            ax.text(
                j,
                i,
                f"{delta_corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(
        im,
        ax=ax,
        label="Correlation"
    )

    ax.set_title(
        "Correlation between Counter Increments"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "07_delta_counter_correlation.png",
        pdf
    )


    # ========================================================
    # 8. POTENTIAL COUNTER RESETS
    # ========================================================

    reset_statistics = []

    for feature in COUNTER_FEATURES:

        delta_col = f"delta_{feature}"

        series = (
            counter_deltas[delta_col]
            .dropna()
        )

        if len(series) == 0:
            percentage = 0

        else:
            percentage = (
                (series < 0).mean()
                * 100
            )

        reset_statistics.append(
            {
                "feature": feature,
                "negative_delta_percent": percentage
            }
        )

    reset_df = pd.DataFrame(
        reset_statistics
    )

    reset_df.to_csv(
        OUTPUT_DIR / "counter_reset_statistics.csv",
        index=False
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        reset_df["feature"],
        reset_df["negative_delta_percent"]
    )

    ax.set_xlabel("Counter")

    ax.set_ylabel(
        "Negative increments (%)"
    )

    ax.set_title(
        "Potential Counter Resets / Inconsistencies"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "08_potential_counter_resets.png",
        pdf
    )


    # ========================================================
    # 9. LAST READOUT DISTRIBUTION
    # ========================================================

    last_readouts = (
        df.groupby("vehicle_id")
        .tail(1)
        .copy()
    )

    feature = "171_0"

    values = (
        last_readouts[feature]
        .dropna()
    )

    log_values = np.log1p(values)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        log_values,
        bins=60
    )

    ax.set_xlabel(
        f"log(1 + {feature})"
    )

    ax.set_ylabel(
        "Number of vehicles"
    )

    ax.set_title(
        f"Distribution of {feature} at Last Vehicle Readout"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "09_last_readout_distribution.png",
        pdf
    )


    # ========================================================
    # 10. HISTOGRAM FEATURE 167
    # ========================================================

    prefix = "167"

    histogram_columns = (
        get_histogram_columns(
            df,
            prefix
        )
    )

    # Top 5 vehicles with most readouts
    selected_vehicles = (
        vehicle_stats
        .sort_values(
            "n_readouts",
            ascending=False
        )
        .head(5)["vehicle_id"]
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for vehicle_id in selected_vehicles:

        vehicle_data = (
            df[
                df["vehicle_id"]
                == vehicle_id
            ]
            .iloc[-1]
        )

        values = (
            vehicle_data[
                histogram_columns
            ]
            .astype(float)
            .values
        )

        total = np.nansum(values)

        if total > 0:
            values = values / total

        bins = range(
            len(histogram_columns)
        )

        ax.plot(
            bins,
            values,
            marker="o",
            label=str(vehicle_id)
        )

    ax.set_xlabel(
        f"{prefix} histogram bin"
    )

    ax.set_ylabel(
        "Share of total histogram"
    )

    ax.set_title(
        f"Normalized Histogram {prefix} Profiles"
    )

    ax.legend(
        title="Vehicle"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    save_figure(
        fig,
        "10_histogram_167_profiles.png",
        pdf
    )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 60)
print("EDA completed")
print("=" * 60)

print(f"\nAll outputs are available in:")
print(OUTPUT_DIR.resolve())

print(f"\nCombined PDF report:")
print(PDF_PATH.resolve())