# Fig. 28 — Seasonal Biodiversity Climatology
# Occurrence-based biodiversity indicators across Antarctic shelf seas
# 2008–2025
#
# thesis
# 1080 dpi
#
# Input:
# Biodiversity_indices_seasonal_ecosystem_coupling_2008_2025_1deg_masked_min10.nc


# MOUNT DRIVE 
from google.colab import drive
drive.mount('/content/drive')

# 0) Install packages

!pip -q install xarray netCDF4 h5netcdf dask



# 1) Imports

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt



# 2) File paths

BIO_FILE = "/content/drive/MyDrive/SAM_Thesis/Data/Biodiversity_Indices/Biodiversity_indices_seasonal_ecosystem_coupling_2008_2025_1deg_masked_min10.nc"

OUT_DIR = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG = os.path.join(
    OUT_DIR,
    "Fig28_seasonal_biodiversity_climatology_2008_2025_1080dpi.png"
)

OUT_CSV = os.path.join(
    OUT_DIR,
    "Fig28_seasonal_biodiversity_climatology_2008_2025.csv"
)



# 3) Main settings

START_YEAR = 2008
END_YEAR   = 2025
SAVE_DPI   = 1080

# Minimum valid grid cells per month
MIN_VALID_CELLS = 3

# Add figure title or not
# For thesis/journal final figure, usually False is cleaner because caption carries title.
ADD_FIG_TITLE = False

# Bar style
BAR_COLOR = "#2b8cbe"
BAR_EDGE_COLOR = "black"
BAR_ALPHA = 0.92
ERROR_COLOR = "black"
LINE_COLOR = "#d95f02"



# 4) Matplotlib style

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "ytick.minor.width": 0.5,
    "savefig.bbox": "tight",
})



# 5) Open dataset

ds = xr.open_dataset(BIO_FILE)

print("Dataset opened successfully:")
print(ds)



# 6) Required variables
required_vars = [
    "occurrence_count",
    "species_richness",
    "shannon_index",
    "simpson_index",
    "pielou_evenness"
]

for v in required_vars:
    if v not in ds.data_vars:
        raise ValueError(f"Variable not found in dataset: {v}")

for d in ["year", "month", "lat", "lon"]:
    if d not in ds.dims:
        raise ValueError(f"Required dimension not found in dataset: {d}")



# 7) Helper functions

def area_weighted_mean(da):
    """
    Area-weighted spatial mean over lat-lon.
    Keeps year-month dimensions.
    """
    weights = np.cos(np.deg2rad(da["lat"]))
    valid = da.notnull()

    weighted_sum = (da * weights).where(valid).sum(
        dim=("lat", "lon"),
        skipna=True
    )

    weight_sum = weights.where(valid).sum(
        dim=("lat", "lon"),
        skipna=True
    )

    return weighted_sum / weight_sum


def valid_count(da):
    """
    Count valid 1-degree grid cells per year-month.
    """
    return da.notnull().sum(dim=("lat", "lon"))


def ci95_from_samples(values):
    """
    Calculate mean and 95% confidence interval from valid seasonal samples.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n == 0:
        return np.nan, np.nan, 0

    mean = np.nanmean(values)

    if n == 1:
        return mean, np.nan, n

    se = np.nanstd(values, ddof=1) / np.sqrt(n)
    ci95 = 1.96 * se

    return mean, ci95, n



# 8) Build monthly dataframe

years = ds["year"].values
months = ds["month"].values

time_index = []

for y in years:
    for m in months:
        time_index.append(pd.Timestamp(year=int(y), month=int(m), day=1))

time_index = pd.DatetimeIndex(time_index)

records = {}

for var in required_vars:

    da = ds[var]

    mean_da = area_weighted_mean(da)
    n_da = valid_count(da)

    mean_1d = mean_da.values.reshape(-1)
    n_1d = n_da.values.reshape(-1)

    # Mask months with insufficient valid grid-cell support
    mean_1d = np.where(n_1d >= MIN_VALID_CELLS, mean_1d, np.nan)

    records[f"{var}_mean"] = mean_1d
    records[f"{var}_ncell"] = n_1d

df = pd.DataFrame(records, index=time_index)
df.index.name = "time"

df = df[
    (df.index.year >= START_YEAR) &
    (df.index.year <= END_YEAR)
].copy()

df["year"] = df.index.year
df["month"] = df.index.month



# 9) Define austral seasons

def assign_austral_season(month):
    """
    Austral seasonal grouping:
    Spring = SON
    Summer = DJF
    Autumn = MAM
    Winter = JJA
    """
    if month in [9, 10, 11]:
        return "Spring"
    elif month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"

df["season"] = df["month"].apply(assign_austral_season)

season_order = ["Spring", "Summer", "Autumn", "Winter"]



# 10) Seasonal climatology statistics

seasonal_records = []

for var in required_vars:

    for season in season_order:

        sub = df[df["season"] == season][f"{var}_mean"].values
        mean, ci95, n = ci95_from_samples(sub)

        ncell_col = f"{var}_ncell"
        mean_ncell = np.nanmean(df[df["season"] == season][ncell_col].values)

        seasonal_records.append({
            "variable": var,
            "season": season,
            "mean": mean,
            "ci95": ci95,
            "n_month_samples": n,
            "mean_valid_grid_cells": mean_ncell
        })

seasonal_df = pd.DataFrame(seasonal_records)
seasonal_df.to_csv(OUT_CSV, index=False)

print("\nSaved seasonal climatology CSV:")
print(OUT_CSV)

print("\nSeasonal climatology table:")
print(seasonal_df)



# 11) Coverage statistics for panel f

coverage_records = []

for season in season_order:

    # Use occurrence_count valid grid cells as sampling-support coverage
    ncell_values = df[df["season"] == season]["occurrence_count_ncell"].values
    mean_ncell, ci95_ncell, n = ci95_from_samples(ncell_values)

    coverage_records.append({
        "season": season,
        "mean_valid_grid_cells": mean_ncell,
        "ci95_valid_grid_cells": ci95_ncell,
        "n_month_samples": n
    })

coverage_df = pd.DataFrame(coverage_records)

print("\nSeasonal valid-cell coverage:")
print(coverage_df)



# 12) Plot information
plot_info = {
    "occurrence_count": {
        "panel": "(a)",
        "title": "Occurrence count",
        "ylabel": "Occurrence count",
        "ylim": None
    },
    "species_richness": {
        "panel": "(b)",
        "title": "Species richness",
        "ylabel": "Species richness",
        "ylim": None
    },
    "shannon_index": {
        "panel": "(c)",
        "title": "Shannon diversity",
        "ylabel": "Shannon index",
        "ylim": None
    },
    "simpson_index": {
        "panel": "(d)",
        "title": "Simpson diversity",
        "ylabel": "Simpson index",
        "ylim": (0, 1.05)
    },
    "pielou_evenness": {
        "panel": "(e)",
        "title": "Pielou evenness",
        "ylabel": "Pielou evenness",
        "ylim": (0, 1.05)
    }
}



# 13) Create figure

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(12.2, 7.2)
)

axes = axes.flatten()

x = np.arange(len(season_order))



# 14) Plot panels a–e

for i, var in enumerate(required_vars):

    ax = axes[i]

    sub = seasonal_df[seasonal_df["variable"] == var].set_index("season").loc[season_order]

    means = sub["mean"].values
    ci95 = sub["ci95"].values
    n_samples = sub["n_month_samples"].values

    ax.bar(
        x,
        means,
        yerr=ci95,
        capsize=4,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.65,
        alpha=BAR_ALPHA,
        error_kw={
            "elinewidth": 1.0,
            "ecolor": ERROR_COLOR,
            "capthick": 1.0
        },
        zorder=3
    )

    ax.plot(
        x,
        means,
        color=LINE_COLOR,
        marker="o",
        markersize=4.2,
        linewidth=1.4,
        zorder=4
    )

    # Add sample size above each bar
    for xi, yi, ci, ni in zip(x, means, ci95, n_samples):
        if np.isfinite(yi):
            offset = 0 if not np.isfinite(ci) else ci
            ax.text(
                xi,
                yi + offset + (np.nanmax(means + np.nan_to_num(ci95, nan=0)) * 0.035),
                f"n={int(ni)}",
                ha="center",
                va="bottom",
                fontsize=7.2
            )

    ax.set_xticks(x)
    ax.set_xticklabels(season_order)
    ax.set_ylabel(plot_info[var]["ylabel"])

    ax.set_title(
        plot_info[var]["title"],
        fontsize=10,
        fontweight="bold",
        pad=8
    )

    ax.text(
        0.02,
        0.92,
        plot_info[var]["panel"],
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center"
    )

    ax.grid(
        True,
        axis="y",
        linewidth=0.35,
        alpha=0.32,
        zorder=0
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if plot_info[var]["ylim"] is not None:
        ax.set_ylim(plot_info[var]["ylim"])
    else:
        upper = np.nanmax(means + np.nan_to_num(ci95, nan=0))
        ax.set_ylim(0, upper * 1.22)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3.2,
        width=0.7
    )



# 15) Panel f — valid grid-cell coverage
ax = axes[5]

coverage_means = coverage_df["mean_valid_grid_cells"].values
coverage_ci95 = coverage_df["ci95_valid_grid_cells"].values
coverage_n = coverage_df["n_month_samples"].values

ax.bar(
    x,
    coverage_means,
    yerr=coverage_ci95,
    capsize=4,
    color="#74a9cf",
    edgecolor="black",
    linewidth=0.65,
    alpha=0.92,
    error_kw={
        "elinewidth": 1.0,
        "ecolor": "black",
        "capthick": 1.0
    },
    zorder=3
)

ax.plot(
    x,
    coverage_means,
    color="#d95f02",
    marker="o",
    markersize=4.2,
    linewidth=1.4,
    zorder=4
)

for xi, yi, ci, ni in zip(x, coverage_means, coverage_ci95, coverage_n):
    if np.isfinite(yi):
        offset = 0 if not np.isfinite(ci) else ci
        ax.text(
            xi,
            yi + offset + (np.nanmax(coverage_means + np.nan_to_num(coverage_ci95, nan=0)) * 0.035),
            f"n={int(ni)}",
            ha="center",
            va="bottom",
            fontsize=7.2
        )

ax.set_xticks(x)
ax.set_xticklabels(season_order)
ax.set_ylabel("Valid grid cells")
ax.set_title(
    "Occurrence-supported coverage",
    fontsize=10,
    fontweight="bold",
    pad=8
)

ax.text(
    0.02,
    0.92,
    "(f)",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
    ha="left",
    va="center"
)

ax.grid(
    True,
    axis="y",
    linewidth=0.35,
    alpha=0.32,
    zorder=0
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

upper = np.nanmax(coverage_means + np.nan_to_num(coverage_ci95, nan=0))
ax.set_ylim(0, upper * 1.22)

ax.tick_params(
    axis="both",
    direction="out",
    length=3.2,
    width=0.7
)



# 16) Optional figure title

if ADD_FIG_TITLE:
    fig.suptitle(
        "Seasonal climatology of occurrence-based biodiversity indicators",
        fontsize=13,
        fontweight="bold",
        y=0.985
    )


# 18) Final layout and save
plt.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.93 if ADD_FIG_TITLE else 0.965,
    bottom=0.095,
    wspace=0.28,
    hspace=0.38
)

plt.savefig(
    OUT_FIG,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nSaved figure:")
print(OUT_FIG)
