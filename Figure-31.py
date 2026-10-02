# Fig. 4 — Sea-wise biodiversity comparison
# Seasonal ecosystem variability and regional biodiversity contrasts
# Antarctic Shelf Seas, 2008–2025
#
# Final thesis / journal-ready version
# 1080 dpi
#
# Panels:
# (a) Grouped seasonal bar plot — Shannon diversity
# (b) Grouped seasonal bar plot — Species richness
# (c) Horizontal bar plot with 95% CI — Regional biodiversity contrast
# (d) Stacked seasonal coverage bar plot — GBIF occurrence coverage


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
    "Fig29_seawise_biodiversity_comparison_2008_2025_1080dpi.png"
)

OUT_MONTHLY_CSV = os.path.join(
    OUT_DIR,
    "Fig29_seawise_monthly_biodiversity_2008_2025.csv"
)

OUT_SEASONAL_CSV = os.path.join(
    OUT_DIR,
    "Fig29_seawise_seasonal_biodiversity_2008_2025.csv"
)

OUT_OVERALL_CSV = os.path.join(
    OUT_DIR,
    "Fig29_seawise_overall_biodiversity_2008_2025.csv"
)



# 3) Main settings

START_YEAR = 2008
END_YEAR   = 2025
SAVE_DPI   = 1080

# Keep low because GBIF sea-wise occurrence coverage is sparse.
# Panel (d) should always be used to judge sampling support.
MIN_VALID_CELLS = 1

ADD_FIG_TITLE = False

SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]

SEASON_COLORS = {
    "Spring": "#2ca25f",
    "Summer": "#f16913",
    "Autumn": "#756bb1",
    "Winter": "#3182bd"
}

BAR_COLOR = "#4c97c2"
BAR_EDGE_COLOR = "black"
ERROR_COLOR = "black"
STACK_ALPHA = 0.90



# 4) Sea information

sea_info = [
    ("WED", "Weddell Sea", "60°W–20°W", -60, -20),
    ("KHV", "King Haakon VII Sea", "20°W–0°", -20, 0),
    ("RLS", "Riiser-Larsen Sea", "0°–10°E", 0, 10),
    ("LAZ", "Lazarev Sea", "10°E–30°E", 10, 30),
    ("COS", "Cosmonauts Sea", "30°E–50°E", 30, 50),
    ("COO", "Cooperation Sea", "50°E–70°E", 50, 70),
    ("DAV", "Davis Sea", "70°E–90°E", 70, 90),
    ("MAW", "Mawson Sea", "90°E–130°E", 90, 130),
    ("DUR", "D'Urville Sea", "130°E–150°E", 130, 150),
    ("SOM", "Somov Sea", "150°E–170°E", 150, 170),
    ("ROS", "Ross Sea", "170°E–130°W", 170, -130),
    ("AMU", "Amundsen Sea", "130°W–100°W", -130, -100),
    ("BEL", "Bellingshausen Sea", "100°W–60°W", -100, -60),
]



# 5) Matplotlib style

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "savefig.bbox": "tight",
})



# 6) Open dataset

ds = xr.open_dataset(BIO_FILE)

print("Dataset opened successfully:")
print(ds)



# 7) Required variables and dimensions

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



# 8) Longitude normalization

def normalize_lon_180(ds):
    """
    Convert longitude to -180 to 180 if needed.
    """
    if float(ds["lon"].max()) > 180:
        new_lon = (((ds["lon"] + 180) % 360) - 180)
        ds = ds.assign_coords(lon=new_lon)
        ds = ds.sortby("lon")
    return ds

ds = normalize_lon_180(ds)



# 9) Helper functions

def lon_mask(lon, lon_min, lon_max):
    """
    Longitude mask including dateline-crossing sectors.
    """
    if lon_min <= lon_max:
        return (lon >= lon_min) & (lon <= lon_max)
    else:
        return (lon >= lon_min) | (lon <= lon_max)


def assign_austral_season(month):
    """
    Austral seasons:
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


def ci95_from_samples(values):
    """
    Mean and approximate 95% confidence interval from valid samples.
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


def area_weighted_sea_monthly_mean(da, lon_min, lon_max):
    """
    Area-weighted sea-wise monthly mean.
    Keeps year and month dimensions.
    """
    lon = da["lon"]
    lat = da["lat"]

    mask_lon = lon_mask(lon, lon_min, lon_max)
    mask_lat = (lat >= -90) & (lat <= -60)

    sub = da.where(mask_lon & mask_lat, drop=True)

    weights = np.cos(np.deg2rad(sub["lat"]))
    valid = sub.notnull()

    weighted_sum = (sub * weights).where(valid).sum(
        dim=("lat", "lon"),
        skipna=True
    )

    weight_sum = weights.where(valid).sum(
        dim=("lat", "lon"),
        skipna=True
    )

    mean = weighted_sum / weight_sum
    ncell = valid.sum(dim=("lat", "lon"))

    return mean, ncell


def clean_axis_ygrid(ax):
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.30, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3.0, width=0.7)


def clean_axis_xgrid(ax):
    ax.grid(True, axis="x", linewidth=0.35, alpha=0.30, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3.0, width=0.7)


def add_panel_label(ax, label):
    """
    Add panel label slightly above/outside each subplot.
    """
    ax.text(
        -0.02,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False
    )



# 10) Build sea-wise monthly dataframe

time_index = []

for y in ds["year"].values:
    for m in ds["month"].values:
        time_index.append(pd.Timestamp(year=int(y), month=int(m), day=1))

time_index = pd.DatetimeIndex(time_index)

monthly_records = []

for sea_short, sea_full, lon_label, lon_min, lon_max in sea_info:

    temp = pd.DataFrame(index=time_index)
    temp["sea"] = sea_short
    temp["sea_full"] = sea_full
    temp["lon_label"] = lon_label
    temp["year"] = temp.index.year
    temp["month"] = temp.index.month
    temp["season"] = temp["month"].apply(assign_austral_season)

    for var in required_vars:

        mean_da, ncell_da = area_weighted_sea_monthly_mean(
            ds[var],
            lon_min,
            lon_max
        )

        mean_1d = mean_da.values.reshape(-1)
        ncell_1d = ncell_da.values.reshape(-1)

        mean_1d = np.where(
            ncell_1d >= MIN_VALID_CELLS,
            mean_1d,
            np.nan
        )

        temp[f"{var}_mean"] = mean_1d
        temp[f"{var}_ncell"] = ncell_1d

    monthly_records.append(temp)

monthly_df = pd.concat(monthly_records).reset_index().rename(columns={"index": "time"})

monthly_df = monthly_df[
    (monthly_df["year"] >= START_YEAR) &
    (monthly_df["year"] <= END_YEAR)
].copy()

monthly_df.to_csv(OUT_MONTHLY_CSV, index=False)

print("\nSaved monthly sea-wise table:")
print(OUT_MONTHLY_CSV)



# 11) Seasonal sea-wise statistics

seasonal_records = []

for sea_short, sea_full, lon_label, lon_min, lon_max in sea_info:

    sea_sub = monthly_df[monthly_df["sea"] == sea_short]

    for season in SEASON_ORDER:

        season_sub = sea_sub[sea_sub["season"] == season]

        row = {
            "sea": sea_short,
            "sea_full": sea_full,
            "season": season
        }

        for var in required_vars:

            values = season_sub[f"{var}_mean"].values
            mean, ci95, n = ci95_from_samples(values)

            row[f"{var}_season_mean"] = mean
            row[f"{var}_season_ci95"] = ci95
            row[f"{var}_season_n"] = n

        coverage_values = season_sub["occurrence_count_ncell"].values
        coverage_mean, coverage_ci95, coverage_n = ci95_from_samples(coverage_values)

        row["coverage_mean_valid_cells"] = coverage_mean
        row["coverage_ci95_valid_cells"] = coverage_ci95
        row["coverage_n"] = coverage_n

        seasonal_records.append(row)

seasonal_df = pd.DataFrame(seasonal_records)
seasonal_df.to_csv(OUT_SEASONAL_CSV, index=False)

print("\nSaved seasonal sea-wise table:")
print(OUT_SEASONAL_CSV)



# 12) Overall sea-wise statistics

overall_records = []

for sea_short, sea_full, lon_label, lon_min, lon_max in sea_info:

    sea_sub = monthly_df[monthly_df["sea"] == sea_short]

    row = {
        "sea": sea_short,
        "sea_full": sea_full,
        "lon_label": lon_label
    }

    for var in required_vars:

        values = sea_sub[f"{var}_mean"].values
        mean, ci95, n = ci95_from_samples(values)

        row[f"{var}_mean"] = mean
        row[f"{var}_ci95"] = ci95
        row[f"{var}_n"] = n

    coverage_values = sea_sub["occurrence_count_ncell"].values
    coverage_mean, coverage_ci95, coverage_n = ci95_from_samples(coverage_values)

    row["coverage_mean_valid_cells"] = coverage_mean
    row["coverage_ci95_valid_cells"] = coverage_ci95
    row["coverage_n"] = coverage_n

    overall_records.append(row)

overall_df = pd.DataFrame(overall_records)
overall_df.to_csv(OUT_OVERALL_CSV, index=False)

print("\nSaved overall sea-wise table:")
print(OUT_OVERALL_CSV)



# 13) Sort seas by mean Shannon diversity

overall_sorted = overall_df.sort_values(
    "shannon_index_mean",
    ascending=False
).reset_index(drop=True)

SEA_ORDER_SORTED = overall_sorted["sea"].tolist()



# 14) Prepare seasonal pivot tables

shannon_plot = (
    seasonal_df
    .pivot(index="sea", columns="season", values="shannon_index_season_mean")
    .reindex(SEA_ORDER_SORTED)
    [SEASON_ORDER]
)

richness_plot = (
    seasonal_df
    .pivot(index="sea", columns="season", values="species_richness_season_mean")
    .reindex(SEA_ORDER_SORTED)
    [SEASON_ORDER]
)

coverage_plot = (
    seasonal_df
    .pivot(index="sea", columns="season", values="coverage_mean_valid_cells")
    .reindex(SEA_ORDER_SORTED)
    [SEASON_ORDER]
)



# 15) Create figure

fig = plt.figure(figsize=(13.4, 8.6))

gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    width_ratios=[1.05, 1.15],
    height_ratios=[1.0, 1.0],
    wspace=0.28,
    hspace=0.40
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

x = np.arange(len(SEA_ORDER_SORTED))
width = 0.18

offsets = {
    "Spring": -1.5 * width,
    "Summer": -0.5 * width,
    "Autumn":  0.5 * width,
    "Winter":  1.5 * width
}



# 16) Panel (a): Grouped seasonal Shannon bars

for season in SEASON_ORDER:
    ax_a.bar(
        x + offsets[season],
        shannon_plot[season].values,
        width=width,
        color=SEASON_COLORS[season],
        edgecolor="black",
        linewidth=0.35,
        alpha=0.90,
        label=season,
        zorder=3
    )

ax_a.set_xticks(x)
ax_a.set_xticklabels(SEA_ORDER_SORTED, rotation=45, ha="right")

ax_a.set_xlabel("Antarctic Shelf Seas")
ax_a.set_ylabel("Shannon index")

ax_a.set_title(
    "Seasonal Shannon diversity variability",
    fontsize=10,
    fontweight="bold",
    pad=8
)

ax_a.set_ylim(0, np.nanmax(shannon_plot.values) * 1.15)

add_panel_label(ax_a, "(a)")
clean_axis_ygrid(ax_a)

ax_a.legend(
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.18),
    handlelength=1.5,
    columnspacing=1.1
)



# 17) Panel (b): Grouped seasonal richness bars

for season in SEASON_ORDER:
    ax_b.bar(
        x + offsets[season],
        richness_plot[season].values,
        width=width,
        color=SEASON_COLORS[season],
        edgecolor="black",
        linewidth=0.35,
        alpha=0.90,
        label=season,
        zorder=3
    )

ax_b.set_xticks(x)
ax_b.set_xticklabels(SEA_ORDER_SORTED, rotation=45, ha="right")

ax_b.set_xlabel("Antarctic Shelf Seas")
ax_b.set_ylabel("Species richness")

ax_b.set_title(
    "Seasonal species richness variability",
    fontsize=10,
    fontweight="bold",
    pad=8
)

ax_b.set_ylim(0, np.nanmax(richness_plot.values) * 1.15)

add_panel_label(ax_b, "(b)")
clean_axis_ygrid(ax_b)



# 18) Panel (c): Regional biodiversity contrast

ypos = np.arange(len(overall_sorted))

ax_c.barh(
    ypos,
    overall_sorted["shannon_index_mean"].values,
    xerr=overall_sorted["shannon_index_ci95"].values,
    color=BAR_COLOR,
    edgecolor=BAR_EDGE_COLOR,
    linewidth=0.65,
    alpha=0.92,
    error_kw={
        "elinewidth": 1.0,
        "ecolor": ERROR_COLOR,
        "capthick": 1.0
    },
    zorder=3
)

ax_c.set_yticks(ypos)
ax_c.set_yticklabels(overall_sorted["sea"].values)
ax_c.invert_yaxis()

ax_c.set_xlabel("Mean Shannon diversity index")
ax_c.set_ylabel("Antarctic Shelf Seas")

ax_c.set_title(
    "Regional biodiversity contrast",
    fontsize=10,
    fontweight="bold",
    pad=8
)

add_panel_label(ax_c, "(c)")
clean_axis_xgrid(ax_c)



# 19) Panel (d): Stacked seasonal coverage

bottom = np.zeros(len(SEA_ORDER_SORTED))

for season in SEASON_ORDER:

    vals = coverage_plot[season].values

    ax_d.bar(
        x,
        vals,
        bottom=bottom,
        color=SEASON_COLORS[season],
        edgecolor="white",
        linewidth=0.5,
        alpha=STACK_ALPHA,
        label=season,
        zorder=3
    )

    bottom = bottom + np.nan_to_num(vals, nan=0.0)

ax_d.set_xticks(x)
ax_d.set_xticklabels(SEA_ORDER_SORTED, rotation=45, ha="right")

ax_d.set_xlabel("Antarctic Shelf Seas")
ax_d.set_ylabel("Mean valid 1° grid cells")

ax_d.set_title(
    "Occurrence-supported seasonal coverage",
    fontsize=10,
    fontweight="bold",
    pad=8
)

add_panel_label(ax_d, "(d)")
clean_axis_ygrid(ax_d)

ax_d.legend(
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.18),
    handlelength=1.5,
    columnspacing=1.1
)



# 20) Optional figure title

if ADD_FIG_TITLE:
    fig.suptitle(
        "Sea-wise biodiversity comparison across Antarctic Shelf Seas",
        fontsize=13,
        fontweight="bold",
        y=0.985
    )




# 22) Final layout and save

plt.subplots_adjust(
    left=0.08,
    right=0.97,
    top=0.935 if ADD_FIG_TITLE else 0.945,
    bottom=0.11,
    wspace=0.28,
    hspace=0.40
)

plt.savefig(
    OUT_FIG,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nSaved Fig. 29:")
print(OUT_FIG)

print("\nSaved monthly table:")
print(OUT_MONTHLY_CSV)

print("\nSaved seasonal table:")
print(OUT_SEASONAL_CSV)

print("\nSaved overall table:")
print(OUT_OVERALL_CSV)
