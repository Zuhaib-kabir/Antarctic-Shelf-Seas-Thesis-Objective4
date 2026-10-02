# Fig. 6 — Seasonal biodiversity–oceanographic overlay

# MOUNT DRIVE 
from google.colab import drive
drive.mount('/content/drive')


# 0) Install packages
!pip -q install xarray netCDF4 h5netcdf dask cartopy scipy


# 1) Imports
import os
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# 2) File paths
CHL_FILE = "/content/drive/MyDrive/SAM_Thesis/Data/CHL_monthly_2008_2025_SO.nc"

SIC_FILE = "/content/drive/MyDrive/SAM_Thesis/Data/OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc"

BIO_FILE = "/content/drive/MyDrive/SAM_Thesis/Data/Biodiversity_Indices/Biodiversity_indices_seasonal_ecosystem_coupling_2008_2025_1deg_masked_min10.nc"

OUT_DIR = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG = os.path.join(
    OUT_DIR,
    "Fig31_seasonal_biodiversity_oceanographic_overlay_FINAL_FIXED_1080dpi.png"
)



# 3) Main settings
SAVE_DPI = 1080

LAT_MIN = -90
LAT_MAX = -60

SEASONS = {
    "Spring": [9, 10, 11],
    "Summer": [12, 1, 2],
    "Early autumn": [3, 4],
}

PANEL_LABELS = ["(a)", "(b)", "(c)"]

# Display smoothing only
CHL_COARSEN = 6
SIC_COARSEN = 5
CHL_SMOOTH_SIGMA = 0.8
SIC_SMOOTH_SIGMA = 1.1

# Chlorophyll-a background
CHL_VMIN = 0.03
CHL_VMAX = 2.0
CHL_CMAP = "turbo"

# SIC contours
SIC_LEVELS = [15, 50, 80]
SIC_LINESTYLES = ["--", "-", ":"]
SIC_LINEWIDTHS = [1.30, 1.55, 1.75]

# Biodiversity filtering
MIN_BIO_VALID_OCCURRENCE = 10
MIN_RICHNESS_TO_PLOT = 1

MAX_RICHNESS_POINTS_PER_PANEL = 135
MAX_SHANNON_POINTS_PER_PANEL = 65

# Species richness squares
RICHNESS_SQUARE_SIZE = 31
RICHNESS_EDGEWIDTH = 1.35

# Shannon circles
SHANNON_CIRCLE_SIZE = 40
SHANNON_EDGEWIDTH = 1.05

# Species richness colors
RICHNESS_BINS = [0, 2, 6, 10, np.inf]
RICHNESS_COLORS = ["#ffffff", "#ffd92f", "#ff9f1c", "#e31a1c"]
RICHNESS_LABELS = ["0–2", "3–6", "7–10", "10+"]

# Shannon colors
SHANNON_BINS = [0, 1, 2, 3, np.inf]
SHANNON_COLORS = ["#ffffff", "#f4a6c6", "#c51b8a", "#4a1486"]

# Short labels prevent text going outside legend box
SHANNON_LABELS = [
    "Low: 0–1",
    "Moderate: 1–2",
    "High: 2–3",
    "Very high: >3",
]


# 4) Matplotlib style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


# 5) Helper functions
def check_file(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found:\n{path}")
    print(f"{name} found:\n{path}")


def standardize_lat_lon(ds):
    rename = {}

    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"

    if rename:
        ds = ds.rename(rename)

    return ds


def normalize_lon_180(ds):
    if "lon" not in ds.coords:
        raise ValueError("No longitude coordinate found.")

    if float(ds["lon"].max()) > 180:
        lon_new = (((ds["lon"] + 180) % 360) - 180)
        ds = ds.assign_coords(lon=lon_new)
        ds = ds.sortby("lon")

    return ds


def detect_time_name(ds):
    for t in ["time", "date", "valid_time"]:
        if t in ds.coords or t in ds.dims:
            return t
    return None


def add_month_coordinate_from_time(ds):
    tname = detect_time_name(ds)

    if tname is None:
        raise ValueError("No time coordinate found.")

    if tname != "time":
        ds = ds.rename({tname: "time"})

    ds["time"] = pd.to_datetime(ds["time"].values)
    ds = ds.assign_coords(month=("time", pd.to_datetime(ds["time"].values).month))

    return ds


def detect_variable(ds, candidates, label):
    for c in candidates:
        if c in ds.data_vars:
            return c

    for v in ds.data_vars:
        dims = ds[v].dims
        if ("lat" in dims) and ("lon" in dims):
            return v

    raise ValueError(
        f"Could not detect {label} variable.\n"
        f"Available variables: {list(ds.data_vars)}"
    )


def subset_lat(da):
    if da["lat"][0] < da["lat"][-1]:
        return da.sel(lat=slice(LAT_MIN, LAT_MAX))
    else:
        return da.sel(lat=slice(LAT_MAX, LAT_MIN))


def coarsen_latlon(da, factor):
    if factor is None or factor <= 1:
        return da

    return da.coarsen(
        lat=factor,
        lon=factor,
        boundary="trim"
    ).mean(skipna=True)


def seasonal_mean_time_dataset(da, months):
    da = da.where(da["month"].isin(months), drop=True)
    da = da.mean(dim="time", skipna=True)
    da = subset_lat(da)
    return da


def seasonal_mean_year_month_dataset(da, months):
    da = da.where(da["month"].isin(months), drop=True)

    dims_to_mean = []
    for d in ["year", "month", "time"]:
        if d in da.dims:
            dims_to_mean.append(d)

    da = da.mean(dim=dims_to_mean, skipna=True)
    da = subset_lat(da)

    return da


def convert_sic_to_percent(da):
    sample = float(da.mean(skipna=True).compute().values)

    if sample <= 1.5:
        print("SIC/SIF appears to be fraction 0–1. Converting to percent.")
        return da * 100.0
    else:
        print("SIC/SIF appears to already be in percent.")
        return da


def clean_chl(da):
    da = da.where(np.isfinite(da))
    da = da.where(da > 0)
    return da


def nan_gaussian_smooth_2d(arr, sigma=1.0, min_weight=0.05):
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)

    filled = np.where(valid, arr, 0.0)
    weights = valid.astype(float)

    smooth_data = gaussian_filter(filled, sigma=sigma, mode="nearest")
    smooth_w = gaussian_filter(weights, sigma=sigma, mode="nearest")

    out = smooth_data / np.where(smooth_w == 0, np.nan, smooth_w)
    out[smooth_w < min_weight] = np.nan

    return out


def make_smoothed_da(da, sigma):
    smoothed = nan_gaussian_smooth_2d(da.values, sigma=sigma)

    return xr.DataArray(
        smoothed,
        coords={"lat": da["lat"].values, "lon": da["lon"].values},
        dims=("lat", "lon")
    )


def circular_boundary(ax):
    theta = np.linspace(0, 2 * np.pi, 256)
    center = [0.5, 0.5]
    radius = 0.5

    verts = np.vstack([
        np.sin(theta) * radius + center[0],
        np.cos(theta) * radius + center[1]
    ]).T

    circle = mpath.Path(verts)
    ax.set_boundary(circle, transform=ax.transAxes)


def add_map_features(ax):
    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#f4f6f7",
        zorder=0
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="0.86",
        edgecolor="0.25",
        linewidth=0.45,
        zorder=6
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.45,
        edgecolor="0.25",
        zorder=7
    )

    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.32,
        color="0.45",
        alpha=0.42,
        linestyle=":"
    )

    circular_boundary(ax)


def add_lon_labels(ax):
    lon_labels = [
        (-30, "30°W"),
        (30, "30°E"),
        (-60, "60°W"),
        (60, "60°E"),
        (-90, "90°W"),
        (90, "90°E"),
        (-120, "120°W"),
        (120, "120°E"),
        (-150, "150°W"),
        (150, "150°E"),
        (180, "180°"),
    ]

    for lo, lab in lon_labels:
        ax.text(
            lo,
            -58.75,
            lab,
            transform=ccrs.PlateCarree(),
            fontsize=8.8,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            zorder=40,
            clip_on=False
        )


def classify_colors(values, bins, colors):
    values = np.asarray(values)
    out = np.full(values.shape, colors[0], dtype=object)

    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]

        if i == 0:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values > lo) & (values <= hi)

        out[mask] = colors[i]

    return out


def prepare_biodiversity_points(shannon_da, richness_da, occ_da):
    sh = shannon_da.values
    ri = richness_da.values
    oc = occ_da.values

    lon2d, lat2d = np.meshgrid(
        shannon_da["lon"].values,
        shannon_da["lat"].values
    )

    mask = (
        np.isfinite(sh) &
        np.isfinite(ri) &
        np.isfinite(oc) &
        (oc >= MIN_BIO_VALID_OCCURRENCE) &
        (ri >= MIN_RICHNESS_TO_PLOT)
    )

    lons = lon2d[mask]
    lats = lat2d[mask]
    shvals = sh[mask]
    rivals = ri[mask]
    ocvals = oc[mask]

    if len(lons) == 0:
        return None

    score = (
        0.50 * np.nan_to_num(ocvals, nan=0)
        + 2.4 * np.nan_to_num(rivals, nan=0)
        + 6.0 * np.nan_to_num(shvals, nan=0)
    )

    idx_sorted = np.argsort(score)[::-1]

    idx_rich = idx_sorted[:min(MAX_RICHNESS_POINTS_PER_PANEL, len(idx_sorted))]
    idx_shan = idx_sorted[:min(MAX_SHANNON_POINTS_PER_PANEL, len(idx_sorted))]

    return (
        lons[idx_rich], lats[idx_rich], shvals[idx_rich], rivals[idx_rich], ocvals[idx_rich],
        lons[idx_shan], lats[idx_shan], shvals[idx_shan], rivals[idx_shan], ocvals[idx_shan]
    )


def add_sic_contours_high_contrast(ax, sic_plot, pc):
    try:
        ax.contour(
            sic_plot["lon"],
            sic_plot["lat"],
            sic_plot,
            levels=SIC_LEVELS,
            colors="white",
            linewidths=[lw + 2.2 for lw in SIC_LINEWIDTHS],
            linestyles=SIC_LINESTYLES,
            transform=pc,
            zorder=11
        )

        cs = ax.contour(
            sic_plot["lon"],
            sic_plot["lat"],
            sic_plot,
            levels=SIC_LEVELS,
            colors="black",
            linewidths=SIC_LINEWIDTHS,
            linestyles=SIC_LINESTYLES,
            transform=pc,
            zorder=12
        )

        labels = ax.clabel(
            cs,
            fmt=lambda x: f"{int(x)}%",
            fontsize=7.2,
            inline=True,
            inline_spacing=4,
            colors="black"
        )

        for txt in labels:
            txt.set_fontweight("bold")
            txt.set_bbox(
                dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.80,
                    pad=0.10
                )
            )

    except Exception as e:
        print(f"Warning: SIC contours failed: {e}")


# 6) Check files
check_file(CHL_FILE, "Chlorophyll-a NetCDF")
check_file(SIC_FILE, "OSTIA sea-ice fraction NetCDF")
check_file(BIO_FILE, "Biodiversity NetCDF")



# 7) Open datasets
print("\nOpening datasets...")

chl_ds = xr.open_dataset(CHL_FILE, chunks={})
sic_ds = xr.open_dataset(SIC_FILE, chunks={})
bio_ds = xr.open_dataset(BIO_FILE, chunks={})

chl_ds = normalize_lon_180(standardize_lat_lon(chl_ds))
sic_ds = normalize_lon_180(standardize_lat_lon(sic_ds))
bio_ds = normalize_lon_180(standardize_lat_lon(bio_ds))

chl_var = detect_variable(
    chl_ds,
    ["CHL", "chl", "chlor_a", "chlorophyll", "chl_a", "CHL_mean"],
    "chlorophyll-a"
)

sic_var = detect_variable(
    sic_ds,
    ["SIC", "sic", "SIF", "sif", "sea_ice_fraction", "ice_fraction", "sea_ice_concentration"],
    "SIC/SIF"
)

print("Detected CHL variable:", chl_var)
print("Detected SIC/SIF variable:", sic_var)

chl_ds = add_month_coordinate_from_time(chl_ds)
sic_ds = add_month_coordinate_from_time(sic_ds)

chl_da = clean_chl(chl_ds[chl_var])
sic_da = convert_sic_to_percent(sic_ds[sic_var])

required_bio_vars = ["shannon_index", "species_richness", "occurrence_count"]

for v in required_bio_vars:
    if v not in bio_ds.data_vars:
        raise ValueError(f"Missing biodiversity variable: {v}")

shannon_da = bio_ds["shannon_index"]
richness_da = bio_ds["species_richness"]
occ_da = bio_ds["occurrence_count"]


# 8) Create figure
proj = ccrs.SouthPolarStereo()
pc = ccrs.PlateCarree()

fig = plt.figure(figsize=(9.9, 13.8))

gs = GridSpec(
    nrows=3,
    ncols=2,
    width_ratios=[1.00, 0.305],
    height_ratios=[1, 1, 1],
    wspace=-0.120,
    hspace=0.115
)

legend_ax = fig.add_subplot(gs[:, 1])
legend_ax.axis("off")

chl_norm = mcolors.LogNorm(vmin=CHL_VMIN, vmax=CHL_VMAX)
chl_cmap = plt.get_cmap(CHL_CMAP).copy()
chl_cmap.set_bad("#f4f6f7")

mappable_chl = None

for i, (season_name, months) in enumerate(SEASONS.items()):

    print(f"\nProcessing {season_name}: months {months}")

    ax = fig.add_subplot(gs[i, 0], projection=proj)

    chl_season = seasonal_mean_time_dataset(chl_da, months)
    sic_season = seasonal_mean_time_dataset(sic_da, months)

    chl_plot = coarsen_latlon(chl_season, CHL_COARSEN).compute()
    sic_plot = coarsen_latlon(sic_season, SIC_COARSEN).compute()

    chl_plot = make_smoothed_da(chl_plot, CHL_SMOOTH_SIGMA)
    sic_plot = make_smoothed_da(sic_plot, SIC_SMOOTH_SIGMA)

    shannon_season = seasonal_mean_year_month_dataset(shannon_da, months).compute()
    richness_season = seasonal_mean_year_month_dataset(richness_da, months).compute()
    occ_season = seasonal_mean_year_month_dataset(occ_da, months).compute()

    add_map_features(ax)

    mappable_chl = ax.pcolormesh(
        chl_plot["lon"],
        chl_plot["lat"],
        chl_plot,
        transform=pc,
        cmap=chl_cmap,
        norm=chl_norm,
        shading="auto",
        zorder=1,
        rasterized=True
    )

    add_sic_contours_high_contrast(ax, sic_plot, pc)

    out = prepare_biodiversity_points(
        shannon_season,
        richness_season,
        occ_season
    )

    if out is not None:

        (
            lons_r, lats_r, sh_r, rich_r, occ_r,
            lons_s, lats_s, sh_s, rich_s, occ_s
        ) = out

        rich_colors = classify_colors(rich_r, RICHNESS_BINS, RICHNESS_COLORS)
        shannon_colors = classify_colors(sh_s, SHANNON_BINS, SHANNON_COLORS)

        ax.scatter(
            lons_r,
            lats_r,
            s=RICHNESS_SQUARE_SIZE,
            marker="s",
            c=rich_colors,
            edgecolor="black",
            linewidth=RICHNESS_EDGEWIDTH,
            transform=pc,
            alpha=0.97,
            zorder=24
        )

        ax.scatter(
            lons_s,
            lats_s,
            s=SHANNON_CIRCLE_SIZE,
            marker="o",
            c=shannon_colors,
            edgecolor="black",
            linewidth=SHANNON_EDGEWIDTH,
            transform=pc,
            alpha=0.97,
            zorder=26
        )

    add_lon_labels(ax)

    ax.text(
        -0.080,
        0.975,
        PANEL_LABELS[i],
        transform=ax.transAxes,
        fontsize=14.0,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False
    )

    ax.text(
        0.50,
        1.025,
        season_name,
        transform=ax.transAxes,
        fontsize=14.0,
        fontweight="bold",
        ha="center",
        va="bottom",
        clip_on=False
    )

    gc.collect()


# 9) Fixed legend panel
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)

# Wider box and safer margins
box_x = 0.040
box_y = 0.070
box_w = 0.900
box_h = 0.830

box = mpatches.FancyBboxPatch(
    (box_x, box_y),
    box_w,
    box_h,
    boxstyle="round,pad=0.010",
    facecolor="white",
    edgecolor="black",
    linewidth=0.95
)
legend_ax.add_patch(box)

# Safe internal x positions
x_left = 0.120
x_right = 0.865
x_symbol = 0.170
x_text = 0.325
x_line1 = 0.145
x_line2 = 0.355

legend_ax.text(
    0.500,
    0.858,
    "LEGEND",
    ha="center",
    va="center",
    fontsize=14.0,
    fontweight="bold"
)


# Background
legend_ax.text(
    x_left,
    0.813,
    "Background",
    ha="left",
    va="top",
    fontsize=10.0,
    fontweight="bold"
)

legend_ax.text(
    x_left,
    0.786,
    "Chlorophyll-a seasonal mean",
    ha="left",
    va="top",
    fontsize=8.3
)

cax_chl = legend_ax.inset_axes([0.145, 0.744, 0.660, 0.021])

cb_chl = plt.colorbar(
    mappable_chl,
    cax=cax_chl,
    orientation="horizontal"
)

cb_chl.set_ticks([0.03, 0.1, 0.3, 1.0, 2.0])
cb_chl.set_ticklabels(["0.03", "0.1", "0.3", "1", "2"])
cb_chl.ax.tick_params(labelsize=7.6, length=2)

legend_ax.text(0.145, 0.724, "Low", fontsize=8.1, ha="left")
legend_ax.text(0.805, 0.724, "High", fontsize=8.1, ha="right")

legend_ax.plot(
    [x_left, x_right],
    [0.696, 0.696],
    color="0.60",
    lw=0.60,
    clip_on=True
)


# SIC contours
legend_ax.text(
    x_left,
    0.674,
    "Sea-ice concentration\ncontours",
    ha="left",
    va="top",
    fontsize=9.4,
    fontweight="bold",
    linespacing=0.82
)

y0 = 0.621

for lev, ls, lw in zip(SIC_LEVELS, SIC_LINESTYLES, SIC_LINEWIDTHS):

    legend_ax.plot(
        [x_line1, x_line2],
        [y0, y0],
        color="white",
        linestyle=ls,
        linewidth=lw + 2.1,
        solid_capstyle="round",
        clip_on=True
    )

    legend_ax.plot(
        [x_line1, x_line2],
        [y0, y0],
        color="black",
        linestyle=ls,
        linewidth=lw,
        solid_capstyle="round",
        clip_on=True
    )

    legend_ax.text(
        0.435,
        y0,
        f"{lev}%",
        fontsize=8.6,
        ha="left",
        va="center"
    )

    y0 -= 0.037

legend_ax.plot(
    [x_left, x_right],
    [0.498, 0.498],
    color="0.60",
    lw=0.60,
    clip_on=True
)


# Biodiversity overlay
legend_ax.text(
    x_left,
    0.474,
    "Biodiversity overlay",
    ha="left",
    va="top",
    fontsize=9.7,
    fontweight="bold"
)

legend_ax.text(
    x_left,
    0.438,
    "Species richness",
    ha="left",
    va="top",
    fontsize=9.2,
    fontweight="bold"
)

y = 0.403

for col, lab in zip(RICHNESS_COLORS, RICHNESS_LABELS):

    legend_ax.scatter(
        x_symbol,
        y,
        s=67,
        marker="s",
        c=col,
        edgecolor="black",
        linewidth=0.95,
        clip_on=True
    )

    legend_ax.text(
        x_text,
        y,
        lab,
        fontsize=8.7,
        ha="left",
        va="center"
    )

    y -= 0.033


# Shannon diversity
legend_ax.text(
    x_left,
    0.256,
    "Shannon diversity (H')",
    ha="left",
    va="top",
    fontsize=9.2,
    fontweight="bold"
)

y = 0.221

for col, lab in zip(SHANNON_COLORS, SHANNON_LABELS):

    legend_ax.scatter(
        x_symbol,
        y,
        s=67,
        marker="o",
        c=col,
        edgecolor="black",
        linewidth=0.85,
        clip_on=True
    )

    legend_ax.text(
        x_text,
        y,
        lab,
        fontsize=8.35,
        ha="left",
        va="center"
    )

    y -= 0.033


# 10) Save figure
plt.subplots_adjust(
    left=0.045,
    right=0.945,
    top=0.985,
    bottom=0.030,
    wspace=-0.120,
    hspace=0.115
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


# 11) Close datasets
chl_ds.close()
sic_ds.close()
bio_ds.close()
gc.collect()
