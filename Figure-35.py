# Fig. 8 — Antarctic Shelf Seas Classification + PCA Matrix
# FINAL FIXED FULL CODE


# MOUNT DRIVE 
from google.colab import drive
drive.mount('/content/drive')



# 0) Install packages

!apt-get -qq install -y libproj-dev proj-data proj-bin libgeos-dev
!pip -q install cartopy xarray netCDF4 h5netcdf dask pandas numpy scipy scikit-learn matplotlib adjustText


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
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from adjustText import adjust_text


# 2) File paths
csv_file = "/content/drive/MyDrive/SAM_Thesis/Processed/sea_classification_input_2008_2025.csv"
bathy_file = "/content/drive/MyDrive/SAM_Thesis/Data/GEBCO_2024_CF.nc"

OUT_DIR = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG = os.path.join(
    OUT_DIR,
    "Fig32_Antarctic_shelf_seas_classification_PCA.png"
)


# 3) Main controls
SAVE_DPI = 1080

# If Colab crashes during GEBCO loading, increase to 36 or 44
BATHY_SKIP = 32

# "kmeans" = data-driven classes
# "conceptual" = cleaner conceptual grouping
CLASS_MODE = "kmeans"

LAT_OUTER = -58
LAT_INNER = -82


# 4) PCA variables
PCA_VARIABLES = [
    "SIC_mean",
    "SIC_trend_per_year",
    "SIC_retreat_month_since_sep",
    "SST_mean_C",
    "MLD_mean",
    "SIT_mean",
    "Stratification_mean",
    "POC_mean",
    "PIC_mean",
    "NPPint_mean",
    "Diatom_contribution_percent",
    "Shannon_mean",
]


# 5) Antarctic shelf sea sectors
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

sea_order = [s[0] for s in sea_info]


# 6) Sea label positions
label_positions = {
    "WED": (-43, -64.8),
    "KHV": (-11, -61.7),
    "RLS": (6, -61.5),
    "LAZ": (21, -62.4),
    "COS": (42, -62.6),
    "COO": (62, -62.7),
    "DAV": (82, -63.8),
    "MAW": (110, -61.5),
    "DUR": (140, -64.7),
    "SOM": (160, -66.6),
    "ROS": (-155, -71.0),
    "AMU": (-118, -67.2),
    "BEL": (-85, -66.0),
}

outside_lon_labels = {
    "0°":     (0.500, 1.018),
    "60°E":  (0.925, 0.810),
    "120°E": (0.925, 0.180),
    "180°":  (0.500, -0.030),
    "120°W": (0.075, 0.180),
    "60°W":  (0.075, 0.810),
}


# 7) Colors
CLASS_COLORS = {
    "Ice-dominated systems":   "#2C7FB8",
    "Ocean-dominated systems": "#74A76A",
    "Mixed systems":           "#F2C84B",
}

CLASS_EDGE = {
    "Ice-dominated systems":   "#1B5D91",
    "Ocean-dominated systems": "#3E7C3E",
    "Mixed systems":           "#B88A00",
}


# 8) Optional conceptual class map
conceptual_class = {
    "WED": "Ice-dominated systems",
    "KHV": "Ice-dominated systems",
    "RLS": "Mixed systems",
    "LAZ": "Ocean-dominated systems",
    "COS": "Ocean-dominated systems",
    "COO": "Ocean-dominated systems",
    "DAV": "Mixed systems",
    "MAW": "Ocean-dominated systems",
    "DUR": "Mixed systems",
    "SOM": "Mixed systems",
    "ROS": "Ice-dominated systems",
    "AMU": "Ocean-dominated systems",
    "BEL": "Ocean-dominated systems",
}


# 9) Helper functions
def normalize_lon(lon):
    return ((lon + 180) % 360) - 180


def get_coord_names(ds):
    all_names = list(ds.coords) + list(ds.dims) + list(ds.data_vars)
    lat_candidates, lon_candidates = [], []

    for name in all_names:
        lname = name.lower()
        if lname in ["lat", "latitude", "nav_lat", "y"] or "lat" in lname:
            lat_candidates.append(name)
        if lname in ["lon", "longitude", "nav_lon", "x"] or "lon" in lname:
            lon_candidates.append(name)

    if len(lat_candidates) == 0 or len(lon_candidates) == 0:
        raise ValueError("Could not detect latitude/longitude names.")

    return lat_candidates[0], lon_candidates[0]


def get_first_numeric_var(ds):
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.number):
            return v
    raise ValueError("No numeric variable found.")


def subset_antarctic(ds, lat_name, south=-90, north=-58):
    lat_values = ds[lat_name].values
    if lat_values[0] < lat_values[-1]:
        return ds.sel({lat_name: slice(south, north)})
    else:
        return ds.sel({lat_name: slice(north, south)})


def get_sector_parts(lon1, lon2):
    if lon1 > lon2:
        return [(lon1, 180), (-180, lon2)]
    return [(lon1, lon2)]


def make_sector_polygon(lon1, lon2, lat_inner=-82, lat_outer=-58, n=700):
    lons_outer = np.linspace(lon1, lon2, n)
    lats_outer = np.full_like(lons_outer, lat_outer, dtype=float)
    lats_inner = np.full_like(lons_outer, lat_inner, dtype=float)

    poly_lons = np.concatenate([lons_outer, lons_outer[::-1]])
    poly_lats = np.concatenate([lats_outer, lats_inner[::-1]])

    return poly_lons, poly_lats


def add_circular_boundary(ax):
    theta = np.linspace(0, 2 * np.pi, 900)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
    ax.set_boundary(circle, transform=ax.transAxes)


def style_pca_axis(ax):
    ax.axhline(0, color="0.45", linewidth=0.75, linestyle="--", zorder=0)
    ax.axvline(0, color="0.45", linewidth=0.75, linestyle="--", zorder=0)

    ax.grid(
        True,
        color="0.90",
        linestyle="-",
        linewidth=0.45,
        alpha=0.75,
        zorder=0
    )

    ax.tick_params(axis="both", labelsize=7.5, width=0.7, length=3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("0.25")


def add_cluster_ellipse(ax, x, y, class_name):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    facecolor = CLASS_COLORS[class_name]
    edgecolor = CLASS_EDGE[class_name]

    if len(x) >= 3:
        cov = np.cov(x, y)
        if not np.all(np.isfinite(cov)):
            return

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        vals = np.maximum(vals, 1e-6)

        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 1.60 * np.sqrt(vals)

    elif len(x) == 2:
        angle = 0
        width = max(1.25, abs(x[1] - x[0]) + 0.9)
        height = max(1.00, abs(y[1] - y[0]) + 0.9)

    elif len(x) == 1:
        angle = 0
        width = 0.95
        height = 0.75

    else:
        return

    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.95,
        alpha=0.15,
        zorder=1
    )
    ax.add_patch(ell)


def plot_score_panel(ax, scores_df, xcol, ycol, title, xlabel, ylabel):
    style_pca_axis(ax)

    texts = []

    for class_name in ["Ice-dominated systems", "Ocean-dominated systems", "Mixed systems"]:
        sub = scores_df[scores_df["Class"] == class_name]

        if len(sub) == 0:
            continue

        add_cluster_ellipse(
            ax,
            sub[xcol].values,
            sub[ycol].values,
            class_name
        )

        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=22,
            color=CLASS_COLORS[class_name],
            edgecolor="black",
            linewidth=0.30,
            zorder=3
        )

        for _, r in sub.iterrows():
            t = ax.text(
                r[xcol] + 0.07,
                r[ycol] + 0.07,
                r["Sea"],
                fontsize=7.3,
                fontweight="bold",
                color="black",
                ha="left",
                va="center",
                zorder=5
            )
            texts.append(t)

    all_x = scores_df[xcol].values
    all_y = scores_df[ycol].values

    xpad = max(0.90, 0.17 * (np.nanmax(all_x) - np.nanmin(all_x)))
    ypad = max(0.85, 0.17 * (np.nanmax(all_y) - np.nanmin(all_y)))

    ax.set_xlim(np.nanmin(all_x) - xpad, np.nanmax(all_x) + xpad)
    ax.set_ylim(np.nanmin(all_y) - ypad, np.nanmax(all_y) + ypad)

    adjust_text(
        texts,
        ax=ax,
        expand_text=(1.15, 1.25),
        expand_points=(1.15, 1.25),
        force_text=(0.35, 0.45),
        force_points=(0.25, 0.35),
        arrowprops=dict(arrowstyle="-", color="0.45", lw=0.35, alpha=0.55),
        only_move={"points": "xy", "text": "xy"}
    )

    ax.set_title(title, fontsize=8.0, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=7.9, fontweight="bold", labelpad=6)
    ax.set_ylabel(ylabel, fontsize=7.9, fontweight="bold", labelpad=6)


def clean_var_label(v):
    label_map = {
        "SIC_mean": "SIC",
        "SIC_trend_per_year": "SIC trend",
        "SIC_retreat_month_since_sep": "Retreat timing",
        "SST_mean_C": "SST",
        "MLD_mean": "MLD",
        "SIT_mean": "SIT",
        "Stratification_mean": "Stratification",
        "POC_mean": "POC",
        "PIC_mean": "PIC",
        "NPPint_mean": "NPPint",
        "Diatom_contribution_percent": "Diatom %",
        "Shannon_mean": "Shannon",
    }
    return label_map.get(v, v)


def variable_color(v):
    if v in ["SIC_mean", "SIC_trend_per_year", "SIC_retreat_month_since_sep", "SIT_mean"]:
        return "#1F5AA6"
    if v in ["SST_mean_C", "MLD_mean", "Stratification_mean"]:
        return "#238B45"
    if v in ["POC_mean", "PIC_mean", "NPPint_mean"]:
        return "#B8860B"
    if v in ["Diatom_contribution_percent", "Shannon_mean"]:
        return "#6A3D9A"
    return "black"


def plot_loading_panel(ax, loadings_df, exp_var):
    ax.axhline(0, color="0.55", linewidth=0.75, linestyle="--", zorder=0)
    ax.axvline(0, color="0.55", linewidth=0.75, linestyle="--", zorder=0)

    unit_circle = plt.Circle(
        (0, 0),
        1.0,
        transform=ax.transData,
        fill=False,
        linestyle="--",
        linewidth=0.7,
        color="0.72",
        zorder=1
    )
    ax.add_patch(unit_circle)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal", adjustable="box")

    # Manually controlled label positions for clearer arrows
    manual_label_pos = {
        "SIC_mean": (-0.78, 0.32),
        "SIC_trend_per_year": (0.78, 0.25),
        "SIC_retreat_month_since_sep": (-0.60, -0.22),
        "SST_mean_C": (0.82, -0.08),
        "MLD_mean": (0.70, 0.48),
        "SIT_mean": (-0.62, 0.52),
        "Stratification_mean": (-0.82, -0.62),
        "POC_mean": (-0.28, 0.72),
        "PIC_mean": (-0.35, -0.15),
        "NPPint_mean": (-0.28, 0.34),
        "Diatom_contribution_percent": (0.38, 0.82),
        "Shannon_mean": (0.42, -0.32),
    }

    for _, r in loadings_df.iterrows():
        v = r["Variable"]
        x = r["PC1"]
        y = r["PC2"]
        color = variable_color(v)

        # arrow endpoint
        ax_x = x * 0.90
        ax_y = y * 0.90

        # clearer arrow
        ax.annotate(
            "",
            xy=(ax_x, ax_y),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.25,
                color=color,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=8,
                alpha=0.95
            ),
            zorder=3
        )

        lx, ly = manual_label_pos.get(v, (x * 1.05, y * 1.05))

        # Colored text with thin black outline, no box
        ax.text(
            lx,
            ly,
            clean_var_label(v),
            fontsize=6.7,
            color=color,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
            path_effects=[
                pe.Stroke(linewidth=1.25, foreground="white"),
                pe.Stroke(linewidth=0.35, foreground="black"),
                pe.Normal()
            ]
        )

    ax.set_title(
        "Variable loadings (PC1–PC2 plane)",
        fontsize=8.0,
        fontweight="bold",
        pad=8
    )

    ax.set_xlabel(
        f"PC1 ({exp_var[0]:.1f}% variance)",
        fontsize=7.9,
        fontweight="bold",
        labelpad=6
    )

    ax.set_ylabel(
        f"PC2 ({exp_var[1]:.1f}% variance)",
        fontsize=7.9,
        fontweight="bold",
        labelpad=6
    )

    ax.tick_params(axis="both", labelsize=7.3, width=0.7, length=3)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("0.25")


# 10) Load CSV
df = pd.read_csv(csv_file)

if "Sea" not in df.columns:
    raise ValueError("CSV must contain a Sea column.")

df["Sea"] = df["Sea"].astype(str).str.upper().str.strip()

if "SST_mean_C" not in df.columns:
    if "SST_mean" in df.columns:
        df["SST_mean_C"] = df["SST_mean"] - 273.15
    else:
        raise ValueError("SST_mean_C missing and SST_mean missing.")

missing = [v for v in PCA_VARIABLES if v not in df.columns]
if missing:
    raise ValueError(f"Missing variables in CSV: {missing}")

df = df[df["Sea"].isin(sea_order)].copy()
df["Sea"] = pd.Categorical(df["Sea"], categories=sea_order, ordered=True)
df = df.sort_values("Sea").reset_index(drop=True)

print("Input data used for PCA:")
display(df[["Sea"] + PCA_VARIABLES])

X = df[PCA_VARIABLES].replace([np.inf, -np.inf], np.nan)
X = X.apply(lambda col: col.fillna(col.median()), axis=0)

scaler = StandardScaler()
Xz = scaler.fit_transform(X)

pca = PCA(n_components=3)
scores = pca.fit_transform(Xz)
EXP_VAR = pca.explained_variance_ratio_ * 100

scores_df = pd.DataFrame(scores, columns=["PC1", "PC2", "PC3"])
scores_df["Sea"] = df["Sea"].astype(str).values


# 11) Orient PC signs
load_tmp = pd.DataFrame(
    pca.components_.T,
    index=PCA_VARIABLES,
    columns=["PC1", "PC2", "PC3"]
)

if load_tmp.loc[["SIC_mean", "SIT_mean"], "PC1"].mean() > 0:
    scores_df["PC1"] *= -1
    pca.components_[0, :] *= -1

load_tmp = pd.DataFrame(
    pca.components_.T,
    index=PCA_VARIABLES,
    columns=["PC1", "PC2", "PC3"]
)

bio_ref = ["POC_mean", "NPPint_mean", "Shannon_mean"]
if load_tmp.loc[bio_ref, "PC2"].mean() < 0:
    scores_df["PC2"] *= -1
    pca.components_[1, :] *= -1


# 12) Classification
if CLASS_MODE == "kmeans":
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=100)
    cluster_id = kmeans.fit_predict(Xz)

    df["cluster_id"] = cluster_id
    scores_df["cluster_id"] = cluster_id

    z_df = pd.DataFrame(Xz, columns=PCA_VARIABLES)
    z_df["cluster_id"] = cluster_id
    cluster_means = z_df.groupby("cluster_id").mean()

    ice_score = (
        cluster_means["SIC_mean"]
        + cluster_means["SIT_mean"]
        + cluster_means["SIC_retreat_month_since_sep"]
        - cluster_means["SST_mean_C"]
    )

    ocean_score = (
        cluster_means["SST_mean_C"]
        + cluster_means["MLD_mean"]
        + cluster_means["Stratification_mean"]
    )

    ice_cluster = ice_score.idxmax()
    remaining = [c for c in cluster_means.index if c != ice_cluster]
    ocean_cluster = ocean_score.loc[remaining].idxmax()
    mixed_cluster = [c for c in cluster_means.index if c not in [ice_cluster, ocean_cluster]][0]

    cluster_to_class = {
        ice_cluster: "Ice-dominated systems",
        ocean_cluster: "Ocean-dominated systems",
        mixed_cluster: "Mixed systems",
    }

    df["Class"] = df["cluster_id"].map(cluster_to_class)
    scores_df["Class"] = scores_df["cluster_id"].map(cluster_to_class)

else:
    df["Class"] = df["Sea"].astype(str).map(conceptual_class)
    scores_df["Class"] = scores_df["Sea"].astype(str).map(conceptual_class)

sea_class_map = dict(zip(df["Sea"].astype(str), df["Class"]))

print("\nExplained variance:")
print(f"PC1 = {EXP_VAR[0]:.2f}%")
print(f"PC2 = {EXP_VAR[1]:.2f}%")
print(f"PC3 = {EXP_VAR[2]:.2f}%")
print(f"PC1 + PC2 = {EXP_VAR[0] + EXP_VAR[1]:.2f}%")
print(f"PC1 + PC2 + PC3 = {EXP_VAR[0] + EXP_VAR[1] + EXP_VAR[2]:.2f}%")

print("\nFinal class assignment:")
display(df[["Sea", "Class"]])


# 13) Load GEBCO bathymetry safely
print("\nLoading GEBCO bathymetry safely...")

ds_bathy = xr.open_dataset(bathy_file, chunks={})
bathy_lat, bathy_lon = get_coord_names(ds_bathy)
bathy_var = "elevation" if "elevation" in ds_bathy.data_vars else get_first_numeric_var(ds_bathy)

ds_bathy = subset_antarctic(ds_bathy, bathy_lat, south=-90, north=-58)

z = ds_bathy[bathy_var].isel({
    bathy_lat: slice(None, None, BATHY_SKIP),
    bathy_lon: slice(None, None, BATHY_SKIP)
}).astype("float32").load()

bathy_lats = z[bathy_lat].values
bathy_lons = z[bathy_lon].values

if np.nanmax(bathy_lons) > 180:
    bathy_lons = normalize_lon(bathy_lons)
    sort_idx = np.argsort(bathy_lons)
    bathy_lons = bathy_lons[sort_idx]
    z = z.isel({bathy_lon: sort_idx})

z_ocean = z.where(z < 0)

ds_bathy.close()
del ds_bathy, z
gc.collect()

print("Bathymetry loaded:", z_ocean.shape)


# 14) PCA loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(loadings, columns=["PC1", "PC2", "PC3"])
loadings_df["Variable"] = PCA_VARIABLES

max_abs = np.nanmax(np.abs(loadings_df[["PC1", "PC2"]].values))
if max_abs > 0:
    loadings_df[["PC1", "PC2", "PC3"]] *= 0.78 / max_abs


# 15) Figure layout
plt.close("all")

fig = plt.figure(figsize=(8.6, 10.6), dpi=180)

# [left, bottom, width, height]
map_pos = [0.200, 0.610, 0.560, 0.300]

# RIGHT vertical colorbar
cbar_pos = [0.785, 0.645, 0.018, 0.230]

# Panel b positions
pc12_pos = [0.080, 0.370, 0.405, 0.155]
pc13_pos = [0.565, 0.370, 0.405, 0.155]
pc23_pos = [0.080, 0.135, 0.405, 0.155]

# Slightly larger and taller loading plot
load_pos = [0.585, 0.125, 0.360, 0.180]

leg_pos = [0.070, 0.035, 0.860, 0.060]


# 16) Panel (a) map
proj = ccrs.SouthPolarStereo()
ax_map = fig.add_axes(map_pos, projection=proj)

ax_map.set_extent([-180, 180, -90, -58], crs=ccrs.PlateCarree())
add_circular_boundary(ax_map)

bathy_levels = [
    -8000, -7000, -6000, -5000, -4000, -3000,
    -2000, -1500, -1000, -500, -200, 0
]

bath = ax_map.contourf(
    bathy_lons,
    bathy_lats,
    z_ocean,
    levels=bathy_levels,
    cmap="Blues_r",
    extend="min",
    transform=ccrs.PlateCarree(),
    zorder=1
)

for abbr, fullname, lontext, lon1, lon2 in sea_info:
    class_name = sea_class_map.get(abbr, "Mixed systems")
    color = CLASS_COLORS[class_name]

    for part_lon1, part_lon2 in get_sector_parts(lon1, lon2):
        poly_lons, poly_lats = make_sector_polygon(
            part_lon1,
            part_lon2,
            lat_inner=LAT_INNER,
            lat_outer=LAT_OUTER,
            n=700
        )

        ax_map.fill(
            poly_lons,
            poly_lats,
            facecolor=color,
            edgecolor="white",
            linewidth=0.30,
            alpha=0.62,
            transform=ccrs.PlateCarree(),
            zorder=2.4
        )

    lab_lon, lab_lat = label_positions[abbr]

    ax_map.text(
        lab_lon,
        lab_lat,
        abbr,
        transform=ccrs.PlateCarree(),
        fontsize=8.3,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.16",
            facecolor="white",
            edgecolor="0.25",
            linewidth=0.35,
            alpha=0.96
        ),
        zorder=20,
        clip_on=False
    )

ax_map.add_feature(
    cfeature.LAND,
    facecolor="#E6E6E6",
    edgecolor="black",
    linewidth=0.45,
    zorder=8.5
)

ax_map.coastlines(
    resolution="110m",
    linewidth=0.55,
    color="black",
    zorder=9
)

ax_map.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,
    linewidth=0.40,
    color="0.45",
    alpha=0.42,
    linestyle="--",
    zorder=3
)

lat_label_positions = {
    "60°S": (178, -60),
    "70°S": (178, -70),
    "80°S": (178, -80),
}

for txt, (lon_pos, lat_pos) in lat_label_positions.items():
    ax_map.text(
        lon_pos,
        lat_pos,
        txt,
        transform=ccrs.PlateCarree(),
        fontsize=8.3,
        fontweight="bold",
        ha="right",
        va="center",
        color="0.25",
        zorder=30,
        clip_on=False
    )

for txt, (x, y) in outside_lon_labels.items():
    ax_map.text(
        x,
        y,
        txt,
        transform=ax_map.transAxes,
        fontsize=8.3,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        zorder=30,
        clip_on=False
    )


# 17) Right-side vertical colorbar
cbar_ax = fig.add_axes(cbar_pos)

cbar = plt.colorbar(
    bath,
    cax=cbar_ax,
    orientation="vertical"
)

cbar.set_label(
    "Bathymetry / Elevation (m)",
    fontsize=8.6,
    fontweight="bold",
    labelpad=7,
    rotation=90
)

cbar.ax.tick_params(labelsize=7.2, width=0.6)

for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight("bold")


# 18) Panel labels and title
fig.text(
    0.035,
    0.955,
    "(a)",
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="top"
)

fig.text(
    0.500,
    0.947,
    "Antarctic shelf seas",
    fontsize=13.5,
    fontweight="bold",
    ha="center",
    va="top"
)

fig.text(
    0.035,
    0.540,
    "(b)",
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="top"
)


# 19) Panel (b) PCA matrix
ax_pc12 = fig.add_axes(pc12_pos)
ax_pc13 = fig.add_axes(pc13_pos)
ax_pc23 = fig.add_axes(pc23_pos)
ax_load = fig.add_axes(load_pos)

plot_score_panel(
    ax_pc12,
    scores_df,
    "PC1",
    "PC2",
    f"PC1 vs PC2 ({EXP_VAR[0] + EXP_VAR[1]:.1f}% total variance)",
    f"PC1 ({EXP_VAR[0]:.1f}% variance)",
    f"PC2 ({EXP_VAR[1]:.1f}% variance)"
)

plot_score_panel(
    ax_pc13,
    scores_df,
    "PC1",
    "PC3",
    f"PC1 vs PC3 ({EXP_VAR[0] + EXP_VAR[2]:.1f}% total variance)",
    f"PC1 ({EXP_VAR[0]:.1f}% variance)",
    f"PC3 ({EXP_VAR[2]:.1f}% variance)"
)

plot_score_panel(
    ax_pc23,
    scores_df,
    "PC2",
    "PC3",
    f"PC2 vs PC3 ({EXP_VAR[1] + EXP_VAR[2]:.1f}% total variance)",
    f"PC2 ({EXP_VAR[1]:.1f}% variance)",
    f"PC3 ({EXP_VAR[2]:.1f}% variance)"
)

plot_loading_panel(
    ax_load,
    loadings_df,
    EXP_VAR
)


# 20) Bottom legend
ax_leg = fig.add_axes(leg_pos)
ax_leg.axis("off")

legend_handles = [
    mpatches.Patch(
        facecolor=CLASS_COLORS["Ice-dominated systems"],
        edgecolor="black",
        linewidth=0.4,
        label="Ice-dominated systems"
    ),
    mpatches.Patch(
        facecolor=CLASS_COLORS["Ocean-dominated systems"],
        edgecolor="black",
        linewidth=0.4,
        label="Ocean-dominated systems"
    ),
    mpatches.Patch(
        facecolor=CLASS_COLORS["Mixed systems"],
        edgecolor="black",
        linewidth=0.4,
        label="Mixed systems"
    ),
]

leg = ax_leg.legend(
    handles=legend_handles,
    loc="center",
    ncol=3,
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    fontsize=9.4,
    handlelength=2.6,
    handleheight=1.4,
    columnspacing=4.0,
    borderpad=0.78
)

leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(0.7)


# 21) Save
plt.savefig(
    OUT_FIG,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nSaved figure:")
print(OUT_FIG)

print("\nExplained variance:")
print(f"PC1 = {EXP_VAR[0]:.2f}%")
print(f"PC2 = {EXP_VAR[1]:.2f}%")
print(f"PC3 = {EXP_VAR[2]:.2f}%")
print(f"PC1 + PC2 = {EXP_VAR[0] + EXP_VAR[1]:.2f}%")
print(f"PC1 + PC2 + PC3 = {EXP_VAR[0] + EXP_VAR[1] + EXP_VAR[2]:.2f}%")

print("\nFinal class assignment:")
display(df[["Sea", "Class"]])
