# Fig. 2
# Occurrence-Based Biodiversity Patterns from GBIF
#
# Panels:
#   (a) Occurrence count
#   (b) Species richness
#   (c) Shannon diversity
#   (d) Simpson diversity
#   (e) Pielou evenness


# MOUNT DRIVE 
from google.colab import drive
drive.mount('/content/drive')


# 0) Install
!pip -q install xarray netCDF4 h5netcdf cartopy matplotlib numpy



# 1) Imports
import os
import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature



# 2) File paths
DATA_FILE = "/content/drive/MyDrive/SAM_Thesis/Data/Biodiversity_Indices/Biodiversity_indices_TOTAL_MONTHLY5000_2008_2025_1deg_masked_min5.nc"

OUT_DIR = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG = os.path.join(
    OUT_DIR,
    "Fig26_biodiversity_patterns_antarctic_shelf_seas_2008_2025.png"
)

SAVE_DPI = 1080

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Input file not found:\n{DATA_FILE}")

print("Input file found:")
print(DATA_FILE)



# 3) Plot settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "savefig.dpi": 300,
})



# Figure layout
FIGSIZE = (16.5, 9.8)

LEFT   = 0.030
RIGHT  = 0.970
TOP    = 0.965
BOTTOM = 0.060
WSPACE = 0.34   # increased
HSPACE = 0.28   # increased



# Variables and display names

PANEL_INFO = [
    {
        "letter": "a",
        "var": "occurrence_count",
        "title": "Occurrence count",
        "cmap": "YlOrRd",
        "vmin": 0,
        "vmax": 2000,
        "ticks": list(range(0, 2001, 250)),
        "ticklabels": ["0", "250", "500", "750", "1,000", "1,250", "1,500", "1,750", "2,000+"],
        "cbar_label": "Occurrences",
        "extend": "max",
    },
    {
        "letter": "b",
        "var": "species_richness",
        "title": "Species richness",
        "cmap": "YlGnBu",
        "vmin": 0,
        "vmax": 150,
        "ticks": list(range(0, 151, 25)),
        "ticklabels": ["0", "25", "50", "75", "100", "125", "150+"],
        "cbar_label": "Species",
        "extend": "max",
    },
    {
        "letter": "c",
        "var": "shannon_index",
        "title": "Shannon diversity",
        "cmap": "PuBu",
        "vmin": 0,
        "vmax": 3.0,
        "ticks": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "ticklabels": ["0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0+"],
        "cbar_label": r"Shannon index ($H'$)",
        "extend": "max",
    },
    {
        "letter": "d",
        "var": "simpson_index",
        "title": "Simpson diversity",
        "cmap": "RdPu",
        "vmin": 0,
        "vmax": 1.0,
        "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "ticklabels": ["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
        "cbar_label": r"Simpson index ($1-D$)",
        "extend": "neither",
    },
    {
        "letter": "e",
        "var": "pielou_evenness",
        "title": "Pielou evenness",
        "cmap": "viridis",
        "vmin": 0,
        "vmax": 1.0,
        "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "ticklabels": ["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
        "cbar_label": r"Pielou evenness ($J'$)",
        "extend": "neither",
    },
]



# 4) Helper functions
def standardize_coords(ds):
    rename_dict = {}

    for c in list(ds.coords):
        cl = c.lower()
        if cl in ["latitude", "nav_lat", "y"]:
            rename_dict[c] = "lat"
        elif cl in ["longitude", "nav_lon", "x"]:
            rename_dict[c] = "lon"

    for d in list(ds.dims):
        dl = d.lower()
        if dl in ["latitude", "nav_lat", "y"] and d not in rename_dict:
            rename_dict[d] = "lat"
        elif dl in ["longitude", "nav_lon", "x"] and d not in rename_dict:
            rename_dict[d] = "lon"

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("Dataset must contain lat and lon coordinates.")

    if float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180)

    ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    return ds


def make_cmap(cmap_name):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")
    return cmap


def mesh_from_da(da):
    lon2, lat2 = np.meshgrid(da["lon"].values, da["lat"].values)
    return lon2, lat2


def polar_ax(fig, gridspec_position):
    proj = ccrs.SouthPolarStereo()

    ax = fig.add_subplot(
        gridspec_position,
        projection=proj
    )

    theta = np.linspace(0, 2 * np.pi, 300)
    center = [0.5, 0.5]
    radius = 0.5

    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    ax.add_feature(
        cfeature.LAND,
        facecolor="0.88",
        edgecolor="black",
        linewidth=0.45,
        zorder=4
    )

    ax.coastlines(linewidth=0.55, zorder=5)

    # keep gridlines, but do not label 60S / 0 / 90E
    lon_grid = [-180, -150, -120, -90, -60, -30, 30, 60, 120, 150]
    lat_grid = [-60, -70, -80]

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.35,
        linestyle=":",
        color="0.50",
        alpha=0.65,
        zorder=6
    )

    gl.xlocator = mticker.FixedLocator(lon_grid)
    gl.ylocator = mticker.FixedLocator(lat_grid)

    # Longitude labels: remove 0° and 90°E
    edge_lat = -57.25

    lon_label_positions = [-150, -120, -90, -60, -30, 30, 60, 120, 150, -180]

    for lo in lon_label_positions:
        if lo == -180:
            label = "180°"
        elif lo < 0:
            label = f"{abs(lo)}°W"
        else:
            label = f"{lo}°E"

        ax.text(
            lo,
            edge_lat,
            label,
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=10
        )

    # Latitude labels: remove 60°S
    for la in [-70, -80]:
        ax.text(
            0,
            la,
            f"{abs(la)}°S",
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=10
        )

    return ax


def add_panel_letter(ax, letter):
    ax.text(
        -0.08,
        1.04,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        clip_on=False
    )


def add_cbar(fig, pcm, ax, label, ticks, ticklabels, extend):
    cb = fig.colorbar(
        pcm,
        ax=ax,
        orientation="vertical",
        shrink=0.84,
        pad=0.055,   # increased
        ticks=ticks,
        extend=extend
    )

    cb.set_label(
        label,
        fontsize=11,
        fontweight="bold",
        labelpad=11
    )

    cb.ax.set_yticklabels(ticklabels)
    cb.ax.tick_params(labelsize=9, width=0.7, length=3)

    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")

    cb.outline.set_linewidth(0.6)
    return cb


def print_basic_stats(ds, panel_info):
    print("\n" + "=" * 70)
    print("BIODIVERSITY DATA SUMMARY")
    print("=" * 70)

    for p in panel_info:
        var = p["var"]

        if var not in ds.data_vars:
            raise KeyError(
                f"Variable '{var}' not found. Available variables: {list(ds.data_vars)}"
            )

        arr = ds[var].values
        valid = np.isfinite(arr)

        print(f"\n{var}")
        print(f"  valid cells : {valid.sum()} / {arr.size}")

        if valid.sum() > 0:
            print(f"  min    : {np.nanmin(arr):.4f}")
            print(f"  p05    : {np.nanpercentile(arr, 5):.4f}")
            print(f"  median : {np.nanpercentile(arr, 50):.4f}")
            print(f"  p95    : {np.nanpercentile(arr, 95):.4f}")
            print(f"  p99    : {np.nanpercentile(arr, 99):.4f}")
            print(f"  max    : {np.nanmax(arr):.4f}")



# 5) Open dataset
ds = xr.open_dataset(
    DATA_FILE,
    decode_times=True,
    engine="h5netcdf"
)

ds = standardize_coords(ds)
ds = ds.sel(lat=slice(-90, -60))

print(ds)
print_basic_stats(ds, PANEL_INFO)



# 6) Create figure
fig = plt.figure(figsize=FIGSIZE)

# More space between figures
gs = fig.add_gridspec(
    2,
    6,
    left=LEFT,
    right=RIGHT,
    top=TOP,
    bottom=BOTTOM,
    wspace=WSPACE,
    hspace=HSPACE
)

positions = [
    gs[0, 0:2],   # (a)
    gs[0, 2:4],   # (b)
    gs[0, 4:6],   # (c)
    gs[1, 1:3],   # (d)
    gs[1, 3:5],   # (e)
]

for p, pos in zip(PANEL_INFO, positions):

    ax = polar_ax(fig, pos)

    da = ds[p["var"]].astype("float32")
    da = da.where(np.isfinite(da))

    lon2, lat2 = mesh_from_da(da)
    cmap = make_cmap(p["cmap"])

    pcm = ax.pcolormesh(
        lon2,
        lat2,
        da,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=p["vmin"],
        vmax=p["vmax"],
        shading="auto",
        zorder=1
    )

    ax.set_title(
        p["title"],
        fontsize=14,
        fontweight="bold",
        pad=10
    )

    add_panel_letter(ax, p["letter"])

    add_cbar(
        fig=fig,
        pcm=pcm,
        ax=ax,
        label=p["cbar_label"],
        ticks=p["ticks"],
        ticklabels=p["ticklabels"],
        extend=p["extend"]
    )



# 7) Save figure
plt.savefig(
    OUT_FIG,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nSaved figure:")
print(OUT_FIG)



# 8) Close
ds.close()
gc.collect()

print("\nDone.")
