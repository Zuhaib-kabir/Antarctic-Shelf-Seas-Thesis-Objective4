# Fig. 5 — Biodiversity relationships with environmental drivers
# Antarctic Shelf Seas, 2008–2025
#
# Style matched to your Fig. 22 / dummy figure
#
# Panels:
# (a) Shannon diversity vs SIC
# (b) Shannon diversity vs chlorophyll-a
# (c) Shannon diversity vs POC
# (d) Shannon diversity vs SST


# MOUNT DRIVE 
from google.colab import drive
drive.mount('/content/drive')



# 0) Install packages

!pip -q install scipy adjustText


# 1) Imports

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from adjustText import adjust_text



# 2) File paths
IN_CSV = "/content/drive/MyDrive/SAM_Thesis/Processed/Fig30_Biodiversity_Environmental_Drivers_ACTIVE_seawise_2008_2025.csv"

OUT_DIR = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG = os.path.join(
    OUT_DIR,
    "Fig30_Biodiversity_Environmental_Drivers_2008_2025_1080dpi.png"
)

OUT_STATS = os.path.join(
    OUT_DIR,
    "Fig30_Biodiversity_Environmental_Drivers_regression_stats.csv"
)


# 3) Main settings
SAVE_DPI = 1080

POINT_SIZE = 62
POINT_ALPHA = 0.95

REG_LINE_COLOR = "black"
REG_LINE_WIDTH = 1.35

CI_COLOR = "0.75"
CI_ALPHA = 0.30

LABEL_FONTSIZE = 7.3
STAT_FONTSIZE = 8.0

MIN_N_REG = 4

USE_SHARED_YLIM = True
Y_PAD_FRAC = 0.18


# 4) Matplotlib style
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 14,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.bbox": "tight",
})


# 5) Load CSV

if not os.path.exists(IN_CSV):
    raise FileNotFoundError(f"Input CSV not found:\n{IN_CSV}")

df = pd.read_csv(IN_CSV)

print("Loaded Fig. 30 CSV:")
print(IN_CSV)
print(df.head())
print("\nColumns:")
print(df.columns.tolist())


# 6) Sea order, names, colors
sea_order = [
    "WED", "KHV", "RLS", "LAZ", "COS", "COO", "DAV",
    "MAW", "DUR", "SOM", "ROS", "AMU", "BEL"
]

sea_full = {
    "WED": "Weddell Sea",
    "KHV": "King Haakon VII Sea",
    "RLS": "Riiser-Larsen Sea",
    "LAZ": "Lazarev Sea",
    "COS": "Cosmonauts Sea",
    "COO": "Cooperation Sea",
    "DAV": "Davis Sea",
    "MAW": "Mawson Sea",
    "DUR": "D’Urville Sea",
    "SOM": "Somov Sea",
    "ROS": "Ross Sea",
    "AMU": "Amundsen Sea",
    "BEL": "Bellingshausen Sea",
}

sea_color = {
    "WED": "#1f78b4",
    "KHV": "#ff7f00",
    "RLS": "#33a02c",
    "LAZ": "#6a3d9a",
    "COS": "#e31a1c",
    "COO": "#8c564b",
    "DAV": "#e377c2",
    "MAW": "#7f7f7f",
    "DUR": "#bcbd22",
    "SOM": "#17becf",
    "ROS": "#08306b",
    "AMU": "#ff7f0e",
    "BEL": "#006d2c",
}

df["sea"] = df["sea"].astype(str)
df["sea_order"] = df["sea"].apply(lambda x: sea_order.index(x) if x in sea_order else 999)
df = df.sort_values("sea_order").reset_index(drop=True)


# 7) Required columns
required_cols = [
    "sea",

    "shannon_for_sic",
    "sic_for_shannon",
    "use_sic_panel",

    "shannon_for_chl",
    "chl_for_shannon",
    "use_chl_panel",

    "shannon_for_poc",
    "poc_for_shannon",
    "use_poc_panel",

    "shannon_for_sst",
    "sst_for_shannon",
    "use_sst_panel",
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns:\n{missing}")


# 8) Helper functions
def to_bool_series(s):
    """
    Convert bool/string bool columns safely.
    """
    if s.dtype == bool:
        return s

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def clean_panel_data(data, xcol, ycol, use_col):
    sub = data[["sea", xcol, ycol, use_col]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan)

    sub[use_col] = to_bool_series(sub[use_col])
    sub = sub[sub[use_col]]

    sub = sub.dropna(subset=[xcol, ycol])

    return sub


def regression_statistics(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)

    if n < MIN_N_REG:
        return {
            "n": n,
            "slope": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "p": np.nan,
            "r2": np.nan
        }

    res = stats.linregress(x, y)

    return {
        "n": n,
        "slope": res.slope,
        "intercept": res.intercept,
        "r": res.rvalue,
        "p": res.pvalue,
        "r2": res.rvalue ** 2
    }


def add_regression_and_ci(ax, x, y):
    st = regression_statistics(x, y)

    if not np.isfinite(st["slope"]):
        return st

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 250)
    yfit = st["intercept"] + st["slope"] * xfit

    ax.plot(
        xfit,
        yfit,
        color=REG_LINE_COLOR,
        linewidth=REG_LINE_WIDTH,
        zorder=2
    )

    n = len(x)
    yhat = st["intercept"] + st["slope"] * x
    resid = y - yhat

    s_err = np.sqrt(np.sum(resid ** 2) / max(n - 2, 1))
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)

    if sxx > 0 and n > 2:
        tval = stats.t.ppf(0.975, df=n - 2)
        ci = tval * s_err * np.sqrt(
            1 / n + ((xfit - x_mean) ** 2) / sxx
        )

        ax.fill_between(
            xfit,
            yfit - ci,
            yfit + ci,
            color=CI_COLOR,
            alpha=CI_ALPHA,
            linewidth=0,
            zorder=1
        )

    return st


def add_stats_box(ax, st):
    if not np.isfinite(st["r"]):
        txt = f"n = {st['n']}"
    else:
        ptxt = "p < 0.001" if st["p"] < 0.001 else f"p = {st['p']:.3f}"
        txt = f"r = {st['r']:.2f}, {ptxt}, n = {st['n']}"

    ax.text(
        0.985,
        0.945,
        txt,
        transform=ax.transAxes,
        fontsize=STAT_FONTSIZE,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="square,pad=0.20",
            facecolor="white",
            edgecolor="0.45",
            linewidth=0.6,
            alpha=0.96
        ),
        zorder=30
    )


def add_labels_adjusted(ax, x, y, labels):
    """
    Add all labels and automatically reduce overlaps.
    """
    texts = []

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xr = np.nanmax(x) - np.nanmin(x)
    yr = np.nanmax(y) - np.nanmin(y)

    dx = 0.010 * xr if np.isfinite(xr) and xr > 0 else 0.02
    dy = 0.010 * yr if np.isfinite(yr) and yr > 0 else 0.02

    for xi, yi, lab in zip(x, y, labels):

        if not np.isfinite(xi) or not np.isfinite(yi):
            continue

        t = ax.text(
            xi + dx,
            yi + dy,
            lab,
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
            ha="left",
            va="center",
            color="black",
            zorder=20
        )

        texts.append(t)

    # Automatic overlap adjustment
    adjust_text(
        texts,
        x=x,
        y=y,
        ax=ax,
        expand_text=(1.15, 1.25),
        expand_points=(1.25, 1.35),
        force_text=(0.45, 0.65),
        force_points=(0.25, 0.35),
        lim=300,
        arrowprops=dict(
            arrowstyle="-",
            color="0.35",
            lw=0.35,
            alpha=0.70
        )
    )

    return texts


def clean_axis(ax):
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.45,
        alpha=0.35,
        zorder=0
    )

    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.9)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3.5,
        width=0.8
    )


def add_panel_label(ax, label):
    ax.text(
        -0.145,
        1.075,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False
    )


def set_axis_padding(ax, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) > 0:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        xpad = (xmax - xmin) * 0.16 if xmax > xmin else 1
        ax.set_xlim(xmin - xpad, xmax + xpad)

    if len(y) > 0 and not USE_SHARED_YLIM:
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        ypad = (ymax - ymin) * Y_PAD_FRAC if ymax > ymin else 0.2
        ax.set_ylim(max(0, ymin - ypad), ymax + ypad)


# 9) Panel setup
panels = [
    {
        "panel": "(a)",
        "title": "Shannon diversity vs SIC",
        "xcol": "sic_for_shannon",
        "ycol": "shannon_for_sic",
        "use_col": "use_sic_panel",
        "xlabel": "Sea-ice concentration, SIC (%)",
        "ylabel": "Shannon diversity index",
    },
    {
        "panel": "(b)",
        "title": "Shannon diversity vs chlorophyll-a",
        "xcol": "chl_for_shannon",
        "ycol": "shannon_for_chl",
        "use_col": "use_chl_panel",
        "xlabel": "Chlorophyll-a (mg m$^{-3}$)",
        "ylabel": "Shannon diversity index",
    },
    {
        "panel": "(c)",
        "title": "Shannon diversity vs POC",
        "xcol": "poc_for_shannon",
        "ycol": "shannon_for_poc",
        "use_col": "use_poc_panel",
        "xlabel": "POC (mg C m$^{-3}$)",
        "ylabel": "Shannon diversity index",
    },
    {
        "panel": "(d)",
        "title": "Shannon diversity vs SST",
        "xcol": "sst_for_shannon",
        "ycol": "shannon_for_sst",
        "use_col": "use_sst_panel",
        "xlabel": "Sea-surface temperature, SST (°C)",
        "ylabel": "Shannon diversity index",
    }
]


# 10) Create figure
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(13.2, 9.0)
)

axes = axes.flatten()

stats_records = []
all_y_values = []

for ax, panel in zip(axes, panels):

    sub = clean_panel_data(
        df,
        panel["xcol"],
        panel["ycol"],
        panel["use_col"]
    )

    x = sub[panel["xcol"]].values
    y = sub[panel["ycol"]].values
    labels = sub["sea"].values

    all_y_values.extend(y.tolist())

    # Points colored by sea
    for xi, yi, lab in zip(x, y, labels):

        ax.scatter(
            xi,
            yi,
            s=POINT_SIZE,
            color=sea_color.get(lab, "0.5"),
            edgecolor="black",
            linewidth=0.55,
            alpha=POINT_ALPHA,
            zorder=5
        )

    # Regression line and confidence interval
    st = add_regression_and_ci(ax, x, y)

    # Axis padding before label adjustment
    set_axis_padding(ax, x, y)

    # Labels with automatic adjustment
    add_labels_adjusted(ax, x, y, labels)

    # Stats box after labels
    add_stats_box(ax, st)

    # Axis style
    clean_axis(ax)

    ax.set_title(
        panel["title"],
        fontsize=14,
        fontweight="bold",
        pad=10
    )

    ax.set_xlabel(
        panel["xlabel"],
        fontsize=10,
        fontweight="bold"
    )

    ax.set_ylabel(
        panel["ylabel"],
        fontsize=10,
        fontweight="bold"
    )

    add_panel_label(ax, panel["panel"])

    stats_records.append({
        "panel": panel["panel"],
        "relationship": panel["title"],
        "x_variable": panel["xcol"],
        "y_variable": panel["ycol"],
        **st
    })


# 11) Shared y-axis limits

if USE_SHARED_YLIM:

    all_y_values = np.asarray(all_y_values, dtype=float)
    all_y_values = all_y_values[np.isfinite(all_y_values)]

    ymin = np.nanmin(all_y_values)
    ymax = np.nanmax(all_y_values)

    ypad = (ymax - ymin) * Y_PAD_FRAC if ymax > ymin else 0.2

    for ax in axes:
        ax.set_ylim(max(0, ymin - ypad), ymax + ypad)


# 12) Bottom sea-abbreviation legend
legend_ax = fig.add_axes([0.075, 0.005, 0.87, 0.115])
legend_ax.axis("off")

legend_ax.plot(
    [0, 1],
    [0.95, 0.95],
    color="0.45",
    linewidth=0.7,
    transform=legend_ax.transAxes,
    clip_on=False
)

legend_ax.text(
    0.00,
    0.80,
    "Sea abbreviations:",
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="center",
    transform=legend_ax.transAxes
)

legend_items = [
    ("WED", "Weddell Sea"),
    ("KHV", "King Haakon VII Sea"),
    ("RLS", "Riiser-Larsen Sea"),
    ("LAZ", "Lazarev Sea"),
    ("COS", "Cosmonauts Sea"),
    ("COO", "Cooperation Sea"),
    ("DAV", "Davis Sea"),
    ("MAW", "Mawson Sea"),
    ("DUR", "D’Urville Sea"),
    ("SOM", "Somov Sea"),
    ("ROS", "Ross Sea"),
    ("AMU", "Amundsen Sea"),
    ("BEL", "Bellingshausen Sea"),
]

# 5 columns so all 13 seas appear cleanly
x_positions = [0.03, 0.245, 0.46, 0.675, 0.86]
y_positions = [0.53, 0.30, 0.08]

idx = 0

for col_x in x_positions:
    for row_y in y_positions:

        if idx >= len(legend_items):
            break

        short, full = legend_items[idx]

        legend_ax.scatter(
            col_x,
            row_y,
            s=36,
            color=sea_color[short],
            edgecolor="black",
            linewidth=0.4,
            transform=legend_ax.transAxes,
            clip_on=False
        )

        legend_ax.text(
            col_x + 0.015,
            row_y,
            f"{short} – {full}",
            fontsize=8.2,
            ha="left",
            va="center",
            transform=legend_ax.transAxes
        )

        idx += 1


# 13) Layout and save
plt.subplots_adjust(
    left=0.075,
    right=0.965,
    top=0.935,
    bottom=0.175,
    wspace=0.28,
    hspace=0.36
)

plt.savefig(
    OUT_FIG,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# 14) Save regression statistics

stats_df = pd.DataFrame(stats_records)
stats_df.to_csv(OUT_STATS, index=False)

print("\nSaved Fig. 30:")
print(OUT_FIG)

print("\nSaved regression stats:")
print(OUT_STATS)

print("\nRegression statistics:")
print(stats_df)
