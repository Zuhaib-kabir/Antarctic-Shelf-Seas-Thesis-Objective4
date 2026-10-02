# Fig. 7 — Monthly Biodiversity Time Series
# Professional thesis / journal-ready version
# With linear trend lines
# 1080 dpi
#
# Variables:
# (a) Occurrence count
# (b) Species richness
# (c) Shannon diversity
# (d) Simpson diversity
# (e) Pielou evenness
#
# Input:
# Biodiversity_indices_seasonal_ecosystem_coupling_
# 2008_2025_1deg_masked_min10.nc


# MOUNT DRIVE 
from google.colab import drive
drive.mount('/content/drive')

# 0) Install packages
!pip -q install xarray netCDF4 h5netcdf dask scipy



# 1) Imports
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.stats import linregress



# 2) File paths
BIO_FILE = (
    "/content/drive/MyDrive/SAM_Thesis/Data/"
    "Biodiversity_Indices/"
    "Biodiversity_indices_seasonal_ecosystem_coupling_"
    "2008_2025_1deg_masked_min10.nc"
)

OUT_DIR = "/content/drive/MyDrive/SAM_Thesis/Fig"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FIG = os.path.join(
    OUT_DIR,
    "Fig27_monthly_biodiversity_time_series_"
    "2008_2025_with_trend_1080dpi.png"
)

OUT_CSV = os.path.join(
    OUT_DIR,
    "Fig27_monthly_biodiversity_time_series_"
    "2008_2025_with_trend.csv"
)



# 3) Main settings
START_YEAR = 2008
END_YEAR   = 2025
SAVE_DPI   = 1080

# Running mean settings
ROLLING_WINDOW = 3

# Minimum valid grid cells required
MIN_VALID_CELLS = 3



# 3.1) Confidence interval style
CI_COLOR = "#9ecae1"
CI_ALPHA = 0.34



# 3.2) Trend-line settings
ADD_LINEAR_TREND = True

# Select the time series used to calculate the trend:
# "mean"   = original monthly spatial mean
# "smooth" = 3-month running mean
TREND_SOURCE = "mean"

TREND_COLOR = "#2166ac"
TREND_LINEWIDTH = 1.45
TREND_LINESTYLE = "--"
TREND_ALPHA = 0.95

# Show slope, R² and p-value inside every subplot
SHOW_TREND_STATISTICS = True

# Display trend as change per decade
TREND_RATE_PERIOD = 10



# 4) Matplotlib style
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



# 5) Open and sort dataset
ds = xr.open_dataset(BIO_FILE)

# Make sure year and month are in ascending order
ds = ds.sortby("year")
ds = ds.sortby("month")

print(ds)



# 6) Required variables
required_vars = [
    "occurrence_count",
    "species_richness",
    "shannon_index",
    "simpson_index",
    "pielou_evenness"
]

for var in required_vars:
    if var not in ds.data_vars:
        raise ValueError(
            f"Required variable was not found: {var}"
        )

for dim in ["year", "month", "lat", "lon"]:
    if dim not in ds.dims:
        raise ValueError(
            f"Required dimension was not found: {dim}"
        )



# 7) Helper functions
def area_weighted_mean(da):
    """
    Calculate the latitude-area-weighted spatial mean.

    The function reduces the latitude and longitude
    dimensions while retaining year and month.
    """

    weights = np.cos(np.deg2rad(da["lat"]))
    valid = da.notnull()

    weighted_sum = (
        (da * weights)
        .where(valid)
        .sum(
            dim=("lat", "lon"),
            skipna=True
        )
    )

    weight_sum = (
        weights
        .where(valid)
        .sum(
            dim=("lat", "lon"),
            skipna=True
        )
    )

    weighted_mean = weighted_sum / weight_sum

    return weighted_mean


def area_weighted_std(da, mean_da):
    """
    Calculate the approximate latitude-area-weighted
    spatial standard deviation.
    """

    weights = np.cos(np.deg2rad(da["lat"]))
    valid = da.notnull()

    squared_difference = (
        (da - mean_da) ** 2
    ).where(valid)

    weighted_squared_sum = (
        (squared_difference * weights)
        .sum(
            dim=("lat", "lon"),
            skipna=True
        )
    )

    weight_sum = (
        weights
        .where(valid)
        .sum(
            dim=("lat", "lon"),
            skipna=True
        )
    )

    weighted_std = np.sqrt(
        weighted_squared_sum / weight_sum
    )

    return weighted_std


def valid_count(da):
    """
    Count valid latitude-longitude grid cells
    for every year-month.
    """

    return da.notnull().sum(
        dim=("lat", "lon")
    )


def datetime_to_elapsed_years(datetime_index, origin):
    """
    Convert datetime values into elapsed decimal years
    from a specified origin.
    """

    elapsed_days = (
        datetime_index - origin
    ) / np.timedelta64(1, "D")

    elapsed_years = (
        np.asarray(elapsed_days, dtype=float)
        / 365.2425
    )

    return elapsed_years


def calculate_linear_trend(series, full_time_index):
    """
    Calculate an ordinary least-squares linear trend.

    Parameters
    ----------
    series : pandas.Series
        Monthly biodiversity time series.

    full_time_index : pandas.DatetimeIndex
        Complete time axis on which the fitted trend
        line will be evaluated.

    Returns
    -------
    trend_series : pandas.Series
        Fitted trend values.

    slope_per_year : float
        Linear slope per year.

    intercept : float
        Regression intercept.

    r_squared : float
        Coefficient of determination.

    p_value : float
        Two-sided p-value for the slope.

    n_valid : int
        Number of valid observations used.
    """

    valid_mask = (
        series.notna()
        & np.isfinite(series)
    )

    n_valid = int(valid_mask.sum())

    if n_valid < 3:
        empty_trend = pd.Series(
            np.nan,
            index=full_time_index
        )

        return (
            empty_trend,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            n_valid
        )

    valid_time = series.index[valid_mask]
    valid_values = (
        series.loc[valid_mask]
        .astype(float)
        .to_numpy()
    )

    # Use the first date of the entire analysis period
    # as the regression origin
    origin = pd.Timestamp(
        f"{START_YEAR}-01-01"
    )

    x_valid = datetime_to_elapsed_years(
        valid_time,
        origin
    )

    regression = linregress(
        x_valid,
        valid_values
    )

    x_full = datetime_to_elapsed_years(
        full_time_index,
        origin
    )

    fitted_values = (
        regression.intercept
        + regression.slope * x_full
    )

    trend_series = pd.Series(
        fitted_values,
        index=full_time_index
    )

    r_squared = regression.rvalue ** 2

    return (
        trend_series,
        regression.slope,
        regression.intercept,
        r_squared,
        regression.pvalue,
        n_valid
    )


def format_trend_value(value):
    """
    Format trend values appropriately for variables
    with different numerical ranges.
    """

    absolute_value = abs(value)

    if absolute_value >= 100:
        return f"{value:.1f}"

    elif absolute_value >= 10:
        return f"{value:.2f}"

    elif absolute_value >= 1:
        return f"{value:.3f}"

    elif absolute_value >= 0.01:
        return f"{value:.4f}"

    else:
        return f"{value:.3e}"


def format_p_value(p_value):
    """
    Format statistical p-values.
    """

    if np.isnan(p_value):
        return "NA"

    elif p_value < 0.001:
        return "<0.001"

    else:
        return f"{p_value:.3f}"



# 8) Build datetime index
years = np.asarray(ds["year"].values)
months = np.asarray(ds["month"].values)

time_index = pd.DatetimeIndex([
    pd.Timestamp(
        year=int(year),
        month=int(month),
        day=1
    )
    for year in years
    for month in months
])



# 9) Create monthly dataframe
records = {}

for var in required_vars:

    da = ds[var]

    mean_da = area_weighted_mean(da)
    std_da  = area_weighted_std(
        da,
        mean_da
    )
    n_da = valid_count(da)

    # Force consistent year-month ordering before flattening
    mean_1d = (
        mean_da
        .transpose("year", "month")
        .values
        .reshape(-1)
    )

    std_1d = (
        std_da
        .transpose("year", "month")
        .values
        .reshape(-1)
    )

    n_1d = (
        n_da
        .transpose("year", "month")
        .values
        .reshape(-1)
    )

    # Mask months containing too few valid grid cells
    valid_month_mask = (
        n_1d >= MIN_VALID_CELLS
    )

    mean_1d = np.where(
        valid_month_mask,
        mean_1d,
        np.nan
    )

    std_1d = np.where(
        valid_month_mask,
        std_1d,
        np.nan
    )

    # Approximate 95% confidence interval
    standard_error = (
        std_1d
        / np.sqrt(np.maximum(n_1d, 1))
    )

    ci95_1d = 1.96 * standard_error

    records[f"{var}_mean"] = mean_1d
    records[f"{var}_ci95"] = ci95_1d
    records[f"{var}_n"] = n_1d


df = pd.DataFrame(
    records,
    index=time_index
)

df.index.name = "time"

# Limit analysis period
df = df.loc[
    (df.index.year >= START_YEAR)
    & (df.index.year <= END_YEAR)
].copy()



# 10) Add 3-month running mean
for var in required_vars:

    df[f"{var}_smooth"] = (
        df[f"{var}_mean"]
        .rolling(
            window=ROLLING_WINDOW,
            center=True,
            min_periods=2
        )
        .mean()
    )



# 11) Calculate trend statistics
trend_results = {}

for var in required_vars:

    if TREND_SOURCE.lower() == "smooth":
        trend_input = df[f"{var}_smooth"]

    else:
        trend_input = df[f"{var}_mean"]

    (
        trend_series,
        slope_per_year,
        intercept,
        r_squared,
        p_value,
        n_valid
    ) = calculate_linear_trend(
        trend_input,
        df.index
    )

    df[f"{var}_trend"] = trend_series

    trend_results[var] = {
        "slope_per_year": slope_per_year,
        "slope_per_decade": (
            slope_per_year
            * TREND_RATE_PERIOD
        ),
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "n_valid": n_valid
    }


# Save complete dataframe
df.to_csv(OUT_CSV)

print("\nSaved monthly dataframe:")
print(OUT_CSV)

print("\nValid monthly observations and trend results:")

for var in required_vars:

    result = trend_results[var]

    print(
        f"\n{var}:"
        f"\n  Valid months       = {result['n_valid']}"
        f"\n  Slope per year     = "
        f"{result['slope_per_year']:.6f}"
        f"\n  Slope per decade   = "
        f"{result['slope_per_decade']:.6f}"
        f"\n  R-squared          = "
        f"{result['r_squared']:.4f}"
        f"\n  p-value            = "
        f"{result['p_value']:.6f}"
    )



# 12) Display information
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
    nrows=5,
    ncols=1,
    figsize=(11.7, 7.6),
    sharex=True
)



# 14) Draw each subplot
for i, var in enumerate(required_vars):

    ax = axes[i]

    y = df[f"{var}_mean"]
    y_smooth = df[f"{var}_smooth"]
    ci = df[f"{var}_ci95"]
    y_trend = df[f"{var}_trend"]

    trend_result = trend_results[var]



    # 14.1) 95% confidence interval

    ax.fill_between(
        df.index,
        y - ci,
        y + ci,
        color=CI_COLOR,
        alpha=CI_ALPHA,
        linewidth=0,
        label="95% CI" if i == 0 else None,
        zorder=1
    )



    # 14.2) Monthly mean

    ax.plot(
        df.index,
        y,
        color="black",
        linewidth=0.78,
        alpha=0.80,
        label="Monthly mean" if i == 0 else None,
        zorder=3
    )



    # 14.3) Three-month running mean
 
    ax.plot(
        df.index,
        y_smooth,
        color="#d95f02",
        linewidth=1.55,
        alpha=0.96,
        label=(
            "3-month running mean"
            if i == 0
            else None
        ),
        zorder=4
    )


  
    # 14.4) Linear trend line

    if ADD_LINEAR_TREND:

        ax.plot(
            df.index,
            y_trend,
            color=TREND_COLOR,
            linewidth=TREND_LINEWIDTH,
            linestyle=TREND_LINESTYLE,
            alpha=TREND_ALPHA,
            label=(
                "Linear trend"
                if i == 0
                else None
            ),
            zorder=5
        )


  
    # 14.5) Panel label
    ax.text(
        0.005,
        0.88,
        plot_info[var]["panel"],
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=8
    )



    # 14.6) Panel title

    ax.text(
        0.052,
        0.88,
        plot_info[var]["title"],
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=8
    )


  
    # 14.7) Trend statistics

    if (
        ADD_LINEAR_TREND
        and SHOW_TREND_STATISTICS
        and np.isfinite(
            trend_result["slope_per_decade"]
        )
    ):

        slope_decade = (
            trend_result["slope_per_decade"]
        )

        slope_text = format_trend_value(
            slope_decade
        )

        p_text = format_p_value(
            trend_result["p_value"]
        )

        trend_text = (
            f"Trend = {slope_text} decade$^{{-1}}$\n"
            f"$R^2$ = {trend_result['r_squared']:.3f}; "
            f"$p$ = {p_text}"
        )

        ax.text(
            0.988,
            0.88,
            trend_text,
            transform=ax.transAxes,
            fontsize=7.5,
            ha="right",
            va="top",
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76
            },
            zorder=10
        )


   
    # 14.8) Y-axis label
    ax.set_ylabel(
        plot_info[var]["ylabel"]
    )



    # 14.9) Grid

    ax.grid(
        True,
        which="major",
        axis="both",
        linewidth=0.35,
        alpha=0.28
    )



    # 14.10) Spines

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


  
    # 14.11) Y-axis limits

    if plot_info[var]["ylim"] is not None:

        ax.set_ylim(
            plot_info[var]["ylim"]
        )

    else:

        valid_y = y.notna() & ci.notna()

        if valid_y.sum() > 0:

            lower_values = (
                y[valid_y] - ci[valid_y]
            )

            upper_values = (
                y[valid_y] + ci[valid_y]
            )

            ymin = np.nanpercentile(
                lower_values,
                1
            )

            ymax = np.nanpercentile(
                upper_values,
                99
            )

            # Include fitted trend values when
            # determining the plot range
            if ADD_LINEAR_TREND:

                finite_trend = y_trend[
                    np.isfinite(y_trend)
                ]

                if len(finite_trend) > 0:

                    ymin = min(
                        ymin,
                        finite_trend.min()
                    )

                    ymax = max(
                        ymax,
                        finite_trend.max()
                    )

            y_range = ymax - ymin

            if not np.isfinite(y_range) or y_range == 0:
                y_range = max(
                    abs(ymax),
                    1
                )

            padding = y_range * 0.12

            ax.set_ylim(
                max(0, ymin - padding),
                ymax + padding
            )


 
    # 14.12) Tick style
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.2,
        width=0.7
    )

    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        length=2.0,
        width=0.5
    )



# 15) X-axis formatting
axes[-1].set_xlabel("Year")

axes[-1].xaxis.set_major_locator(
    mdates.YearLocator(2)
)

axes[-1].xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)

axes[-1].xaxis.set_minor_locator(
    mdates.YearLocator(1)
)

for ax in axes:

    ax.set_xlim(
        pd.Timestamp(
            f"{START_YEAR}-01-01"
        ),
        pd.Timestamp(
            f"{END_YEAR}-12-31"
        )
    )



# 16) Figure legend

axes[0].legend(
    loc="upper right",
    frameon=False,
    ncol=4,
    bbox_to_anchor=(1.0, 1.23),
    handlelength=2.7,
    columnspacing=1.25,
    handletextpad=0.55
)



# 17) Final spacing

plt.subplots_adjust(
    left=0.085,
    right=0.985,
    top=0.94,
    bottom=0.085,
    hspace=0.20
)



# 18) Save and display

plt.savefig(
    OUT_FIG,
    dpi=SAVE_DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nSaved figure:")
print(OUT_FIG)

print("\nSaved trend-data CSV:")
print(OUT_CSV)
