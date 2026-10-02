# Antarctic Shelf Seas Thesis — Objective 4

This repository contains the Python workflows used for **Objective 4 of the Antarctic shelf-seas thesis**, focusing on **occurrence-based marine biodiversity patterns and their environmental context across the Antarctic shelf seas**.

The workflows use biodiversity indices derived from GBIF occurrence records and combine them with oceanographic and cryospheric variables to examine spatial patterns, seasonal variability, regional contrasts, environmental relationships, temporal trends, and multivariate classification of the 13 Antarctic shelf seas.

Most analyses cover **2008–2025** and are designed primarily for execution in **Google Colab**, with input and output files referenced from Google Drive.

---

## Objective 4 analysis scope

The repository analyzes the following biodiversity indicators:

- occurrence count;
- species richness;
- Shannon diversity;
- Simpson diversity;
- Pielou evenness.

These indicators are analyzed through:

- total spatial biodiversity maps;
- seasonal climatologies;
- sea-wise seasonal comparisons;
- regional biodiversity contrasts;
- occurrence-supported coverage diagnostics;
- biodiversity relationships with SIC, chlorophyll-a, POC, and SST;
- seasonal biodiversity–oceanographic overlays;
- monthly biodiversity time series and linear trends;
- principal component analysis (PCA);
- K-means classification of Antarctic shelf-sea systems.

> **Important interpretation note:** the biodiversity metrics in this repository are derived from GBIF occurrence records. They are occurrence-based biodiversity indicators and should not be interpreted as direct abundance, biomass, population-density, or complete-community census measurements.

---

## Antarctic shelf-sea sectors

Most regional analyses use the same 13 Antarctic shelf seas:

1. Weddell Sea (WED)
2. King Haakon VII Sea (KHV)
3. Riiser-Larsen Sea (RLS)
4. Lazarev Sea (LAZ)
5. Cosmonauts Sea (COS)
6. Cooperation Sea (COO)
7. Davis Sea (DAV)
8. Mawson Sea (MAW)
9. D'Urville Sea (DUR)
10. Somov Sea (SOM)
11. Ross Sea (ROS)
12. Amundsen Sea (AMU)
13. Bellingshausen Sea (BEL)

The Ross Sea is handled as a date-line-crossing longitude sector where required.

---

## Repository contents

| Script | Main purpose |
|---|---|
| [`Figure-29.py`](./Figure-29.py) | Total occurrence-based biodiversity patterns across the Southern Ocean |
| [`Figure-30.py`](./Figure-30.py) | Seasonal biodiversity climatology and occurrence-supported coverage |
| [`Figure-31.py`](./Figure-31.py) | Sea-wise seasonal biodiversity comparison and regional contrasts |
| [`Figure-32.py`](./Figure-32.py) | Biodiversity relationships with SIC, chlorophyll-a, POC, and SST |
| [`Figure-33.py`](./Figure-33.py) | Seasonal biodiversity–oceanographic overlay |
| [`Figure-34.py`](./Figure-34.py) | Monthly biodiversity time series and linear trends |
| [`Figure-35.py`](./Figure-35.py) | Antarctic shelf-sea PCA and K-means classification |

---

# Figure workflows

## `Figure-29.py` — Occurrence-based biodiversity patterns

This workflow maps the spatial distribution of five biodiversity indicators across the Antarctic / Southern Ocean domain.

### Panels

```text
(a) Occurrence count
(b) Species richness
(c) Shannon diversity
(d) Simpson diversity
(e) Pielou evenness
```

### Main input

```text
Biodiversity_indices_TOTAL_MONTHLY5000_2008_2025_1deg_masked_min5.nc
```

### Main biodiversity variables

```text
occurrence_count
species_richness
shannon_index
simpson_index
pielou_evenness
```

### Spatial domain

The plotting workflow uses a South Polar Stereographic map covering approximately:

```text
90°S to 60°S
180°W to 180°E
```

The dataset coordinates are standardized and longitude is converted to:

```text
[-180°, 180°)
```

when required.

Missing values are displayed as white rather than being assigned zero.

### Display ranges

The source script uses fixed display ranges for interpretability, including:

```text
Occurrence count:  0–2000+
Species richness:  0–150+
Shannon index:     0–3+
Simpson index:     0–1
Pielou evenness:   0–1
```

---

## `Figure-30.py` — Seasonal biodiversity climatology

This workflow calculates seasonal climatology statistics for the five biodiversity indicators.

### Main input

```text
Biodiversity_indices_seasonal_ecosystem_coupling_2008_2025_1deg_masked_min10.nc
```

### Analysis period

```text
2008–2025
```

### Austral seasons

```text
Spring = SON
Summer = DJF
Autumn = MAM
Winter = JJA
```

### Spatial averaging

For each year-month and biodiversity variable, the script calculates a latitude-area-weighted spatial mean using:

```text
weight = cos(latitude)
```

Months are masked if they contain fewer than:

```text
MIN_VALID_CELLS = 3
```

valid 1° grid cells.

### Seasonal statistics

For each biodiversity variable and season, the code calculates:

- seasonal mean;
- approximate 95% confidence interval;
- number of valid monthly samples;
- mean number of valid grid cells.

The confidence interval is calculated as:

```text
mean ± 1.96 × standard error
```

### Occurrence-supported coverage

A sixth panel uses the number of valid `occurrence_count` grid cells as a direct sampling-support diagnostic.

This is important because seasonal biodiversity differences may partly reflect changing GBIF observation coverage.

### Main output table

```text
Fig28_seasonal_biodiversity_climatology_2008_2025.csv
```

---

## `Figure-31.py` — Sea-wise biodiversity comparison

This workflow compares biodiversity among the 13 Antarctic shelf seas.

### Main input

```text
Biodiversity_indices_seasonal_ecosystem_coupling_2008_2025_1deg_masked_min10.nc
```

### Panels

```text
(a) Seasonal Shannon diversity
(b) Seasonal species richness
(c) Regional biodiversity contrast
(d) Seasonal GBIF occurrence coverage
```

### Regional processing

For each sea, the code:

- selects the corresponding longitude sector;
- restricts latitude to 90°S–60°S;
- calculates cosine-latitude area-weighted monthly biodiversity means;
- retains occurrence-supported grid-cell counts;
- calculates seasonal means and approximate 95% confidence intervals;
- calculates overall 2008–2025 sea-wise biodiversity statistics.

### Minimum spatial support

The source uses:

```text
MIN_VALID_CELLS = 1
```

for this sea-wise workflow because GBIF occurrence coverage is sparse.

The coverage panel should therefore be considered when interpreting regional biodiversity differences.

### Main outputs

```text
Fig29_seawise_monthly_biodiversity_2008_2025.csv
Fig29_seawise_seasonal_biodiversity_2008_2025.csv
Fig29_seawise_overall_biodiversity_2008_2025.csv
```

The seas are sorted by mean Shannon diversity for the regional comparison.

---

## `Figure-32.py` — Biodiversity relationships with environmental drivers

This workflow evaluates sea-wise relationships between Shannon diversity and four environmental variables.

### Panels

```text
(a) Shannon diversity vs SIC
(b) Shannon diversity vs chlorophyll-a
(c) Shannon diversity vs POC
(d) Shannon diversity vs SST
```

### Main input

```text
Fig30_Biodiversity_Environmental_Drivers_ACTIVE_seawise_2008_2025.csv
```

The input table contains panel-specific Shannon and environmental values plus Boolean inclusion flags.

Examples include:

```text
shannon_for_sic
sic_for_shannon
use_sic_panel

shannon_for_chl
chl_for_shannon
use_chl_panel

shannon_for_poc
poc_for_shannon
use_poc_panel

shannon_for_sst
sst_for_shannon
use_sst_panel
```

### Regression method

For each panel, the script performs ordinary least-squares linear regression using:

```text
scipy.stats.linregress
```

and calculates:

- slope;
- intercept;
- correlation coefficient `r`;
- p-value;
- R²;
- number of valid seas.

The minimum regression sample size is:

```text
MIN_N_REG = 4
```

### Confidence interval

The displayed regression line includes an approximate 95% confidence interval based on residual standard error and Student's t distribution.

### Main statistical output

```text
Fig30_Biodiversity_Environmental_Drivers_regression_stats.csv
```

Sea labels are adjusted automatically using `adjustText` to reduce overlap.

---

## `Figure-33.py` — Seasonal biodiversity–oceanographic overlay

This workflow combines biodiversity with chlorophyll-a and sea-ice conditions.

### Main inputs

```text
CHL_monthly_2008_2025_SO.nc
OSTIA_sea_ice_fraction_monthly_2008_2025_SO.nc
Biodiversity_indices_seasonal_ecosystem_coupling_2008_2025_1deg_masked_min10.nc
```

### Seasonal panels

```text
Spring
Summer
Early autumn
```

with:

```text
Spring       = September–November
Summer       = December–February
Early autumn = March–April
```

### Figure components

Background:

```text
chlorophyll-a
```

Sea-ice contours:

```text
15%
50%
80%
```

Biodiversity overlays:

```text
species richness
Shannon diversity
```

### Biodiversity filtering

The source requires at least:

```text
MIN_BIO_VALID_OCCURRENCE = 10
```

occurrences for biodiversity display.

Species richness is shown with square markers and Shannon diversity with circular markers.

### Display-only smoothing

The chlorophyll and SIC fields are spatially coarsened and smoothed only for visualization.

The script uses NaN-aware Gaussian smoothing so missing data are not directly treated as zero during smoothing.

---

## `Figure-34.py` — Monthly biodiversity time series and trends

This workflow calculates Southern Ocean monthly biodiversity time series during **2008–2025**.

### Variables

```text
Occurrence count
Species richness
Shannon diversity
Simpson diversity
Pielou evenness
```

### Main input

```text
Biodiversity_indices_seasonal_ecosystem_coupling_2008_2025_1deg_masked_min10.nc
```

### Monthly spatial statistics

For each variable, the script calculates:

- cosine-latitude area-weighted spatial mean;
- approximate area-weighted spatial standard deviation;
- number of valid grid cells;
- approximate 95% confidence interval.

Months with fewer than:

```text
MIN_VALID_CELLS = 3
```

are masked.

### Running mean

The source uses a centered:

```text
3-month running mean
```

with at least two valid months.

### Trend method

The source uses ordinary least-squares linear regression with:

```text
scipy.stats.linregress
```

The default trend source is:

```text
TREND_SOURCE = "mean"
```

meaning the trend is fitted to the original monthly area-weighted mean rather than the smoothed series.

For each biodiversity indicator the script reports:

- slope per year;
- slope per decade;
- intercept;
- R²;
- p-value;
- number of valid monthly observations.

### Main output

```text
Fig27_monthly_biodiversity_time_series_2008_2025_with_trend.csv
```

This CSV contains the monthly means, approximate 95% confidence intervals, valid-cell counts, smoothed series, and fitted trend series.

---

## `Figure-35.py` — Antarctic shelf-sea classification and PCA

This workflow integrates physical, sea-ice, carbon, productivity, phytoplankton, and biodiversity indicators to classify the 13 Antarctic shelf seas.

### Main input table

```text
sea_classification_input_2008_2025.csv
```

### Bathymetry input

```text
GEBCO_2024_CF.nc
```

### PCA variables

The current workflow uses 12 variables:

```text
SIC_mean
SIC_trend_per_year
SIC_retreat_month_since_sep
SST_mean_C
MLD_mean
SIT_mean
Stratification_mean
POC_mean
PIC_mean
NPPint_mean
Diatom_contribution_percent
Shannon_mean
```

These variables integrate:

- sea-ice state;
- sea-ice trend;
- sea-ice retreat timing;
- SST;
- mixed-layer depth;
- sea-ice thickness;
- stratification;
- particulate organic carbon;
- particulate inorganic carbon;
- integrated NPP;
- diatom contribution;
- Shannon diversity.

### Preprocessing

Before PCA, the script:

- retains the 13 defined shelf seas;
- converts SST from Kelvin to Celsius if necessary;
- replaces non-finite values with missing;
- fills missing values by the median of each variable;
- standardizes all variables using `StandardScaler`.

### PCA

The source calculates:

```text
3 principal components
```

and reports explained variance for:

```text
PC1
PC2
PC3
PC1 + PC2
PC1 + PC2 + PC3
```

The signs of PC1 and PC2 are oriented after calculation so the axes have a consistent interpretation relative to the sea-ice and biological variables.

### Classification

The default setting is:

```text
CLASS_MODE = "kmeans"
```

with:

```text
n_clusters = 3
random_state = 42
n_init = 100
```

The three clusters are interpreted as:

```text
Ice-dominated systems
Ocean-dominated systems
Mixed systems
```

Cluster names are assigned using standardized cluster means and the source-defined ice/ocean scores.

The script also contains an optional conceptual classification map that can be selected instead of K-means.

### PCA display

The final figure includes:

- Antarctic shelf-sea classification map;
- PC1–PC2 scores;
- PC1–PC3 scores;
- PC2–PC3 scores;
- variable loadings in the PC1–PC2 plane;
- cluster ellipses;
- sea labels;
- GEBCO bathymetric context.

---

# Biodiversity variables

Across the repository, the principal occurrence-based biodiversity variables are:

### Occurrence count

Number of retained GBIF occurrence records represented in the biodiversity grid.

### Species richness

Number of distinct species represented in a grid cell or aggregated analysis unit.

### Shannon diversity

The source variable is:

```text
shannon_index
```

and is presented as:

```text
Shannon index (H')
```

### Simpson diversity

The source uses:

```text
simpson_index
```

displayed as:

```text
Simpson index (1-D)
```

### Pielou evenness

The source uses:

```text
pielou_evenness
```

displayed as:

```text
Pielou evenness (J')
```

These metrics remain dependent on the underlying GBIF occurrence sampling and quality-control framework.

---

# Seasonal definitions

The biodiversity climatology and sea-wise comparison workflows use four austral seasons:

```text
Spring = SON
Summer = DJF
Autumn = MAM
Winter = JJA
```

The seasonal biodiversity–oceanographic overlay uses:

```text
Spring       = SON
Summer       = DJF
Early autumn = March–April
```

This distinction is preserved because it is explicitly defined in the individual source scripts.

---

# Main biodiversity input products

The repository primarily uses two precomputed biodiversity NetCDF products.

### Total biodiversity product

```text
Biodiversity_indices_TOTAL_MONTHLY5000_2008_2025_1deg_masked_min5.nc
```

used by the total spatial biodiversity-pattern workflow.

### Seasonal ecosystem-coupling product

```text
Biodiversity_indices_seasonal_ecosystem_coupling_2008_2025_1deg_masked_min10.nc
```

used by the seasonal, sea-wise, overlay, and monthly-trend workflows.

The filenames indicate different minimum-record masks, and the scripts preserve those products as distinct analysis inputs.

---

# Environmental variables used with biodiversity

Depending on the workflow, Objective 4 combines biodiversity with:

- sea-ice concentration / fraction;
- chlorophyll-a;
- particulate organic carbon;
- sea-surface temperature;
- mixed-layer depth;
- sea-ice thickness;
- upper-ocean stratification;
- particulate inorganic carbon;
- integrated NPP;
- diatom contribution;
- sea-ice retreat timing;
- sea-ice trends.

---

# Analysis-ready supporting files

Several workflows depend on processed tables created by the wider thesis processing pipeline.

Examples include:

```text
Fig30_Biodiversity_Environmental_Drivers_ACTIVE_seawise_2008_2025.csv
sea_classification_input_2008_2025.csv
```

The sea-wise biodiversity script creates:

```text
Fig29_seawise_monthly_biodiversity_2008_2025.csv
Fig29_seawise_seasonal_biodiversity_2008_2025.csv
Fig29_seawise_overall_biodiversity_2008_2025.csv
```

The seasonal climatology workflow creates:

```text
Fig28_seasonal_biodiversity_climatology_2008_2025.csv
```

The monthly time-series workflow creates:

```text
Fig27_monthly_biodiversity_time_series_2008_2025_with_trend.csv
```

The biodiversity–environment regression workflow creates:

```text
Fig30_Biodiversity_Environmental_Drivers_regression_stats.csv
```

---

# Google Drive directory structure

The scripts primarily reference:

```text
/content/drive/MyDrive/SAM_Thesis/Data/
/content/drive/MyDrive/SAM_Thesis/Data/Biodiversity_Indices/
/content/drive/MyDrive/SAM_Thesis/Processed/
/content/drive/MyDrive/SAM_Thesis/Fig/
```

Before running a script, confirm that the required input files exist at the configured paths.

Large GBIF-derived, environmental, and bathymetric datasets are referenced by path and are not stored directly in this repository.

---

# Python environment

The workflows are designed primarily for **Google Colab**.

Packages used across the repository include:

```text
numpy
pandas
xarray
dask
netCDF4
h5netcdf
scipy
matplotlib
cartopy
scikit-learn
adjustText
```

`Figure-35.py` also installs projection/geospatial system packages required by Cartopy:

```text
libproj-dev
proj-data
proj-bin
libgeos-dev
```

Some scripts use Colab/Jupyter installation syntax such as:

```text
!pip
!apt-get
```

and therefore should be run in Google Colab or another compatible notebook environment unless those commands are converted to standard Python/system installation steps.

---

# Typical workflow

1. Open the required script in Google Colab.
2. Mount Google Drive.
3. Verify that the required biodiversity and environmental products exist.
4. Install missing Python/system packages.
5. Run the script from top to bottom.
6. Review the generated CSV and figure outputs in the configured directories.

A simplified analysis sequence is:

```text
GBIF-derived biodiversity NetCDF products
            ↓
spatial / seasonal biodiversity analysis
            ↓
sea-wise biodiversity summaries
            ↓
environmental-driver relationships
            ↓
monthly trend analysis
            ↓
integrated physical–biological classification
            ↓
PCA + K-means shelf-sea system classes
```

---

# Reproducibility notes

The source workflows preserve several important analysis choices:

- main analysis period: **2008–2025**;
- 13 Antarctic shelf-sea sectors;
- longitude normalization to `[-180°, 180°)`;
- cosine-latitude area weighting;
- explicit minimum-valid-grid-cell thresholds;
- occurrence-supported coverage diagnostics;
- approximate 95% confidence intervals based on standard errors;
- ordinary least-squares regression through `scipy.stats.linregress`;
- 3-month centered running means in the monthly biodiversity workflow;
- fixed random seed for K-means classification;
- median filling of missing PCA predictors;
- standardization before PCA and clustering;
- three-component PCA;
- three-cluster K-means classification.

Because GBIF sampling effort is spatially and temporally uneven, occurrence count and valid-grid-cell coverage should be considered when interpreting richness and diversity patterns.

---

# Repository structure

```text
.
├── Figure-29.py
├── Figure-30.py
├── Figure-31.py
├── Figure-32.py
├── Figure-33.py
├── Figure-34.py
├── Figure-35.py
├── .gitignore
├── LICENSE
└── README.md
```

---

# License

This repository is distributed under the license included in [`LICENSE`](./LICENSE).

---

# Repository scope

This repository provides the Python workflows for **Objective 4**, documenting occurrence-based Antarctic marine biodiversity patterns, seasonal and regional contrasts, biodiversity–environment relationships, monthly biodiversity trends, ecosystem overlays, and integrated multivariate classification of Antarctic shelf-sea systems.
