# %% [markdown]
# # Project: Spectral Analysis of AMO and NAO
#
# This project compares observed NOAA AMO and NAO indices with indices
# derived from TraCE-21K-II modeled sea surface temperature and sea level
# pressure. This exploratory analysis looks at each dataset's structure,
# spatial and temporal coverage, resolution, and suitability for comparison.

# %% Imports and data path
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
data_folder = Path(r"C:\Users\dunwo\NAO-AMO-Spectral-Analysis\DATA")

# runfile C:/Users/dunwo/NAO-AMO-Spectral-Analysis/EDA_project_Outline_JD.py --wdir

# % Observed NAO
nao_obs = xr.open_dataset(data_folder / "nao.long.nc")
#print(nao_obs)
#print(nao_obs["value"].attrs)
#print("Missing values:", int(nao_obs["value"].isnull().sum()))
#print(nao_obs["value"].to_series().describe())


# %% Observed NAO time series
#print("Missing months:")
#print(nao_obs.time.where(nao_obs["value"].isnull(), drop=True).values)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nao_obs.time, nao_obs["value"], linewidth=0.6)
ax.axhline(0, color="black", linewidth=0.8)

ax.set(
    title="Observed Monthly NAO, January 1821–June 2025",
    xlabel="Year",
    ylabel="NAO index")

ax.text(
    0.02, 0.97,
    "Source: NOAA PSL\n"
    "Resolution: monthly\n",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="lightgray"))

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Observed NAO
#
# This dataset is the monthly observed NAO index from the Climatic
# Research Unit (CRU), distributed by NOAA's Physical Sciences Laboratory.
# It is available at https://psl.noaa.gov/data/timeseries/month/DS/NAO/
# and was downloaded as a NetCDF file with name `nao.long.nc`.
#
# The file contains one variable, `value`, along a time dimension with
# 2,454 monthly entries spanning January 1821 to June 2025.
# 2,438 months are valid and 16 are listed as missing.
#
# Spatially, this index represents the sea level pressure contrast between
# Gibraltar and southwest Iceland (Reykjavik). It is calculated as the difference
# between their normalized pressure records. It describes North Atlantic
# atmospheric variability using station records rather than a spatial grid.


# %% Observed AMO
amo_table = pd.read_csv(
    data_folder / "amon.us.long.data.txt",
    sep=r"\s+",
    skiprows=1,
    nrows=168,
    header=None,
    index_col=0,
    na_values=-99.99)

# Convert yearly rows into a monthly time series
amo_obs = pd.Series(
    amo_table.to_numpy().ravel(),
    index=pd.date_range("1856-01-01", periods=amo_table.size, freq="MS"),
    name="AMO")

#print(amo_obs.describe())
#print("Missing months:", amo_obs.isna().sum())
#print("Last valid month:", amo_obs.last_valid_index())


# %% Observed AMO time series
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(amo_obs.index, amo_obs, linewidth=0.7)
ax.axhline(0, color="black", linewidth=0.8)

ax.set(
    title="Observed Monthly AMO, January 1856–January 2023",
    xlabel="Year",
    ylabel="AMO index (°C)")

ax.text(
    0.02, 0.97,
    "Source: NOAA PSL, Kaplan SST\n"
    "Resolution: monthly\n",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="lightgray"))

fig.tight_layout()
plt.show()


# %% [markdown]
# ## Observed AMO
#
# This dataset contains the unsmoothed monthly AMO index calculated by
# NOAA PSL from Kaplan SST Version 2. It was downloaded as `amon.us.long.data.txt` file
# from https://psl.noaa.gov/data/timeseries/AMO/
#
# The index represents area "weighted North Atlantic sea surface temperature"
# variability over approximately 0 – 70°N, in degrees Celsius.
# The source SST data have a 5° × 5° grid, but this file contains a single
# index rather than individual grid cells.
#
# The text file contains one row per year and 12 monthly columns.
# There are 2,005 valid monthly values from January 1856 to January 2023.
# February - December 2023 contain missing values with placeholders (-99.99),
# which are converted to NaN when loading.
#
# NOAA has already detrended this index. Unsmoothed means that no running
# average has been applied; its temporal resolution is one month.


# %% Modeled sea level pressure
psl_model = xr.open_dataset(data_folder / "TraCE-21K-II.monthly.PSL.nc", decode_times=False)

#print(psl_model)
#print(psl_model["time"].attrs)
#print(psl_model["PSL"].attrs)
#print(psl_model.attrs)
#print("Time endpoints:", psl_model.time.values[[0, -1]])
#print("First time steps:", np.diff(psl_model.time.values[:5]))


# %% Modeled pressure: North Atlantic spatial coverage
# Plot the final monthly pressure field; convert Pa to hPa.
pressure = psl_model["PSL"].isel(time=-1) / 100

fig, ax = plt.subplots(
    figsize=(10, 6),
    subplot_kw={"projection": ccrs.PlateCarree()})
pressure.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    cbar_kwargs={"label": "Sea level pressure (hPa)"})
ax.set_extent([-80, 20, 20, 80], crs=ccrs.PlateCarree())
ax.coastlines()
grid = ax.gridlines(draw_labels=True, linestyle=":", alpha=0.5)
grid.top_labels = False
grid.right_labels = False

# Reference locations for the observed station-based NAO index.
ax.scatter(
    [-5.35, -21.94], [36.14, 64.15],
    color="red", edgecolor="black", s=55,
    transform=ccrs.PlateCarree(), zorder=5)
ax.text(-3, 36.14, "Gibraltar", transform=ccrs.PlateCarree(), fontsize=10)
ax.text(-19, 64.15, "Reykjavik",transform=ccrs.PlateCarree(), fontsize=10)
ax.set_title("TraCE-21K-II: Final Monthly North Atlantic Sea Level Pressure")
fig.tight_layout()
plt.show()

# %% [markdown]
# The map shows monthly sea level pressure field from TraCE-21K-II
# over the North Atlantic. Pressure is displayed in hPa, converted from Pa.
# The visible cells show the model's spatial resolution.
#
# Gibraltar and Reykjavik are marked as reference locations for the observed
# stationcvbased NAO index. Model pressure near these locations will be used
# to derive a comparable modeled index.
#
# The figure generated shows the spatial coverage and its pressure field.
# Variability through time will be examined after extracting the pressure
# records and calculating the modeled NAO index.


# %% Extract modeled pressure at NAO parameters
# Grab nearest grid cells across all months and convert Pa to hPa.
# Longitude uses the model's 0 - 360E system.
psl_gibraltar = (
    psl_model["PSL"]
    .sel(lat=36.14, lon=354.65, method="nearest")
    .load() / 100)

psl_reykjavik = (
    psl_model["PSL"]
    .sel(lat=64.15, lon=338.06, method="nearest")
    .load() / 100)

# Report selected grid locations, record lengths, and missing months.
for name, record in [
    ("Gibraltar", psl_gibraltar),
    ("Reykjavik", psl_reykjavik)]:
    print(
        f"{name}: {record.lat.item():.2f}°N, "
        f"{record.lon.item():.2f}°E | "
        f"{record.size:,} months | "
        f"{record.isnull().sum().item()} missing")