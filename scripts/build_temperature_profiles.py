"""Build temperature profiles."""

import geopandas as gpd
import atlite
import pandas as pd
import xarray as xr
import numpy as np
from helper import clean_invalid_geometries

if __name__ == '__main__':

    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.safe_load(f)
        snakemake.input = Dict()
        snakemake.output = Dict()

    time = pd.date_range(freq='h', **snakemake.config['snapshots'])
    cutout_config = snakemake.config['atlite']['cutout']
    cutout = atlite.Cutout(cutout_config).sel(time=time)

    clustered_regions = gpd.read_file(
        snakemake.input.regions_onshore).set_index('name').squeeze()

    clean_invalid_geometries(clustered_regions)

    I = cutout.indicatormatrix(clustered_regions)

    for area in ["total", "rural", "urban"]:

        pop_layout = xr.open_dataarray(snakemake.input[f'pop_layout_{area}'])

        stacked_pop = pop_layout.stack(spatial=('y', 'x'))
        M = I.T.dot(np.diag(I.dot(stacked_pop)))

        nonzero_sum = M.sum(axis=0, keepdims=True)
        nonzero_sum[nonzero_sum == 0.] = 1.
        M_tilde = M / nonzero_sum

        temp_air = cutout.temperature(
            matrix=M_tilde.T, index=clustered_regions.index)

        temp_air.to_netcdf(snakemake.output[f"temp_air_{area}"])

        temp_soil = cutout.soil_temperature(
            matrix=M_tilde.T, index=clustered_regions.index)

        temp_soil.to_netcdf(snakemake.output[f"temp_soil_{area}"])
