"""Build solar thermal collector time series."""

import geopandas as gpd
import atlite
import pandas as pd
import xarray as xr
import numpy as np

if __name__ == '__main__':
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'build_solar_thermal_profiles',
            simpl='',
            clusters=48,
        )

    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.safe_load(f)
        snakemake.input = Dict()
        snakemake.output = Dict()

    config = snakemake.config['solar_thermal']

    weather_year = snakemake.wildcards.weather_year
    cutout_source = snakemake.config['cutout'].split('-')[1]

    if len(weather_year) > 0:
        time = pd.date_range(str(snakemake.wildcards.weather_year) + '-01-01',str(int(snakemake.wildcards.weather_year)+1) + '-01-01',freq='h')[0:-1]
        cutout_name = '../pypsa-eur/cutouts/europe-' + str(weather_year) + '-' + cutout_source + '.nc'
    else:
        time = pd.date_range('2013-01-01','2014-01-01',freq='h')[0:-1]
        cutout_name = '../pypsa-eur/cutouts/europe-2013-era5.nc'

    cutout = atlite.Cutout(cutout_name).sel(time=time)

    clustered_regions = gpd.read_file(
        snakemake.input.regions_onshore).set_index('name').buffer(0).squeeze()

    I = cutout.indicatormatrix(clustered_regions)

    for area in ["total", "rural", "urban"]:

        pop_layout = xr.open_dataarray(snakemake.input[f'pop_layout_{area}'])

        stacked_pop = pop_layout.stack(spatial=('y', 'x'))
        M = I.T.dot(np.diag(I.dot(stacked_pop)))

        nonzero_sum = M.sum(axis=0, keepdims=True)
        nonzero_sum[nonzero_sum == 0.] = 1.
        M_tilde = M / nonzero_sum

        solar_thermal = cutout.solar_thermal(**config, matrix=M_tilde.T,
                                             index=clustered_regions.index)

        solar_thermal.to_netcdf(snakemake.output[f"solar_thermal_{area}"])
