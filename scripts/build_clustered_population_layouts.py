"""Build clustered population layouts."""

import geopandas as gpd
import xarray as xr
import pandas as pd
import atlite
from helper import clean_invalid_geometries


if __name__ == '__main__':

    cutout = atlite.Cutout(snakemake.config['atlite']['cutout'])

    clustered_regions = gpd.read_file(
        snakemake.input.regions_onshore).set_index('name').squeeze()

    clean_invalid_geometries(clustered_regions)

    I = cutout.indicatormatrix(clustered_regions)

    pop = {}
    for item in ["total", "urban", "rural"]:
        pop_layout = xr.open_dataarray(snakemake.input[f'pop_layout_{item}'])
        pop[item] = I.dot(pop_layout.stack(spatial=('y', 'x')))

    pop = pd.DataFrame(pop, index=clustered_regions.index)

    pop.to_csv(snakemake.output.clustered_pop_layout)
