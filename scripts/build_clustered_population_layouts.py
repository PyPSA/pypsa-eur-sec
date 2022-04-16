"""Build clustered population layouts."""

import geopandas as gpd
import xarray as xr
import pandas as pd
import atlite


if __name__ == '__main__':
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'build_clustered_population_layouts',
            simpl='',
            clusters=48,
        )

    weather_year = snakemake.wildcards.weather_year
    cutout_source = snakemake.config['cutout'].split('-')[1]
    if len(weather_year) > 0:
        cutout_name = '../pypsa-eur/cutouts/europe-' + str(weather_year) + '-' + cutout_source + '.nc'
    else:
        cutout_name = '../pypsa-eur/cutouts/europe-2013-era5.nc'

    print(cutout_name)
    cutout = atlite.Cutout(cutout_name)

    clustered_regions = gpd.read_file(
        snakemake.input.regions_onshore).set_index('name').buffer(0).squeeze()

    I = cutout.indicatormatrix(clustered_regions)

    pop = {}
    for item in ["total", "urban", "rural"]:
        pop_layout = xr.open_dataarray(snakemake.input[f'pop_layout_{item}'])
        pop[item] = I.dot(pop_layout.stack(spatial=('y', 'x')))

    pop = pd.DataFrame(pop, index=clustered_regions.index)

    pop["ct"] = pop.index.str[:2]
    country_population = pop.total.groupby(pop.ct).sum()
    pop["fraction"] = pop.total / pop.ct.map(country_population)

    pop.to_csv(snakemake.output.clustered_pop_layout)
