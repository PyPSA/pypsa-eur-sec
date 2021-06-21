
import geopandas as gpd
import xarray as xr
import pandas as pd
import atlite
import helper

cutout_path = snakemake.config['atlite']['cutout_dir'] + "/" + snakemake.config['atlite']['cutout_name']+ ".nc"

cutout = atlite.Cutout(path=cutout_path)


clustered_busregions_as_geopd = gpd.read_file(snakemake.input.regions_onshore).set_index('name', drop=True)

clustered_busregions = pd.Series(clustered_busregions_as_geopd.geometry, index=clustered_busregions_as_geopd.index)

helper.clean_invalid_geometries(clustered_busregions)

I = cutout.indicatormatrix(clustered_busregions)


items = ["total","urban","rural"]

pop = pd.DataFrame(columns=items,
                   index=clustered_busregions.index)


for item in items:
    pop_layout = xr.open_dataarray(snakemake.input['pop_layout_'+item])
    pop[item] = I.dot(pop_layout.stack(spatial=('y', 'x')))

pop.to_csv(snakemake.output.clustered_pop_layout)
