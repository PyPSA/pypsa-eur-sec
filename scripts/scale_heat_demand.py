"""
scale (and possibly reshape) the default heat demand time serieses
"""
import shutil
import netCDF4 as nc
import xarray as xr
import pandas as pd

# TODO: check if scaling "total" heat demand is redundant to scaling urban and rural
# they don't seem to add up at first glance


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helper import mock_snakemake

        snakemake = mock_snakemake(
            "scale_heat_demands",
            simpl="",
            clusters=37,
        )

# shutil.copy2(snakemake.input.heat_demand_urban_unscaled,snakemake.output.heat_demand_urban)
# shutil.copy2(snakemake.input.heat_demand_rural_unscaled,snakemake.output.heat_demand_rural)
# shutil.copy2(snakemake.input.heat_demand_total_unscaled,snakemake.output.heat_demand_total)

#%% Load Demand & Scaling
d_urban=xr.open_dataset(snakemake.input.heat_demand_urban_unscaled)
d_rural=xr.open_dataset(snakemake.input.heat_demand_rural_unscaled)
d_total=xr.open_dataset(snakemake.input.heat_demand_total_unscaled)

scale = pd.read_csv(snakemake.input.heat_demand_scaling, index_col=0)
scale = scale.fillna(1) # if no scaling specified, fill in unit scaling

#%% scaling

if snakemake.config['scaling']['heat']['by_nation']:
    # TODO : scaling at national levels - requires additional file structures
    node_names=d_urban.name.to_series().reset_index(drop=True)
    for nation in scale.index[:-1]:
        national_nodes = [n for n in node_names if n.startswith(nation)]
        d_urban.loc[dict(name=national_nodes)] = d_urban.loc[dict(name=national_nodes)] * scale["urban"][nation]
        d_rural.loc[dict(name=national_nodes)] = d_rural.loc[dict(name=national_nodes)] * scale["rural"][nation]
        d_total.loc[dict(name=national_nodes)] = d_total.loc[dict(name=national_nodes)] * scale["total"][nation]
    
     
else:
    # scale all countries the same
    d_urban['heat_demand'] = snakemake.config['scaling']['heat']['urban'] * d_urban['heat_demand']
    d_rural['heat_demand'] = snakemake.config['scaling']['heat']['rural'] * d_rural['heat_demand']
    d_total['heat_demand'] = snakemake.config['scaling']['heat']['total'] * d_total['heat_demand']

#%% writing to file

d_urban.to_netcdf(snakemake.output.heat_demand_urban,mode="w")
d_rural.to_netcdf(snakemake.output.heat_demand_rural,mode="w")
d_total.to_netcdf(snakemake.output.heat_demand_total,mode="w")
d_urban.close()
d_rural.close()
d_total.close()