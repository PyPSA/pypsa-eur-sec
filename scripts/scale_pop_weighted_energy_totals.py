"""
Scale energy totals
"""

import shutil
import pandas as pd

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helper import mock_snakemake

        snakemake = mock_snakemake(
            "scale_pop_weighted_energy_totals",
            simpl="",
            clusters=37,
            scal="Grubler"
        )


# data = pd.read_csv(snakemake.input.pop_weighted_energy_totals_unscaled,index_col=0).T
# scaling = pd.read_csv(snakemake.input.scaling,index_col=0)
# scaling = scaling.fillna(1).T

# if snakemake.config["scaling"]["energy_totals"]["by_nation"]:
    
#     node_names = data.columns
    
#     for nation in scaling.columns[:-1]:
#         national_nodes = [n for n in node_names if n.startswith(nation)]
        
#         if snakemake.config["scaling"]["energy_totals"]["by_sector"]:
#             for sector in data.index:
#                 data.loc[sector,national_nodes]*=scaling.loc[sector,nation]
#         else:
#             data[national_nodes] *= scaling[nation]["all_sectors"]
    
# else:
#     if snakemake.config["scaling"]["energy_totals"]["by_sector"]:
#         for sector in data.index:
#             data.loc[sector]*=scaling.loc[sector]["all_countries"]
#     else:
#         data = data * scaling.loc["all_sectors"]["all_countries"]
        
# data = data.T
# data.to_csv(snakemake.output[0],float_format="%.4f")

#%% New method : first scale all subsectors, then perform aggregation to scale all

# 1. Scale all subsectors

data = pd.read_csv(snakemake.input.pop_weighted_energy_totals_unscaled,index_col=0) #s
scaling = pd.read_csv(snakemake.input.scaling,index_col=0) #snakemake.input.scaling
scaling = scaling.fillna(1)
data_new = data.copy()

scaled_sectors = scaling.columns[:-1]

# snakemake.config["scaling"]["energy_totals"]["by_nation"] = False
# snakemake.config["scaling"]["energy_totals"]["by_sector"] = False


if snakemake.config["scaling"]["energy_totals"]["by_nation"]:
    
    node_names = data_new.index
    
    for nation in scaling.index[:-1]:
        national_nodes = [n for n in node_names if n.startswith(nation)]
        
        if snakemake.config["scaling"]["energy_totals"]["by_sector"]:
            for sector in scaled_sectors:
                data_new.loc[national_nodes,sector]*=scaling.loc[nation,sector]
        else:
            data_new.loc[national_nodes,scaled_sectors] *= scaling.loc[nation]["all_sectors"]
    
else:
    if snakemake.config["scaling"]["energy_totals"]["by_sector"]:
        for sector in scaled_sectors:
            data_new[sector]*=scaling.loc["all_countries"][sector]
    else:
        data_new[scaled_sectors] = data_new[scaled_sectors] * scaling.loc["all_countries"]["all_sectors"]
    
# 2. Aggregate subsectors to sector totals (for road, rail, aviation, and shipping)        

agg_dict = {
                "total road" : ["total two-wheel",	"total passenger cars",	"total other road passenger",	"total light duty road freight",	"total heavy duty road freight"],
                "total rail" : ["total rail passenger",	"total rail freight"],
                "total domestic aviation" : ["total domestic aviation passenger","total domestic aviation freight"],
                "total international aviation" : ["total international aviation passenger","total international aviation freight"]    
            }
    
for sect in agg_dict.keys():
    data_new[sect] = data_new[agg_dict[sect]].sum(axis=1)
  
data_new.to_csv(snakemake.output[0])