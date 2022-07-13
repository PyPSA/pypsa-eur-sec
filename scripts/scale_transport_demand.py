"""
scale (and maybe reshape) the default transport data
"""

import shutil
import pandas as pd

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helper import mock_snakemake

        snakemake = mock_snakemake(
            "scale_transport_demand",
            simpl="",
            clusters=37,
        )


# shutil.copy2(snakemake.input.transport_demand_unscaled, snakemake.output.transport_demand)
# shutil.copy2(snakemake.input.transport_data_unscaled, snakemake.output.transport_data)
# shutil.copy2(snakemake.input.avail_profile_unscaled, snakemake.output.avail_profile)
# shutil.copy2(snakemake.input.dsm_profile_unscaled, snakemake.output.dsm_profile)

data = pd.read_csv(snakemake.input.transport_demand_unscaled,index_col=0)
scaling = pd.read_csv(snakemake.input.transport_scaling,index_col=0)


if snakemake.config["scaling"]["transport"]["by_nation"]:
    node_names = data.columns
    for nation in scaling.index:
        national_nodes = [n for n in node_names if n.startswith(nation)]
        data[national_nodes] = data[national_nodes] * float(scaling.loc[nation])
else:
    data=data * float(scaling.loc["all_countries"])#snakemake.config["scaling"]["transport"]["total"]

data.to_csv(snakemake.output.transport_demand)