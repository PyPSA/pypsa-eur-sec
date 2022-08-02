"""
Scale the total demand of industrial production per country

TODO : add adjustments for target years
"""


import shutil
import pandas as pd

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helper import mock_snakemake

        snakemake = mock_snakemake(
            "scale_industrial_production_per_country_tomorrow",
            simpl="",
            clusters=37,
            planning_horizons=2030
        )

# shutil.copy2(snakemake.input.industrial_production_per_country_tomorrow_unscaled, snakemake.output.industrial_production_per_country_tomorrow)

prod = pd.read_csv(snakemake.input.industrial_production_per_country_tomorrow_unscaled, index_col=0)
scale = pd.read_csv(snakemake.input.industrial_production_scaling, index_col=0)
scale=scale.fillna(1) # if no scaling specified, fill in unit scaling


if snakemake.config["scaling"]["industry"]["by_nation"]:
    if snakemake.config["scaling"]["industry"]["by_sector"]:
        # scale each sector in each nation individualls
        prod = prod * scale.iloc[:-1,:-1]
    else:
        #scale by nation, but not sector
        for n in prod.index:
            prod.loc[n]=prod.loc[n]*scale['all_sectors'][n]
else:
    if snakemake.config["scaling"]["industry"]["by_sector"]:
        # scale by sector, all nations the same
        for sector in prod.columns:
            prod[sector]=prod[sector]*scale.loc["all_countries"][sector]
    else:
        #global scaling : all nations the same
        prod = prod * scale.loc["all_countries"]["all_sectors"]
        
prod.to_csv(snakemake.output.industrial_production_per_country_tomorrow)