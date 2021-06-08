# coding: utf-8

import logging
logger = logging.getLogger(__name__)

import pandas as pd
idx = pd.IndexSlice

import pypsa
import yaml

from add_existing_baseyear import add_build_year_to_new_assets
from helper import override_component_attrs


def add_brownfield(n, n_p, year):

    print("adding brownfield")

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):

        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.mremove(
            c.name,
            c.df.index[c.df.lifetime.isna()]
        )

        # remove assets whose build_year + lifetime < year
        n_p.mremove(
            c.name,
            c.df.index[c.df.build_year + c.df.lifetime < year]
        )

        # remove assets if their optimized nominal capacity is lower than a threshold
        # since CHP heat Link is proportional to CHP electric Link, make sure threshold is compatible
        chp_heat = c.df.index[(
            c.df[attr + "_nom_extendable"]
            & c.df.index.str.contains("urban central")
            & c.df.index.str.contains("CHP")
            & c.df.index.str.contains("heat")
        )]

        threshold = snakemake.config['existing_capacities']['threshold_capacity']
        
        if not chp_heat.empty:
            threshold_chp_heat = (threshold
                * c.df.efficiency[chp_heat.str.replace("heat", "electric")].values
                * c.df.p_nom_ratio[chp_heat.str.replace("heat", "electric")].values
                / c.df.efficiency[chp_heat].values
            )
            n_p.mremove(
                c.name,
                chp_heat[c.df.loc[chp_heat, attr + "_nom_opt"] < threshold_chp_heat]
            )
        
        n_p.mremove(
            c.name,
            c.df.index[c.df[attr + "_nom_extendable"] & ~c.df.index.isin(chp_heat) & (c.df[attr + "_nom_opt"] < threshold)]
        )

        # copy over assets but fix their capacity
        c.df[attr + "_nom"] = c.df[attr + "_nom_opt"]
        c.df[attr + "_nom_extendable"] = False

        n.import_components_from_dataframe(c.df, c.name)

        # copy time-dependent
        selection = (
            n.component_attrs[c.name].type.str.contains("series")
            & n.component_attrs[c.name].status.str.contains("Input")
        )
        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr], c.name, tattr)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake
        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='', clusters='37', lv='1.0',
                           sector_opts='Co2L0-168H-T-H-B-I-solar3-dist1',
                           co2_budget_name='go',
                           planning_horizons='2030'),
            input=dict(network='pypsa-eur-sec/results/test/prenetworks/elec_s{simpl}_{clusters}_lv{lv}__{sector_opts}_{co2_budget_name}_{planning_horizons}.nc',
                       network_p='pypsa-eur-sec/results/test/postnetworks/elec_s{simpl}_{clusters}_lv{lv}__{sector_opts}_{co2_budget_name}_2020.nc',
                       costs='pypsa-eur-sec/data/costs/costs_{planning_horizons}.csv',
                       cop_air_total="pypsa-eur-sec/resources/cop_air_total_elec_s{simpl}_{clusters}.nc",
                       cop_soil_total="pypsa-eur-sec/resources/cop_soil_total_elec_s{simpl}_{clusters}.nc"),
            output=['pypsa-eur-sec/results/test/prenetworks_brownfield/elec_s{simpl}_{clusters}_lv{lv}__{sector_opts}_{planning_horizons}.nc']
        )
        import yaml
        with open('config.yaml', encoding='utf8') as f:
            snakemake.config = yaml.safe_load(f)

    print(snakemake.input.network_p)
    logging.basicConfig(level=snakemake.config['logging_level'])

    year = int(snakemake.wildcards.planning_horizons)

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p, override_component_attrs=overrides)

    add_brownfield(n, n_p, year)

    n.export_to_netcdf(snakemake.output[0])
