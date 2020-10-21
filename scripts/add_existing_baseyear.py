# coding: utf-8

import logging
logger = logging.getLogger(__name__)
import pandas as pd
idx = pd.IndexSlice

import numpy as np
import scipy as sp
import xarray as xr
import re, os

from six import iteritems, string_types

import pypsa

import yaml

import pytz

from vresutils.costdata import annuity

from prepare_sector_network import prepare_costs

#First tell PyPSA that links can have multiple outputs by
#overriding the component_attrs. This can be done for
#as many buses as you need with format busi for i = 2,3,4,5,....
#See https://pypsa.org/doc/components.html#link-with-multiple-outputs-or-inputs

override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus3"] = ["string",np.nan,np.nan,"3rd bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency3"] = ["static or series","per unit",1.,"3rd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]
override_component_attrs["Link"].loc["p3"] = ["series","MW",0.,"3rd bus output","Output"]

override_component_attrs["Link"].loc["build_year"] = ["integer","year",np.nan,"build year","Input (optional)"]
override_component_attrs["Link"].loc["lifetime"] = ["float","years",np.nan,"build year","Input (optional)"]
override_component_attrs["Generator"].loc["build_year"] = ["integer","year",np.nan,"build year","Input (optional)"]
override_component_attrs["Generator"].loc["lifetime"] = ["float","years",np.nan,"build year","Input (optional)"]
override_component_attrs["Store"].loc["build_year"] = ["integer","year",np.nan,"build year","Input (optional)"]
override_component_attrs["Store"].loc["lifetime"] = ["float","years",np.nan,"build year","Input (optional)"]


def add_build_year_to_new_assets(n, baseyear):

    """

    Parameters
    ----------
    n : network

    baseyear: year in which optimized assets are built
    """

    #Give assets with lifetimes and no build year the build year baseyear
    for c in n.iterate_components(["Link", "Generator", "Store"]):

        assets = c.df.index[~c.df.lifetime.isna() & c.df.build_year.isna()]
        c.df.loc[assets, "build_year"] = baseyear

        #add -baseyear to name
        rename = pd.Series(c.df.index, c.df.index)
        rename[assets] += "-" + str(baseyear)
        c.df.rename(index=rename, inplace=True)

        #rename time-dependent
        for attr in n.component_attrs[c.name].index[(n.component_attrs[c.name].type.str.contains("series") &
                                                     n.component_attrs[c.name].status.str.contains("Input"))]:
            c.pnl[attr].rename(columns=rename, inplace=True)

def add_existing_renewables(df_agg):
    """
    Append existing renewables to the df_agg pd.DataFrame
    with the conventional power plants.
    """

    cc = pd.read_csv('data/Country_codes.csv',
                     index_col=0)

    carriers = {"solar" : "solar",
                "onwind" : "onwind",
                "offwind" : "offwind-ac"}

    for tech in ['solar', 'onwind', 'offwind']:
        carrier = carriers[tech]
        df = pd.read_csv('data/existing_infrastructure/{}_capacity_IRENA.csv'.format(tech),
                         index_col=0)
        df = df.fillna(0.)
        df.columns = df.columns.astype(int)

        df.rename(index={'Czechia':'Czech Republic',
                         'UK':'United Kingdom',
                         'Bosnia Herzg':'Bosnia Herzegovina',
                         'North Macedonia': 'Macedonia'}, inplace=True)

        df.rename(index=cc["2 letter code (ISO-3166-2)"], inplace=True)

        # calculate yearly differences
        df.insert(loc=0, value=.0, column='1999')
        df = df.diff(axis=1).drop('1999', axis=1)
        df = df.clip(lower=0)


        #distribute capacities among nodes according to capacity factor
        #weighting with nodal_fraction
        elec_buses = n.buses.index[n.buses.carrier == "AC"].union(n.buses.index[n.buses.carrier == "DC"])
        nodal_fraction = pd.Series(0.,elec_buses)

        for country in n.buses.loc[elec_buses,"country"].unique():
            gens = n.generators.index[(n.generators.index.str[:2] == country) & (n.generators.carrier == carrier)]
            cfs = n.generators_t.p_max_pu[gens].mean()
            cfs_key = cfs/cfs.sum()
            nodal_fraction.loc[n.generators.loc[gens,"bus"]] = cfs_key.values

        nodal_df = df.loc[n.buses.loc[elec_buses,"country"]]
        nodal_df.index = elec_buses
        nodal_df = nodal_df.multiply(nodal_fraction,axis=0)

        for year in nodal_df.columns:
            for node in nodal_df.index:
                name = f"{node}-{tech}-{year}"
                capacity = nodal_df.loc[node,year]
                if capacity > 0.:
                    df_agg.at[name,"Fueltype"] = tech
                    df_agg.at[name,"Capacity"] = capacity
                    df_agg.at[name,"YearCommissioned"] = year
                    df_agg.at[name,"cluster_bus"] = node

def add_power_capacities_installed_before_baseyear(n, grouping_years, costs, baseyear):
    """

    Parameters
    ----------
    n : network

    grouping_years : intervals to group existing capacities

    costs : to read lifetime to estimate YearDecomissioning


    """
    print("adding power capacities installed before baseyear")


    ### add conventional capacities using 'powerplants.csv'
    df_agg = pd.read_csv(snakemake.input.powerplants, index_col=0)

    rename_fuel = {'Hard Coal':'coal',
                   'Lignite':'lignite',
                   'Nuclear':'nuclear',
                   'Oil':'oil',
                   'OCGT':'OCGT',
                   'CCGT':'CCGT',
                   'Natural Gas':'gas',}
    fueltype_to_drop = ['Hydro',
                        'Wind',
                        'Solar',
                        'Geothermal',
                        'Bioenergy',
                        'Waste',
                        'Other',
                        'CCGT, Thermal']
    technology_to_drop = ['Pv',
                          'Storage Technologies']

    df_agg.drop(df_agg.index[df_agg.Fueltype.isin(fueltype_to_drop)],inplace=True)
    df_agg.drop(df_agg.index[df_agg.Technology.isin(technology_to_drop)],inplace=True)
    df_agg.Fueltype = df_agg.Fueltype.map(rename_fuel)

    #assign clustered bus
    busmap_s = pd.read_csv(snakemake.input.busmap_s, index_col=0).squeeze()
    busmap = pd.read_csv(snakemake.input.busmap, index_col=0).squeeze()
    clustermaps = busmap_s.map(busmap)
    clustermaps.index = clustermaps.index.astype(int)

    df_agg["cluster_bus"] = df_agg.bus.map(clustermaps)


    #include renewables in df_agg
    add_existing_renewables(df_agg)

    df_agg["grouping_year"] = np.take(grouping_years,
                                      np.digitize(df_agg.YearCommissioned,
                                                  grouping_years,
                                                  right=True))

    df = df_agg.pivot_table(index=["grouping_year",'Fueltype'], columns='cluster_bus',
                            values='Capacity', aggfunc='sum')

    carrier = {"OCGT" : "gas",
               "CCGT" : "gas",
               "coal" : "coal",
               "oil" : "oil",
               "lignite" : "lignite",
               "nuclear" : "uranium"}

    for grouping_year, generator in df.index:
        #capacity is the capacity in MW at each node for this
        capacity = df.loc[grouping_year, generator]
        capacity = capacity[~capacity.isna()]
        capacity = capacity[capacity > snakemake.config['existing_capacities']['threshold_capacity']]

        if generator in ['solar', 'onwind', 'offwind']:
            if generator =='offwind':
                p_max_pu=n.generators_t.p_max_pu[capacity.index + ' offwind-ac' + '-' + str(baseyear)]
            else:
                p_max_pu=n.generators_t.p_max_pu[capacity.index + ' ' + generator + '-' + str(baseyear)]

            n.madd("Generator",
                   capacity.index,
                   suffix=' ' + generator +"-"+ str(grouping_year),
                   bus=capacity.index,
                   carrier=generator,
                   p_nom=capacity,
                   marginal_cost=costs.at[generator,'VOM'],
                   capital_cost=costs.at[generator,'fixed'],
                   efficiency=costs.at[generator, 'efficiency'],
                   p_max_pu=p_max_pu.rename(columns=n.generators.bus),
                   build_year=grouping_year,
                   lifetime=costs.at[generator,'lifetime'])
        else:
            n.madd("Link",
                   capacity.index,
                   suffix= " " + generator +"-" + str(grouping_year),
                   bus0="EU " + carrier[generator],
                   bus1=capacity.index,
                   bus2="co2 atmosphere",
                   carrier=generator,
                   marginal_cost=costs.at[generator,'efficiency']*costs.at[generator,'VOM'], #NB: VOM is per MWel
                   capital_cost=costs.at[generator,'efficiency']*costs.at[generator,'fixed'], #NB: fixed cost is per MWel
                   p_nom=capacity/costs.at[generator,'efficiency'],
                   efficiency=costs.at[generator,'efficiency'],
                   efficiency2=costs.at[carrier[generator],'CO2 intensity'],
                   build_year=grouping_year,
                   lifetime=costs.at[generator,'lifetime'])


def add_heating_capacities_installed_before_baseyear(n, baseyear, grouping_years, ashp_cop, gshp_cop, time_dep_hp_cop, costs, default_lifetime):

    """

    Parameters
    ----------
    n : network

    baseyear: last year covered in the existing capacities database

    grouping_years : intervals to group existing capacities

    linear decomissioning of heating capacities from 2020 to 2045 is
    currently assumed

    heating capacities split between residential and services proportional
    to heating load in both
    50% capacities in rural busess 50% in urban buses
    """
    print("adding heating capacities installed before baseyear")

    # Add existing heating capacities, data comes from the study
    # "Mapping and analyses of the current and future (2020 - 2030)
    # heating/cooling fuel deployment (fossil/renewables) "
    # https://ec.europa.eu/energy/studies/mapping-and-analyses-current-and-future-2020-2030-heatingcooling-fuel-deployment_en?redir=1
    # file: "WP2_DataAnnex_1_BuildingTechs_ForPublication_201603.xls" -> "existing_heating_raw.csv".

    # retrieve existing heating capacities
    techs = ['gas boiler',
             'oil boiler',
             'resistive heater',
             'air heat pump',
             'ground heat pump']
    df = pd.read_csv('data/existing_infrastructure/existing_heating_raw.csv',
                     index_col=0,
                     header=0)
    # data for Albania, Montenegro and Macedonia not included in database
    df.loc['Albania']=np.nan
    df.loc['Montenegro']=np.nan
    df.loc['Macedonia']=np.nan
    df.fillna(0, inplace=True)
    df *= 1e3 # GW to MW

    cc = pd.read_csv('data/Country_codes.csv',
                     index_col=0)

    df.rename(index=cc["2 letter code (ISO-3166-2)"], inplace=True)

    # coal and oil boilers are assimilated to oil boilers
    df['oil boiler'] =df['oil boiler'] + df['coal boiler']
    df.drop(['coal boiler'], axis=1, inplace=True)

    # distribute technologies to nodes by population
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout,
                             index_col=0)
    pop_layout["ct"] = pop_layout.index.str[:2]
    ct_total = pop_layout.total.groupby(pop_layout["ct"]).sum()
    pop_layout["ct_total"] = pop_layout["ct"].map(ct_total.get)
    pop_layout["fraction"] = pop_layout["total"]/pop_layout["ct_total"]

    nodal_df = df.loc[pop_layout.ct]
    nodal_df.index = pop_layout.index
    nodal_df = nodal_df.multiply(pop_layout.fraction,axis=0)

    # split existing capacities between residential and services
    # proportional to energy demand
    ratio_residential=pd.Series([(n.loads_t.p_set.sum()['{} residential rural heat'.format(node)] /
               (n.loads_t.p_set.sum()['{} residential rural heat'.format(node)] +
                n.loads_t.p_set.sum()['{} services rural heat'.format(node)] ))
                   for node in nodal_df.index], index=nodal_df.index)

    for tech in techs:
        nodal_df['residential ' + tech] = nodal_df[tech]*ratio_residential
        nodal_df['services ' + tech] = nodal_df[tech]*(1-ratio_residential)

    nodes={}
    p_nom={}
    for name in ["residential rural",
                 "services rural",
                 "residential urban decentral",
                 "services urban decentral",
                 "urban central"]:

        name_type = "central" if name == "urban central" else "decentral"
        nodes[name] = pd.Index([n.buses.at[index,"location"] for index in n.buses.index[n.buses.index.str.contains(name) & n.buses.index.str.contains('heat')]])
        heat_pump_type = "air" if "urban" in name else "ground"
        heat_type= "residential" if "residential" in name else "services"

        if name == "urban central":
            p_nom[name]=nodal_df['air heat pump'][nodes[name]]
        else:
            p_nom[name] =  nodal_df['{} {} heat pump'.format(heat_type, heat_pump_type)][nodes[name]]

        # Add heat pumps
        costs_name = "{} {}-sourced heat pump".format("decentral", heat_pump_type)


        cop = {"air" : ashp_cop, "ground" : gshp_cop}
        efficiency = cop[heat_pump_type][nodes[name]] if time_dep_hp_cop else costs.at[costs_name,'efficiency']
        for i,grouping_year in enumerate(grouping_years):
            if int(grouping_year) + default_lifetime <= int(baseyear):
                ratio=0
            else:
                #installation is assumed to be linear for the past 25 years (default lifetime)
                ratio = (int(grouping_year)-int(grouping_years[i-1]))/default_lifetime

            n.madd("Link",
                   nodes[name],
                   suffix=" {} {} heat pump-{}".format(name,heat_pump_type, grouping_year),
                   bus0=nodes[name],
                   bus1=nodes[name] + " " + name + " heat",
                   carrier="{} {} heat pump".format(name,heat_pump_type),
                   efficiency=efficiency,
                   capital_cost=costs.at[costs_name,'efficiency']*costs.at[costs_name,'fixed'],
                   p_nom=p_nom[name]*ratio/costs.at[costs_name,'efficiency'],
                   build_year=int(grouping_year),
                   lifetime=costs.at[costs_name,'lifetime'])

            # add resistive heater, gas boilers and oil boilers
            # (50% capacities to rural buses, 50% to urban buses)
            n.madd("Link",
                   nodes[name],
                   suffix= " " + name + " resistive heater-{}".format(grouping_year),
                   bus0=nodes[name],
                   bus1=nodes[name] + " " + name + " heat",
                   carrier=name + " resistive heater",
                   efficiency=costs.at[name_type + ' resistive heater','efficiency'],
                   capital_cost=costs.at[name_type + ' resistive heater','efficiency']*costs.at[name_type + ' resistive heater','fixed'],
                   p_nom=0.5*nodal_df['{} resistive heater'.format(heat_type)][nodes[name]]*ratio/costs.at[name_type + ' resistive heater','efficiency'],
                   build_year=int(grouping_year),
                   lifetime=costs.at[costs_name,'lifetime'])

            n.madd("Link",
                   nodes[name],
                   suffix= " " + name + " gas boiler-{}".format(grouping_year),
                   bus0=["EU gas"]*len(nodes[name]),
                   bus1=nodes[name] + " " + name + " heat",
                   bus2="co2 atmosphere",
                   carrier=name + " gas boiler",
                   efficiency=costs.at[name_type + ' gas boiler','efficiency'],
                   efficiency2=costs.at['gas','CO2 intensity'],
                   capital_cost=costs.at[name_type + ' gas boiler','efficiency']*costs.at[name_type + ' gas boiler','fixed'],
                   p_nom=0.5*nodal_df['{} gas boiler'.format(heat_type)][nodes[name]]*ratio/costs.at[name_type + ' gas boiler','efficiency'],
                   build_year=int(grouping_year),
                   lifetime=costs.at[name_type + ' gas boiler','lifetime'])
            n.madd("Link",
                   nodes[name],
                   suffix=" " + name + " oil boiler-{}".format(grouping_year),
                   bus0=["EU oil"]*len(nodes[name]),
                   bus1=nodes[name] + " " + name + " heat",
                   bus2="co2 atmosphere",
                   carrier=name + " oil boiler",
                   efficiency=costs.at['decentral oil boiler','efficiency'],
                   efficiency2=costs.at['oil','CO2 intensity'],
                   capital_cost=costs.at['decentral oil boiler','efficiency']*costs.at['decentral oil boiler','fixed'],
                   p_nom=0.5*nodal_df['{} oil boiler'.format(heat_type)][nodes[name]]*ratio/costs.at['decentral oil boiler','efficiency'],
                   build_year=int(grouping_year),
                   lifetime=costs.at[name_type + ' gas boiler','lifetime'])

            # delete links with p_nom=nan corresponding to extra nodes in country
            n.mremove("Link", [index for index in n.links.index.to_list() if str(grouping_year) in index and np.isnan(n.links.p_nom[index])])

            # delete links if their lifetime is over and p_nom=0
            n.mremove("Link", [index for index in n.links.index.to_list() if str(grouping_year) in index and n.links.p_nom[index]<snakemake.config['existing_capacities']['threshold_capacity']])


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake
        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='', clusters='39', lv='1.0',
                           sector_opts='Co2L0-168H-T-H-B-I-solar3-dist1',
                           co2_budget_name='b30b3',
                           planning_horizons='2020'),
            input=dict(network='pypsa-eur-sec/results/test/prenetworks/{network}_s{simpl}_{clusters}_lv{lv}__{sector_opts}_{co2_budget_name}_{planning_horizons}.nc',
                       powerplants='pypsa-eur/resources/powerplants.csv',
                       busmap_s='pypsa-eur/resources/busmap_{network}_s{simpl}.csv',
                       busmap='pypsa-eur/resources/busmap_{network}_s{simpl}_{clusters}.csv',
                       costs='pypsa-eur-sec/data/costs/costs_{planning_horizons}.csv',
                       cop_air_total="pypsa-eur-sec/resources/cop_air_total_{network}_s{simpl}_{clusters}.nc",
                       cop_soil_total="pypsa-eur-sec/resources/cop_soil_total_{network}_s{simpl}_{clusters}.nc"),
            output=['pypsa-eur-sec/results/test/prenetworks_brownfield/{network}_s{simpl}_{clusters}_lv{lv}__{sector_opts}_{planning_horizons}.nc'],
        )
        import yaml
        with open('config.yaml', encoding='utf8') as f:
            snakemake.config = yaml.safe_load(f)


    logging.basicConfig(level=snakemake.config['logging_level'])

    options = snakemake.config["sector"]
    opts = snakemake.wildcards.sector_opts.split('-')

    baseyear= snakemake.config['scenario']["planning_horizons"][0]

    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=override_component_attrs)

    add_build_year_to_new_assets(n, baseyear)

    Nyears = n.snapshot_weightings.sum()/8760.
    costs = prepare_costs(snakemake.input.costs,
                          snakemake.config['costs']['USD2013_to_EUR2013'],
                          snakemake.config['costs']['discountrate'],
                          Nyears)

    grouping_years=snakemake.config['existing_capacities']['grouping_years']
    add_power_capacities_installed_before_baseyear(n, grouping_years, costs, baseyear)

    if "H" in opts:
        time_dep_hp_cop = options["time_dep_hp_cop"]
        ashp_cop = xr.open_dataarray(snakemake.input.cop_air_total).T.to_pandas().reindex(index=n.snapshots)
        gshp_cop = xr.open_dataarray(snakemake.input.cop_soil_total).T.to_pandas().reindex(index=n.snapshots)
        default_lifetime = snakemake.config['costs']['lifetime']
        add_heating_capacities_installed_before_baseyear(n, baseyear, grouping_years, ashp_cop, gshp_cop, time_dep_hp_cop, costs, default_lifetime)

    n.export_to_netcdf(snakemake.output[0])
