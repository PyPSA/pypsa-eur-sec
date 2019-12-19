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



def remove_elec_base_techs(n):
    """remove conventional generators (e.g. OCGT) and storage units (e.g. batteries and H2)
    from base electricity-only network, since they're added here differently using links
    """
    to_keep = {"generators" : snakemake.config["plotting"]["vre_techs"],
               "storage_units" : snakemake.config["plotting"]["renewable_storage_techs"]}

    n.carriers = n.carriers.loc[to_keep["generators"] + to_keep["storage_units"]]

    for components, techs in iteritems(to_keep):
        df = getattr(n,components)
        to_remove = df.carrier.value_counts().index^techs
        print("removing {} with carrier {}".format(components,to_remove))
        df.drop(df.index[df.carrier.isin(to_remove)],inplace=True)


def add_co2_tracking(n):


    #minus sign because opposite to how fossil fuels used:
    #CH4 burning puts CH4 down, atmosphere up
    n.add("Carrier","co2",
          co2_emissions=-1.)

    #this tracks CO2 in the atmosphere
    n.add("Bus","co2 atmosphere",
          carrier="co2")

    #NB: can also be negative
    n.madd("Store",["co2 atmosphere"],
           e_nom_extendable=True,
           e_min_pu=-1,
           carrier="co2",
           bus="co2 atmosphere")

    #this tracks CO2 stored, e.g. underground
    n.add("Bus","co2 stored",
          carrier="co2 stored")

    #TODO move cost to data/costs.csv
    #TODO move maximum somewhere more transparent
    n.madd("Store",["co2 stored"],
           e_nom_extendable = True,
           e_nom_max=2e8,
           capital_cost=20.,
           carrier="co2 stored",
           bus="co2 stored")

    if options['co2_vent']:
        n.madd("Link",["co2 vent"],
               bus0="co2 stored",
               bus1="co2 atmosphere",
               carrier="co2 vent",
               efficiency=1.,
               p_nom_extendable=True)

    if options['dac']:
        #direct air capture consumes electricity to take CO2 from the air to the underground store
        #TODO do with cost from Breyer - later use elec and heat and capital cost
        n.madd("Link",["DAC"],
               bus0="co2 atmosphere",
               bus1="co2 stored",
               carrier="DAC",
               marginal_cost=75.,
               efficiency=1.,
               p_nom_extendable=True)


def add_co2limit(n, Nyears=1.,limit=0.):

    cts = pop_layout.ct.value_counts().index

    co2_limit = co2_totals.loc[cts, "electricity"].sum()

    if "T" in opts:
        co2_limit += co2_totals.loc[cts, [i+ " non-elec" for i in ["rail","road"]]].sum().sum()
    if "H" in opts:
        co2_limit += co2_totals.loc[cts, [i+ " non-elec" for i in ["residential","services"]]].sum().sum()
    if "I" in opts:
        co2_limit += co2_totals.loc[cts, ["industrial non-elec","industrial processes",
                                          "domestic aviation","international aviation",
                                          "domestic navigation","international navigation"]].sum().sum()

    co2_limit *= limit*Nyears

    n.add("GlobalConstraint", "CO2Limit",
          carrier_attribute="co2_emissions", sense="<=",
          constant=co2_limit)

def add_emission_prices(n, emission_prices=None, exclude_co2=False):
    assert False, "Needs to be fixed, adds NAN"

    if emission_prices is None:
        emission_prices = snakemake.config['costs']['emission_prices']
    if exclude_co2: emission_prices.pop('co2')
    ep = (pd.Series(emission_prices).rename(lambda x: x+'_emissions') * n.carriers).sum(axis=1)
    n.generators['marginal_cost'] += n.generators.carrier.map(ep)
    n.storage_units['marginal_cost'] += n.storage_units.carrier.map(ep)

def set_line_s_max_pu(n):
    # set n-1 security margin to 0.5 for 37 clusters and to 0.7 from 200 clusters
    # 128 reproduces 98% of line volume in TWkm, but clustering distortions inside node
    n_clusters = len(n.buses.index[n.buses.carrier == "AC"])
    s_max_pu = np.clip(0.5 + 0.2 * (n_clusters - 37) / (200 - 37), 0.5, 0.7)
    n.lines['s_max_pu'] = s_max_pu

    dc_b = n.links.carrier == 'DC'
    n.links.loc[dc_b, 'p_max_pu'] = snakemake.config['links']['p_max_pu']
    n.links.loc[dc_b, 'p_min_pu'] = - snakemake.config['links']['p_max_pu']

def set_line_volume_limit(n, lv):

    dc_b = n.links.carrier == 'DC'

    if lv != "opt":
        lv = float(lv)

        # Either line_volume cap or cost
        n.lines['capital_cost'] = 0.
        n.links.loc[dc_b,'capital_cost'] = 0.
    else:
        n.lines['capital_cost'] = (n.lines['length'] *
                                   costs.at['HVAC overhead', 'fixed'])

        #add HVDC inverter post factor, to maintain consistency with LV limit
        n.links.loc[dc_b, 'capital_cost'] = (n.links.loc[dc_b, 'length'] *
                                             costs.at['HVDC overhead', 'fixed'])# +
                                             #costs.at['HVDC inverter pair', 'fixed'])



    if lv != 1.0:
        lines_s_nom = n.lines.s_nom.where(
            n.lines.type == '',
            np.sqrt(3) * n.lines.num_parallel *
            n.lines.type.map(n.line_types.i_nom) *
            n.lines.bus0.map(n.buses.v_nom)
        )

        n.lines['s_nom_min'] = lines_s_nom

        n.links.loc[dc_b,'p_nom_min'] = n.links['p_nom']

        n.lines['s_nom_extendable'] = True
        n.links.loc[dc_b,'p_nom_extendable'] = True

        if lv != "opt":
            n.line_volume_limit = lv * ((lines_s_nom * n.lines['length']).sum() +
                                        n.links.loc[dc_b].eval('p_nom * length').sum())

    return n

def average_every_nhours(n, offset):
    logger.info('Resampling the network to {}'.format(offset))
    m = n.copy(with_time=False)

    #fix copying of network attributes
    #copied from pypsa/io.py, should be in pypsa/components.py#Network.copy()
    allowed_types = (float,int,bool) + string_types + tuple(np.typeDict.values())
    attrs = dict((attr, getattr(n, attr))
                 for attr in dir(n)
                 if (not attr.startswith("__") and
                     isinstance(getattr(n,attr), allowed_types)))
    for k,v in iteritems(attrs):
        setattr(m,k,v)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name+"_t")
        for k, df in iteritems(c.pnl):
            if not df.empty:
                if c.list_name == "stores" and k == "e_max_pu":
                    pnl[k] = df.resample(offset).min()
                elif c.list_name == "stores" and k == "e_min_pu":
                    pnl[k] = df.resample(offset).max()
                else:
                    pnl[k] = df.resample(offset).mean()

    return m


def generate_periodic_profiles(dt_index=pd.date_range("2011-01-01 00:00","2011-12-31 23:00",freq="H",tz="UTC"),
                               nodes=[],
                               weekly_profile=range(24*7)):
    """Give a 24*7 long list of weekly hourly profiles, generate this for
       each country for the period dt_index, taking account of time
       zones and Summer Time.

    """


    weekly_profile = pd.Series(weekly_profile,range(24*7))

    week_df = pd.DataFrame(index=dt_index,columns=nodes)

    for ct in nodes:
        week_df[ct] = [24*dt.weekday()+dt.hour for dt in dt_index.tz_convert(pytz.timezone(timezone_mappings[ct[:2]]))]
        week_df[ct] = week_df[ct].map(weekly_profile)

    return week_df



def shift_df(df,hours=1):
    """Works both on Series and DataFrame"""
    df = df.copy()
    df.values[:] = np.concatenate([df.values[-hours:],
                                   df.values[:-hours]])
    return df

def transport_degree_factor(temperature,deadband_lower=15,deadband_upper=20,
                            lower_degree_factor=0.5,
                            upper_degree_factor=1.6):

    """Work out how much energy demand in vehicles increases due to heating and cooling.

    There is a deadband where there is no increase.

    Degree factors are % increase in demand compared to no heating/cooling fuel consumption.

    Returns per unit increase in demand for each place and time
    """

    dd = temperature.copy()

    dd[(temperature > deadband_lower) & (temperature < deadband_upper)] = 0.

    dd[temperature < deadband_lower] = lower_degree_factor/100.*(deadband_lower-temperature[temperature < deadband_lower])

    dd[temperature > deadband_upper] = upper_degree_factor/100.*(temperature[temperature > deadband_upper]-deadband_upper)

    return dd


def prepare_data(network):


    ##############
    #Heating
    ##############


    ashp_cop = xr.open_dataarray(snakemake.input.cop_air_total).T.to_pandas().reindex(index=network.snapshots)
    gshp_cop = xr.open_dataarray(snakemake.input.cop_soil_total).T.to_pandas().reindex(index=network.snapshots)

    solar_thermal = xr.open_dataarray(snakemake.input.solar_thermal_total).T.to_pandas().reindex(index=network.snapshots)
    #1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
    solar_thermal = options['solar_cf_correction'] * solar_thermal/1e3

    energy_totals = pd.read_csv(snakemake.input.energy_totals_name,index_col=0)

    nodal_energy_totals = energy_totals.loc[pop_layout.ct].fillna(0.)
    nodal_energy_totals.index = pop_layout.index
    nodal_energy_totals = nodal_energy_totals.multiply(pop_layout.fraction,axis=0)

    #copy forward the daily average heat demand into each hour, so it can be multipled by the intraday profile
    daily_space_heat_demand = xr.open_dataarray(snakemake.input.heat_demand_total).T.to_pandas().reindex(index=network.snapshots, method="ffill")

    intraday_profiles = pd.read_csv(snakemake.input.heat_profile,index_col=0)

    sectors = ["residential","services"]
    uses = ["water","space"]

    heat_demand = {}
    electric_heat_supply = {}
    for sector in sectors:
        for use in uses:
            intraday_year_profile = generate_periodic_profiles(daily_space_heat_demand.index.tz_localize("UTC"),
                                                               nodes=daily_space_heat_demand.columns,
                                                               weekly_profile=(list(intraday_profiles["{} {} weekday".format(sector,use)])*5 + list(intraday_profiles["{} {} weekend".format(sector,use)])*2)).tz_localize(None)

            if use == "space":
                heat_demand_shape = daily_space_heat_demand*intraday_year_profile
                factor = options['space_heating_fraction']
            else:
                heat_demand_shape = intraday_year_profile
                factor = 1.

            heat_demand["{} {}".format(sector,use)] = factor*(heat_demand_shape/heat_demand_shape.sum()).multiply(nodal_energy_totals["total {} {}".format(sector,use)])*1e6
            electric_heat_supply["{} {}".format(sector,use)] = (heat_demand_shape/heat_demand_shape.sum()).multiply(nodal_energy_totals["electricity {} {}".format(sector,use)])*1e6

    heat_demand = pd.concat(heat_demand,axis=1)
    electric_heat_supply = pd.concat(electric_heat_supply,axis=1)

    #subtract from electricity load since heat demand already in heat_demand
    electric_nodes = n.loads.index[n.loads.carrier == "electricity"]
    n.loads_t.p_set[electric_nodes] = n.loads_t.p_set[electric_nodes] - electric_heat_supply.groupby(level=1,axis=1).sum()[electric_nodes]

    ##############
    #Transport
    ##############


    ## Get overall demand curve for all vehicles

    dir_name = "data/emobility/"
    traffic = pd.read_csv(os.path.join(dir_name,"KFZ__count"),skiprows=2)["count"]

    #Generate profiles
    transport_shape = generate_periodic_profiles(dt_index=network.snapshots.tz_localize("UTC"),
                                                 nodes=pop_layout.index,
                                                 weekly_profile=traffic.values).tz_localize(None)
    transport_shape = transport_shape/transport_shape.sum()

    transport_data = pd.read_csv(snakemake.input.transport_name,
                                 index_col=0)

    nodal_transport_data = transport_data.loc[pop_layout.ct].fillna(0.)
    nodal_transport_data.index = pop_layout.index
    nodal_transport_data["number cars"] = pop_layout["fraction"]*nodal_transport_data["number cars"]
    nodal_transport_data.loc[nodal_transport_data["average fuel efficiency"] == 0.,"average fuel efficiency"] = transport_data["average fuel efficiency"].mean()


    #electric motors are more efficient, so alter transport demand

    #kWh/km from EPA https://www.fueleconomy.gov/feg/ for Tesla Model S
    plug_to_wheels_eta = 0.20
    battery_to_wheels_eta = plug_to_wheels_eta*0.9

    efficiency_gain = nodal_transport_data["average fuel efficiency"]/battery_to_wheels_eta


    #get heating demand for correction to demand time series
    temperature = xr.open_dataarray(snakemake.input.temp_air_total).T.to_pandas()

    #correction factors for vehicle heating
    dd_ICE = transport_degree_factor(temperature,
                                     options['transport_heating_deadband_lower'],
                                     options['transport_heating_deadband_upper'],
                                     options['ICE_lower_degree_factor'],
                                     options['ICE_upper_degree_factor'])

    dd_EV = transport_degree_factor(temperature,
                                    options['transport_heating_deadband_lower'],
                                    options['transport_heating_deadband_upper'],
                                    options['EV_lower_degree_factor'],
                                    options['EV_upper_degree_factor'])

    #divide out the heating/cooling demand from ICE totals
    ICE_correction = (transport_shape*(1+dd_ICE)).sum()/transport_shape.sum()

    transport = (transport_shape.multiply(nodal_energy_totals["total road"] + nodal_energy_totals["total rail"]
                                         - nodal_energy_totals["electricity rail"])*1e6*Nyears).divide(efficiency_gain*ICE_correction)

    #multiply back in the heating/cooling demand for EVs
    transport = transport.multiply(1+dd_EV)


    ## derive plugged-in availability for PKW's (cars)

    traffic = pd.read_csv(os.path.join(dir_name,"Pkw__count"),skiprows=2)["count"]

    avail_max = 0.95

    avail_mean = 0.8

    avail = avail_max - (avail_max - avail_mean)*(traffic - traffic.min())/(traffic.mean() - traffic.min())

    avail_profile = generate_periodic_profiles(dt_index=network.snapshots.tz_localize("UTC"),
                                               nodes=pop_layout.index,
                                               weekly_profile=avail.values).tz_localize(None)

    dsm_week = np.zeros((24*7,))

    dsm_week[(np.arange(0,7,1)*24+options['dsm_restriction_time'])] = options['dsm_restriction_value']

    dsm_profile = generate_periodic_profiles(dt_index=network.snapshots.tz_localize("UTC"),
                                             nodes=pop_layout.index,
                                             weekly_profile=dsm_week).tz_localize(None)


    ###############
    #CO2
    ###############

    #1e6 to convert Mt to tCO2
    co2_totals = 1e6*pd.read_csv(snakemake.input.co2_totals_name,index_col=0)



    return nodal_energy_totals, heat_demand, ashp_cop, gshp_cop, solar_thermal, transport, avail_profile, dsm_profile, co2_totals, nodal_transport_data

def prepare_costs():

    #set all asset costs and other parameters
    costs = pd.read_csv(snakemake.input.costs,index_col=list(range(3))).sort_index()

    #correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"),"value"]*=1e3
    costs.loc[costs.unit.str.contains("USD"),"value"]*=snakemake.config['costs']['USD2013_to_EUR2013']

    cost_year = snakemake.config['costs']['year']

    costs = costs.loc[idx[:,cost_year,:],"value"].unstack(level=2).groupby(level="technology").sum(min_count=1)
    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : snakemake.config['costs']['discountrate'],
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : 25
    })

    costs["fixed"] = [(annuity(v["lifetime"],v["discount rate"])+v["FOM"]/100.)*v["investment"]*Nyears for i,v in costs.iterrows()]
    return costs


def add_generation(network):
    print("adding electricity generation")
    nodes = pop_layout.index

    conventionals = [("OCGT","gas")]

    for generator,carrier in [("OCGT","gas")]:
        network.add("Carrier",
                    carrier)

        network.add("Bus",
                    "EU " + carrier,
                    carrier=carrier)

        #use madd to get carrier inserted
        network.madd("Store",
                     ["EU " + carrier + " Store"],
                     bus=["EU " + carrier],
                     e_nom_extendable=True,
                     e_cyclic=True,
                     carrier=carrier,
                     capital_cost=0.) #could correct to e.g. 0.2 EUR/kWh * annuity and O&M

        network.add("Generator",
                    "EU fossil " + carrier,
                    bus="EU " + carrier,
                    p_nom_extendable=True,
                    carrier=carrier,
                    capital_cost=0.,
                    marginal_cost=costs.at[carrier,'fuel'])


        network.madd("Link",
                     nodes + " " + generator,
                     bus0=["EU " + carrier]*len(nodes),
                     bus1=nodes,
                     bus2="co2 atmosphere",
                     marginal_cost=costs.at[generator,'efficiency']*costs.at[generator,'VOM'], #NB: VOM is per MWel
                     capital_cost=costs.at[generator,'efficiency']*costs.at[generator,'fixed'], #NB: fixed cost is per MWel
                     p_nom_extendable=True,
                     carrier=generator,
                     efficiency=costs.at[generator,'efficiency'],
                     efficiency2=costs.at[carrier,'CO2 intensity'])


def add_storage(network):
    print("adding electricity storage")
    nodes = pop_layout.index

    network.add("Carrier","H2")


    network.madd("Bus",
                 nodes+ " H2",
                 carrier="H2")

    network.madd("Link",
                 nodes + " H2 Electrolysis",
                 bus1=nodes + " H2",
                 bus0=nodes,
                 p_nom_extendable=True,
                 carrier="H2 Electrolysis",
                 efficiency=costs.at["electrolysis","efficiency"],
                 capital_cost=costs.at["electrolysis","fixed"])

    network.madd("Link",
                 nodes + " H2 Fuel Cell",
                 bus0=nodes + " H2",
                 bus1=nodes,
                 p_nom_extendable=True,
                 carrier ="H2 Fuel Cell",
                 efficiency=costs.at["fuel cell","efficiency"],
                 capital_cost=costs.at["fuel cell","fixed"]*costs.at["fuel cell","efficiency"])  #NB: fixed cost is per MWel

    if options['hydrogen_underground_storage']:
        h2_capital_cost = costs.at["hydrogen underground storage","fixed"]
    else:
        h2_capital_cost = costs.at["hydrogen storage","fixed"]

    network.madd("Store",
                 nodes + " H2 Store",
                 bus=nodes + " H2",
                 e_nom_extendable=True,
                 e_cyclic=True,
                 carrier="H2 Store",
                 capital_cost=h2_capital_cost)

    h2_links = pd.DataFrame(columns=["bus0","bus1","length"])
    prefix = "H2 pipeline "
    connector = " -> "
    attrs = ["bus0","bus1","length"]

    candidates = pd.concat([n.lines[attrs],n.links.loc[n.links.carrier == "DC",attrs]])

    for candidate in candidates.index:
        buses = [candidates.at[candidate,"bus0"],candidates.at[candidate,"bus1"]]
        buses.sort()
        name = prefix + buses[0] + connector + buses[1]
        if name not in h2_links.index:
            h2_links.at[name,"bus0"] = buses[0]
            h2_links.at[name,"bus1"] = buses[1]
            h2_links.at[name,"length"] = candidates.at[candidate,"length"]

    #TODO Add efficiency losses
    network.madd("Link",
                 h2_links.index,
                 bus0=h2_links.bus0.values + " H2",
                 bus1=h2_links.bus1.values + " H2",
                 p_min_pu=-1,
                 p_nom_extendable=True,
                 length=h2_links.length.values,
                 capital_cost=costs.at['H2 pipeline','fixed']*h2_links.length.values,
                 carrier="H2 pipeline")


    network.add("Carrier","battery")

    network.madd("Bus",
                 nodes + " battery",
                 carrier="battery")

    network.madd("Store",
                 nodes + " battery",
                 bus=nodes + " battery",
                 e_cyclic=True,
                 e_nom_extendable=True,
                 carrier="battery",
                 capital_cost=costs.at['battery storage','fixed'])

    network.madd("Link",
                 nodes + " battery charger",
                 bus0=nodes,
                 bus1=nodes + " battery",
                 carrier="battery charger",
                 efficiency=costs.at['battery inverter','efficiency']**0.5,
                 capital_cost=costs.at['battery inverter','fixed'],
                 p_nom_extendable=True)

    network.madd("Link",
                 nodes + " battery discharger",
                 bus0=nodes + " battery",
                 bus1=nodes,
                 carrier="battery discharger",
                 efficiency=costs.at['battery inverter','efficiency']**0.5,
                 marginal_cost=options['marginal_cost_storage'],
                 p_nom_extendable=True)


    if options['methanation']:
        network.madd("Link",
                     nodes + " Sabatier",
                     bus0=nodes+" H2",
                     bus1=["EU gas"]*len(nodes),
                     bus2="co2 stored",
                     p_nom_extendable=True,
                     carrier="Sabatier",
                     efficiency=costs.at["methanation","efficiency"],
                     efficiency2=-costs.at["methanation","efficiency"]*costs.at['gas','CO2 intensity'],
                     capital_cost=costs.at["methanation","fixed"])

    if options['helmeth']:
        network.madd("Link",
                     nodes + " helmeth",
                     bus0=nodes,
                     bus1=["EU gas"]*len(nodes),
                     bus2="co2 stored",
                     carrier="helmeth",
                     p_nom_extendable=True,
                     efficiency=costs.at["helmeth","efficiency"],
                     efficiency2=-costs.at["helmeth","efficiency"]*costs.at['gas','CO2 intensity'],
                     capital_cost=costs.at["helmeth","fixed"])


    if options['SMR']:
        network.madd("Link",
                     nodes + " SMR CCS",
                     bus0=["EU gas"]*len(nodes),
                     bus1=nodes+" H2",
                     bus2="co2 atmosphere",
                     bus3="co2 stored",
                     p_nom_extendable=True,
                     carrier="SMR CCS",
                     efficiency=costs.at["SMR CCS","efficiency"],
                     efficiency2=costs.at['gas','CO2 intensity']*(1-options["ccs_fraction"]),
                     efficiency3=costs.at['gas','CO2 intensity']*options["ccs_fraction"],
                     capital_cost=costs.at["SMR CCS","fixed"])

        network.madd("Link",
                     nodes + " SMR",
                     bus0=["EU gas"]*len(nodes),
                     bus1=nodes+" H2",
                     bus2="co2 atmosphere",
                     p_nom_extendable=True,
                     carrier="SMR",
                     efficiency=costs.at["SMR","efficiency"],
                     efficiency2=costs.at['gas','CO2 intensity'],
                     capital_cost=costs.at["SMR","fixed"])


def add_transport(network):
    print("adding transport")
    nodes = pop_layout.index

    network.add("Carrier","Li ion")

    network.madd("Bus",
                 nodes,
                 suffix=" EV battery",
                 carrier="Li ion")

    network.madd("Load",
                 nodes,
                 suffix=" transport",
                 bus=nodes + " EV battery",
                 carrier="transport",
                 p_set=(1-options['transport_fuel_cell_share'])*(transport[nodes]+shift_df(transport[nodes],1)+shift_df(transport[nodes],2))/3.)

    p_nom = nodal_transport_data["number cars"]*0.011*(1-options['transport_fuel_cell_share'])  #3-phase charger with 11 kW * x% of time grid-connected

    network.madd("Link",
                 nodes,
                 suffix= " BEV charger",
                 bus0=nodes,
                 bus1=nodes + " EV battery",
                 p_nom=p_nom,
                 carrier="BEV charger",
                 p_max_pu=avail_profile[nodes],
                 efficiency=0.9, #[B]
                 #These were set non-zero to find LU infeasibility when availability = 0.25
                 #p_nom_extendable=True,
                 #p_nom_min=p_nom,
                 #capital_cost=1e6,  #i.e. so high it only gets built where necessary
    )

    if options["v2g"]:

        network.madd("Link",
                     nodes,
                     suffix=" V2G",
                     bus1=nodes,
                     bus0=nodes + " EV battery",
                     p_nom=p_nom,
                     carrier="V2G",
                     p_max_pu=avail_profile[nodes],
                     efficiency=0.9)  #[B]



    if options["bev"]:

        network.madd("Store",
                     nodes,
                     suffix=" battery storage",
                     bus=nodes + " EV battery",
                     carrier="battery storage",
                     e_cyclic=True,
                     e_nom=nodal_transport_data["number cars"]*0.05*options["bev_availability"]*(1-options['transport_fuel_cell_share']), #50 kWh battery http://www.zeit.de/mobilitaet/2014-10/auto-fahrzeug-bestand
                     e_max_pu=1,
                     e_min_pu=dsm_profile[nodes])


    if options['transport_fuel_cell_share'] != 0:

        network.madd("Load",
                     nodes,
                     suffix=" transport fuel cell",
                     bus=nodes + " H2",
                     carrier="transport fuel cell",
                     p_set=options['transport_fuel_cell_share']/costs.at["fuel cell","efficiency"]*transport[nodes])




def add_heat(network):

    print("adding heat")

    sectors = ["residential","services"]

    #stores the different groups of nodes
    nodes = {}

    #rural are areas with low heating density and individual heating
    #urban are areas with high heating density
    #urban can be split into district heating (central) and individual heating (decentral)
    for sector in sectors:
        nodes[sector + " rural"] = pop_layout.index

        if options["central"]:
            urban_decentral_ct = pd.Index(["ES","GR","PT","IT","BG"])
            nodes[sector + " urban decentral"] = pop_layout.index[pop_layout.ct.isin(urban_decentral_ct)]
        else:
            nodes[sector + " urban decentral"] = pop_layout.index

    #for central nodes, residential and services are aggregated
    nodes["urban central"] = pop_layout.index ^ nodes["residential urban decentral"]

    #NB: must add costs of central heating afterwards (EUR 400 / kWpeak, 50a, 1% FOM from Fraunhofer ISE)

    urban_fraction = options['central_fraction']*pop_layout["urban"]/(pop_layout[["urban","rural"]].sum(axis=1))

    for name in ["residential rural","services rural","residential urban decentral","services urban decentral","urban central"]:

        name_type = "central" if name == "urban central" else "decentral"

        network.add("Carrier",name + " heat")

        network.madd("Bus",
                     nodes[name] + " " + name + " heat",
                     carrier=name + " heat")

        ## Add heat load

        for sector in sectors:
            if "rural" in name:
                factor = 1-urban_fraction[nodes[name]]
            elif "urban" in name:
                factor = urban_fraction[nodes[name]]
            else:
                factor = None
            if sector in name:
                heat_load = heat_demand[[sector + " water",sector + " space"]].groupby(level=1,axis=1).sum()[nodes[name]].multiply(factor)

        if name == "urban central":
            heat_load = heat_demand.groupby(level=1,axis=1).sum()[nodes[name]].multiply(urban_fraction[nodes[name]]*(1+options['district_heating_loss']))

        network.madd("Load",
                     nodes[name],
                     suffix=" " + name + " heat",
                     bus=nodes[name] + " " + name + " heat",
                     carrier=name + " heat",
                     p_set=heat_load)


        ## Add heat pumps

        heat_pump_type = "air" if "urban" in name else "ground"

        costs_name = "{} {}-sourced heat pump".format(name_type,heat_pump_type)
        cop = {"air" : ashp_cop, "ground" : gshp_cop}
        efficiency = cop[heat_pump_type][nodes[name]] if options["time_dep_hp_cop"] else costs.at[costs_name,'efficiency']

        network.madd("Link",
                     nodes[name],
                     suffix=" {} {} heat pump".format(name,heat_pump_type),
                     bus0=nodes[name],
                     bus1=nodes[name] + " " + name + " heat",
                     carrier="{} {} heat pump".format(name,heat_pump_type),
                     efficiency=efficiency,
                     capital_cost=costs.at[costs_name,'efficiency']*costs.at[costs_name,'fixed'],
                     p_nom_extendable=True)


        if options["tes"]:

            network.add("Carrier",name + " water tanks")

            network.madd("Bus",
                         nodes[name] + " " + name + " water tanks",
                         carrier=name + " water tanks")

            network.madd("Link",
                         nodes[name] + " " + name + " water tanks charger",
                         bus0=nodes[name] + " " + name + " heat",
                         bus1=nodes[name] + " " + name + " water tanks",
                         efficiency=costs.at['water tank charger','efficiency'],
                         carrier=name + " water tanks charger",
                         p_nom_extendable=True)

            network.madd("Link",
                         nodes[name] + " " + name + " water tanks discharger",
                         bus0=nodes[name] + " " + name + " water tanks",
                         bus1=nodes[name] + " " + name + " heat",
                         carrier=name + " water tanks discharger",
                         efficiency=costs.at['water tank discharger','efficiency'],
                         p_nom_extendable=True)

            # [HP] 180 day time constant for centralised, 3 day for decentralised
            tes_time_constant_days = options["tes_tau"] if name_type == "decentral" else 180.

            network.madd("Store",
                         nodes[name] + " " + name + " water tanks",
                         bus=nodes[name] + " " + name + " water tanks",
                         e_cyclic=True,
                         e_nom_extendable=True,
                         carrier=name + " water tanks",
                         standing_loss=1-np.exp(-1/(24.*tes_time_constant_days)),
                         capital_cost=costs.at[name_type + ' water tank storage','fixed']/(1.17e-3*40)) #conversion from EUR/m^3 to EUR/MWh for 40 K diff and 1.17 kWh/m^3/K


        if options["boilers"]:

            network.madd("Link",
                         nodes[name] + " " + name + " resistive heater",
                         bus0=nodes[name],
                         bus1=nodes[name] + " " + name + " heat",
                         carrier=name + " resistive heater",
                         efficiency=costs.at[name_type + ' resistive heater','efficiency'],
                         capital_cost=costs.at[name_type + ' resistive heater','efficiency']*costs.at[name_type + ' resistive heater','fixed'],
                         p_nom_extendable=True)

            network.madd("Link",
                         nodes[name] + " " + name + " gas boiler",
                         p_nom_extendable=True,
                         bus0=["EU gas"]*len(nodes[name]),
                         bus1=nodes[name] + " " + name + " heat",
                         bus2="co2 atmosphere",
                         carrier=name + " gas boiler",
                         efficiency=costs.at[name_type + ' gas boiler','efficiency'],
                         efficiency2=costs.at['gas','CO2 intensity'],
                         capital_cost=costs.at[name_type + ' gas boiler','efficiency']*costs.at[name_type + ' gas boiler','fixed'])



        if options["solar_thermal"]:

            network.add("Carrier",name + " solar thermal")

            network.madd("Generator",
                         nodes[name],
                         suffix=" " + name + " solar thermal collector",
                         bus=nodes[name] + " " + name + " heat",
                         carrier=name + " solar thermal",
                         p_nom_extendable=True,
                         capital_cost=costs.at[name_type + ' solar thermal','fixed'],
                         p_max_pu=solar_thermal[nodes[name]])


        if options["chp"]:

            if name == "urban central":
                #add gas CHP; biomass CHP is added in biomass section
                network.madd("Link",
                             nodes[name] + " urban central gas CHP electric",
                             bus0="EU gas",
                             bus1=nodes[name],
                             bus2="co2 atmosphere",
                             carrier="urban central gas CHP electric",
                             p_nom_extendable=True,
                             capital_cost=costs.at['central gas CHP','fixed']*costs.at['central gas CHP','efficiency'],
                             marginal_cost=costs.at['central gas CHP','VOM'],
                             efficiency=costs.at['central gas CHP','efficiency'],
                             efficiency2=costs.at['gas','CO2 intensity'],
                             c_b=costs.at['central gas CHP','c_b'],
                             c_v=costs.at['central gas CHP','c_v'],
                             p_nom_ratio=costs.at['central gas CHP','p_nom_ratio'])

                network.madd("Link",
                             nodes[name] + " urban central gas CHP heat",
                             bus0="EU gas",
                             bus1=nodes[name] + " urban central heat",
                             bus2="co2 atmosphere",
                             carrier="urban central gas CHP heat",
                             p_nom_extendable=True,
                             marginal_cost=costs.at['central gas CHP','VOM'],
                             efficiency=costs.at['central gas CHP','efficiency']/costs.at['central gas CHP','c_v'],
                             efficiency2=costs.at['gas','CO2 intensity'])

                network.madd("Link",
                             nodes[name] + " urban central gas CHP CCS electric",
                             bus0="EU gas",
                             bus1=nodes[name],
                             bus2="co2 atmosphere",
                             bus3="co2 stored",
                             carrier="urban central gas CHP CCS electric",
                             p_nom_extendable=True,
                             capital_cost=costs.at['central gas CHP CCS','fixed']*costs.at['central gas CHP CCS','efficiency'],
                             marginal_cost=costs.at['central gas CHP CCS','VOM'],
                             efficiency=costs.at['central gas CHP CCS','efficiency'],
                             efficiency2=costs.at['gas','CO2 intensity']*(1-options["ccs_fraction"]),
                             efficiency3=costs.at['gas','CO2 intensity']*options["ccs_fraction"],
                             c_b=costs.at['central gas CHP CCS','c_b'],
                             c_v=costs.at['central gas CHP CCS','c_v'],
                             p_nom_ratio=costs.at['central gas CHP CCS','p_nom_ratio'])

                network.madd("Link",
                             nodes[name] + " urban central gas CHP CCS heat",
                             bus0="EU gas",
                             bus1=nodes[name] + " urban central heat",
                             bus2="co2 atmosphere",
                             bus3="co2 stored",
                             carrier="urban central gas CHP CCS heat",
                             p_nom_extendable=True,
                             marginal_cost=costs.at['central gas CHP CCS','VOM'],
                             efficiency=costs.at['central gas CHP CCS','efficiency']/costs.at['central gas CHP CCS','c_v'],
                             efficiency2=costs.at['gas','CO2 intensity']*(1-options["ccs_fraction"]),
                             efficiency3=costs.at['gas','CO2 intensity']*options["ccs_fraction"])

            else:
                network.madd("Link",
                             nodes[name] + " " + name + " micro gas CHP",
                             p_nom_extendable=True,
                             bus0="EU gas",
                             bus1=nodes[name],
                             bus2=nodes[name] + " " + name + " heat",
                             bus3="co2 atmosphere",
                             carrier=name + " micro gas CHP",
                             efficiency=costs.at['micro CHP','efficiency'],
                             efficiency2=costs.at['micro CHP','efficiency-heat'],
                             efficiency3=costs.at['gas','CO2 intensity'],
                             capital_cost=costs.at['micro CHP','fixed'])


    #NB: this currently doesn't work for pypsa-eur model
    if options['retrofitting']:

        retro_nodes = pd.Index(["DE"])

        space_heat_demand = space_heat_demand[retro_nodes]

        square_metres = population[retro_nodes]/population['DE']*5.7e9   #HPI 3.4e9m^2 for DE res, 2.3e9m^2 for tert https://doi.org/10.1016/j.rser.2013.09.012

        space_peak = space_heat_demand.max()

        space_pu = space_heat_demand.divide(space_peak)

        network.add("Carrier", "retrofitting")

        network.madd('Generator',
                     retro_nodes,
                     suffix=' retrofitting I',
                     bus=retro_nodes+' heat',
                     carrier="retrofitting",
                     p_nom_extendable=True,
                     p_nom_max=options['retroI-fraction']*space_peak*(1-urban_fraction),
                     p_max_pu=space_pu,
                     p_min_pu=space_pu,
                     capital_cost=options['retrofitting-cost_factor']*costs.at['retrofitting I','fixed']*square_metres/(options['retroI-fraction']*space_peak))

        network.madd('Generator',
                     retro_nodes,
                     suffix=' retrofitting II',
                     bus=retro_nodes+' heat',
                     carrier="retrofitting",
                     p_nom_extendable=True,
                     p_nom_max=options['retroII-fraction']*space_peak*(1-urban_fraction),
                     p_max_pu=space_pu,
                     p_min_pu=space_pu,
                     capital_cost=options['retrofitting-cost_factor']*costs.at['retrofitting II','fixed']*square_metres/(options['retroII-fraction']*space_peak))

        network.madd('Generator',
                     retro_nodes,
                     suffix=' urban retrofitting I',
                     bus=retro_nodes+' urban heat',
                     carrier="retrofitting",
                     p_nom_extendable=True,
                     p_nom_max=options['retroI-fraction']*space_peak*urban_fraction,
                     p_max_pu=space_pu,
                     p_min_pu=space_pu,
                     capital_cost=options['retrofitting-cost_factor']*costs.at['retrofitting I','fixed']*square_metres/(options['retroI-fraction']*space_peak))

        network.madd('Generator',
                     retro_nodes,
                     suffix=' urban retrofitting II',
                     bus=retro_nodes+' urban heat',
                     carrier="retrofitting",
                     p_nom_extendable=True,
                     p_nom_max=options['retroII-fraction']*space_peak*urban_fraction,
                     p_max_pu=space_pu,
                     p_min_pu=space_pu,
                     capital_cost=options['retrofitting-cost_factor']*costs.at['retrofitting II','fixed']*square_metres/(options['retroII-fraction']*space_peak))




def add_biomass(network):

    print("adding biomass")

    nodes = pop_layout.index

    #biomass distributed at country level - i.e. transport within country allowed
    cts = pop_layout.ct.value_counts().index

    biomass_potentials = pd.read_csv(snakemake.input.biomass_potentials,
                                     index_col=0)

    network.add("Carrier","biogas")
    network.add("Carrier","solid biomass")

    network.madd("Bus",
                 ["EU biogas"],
                 carrier="biogas")

    network.madd("Bus",
                 ["EU solid biomass"],
                 carrier="solid biomass")

    network.madd("Store",
                 ["EU biogas"],
                 bus="EU biogas",
                 carrier="biogas",
                 e_nom=biomass_potentials.loc[cts,"biogas"].sum(),
                 marginal_cost=costs.at['biogas','fuel'],
                 e_initial=biomass_potentials.loc[cts,"biogas"].sum())

    network.madd("Store",
                 ["EU solid biomass"],
                 bus="EU solid biomass",
                 carrier="solid biomass",
                 e_nom=biomass_potentials.loc[cts,"solid biomass"].sum(),
                 marginal_cost=costs.at['solid biomass','fuel'],
                 e_initial=biomass_potentials.loc[cts,"solid biomass"].sum())

    network.madd("Link",
                 ["biogas to gas"],
                 bus0="EU biogas",
                 bus1="EU gas",
                 bus2="co2 atmosphere",
                 carrier="biogas to gas",
                 efficiency2=-costs.at['gas','CO2 intensity'],
                 p_nom_extendable=True)


    #AC buses with district heating
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty and options["chp"]:
        urban_central = urban_central.str[:-len(" urban central heat")]

        network.madd("Link",
                     urban_central + " urban central solid biomass CHP electric",
                     bus0="EU solid biomass",
                     bus1=urban_central,
                     carrier="urban central solid biomass CHP electric",
                     p_nom_extendable=True,
                     capital_cost=costs.at['central solid biomass CHP','fixed']*costs.at['central solid biomass CHP','efficiency'],
                     marginal_cost=costs.at['central solid biomass CHP','VOM'],
                     efficiency=costs.at['central solid biomass CHP','efficiency'],
                     c_b=costs.at['central solid biomass CHP','c_b'],
                     c_v=costs.at['central solid biomass CHP','c_v'],
                     p_nom_ratio=costs.at['central solid biomass CHP','p_nom_ratio'])


        network.madd("Link",
                     urban_central + " urban central solid biomass CHP heat",
                     bus0="EU solid biomass",
                     bus1=urban_central + " urban central heat",
                     carrier="urban central solid biomass CHP heat",
                     p_nom_extendable=True,
                     marginal_cost=costs.at['central solid biomass CHP','VOM'],
                     efficiency=costs.at['central solid biomass CHP','efficiency']/costs.at['central solid biomass CHP','c_v'])

        network.madd("Link",
                     urban_central + " urban central solid biomass CHP CCS electric",
                     bus0="EU solid biomass",
                     bus1=urban_central,
                     bus2="co2 atmosphere",
                     bus3="co2 stored",
                     carrier="urban central solid biomass CHP CCS electric",
                     p_nom_extendable=True,
                     capital_cost=costs.at['central solid biomass CHP CCS','fixed']*costs.at['central solid biomass CHP CCS','efficiency'],
                     marginal_cost=costs.at['central solid biomass CHP CCS','VOM'],
                     efficiency=costs.at['central solid biomass CHP CCS','efficiency'],
                     efficiency2=-costs.at['solid biomass','CO2 intensity']*options["ccs_fraction"],
                     efficiency3=costs.at['solid biomass','CO2 intensity']*options["ccs_fraction"],
                     c_b=costs.at['central solid biomass CHP','c_b'],
                     c_v=costs.at['central solid biomass CHP','c_v'],
                     p_nom_ratio=costs.at['central solid biomass CHP','p_nom_ratio'])

        network.madd("Link",
                     urban_central + " urban central solid biomass CHP CCS heat",
                     bus0="EU solid biomass",
                     bus1=urban_central + " urban central heat",
                     bus2="co2 atmosphere",
                     bus3="co2 stored",
                     carrier="urban central solid biomass CHP CCS heat",
                     p_nom_extendable=True,
                     marginal_cost=costs.at['central solid biomass CHP CCS','VOM'],
                     efficiency=costs.at['central solid biomass CHP CCS','efficiency']/costs.at['central solid biomass CHP CCS','c_v'],
                     efficiency2=-costs.at['solid biomass','CO2 intensity']*options["ccs_fraction"],
                     efficiency3=costs.at['solid biomass','CO2 intensity']*options["ccs_fraction"])


def add_industry(network):

    print("adding industrial demand")

    nodes = pop_layout.index

    #1e6 to convert TWh to MWh
    industrial_demand = 1e6*pd.read_csv(snakemake.input.industrial_demand,
                                        index_col=0)

    solid_biomass_by_country = industrial_demand["solid biomass"].groupby(pop_layout.ct).sum()
    countries = solid_biomass_by_country.index

    network.madd("Bus",
                 ["solid biomass for industry"],
                 carrier="solid biomass for industry")

    network.madd("Load",
                 ["solid biomass for industry"],
                 bus="solid biomass for industry",
                 carrier="solid biomass for industry",
                 p_set=solid_biomass_by_country.sum()/8760.)

    network.madd("Link",
                 ["solid biomass for industry"],
                 bus0="EU solid biomass",
                 bus1="solid biomass for industry",
                 carrier="solid biomass for industry",
                 p_nom_extendable=True,
                 efficiency=1.)

    network.madd("Link",
                 ["solid biomass for industry CCS"],
                 bus0="EU solid biomass",
                 bus1="solid biomass for industry",
                 bus2="co2 atmosphere",
                 bus3="co2 stored",
                 carrier="solid biomass for industry CCS",
                 p_nom_extendable=True,
                 capital_cost=costs.at["industry CCS","fixed"]*costs.at['solid biomass','CO2 intensity']*8760, #8760 converts EUR/(tCO2/a) to EUR/(tCO2/h)
                 efficiency=0.9,
                 efficiency2=-costs.at['solid biomass','CO2 intensity']*options["ccs_fraction"],
                 efficiency3=costs.at['solid biomass','CO2 intensity']*options["ccs_fraction"])


    network.madd("Bus",
                 ["gas for industry"],
                 carrier="gas for industry")

    network.madd("Load",
                 ["gas for industry"],
                 bus="gas for industry",
                 carrier="gas for industry",
                 p_set=industrial_demand.loc[nodes,"methane"].sum()/8760.)

    network.madd("Link",
                 ["gas for industry"],
                 bus0="EU gas",
                 bus1="gas for industry",
                 bus2="co2 atmosphere",
                 carrier="gas for industry",
                 p_nom_extendable=True,
                 efficiency=1.,
                 efficiency2=costs.at['gas','CO2 intensity'])

    network.madd("Link",
                 ["gas for industry CCS"],
                 bus0="EU gas",
                 bus1="gas for industry",
                 bus2="co2 atmosphere",
                 bus3="co2 stored",
                 carrier="gas for industry CCS",
                 p_nom_extendable=True,
                 capital_cost=costs.at["industry CCS","fixed"]*costs.at['gas','CO2 intensity']*8760, #8760 converts EUR/(tCO2/a) to EUR/(tCO2/h)
                 efficiency=0.9,
                 efficiency2=costs.at['gas','CO2 intensity']*(1-options["ccs_fraction"]),
                 efficiency3=costs.at['gas','CO2 intensity']*options["ccs_fraction"])


    network.madd("Load",
                 nodes,
                 suffix=" H2 for industry",
                 bus=nodes + " H2",
                 carrier="H2 for industry",
                 p_set=industrial_demand.loc[nodes,"hydrogen"]/8760.)


    network.madd("Load",
                 nodes,
                 suffix=" H2 for shipping",
                 bus=nodes + " H2",
                 carrier="H2 for shipping",
                 p_set = nodal_energy_totals.loc[nodes,["total international navigation","total domestic navigation"]].sum(axis=1)*1e6*options['shipping_average_efficiency']/costs.at["fuel cell","efficiency"]/8760.)

    network.add("Bus",
                "Fischer-Tropsch",
                carrier="Fischer-Tropsch")

    #use madd to get carrier inserted
    network.madd("Store",
                 ["Fischer-Tropsch Store"],
                 bus="Fischer-Tropsch",
                 e_nom_extendable=True,
                 e_cyclic=True,
                 carrier="Fischer-Tropsch",
                 capital_cost=0.) #could correct to e.g. 0.001 EUR/kWh * annuity and O&M

    network.add("Generator",
                "fossil oil",
                bus="Fischer-Tropsch",
                p_nom_extendable=True,
                carrier="oil",
                capital_cost=0.,
                marginal_cost=costs.at["oil",'fuel'])

    network.madd("Link",
                 nodes + " Fischer-Tropsch",
                 bus0=nodes + " H2",
                 bus1="Fischer-Tropsch",
                 bus2="co2 stored",
                 carrier="Fischer-Tropsch",
                 efficiency=costs.at["Fischer-Tropsch",'efficiency'],
                 capital_cost=costs.at["Fischer-Tropsch",'fixed'],
                 efficiency2=-costs.at["oil",'CO2 intensity']*costs.at["Fischer-Tropsch",'efficiency'],
                 p_nom_extendable=True)

    network.madd("Load",
                 ["naphtha for industry"],
                 bus="Fischer-Tropsch",
                 carrier="naphtha for industry",
                 p_set = industrial_demand.loc[nodes,"naphtha"].sum()/8760.)

    network.madd("Load",
                 ["kerosene for aviation"],
                 bus="Fischer-Tropsch",
                 carrier="kerosene for aviation",
                 p_set = nodal_energy_totals.loc[nodes,["total international aviation","total domestic aviation"]].sum(axis=1).sum()*1e6/8760.)

    #NB: CO2 gets released again to atmosphere when plastics decay or kerosene is burned
    #except for the process emissions when naphtha is used for petrochemicals, which can be captured with other industry process emissions
    #tco2 per hour
    co2 = network.loads.loc[["naphtha for industry","kerosene for aviation"],"p_set"].sum()*costs.at["oil",'CO2 intensity'] - industrial_demand.loc[nodes,"process emission from feedstock"].sum()/8760.

    network.madd("Load",
                 ["Fischer-Tropsch emissions"],
                 bus="co2 atmosphere",
                 carrier="Fischer-Tropsch emissions",
                 p_set=-co2)

    network.madd("Load",
                 nodes,
                 suffix=" low-temperature heat for industry",
                 bus=[node + " urban central heat" if node + " urban central heat" in network.buses.index else node + " services urban decentral heat" for node in nodes],
                 carrier="low-temperature heat for industry",
                 p_set=industrial_demand.loc[nodes,"low-temperature heat"]/8760.)

    network.madd("Load",
                 nodes,
                 suffix=" industry new electricity",
                 bus=nodes,
                 carrier="industry new electricity",
                 p_set = (industrial_demand.loc[nodes,"electricity"]-industrial_demand.loc[nodes,"current electricity"])/8760.)

    network.madd("Bus",
                 ["process emissions"],
                 carrier="process emissions")

    #this should be process emissions fossil+feedstock
    #then need load on atmosphere for feedstock emissions that are currently going to atmosphere via Link Fischer-Tropsch demand
    network.madd("Load",
                 ["process emissions"],
                 bus="process emissions",
                 carrier="process emissions",
                 p_set = -industrial_demand.loc[nodes,["process emission","process emission from feedstock"]].sum(axis=1).sum()/8760.)

    network.madd("Link",
                 ["process emissions"],
                 bus0="process emissions",
                 bus1="co2 atmosphere",
                 carrier="process emissions",
                 p_nom_extendable=True,
                 efficiency=1.)

    #assume enough local waste heat for CCS
    network.madd("Link",
                 ["process emissions CCS"],
                 bus0="process emissions",
                 bus1="co2 atmosphere",
                 bus2="co2 stored",
                 carrier="process emissions CCS",
                 p_nom_extendable=True,
                 capital_cost=costs.at["industry CCS","fixed"]*8760, #8760 converts EUR/(tCO2/a) to EUR/(tCO2/h)
                 efficiency=(1-options["ccs_fraction"]),
                 efficiency2=options["ccs_fraction"])



def add_waste_heat(network):

    print("adding possibility to use industrial waste heat in district heating")

    #AC buses with district heating
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty:
        urban_central = urban_central.str[:-len(" urban central heat")]

        if options['use_fischer_tropsch_waste_heat']:
            n.links.loc[urban_central + " Fischer-Tropsch","bus3"] = urban_central + " urban central heat"
            n.links.loc[urban_central + " Fischer-Tropsch","efficiency3"] = 0.95 - n.links.loc[urban_central + " Fischer-Tropsch","efficiency"]

        if options['use_fuel_cell_waste_heat']:
            n.links.loc[urban_central + " H2 Fuel Cell","bus2"] = urban_central + " urban central heat"
            n.links.loc[urban_central + " H2 Fuel Cell","efficiency2"] = 0.95 - n.links.loc[urban_central + " H2 Fuel Cell","efficiency"]


def restrict_technology_potential(n,tech,limit):
    print("restricting potentials (p_nom_max) for {} to {} of technical potential".format(tech,limit))
    gens = n.generators.index[n.generators.carrier.str.contains(tech)]
    #beware if limit is 0 and p_nom_max is np.inf, 0*np.inf is nan
    n.generators.loc[gens,"p_nom_max"] *=limit

def decentral(n):
    n.lines.drop(n.lines.index,inplace=True)
    n.links.drop(n.links.index[n.links.carrier.isin(["DC","B2B"])],inplace=True)

def remove_h2_network(n):

    nodes = pop_layout.index

    n.links.drop(n.links.index[n.links.carrier.isin(["H2 pipeline"])],inplace=True)

    n.stores.drop(["EU H2 Store"],inplace=True)

    if options['hydrogen_underground_storage']:
        h2_capital_cost = costs.at["hydrogen underground storage","fixed"]
    else:
        h2_capital_cost = costs.at["hydrogen storage","fixed"]

    #put back nodal H2 storage
    n.madd("Store",
           nodes + " H2 Store",
           bus=nodes + " H2",
           e_nom_extendable=True,
           e_cyclic=True,
           carrier="H2 Store",
           capital_cost=h2_capital_cost)



if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake
        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='', clusters='37', lv='2', opts='Co2L-3H'),
            input=dict(network='../pypsa-eur/networks/{network}_s{simpl}_{clusters}.nc', timezone_mappings='data/timezone_mappings.csv'),
            output=['networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc']
        )
        import yaml
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)


    logging.basicConfig(level=snakemake.config['logging_level'])

    timezone_mappings = pd.read_csv(snakemake.input.timezone_mappings,index_col=0,squeeze=True,header=None)

    options = snakemake.config["sector"]

    opts = snakemake.wildcards.sector_opts.split('-')

    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=override_component_attrs)

    Nyears = n.snapshot_weightings.sum()/8760.

    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout,index_col=0)
    pop_layout["ct"] = pop_layout.index.str[:2]
    ct_total = pop_layout.total.groupby(pop_layout["ct"]).sum()
    pop_layout["ct_total"] = pop_layout["ct"].map(ct_total.get)
    pop_layout["fraction"] = pop_layout["total"]/pop_layout["ct_total"]

    costs = prepare_costs()

    remove_elec_base_techs(n)

    n.loads["carrier"] = "electricity"

    add_co2_tracking(n)

    add_generation(n)

    add_storage(n)

    for o in opts:
        if "space" in o:
            limit = o[o.find("space")+5:]
            limit = float(limit.replace("p",".").replace("m","-"))
            print(o,limit)
            options['space_heating_fraction'] = limit

    nodal_energy_totals, heat_demand, ashp_cop, gshp_cop, solar_thermal, transport, avail_profile, dsm_profile, co2_totals, nodal_transport_data = prepare_data(n)

    if "nodistrict" in opts:
        options["central"] = False

    if "T" in opts:
        add_transport(n)

    if "H" in opts:
        add_heat(n)

    if "B" in opts:
        add_biomass(n)

    if "I" in opts:
        add_industry(n)

    if "I" in opts and "H" in opts:
        add_waste_heat(n)

    if "decentral" in opts:
        decentral(n)

    if "noH2network" in opts:
        remove_h2_network(n)

    for o in opts:
        m = re.match(r'^\d+h$', o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break
    else:
        logger.info("No resampling")

    for o in opts:
        if "Co2L" in o:

            limit = o[o.find("Co2L")+4:]
            print(o,limit)
            if limit == "":
                limit = snakemake.config['co2_reduction']
            else:
                limit = float(limit.replace("p",".").replace("m","-"))
            add_co2limit(n, Nyears, limit)
        # add_emission_prices(n, exclude_co2=True)

    # if 'Ep' in opts:
    #     add_emission_prices(n)

        for tech in ["solar","onwind","offwind"]:
            if tech in o:
                limit = o[o.find(tech)+len(tech):]
                limit = float(limit.replace("p",".").replace("m","-"))
                restrict_technology_potential(n,tech,limit)

    n.export_to_netcdf(snakemake.output[0])
