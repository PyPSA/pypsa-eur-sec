
from six import iteritems

import sys

sys.path.append("../pypsa-eur/scripts")

import pandas as pd

import numpy as np

import pypsa

from vresutils.costdata import annuity

from prepare_sector_network import generate_periodic_profiles

from add_electricity import load_costs

import yaml

idx = pd.IndexSlice

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}

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
override_component_attrs["StorageUnit"].loc["p_dispatch"] = ["series","MW",0.,"Storage discharging.","Output"]
override_component_attrs["StorageUnit"].loc["p_store"] = ["series","MW",0.,"Storage charging.","Output"]




def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"


def assign_locations(n):
    for c in n.iterate_components(n.one_port_components|n.branch_components):

        ifind = pd.Series(c.df.index.str.find(" ",start=4),c.df.index)

        for i in ifind.unique():
            names = ifind.index[ifind == i]

            if i == -1:
                c.df.loc[names,'location'] = ""
            else:
                c.df.loc[names,'location'] = names.str[:i]

def calculate_nodal_cfs(n,label,nodal_cfs):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components((n.branch_components^{"Line","Transformer"})|n.controllable_one_port_components^{"Load","StorageUnit"}):
        capacities_c = c.df[opt_name.get(c.name,"p") + "_nom_opt"].groupby((c.df.location,c.df.carrier)).sum()

        if c.name == "Link":
            p = c.pnl.p0.abs().mean()
        elif c.name == "Generator":
            p = c.pnl.p.abs().mean()
        elif c.name == "Store":
            p = c.pnl.e.abs().mean()
        else:
            sys.exit()

        p_c = p.groupby((c.df.location,c.df.carrier)).sum()

        cf_c = p_c/capacities_c

        index = pd.MultiIndex.from_tuples([(c.list_name,) + t for t in cf_c.index.to_list()])
        nodal_cfs = nodal_cfs.reindex(index|nodal_cfs.index)
        nodal_cfs.loc[index,label] = cf_c.values

    return nodal_cfs





def calculate_cfs(n,label,cfs):

    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load","StorageUnit"}):
        capacities_c = c.df[opt_name.get(c.name,"p") + "_nom_opt"].groupby(c.df.carrier).sum()

        if c.name in ["Link","Line","Transformer"]:
            p = c.pnl.p0.abs().mean()
        elif c.name == "Store":
            p = c.pnl.e.abs().mean()
        else:
            p = c.pnl.p.abs().mean()

        p_c = p.groupby(c.df.carrier).sum()

        cf_c = p_c/capacities_c

        cf_c = pd.concat([cf_c], keys=[c.list_name])

        cfs = cfs.reindex(cf_c.index|cfs.index)

        cfs.loc[cf_c.index,label] = cf_c

    return cfs




def calculate_nodal_costs(n,label,nodal_costs):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        capital_costs = (c.df.capital_cost*c.df[opt_name.get(c.name,"p") + "_nom_opt"]).groupby((c.df.location,c.df.carrier)).sum()
        index = pd.MultiIndex.from_tuples([(c.list_name,"capital") + t for t in capital_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index|nodal_costs.index)
        nodal_costs.loc[index,label] = capital_costs.values

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings,axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings,axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings,axis=0).sum()

        #correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.)]
            c.df.loc[items,"marginal_cost"] = -20.

        marginal_costs = (p*c.df.marginal_cost).groupby((c.df.location,c.df.carrier)).sum()
        index = pd.MultiIndex.from_tuples([(c.list_name,"marginal") + t for t in marginal_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index|nodal_costs.index)
        nodal_costs.loc[index,label] = marginal_costs.values

    return nodal_costs


def calculate_costs(n,label,costs):

    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        capital_costs = c.df.capital_cost*c.df[opt_name.get(c.name,"p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(capital_costs_grouped.index|costs.index)

        costs.loc[capital_costs_grouped.index,label] = capital_costs_grouped

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings,axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings,axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings,axis=0).sum()

        #correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.)]
            c.df.loc[items,"marginal_cost"] = -20.

        marginal_costs = p*c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(marginal_costs_grouped.index|costs.index)

        costs.loc[marginal_costs_grouped.index,label] = marginal_costs_grouped

    #add back in costs of links if there is a line volume limit
    if label[1] != "opt":
        costs.loc[("links-added","capital","transmission lines"),label] = ((costs_db.at['HVDC overhead', 'capital_cost']*n.links.length + costs_db.at['HVDC inverter pair', 'capital_cost'])*n.links.p_nom_opt)[n.links.carrier == "DC"].sum()
        costs.loc[("lines-added","capital","transmission lines"),label] = costs_db.at["HVAC overhead", "capital_cost"]*(n.lines.length*n.lines.s_nom_opt).sum()
    else:
        costs.loc[("links-added","capital","transmission lines"),label] = (costs_db.at['HVDC inverter pair', 'capital_cost']*n.links.p_nom_opt)[n.links.carrier == "DC"].sum()


    #add back in all hydro
    #costs.loc[("storage_units","capital","hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro","p_nom"].sum()
    #costs.loc[("storage_units","capital","PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS","p_nom"].sum()
    #costs.loc[("generators","capital","ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror","p_nom"].sum()

    return costs


def calculate_nodal_capacities(n,label,nodal_capacities):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        nodal_capacities_c = c.df[opt_name.get(c.name,"p") + "_nom_opt"].groupby((c.df.location,c.df.carrier)).sum()
        index = pd.MultiIndex.from_tuples([(c.list_name,) + t for t in nodal_capacities_c.index.to_list()])
        nodal_capacities = nodal_capacities.reindex(index|nodal_capacities.index)
        nodal_capacities.loc[index,label] = nodal_capacities_c.values

    return nodal_capacities




def calculate_capacities(n,label,capacities):

    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        capacities_grouped = c.df[opt_name.get(c.name,"p") + "_nom_opt"].groupby(c.df.carrier).sum()
        capacities_grouped = pd.concat([capacities_grouped], keys=[c.list_name])

        capacities = capacities.reindex(capacities_grouped.index|capacities.index)

        capacities.loc[capacities_grouped.index,label] = capacities_grouped

    return capacities


def calculate_curtailment(n,label,curtailment):

    avail = n.generators_t.p_max_pu.multiply(n.generators.p_nom_opt).sum().groupby(n.generators.carrier).sum()
    used = n.generators_t.p.sum().groupby(n.generators.carrier).sum()

    curtailment[label] = (((avail - used)/avail)*100).round(3)

    return curtailment

def calculate_energy(n,label,energy):

    for c in n.iterate_components(n.one_port_components|n.branch_components):

        if c.name in n.one_port_components:
            c_energies = c.pnl.p.multiply(n.snapshot_weightings,axis=0).sum().multiply(c.df.sign).groupby(c.df.carrier).sum()
        else:
            c_energies = pd.Series(0.,c.df.carrier.unique())
            for port in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
                c_energies -= c.pnl["p"+port].multiply(n.snapshot_weightings,axis=0).sum().groupby(c.df.carrier).sum()

        c_energies = pd.concat([c_energies], keys=[c.list_name])

        energy = energy.reindex(c_energies.index|energy.index)

        energy.loc[c_energies.index,label] = c_energies

    return energy


def calculate_supply(n,label,supply):
    """calculate the max dispatch of each component at the buses aggregated by carrier"""

    bus_carriers = n.buses.carrier.unique()

    for i in bus_carriers:
        bus_map = (n.buses.carrier == i)
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map)]

            if len(items) == 0:
                continue

            s = c.pnl.p[items].max().multiply(c.df.loc[items,'sign']).groupby(c.df.loc[items,'carrier']).sum()
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply = supply.reindex(s.index|supply.index)
            supply.loc[s.index,label] = s


        for c in n.iterate_components(n.branch_components):

            for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

                items = c.df.index[c.df["bus" + end].map(bus_map,na_action=False)]

                if len(items) == 0:
                    continue

                #lots of sign compensation for direction and to do maximums
                s = (-1)**(1-int(end))*((-1)**int(end)*c.pnl["p"+end][items]).max().groupby(c.df.loc[items,'carrier']).sum()
                s.index = s.index+end
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply = supply.reindex(s.index|supply.index)
                supply.loc[s.index,label] = s

    return supply

def calculate_supply_energy(n,label,supply_energy):
    """calculate the total energy supply/consuption of each component at the buses aggregated by carrier"""


    bus_carriers = n.buses.carrier.unique()

    for i in bus_carriers:
        bus_map = (n.buses.carrier == i)
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map)]

            if len(items) == 0:
                continue

            s = c.pnl.p[items].multiply(n.snapshot_weightings,axis=0).sum().multiply(c.df.loc[items,'sign']).groupby(c.df.loc[items,'carrier']).sum()
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply_energy = supply_energy.reindex(s.index|supply_energy.index)
            supply_energy.loc[s.index,label] = s


        for c in n.iterate_components(n.branch_components):

            for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

                items = c.df.index[c.df["bus" + str(end)].map(bus_map,na_action=False)]

                if len(items) == 0:
                    continue

                s = (-1)*c.pnl["p"+end][items].multiply(n.snapshot_weightings,axis=0).sum().groupby(c.df.loc[items,'carrier']).sum()
                s.index = s.index+end
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply_energy = supply_energy.reindex(s.index|supply_energy.index)

                supply_energy.loc[s.index,label] = s


    return supply_energy

def calculate_metrics(n,label,metrics):

    metrics = metrics.reindex(pd.Index(["line_volume","line_volume_limit","line_volume_AC","line_volume_DC","line_volume_shadow","co2_shadow"])|metrics.index)

    metrics.at["line_volume_DC",label] = (n.links.length*n.links.p_nom_opt)[n.links.carrier == "DC"].sum()
    metrics.at["line_volume_AC",label] = (n.lines.length*n.lines.s_nom_opt).sum()
    metrics.at["line_volume",label] = metrics.loc[["line_volume_AC","line_volume_DC"],label].sum()

    if hasattr(n,"line_volume_limit"):
        metrics.at["line_volume_limit",label] = n.line_volume_limit
        metrics.at["line_volume_shadow",label] = n.line_volume_limit_dual

    if "CO2Limit" in n.global_constraints.index:
        metrics.at["co2_shadow",label] = n.global_constraints.at["CO2Limit","mu"]

    return metrics


def calculate_prices(n,label,prices):

    prices = prices.reindex(prices.index|n.buses.carrier.unique())

    #WARNING: this is time-averaged, see weighted_prices for load-weighted average
    prices[label] = n.buses_t.marginal_price.mean().groupby(n.buses.carrier).mean()

    return prices



def calculate_weighted_prices(n,label,weighted_prices):
    # Warning: doesn't include storage units as loads


    weighted_prices = weighted_prices.reindex(pd.Index(["electricity","heat","space heat","urban heat","space urban heat","gas","H2"]))

    link_loads = {"electricity" :  ["heat pump", "resistive heater", "battery charger", "H2 Electrolysis"],
                  "heat" : ["water tanks charger"],
                  "urban heat" : ["water tanks charger"],
                  "space heat" : [],
                  "space urban heat" : [],
                  "gas" : ["OCGT","gas boiler","CHP electric","CHP heat"],
                  "H2" : ["Sabatier", "H2 Fuel Cell"]}

    for carrier in link_loads:

        if carrier == "electricity":
            suffix = ""
        elif carrier[:5] == "space":
            suffix = carrier[5:]
        else:
            suffix =  " " + carrier

        buses = n.buses.index[n.buses.index.str[2:] == suffix]

        if buses.empty:
            continue

        if carrier in ["H2","gas"]:
            load = pd.DataFrame(index=n.snapshots,columns=buses,data=0.)
        elif carrier[:5] == "space":
            load = heat_demand_df[buses.str[:2]].rename(columns=lambda i: str(i)+suffix)
        else:
            load = n.loads_t.p_set[buses]


        for tech in link_loads[carrier]:

            names = n.links.index[n.links.index.to_series().str[-len(tech):] == tech]

            if names.empty:
                continue

            load += n.links_t.p0[names].groupby(n.links.loc[names,"bus0"],axis=1).sum()

        #Add H2 Store when charging
        #if carrier == "H2":
        #    stores = n.stores_t.p[buses+ " Store"].groupby(n.stores.loc[buses+ " Store","bus"],axis=1).sum(axis=1)
        #    stores[stores > 0.] = 0.
        #    load += -stores

        weighted_prices.loc[carrier,label] = (load*n.buses_t.marginal_price[buses]).sum().sum()/load.sum().sum()

        if carrier[:5] == "space":
            print(load*n.buses_t.marginal_price[buses])

    return weighted_prices




def calculate_market_values(n, label, market_values):
    # Warning: doesn't include storage units

    carrier = "AC"

    buses = n.buses.index[n.buses.carrier == carrier]

    ## First do market value of generators ##

    generators = n.generators.index[n.buses.loc[n.generators.bus,"carrier"] == carrier]

    techs = n.generators.loc[generators,"carrier"].value_counts().index

    market_values = market_values.reindex(market_values.index | techs)


    for tech in techs:
        gens = generators[n.generators.loc[generators,"carrier"] == tech]

        dispatch = n.generators_t.p[gens].groupby(n.generators.loc[gens,"bus"],axis=1).sum().reindex(columns=buses,fill_value=0.)

        revenue = dispatch*n.buses_t.marginal_price[buses]

        market_values.at[tech,label] = revenue.sum().sum()/dispatch.sum().sum()



    ## Now do market value of links ##

    for i in ["0","1"]:
        all_links = n.links.index[n.buses.loc[n.links["bus"+i],"carrier"] == carrier]

        techs = n.links.loc[all_links,"carrier"].value_counts().index

        market_values = market_values.reindex(market_values.index | techs)

        for tech in techs:
            links = all_links[n.links.loc[all_links,"carrier"] == tech]

            dispatch = n.links_t["p"+i][links].groupby(n.links.loc[links,"bus"+i],axis=1).sum().reindex(columns=buses,fill_value=0.)

            revenue = dispatch*n.buses_t.marginal_price[buses]

            market_values.at[tech,label] = revenue.sum().sum()/dispatch.sum().sum()

    return market_values


def calculate_price_statistics(n, label, price_statistics):


    price_statistics = price_statistics.reindex(price_statistics.index|pd.Index(["zero_hours","mean","standard_deviation"]))

    buses = n.buses.index[n.buses.carrier == "AC"]

    threshold = 0.1 #higher than phoney marginal_cost of wind/solar

    df = pd.DataFrame(data=0.,columns=buses,index=n.snapshots)

    df[n.buses_t.marginal_price[buses] < threshold] = 1.

    price_statistics.at["zero_hours", label] = df.sum().sum()/(df.shape[0]*df.shape[1])

    price_statistics.at["mean", label] = n.buses_t.marginal_price[buses].unstack().mean()

    price_statistics.at["standard_deviation", label] = n.buses_t.marginal_price[buses].unstack().std()

    return price_statistics


outputs = ["nodal_costs",
           "nodal_capacities",
           "nodal_cfs",
           "cfs",
           "costs",
           "capacities",
           "curtailment",
           "energy",
           "supply",
           "supply_energy",
           "prices",
           "weighted_prices",
           "price_statistics",
           "market_values",
           "metrics",
           ]

def make_summaries(networks_dict):

    columns = pd.MultiIndex.from_tuples(networks_dict.keys(),names=["cluster","lv","opt"])

    df = {}

    for output in outputs:
        df[output] = pd.DataFrame(columns=columns,dtype=float)

    for label, filename in iteritems(networks_dict):
        print(label, filename)

        n = pypsa.Network(filename,
                          override_component_attrs=override_component_attrs)


        assign_carriers(n)
        assign_locations(n)

        for output in outputs:
            df[output] = globals()["calculate_" + output](n, label, df[output])

    return df


def to_csv(df):

    for key in df:
        df[key].to_csv(snakemake.output[key])


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils import Dict
        import yaml
        snakemake = Dict()
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)

        #overwrite some options
        snakemake.config["run"] = "190418-test-rebase"
        snakemake.config["scenario"]["lv"] = [1.0, 1.25]
        snakemake.config["scenario"]["sector_opts"] = ["Co2L0-3H-T-H-B-I","Co2L0-3H-T-H-B-I-onwind0","Co2L0p1-3H-T-H-B-I","Co2L0-3H-T-H-B-I-onwind0-solar2-offwind2"]
        snakemake.input = Dict()
        snakemake.input['heat_demand_name'] = 'data/heating/daily_heat_demand.h5'
        snakemake.output = Dict()
        for item in outputs:
            snakemake.output[item] = snakemake.config['summary_dir'] + '/{name}/csvs/{item}.csv'.format(name=snakemake.config['run'],item=item)

    networks_dict = {(cluster,lv,opt+sector_opt) :
                     snakemake.config['results_dir'] + snakemake.config['run'] + '/postnetworks/elec_s_{cluster}_lv{lv}_{opt}_{sector_opt}.nc'\
                     .format(cluster=cluster,
                             opt=opt,
                             lv=lv,
                             sector_opt=sector_opt)\
                     for cluster in snakemake.config['scenario']['clusters'] \
                     for opt in snakemake.config['scenario']['opts'] \
                     for sector_opt in snakemake.config['scenario']['sector_opts'] \
                     for lv in snakemake.config['scenario']['lv']}
    print(networks_dict)

    costs_db = load_costs(Nyears=1.,tech_costs="data/costs.csv",config=snakemake.config["costs"],elec_config=snakemake.config['electricity'])

    df = make_summaries(networks_dict)

    df["metrics"].loc["total costs"] =  df["costs"].sum()

    to_csv(df)
