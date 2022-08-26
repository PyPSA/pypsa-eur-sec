from six import iteritems

import sys

import pandas as pd

import numpy as np

import pypsa
from vresutils.costdata import annuity

from helper import override_component_attrs

from pypsa.descriptors import (
    nominal_attrs,
    get_active_assets,
)

from packaging.version import Version, parse
agg_group_kwargs = (
    dict(numeric_only=False) if parse(pd.__version__) >= Version("1.3") else {}
)

idx = pd.IndexSlice

opt_name = {"Store": "e", "Line": "s", "Transformer": "s"}



def prepare_costs(cost_file, USD_to_EUR, discount_rate, Nyears, lifetime):

    # set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=list(range(2))).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = (
        costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    )
    costs = costs.fillna(
        {
            "CO2 intensity": 0,
            "FOM": 0,
            "VOM": 0,
            "discount rate": discount_rate,
            "efficiency": 1,
            "fuel": 0,
            "investment": 0,
            "lifetime": lifetime,
        }
    )

    costs["fixed"] = [
        (annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100.0)
        * v["investment"]
        * Nyears
        for i, v in costs.iterrows()
    ]
    return costs


def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"
    n.loads["carrier"] = n.loads.bus.apply(lambda x: n.buses.loc[x, "carrier"])


def assign_locations(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):

        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)

        for i in ifind.unique():
            names = ifind.index[ifind == i]

            if i == -1:
                c.df.loc[names, "location"] = ""
            else:
                c.df.loc[names, "location"] = names.str[:i]


def calculate_nodal_cfs(n, label, nodal_cfs):
    # Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(
        (n.branch_components ^ {"Line", "Transformer"})
        | n.controllable_one_port_components ^ {"Load", "StorageUnit"}
    ):
        capacities_c = c.df.groupby(["location", "carrier"])[
            opt_name.get(c.name, "p") + "_nom_opt"
        ].sum(**agg_group_kwargs)

        if c.name == "Link":
            p = c.pnl.p0.abs().mean()
        elif c.name == "Generator":
            p = c.pnl.p.abs().mean()
        elif c.name == "Store":
            p = c.pnl.e.abs().mean()
        else:
            sys.exit()

        c.df["p"] = p
        p_c = c.df.groupby(["location", "carrier"])["p"].sum(**agg_group_kwargs)

        cf_c = p_c / capacities_c

        index = pd.MultiIndex.from_tuples(
            [(c.list_name,) + t for t in cf_c.index.to_list()]
        )
        nodal_cfs = nodal_cfs.reindex(index.union(nodal_cfs.index))
        nodal_cfs.loc[index, label] = cf_c.values

    return nodal_cfs


def calculate_cfs(n, label, cfs):

    for c in n.iterate_components(
        n.branch_components
        | n.controllable_one_port_components ^ {"Load", "StorageUnit"}
    ):
        capacities_c = (
            c.df[opt_name.get(c.name, "p") + "_nom_opt"]
            .groupby(c.df.carrier)
            .sum(**agg_group_kwargs)
        )

        if c.name in ["Link", "Line", "Transformer"]:
            p = c.pnl.p0.abs().mean()
        elif c.name == "Store":
            p = c.pnl.e.abs().mean()
        else:
            p = c.pnl.p.abs().mean()

        p_c = p.groupby(c.df.carrier).sum(**agg_group_kwargs)

        cf_c = p_c / capacities_c

        cf_c = pd.concat([cf_c], keys=[c.list_name])

        cfs = cfs.reindex(cf_c.index.union(cfs.index))

        cfs.loc[cf_c.index, label] = cf_c

    return cfs


def calculate_nodal_costs(n, label, nodal_costs):
    # Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        c.df["capital_costs"] = (
            c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        )
        capital_costs = c.df.groupby(["location", "carrier"])["capital_costs"].sum(
            **agg_group_kwargs
        )
        index = pd.MultiIndex.from_tuples(
            [(c.list_name, "capital") + t for t in capital_costs.index.to_list()]
        )
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))
        nodal_costs.loc[index, label] = capital_costs.values

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.objective, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.objective, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.objective, axis=0).sum()

        # correct sequestration cost
        if c.name == "Store":
            items = c.df.index[
                (c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)
            ]
            c.df.loc[items, "marginal_cost"] = -20.0

        c.df["marginal_costs"] = p * c.df.marginal_cost
        marginal_costs = c.df.groupby(["location", "carrier"])["marginal_costs"].sum(
            **agg_group_kwargs
        )
        index = pd.MultiIndex.from_tuples(
            [(c.list_name, "marginal") + t for t in marginal_costs.index.to_list()]
        )
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))
        nodal_costs.loc[index, label] = marginal_costs.values

    return nodal_costs


def calculate_costs(n, label, costs):

    investments = n.snapshots.levels[0]
    helper = [costs.columns.levels[i] for i in range(len(label))]
    helper.insert(len(label), investments)
    cols = pd.MultiIndex.from_product(
        helper,
        names=costs.columns.names[:len(label)] + ["year"],
    )
    costs = costs.reindex(cols, axis=1)

    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        capital_costs = c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        active = pd.concat(
            [
                get_active_assets(n, c.name, inv_p).rename(inv_p)
                for inv_p in investments
            ],
            axis=1,
        ).astype(int)
        capital_costs = active.mul(capital_costs, axis=0)
        discount = (
            n.investment_period_weightings["objective"]
            / n.investment_period_weightings["years"]
        )
        capital_costs_grouped = (
            capital_costs.groupby(c.df.carrier).sum(**agg_group_kwargs).mul(discount)
        )

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(capital_costs_grouped.index.union(costs.index))

        costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped.values

        if c.name == "Link":
            p = (
                c.pnl.p0.multiply(n.snapshot_weightings.objective, axis=0)
                .groupby(level=0)
                .sum()
            )
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.objective, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.groupby(level=0).sum()
        else:
            p = (
                round(c.pnl.p, ndigits=2)
                .multiply(n.snapshot_weightings.objective, axis=0)
                .groupby(level=0)
                .sum()
            )

        # correct sequestration cost
        if c.name == "Store":
            items = c.df.index[
                (c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)
            ]
            c.df.loc[items, "marginal_cost"] = -20.0

        marginal_costs = p.mul(c.df.marginal_cost).T
        # marginal_costs = active.mul(marginal_costs, axis=0)
        marginal_costs_grouped = (
            marginal_costs.groupby(c.df.carrier).sum(**agg_group_kwargs).mul(discount)
        )

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))

        costs.loc[marginal_costs_grouped.index, label] = marginal_costs_grouped.values

    # add back in all hydro
    # costs.loc[("storage_units","capital","hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro","p_nom"].sum()
    # costs.loc[("storage_units","capital","PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS","p_nom"].sum()
    # costs.loc[("generators","capital","ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror","p_nom"].sum()

    return costs


def calculate_capital_cost(n, label, capital_cost):
    """Calculate costs without the non-learning costs."""
    investments = n.snapshots.levels[0]
    helper = [capital_cost.columns.levels[i] for i in range(len(label))]
    helper.insert(len(label), investments)
    cols = pd.MultiIndex.from_product(
        helper,
        names=capital_cost.columns.names[:len(label)] + ["year"],
    )

    capital_cost = capital_cost.reindex(cols, axis=1)

    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        cap = c.df.capital_cost
        if "nolearning_cost" in c.df.columns:
            cap -= c.df.nolearning_cost.fillna(0)
        cap = cap.groupby([c.df.carrier, c.df.build_year]).mean().unstack().reindex(columns=investments)
        capital_cost = capital_cost.reindex(capital_cost.index.union(cap.index))
        capital_cost.loc[cap.index, label] = cap.values

    return capital_cost


def calculate_nodal_capacities(n, label, nodal_capacities):
    # Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        nodal_capacities_c = c.df.groupby(["location", "carrier"])[
            opt_name.get(c.name, "p") + "_nom_opt"
        ].sum(**agg_group_kwargs)
        index = pd.MultiIndex.from_tuples(
            [(c.list_name,) + t for t in nodal_capacities_c.index.to_list()]
        )
        nodal_capacities = nodal_capacities.reindex(index.union(nodal_capacities.index))
        nodal_capacities.loc[index, label] = nodal_capacities_c.values

    return nodal_capacities


def calculate_capacities(n, label, capacities):

    investments = n.snapshots.levels[0]
    helper = [capacities.columns.levels[i] for i in range(len(label))]
    helper.insert(len(label), investments)
    cols = pd.MultiIndex.from_product(
        helper,
        names=capacities.columns.names[:len(label)] + ["year"],
    )

    capacities = capacities.reindex(cols, axis=1)

    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        active = pd.concat(
            [
                get_active_assets(n, c.name, inv_p).rename(inv_p)
                for inv_p in investments
            ],
            axis=1,
        ).astype(int)
        caps = c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        caps = active.mul(caps, axis=0)
        capacities_grouped = (
            caps.groupby(c.df.carrier)
            .sum(**agg_group_kwargs)
            .drop("load", errors="ignore")
        )
        capacities_grouped = pd.concat([capacities_grouped], keys=[c.list_name])

        capacities = capacities.reindex(
            capacities_grouped.index.union(capacities.index)
        )

        capacities.loc[capacities_grouped.index, label] = capacities_grouped.values

    return capacities


def calculate_curtailment(n, label, curtailment):

    avail = (
        n.generators_t.p_max_pu.multiply(n.generators.p_nom_opt)
        .sum()
        .groupby(n.generators.carrier)
        .sum(**agg_group_kwargs)
    )
    used = n.generators_t.p.sum().groupby(n.generators.carrier).sum(**agg_group_kwargs)

    curtailment[label] = (((avail - used) / avail) * 100).round(3)

    return curtailment


def calculate_energy(n, label, energy):

    for c in n.iterate_components(n.one_port_components | n.branch_components):

        if c.name in n.one_port_components:
            c_energies = (
                c.pnl.p.multiply(n.snapshot_weightings, axis=0)
                .sum()
                .multiply(c.df.sign)
                .groupby(c.df.carrier)
                .sum(**agg_group_kwargs)
            )
        else:
            c_energies = pd.Series(0.0, c.df.carrier.unique())
            for port in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
                totals = c.pnl["p" + port].multiply(n.snapshot_weightings.generators, axis=0).sum()
                # remove values where bus is missing (bug in nomopyomo)
                no_bus = c.df.index[c.df["bus" + port] == ""]
                totals.loc[no_bus] = n.component_attrs[c.name].loc[
                    "p" + port, "default"
                ]
                c_energies -= totals.groupby(c.df.carrier).sum(**agg_group_kwargs)

        c_energies = pd.concat([c_energies], keys=[c.list_name])

        energy = energy.reindex(c_energies.index.union(energy.index))

        energy.loc[c_energies.index, label] = c_energies

    return energy


def calculate_supply(n, label, supply):
    """calculate the max dispatch of each component at the buses aggregated by carrier"""

    bus_carriers = n.buses.carrier.unique()

    for i in bus_carriers:
        bus_map = n.buses.carrier == i
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

            if len(items) == 0:
                continue

            s = (
                c.pnl.p[items]
                .max()
                .multiply(c.df.loc[items, "sign"])
                .groupby(c.df.loc[items, "carrier"])
                .sum(**agg_group_kwargs)
            )
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply = supply.reindex(s.index.union(supply.index))
            supply.loc[s.index, label] = s

        for c in n.iterate_components(n.branch_components):

            for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

                items = c.df.index[c.df["bus" + end].map(bus_map, na_action=False)]

                if len(items) == 0:
                    continue

                # lots of sign compensation for direction and to do maximums
                s = (-1) ** (1 - int(end)) * (
                    (-1) ** int(end) * c.pnl["p" + end][items]
                ).max().groupby(c.df.loc[items, "carrier"]).sum(**agg_group_kwargs)
                s.index = s.index + end
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply = supply.reindex(s.index.union(supply.index))
                supply.loc[s.index, label] = s

    return supply


def calculate_supply_energy(n, label, supply_energy):
    """calculate the total energy supply/consuption of each component at the buses aggregated by carrier"""

    investments = n.snapshots.levels[0]
    helper = [supply_energy.columns.levels[i] for i in range(len(label))]
    helper.insert(len(label), investments)
    cols = pd.MultiIndex.from_product(
        helper,
        names=supply_energy.columns.names[:len(label)] + ["year"],
    )

    supply_energy = supply_energy.reindex(cols, axis=1)

    bus_carriers = n.buses.carrier.unique()

    for i in bus_carriers:
        bus_map = n.buses.carrier == i
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

            if len(items) == 0:
                continue

            if c.name == "Generator":
                weightings = n.snapshot_weightings.generators
            else:
                weightings = n.snapshot_weightings.stores

            if i in ["oil", "co2", "H2"]:
                if c.name=="Load":
                    c.df.loc[items, "carrier"] = [load.split("-202")[0] for load in items]
                if i=="oil" and c.name=="Generator":
                    c.df.loc[items, "carrier"] = "imported oil"
            s = (
                c.pnl.p[items]
                .multiply(weightings, axis=0)
                .groupby(level=0)
                .sum()
                .multiply(c.df.loc[items, "sign"])
                .groupby(c.df.loc[items, "carrier"], axis=1)
                .sum(**agg_group_kwargs)
                .T
            )
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply_energy = supply_energy.reindex(
                s.index.union(supply_energy.index, sort=False)
            )
            supply_energy.loc[s.index, label] = s.values

        for c in n.iterate_components(n.branch_components):

            for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

                items = c.df.index[c.df["bus" + str(end)].map(bus_map, na_action=False)]

                if len(items) == 0:
                    continue

                s = (
                    (-1)
                    * c.pnl["p" + end]
                    .reindex(items, axis=1)
                    .multiply(n.snapshot_weightings.objective, axis=0)
                    .groupby(level=0)
                    .sum()
                    .groupby(c.df.loc[items, "carrier"], axis=1)
                    .sum(**agg_group_kwargs)
                ).T
                s.index = s.index + end
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply_energy = supply_energy.reindex(
                    s.index.union(supply_energy.index, sort=False)
                )

                supply_energy.loc[s.index, label] = s.values

    return supply_energy


def calculate_metrics(n, label, metrics):

    metrics = metrics.reindex(
        pd.Index(
            [
                "line_volume",
                "line_volume_limit",
                "line_volume_AC",
                "line_volume_DC",
                "line_volume_shadow",
                "co2_shadow",
            ]
        ).union(metrics.index)
    )

    metrics.at["line_volume_DC", label] = (n.links.length * n.links.p_nom_opt)[
        n.links.carrier == "DC"
    ].sum()
    metrics.at["line_volume_AC", label] = (n.lines.length * n.lines.s_nom_opt).sum()
    metrics.at["line_volume", label] = metrics.loc[
        ["line_volume_AC", "line_volume_DC"], label
    ].sum()

    if hasattr(n, "line_volume_limit"):
        metrics.at["line_volume_limit", label] = n.line_volume_limit
        metrics.at["line_volume_shadow", label] = n.line_volume_limit_dual

    if "CO2Limit" in n.global_constraints.index:
        metrics.at["co2_shadow", label] = n.global_constraints.at["CO2Limit", "mu"]

    return metrics


def calculate_prices(n, label, prices):

    prices = prices.reindex(prices.index.union(n.buses.carrier.unique()))

    # WARNING: this is time-averaged, see weighted_prices for load-weighted average
    prices[label] = (
        n.buses_t.marginal_price.mean()
        .groupby(n.buses.carrier)
        .mean(**agg_group_kwargs)
    )

    return prices


def calculate_weighted_prices(n, label, weighted_prices):
    # Warning: doesn't include storage units as loads

    weighted_prices = weighted_prices.reindex(
        pd.Index(
            [
                "electricity",
                "heat",
                "space heat",
                "urban heat",
                "space urban heat",
                "gas",
                "H2",
            ]
        )
    )

    link_loads = {
        "electricity": [
            "heat pump",
            "resistive heater",
            "battery charger",
            "H2 Electrolysis",
        ],
        "heat": ["water tanks charger"],
        "urban heat": ["water tanks charger"],
        "space heat": [],
        "space urban heat": [],
        "gas": ["OCGT", "gas boiler", "CHP electric", "CHP heat"],
        "H2": ["Sabatier", "H2 Fuel Cell"],
    }

    for carrier in link_loads:

        if carrier == "electricity":
            suffix = ""
        elif carrier[:5] == "space":
            suffix = carrier[5:]
        else:
            suffix = " " + carrier

        buses = n.buses.index[n.buses.index.str[2:] == suffix]

        if buses.empty:
            continue

        if carrier in ["H2", "gas"]:
            load = pd.DataFrame(index=n.snapshots, columns=buses, data=0.0)

        else:
            load = n.loads_t.p_set.reindex(buses, axis=1)

        for tech in link_loads[carrier]:

            names = n.links.index[n.links.index.to_series().str[-len(tech) :] == tech]

            if names.empty:
                continue

            load += (
                n.links_t.p0[names].groupby(n.links.loc[names, "bus0"], axis=1).sum()
            )

        # Add H2 Store when charging
        # if carrier == "H2":
        #    stores = n.stores_t.p[buses+ " Store"].groupby(n.stores.loc[buses+ " Store","bus"],axis=1).sum(axis=1)
        #    stores[stores > 0.] = 0.
        #    load += -stores

        weighted_prices.loc[carrier, label] = (
            load * n.buses_t.marginal_price[buses]
        ).sum().sum() / load.sum().sum()

        if carrier[:5] == "space":
            print(load * n.buses_t.marginal_price[buses])

    return weighted_prices


def calculate_market_values(n, label, market_values):
    # Warning: doesn't include storage units

    carrier = "AC"

    buses = n.buses.index[n.buses.carrier == carrier]

    ## First do market value of generators ##

    generators = n.generators.index[n.buses.loc[n.generators.bus, "carrier"] == carrier]

    techs = n.generators.loc[generators, "carrier"].value_counts().index

    market_values = market_values.reindex(market_values.index.union(techs))

    for tech in techs:
        gens = generators[n.generators.loc[generators, "carrier"] == tech]

        dispatch = (
            n.generators_t.p[gens]
            .groupby(n.generators.loc[gens, "bus"], axis=1)
            .sum()
            .reindex(columns=buses, fill_value=0.0)
        )

        revenue = dispatch * n.buses_t.marginal_price[buses]

        market_values.at[tech, label] = revenue.sum().sum() / dispatch.sum().sum()

    ## Now do market value of links ##

    for i in ["0", "1"]:
        all_links = n.links.index[n.buses.loc[n.links["bus" + i], "carrier"] == carrier]

        techs = n.links.loc[all_links, "carrier"].value_counts().index

        market_values = market_values.reindex(market_values.index.union(techs))

        for tech in techs:
            links = all_links[n.links.loc[all_links, "carrier"] == tech]

            dispatch = (
                n.links_t["p" + i][links]
                .groupby(n.links.loc[links, "bus" + i], axis=1)
                .sum()
                .reindex(columns=buses, fill_value=0.0)
            )

            revenue = dispatch * n.buses_t.marginal_price[buses]

            market_values.at[tech, label] = revenue.sum().sum() / dispatch.sum().sum()

    return market_values


def calculate_price_statistics(n, label, price_statistics):

    price_statistics = price_statistics.reindex(
        price_statistics.index.union(
            pd.Index(["zero_hours", "mean", "standard_deviation"])
        )
    )

    buses = n.buses.index[n.buses.carrier == "AC"]

    threshold = 0.1  # higher than phoney marginal_cost of wind/solar

    df = pd.DataFrame(data=0.0, columns=buses, index=n.snapshots)

    df[n.buses_t.marginal_price[buses] < threshold] = 1.0

    price_statistics.at["zero_hours", label] = df.sum().sum() / (
        df.shape[0] * df.shape[1]
    )

    price_statistics.at["mean", label] = (
        n.buses_t.marginal_price[buses].mean().mean()
    )

    price_statistics.at["standard_deviation", label] = (
        n.buses_t.marginal_price[buses].stack().std()
    )

    return price_statistics


def calculate_co2_emissions(n, label, df):

    investments = n.snapshots.levels[0]

    carattr = "co2_emissions"
    emissions = n.carriers.query(f"{carattr} != 0")[carattr]

    if emissions.empty:
        return

    weightings = n.snapshot_weightings.mul(
        n.investment_period_weightings["years"]
        .reindex(n.snapshots)
        .fillna(method="bfill")
        .fillna(1.0),
        axis=0,
    )

    # generators
    gens = n.generators.query("carrier in @emissions.index")
    if not gens.empty:
        em_pu = gens.carrier.map(emissions) / gens.efficiency
        em_pu = (
            weightings["generators"].to_frame("weightings")
            @ em_pu.to_frame("weightings").T
        )
        emitted = n.generators_t.p[gens.index].mul(em_pu)

        emitted_grouped = (
            emitted.groupby(level=0)
            .sum()
            .groupby(n.generators.carrier, axis=1)
            .sum(**agg_group_kwargs)
            .T
        )

        df = df.reindex(emitted_grouped.index.union(df.index))

        df.loc[emitted_grouped.index, label] = emitted_grouped.values

    if any(n.stores.carrier == "co2"):
        co2_i = n.stores[n.stores.carrier == "co2"].index
        df[label] = n.stores_t.e.groupby(level=0).last()[co2_i].iloc[:, 0]

    return df


outputs = [
    "nodal_costs",
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
    "weighted_prices",  # TODO
    "price_statistics",
    "market_values",
    "metrics",
    "co2_emissions",
    "capital_cost",
]


def make_summaries(networks_dict):
    if snakemake.rule=="make_summary_perfect":
        names = ["cluster", "lv", "opt"]
    else:

        names =  ["clusters", "lv", "sector_opt", "epsilon", "sense", "mga_tech"]
    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(), names=names
    )
    df = {}

    for output in outputs:
        df[output] = pd.DataFrame(columns=columns, dtype=float)

    overrides = override_component_attrs(snakemake.input.overrides)
    for label, filename in iteritems(networks_dict):
        print(label, filename)
        try:
            n = pypsa.Network(
                filename, override_component_attrs=overrides
            )
        except OSError:
            print(label, " not solved yet.")
            continue

        if not hasattr(n, "objective"):
            print(label, " not solved correctly. Check log if infeasible or unbounded.")
            continue
        assign_carriers(n)
        assign_locations(n)

        for output in outputs:
            df[output] = globals()["calculate_" + output](n, label, df[output])

    return df


def to_csv(df):

    for key in df:
        df[key] = df[key].apply(lambda x: pd.to_numeric(x))
        df[key].to_csv(snakemake.output[key])


#%%
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'make_summary_mga',
        )

    if snakemake.rule=="make_summary_perfect":
        networks_dict = {
            (clusters, lv, sector_opt): "results/"
            + snakemake.config["run"]
            + "/postnetworks/elec_s{simpl}_{clusters}_lv{lv}_{opts}_{sector_opts}_brownfield_all_years.nc".format(
            simpl=simpl,
            clusters=clusters,
            lv=lv,
            opts=opts,
            sector_opts=sector_opt
            )
            for simpl in snakemake.config["scenario"]["simpl"]
            for clusters in snakemake.config["scenario"]["clusters"]
            for opts in snakemake.config["scenario"]["opts"]
            for sector_opt in snakemake.config["scenario"]["sector_opts"]
            for lv in snakemake.config["scenario"]["lv"]
        }
    else:
        networks_dict = {
            (clusters, lv, sector_opt, epsilon, sense, mga_tech): "results/"
            + snakemake.config["run"]
            + "/alternatives/elec_s{simpl}_{clusters}_lv{lv}_{opts}_{sector_opts}_brownfield_all_years_tol{epsilon}_obj+{mga_tech}+{sense}.nc".format(
            simpl=simpl,
            clusters=clusters,
            lv=lv,
            opts=opts,
            sector_opts=sector_opt,
            epsilon=epsilon,
            sense=sense,
            mga_tech=mga_tech,
            )
            for simpl in snakemake.config["scenario"]["simpl"]
            for clusters in snakemake.config["scenario"]["clusters"]
            for opts in snakemake.config["scenario"]["opts"]
            for sector_opt in snakemake.config["scenario"]["sector_opts"]
            for lv in snakemake.config["scenario"]["lv"]
            for epsilon in snakemake.config["scenario"]["epsilon"]
            for mga_tech in snakemake.config["scenario"]["mga_tech"]
            for sense in snakemake.config["scenario"]["sense"]
        }

    print(networks_dict)

    Nyears = 1

    costs_db = prepare_costs(
        snakemake.input.costs,
        snakemake.config["costs"]["USD2013_to_EUR2013"],
        snakemake.config["costs"]["discountrate"],
        Nyears,
        snakemake.config["costs"]["lifetime"],
    )

    df = make_summaries(networks_dict)

    df["metrics"].loc["total costs"] = df["costs"].sum().groupby(level=[0, 1, 2]).sum()

    to_csv(df)
