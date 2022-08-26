"""Solve network."""

import pypsa

import numpy as np
import pandas as pd
import math

from pypsa.linopt import get_var, linexpr, define_constraints, write_objective

from pypsa.linopf import network_lopf, ilopf, lookup

from pypsa.descriptors import (get_active_assets, expand_series, nominal_attrs,
                               get_extendable_i, get_non_extendable_i)

from pypsa.pf import get_switchable_as_dense as get_as_dense

from vresutils.benchmark import memory_logger

from helper import override_component_attrs, update_config_with_sector_opts

from packaging.version import Version, parse
agg_group_kwargs = (
    dict(numeric_only=False) if parse(pd.__version__) >= Version("1.3") else {}
)

import logging
logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def add_land_use_constraint(n):

    if (snakemake.config["foresight"] == "perfect") and ('m' in snakemake.wildcards.clusters):
        raise NotImplementedError(
            "The clustermethod  m is not implemented for perfect foresight"
        )

    if 'm' in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n)
    else:
        _add_land_use_constraint(n)

def add_land_use_constraint_perfect(n):
    c, attr = "Generator", "p_nom"
    investments = n.snapshots.levels[0]
    df = n.df(c).copy()
    res = df[df.p_nom_max!=np.inf].carrier.unique()
    # extendable assets
    ext_i = n.df(c)[(n.df(c)["carrier"].isin(res)) & (n.df(c)["p_nom_extendable"])].index
    # dataframe when assets are active
    active_i =  pd.concat([get_active_assets(n, c, inv_p).rename(inv_p)
                           for inv_p in investments],axis=1).astype(int)
    # extendable and active assets
    ext_and_active = active_i.T[active_i.index.intersection(ext_i)]
    if ext_and_active.empty:
        return
    p_nom = get_var(n, c, attr)[ext_and_active.columns]
    # sum over each bus and carrier should be smaller than p_nom_max
    lhs = (
        linexpr((ext_and_active, p_nom))
        .T.groupby([n.df(c).carrier, n.df(c).bus])
        .sum(**agg_group_kwargs)
        .T
    )
    # maximum technical potential
    p_nom_max_w = n.df(c).p_nom_max.loc[ext_and_active.columns]
    p_nom_max_t = expand_series(p_nom_max_w, investments).T

    rhs = (
        p_nom_max_t.mul(ext_and_active)
        .groupby([n.df(c).carrier, n.df(c).bus], axis=1)
        .max(**agg_group_kwargs)
    ).reindex(columns=lhs.columns)

    # TODO this currently leads to infeasibilities in ES and DE
    # rename existing offwind to reduce technical potential
   # df.carrier.replace({"offwind":"offwind-ac"},inplace=True)
    #existing_p_nom = df.groupby([df.carrier, df.bus]).sum().p_nom.loc[rhs.columns]
    #rhs -= existing_p_nom
    # make sure that rhs is not negative because existing capacities > tech potential
    #rhs.clip(lower=0, inplace=True)
    define_constraints(n, lhs, "<=", rhs, "GlobalConstraint", "land_use_constraint")

def _add_land_use_constraint(n):
    #warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    res = n.generators[n.generators.p_nom_max!=np.inf].carrier.unique()

    for carrier in res:
        existing = n.generators.loc[n.generators.carrier==carrier,"p_nom"].groupby(n.generators.bus.map(n.buses.location)).sum()
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index,"p_nom_max"] -= existing

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    grouping_years = snakemake.config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    res = n.generators[n.generators.p_nom_max!=np.inf].carrier.unique()
    for carrier in res:

        existing = n.generators.loc[n.generators.carrier==carrier,"p_nom"]
        ind = list(set([i.split(sep=" ")[0] + ' ' + i.split(sep=" ")[1] for i in existing.index]))

        previous_years = [
            str(y) for y in
            planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [i for i in ind if i + " " + carrier + "-" + p_year in existing.index]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[sel_p_year].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def prepare_network(n, solve_opts=None):

    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.generators_t.p_min_pu, n.storage_units_t.inflow):
            df.where(df>solve_opts['clip_p_max_pu'], other=0., inplace=True)

    if solve_opts.get('load_shedding'):
        n.add("Carrier", "Load")
        n.madd("Generator", n.buses.index, " load",
               bus=n.buses.index,
               carrier='load',
               sign=1e-3, # Adjust sign to measure p and p_nom in kW instead of MW
               marginal_cost=1e2, # Eur/kWh
               # intersect between macroeconomic and surveybased
               # willingness to pay
               # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
               p_nom=1e9 # kW
        )

    if solve_opts.get('noisy_costs'):
        for t in n.iterate_components():
            #if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if 'marginal_cost' in t.df:
                np.random.seed(174)
                t.df['marginal_cost'] += 1e-2 + 2e-3 * (np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(['Line', 'Link']):
            np.random.seed(123)
            t.df['capital_cost'] += (1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)) * t.df['length']

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760./nhours

    if snakemake.config['foresight']=="myopic":
        add_land_use_constraint(n)

    return n


def add_battery_constraints(n):

    chargers_b = n.links.carrier.str.contains("battery charger")
    chargers = n.links.index[chargers_b & n.links.p_nom_extendable]
    dischargers = chargers.str.replace("charger", "discharger")

    if chargers.empty or ('Link', 'p_nom') not in n.variables.index:
        return

    link_p_nom = get_var(n, "Link", "p_nom")

    lhs = linexpr((1,link_p_nom[chargers]),
                  (-n.links.loc[dischargers, "efficiency"].values,
                   link_p_nom[dischargers].values))

    define_constraints(n, lhs, "=", 0, 'Link', 'charger_ratio')


def add_chp_constraints(n):

    electric_bool = (n.links.index.str.contains("urban central")
                     & n.links.index.str.contains("CHP")
                     & n.links.index.str.contains("electric"))
    heat_bool = (n.links.index.str.contains("urban central")
                 & n.links.index.str.contains("CHP")
                 & n.links.index.str.contains("heat"))

    electric = n.links.index[electric_bool]
    heat = n.links.index[heat_bool]

    electric_ext = n.links.index[electric_bool & n.links.p_nom_extendable]
    heat_ext = n.links.index[heat_bool & n.links.p_nom_extendable]

    electric_fix = n.links.index[electric_bool & ~n.links.p_nom_extendable]
    heat_fix = n.links.index[heat_bool & ~n.links.p_nom_extendable]

    link_p = get_var(n, "Link", "p")

    if not electric_ext.empty:

        link_p_nom = get_var(n, "Link", "p_nom")

        #ratio of output heat to electricity set by p_nom_ratio
        lhs = linexpr((n.links.loc[electric_ext, "efficiency"]
                       *n.links.loc[electric_ext, "p_nom_ratio"],
                       link_p_nom[electric_ext]),
                      (-n.links.loc[heat_ext, "efficiency"].values,
                       link_p_nom[heat_ext].values))

        define_constraints(n, lhs, "=", 0, 'chplink', 'fix_p_nom_ratio')

        #top_iso_fuel_line for extendable
        lhs = linexpr((1,link_p[heat_ext]),
                      (1,link_p[electric_ext].values),
                      (-1,link_p_nom[electric_ext].values))

        define_constraints(n, lhs, "<=", 0, 'chplink', 'top_iso_fuel_line_ext')

    if not electric_fix.empty:

        #top_iso_fuel_line for fixed
        lhs = linexpr((1,link_p[heat_fix]),
                      (1,link_p[electric_fix].values))

        rhs = n.links.loc[electric_fix, "p_nom"].values

        define_constraints(n, lhs, "<=", rhs, 'chplink', 'top_iso_fuel_line_fix')

    if not electric.empty:

        #backpressure
        lhs = linexpr((n.links.loc[electric, "c_b"].values
                       *n.links.loc[heat, "efficiency"],
                       link_p[heat]),
                      (-n.links.loc[electric, "efficiency"].values,
                       link_p[electric].values))

        define_constraints(n, lhs, "<=", 0, 'chplink', 'backpressure')

def basename(x):
     return x.split("-2")[0]

def add_pipe_retrofit_constraint(n):
    """Add constraint for retrofitting existing CH4 pipelines to H2 pipelines."""

    gas_pipes_i = n.links.query("carrier == 'gas pipeline' and p_nom_extendable").index
    h2_retrofitted_i = n.links.query("carrier == 'H2 pipeline retrofitted' and p_nom_extendable").index

    if h2_retrofitted_i.empty or gas_pipes_i.empty: return

    link_p_nom = get_var(n, "Link", "p_nom")

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    fr = "H2 pipeline retrofitted"
    to = "gas pipeline"

    pipe_capacity = n.links.loc[gas_pipes_i, 'p_nom'].rename(basename)

    lhs = linexpr(
        (CH4_per_H2, link_p_nom.loc[h2_retrofitted_i].rename(index=lambda x: x.replace(fr, to))),
        (1, link_p_nom.loc[gas_pipes_i])
    )

    lhs.rename(basename, inplace=True)
    define_constraints(n, lhs, "=", pipe_capacity, 'Link', 'pipe_retrofit')


def add_co2_sequestration_limit(n, sns):
    # TODO isn't it better to have the limit on the e_nom instead of last sn e?
    co2_stores = n.stores.loc[n.stores.carrier=='co2 stored'].index

    if co2_stores.empty or ('Store', 'e') not in n.variables.index:
        return
    if snakemake.config["foresight"]:
        last_sn = (n.snapshot_weightings.loc[sns].reset_index(level=1, drop=False)
                   .groupby(level=0).last().reset_index()
                   .set_index(["period", "timestep"]).index)
    else:
        last_sn = sns[-1]
    vars_final_co2_stored = get_var(n, 'Store', 'e').loc[last_sn, co2_stores]

    lhs = linexpr((1, vars_final_co2_stored)).sum(axis=1)

    limit = n.config["sector"].get("co2_sequestration_potential", 200) * 1e6
    for o in opts:
        if not "seq" in o: continue
        limit = float(o[o.find("seq")+3:])
        break

    name = 'co2_sequestration_limit'
    sense = "<="

    n.add("GlobalConstraint", name, sense=sense, constant=limit,
          type=np.nan, carrier_attribute=np.nan)

    define_constraints(n, lhs, sense, limit, 'GlobalConstraint',
                       'mu', axes=pd.Index([name]), spec=name)

def add_carbon_neutral_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Neutral"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        time_valid = int(glc.loc["investment_period"])
        if not stores.empty:
            final_e = get_var(n, "Store", "e").groupby(level=0).last()[stores.index]
            lhs = linexpr(
                (-1, final_e.shift().loc[time_valid]), (1, final_e.loc[time_valid])
            )
            define_constraints(n, lhs,  glc.sense, rhs,  "GlobalConstraint", "mu",
                               axes=pd.Index([name]), spec=name)


def add_carbon_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2constraint"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        sense = glc.sense
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]
        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            time_valid = int(glc.loc["investment_period"]) if not math.isnan(glc.loc["investment_period"]) else n.snapshots.levels[0]
            time_weightings = n.investment_period_weightings.loc[time_valid, "years"]
            if type(time_weightings) == pd.Series:
                time_weightings = expand_series(time_weightings, stores.index)
            final_e = get_var(n, "Store", "e").groupby(level=0).last().loc[time_valid, stores.index]

            lhs = linexpr((time_weightings, final_e))
            define_constraints(n, lhs,  sense, rhs,  "GlobalConstraint", "mu",
                               axes=pd.Index([name]), spec=name)


def extra_functionality(n, snapshots):
    add_battery_constraints(n)
    add_pipe_retrofit_constraint(n)
    add_co2_sequestration_limit(n, snapshots)

    if snakemake.config['foresight']=="perfect":
        add_land_use_constraint_perfect(n)
        add_carbon_constraint(n, snapshots)
        add_carbon_neutral_constraint(n, snapshots)

    # MGA
    mga_tech = snakemake.wildcards.mga_tech.split("-")
    logger.info('MGA tech {}'.format(mga_tech))
    component = mga_tech[0]
    carrier = mga_tech[1]
    sense = snakemake.wildcards.sense
    process_objective_wildcard(n, component, carrier, sense)
    define_mga_constraint(n, snapshots)
    define_mga_objective(n)


def process_objective_wildcard(n, component, carrier, sense):
    """
    Parameters
    ----------
    n : pypsa.Network
    n.mga_obj : list-like
        [component, carrier, sense]
    """
    lookup_to_int = {"max": -1, "min": 1}

    mga_obj = [component, carrier, lookup_to_int[sense]]

    # attach to network
    n.mga_obj = mga_obj
    # print mga_obj to console
    logger.info("MGA objective {}".format(mga_obj))


def objective_constant(n, sns, ext=True, nonext=True):
    """Add capital cost of existing capacities.
    """

    if not (ext or nonext):
        return 0.0

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.objective[
                sns.unique("period")
            ]
    constant = 0.0
    for c, attr in nominal_attrs.items():
        i = pd.Index([])
        if ext:
            i = i.append(get_extendable_i(n, c))
        if nonext:
            i = i.append(get_non_extendable_i(n, c))
        cost = n.df(c)[attr][i] @ n.df(c).capital_cost[i]
        if n._multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(n, c, period)[i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        constant += cost @ n.df(c)[attr][i]

    return constant


def define_mga_constraint(n, snapshots, epsilon=None, with_fix=None):
    """Build constraint defining near-optimal feasible space
    Parameters
    ----------
    n : pypsa.Network
    snapshots : Series|list-like
        snapshots
    epsilon : float, optional
        Allowed added cost compared to least-cost solution, by default None
    with_fix : bool, optional
        Calculation of allowed cost penalty should include cost of non-extendable components, by default None
    """

    if epsilon is None:
        epsilon = float(snakemake.wildcards.epsilon)

    if with_fix is None:
        with_fix = snakemake.config.get("include_non_extendable", True)

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.objective[
            snapshots.unique("period")
        ]

    expr = pd.Series(dtype="object")
    # marginal cost
    if n._multi_invest:
        weighting = n.snapshot_weightings.objective.mul(period_weighting, level=0).loc[
            snapshots
        ]
    else:
        weighting = n.snapshot_weightings.objective.loc[snapshots]

    for c, attr in lookup.query("marginal_cost").index:
        cost = (
            get_as_dense(n, c, "marginal_cost", snapshots)
            .loc[:, lambda ds: (ds != 0).all()]
            .mul(weighting, axis=0)
        )
        if cost.empty:
            continue
        marginal_cost = linexpr((cost, get_var(n, c, attr).loc[snapshots, cost.columns])).sum()
        expr = pd.concat([expr, marginal_cost])

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(n, c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(n, c, period)[ext_i]
                    for period in snapshots.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        caps = get_var(n, c, attr).loc[ext_i]
        expr = pd.concat([expr, linexpr((cost, caps))])

    lhs = expr.sum()
    if with_fix:
        ext_const =  n.objective # objective_constant(n,snapshots, ext=True, nonext=False)
        nonext_const = n.objective_constant # objective_constant(n, snapshots, ext=False, nonext=True)
        limit = (1 + epsilon) * (n.objective + ext_const + nonext_const) - nonext_const
    else:
        ext_const = n.objective # objective_constant(n, snapshots)
        limit = (1 + epsilon) * (n.objective + ext_const)

    name = 'CostMax'
    sense = '<='

    logger.info('Add MGA constraint with epsilon: {} and limit {}'.format(epsilon, limit))

    n.add("GlobalConstraint", name, sense=sense, constant=limit,
          type=np.nan, carrier_attribute=np.nan)

    define_constraints(n, lhs, sense, limit, "GlobalConstraint", 'mu_epsilon', spec=name)


def define_mga_objective(n):

    components, pattern, sense = n.mga_obj

    if isinstance(components, str):
        components = [components]

    terms = []
    for c in components:
        if pattern == "heat pump":
            variables = get_var(n, c, nominal_attrs[c])[n.df(c)["carrier"].str.contains(pattern)]
        else:
            variables = get_var(n, c, nominal_attrs[c])[n.df(c)["carrier"]==pattern]

        if c in ["Link", "Line"] and pattern in ["", "LN|LK", "LK|LN"]:
            coeffs = sense * n.df(c).loc[variables.index, "length"]
        else:
            coeffs = sense

        terms.append(linexpr((coeffs, variables)))

    joint_terms = pd.concat(terms)

    write_objective(n, joint_terms)

def solve_network(n, config, opts='', **kwargs):
    solver_options = config['solving']['solver'].copy()
    solver_name = solver_options.pop('name')
    cf_solving = config['solving']['options']
    if snakemake.config["foresight"] == "perfect":
        cf_solving["multi_investment_periods"] = True
    track_iterations = cf_solving.get('track_iterations', False)
    min_iterations = cf_solving.get('min_iterations', 4)
    max_iterations = cf_solving.get('max_iterations', 6)
    keep_shadowprices = cf_solving.get('keep_shadowprices', True)
    multi_investment = cf_solving.get("multi_investment_periods", False)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    if cf_solving.get('skip_iterations', False):
        network_lopf(n, solver_name=solver_name, solver_options=solver_options,
                     extra_functionality=extra_functionality,
                     keep_shadowprices=keep_shadowprices,
                     multi_investment_periods=multi_investment, **kwargs)
    else:
        ilopf(n, solver_name=solver_name, solver_options=solver_options,
              track_iterations=track_iterations,
              min_iterations=min_iterations,
              max_iterations=max_iterations,
              extra_functionality=extra_functionality,
              keep_shadowprices=keep_shadowprices,
              multi_investment_periods=multi_investment,
              **kwargs)
    return n

#%%
if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'generate_alternative',
            simpl='',
            opts="",
            clusters="37",
            lv=1.0,
            sector_opts='365h-T-H-B-I-A-solar+p3-dist1-co2min',
            mga_tech="Generator-solar",
            sense="min",
            epsilon=0.1,
        )

    logging.basicConfig(filename=snakemake.log.python,
                        level=snakemake.config['logging_level'])

    update_config_with_sector_opts(snakemake.config, snakemake.wildcards.sector_opts)

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        from pathlib import Path
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
    opts = snakemake.wildcards.sector_opts.split('-')
    solve_opts = snakemake.config['solving']['options']

    fn = getattr(snakemake.log, 'memory', None)
    with memory_logger(filename=fn, interval=30.) as mem:

        overrides = override_component_attrs(snakemake.input.overrides)
        n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
        del n.global_constraints_t["land_use_constraint"]
        n.mremove("GlobalConstraint", ["co2_sequestration_limit"])

        n = prepare_network(n, solve_opts)


        n = solve_network(n, config=snakemake.config, opts=opts,
                          solver_dir=tmpdir,
                          solver_logfile=snakemake.log.solver,
                          skip_objective=True)

        if "lv_limit" in n.global_constraints.index:
            n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
            n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

        # rename columns since multi columns not working with pypsa.io
        for key in n.global_constraints_t.keys():
            n.global_constraints_t[key].columns = ['{} {}'.format(i, j) for i, j in n.global_constraints_t[key].columns]

        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
