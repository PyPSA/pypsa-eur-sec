import os

os.system("conda config --add channels http://conda.anaconda.org/gurobi")

os.system("conda install -y gurobi=8.1.0")

os.system("conda install -y git")

os.system("pip install -U git+git://github.com/PyPSA/pypsa.git@nomopyomo#egg=pypsa")

#import sys

#sys.path = ["pypsa"] + sys.path



import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
import gc
import os

import pypsa

from pypsa.linopt import get_var, linexpr, define_constraints

from pypsa.descriptors import free_output_series_dataframes

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger



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



def patch_pyomo_tmpdir(tmpdir):
    # PYOMO should write its lp files into tmp here
    import os
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    from pyutilib.services import TempfileManager
    TempfileManager.tempdir = tmpdir

def prepare_network(n, solve_opts=None):
    if solve_opts is None:
        solve_opts = snakemake.config['solving']['options']

    if 'clip_p_max_pu' in solve_opts:
        for df in (n.generators_t.p_max_pu, n.storage_units_t.inflow):
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
                t.df['marginal_cost'] += 1e-2 + 2e-3*(np.random.random(len(t.df)) - 0.5)

        for t in n.iterate_components(['Line', 'Link']):
            np.random.seed(123)
            t.df['capital_cost'] += (1e-1 + 2e-2*(np.random.random(len(t.df)) - 0.5)) * t.df['length']

    if solve_opts.get('nhours'):
        nhours = solve_opts['nhours']
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760./nhours

    return n

def add_opts_constraints(n, opts=None):
    if opts is None:
        opts = snakemake.wildcards.opts.split('-')

    if 'BAU' in opts:
        mincaps = snakemake.config['electricity']['BAU_mincapacities']
        def bau_mincapacities_rule(model, carrier):
            gens = n.generators.index[n.generators.p_nom_extendable & (n.generators.carrier == carrier)]
            return sum(model.generator_p_nom[gen] for gen in gens) >= mincaps[carrier]
        n.model.bau_mincapacities = pypsa.opt.Constraint(list(mincaps), rule=bau_mincapacities_rule)

    if 'SAFE' in opts:
        peakdemand = (1. + snakemake.config['electricity']['SAFE_reservemargin']) * n.loads_t.p_set.sum(axis=1).max()
        conv_techs = snakemake.config['plotting']['conv_techs']
        exist_conv_caps = n.generators.loc[n.generators.carrier.isin(conv_techs) & ~n.generators.p_nom_extendable, 'p_nom'].sum()
        ext_gens_i = n.generators.index[n.generators.carrier.isin(conv_techs) & n.generators.p_nom_extendable]
        n.model.safe_peakdemand = pypsa.opt.Constraint(expr=sum(n.model.generator_p_nom[gen] for gen in ext_gens_i) >= peakdemand - exist_conv_caps)

def add_eps_storage_constraint(n):
    if not hasattr(n, 'epsilon'):
        n.epsilon = 1e-5
    fix_sus_i = n.storage_units.index[~ n.storage_units.p_nom_extendable]
    n.model.objective.expr += sum(n.epsilon * n.model.state_of_charge[su, n.snapshots[0]] for su in fix_sus_i)

def add_battery_constraints(n):

    nodes = n.buses.index[n.buses.carrier == "battery"]

    link_p_nom = get_var(n, "Link", "p_nom")

    lhs = linexpr((1,link_p_nom[nodes + " charger"]),
                  (-n.links.loc[nodes + " discharger", "efficiency"].values,
                   link_p_nom[nodes + " discharger"].values))
    define_constraints(n, lhs, "=", 0, 'Link', 'charger_ratio')


def add_chp_constraints(n):

    electric = n.links.index[n.links.index.str.contains("urban central") & n.links.index.str.contains("CHP") & n.links.index.str.contains("electric")]
    heat = n.links.index[n.links.index.str.contains("urban central") & n.links.index.str.contains("CHP") & n.links.index.str.contains("heat")]

    if not electric.empty:

        link_p_nom = get_var(n, "Link", "p_nom")

        #ratio of output heat to electricity set by p_nom_ratio
        lhs = linexpr((n.links.loc[electric,"efficiency"]
                       *n.links.loc[electric,'p_nom_ratio'],
                       link_p_nom[electric]),
                      (-n.links.loc[heat,"efficiency"].values,
                       link_p_nom[heat].values))
        define_constraints(n, lhs, "=", 0, 'chplink', 'fix_p_nom_ratio')

        link_p = get_var(n, "Link", "p")

        #backpressure
        lhs = linexpr((n.links.loc[electric,'c_b'].values
                       *n.links.loc[heat,"efficiency"],
                       link_p[heat]),
                      (-n.links.loc[electric,"efficiency"].values,
                       link_p[electric].values))

        define_constraints(n, lhs, "<=", 0, 'chplink', 'backpressure')

        #top_iso_fuel_line
        lhs = linexpr((1,link_p[heat]),
                      (1,link_p[electric].values),
                      (-1,link_p_nom[electric].values))

        define_constraints(n, lhs, "<=", 0, 'chplink', 'top_iso_fuel_line')

def extra_functionality(n, snapshots):
    #add_opts_constraints(n, opts)
    #add_eps_storage_constraint(n)
    add_chp_constraints(n)
    add_battery_constraints(n)



def fix_branches(n, lines_s_nom=None, links_p_nom=None):
    if lines_s_nom is not None and len(lines_s_nom) > 0:
        n.lines.loc[lines_s_nom.index,"s_nom"] = lines_s_nom.values
        n.lines.loc[lines_s_nom.index,"s_nom_extendable"] = False
    if links_p_nom is not None and len(links_p_nom) > 0:
        n.links.loc[links_p_nom.index,"p_nom"] = links_p_nom.values
        n.links.loc[links_p_nom.index,"p_nom_extendable"] = False

def solve_network(n, config=None, solver_log=None, opts=None):
    if config is None:
        config = snakemake.config['solving']
    solve_opts = config['options']

    solver_options = config['solver'].copy()
    if solver_log is None:
        solver_log = snakemake.log.solver
    solver_name = solver_options.pop('name')

    def run_lopf(n, allow_warning_status=False, fix_zero_lines=False, fix_ext_lines=False):
        free_output_series_dataframes(n)

        if fix_zero_lines:
            fix_lines_b = (n.lines.s_nom_opt == 0.) & n.lines.s_nom_extendable
            fix_links_b = (n.links.carrier=='DC') & (n.links.p_nom_opt == 0.) & n.links.p_nom_extendable
            fix_branches(n,
                         lines_s_nom=pd.Series(0., n.lines.index[fix_lines_b]),
                         links_p_nom=pd.Series(0., n.links.index[fix_links_b]))

        if fix_ext_lines:
            fix_branches(n,
                         lines_s_nom=n.lines.loc[n.lines.s_nom_extendable, 's_nom_opt'],
                         links_p_nom=n.links.loc[(n.links.carrier=='DC') & n.links.p_nom_extendable, 'p_nom_opt'])
            if "line_volume_constraint" in n.global_constraints.index:
                n.global_constraints.drop("line_volume_constraint",inplace=True)
        else:
            if "line_volume_constraint" not in n.global_constraints.index:
                line_volume = getattr(n, 'line_volume_limit', None)
                if line_volume is not None and not np.isinf(line_volume):
                    n.add("GlobalConstraint",
                          "line_volume_constraint",
                          type="transmission_volume_expansion_limit",
                          carrier_attribute="AC,DC",
                          sense="<=",
                          constant=line_volume)


        # Firing up solve will increase memory consumption tremendously, so
        # make sure we freed everything we can
        gc.collect()

        #from pyomo.opt import ProblemFormat
        #print("Saving model to MPS")
        #n.model.write('/home/ka/ka_iai/ka_kc5996/projects/pypsa-eur/128-B-I.mps', format=ProblemFormat.mps)
        #print("Model is saved to MPS")
        #sys.exit()


        status, termination_condition = n.lopf(pyomo=False,
                                               solver_name=solver_name,
                                               solver_logfile=solver_log,
                                               solver_options=solver_options,
                                               extra_functionality=extra_functionality,
                                               formulation=solve_opts['formulation'])
                                               #extra_postprocessing=extra_postprocessing
                                               #keep_files=True
                                               #free_memory={'pypsa'}

        assert status == "ok" or allow_warning_status and status == 'warning', \
            ("network_lopf did abort with status={} "
             "and termination_condition={}"
             .format(status, termination_condition))

        if not fix_ext_lines and "line_volume_constraint" in n.global_constraints.index:
            n.line_volume_limit_dual = n.global_constraints.at["line_volume_constraint","mu"]
            print("line volume limit dual:",n.line_volume_limit_dual)

        return status, termination_condition

    lines_ext_b = n.lines.s_nom_extendable
    if lines_ext_b.any():
        # puh: ok, we need to iterate, since there is a relation
        # between s/p_nom and r, x for branches.
        msq_threshold = 0.01
        lines = pd.DataFrame(n.lines[['r', 'x', 'type', 'num_parallel']])

        lines['s_nom'] = (
            np.sqrt(3) * n.lines['type'].map(n.line_types.i_nom) *
            n.lines.bus0.map(n.buses.v_nom)
        ).where(n.lines.type != '', n.lines['s_nom'])

        lines_ext_typed_b = (n.lines.type != '') & lines_ext_b
        lines_ext_untyped_b = (n.lines.type == '') & lines_ext_b

        def update_line_parameters(n, zero_lines_below=10, fix_zero_lines=False):
            if zero_lines_below > 0:
                n.lines.loc[n.lines.s_nom_opt < zero_lines_below, 's_nom_opt'] = 0.
                n.links.loc[(n.links.carrier=='DC') & (n.links.p_nom_opt < zero_lines_below), 'p_nom_opt'] = 0.

            if lines_ext_untyped_b.any():
                for attr in ('r', 'x'):
                    n.lines.loc[lines_ext_untyped_b, attr] = (
                        lines[attr].multiply(lines['s_nom']/n.lines['s_nom_opt'])
                    )

            if lines_ext_typed_b.any():
                n.lines.loc[lines_ext_typed_b, 'num_parallel'] = (
                    n.lines['s_nom_opt']/lines['s_nom']
                )
                logger.debug("lines.num_parallel={}".format(n.lines.loc[lines_ext_typed_b, 'num_parallel']))

        iteration = 1

        lines['s_nom_opt'] = lines['s_nom'] * n.lines['num_parallel'].where(n.lines.type != '', 1.)
        status, termination_condition = run_lopf(n, allow_warning_status=True)

        def msq_diff(n):
            lines_err = np.sqrt(((n.lines['s_nom_opt'] - lines['s_nom_opt'])**2).mean())/lines['s_nom_opt'].mean()
            logger.info("Mean square difference after iteration {} is {}".format(iteration, lines_err))
            return lines_err

        min_iterations = solve_opts.get('min_iterations', 2)
        max_iterations = solve_opts.get('max_iterations', 999)

        while msq_diff(n) > msq_threshold or iteration < min_iterations:
            if iteration >= max_iterations:
                logger.info("Iteration {} beyond max_iterations {}. Stopping ...".format(iteration, max_iterations))
                break

            update_line_parameters(n)
            lines['s_nom_opt'] = n.lines['s_nom_opt']
            iteration += 1

            status, termination_condition = run_lopf(n, allow_warning_status=True)

        update_line_parameters(n, zero_lines_below=100)

        logger.info("Starting last run with fixed extendable lines")

        # Not really needed, could also be taken out
        # if 'snakemake' in globals():
        #     fn = os.path.basename(snakemake.output[0])
        #     n.export_to_netcdf('/home/vres/data/jonas/playground/pypsa-eur/' + fn)

    status, termination_condition = run_lopf(n, fix_ext_lines=True)

    # Drop zero lines from network
    # zero_lines_i = n.lines.index[(n.lines.s_nom_opt == 0.) & n.lines.s_nom_extendable]
    # if len(zero_lines_i):
    #     n.mremove("Line", zero_lines_i)
    # zero_links_i = n.links.index[(n.links.p_nom_opt == 0.) & n.links.p_nom_extendable]
    # if len(zero_links_i):
    #     n.mremove("Link", zero_links_i)


    return n

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict
        snakemake = MockSnakemake(
            wildcards=dict(network='elec', simpl='', clusters='45', lv='1.25', opts='Co2L-3H-T-H'),
            input=dict(network="networks/{network}_s{simpl}_{clusters}_lv{lv}_{opts}.nc"),
            output=["results/networks/s{simpl}_{clusters}_lv{lv}_{opts}-test.nc"],
            log=dict(gurobi="logs/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_gurobi-test.log",
                     python="logs/{network}_s{simpl}_{clusters}_lv{lv}_{opts}_python-test.log")
        )


    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        patch_pyomo_tmpdir(tmpdir)

    logging.basicConfig(filename=snakemake.log.python,
                        level=snakemake.config['logging_level'])

    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:
        n = pypsa.Network(snakemake.input.network,
                          override_component_attrs=override_component_attrs)

        n = prepare_network(n)
        n = solve_network(n)

        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
