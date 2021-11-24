"""Plot Sankey diagram"""

import pypsa
import pandas as pd
import numpy as np
import warnings

from pypsa.descriptors import get_switchable_as_dense as as_dense
from matplotlib.colors import to_rgba
from helper import override_component_attrs


def prepare_sankey(n):

    columns = ["label", "source", "target", "value"]

    gen = (n.snapshot_weightings.generators @ n.generators_t.p).groupby([
        n.generators.carrier, n.generators.carrier, n.generators.bus.map(n.buses.carrier)
    ]).sum().div(1e6) # TWh

    gen.index.set_names(columns[:-1], inplace=True)
    gen = gen.reset_index(name='value')
    gen = gen.loc[gen.value>0.1]

    gen["source"] = gen["source"].replace({
        "gas": "fossil gas",
        "oil": "fossil oil"
    })

    sto = (n.snapshot_weightings.generators @ n.stores_t.p).groupby([
        n.stores.carrier, n.stores.carrier, n.stores.bus.map(n.buses.carrier)
    ]).sum().div(1e6)
    sto.index.set_names(columns[:-1], inplace=True)
    sto = sto.reset_index(name='value')
    sto = sto.loc[sto.value>.1]

    su = (n.snapshot_weightings.generators @ n.storage_units_t.p).groupby([
        n.storage_units.carrier, n.storage_units.carrier, n.storage_units.bus.map(n.buses.carrier)
    ]).sum().div(1e6)
    su.index.set_names(columns[:-1], inplace=True)
    su = su.reset_index(name='value')
    su = su.loc[su.value>.1]

    load = (n.snapshot_weightings.generators @ as_dense(n, "Load", "p_set")).groupby([
        n.loads.carrier, n.loads.carrier, n.loads.bus.map(n.buses.carrier)
    ]).sum().div(1e6).swaplevel() # TWh
    load.index.set_names(columns[:-1], inplace=True)
    load = load.reset_index(name='value')
    
    load = load.loc[~load.label.str.contains("emissions")]
    load.target += " demand"
    
    for i in range(5):
        n.links[f"total_e{i}"] = (n.snapshot_weightings.generators @ n.links_t[f"p{i}"]).div(1e6) # TWh
        n.links[f"carrier_bus{i}"] = n.links[f"bus{i}"].map(n.buses.carrier)
        
    def calculate_losses(x):
        energy_ports = x.loc[
            x.index.str.contains("carrier_bus") &
            ~x.str.contains("co2", na=False)
        ].index.str.replace("carrier_bus", "total_e")
        return -x.loc[energy_ports].sum()

    n.links["total_e5"] = n.links.apply(calculate_losses, axis=1)
    n.links["carrier_bus5"] = "losses"
    
    df = pd.concat([
        n.links.groupby(["carrier", "carrier_bus0", "carrier_bus" + str(i)]).sum()["total_e" + str(i)] for i in range(1,6)
    ]).reset_index()
    df.columns = columns
    
    # fix heat pump energy balance

    hp = n.links.loc[n.links.carrier.str.contains("heat pump")]

    hp_t_elec = n.links_t.p0.filter(like="heat pump")

    grouper = [hp["carrier"], hp["carrier_bus0"], hp["carrier_bus1"]]
    hp_elec = (-n.snapshot_weightings.generators @ hp_t_elec).groupby(grouper).sum().div(1e6).reset_index()
    hp_elec.columns = columns

    df = df.loc[~(df.label.str.contains("heat pump") & (df.target == 'losses'))]

    df.loc[df.label.str.contains("heat pump"), "value"] -= hp_elec["value"].values

    df.loc[df.label.str.contains("air heat pump"), "source"] = "air-sourced ambient"
    df.loc[df.label.str.contains("ground heat pump"), "source"] = "ground-sourced ambient"

    df = pd.concat([df, hp_elec])
    df = df.set_index(["label", "source", "target"]).squeeze()
    df = pd.concat([
        df.loc[df<0].mul(-1),
        df.loc[df>0].swaplevel(1, 2),
    ]).reset_index()
    df.columns = columns
    
    # make DAC demand
    df.loc[df.label=='DAC', "target"] = "DAC"
    
    to_concat = [df, gen, su, sto, load]
    connections = pd.concat(to_concat).sort_index().reset_index(drop=True)
    
    # aggregation

    src_contains = connections.source.str.contains
    trg_contains = connections.target.str.contains

    connections.loc[src_contains("low voltage"), "source"] = "AC"
    connections.loc[trg_contains("low voltage"), "target"] = "AC"
    connections.loc[src_contains("water tank"), "source"] = "water tank"
    connections.loc[trg_contains("water tank"), "target"] = "water tank"
    connections.loc[src_contains("solar thermal"), "source"] = "solar thermal"
    connections.loc[src_contains("battery"), "source"] = "battery"
    connections.loc[trg_contains("battery"), "target"] = "battery"
    connections.loc[src_contains("Li ion"), "source"] = "battery"
    connections.loc[trg_contains("Li ion"), "target"] = "battery"

    connections.loc[src_contains("heat") & ~src_contains("demand"), "source"] = "heat"
    connections.loc[trg_contains("heat") & ~trg_contains("demand"), "target"] = "heat"

    connections = connections.loc[
        ~(connections.source == connections.target) 
        & ~connections.source.str.contains("co2")
        & ~connections.target.str.contains("co2")
        & ~connections.source.str.contains("emissions")
        & ~connections.source.isin(['gas for industry', "solid biomass for industry"])
        & (connections.value >= 0.5)
    ]

    where = connections.label=='urban central gas boiler'
    connections.loc[where] = connections.loc[where].replace("losses", "fossil gas")

    connections.replace("AC", "electricity grid", inplace=True)
    
    return connections


def plot_sankey(connections, fn=None):

    if find_spec('plotly') is None:
        warnings.warn("Optional dependency 'plotly' not found. Plotting of Sankey diagram skipped. Install via 'conda install -c conda-forge plotly' or 'pip install plotly'")
        return
           
    import plotly.graph_objects as go

    labels = np.unique(connections[["source", "target"]])

    nodes = pd.Series({v: i for i, v in enumerate(labels)})

    node_colors = pd.Series(nodes.index.map(colors).fillna("grey"), index=nodes.index)

    link_colors = ["rgba{}".format(to_rgba(node_colors[src], alpha=0.5)) for src in connections.source]

    fig = go.Figure(go.Sankey(
        arrangement="snap", # [snap, nodepad, perpendicular, fixed]
        valuesuffix = "TWh",
        valueformat = ".1f",
        node = dict(
            pad=20,
            thickness=20,
            label=nodes.index,
            color=node_colors
        ),
        link = dict(
            source=connections.source.map(nodes),
            target=connections.target.map(nodes),
            value=connections.value,
            label=connections.label,
            color=link_colors,
        )
    ))

    fig.update_layout(
        title=f"Sankey Diagram: {scenario}",
        font_size=15
    )

    if fn is not None:
        fig.write_html(fn)


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'plot_sankey',
            simpl='',
            clusters=45,
            lv=1.5,
            opts='',
            sector_opts='Co2L0-168H-T-H-B-I-solar+p3-dist1',
            planning_horizons=2030,
        )


    colors = snakemake.config["plotting"]["tech_colors"]

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    connections = prepare_sankey(n)

    connections.to_csv(snakemake.output.connections)

    plot_sankey(connections, snakemake.output.sankey)
