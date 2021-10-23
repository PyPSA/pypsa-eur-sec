import pypsa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse

from make_summary import assign_carriers
from plot_summary import rename_techs, preferred_order
from helper import override_component_attrs

plt.style.use('ggplot')


def rename_techs_tyndp(tech):
    tech = rename_techs(tech)
    if "heat pump" in tech or "resistive heater" in tech:
        return "power-to-heat"
    elif tech in ["H2 Electrolysis", "methanation", "helmeth", "H2 liquefaction"]:
        return "power-to-gas"
    elif tech == "H2":
        return "H2 storage"
    elif tech in ["OCGT", "CHP", "gas boiler", "H2 Fuel Cell"]:
        return "gas-to-power/heat"
    elif "solar" in tech:
        return "solar"
    elif tech == "Fischer-Tropsch":
        return "power-to-liquid"
    elif "offshore wind" in tech:
        return "offshore wind"
    elif "CC" in tech or "sequestration" in tech:
        return "CCS"
    else:
        return tech


def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0] * (72. / fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5 * width - 0.5 * xdescent, 0.5 *
                        height - 0.5 * ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}


def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]


def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1: continue
            names = ifind.index[ifind == i]
            c.df.loc[names, 'location'] = names.str[:i]


def plot_map(network, components=["links", "stores", "storage_units", "generators"],
             bus_size_factor=1.7e10, transmission=False):

    n = network.copy()
    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    costs = pd.DataFrame(index=n.buses.index)

    for comp in components:
        df_c = getattr(n, comp)
        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        costs_c = ((df_c.capital_cost * df_c[attr])
                   .groupby([df_c.location, df_c.nice_group]).sum()
                   .unstack().fillna(0.))
        costs = pd.concat([costs, costs_c], axis=1)

        print(comp, costs)

    costs = costs.groupby(costs.columns, axis=1).sum()

    costs.drop(list(costs.columns[(costs == 0.).all()]), axis=1, inplace=True)

    new_columns = (preferred_order.intersection(costs.columns)
                   .append(costs.columns.difference(preferred_order)))
    costs = costs[new_columns]

    for item in new_columns:
        if item not in snakemake.config['plotting']['tech_colors']:
            print("Warning!",item,"not in config/plotting/tech_colors")

    costs = costs.stack()  # .sort_index()

    # hack because impossible to drop buses...
    n.buses.loc["EU gas", ["x", "y"]] = n.buses.loc["DE0 0", ["x", "y"]]

    n.links.drop(n.links.index[(n.links.carrier != "DC") & (
        n.links.carrier != "B2B")], inplace=True)

    # drop non-bus
    to_drop = costs.index.levels[0].symmetric_difference(n.buses.index)
    if len(to_drop) != 0:
        print("dropping non-buses", to_drop)
        costs.drop(to_drop, level=0, inplace=True, axis=0)

    # make sure they are removed from index
    costs.index = pd.MultiIndex.from_tuples(costs.index.values)

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.
    line_upper_threshold = 1e4
    linewidth_factor = 2e3
    ac_color = "gray"
    dc_color = "m"

    if snakemake.wildcards["lv"] == "1.0":
        # should be zero
        line_widths = n.lines.s_nom_opt - n.lines.s_nom
        link_widths = n.links.p_nom_opt - n.links.p_nom
        title = "Transmission reinforcement"

        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            linewidth_factor = 2e3
            line_lower_threshold = 0.
            title = "Today's transmission"
    else:
        line_widths = n.lines.s_nom_opt - n.lines.s_nom_min
        link_widths = n.links.p_nom_opt - n.links.p_nom_min
        title = "Transmission reinforcement"

        if transmission:
            line_widths = n.lines.s_nom_opt
            link_widths = n.links.p_nom_opt
            title = "Total transmission"

    line_widths[line_widths < line_lower_threshold] = 0.
    link_widths[link_widths < line_lower_threshold] = 0.

    line_widths[line_widths > line_upper_threshold] = line_upper_threshold
    link_widths[link_widths > line_upper_threshold] = line_upper_threshold

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    fig.set_size_inches(7, 6)

    n.plot(
        bus_sizes=costs / bus_size_factor,
        bus_colors=snakemake.config['plotting']['tech_colors'],
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,  **map_opts
    )

    handles = make_legend_circles_for(
        [5e9, 1e9],
        scale=bus_size_factor,
        facecolor="gray"
    )

    labels = ["{} bEUR/a".format(s) for s in (5, 1)]

    l2 = ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 1.01),
        labelspacing=1.0,
        frameon=False,
        title='System cost',
        handler_map=make_handler_map_to_scale_circles_as_in(ax)
    )

    ax.add_artist(l2)

    handles = []
    labels = []

    for s in (10, 5):
        handles.append(plt.Line2D([0], [0], color=ac_color,
                                  linewidth=s * 1e3 / linewidth_factor))
        labels.append("{} GW".format(s))

    l1_1 = ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(0.22, 1.01),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1.5,
        title=title
    )

    ax.add_artist(l1_1)

    fig.savefig(
        snakemake.output.map,
        transparent=True,
        bbox_inches="tight"
    )


def plot_h2_map(network):

    n = network.copy()
    if "H2 pipeline" not in n.links.carrier.unique():
        return

    assign_location(n)

    bus_size_factor = 1e5
    linewidth_factor = 1e4
    # MW below which not drawn
    line_lower_threshold = 1e3
    bus_color = "m"
    link_color = "c"

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    elec = n.links.index[n.links.carrier == "H2 Electrolysis"]

    bus_sizes = n.links.loc[elec,"p_nom_opt"].groupby(n.links.loc[elec, "bus0"]).sum() / bus_size_factor

    # make a fake MultiIndex so that area is correct for legend
    bus_sizes.index = pd.MultiIndex.from_product(
        [bus_sizes.index, ["electrolysis"]])

    n.links.drop(n.links.index[n.links.carrier != "H2 pipeline"], inplace=True)

    link_widths = n.links.p_nom_opt / linewidth_factor
    link_widths[n.links.p_nom_opt < line_lower_threshold] = 0.

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")

    print(link_widths.sort_values())

    print(n.links[["bus0", "bus1"]])

    fig, ax = plt.subplots(
        figsize=(7, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    n.plot(
        bus_sizes=bus_sizes,
        bus_colors={"electrolysis": bus_color},
        link_colors=link_color,
        link_widths=link_widths,
        branch_components=["Link"],
        ax=ax,  **map_opts
    )

    handles = make_legend_circles_for(
        [50000, 10000],
        scale=bus_size_factor,
        facecolor=bus_color
    )

    labels = ["{} GW".format(s) for s in (50, 10)]

    l2 = ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.01),
        labelspacing=1.0,
        frameon=False,
        title='Electrolyzer capacity',
        handler_map=make_handler_map_to_scale_circles_as_in(ax)
    )

    ax.add_artist(l2)

    handles = []
    labels = []

    for s in (50, 10):
        handles.append(plt.Line2D([0], [0], color=link_color,
                                  linewidth=s * 1e3 / linewidth_factor))
        labels.append("{} GW".format(s))

    l1_1 = ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(0.28, 1.01),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1.5,
        title='H2 pipeline capacity'
    )

    ax.add_artist(l1_1)

    fig.savefig(
        snakemake.output.map.replace("-costs-all","-h2_network"),
        transparent=True,
        bbox_inches="tight"
    )


def plot_map_without(network):

    n = network.copy()
    assign_location(n)

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    fig, ax = plt.subplots(
        figsize=(7, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # PDF has minimum width, so set these to zero
    line_lower_threshold = 200.
    line_upper_threshold = 1e4
    linewidth_factor = 2e3
    ac_color = "gray"
    dc_color = "m"

    # hack because impossible to drop buses...
    n.buses.loc["EU gas", ["x", "y"]] = n.buses.loc["DE0 0", ["x", "y"]]

    to_drop = n.links.index[(n.links.carrier != "DC") & (n.links.carrier != "B2B")]
    n.links.drop(to_drop, inplace=True)

    if snakemake.wildcards["lv"] == "1.0":
        line_widths = n.lines.s_nom
        link_widths = n.links.p_nom
    else:
        line_widths = n.lines.s_nom_min
        link_widths = n.links.p_nom_min

    line_widths[line_widths < line_lower_threshold] = 0.
    link_widths[link_widths < line_lower_threshold] = 0.

    line_widths[line_widths > line_upper_threshold] = line_upper_threshold
    link_widths[link_widths > line_upper_threshold] = line_upper_threshold

    n.plot(
        bus_colors="k",
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax, **map_opts
    )

    handles = []
    labels = []

    for s in (10, 5):
        handles.append(plt.Line2D([0], [0], color=ac_color,
                                  linewidth=s * 1e3 / linewidth_factor))
        labels.append("{} GW".format(s))
    l1_1 = ax.legend(handles, labels,
                     loc="upper left", bbox_to_anchor=(0.05, 1.01),
                     frameon=False,
                     labelspacing=0.8, handletextpad=1.5,
                     title='Today\'s transmission')
    ax.add_artist(l1_1)

    fig.savefig(
        snakemake.output.today,
        transparent=True,
        bbox_inches="tight"
    )


def plot_series(network, carrier="AC", name="test"):

    n = network.copy()
    assign_location(n)
    assign_carriers(n)

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):
        n_port = 4 if c.name=='Link' else 2
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),
                               axis=1)

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = supply.groupby(rename_techs_tyndp, axis=1).sum()

    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix

    supply = pd.concat((supply, negative_supply), axis=1)

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = "2013-02-19"
    stop = "2013-02-26"

    threshold = 10e3

    to_drop = supply.columns[(abs(supply) < threshold).all()]

    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    supply.index.name = None

    supply = supply / 1e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand"},
                  inplace=True)
    supply.columns = supply.columns.str.replace("residential ", "")
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["electric demand",
                                "transmission lines",
                                "hydroelectricity",
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
                                "CHP",
                                "onshore wind",
                                "offshore wind",
                                "solar PV",
                                "solar thermal",
                                "building retrofitting",
                                "ground heat pump",
                                "air heat pump",
                                "resistive heater",
                                "OCGT",
                                "gas boiler",
                                "gas",
                                "natural gas",
                                "methanation",
                                "hydrogen storage",
                                "battery storage",
                                "hot water storage"])

    new_columns = (preferred_order.intersection(supply.columns)
                   .append(supply.columns.difference(preferred_order)))

    supply =  supply.groupby(supply.columns, axis=1).sum()
    fig, ax = plt.subplots()
    fig.set_size_inches((8, 5))

    (supply.loc[start:stop, new_columns]
     .plot(ax=ax, kind="area", stacked=True, linewidth=0.,
           color=[snakemake.config['plotting']['tech_colors'][i.replace(suffix, "")]
                  for i in new_columns]))

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])

    ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
    ax.set_xlim([start, stop])
    ax.set_ylim([-1300, 1900])
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    fig.tight_layout()

    fig.savefig("{}{}/maps/series-{}-{}-{}-{}-{}.pdf".format(
        snakemake.config['results_dir'], snakemake.config['run'],
        snakemake.wildcards["lv"],
        carrier, start, stop, name),
        transparent=True)


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'plot_network',
            simpl='',
            clusters=45,
            lv=1.5,
            opts='',
            sector_opts='Co2L0-168H-T-H-B-I-solar+p3-dist1',
            planning_horizons=2030,
        )

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    map_opts = snakemake.config['plotting']['map']

    plot_map(n,
        components=["generators", "links", "stores", "storage_units"],
        bus_size_factor=1.5e10,
        transmission=False
    )

    plot_h2_map(n)
    plot_map_without(n)

    #plot_series(n, carrier="AC", name=suffix)
    #plot_series(n, carrier="heat", name=suffix)
