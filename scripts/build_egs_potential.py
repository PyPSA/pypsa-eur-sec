"""
Inputs
------
- ``data/egs_data/egs_global_potential.xlsx``: for potential of electricity from enhanced geothermal in 5 year
steps between 2015 and 2015 in a country-resolution (from "From hot rock to useful...")

- ``data/egs_data/egs_costs.xlsx``: Marginal and capital cost

Note the data comes in three steps of LCOE: 50, 100, 150 Euro/MWh
For each three we have: Maximal nominal power, marginal cost, capital cost
The current approach models this by splitting the steps into 3 generators

Used data comes from the paper
"From hot rock to useful energy: A global estimate of enhanced geothermal potential"
~ Aghahosseini, Breyer (2020)

Other useful resources:
    For estimations of used efficiencies:
    "Overcoming challenges in the classification of deep geothermal potential"
    ~ Breede et al. 2015
    For organic rankine cycles relevant at the production temperatures of 150-300 degrees:
    "Techno-economic survey of Organic Rankine Cycle (ORC) systems"
    ~ Quoilin et al. 2013

Outputs
-------
(see Snakefile)
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import pycountry
from copy import deepcopy

# clearing inconsistencies in source data
countryname_mapper = {
    'Macedonia': 'North Macedonia',
    'Bosnia and Herz.': 'Bosnia and Herzegovina',
    'Czech Republic': 'Czechia',
}


def get_egs_potentials(potentials_file,
                       sustainable_potentials_file,
                       costs_file,
                       shapes_file,
                       ):
    """
    Disaggregates data to the provided shapefile

    Args:
        potentials_file(str or pathlib.Path): file with potentials
        sustainable_potentials_file(str or pathlib.Path): currently not considered
        costs_file(str or pathlib.Path): file with capital and marginal costs
        shapes_file(pathlib.Path): path to shapefiles to which data is disagreggated
    """
    shapes = gpd.read_file(shapes_file)

    times = pd.date_range('2015-01-01', '2055-01-01', freq='5y')

    # Concerning the cost of egs:
    # The originating paper (see above) provides LCOE via 
    # (CAPEX + OPEX_fixed) / E + OPEX_var
    # where OPEX_var is zero (see the paper's supp)
    # E is the total energy generated during lifetime
    # Upon request, the authors shared CAPEX and OPEX_fixed on a 
    # country level.
    # CAPEX is interpreted as investment cost as in pypsa-eur/data/costs.csv
    # OPEX_fixed is interpreted as FOM as in pypsa-eur/data/costs.csv
    # OPEX_var is interpreted as VOM as in pypsa-eur/data/costs.csv
    # from this, Nyears is the egs lifetime, discount rate is taken from
    # config.yaml
    # The procedure to compute capital and marginal cost for the optimization
    # is copied from pypsa-eur/scripts/add_electricity.py (see load_costs(...))

    # The conversion from opex, capex -> capital, marginal cost is executed 
    # in scripts/prepare_sector_network.py

    cost_cutoffs = ["150", "100", "50"]
    egs_data = dict()

    for cutoff in cost_cutoffs:
        inner = dict()
        inner['potential'] = pd.DataFrame(index=times, columns=shapes['name'])
        inner['opex_fixed'] = deepcopy(inner['potential'])
        inner['capex'] = deepcopy(inner['potential'])
        egs_data[cutoff] = inner

    shapes['area'] = shapes.geometry.apply(lambda geom: geom.area)
    shapes['country'] = shapes['name'].apply(
        lambda name: pycountry.countries.get(alpha_2=name[:2]).name)

    areas = pd.DataFrame(shapes.groupby('country').sum()['area'])

    def get_overlap(country_row, shape_row):
        if shape_row.country != country_row.name:
            return 0.
        else:
            return shape_row.geometry.area / country_row.area

    for i, shape_row in shapes.iterrows():
        areas[shape_row['name']] = areas.apply(
            lambda country_row: get_overlap(country_row, shape_row),
            axis=1
        )

    country_shares = areas.drop(columns=['area'])
    assigner = np.ceil(country_shares)

    cutoff_slices = [slice(18,19), slice(8,18), slice(0,8)]

    for cutoff, cutoff_slice in zip(cost_cutoffs, cutoff_slices):

        for i, time in enumerate(times):

            potential = pd.read_excel(potentials_file, sheet_name=(i+1), index_col=0)
            potential = potential[[col for col in potential.columns if 'Power' in col]]
            potential = potential.loc["Afghanistan":"Zimbabwe"]
            potential = potential.rename(index=countryname_mapper)

            potential = potential.loc[areas.index]
            potential = potential[potential.columns[cutoff_slice]]
            potential = potential.sum(axis=1)
            
            potential = country_shares.transpose() @ potential

            egs_data[cutoff]['potential'].loc[time] = potential

    # loading capex and opex at difference LCOE cutoffs
    cutoff_skiprows = [1, 44, 87]
    capex_usecols = slice(1, 9)
    opex_usecols = slice(10, 18)

    for cutoff, skiprows in zip(cost_cutoffs, cutoff_skiprows):

        prices = pd.read_excel(costs_file,
                            sheet_name=1,
                            index_col=0,
                            skiprows=skiprows,
                            )

        prices = prices.iloc[:38]
        prices = prices.rename(index=countryname_mapper)
        prices = prices.loc[areas.index]

        capex = prices[prices.columns[capex_usecols]]
        opex = prices[prices.columns[opex_usecols]]

        opex.columns = times
        capex.columns = times

        for col in capex.columns:

            capex_by_shape = assigner.transpose() @ capex[col]
            opex_by_shape = assigner.transpose() @ opex[col]

            egs_data[cutoff]['capex'].loc[col] = capex_by_shape
            egs_data[cutoff]['opex_fixed'].loc[col] = opex_by_shape

    return egs_data


logger = logging.getLogger(__name__)

if __name__ == "__main__":

    if "snakemake" not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            "build_egs_potential",
            simpl="",
            clusters=48,
        )

    egs_data = get_egs_potentials(
        snakemake.input["egs_potential"],
        snakemake.input["egs_sustainable_potential"],
        snakemake.input["egs_cost"],
        snakemake.input["shapes"],
        )

    for cutoff in ["50", "100", "150"]:
        data = egs_data[cutoff]

        time = data["potential"].index
        countries = data["potential"].columns
        dims = ["time", "countries"]

        ds = xr.Dataset(
            data_vars=dict(
                potential=(dims, data["potential"]),
                opex_fixed=(dims, data["opex_fixed"]),
                capex=(dims, data["capex"]),
                ),
            coords={
                'time': (("time"), time),
                'countries': (("countries"), countries),
                },
            attrs=dict(units='Fill in units!'))

        ds.to_netcdf(snakemake.output[f"egs_potential_{cutoff}"])
