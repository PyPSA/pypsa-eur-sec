"""
Inputs
------
- ``mmmc3.xlsx``: for potential of electricity from enhanced geothermal in 5 year
steps between 2015 and 2015 in a country-resolution (from "From hot rock to useful...")

- ``Geothermal_CapexOpexEurope.xlsx``: Marginal and capital cost

Note the data comes in three steps of LCOE: 50, 100, 150 Euro/MWh
For each three we have: Maximal nominal power, marginal cost, capital cost
An approach to modelling this would be splitting the steps into 3 generators

Used data comes from the paper
"From hot rock to useful energy: A global estimate of enhanced geothermal potential"
~ Aghahosseini, Breyer (2020)

Outputs
-------

- ``lukas_resources/egs_potential_profiles_50.nc``
- ``lukas_resources/egs_potential_profiles_100.nc``
- ``lukas_resources/egs_potential_profiles_150.nc``
"""

import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import pycountry
from copy import deepcopy

# clearing inconsistencies in source data
countryname_mapper = {
    'Macedonia': 'North Macedonia',
    'Bosnia and Herz.': 'Bosnia and Herzegovina',
    'Czech Republic': 'Czechia',
}

# for i, row in pypsa_countries.iterrows():
#     if row.full_names in countryname_mapper:
#         pypsa_countries.at[i, 'full_names'] = countryname_mapper[row.full_names]

def get_egs_potentials(potentials_file, costs_file, shapes_file):
    """
    Disaggregates data to the provided shapefile
    
    Args:
        potentials_file(str or pathlib.Path): file with potentials
        costs_file(str or pathlib.Path): file with capital and marginal costs
        shapes_file(pathlib.Path): path to shapefiles to which data is disagreggated
    """
    shapes = gpd.read_file(shapes_file)

    times = pd.date_range('2015-01-01', '2055-01-01', freq='5y')

    cost_cutoffs = ["150", "100", "50"]
    egs_data = dict()
    
    for cutoff in cost_cutoffs:
        inner = dict()
        inner['sus_potential'] = pd.DataFrame(index=times, columns=shapes['name'])
        inner['marginal_cost'] = deepcopy(inner['sus_potential'])
        inner['capital_cost'] = deepcopy(inner['sus_potential'])
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

    # print("Potential data is not correct yet!")
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

            egs_data[cutoff]['sus_potential'].loc[time] = potential
    
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

            egs_data[cutoff]['capital_cost'].loc[col] = capex_by_shape
            egs_data[cutoff]['marginal_cost'].loc[col] = opex_by_shape

    return egs_data



logger = logging.getLogger(__name__)

from pathlib import Path

### temporary mess
pypsa_eur_path = Path.cwd() / '..' / '..' / 'lab_pypsa_eur_sec' / 'pypsa-eur'
respath = pypsa_eur_path / 'resources' 
egspath = Path.cwd() / "lukas_resources"

class MegaMockSnakemake:
    egs_potential = egspath / 'mmc3.xlsx'
    egs_cost = egspath / 'Geothermal_CapexOpex_Europe.xlsx'
    input_shape = respath / 'regions_onshore_elec_s_256.geojson'
    
    output = dict(
        egs_potential_50=respath/"egs_potential_profiles_50.csv",
        egs_potential_100=respath/"egs_potential_profiles_100.csv",
        egs_potential_150=respath/"egs_potential_profiles_150.csv",
        )

    def __init__(self):
        pass


if __name__ == "__main__":
    mms = MegaMockSnakemake()
    
    egs_data = get_egs_potentials(
        mms.egs_potential,
        mms.egs_cost,
        mms.input_shape 
        )
    
    for cutoff in ['50', '100', '150']:
        data = egs_data[cutoff]
        data.to_csv(mms.output[f"egs_potential_{cutoff}"])