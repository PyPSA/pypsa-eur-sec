import pandas as pd
import geopandas as gpd
import multiprocessing as mp
from itertools import repeat
import numpy as np

idx = pd.IndexSlice

def cartesian(s1, s2):
    """Cartesian product of two pd.Series"""
    return pd.DataFrame(np.outer(s1, s2), index=s1.index, columns=s2.index)

# translations for Eurostat
eurostat_country_to_alpha2 = {
    "EU28": "EU",
    "EA19": "EA",
    "Belgium": "BE",
    "Bulgaria": "BG",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Germany": "DE",
    "Estonia": "EE",
    "Ireland": "IE",
    "Greece": "GR",
    "Spain": "ES",
    "France": "FR",
    "Croatia": "HR",
    "Italy": "IT",
    "Cyprus": "CY",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Hungary": "HU",
    "Malta": "MA",
    "Netherlands": "NL",
    "Austria": "AT",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Slovenia": "SI",
    "Slovakia": "SK",
    "Finland": "FI",
    "Sweden": "SE",
    "United Kingdom": "GB",
    "Iceland": "IS",
    "Norway": "NO",
    "Montenegro": "ME",
    "FYR of Macedonia": "MK",
    "Albania": "AL",
    "Serbia": "RS",
    "Turkey": "TU",
    "Bosnia and Herzegovina": "BA",
    "Kosovo\n(UNSCR 1244/99)": "KO",  # 2017 version
    # 2016 version
    "Kosovo\n(under United Nations Security Council Resolution 1244/99)": "KO",
    "Moldova": "MO",
    "Ukraine": "UK",
    "Switzerland": "CH",
}

non_EU = ["NO", "CH", "ME", "MK", "RS", "BA", "AL"]

idees_rename = {"GR": "EL", "GB": "UK"}

eu28 = [
    "FR",
    "DE",
    "GB",
    "IT",
    "ES",
    "PL",
    "SE",
    "NL",
    "BE",
    "FI",
    "CZ",
    "DK",
    "PT",
    "RO",
    "AT",
    "BG",
    "EE",
    "GR",
    "LV",
    "HU",
    "IE",
    "SK",
    "LT",
    "HR",
    "LU",
    "SI",
] + ["CY", "MT"]

eu28_eea = eu28.copy()
eu28_eea.remove("GB")
eu28_eea.append("UK")


def build_eurostat(countries, year):
    """Return multi-index for all countries' energy data in TWh/a."""

    # TODO: handle in Snakefile
    # 2016 includes BA, 2017 does not
    publication_year = 2016
    fns = {
        2016: f"data/eurostat-energy_balances-june_2016_edition/{year}-Energy-Balances-June2016edition.xlsx",
        2017: f"data/eurostat-energy_balances-june_2017_edition/{year}-ENERGY-BALANCES-June2017edition.xlsx",
    }
    
    dfs = pd.read_excel(
        fns[publication_year],
        sheet_name=None,
        skiprows=1,
        index_col=list(range(4)),
    )

    # sorted_index necessary for slicing
    lookup = eurostat_country_to_alpha2
    labelled_dfs = {lookup[df.columns[0]]: df
                    for df in dfs.values()
                    if lookup[df.columns[0]] in countries}
    df = pd.concat(labelled_dfs, sort=True).sort_index()
    
    # drop non-numeric and country columns 
    non_numeric_cols = df.columns[df.dtypes != float]
    country_cols = df.columns.intersection(lookup.keys())
    to_drop = non_numeric_cols.union(country_cols)                             
    df.drop(to_drop, axis=1, inplace=True)
    
    # drop countries not included
    #include = df.index.get_level_values(0).isin(countries)
    #df = df.loc[include]

    # convert ktoe/a to TWh/a
    df *= 11.63 / 1e3

    return df


def build_swiss(year):
    """Return a pd.Series of Swiss energy data in TWh/a"""
    # TODO handle in Snakefile
    fn = "../../data/switzerland-sfoe/switzerland-new_format.csv"

    df = pd.read_csv(fn, index_col=[0,1]).loc["CH", str(year)]

    # convert PJ/a to TWh/a
    df /= 3.6

    return df


def idees_per_country(ct, year):

    ct_totals = {}

    ct_idees = idees_rename.get(ct, ct)
    fn_residential = f"data/jrc-idees-2015/JRC-IDEES-2015_Residential_{ct_idees}.xlsx"
    fn_services = f"data/jrc-idees-2015/JRC-IDEES-2015_Tertiary_{ct_idees}.xlsx"
    fn_transport = f"data/jrc-idees-2015/JRC-IDEES-2015_Transport_{ct_idees}.xlsx"

    # residential
    
    df = pd.read_excel(fn_residential, "RES_hh_fec", index_col=0)[year]

    ct_totals["total residential space"] = df["Space heating"]

    rows = ["Advanced electric heating", "Conventional electric heating"]
    ct_totals["electricity residential space"] = df[rows].sum()

    ct_totals["total residential water"] = df.at["Water heating"]

    assert df.index[23] == "Electricity"
    ct_totals["electricity residential water"] = df[23]

    ct_totals["total residential cooking"] = df["Cooking"]

    assert df.index[30] == "Electricity"
    ct_totals["electricity residential cooking"] = df[30]

    df = pd.read_excel(fn_residential, "RES_summary", index_col=0)[year]

    row = "Energy consumption by fuel - Eurostat structure (ktoe)"
    ct_totals["total residential"] = df[row]

    assert df.index[47] == "Electricity"
    ct_totals["electricity residential"] = df[47]
    
    # services

    df = pd.read_excel(fn_services, "SER_hh_fec", index_col=0)[year]

    ct_totals["total services space"] = df["Space heating"]

    rows = ["Advanced electric heating", "Conventional electric heating"]
    ct_totals["electricity services space"] = df[rows].sum()

    ct_totals["total services water"] = df["Hot water"]

    assert df.index[24] == "Electricity"
    ct_totals["electricity services water"] = df[24]

    ct_totals["total services cooking"] = df["Catering"]

    assert df.index[31] == "Electricity"
    ct_totals["electricity services cooking"] = df[31]

    df = pd.read_excel(fn_services, "SER_summary", index_col=0)[year]

    row = "Energy consumption by fuel - Eurostat structure (ktoe)"
    ct_totals["total services"] = df[row]

    assert df.index[50] == "Electricity"
    ct_totals["electricity services"] = df[50]
    
    # transport

    df = pd.read_excel(fn_transport, "TrRoad_ene", index_col=0)[year]

    ct_totals["total road"] = df["by fuel (EUROSTAT DATA)"]

    ct_totals["electricity road"] = df["Electricity"]

    ct_totals["total two-wheel"] = df["Powered 2-wheelers (Gasoline)"]

    assert df.index[19] == "Passenger cars"
    ct_totals["total passenger cars"] = df[19]

    assert df.index[30] == "Battery electric vehicles"
    ct_totals["electricity passenger cars"] = df[30]

    assert df.index[31] == "Motor coaches, buses and trolley buses"
    ct_totals["total other road passenger"] = df[31]

    assert df.index[39] == "Battery electric vehicles"
    ct_totals["electricity other road passenger"] = df[39]

    assert df.index[41] == "Light duty vehicles"
    ct_totals["total light duty road freight"] = df[41]

    assert df.index[49] == "Battery electric vehicles"
    ct_totals["electricity light duty road freight"] = df[49]

    row = "Heavy duty vehicles (Diesel oil incl. biofuels)"
    ct_totals["total heavy duty road freight"] = df[row]

    assert df.index[61] == "Passenger cars"
    ct_totals["passenger car efficiency"] = df[61]

    df = pd.read_excel(fn_transport, "TrRail_ene", index_col=0)[year]

    ct_totals["total rail"] = df["by fuel (EUROSTAT DATA)"]

    ct_totals["electricity rail"] = df["Electricity"]

    assert df.index[15] == "Passenger transport"
    ct_totals["total rail passenger"] = df[15]

    assert df.index[16] == "Metro and tram, urban light rail"
    assert df.index[19] == "Electric"
    assert df.index[20] == "High speed passenger trains"
    ct_totals["electricity rail passenger"] = df[[16, 19, 20]].sum()

    assert df.index[21] == "Freight transport"
    ct_totals["total rail freight"] = df[21]

    assert df.index[23] == "Electric"
    ct_totals["electricity rail freight"] = df[23]

    df = pd.read_excel(fn_transport, "TrAvia_ene", index_col=0)[year]

    assert df.index[6] == "Passenger transport"
    ct_totals["total aviation passenger"] = df[6]

    assert df.index[10] == "Freight transport"
    ct_totals["total aviation freight"] = df[10]

    assert df.index[7] == "Domestic"
    ct_totals["total domestic aviation passenger"] = df[7]

    assert df.index[8] == "International - Intra-EU"
    assert df.index[9] == "International - Extra-EU"
    ct_totals["total international aviation passenger"] = df[[8,9]].sum()

    assert df.index[11] == "Domestic and International - Intra-EU"
    ct_totals["total domestic aviation freight"] = df[11]

    assert df.index[12] == "International - Extra-EU"
    ct_totals["total international aviation freight"] = df[12]

    ct_totals["total domestic aviation"] = ct_totals["total domestic aviation freight"] \
                                         + ct_totals["total domestic aviation passenger"]

    ct_totals["total international aviation"] = ct_totals["total international aviation freight"] \
                                              + ct_totals["total international aviation passenger"]

    df = pd.read_excel(fn_transport, "TrNavi_ene", index_col=0)[year]

    # coastal and inland
    ct_totals["total domestic navigation"] = df["by fuel (EUROSTAT DATA)"]

    df = pd.read_excel(fn_transport, "TrRoad_act", index_col=0)[year]

    assert df.index[85] == "Passenger cars"
    ct_totals["passenger cars"] = df[85]
    
    return pd.Series(ct_totals, name=ct)


def build_idees(countries, year):

    nprocesses = mp.cpu_count()
    chunksize = max(1, int(len(countries) / nprocesses))
    args = zip(countries, repeat(2011))
    with mp.Pool(processes=nprocesses) as pool:
        totals_list = pool.starmap(idees_per_country, args, chunksize)

    totals = pd.concat(totals_list, axis=1)

    # convert ktoe to TWh
    exclude = totals.index.str.fullmatch("passenger cars")
    totals.loc[~exclude] *= 11.63 / 1e3

    # convert TWh/100km to kWh/km
    totals.loc["passenger car efficiency"] *= 10

    return totals.T

    return totals


def build_energy_totals(eurostat, swiss, idees):

    clean_df = idees.reindex(population.index).drop(
        ["passenger cars", "passenger car efficiency"], axis=1
    )

    print("International navigation")
    in_eurostat = clean_df.index.intersection(eurostat.index.levels[0])
    clean_df.loc[in_eurostat, "total international navigation"] = (
        eurostat.loc[idx[in_eurostat, :, "Bunkers", :], "Total all products"]
        .groupby(level=0)
        .sum()
    )

    clean_df.loc["CH"] = swiss

    # get values for missing countries based on Eurostat EnergyBalances
    # divide cooking/space/water according to averages in EU28

    missing = clean_df.index[clean_df["total residential"].isnull()]
    missing_in_eurostat = missing.intersection(eurostat.index.levels[0])
    uses = ["space", "cooking", "water"]

    for sector, eurostat_sector in [
        ("residential", "Residential"),
        ("services", "Services"),
        ("road", "Road"),
        ("rail", "Rail"),
    ]:
        for fuel, eurostat_fuel in [
            ("electricity", "Electricity"),
            ("total", "Total all products"),
        ]:
            clean_df.loc[missing_in_eurostat, "{} {}".format(fuel, sector)] = (
                eurostat.loc[
                    idx[missing_in_eurostat, :, :, eurostat_sector], eurostat_fuel
                ]
                .groupby(level=0)
                .sum()
            )

        if sector in ["road", "rail"]:
            continue

        fuel = "electricity"
        for use in uses:
            avg = (
                clean_df["{} {} {}".format(fuel, sector, use)]
                / clean_df["{} {}".format(fuel, sector)]
            ).mean()
            print(
                "{}: average fraction of {} for {} is {}".format(sector, fuel, use, avg)
            )
            clean_df.loc[missing_in_eurostat, "{} {} {}".format(fuel, sector, use)] = (
                avg * clean_df.loc[missing_in_eurostat, "{} {}".format(fuel, sector)]
            )

        fuel = "total"
        for use in uses:
            avg = (
                (
                    clean_df["{} {} {}".format("total", sector, use)]
                    - clean_df["{} {} {}".format("electricity", sector, use)]
                )
                / (
                    clean_df["{} {}".format("total", sector)]
                    - clean_df["{} {}".format("electricity", sector)]
                )
            ).mean()
            print(
                "{}: average fraction of non-electric for {} is {}".format(
                    sector, use, avg
                )
            )
            clean_df.loc[
                missing_in_eurostat, "{} {} {}".format(fuel, sector, use)
            ] = clean_df.loc[
                missing_in_eurostat, "{} {} {}".format("electricity", sector, use)
            ] + avg * (
                clean_df.loc[missing_in_eurostat, "{} {}".format("total", sector)]
                - clean_df.loc[
                    missing_in_eurostat, "{} {}".format("electricity", sector)
                ]
            )

    # Fix Norway space and water heating fractions
    # http://www.ssb.no/en/energi-og-industri/statistikker/husenergi/hvert-3-aar/2014-07-14
    # The main heating source for about 73 per cent of the households is based on electricity
    # => 26% is non-electric
    elec_fraction = 0.73

    without_norway = clean_df.drop("NO")

    for sector in ["residential", "services"]:

        # assume non-electric is heating
        total_heating = (
            clean_df.loc["NO", "{} {}".format("total", sector)]
            - clean_df.loc["NO", "{} {}".format("electricity", sector)]
        ) / (1 - elec_fraction)

        for use in uses:
            fraction = (
                (
                    without_norway["{} {} {}".format("total", sector, use)]
                    - without_norway["{} {} {}".format("electricity", sector, use)]
                )
                / (
                    without_norway["{} {}".format("total", sector)]
                    - without_norway["{} {}".format("electricity", sector)]
                )
            ).mean()
            clean_df.loc["NO", "{} {} {}".format("total", sector, use)] = (
                total_heating * fraction
            )
            clean_df.loc["NO", "{} {} {}".format("electricity", sector, use)] = (
                total_heating * fraction * elec_fraction
            )

    # Missing aviation
    print("Aviation")
    clean_df.loc[missing_in_eurostat, "total domestic aviation"] = (
        eurostat.loc[
            idx[missing_in_eurostat, :, :, "Domestic aviation"], "Total all products"
        ]
        .groupby(level=0)
        .sum()
    )
    clean_df.loc[missing_in_eurostat, "total international aviation"] = (
        eurostat.loc[
            idx[missing_in_eurostat, :, :, "International aviation"],
            "Total all products",
        ]
        .groupby(level=0)
        .sum()
    )

    print("Domestic navigation")
    clean_df.loc[missing_in_eurostat, "total domestic navigation"] = (
        eurostat.loc[
            idx[missing_in_eurostat, :, :, "Domestic Navigation"], "Total all products"
        ]
        .groupby(level=0)
        .sum()
    )

    # split road traffic for non-IDEES
    missing = clean_df.index[clean_df["total passenger cars"].isnull()]
    for fuel in ["total", "electricity"]:
        selection = [
            fuel + " passenger cars",
            fuel + " other road passenger",
            fuel + " light duty road freight",
        ]
        if fuel == "total":
            selection = (
                [fuel + " two-wheel"] + selection + [fuel + " heavy duty road freight"]
            )
        road = clean_df[selection].sum()
        road_fraction = road / road.sum()
        for i in road_fraction.index:
            clean_df.loc[missing, i] = (
                road_fraction[i] * clean_df.loc[missing, fuel + " road"]
            )

    # split rail traffic for non-IDEES
    missing = clean_df.index[clean_df["total rail passenger"].isnull()]
    for fuel in ["total", "electricity"]:
        selection = [fuel + " rail passenger", fuel + " rail freight"]
        rail = clean_df[selection].sum()
        rail_fraction = rail / rail.sum()
        for i in rail_fraction.index:
            clean_df.loc[missing, i] = (
                rail_fraction[i] * clean_df.loc[missing, fuel + " rail"].values
            )

    # split aviation traffic for non-IDEES
    missing = clean_df.index[clean_df["total domestic aviation passenger"].isnull()]
    for destination in ["domestic", "international"]:
        selection = [
            "total " + destination + " aviation passenger",
            "total " + destination + " aviation freight",
        ]
        aviation = clean_df[selection].sum()
        aviation_fraction = aviation / aviation.sum()
        for i in aviation_fraction.index:
            clean_df.loc[missing, i] = (
                aviation_fraction[i]
                * clean_df.loc[missing, "total " + destination + " aviation"].values
            )
    clean_df.loc[missing, "total aviation passenger"] = clean_df.loc[
        missing,
        ["total domestic aviation passenger", "total international aviation passenger"],
    ].sum(axis=1)
    clean_df.loc[missing, "total aviation freight"] = clean_df.loc[
        missing,
        ["total domestic aviation freight", "total international aviation freight"],
    ].sum(axis=1)

    if "BA" in clean_df.index:
        # fix missing data for BA (services and road energy data)
        missing = clean_df.loc["BA"] == 0.0

        # add back in proportional to RS with ratio of total residential demand
        clean_df.loc["BA", missing] = (
            clean_df.loc["BA", "total residential"]
            / clean_df.loc["RS", "total residential"]
            * clean_df.loc["RS", missing]
        )

    clean_df.to_csv(snakemake.output.energy_name)

    return clean_df


def build_eea_co2(year=1990):
    # see ../notebooks/compute_1990_Europe_emissions_for_targets.ipynb

    # https://www.eea.europa.eu/data-and-maps/data/national-emissions-reported-to-the-unfccc-and-to-the-eu-greenhouse-gas-monitoring-mechanism-16
    # downloaded 201228 (modified by EEA last on 201221)
    fn = "data/eea/UNFCCC_v23.csv"
    df = pd.read_csv(fn, encoding="latin-1")
    df.loc[df["Year"] == "1985-1987", "Year"] = 1986
    df["Year"] = df["Year"].astype(int)
    df = df.set_index(
        ["Country_code", "Pollutant_name", "Year", "Sector_name"]
    ).sort_index()

    e = pd.Series()
    e["electricity"] = "1.A.1.a - Public Electricity and Heat Production"
    e["residential non-elec"] = "1.A.4.b - Residential"
    e["services non-elec"] = "1.A.4.a - Commercial/Institutional"
    e["rail non-elec"] = "1.A.3.c - Railways"
    e["road non-elec"] = "1.A.3.b - Road Transportation"
    e["domestic navigation"] = "1.A.3.d - Domestic Navigation"
    e["international navigation"] = "1.D.1.b - International Navigation"
    e["domestic aviation"] = "1.A.3.a - Domestic Aviation"
    e["international aviation"] = "1.D.1.a - International Aviation"
    e["total energy"] = "1 - Energy"
    e["industrial processes"] = "2 - Industrial Processes and Product Use"
    e["agriculture"] = "3 - Agriculture"
    e["LULUCF"] = "4 - Land Use, Land-Use Change and Forestry"
    e["waste management"] = "5 - Waste management"
    e["other"] = "6 - Other Sector"
    e["indirect"] = "ind_CO2 - Indirect CO2"
    e["total wL"] = "Total (with LULUCF)"
    e["total woL"] = "Total (without LULUCF)"

    pol = "CO2"  # ["All greenhouse gases - (CO2 equivalent)","CO2"]

    cts = ["CH", "EUA", "NO"] + eu28_eea

    emissions = (
        df.loc[idx[cts, pol, year, e.values], "emissions"]
        .unstack("Sector_name")
        .rename(columns=pd.Series(e.index, e.values))
        .rename(index={"All greenhouse gases - (CO2 equivalent)": "GHG"}, level=1)
    )

    # only take level 0, since level 1 (pol) and level 2 (year) are trivial
    emissions = emissions.groupby(level=0, axis=0).sum()

    emissions.rename(index={"EUA": "EU28", "UK": "GB"}, inplace=True)

    emissions["industrial non-elec"] = emissions["total energy"] - emissions[
        [
            "electricity",
            "services non-elec",
            "residential non-elec",
            "road non-elec",
            "rail non-elec",
            "domestic aviation",
            "international aviation",
            "domestic navigation",
            "international navigation",
        ]
    ].sum(axis=1)

    emissions.drop(columns=["total energy", "total wL", "total woL"], inplace=True)

    return emissions / 1e3


def build_eurostat_co2(year=1990):

    eurostat_for_co2 = build_eurostat(year)

    se = pd.Series(index=eurostat_for_co2.columns, dtype=float)

    # emissions in tCO2_equiv per MWh_th
    se["Solid fuels"] = 0.36  # Approximates coal
    se["Oil (total)"] = 0.285  # Average of distillate and residue
    se["Gas"] = 0.2  # For natural gas

    # oil values from https://www.eia.gov/tools/faqs/faq.cfm?id=74&t=11
    # Distillate oil (No. 2)  0.276
    # Residual oil (No. 6)  0.298
    # https://www.eia.gov/electricity/annual/html/epa_a_03.html

    eurostat_co2 = eurostat_for_co2.multiply(se).sum(axis=1)

    return eurostat_co2


def build_co2_totals(eea_co2, eurostat_co2):

    co2 = eea_co2.reindex(["EU28", "NO", "CH", "BA", "RS", "AL", "ME", "MK"] + eu28)

    for ct in ["BA", "RS", "AL", "ME", "MK"]:
        co2.loc[ct, "electricity"] = eurostat_co2[
            ct, "+", "Conventional Thermal Power Stations", "of which From Coal"
        ].sum()
        co2.loc[ct, "residential non-elec"] = eurostat_co2[
            ct, "+", "+", "Residential"
        ].sum()
        co2.loc[ct, "services non-elec"] = eurostat_co2[ct, "+", "+", "Services"].sum()
        co2.loc[ct, "road non-elec"] = eurostat_co2[ct, "+", "+", "Road"].sum()
        co2.loc[ct, "rail non-elec"] = eurostat_co2[ct, "+", "+", "Rail"].sum()
        co2.loc[ct, "domestic navigation"] = eurostat_co2[
            ct, "+", "+", "Domestic Navigation"
        ].sum()
        co2.loc[ct, "international navigation"] = eurostat_co2[ct, "-", "Bunkers"].sum()
        co2.loc[ct, "domestic aviation"] = eurostat_co2[
            ct, "+", "+", "Domestic aviation"
        ].sum()
        co2.loc[ct, "international aviation"] = eurostat_co2[
            ct, "+", "+", "International aviation"
        ].sum()
        # doesn't include industrial process emissions or fuel processing/refining
        co2.loc[ct, "industrial non-elec"] = eurostat_co2[ct, "+", "Industry"].sum()
        # doesn't include non-energy emissions
        co2.loc[ct, "agriculture"] = eurostat_co2[
            ct, "+", "+", "Agriculture / Forestry"
        ].sum()

    return co2


def build_transport_data():

    transport_data = pd.DataFrame(
        columns=["number cars", "average fuel efficiency"], index=population.index
    )

    # collect number of cars

    transport_data["number cars"] = idees["passenger cars"]

    # CH from http://ec.europa.eu/eurostat/statistics-explained/index.php/Passenger_cars_in_the_EU#Luxembourg_has_the_highest_number_of_passenger_cars_per_inhabitant
    transport_data.loc["CH", "number cars"] = 4.136e6

    missing = transport_data.index[transport_data["number cars"].isnull()]

    print("Missing data on cars from:")

    print(missing)

    cars_pp = transport_data["number cars"] / population

    transport_data.loc[missing, "number cars"] = cars_pp.mean() * population

    # collect average fuel efficiency in kWh/km

    transport_data["average fuel efficiency"] = idees["passenger car efficiency"]

    missing = transport_data.index[transport_data["average fuel efficiency"].isnull()]

    print("Missing data on fuel efficiency from:")

    print(missing)

    transport_data.loc[missing, "average fuel efficiency"] = transport_data[
        "average fuel efficiency"
    ].mean()

    transport_data.to_csv(snakemake.output.transport_name)

    return transport_data


if __name__ == "__main__":

    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from vresutils import Dict

        snakemake = Dict()
        snakemake.output = Dict()
        snakemake.output["energy_name"] = "data/energy_totals.csv"
        snakemake.output["co2_name"] = "data/co2_totals.csv"
        snakemake.output["transport_name"] = "data/transport_data.csv"

        snakemake.input = Dict()
        snakemake.input["nuts3_shapes"] = "../pypsa-eur/resources/nuts3_shapes.geojson"

    nuts3 = gpd.read_file(snakemake.input.nuts3_shapes).set_index("index")
    population = nuts3["pop"].groupby(nuts3.country).sum()

    data_year = 2011
    eurostat = build_eurostat(data_year)
    swiss = build_swiss(data_year)
    idees = build_idees(data_year)

    build_energy_totals(eurostat, swiss, idees)

    base_year_emissions = 1990
    eea_co2 = build_eea_co2(base_year_emissions)
    eurostat_co2 = build_eurostat_co2(base_year_emissions)

    co2 = build_co2_totals(eea_co2, eurostat_co2)
    co2.to_csv(snakemake.output.co2_name)

    build_transport_data()
