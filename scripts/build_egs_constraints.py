import logging
from copy import deepcopy

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as xrx
from tqdm import tqdm
import json

from shapely.geometry import MultiPoint, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union


def prepare_egs_data(egs_file):

    with open(egs_file) as f:
        jsondata = json.load(f)

    def point_to_square(p, lon_extent=1., lat_extent=1.):

        try:
            x, y = p.coords.xy[0][0], p.coords.xy[1][0]
        except IndexError:
            return p
        
        return Polygon([
            [x-lon_extent/2, y-lat_extent/2],
            [x-lon_extent/2, y+lat_extent/2],
            [x+lon_extent/2, y+lat_extent/2],
            [x+lon_extent/2, y-lat_extent/2],
            ])

    years = [2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
    lcoes = ["LCOE50", "LCOE100", "LCOE150"]

    for year in years:
        df = pd.DataFrame(columns=["Lon", "Lat", "CAPEX", "HeatSust", "PowerSust"])

        for lcoe in lcoes:

            for country_data in jsondata[lcoe]:
                try:
                    country_df = pd.DataFrame(columns=df.columns, 
                                              index=range(len(country_data[0][years.index(year)]["Lon"])))
                except TypeError:
                    country_df = pd.DataFrame(columns=df.columns, index=range(0))

                for col in df.columns:
                    country_df[col] = country_data[0][years.index(year)][col]

                df = pd.concat((df, country_df.dropna()), axis=0, ignore_index=True)

        gdf = gpd.GeoDataFrame(df.drop(columns=["Lon", "Lat"]), geometry=gpd.points_from_xy(df.Lon, df.Lat)).reset_index()
        gdf["geometry"] = gdf.geometry.apply(lambda geom: point_to_square(geom))

        (Path.cwd() / "data" / "egs_data").mkdir(exist_ok=True)
        gdf.to_file(f"data/egs_data/egs_potential_{year}.geojson", driver="GeoJSON")



def build_egs_potentials(
                    cost_year,
                    faults_file,
                    network_regions_file,
                    heat_demand_density_file,
                    fault_buffer,
                    ):
    """

    """

    network_regions = gpd.read_file(network_regions_file).set_crs(epsg=4326)
    network_regions.index = network_regions["name"]

    egs_data = gpd.read_file(
        f"data/egs_data/egs_potential_{cost_year}.geojson").set_crs(epsg=4326)
    egs_data.index = egs_data.geometry.astype(str)
    egs_shapes = egs_data.geometry
    
    overlap_matrix = pd.DataFrame(index=network_regions.index, columns=egs_shapes.index)
    
    for name, polygon in network_regions.geometry.items():
        overlap_matrix.loc[name] = egs_shapes.intersection(polygon).area / egs_shapes.area

    indicator_matrix = pd.DataFrame(np.ceil(overlap_matrix.values),
        index=network_regions.index, columns=egs_shapes.index)

    faults = gpd.read_file(faults_file).set_crs(epsg=4326)
    network_regions = network_regions.geometry
    heat_demand_da = xrx.open_rasterio(heat_demand_density_file)

    # determine radius from km to utm
    centroid = gpd.GeoSeries([MultiPoint(faults.centroid.tolist()).centroid])
    utm_epsg = centroid.set_crs(epsg=4326).estimate_utm_crs()
    print("Computing buffer around fault lines in epsg {}".format(utm_epsg))

    # getting a line of length exclusion_radius, transform it to UTM,
    # and use the length of that new line as exclusion radius
    delta_lat = fault_buffer / 20_015.11 * 180
    c_lon, c_lat = np.array(centroid.iloc[0].coords.xy).flatten().tolist()

    exclusion_radius = (
        gpd.GeoSeries([
            LineString([[c_lon, c_lat], [c_lon, c_lat + delta_lat]])
        ])
        .set_crs(epsg=4326)
        .to_crs(utm_epsg)
        .iloc[0]
        .length
    )

    # network_regions = network_regions.set_crs(epsg=4326).to_crs(utm_epsg)
    egs_shapes = egs_shapes.to_crs(utm_epsg)

    faults = faults.set_crs(epsg=4326).to_crs(utm_epsg)

    faults = faults.buffer(exclusion_radius)
    remaining_area = list()

    # for i, region in network_regions.items():
    for i, region in egs_shapes.items():

        mask = faults.apply(lambda geom: geom.intersects(region))
        excluded_zones = faults.loc[mask]
        region = region.buffer(0)

        for _, geom in excluded_zones.items():
            
            if not geom.area:
                continue
            region = region.difference(geom)
        
        remaining_area.append(region)
    
    remaining_area = gpd.GeoSeries(remaining_area).set_crs(utm_epsg)

    # assert len(remaining_area) == len(network_regions), "lost regions"
    assert len(remaining_area) == len(egs_shapes), "lost regions"

    # remaining_area.index = network_regions.index
    remaining_area.index = egs_shapes.index

    egs_constraints = pd.DataFrame(
        np.zeros((len(remaining_area), 2)),
        columns=["dh_share", "rural_share"],
        index=remaining_area.index
    )

    for i, idx in tqdm(enumerate(remaining_area.index)):
        
        # available_share = remaining_area.iloc[i].area / network_regions.iloc[i].area
        available_share = remaining_area.iloc[i].area / egs_shapes.iloc[i].area

        remainder = remaining_area.iloc[i:i+1].to_crs(epsg=3035)

        minx, miny, maxx, maxy = remainder.total_bounds
        remainder = remainder.iloc[0]

        subset = heat_demand_da.sel(x=slice(minx, maxx), y=slice(maxy, miny))
        subset = subset.coarsen(x=50, boundary="trim").mean()
        subset = subset.coarsen(y=50, boundary="trim").mean()

        x, y = np.meshgrid(subset.x.values, subset.y.values)
        x, y = x.flatten(), y.flatten() 
    
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_xy(x, y)).set_crs(epsg=3035)
        gdf["heat_demand"] = subset.values.flatten()
        
        in_geom_mask = gdf.geometry.within(remainder)
        gdf = gdf.loc[in_geom_mask]

        district_mask = (gdf.heat_demand > 10)

        if len(district_mask):
            district_heating_share = district_mask.sum() / len(district_mask)
        else:
            district_heating_share = 0.

        rural_share = 1 - district_heating_share


        egs_constraints.loc[idx, "dh_share"] = available_share * district_heating_share
        egs_constraints.loc[idx, "rural_share"] = available_share * rural_share

    overlap_matrix.columns = egs_shapes.astype(str).values
    indicator_matrix.columns = egs_shapes.astype(str).values

    sustainability_factor = 0.0025

    egs_data["Power"] = egs_data["PowerSust"] / sustainability_factor
    egs_data["p_nom_max"] = egs_data["Power"] * egs_constraints[["dh_share", "rural_share"]].sum(axis=1)
    egs_data["dh_share"] = egs_constraints["dh_share"].divide(egs_constraints[["dh_share", "rural_share"]].sum(axis=1))

    egs_data = pd.concat((egs_data, egs_constraints), axis=1)
    egs_data = egs_data[["p_nom_max", "CAPEX", "dh_share"]] 
    
    return egs_data, overlap_matrix, indicator_matrix


def get_capacity_factors(network_regions_file,
                         air_temperatures_file):
    """
    Performance of EGS is higher for lower temperatures, due to more efficient air cooling
    Data from Ricks et al.: The Role of Flexible Geothermal Power in Decarbonized Elec Systems
    """ 

    delta_T = [-15, -10, -5, 0, 5, 10, 15, 20]
    cf = [1.17, 1.13, 1.07, 1, 0.925, 0.84, 0.75, 0.65]

    x = np.linspace(-15, 20, 200)
    y = np.interp(x, delta_T, cf)

    upper_x = np.linspace(20, 25, 50)
    m_upper = (y[-1] - y[-2]) / (x[-1] - x[-2])
    upper_y = upper_x * m_upper - x[-1] * m_upper + y[-1]

    lower_x = np.linspace(-20, -15, 50)
    m_lower = (y[1] - y[0]) / (x[1] - x[0])
    lower_y = lower_x * m_lower - x[0] * m_lower + y[0]

    x = np.hstack((lower_x, x, upper_x))
    y = np.hstack((lower_y, y, upper_y))

    network_regions = gpd.read_file(network_regions_file).set_crs(epsg=4326)
    index = network_regions["name"]

    # air_temp = pd.read_csv(air_temperatures_file, index_col=0, parse_dates=True)
    air_temp = xr.open_dataset(air_temperatures_file)

    snapshots = air_temp.sel(name="AL1 0").to_dataframe()["temperature"].index
    capacity_factors = pd.DataFrame(index=snapshots)

    for bus in index:
        temp = air_temp.sel(name=bus).to_dataframe()["temperature"]
        capacity_factors[bus] = np.interp((temp - temp.mean()).values, x, y)
    
    return capacity_factors



logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_egs_potential",
            simpl="",
            clusters=37,
        )
    
    prepare_egs_data(snakemake.input["egs_costs"])

    config = snakemake.config

    exclusion_radius = config["sector"]["egs_fault_distance"] # default 10 km

    egs_potentials, overlap_matrix, indicator_matrix = build_egs_potentials(
                            config["costs"]["year"],
                            snakemake.input["faultlines"],
                            snakemake.input["shapes"],
                            snakemake.input["heat_demand_density"],
                            exclusion_radius,
                            )

    egs_potentials.to_csv(snakemake.output["egs_potentials"])
    overlap_matrix.to_csv(snakemake.output["egs_overlap_matrix"])
    indicator_matrix.to_csv(snakemake.output["egs_indicator_matrix"])

    capacity_factors = get_capacity_factors(
        snakemake.input["shapes"],
        snakemake.input["air_temperature"],
    )

    capacity_factors.to_csv(snakemake.output["egs_capacity_factors"])
               