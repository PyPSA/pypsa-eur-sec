import logging
from copy import deepcopy

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as xrx

from shapely.geometry import MultiPoint, LineString, MultiPolygon, Polygon
# from shapely.ops import cascaded_union
from shapely.ops import unary_union


def get_spatial_exclusion_factors(faults_file,
                                  network_regions_file,
                                  heat_demand_density_file,
                                  fault_buffer,
                                  ):
    """

    """

    faults = gpd.read_file(faults_file).set_crs(epsg=4326)
    network_regions = gpd.read_file(network_regions_file).set_crs(epsg=4326)
    network_regions.index = network_regions["name"]
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

    network_regions = network_regions.set_crs(epsg=4326).to_crs(utm_epsg)
    faults = faults.set_crs(epsg=4326).to_crs(utm_epsg)

    faults = faults.buffer(exclusion_radius)
    remaining_area = list()

    for i, region in network_regions.items():

        mask = faults.apply(lambda geom: geom.intersects(region))
        excluded_zones = faults.loc[mask]
        region = region.buffer(0)

        for _, geom in excluded_zones.items():
            
            if not geom.area:
                continue
            region = region.difference(geom)
        
        remaining_area.append(region)
    
    remaining_area = gpd.GeoSeries(remaining_area).set_crs(utm_epsg)

    assert len(remaining_area) == len(network_regions), "lost regions"

    remaining_area.index = network_regions.index

    egs_constraints = pd.DataFrame(
        np.zeros((len(remaining_area), 2)),
        columns=["district_heating_share", "rural_share"],
        index=remaining_area.index
    )

    from tqdm import tqdm
    for i, idx in tqdm(enumerate(remaining_area.index)):
        
        available_share = remaining_area.iloc[i].area / network_regions.iloc[i].area

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

        district_heating_share = district_mask.sum() / len(district_mask)
        rural_share = 1 - district_heating_share

        egs_constraints.loc[idx, "district_heating_share"] = available_share * district_heating_share
        egs_constraints.loc[idx, "rural_share"] = available_share * rural_share

    egs_constraints.to_csv("egs_constraints.csv")
    return egs_constraints 


def get_capacity_factors(do_capacity_variation,
                         network_regions_file,
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

    config = snakemake.config

    exclusion_radius = config["sector"]["egs_fault_distance"] # default 10 km

    egs_constraints = get_spatial_exclusion_factors(
                            snakemake.input["faultlines"],
                            snakemake.input["shapes"],
                            snakemake.input["heat_demand_density"],
                            exclusion_radius,
                            )

    egs_constraints.to_csv(snakemake.output["egs_spatial_constraints"])

    """
    capacity_factors = get_capacity_factors(
        config["sector"]["egs_capacity_variation"],
        snakemake.input["air_temperature"],
    )

    capacity_factors.to_csv(snakemake.output["egs_capacity_factors"])
    """
               