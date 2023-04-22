import logging
from copy import deepcopy

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as xrx

from shapely.geometry import MultiPoint, LineString, MultiPolygon, Polygon
from shapely.ops import cascaded_union


def get_spatial_exclusion_factors(faults_file,
                                  network_regions_file,
                                  heat_demand_density_file,
                                  fault_buffer,
                                  ):
    """

    """    

    faults = gpd.read_file(faults_file).set_crs(epsg=4326)
    network_regions = gpd.read_file(network_regions_file).set_crs(epsg=4326)
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

    excluded_zones = (
        gpd.GeoSeries([cascaded_union(faults.buffer(exclusion_radius).tolist())])
        .set_crs(utm_epsg)
    )

    remaining_area = list()
    
    for region in network_regions:
        mask = excluded_zones.apply(lambda geom: geom.intersects(region))

        for i, geom in excluded_zones.loc[mask].iteritems():
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

    for i, idx in enumerate(remaining_area.index):
        
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

    return egs_constraints 

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helper import mock_snakemake

        snakemake = mock_snakemake(
            "build_egs_potential",
            simpl="",
            clusters=48,
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
               