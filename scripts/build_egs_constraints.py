import logging
from copy import deepcopy

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import MultiPoint, LineString, MultiPolygon, Polygon
from shapely.ops import cascaded_union


def get_spatial_exclusion_factors(faults_file,
                                  catchment_areas_file):
    """

    """    

    faults = gpd.read_file(faults_file)
    catchment_areas = gpd.read_file(catchment_areas_file)

    # determine radius from km to utm
    centroid = gpd.GeoSeries([MultiPoint(faults.centroid.tolist()).centroid])
    utm_epsg = centroid.set_crs(epsg=4326).estimate_utm_crs()
    print("utm epsg: {}".format(utm_epsg))

    # getting a line of length exclusion_radius, transform it to UTM,
    # and use to the length of that new line as exclusion radius
    delta_lat = exclusion_radius / 20_015.11 * 180
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

    polygons = polygons.set_crs(epsg=4326).to_crs(utm_epsg)
    faults = faults.set_crs(epsg=4326).to_crs(utm_epsg)

    excluded_zones = (
        gpd.GeoSeries([cascaded_union(faults.buffer(exclusion_radius).tolist())])
        .set_crs(utm_epsg)
    )

    
    