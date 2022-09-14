#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:21:12 2022

@author: poweruser
"""

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
from plot_network import plot_map_without, plot_series

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake(
            'plot_network',
            simpl='',
            clusters="128",
            scal="Tango",
            lv=1.0,
            opts='',
            sector_opts='3H-T-H-B-I-A',
            planning_horizons="2050",
        )


net = pypsa.Network("../results/Tango_128n_3H_netzero/postnetworks/elec_sc_Tango_s_128_lvopt__Co2L0p00-3H-T-H-B-I-A_2050.nc")

#plot_series(net, carrier="AC", name="test")
plot_map_without(net)

