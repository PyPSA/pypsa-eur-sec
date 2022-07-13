#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given a scaling folder, create a number of scaling folders that sweep from 0 scaling to those values
"""

import os
import glob
import pandas as pd
import shutil
import numpy as np


STEPS = 10
noscaling_dir = "noscaling"
scaling_dir = "Grubler"

files = {}
for f in glob.glob("../scaling/"+scaling_dir+"/*.csv"):
    files[f.split("/")[-1]] = pd.read_csv(f,index_col=0)
    
for s in range(1,STEPS+1):
    try:
        newdir="../scaling/"+scaling_dir+"-"+str(s)+"of"+str(STEPS)
        os.mkdir(newdir)
        
    except OSError as error:
        print(error)    
        
    for f in files.keys():
        newf = files[f].copy()
        for i in files[f].index:
            for c in files[f].columns:
                if not np.isnan(newf[c][i]):
                    newf.loc[i,c] = 1 - (s/STEPS) * (1-newf.loc[i,c])
        print(newdir+"/"+f)
        newf.to_csv(newdir+"/"+f)