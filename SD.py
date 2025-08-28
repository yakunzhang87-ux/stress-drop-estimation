#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:43:38 2025

@author: Zhang et al.
"""
import sys
import numpy as np
import time
import traceback
from SDpy import StressDrop as sd

try:
    p_dict_init=sd.Para_init()
    controlfile=str(sys.argv[1])
    p_dict_init['controlfile']=controlfile
except:
    traceback.print_exc()

if p_dict_init['controlfile']:
    start_time = time.time()
    final_out=sd.stressdrop(controlfile)
    print(f"Cost Time: {np.ceil(time.time() - start_time)} seconds")
else:
    print('The control file path is not provided!')
