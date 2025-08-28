#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:30:19 2024

@author: Zhang et al.
"""

from setuptools import setup, find_packages

setup(
    name='Stressdrop',
    version='0.10',
    packages=find_packages(),
    install_requires=[
        'obspy', 'numpy', 'pandas', 'matplotlib', 'scipy', 
        'joblib', 'pyproj','multitaper','openpyxl'
    ],
    
)
