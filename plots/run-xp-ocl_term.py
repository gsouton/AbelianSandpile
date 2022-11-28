#!/usr/bin/env python3
from expTools import *

easypapOptions = {
    "-k ": ["ssandPile"],
    "-o ": [""],
    "-v ": ["ocl_term"],
    "-s ": [512],
    "-n ": [""],
    "-of ": ["ocl_term.csv"]
}

# OMP Internal Control Variable
ompICV = {
    "OMP_NUM_THREADS=": [69190] + list(range(1, 20000, 500))
}

nbrun = 1
# Lancement des experiences
execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

