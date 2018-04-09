#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<2017>
  Purpose: Prediict PRS using deep nets
  Created: 04/5/18
"""
import keras
import numpy as np
from utilities4cotagging import read_geno
from qtraitsimulation import qtraits_simulation
from Latinos_simulation import main as simulate

def generate_data(prefix, lenght, plink_exe, pops, threads):
    seed = 12345
    required_sample = lenght * 10
    sample_per_pop = 5000
    nhaps = [sample_per_pop] * 5
    nvars = lenght
    maf = 0.01
    to_bed = plink_exe
    ncausal = int(1E6 * 0.001)
    nsim = required_sample / sample_per_pop
    causal_pos = np.random.randint(0, nvars, )
    for i in range(nsim):
        simulate(nhaps=nhaps, nvars=nvars, maf=maf, to_bed=to_bed,
                 threads=threads, split_out=True, plot_pca=False,
                 focus_pops=pops)
        for pop in pops:
            out = qtraits_simulation(prefix, pop, 0.5, ncausal)
            pheno, realized_h2, (g, bim, truebeta, causals) = out



