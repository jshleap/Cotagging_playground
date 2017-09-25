'''
Co-Tagging
'''
__author__ = 'jshleap'

import argparse, os
from subprocess import Popen
import numpy as np
from time import sleep
from glob import glob as G

## Constants
qsub='''#!/bin/bash
#PBS -l nodes=1:ppn=16,walltime=%s:00:00
#PBS -A ams-754-aa
#PBS -o %s.out
#PBS -e %s.err
#PBS -N %s
    
module load python/2.7.9
module load foss/2015b R/3.3.1

cd  $PBS_O_WORKDIR

%s
'''
## Functions
#----------------------------------------------------------------------
def Parallel_GRM(args,qsub=qsub):
    """
    Create the triangular genetic relationship matrix using plink in a HPC 
    cluster (Guillimin)
    """
    pref = args.prefix
    comm = ('%s --bfile %s --keep-allele-order --make-rel --parallel %d %d -out'
            ' %s.GRM')
    cat = ['cat']
    for i in xrange(1, args.chunks+1):
        tname = '%s_%d'%(pref,i)
        c = comm%(args.PlinkExe, args.GenotypeFile, i, args.chunks, pref)
        with open('temp.%d.sh'%(i)) as qs:
            qs.write(qsub%(tname, tname, tname, c), 'w')
        ch = Popen('qsub temp.%d.sh'%(i), shell=True)
        ch.communicate()
        cat.append('%s.GRM.rel'%(pref))
    ## join the results
    #count=0
    while not os.path.isfile('%s.GRM.rel.bin.%d'%(pref, args.chunks+1)) and \
          (len(G('%s.GRM.rel.bin.*'%(pref))) != args.chunks):
        sleep(1800)
        #count += 1
    fn = '%s.fullGRM'%(pref)
    cat = Popen('%s > %s'%(' '.join(cat), fn), shell=True)
    cat.communicate()
    return fn

#----------------------------------------------------------------------
def single_GRM(args):
    """Create the triangular genetic relationship matrix using plink"""
    pref = args.prefix
    comm = '%s --bfile %s --make-rel -out %s.GRM --keep-allele-order'    
    c = comm%(args.PlinkExe, args.GenotypeFile, i, args.chunks, pref)
    fn = '%s.GRM'%(pref)
    ch = Popen(c, shell=True)
    ch.communicate()
    return fn

#Create a genetic relationship matrix

#Get Clusters

#Get LD matrices

#Co-citation/ dot product


if __name__ == '__main__':
    defplink = '/lb/project/gravel/hleap_projects/UKBB/PRSice_v1.25/plink_1.9_l'
    defplink += 'inux_160914'
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-G', '--Guillimin', help='If ran in Guillimin', 
                        default=False)
    parser.add_argument('-n', '--PlinkExe', help='Path and executable file of \
    plink', default= defplink)    
    parser.add_argument('-g', '--GenotypeFile', help='Path and prefix of the \
    genotype files (bed, bim, fam)', required=True)
    parser.add_argument('-c', '--chunks', help='Number of chunks of the GRM to \
    be used', type=int, default=10)    
    
    args = parser.parse_args()

    