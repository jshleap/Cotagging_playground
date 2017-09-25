'''
LD estimation
'''
__author__ = 'jshleap'

import argparse, os
from subprocess import Popen
import numpy as np
from time import sleep
from glob import glob as G

## Constants
qsub='''#!/bin/bash
#PBS -l nodes=1:ppn=16,walltime=%d:00:00
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
def Parallel_LD(args,qsub=qsub):
    """
    Create the triangular genetic relationship matrix using plink in a HPC 
    cluster (Guillimin)
    """
    pref = '%s_%s'%(args.prefix,args.LDwindow)
    comm = '%s --bfile %s --ld-window-cm %d --%s d with-freqs --parallel %d %d '
    comm += '--keep-allele-order --out %s'
    cat = ['cat']
    shs = []
    for i in xrange(1, args.chunks+1):
        tname = '%s_%d'%(pref,i)
        c = comm%(args.PlinkExe, args.GenotypeFile, args.LDwindow, args.typeLD, 
                  i, args.chunks, pref)
        with open('%s.sh'%(tname),'w') as qs:
            qs.write(qsub%(args.walltime, tname, tname, tname, c))
            shs.append('%s.sh'%(tname))
        ch = Popen('qsub %s.sh'%(tname), shell=True)
        ch.communicate()
        cat.append('%s.ld.%d'%(pref, i))
    ## join the results
    #count=0
    while not os.path.isfile('%s.ld.%d'%(pref, args.chunks)) and \
          (len(G('%s.ld.*'%(pref))) != args.chunks):
        sleep(600)
        #count += 1
    fn = '%s.%s.fullLD'%(pref, args.typeLD)
    catp = Popen('%s > %s'%(' '.join(cat), fn), shell=True)
    catp.communicate()
    map(os.remove, cat[1:])
    map(os.remove, shs)
    return fn

if __name__ == '__main__':
    defplink = '/lb/project/gravel/hleap_projects/UKBB/PRSice_v1.25/plink_1.9_l'
    defplink += 'inux_160914'
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-n', '--PlinkExe', help='Path and executable file of \
    plink', default= defplink)    
    parser.add_argument('-g', '--GenotypeFile', help='Path and prefix of the \
    genotype files (bed, bim, fam)', required=True)
    parser.add_argument('-c', '--chunks', help='Number of chunks of the LD to \
    be used', type=int, default=10)    
    parser.add_argument('-w', '--walltime', help='Walltime', type=int, 
                        default=30)
    parser.add_argument('-r', '--typeLD', help='type of LD to use (r2, r)', 
                        type=str, default='r')    
    parser.add_argument('-l', '--LDwindow', help='Window of LD (default 5cm)',
                        type=int, default=5)
    
    args = parser.parse_args()
    Parallel_LD(args)
