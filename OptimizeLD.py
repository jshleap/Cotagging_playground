"""
Optimize the LD r2 threshold for clumping using 
"""
__author__ = 'jshleap'

import argparse
import numpy as np
from PRS import PRSiceCommand
from PRS import main as PRS


def main(args):
    rang = np.arange(0.1,0.6,0.1)
    
    for i in rang:
        name = '%s_%.1f'%(args.prefix,i)
        args.LDthreshold = i
        args.prefix = name
        PRS(args,PRSiceCommand)
    

if __name__ == '__main__':
    ## Define the path of the default plink and PRSice which is in abacus 
    ##(McGill Genome Center cluster)
    defplink = '/lb/project/gravel/hleap_projects/UKBB/PRSice_v1.25/plink_1.9_l'
    defplink += 'inux_160914'
    defprsice = '/lb/project/gravel/hleap_projects/UKBB/PRSice_v1.25/PRSice_v1.'
    defprsice += '25.R'
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-s', '--SummaryStats', help='Filename of the summary \
    statistics', required=True)    
    parser.add_argument('-g', '--GenotypeFile', help='Path and prefix of the \
    genotype files (bed, bim, fam) (A.K.A. target)', required=True)
    parser.add_argument('-P', '--PhenotypeFile', help='Path and filename of the\
    phenotype file', required=True)
    parser.add_argument('-k', '--KeepFile', help='Path and filename of the file\
    with the individuals to keep', required=True)    
    parser.add_argument('-n', '--PlinkExe', help='Path and executable file of \
    plink', default= defplink)
    parser.add_argument('-r', '--PRSiceExe', help='Path and executable file of \
    PRSice', default= defprsice)
    
    args = parser.parse_args()
    main(args) 