#!/usr/bin/env python
#coding:utf-8
"""
  Author:  ppc_pipeline --<>
  Purpose: Run simulation, P+T, null model and P+C
  Created: 10/04/17
"""
import os
import argparse
import numpy as np
import pandas as pd
from ppt import pplust
from ranumo import ranumo
from plinkGWAS import plink_gwas
from prankcster import prankcster
from qtraitsimulation_old import qtraits_simulation

def execute(args):
    """
    Execute pipeline
    """
    # TODO: expand the Tar files if relaunched
    rstart, rstop, rstep = args.r2range
    Ps = [float('%.1g' % float(x)) for x in args.pvals.split(',')]
    cwd = os.getcwd()
    ref, tar = args.labels
    if not os.path.isdir(ref):
        os.mkdir(ref)
    os.chdir(ref)
    prs_ref, validsnpfile = qtraits_simulation(ref, args.refb, args.h2, 
                                               args.ncausal, args.plinkexe,
                                               maxmem=args.maxmem, 
                                               threads=args.threads)
    gwas_ref, sumstats, train_eur, test_eur = plink_gwas(
        args.plinkexe, args.refb, args.prefix, '%s.pheno' % ref, nosex=True, 
        threads=args.threads, maxmem=args.maxmem, validate=5,
        validsnpsfile=validsnpfile, plot=True)
    r2range =  [x if x <= 0.99 else 0.99 for x in sorted(np.arange(
        rstart, rstop + rstep, rstep), reverse=True)]
    phenoref = os.path.join(os.getcwd(), '%s.pheno' % ref)
    resE, pptrfn = pplust(ref, test_eur, sumstats, r2range, Ps, args.LDwindow, 
                  phenoref, args.plinkexe, plot=True, 
                  clean=True)
    causaleff = os.path.join(os.getcwd(), '%s.full' % ref)
    assert os.path.isfile(causaleff)
    os.chdir(cwd)
    if not os.path.isdir(tar):
        os.mkdir(tar)
    os.chdir(tar)
    tarpref = '%s-%s' % (tar, ref)
    prs_tar, validsnpfile = qtraits_simulation(tarpref, args.tarb, 
                                               args.h2, args.ncausal, 
                                               args.plinkexe, 
                                               maxmem=args.maxmem, 
                                               threads=args.threads,
                                               causaleff=causaleff)
    phenotar = os.path.join(cwd, tar, '%s.pheno' % tarpref)
    resT, pptfn = pplust(tarpref, args.tarb, sumstats, r2range,Ps,args.LDwindow, 
                  phenotar, args.plinkexe, plot=True, 
                  clean=True)    
    os.chdir(cwd)
    if not os.path.isdir('Null'):
        os.mkdir('Null')
    os.chdir('Null')
    sortresults, qrfn = ranumo(args.prefix, args.tarb, args.refb, sumstats, 
                         args.cotagfn, args.plinkexe, args.labels, phenotar, 
                         phenoref, pptR=resE, pptT=resT, hline=args.h2,
                         check_freqs=args.freq_threshold, step=args.prune_step,
                         quality='pdf')
    os.chdir(cwd)
    prankcster(args.prefix, args.tarb, args.refb, args.cotagfn, pptfn, pptrfn, 
               sumstats, phenotar, args.plinkexe, args.alpha_step, args.labels, 
               args.prune_step, sortresults, freq_threshold=args.freq_threshold,
               h2=args.h2, qrangefn=qrfn, maxmem=args.maxmem, 
               threads=args.threads)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-m', '--ncausal', type=int, default=100)
    parser.add_argument('-b', '--refb', help=('prefix of the bed fileset in '
                                              'reference'), 
                                             required=True)    
    parser.add_argument('-c', '--tarb', help=('prefix of the bed fileset in '
                                                'target'), required=True)
    parser.add_argument('-L', '--labels', help=('Space separated string with '
                                                'labels of reference and target '
                                                'populations'), nargs=2)
    parser.add_argument('-d', '--cotagfn', help=('Filename tsv with cotag '
                                                 'results'), required=True) 
    parser.add_argument('-S', '--alpha_step', help=('Step for the granularity of'
                                                    ' the grid search. Default: '
                                                    '.1'), default=0.1, 
                                              type=float) 
    parser.add_argument('-E', '--prune_step', help=('Percentage of snps to be '
                                                    'tested at each step is 1'
                                                    ), default=1, type=float) 
    parser.add_argument('-v', '--pvals', default='1.0,0.5,0.1,10E-3,10E-7')
    parser.add_argument('-r', '--r2range', help=('Space separated rstart, rstop'
                                                 ', rstep for LD clumping'), 
                                           nargs=3, type=float)
    parser.add_argument('-P', '--plinkexe')
    parser.add_argument('-t', '--threads', default=-1, type=int) 
    parser.add_argument('-H', '--h2', default=0.66, type=float, 
                        help=('Heritability of the simulated phenotype'))     
    parser.add_argument('-M', '--maxmem', default=1700, type=int) 
    parser.add_argument('-F', '--freq_threshold', default=0.1, type=float) 
    parser.add_argument('-Q', '--qrangefn', default=None, help=(
        'Specific pre-made qrange file'))    
    parser.add_argument('-l', '--LDwindow', help='Physical distance threshold '+
                        'for clumping in kb (250kb by default)', type=int, 
                        default=250)    
    args = parser.parse_args()
    args.refb = os.path.abspath(args.refb)
    args.tarb = os.path.abspath(args.tarb)
    args.plinkexe = os.path.abspath(args.plinkexe)
    args.cotagfn = os.path.abspath(args.cotagfn)
    execute(args)