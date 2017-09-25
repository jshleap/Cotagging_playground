'''
compute just the plink part of plot covariances in abacus
'''
from joblib import Parallel, delayed
from subprocess import Popen
import pandas as pd
import numpy as np
import argparse


def snpPRS(plinkexe, bedfile1, snpallelebetatuple, phenofn, freqfile, labels):
    """
    Compute the B_i*g_i from a single SNP
    """
    snp = snpallelebetatuple[0]
    # write score to file
    with open('temp%s.score'%(snp), 'w') as t:
        t.write(' '.join(snpallelebetatuple))
    line = '%s --bfile %s -score temp%s.score sum --pheno %s --allow-no-sex '
    line += '--keep-allele-order --out temp%s'
    line = line % (plinkexe, bedfile1, snp, phenofn, snp)
    # Execute plink
    exe = Popen(line, shell=True)
    exe.communicate()
    # read the profile and cleanup
    profile = pd.read_table('temp%s.profile' % snp, delim_whitespace=True)
    rm = Popen('rm temp%s.*' % snp, shell=True)
    rm.communicate()
    #cov = np.diag(profile.loc[:,['PHENO','SCORESUM']].cov(),k=1)[0]
    cov = np.diag(profile.loc[:,['PHENO','SCORE']].cov(),k=1)[0]
    covname = '$Cov(\\beta_{i}^{%s}g_{i}^{%s}, Y)$' % tuple(labels)
    return pd.DataFrame([{'SNP': snp, covname: cov}])

#----------------------------------------------------------------------
def GWASnCOV(freqfn, gwasfn):
    """
    Read the GWAS file (fn; plink format), execute readFreq and compute
    """
    freq = readFreq(freqfn)
    gwas = pd.read_table(gwasfn, delim_whitespace=True)
    gwas['Cov'] = gwas.BETA * freq.Var
    gwas['-log(P)'] = -np.log(gwas.P)
    return gwas.dropna()
#----------------------------------------------------------------------
def readFreq(fn):
    """
    read the plink frequency file (with --freq or --freqx) and compute the 
    variant variance
    """
    freq = pd.read_table(fn, delim_whitespace=True)
    freq['Var'] = freq.MAF * (1 - freq.MAF)
    return freq

#----------------------------------------------------------------------
def main(args):
    """
    Executes the code
    args.bedfile1 = bedfile of evaluating population (a.k.a. AFR)
    args.pheno1fn = phenotype file of evaluating population (a.k.a. AFR)
    args.labels = labels of both populations being analized. Index 0 is the pop
                 being analyzed (a.k.a. AFR); Index 1 is the reference pop
    args.freq1file = frequency file of the evaluating population (a.k.a. AFR)
    
    """
    # get causal snps
    causals = pd.read_table(args.causalfn, delim_whitespace=True, header=None, 
                            names=['SNP', 'A1', 'True_beta'])    
    # Get GWAS and compute the covariance
    #gwas1 = GWASnCOV(args.freq, args.gwasfn1)
    gwas2 = GWASnCOV(args.freq1file, args.gwasfn2)
    cotags = pd.read_table(args.cotagfn, sep='\t')
    #cotags = cotagDF.nlargest(top, 'Cotagging')
    snpPRSes = pd.concat(Parallel(n_jobs=args.cpus)(delayed(snpPRS)(
        args.plinkexe, args.bedfile1, (i.SNP, i.A1, str(i.BETA)), args.pheno1fn,
        args.freq1file, args.labels) for i in gwas2.itertuples()))     
    snpPRSes.to_csv('%s_SNPwiseregr.tsv' % args.prefix, sep='\t', index=False)

if __name__ == '__main__':   
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)    
    parser.add_argument('-f', '--freq1file', help='Filename of the frequencies', 
                        required=True)  
    parser.add_argument('-i', '--gwasfn2', help=('Filename of the GWAS '
                                                 'reference population'),
                        required=True) 
    parser.add_argument('-d', '--cotagfn', help=('Filename tsv with cotag'
                                                 ' results'), required=True)  
    parser.add_argument('-l', '--labels', help=('name of the populations being'
                                                ' in order of pop1 pop2'), 
                                          action='append')     
    parser.add_argument('-C', '--causalfn', help='Filename of causal/score', 
                        required=True)     
    parser.add_argument('-n', '--plinkexe', help='path to plink executable', 
                        required=True)    
    parser.add_argument('-b', '--bedfile1', help=('Filename of bedfile for '
                                                  'population 1'), required=True
                                            )
    parser.add_argument('-P', '--pheno1fn', help=('Filename of phenotype file '
                                                  'for pop 1'), required=True)
    parser.add_argument('-u', '--cpus', help='Number of processors', default=-1,
                        type=int)    
    args = parser.parse_args()
    main(args)       