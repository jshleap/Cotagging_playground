#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Jose Sergio Hleap --<>
  Purpose: Simulate quantitative phenotypes given 
  Created: 09/30/17
"""
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from utilities4cotagging import read_freq, executeLine

def TruePRS(outprefix, bfile, h2, ncausal, plinkexe, snps=None, frq=None, 
            causaleff=None, bfile2=None, freqthreshold=0.1, maxmem=1700, 
            threads=1, seed=None):
    """
    Generate TRUE causal effects and PRS. Also, subset the SNPs based on 
    frequency
    
    :param str outprefix: Prefix for outputs
    :param str bfile: prefix of the plink bedfileset
    :param float h2: Desired heritability
    :param int ncausal: Number of causal variants to simulate
    :param str plinkexe: path to plink executable
    :param :class pd.Series snps: Series with the names of causal snps
    :param :class pd.DataFrame frq: DataFrame with the MAF frequencies
    :param :class pd.DataFrame causaleff: DataFrame with the true causal effects
    :param str bfile2: prefix of the plink bedfileset on a second population
    :param float freqthreshold: Lower threshold to filter MAF by
    """
    # set the seed
    seed = np.random.randint(1e4) if seed is None else seed
    print('using seed %d' % seed)
    np.random.seed(seed=seed)
    # Get the per snp heritability
    h2_snp = h2/ncausal
    if not os.path.isfile('%s.full'%(outprefix)):
        # Read bim file
        if frq is None:
            frq = read_freq(bfile, plinkexe)
            if bfile2 is not None:
                frq = frq.merge(read_freq(bfile2, plinkexe), on=['CHR', 'SNP'])
        allsnps = frq.shape[0]
        print ('Total Number of variants available: %d' % allsnps)
        totalsnps = '%s.totalsnps' % outprefix
        frq.to_csv(totalsnps, sep=' ', header=False, index=False)
        # Get causal mutation indices randomly distributed
        if ncausal > allsnps:
            print('More causals than available snps. Setting it to %d' %allsnps)
            ncausal = allsnps        
        if causaleff is not None:
            causals = frq[frq.SNP.isin(causaleff.SNP)]
            causals = causals.merge(causaleff, on='SNP')
        elif snps is None:
            causals = frq.sample(ncausal, replace=False, random_state=seed)
        else:
            causals = frq[frq.SNP.isin(snps)]
        # If causal effects are provided use them, otherwise get them
        if causaleff is None:
            causals['eff'] = np.random.normal(loc=0, scale=np.sqrt(h2_snp), 
                                              size=ncausal)
        # write snps and effect to score file
        causals.loc[:, 'norm'] = np.sqrt((2 * causals.MAF_x) * (
            1 - causals.MAF_x))
        causals.loc[:, 'beta'] = causals.loc[:, 'eff']/causals.norm      
        scfile = causals.sort_index()
        # Write score to file
        scfile.loc[:, ['SNP', 'A1_x', 'beta']].to_csv('%s.score'%(outprefix), 
                                                    sep=' ', header=False, 
                                                    index=False)
        scfile.to_csv('%s.full'%(outprefix), sep=' ', index=False)        
    else:
        scfile = pd.read_table('%s.full'%(outprefix), delim_whitespace=True, 
                               header=None, names=['SNP', 'Allele', 'beta'])
    # Score using plink
    score = ('%s --bfile %s --score %s.score sum --allow-no-sex --extract %s '
             '--keep-allele-order --out %s --memory %d --threads '
             '%d')
    if not os.path.isfile('%s.profile'%(outprefix)):
        executeLine(score%(plinkexe, bfile, outprefix, totalsnps, outprefix,
                           maxmem, threads))
    # Read scored and rename SCORE column,
    score = pd.read_table('%s.profile'%(outprefix), delim_whitespace=True)
    score = score.rename(columns={'SCORESUM':'gen_eff'})
    return score, scfile, totalsnps

#----------------------------------------------------------------------
def create_pheno(prefix, h2, prs_true, noenv=False):
    """
    Generate phenotypes and real betas.
    
    :param str prefix: Prefix for outputs
    :param float h2: Desired heritability
    :param :class pd.DataFrame prs_true: First Dataframe outputted by TruePRS
    :param bool noenv: whether or not environmental effect should be added
    """
    # Deal with no environment
    if h2 == 1:
        noenv = True
    nind = prs_true.shape[0]
    if noenv:
        env_effect = np.zeros(nind)
    else:
        env_effect = np.random.normal(loc=0,scale=np.sqrt(1-h2), size=nind)
    # Include environmental effects into the dataframe
    prs_true['env_eff'] = env_effect
    # Generate the phenotype from the model Phenotype = genetics + environment
    prs_true['PHENO'] = prs_true.gen_eff + prs_true.env_eff
    # Write it to file
    prs_true.to_csv('%s.prs_pheno.gz'%(prefix), sep='\t', compression='gzip',
                    index=False)
    prs_true.loc[: , ['FID', 'IID', 'PHENO']].to_csv('%s.pheno'%(prefix), 
                                                         sep=' ', header=False, 
                                                         index=False)    
    # return the dataframe
    return prs_true

def plot_pheno(prefix, prs_true, quality='pdf'):
    """
    Plot phenotype histogram
    
    :param :class pd.DataFrame prs_true: Output of the create_pheno function 
    with true PRSs
    :param str prefix: prefix for outputs
    :param str quality: quality of the plot (e.g. pdf, png, jpg)
    """
    prs_true.loc[:, ['PHENO', 'gen_eff', 'env_eff']].hist(alpha=0.5)
    plt.savefig('%s.%s' % (prefix, quality))
    
def qtraits_simulation(outprefix, bfile, h2, ncausal, plinkexe, snps=None, 
                       frq=None, causaleff=None, noenv=False, plothist=False,
                       freqthreshold=0.1, bfile2=None, quality='png', 
                       maxmem=1700, threads=1, seed=None):
    """
    Execute the code. This code should output a score file, a pheno file, and 
    intermediate files with the dataframes produced
    
    :param str outprefix: Prefix for outputs
    :param str bfile: prefix of the plink bedfileset
    :param float h2: Desired heritability
    :param int ncausal: Number of causal variants to simulate
    :param str plinkexe: path to plink executable
    :param :class pd.Series snps: Series with the names of causal snps
    :param :class pd.DataFrame frq: DataFrame with the MAF frequencies
    :param :class pd.DataFrame causaleff: DataFrame with the true causal effects
    :param bool noenv: whether or not environmental effect should be added
    :param float freqthreshold: Lower threshold to filter MAF by
    :param str bfile2: prefix of the plink bedfileset on a second population
    :param str quality: quality of the plot (e.g. pdf, png, jpg)
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param int seed: random seed to use in samplinh
    """
    print('Performing qtraits_simulation on %s' % outprefix)
    if causaleff is not None:
        if isinstance(causaleff, str):
            causaleff = pd.read_table('%s'%(causaleff), delim_whitespace=True)
        causaleff = causaleff.loc[:, ['SNP', 'eff']]
        assert causaleff.shape[0] == ncausal
    if not os.path.isfile('%s.full'%(outprefix)):
        geneff, truebeta , validsnpfile = TruePRS(outprefix, bfile, h2, ncausal, 
                                                  plinkexe, snps=snps, frq=frq, 
                                                  causaleff=causaleff,
                                                  bfile2=bfile2,
                                                  freqthreshold=freqthreshold, 
                                                  seed=seed)
    else:
        truebeta = pd.read_table('%s.full' % outprefix, delim_whitespace=True) 
        validsnpfile = '%s.totalsnps' % outprefix   
        score = pd.read_table('%s.profile' % outprefix, delim_whitespace=True)
        geneff = score.rename(columns={'SCORESUM':'gen_eff'})
    if not os.path.isfile('%s.prs_pheno.gz' % outprefix):
        prs_true = create_pheno(outprefix, h2, geneff, noenv=noenv)
    else:
        prs_true = pd.read_table('%s.prs_pheno.gz' % outprefix, sep='\t')
    if plothist:
        plot_pheno(outprefix, prs_true, quality=quality)
    return prs_true, validsnpfile
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-m', '--ncausal', type=int, default=200)
    parser.add_argument('-b', '--h2', type=float, default=0.66)
    parser.add_argument('-B', '--bfile', default='EUR')
    parser.add_argument('-P', '--plinkexe', default='~/Programs/plink_mac/plink'
                        )
    parser.add_argument('-e', '--noenv', default=False, action='store_true')
    parser.add_argument('-q', '--quality', help=('type of plots (png, pdf)'), 
                        default='png')
    parser.add_argument('--plothist', help=('Plot histogram of phenotype'), 
                        default=False, action='store_true') 
    parser.add_argument('--causal_eff', help=('Provide causal effects file as'
                                              'produced by a previous run of '
                                              'this code with the extension '
                                              'full'), default=None)  
    parser.add_argument('-f', '--freqthreshold', help=('Lower threshold to filt'
                                                       'er MAF by'),
                        default=0.1, type=float)
    parser.add_argument('-2', '--bfile2', help=('prefix of the plink bedfileset'
                                                'o n a second population'))
    
    parser.add_argument('-t', '--threads', default=False, action='store')
    parser.add_argument('-M', '--maxmem', default=False, action='store')  
    parser.add_argument('-s', '--seed', default=None, type=int)  
    
    args = parser.parse_args()
    qtraits_simulation(args.prefix, args.bfile, args.h2, args.ncausal, 
                       args.plinkexe, plothist=args.plothist, 
                       causaleff=args.causal_eff, quality=args.quality, 
                       freqthreshold=args.freqthreshold, bfile2=args.bfile2,
                       maxmem=args.maxmem, threads=args.threads, seed=args.seed)     
    