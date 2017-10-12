#!/usr/bin/env python
#coding:utf-8
"""
  Author: Jose Sergio Hleap  --<>
  Purpose: Optimize the mixing of cotagging and P + T and chose the snps based 
  pn the optimized ranking
  Created: 10/02/17
"""
import os
import shutil
import tarfile
import argparse
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
from glob import glob
from tqdm import tqdm
from utilities4cotagging import *
from scipy.stats import linregress
from matplotlib import pyplot as plt
from joblib import delayed, Parallel
plt.style.use('ggplot')

#---------------------------------------------------------------------------
def read_n_sort_clumped(resultsfn, allsnps):
    """
    Read the <target>.results file, get the best clumping in the P+T and do the
    "sorting"
    
    :param str resultsfn: filename of the result file of a P+T.py run
    :param :class pd.Series allsnps: Series with allowed set of snps
    """
    # Get the path to the results
    path = os.path.split(resultsfn)[0]
    # Read the Clumped results file
    res = pd.read_table(resultsfn, sep='\t')
    # Set the path and filename of the best clumped file
    clumpfn = '%s/%s.clumped' % (path if path != '' else '.', 
                                 res.nlargest(1, 'R2').File.iloc[0])
    # Return the dataftame of the parse of the clumpled file
    return parse_sort_clump(clumpfn, allsnps)

#---------------------------------------------------------------------------
def read_scored_qr(profilefn, phenofile, alpha, nsnps):
    """
    Read the profile file a.k.a. PRS file or scoresum
    
    :param str profilefn: Filename of profile being read
    :param str phenofile: Filename of phenotype
    :param int nsnps: Number of snps that produced the profile
    """
    # Read the profile
    sc = pd.read_table(profilefn, delim_whitespace=True)
    # Read the phenotype file
    pheno = pd.read_table(phenofile, delim_whitespace=True, header=None, names=[
    'FID', 'IID', 'pheno'])
    # Merge the two dataframes
    sc = sc.merge(pheno, on=['FID', 'IID'])
    # Compute the linear regression between the score and the phenotype
    lr = linregress(sc.pheno, sc.SCORE)
    # Return results in form of dictionary
    dic = {'File':profilefn, 'alpha':alpha, 'R2':lr.rvalue**2, 'SNP kept':nsnps}
    return dic

#----------------------------------------------------------------------
def strategy_sum(x, y, alpha):
    """
    strategy for new rank
    
    :param :class pd.Series x: Series with the first range to be combined
    :param :class pd.Series y: Series with the second range to be combined
    :param float alpha: Float with the weight to be combined by
    """
    return (alpha * x) + ((1-alpha) * y)
    
    
#---------------------------------------------------------------------------
def single_alpha_qr(prefix, alpha, merge, plinkexe, bfile, sumstats, 
                    qrange, phenofile, frac_snps, qr, tar, maxmem=1700, 
                    threads=8, strategy='sum'):
    """
    Single execution of the alpha loop for paralellization
    
    :param str prefix: Prefix for outputs
    :param float alpha: proportion of cotagging included
    :param :class pd.DataFrame: Dataframe with the merge cotagging and P + T
    :param str plinkexe: Path and executable of plink
    :param str bfile: Prefix of plink-bed fileset
    :param str sumstats: File with the summary statistics in plink format 
    :param str qrange: File with the ranges to be passed to the --q-score-range
    :param str phenofile: Filename with the phenotype
    :param float frac_snps: Numer of SNPs in one percent
    :param :class pd.DataFrame qr: DataFrame with the qranges
    :param str tar: label of target population
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    """
    # Define qifile name
    qfile = '%s.qfile' 
    # Define output prefix including the alpha
    ou = '%s_%.2f' % (prefix, alpha)
    qfile = qfile % ou
    # Get the apropriate rankings
    cot = merge.Index_Cotag
    ppt = merge.loc[:, 'Index_%sPpT' % tar]
    # Compute the new ranking with the given strategy
    strategy_fnc = globals()['strategy_%s' % strategy]
    merge['New_rank'] = strategy_fnc(cot, ppt, alpha)
    new_rank = merge.sort_values('New_rank')
    new_rank['New_rank'] = new_rank.reset_index(drop=True).index.tolist()         
    # Wirte results of the new ranking to file
    new_rank.loc[:,['SNP', 'New_rank']].to_csv(qfile, sep=' ', header=False,
                                               index=False)    
    # Score files with the new ranking
    score = ('%s --bfile %s --score %s 2 4 7 header --q-score-range %s '
             '%s --allow-no-sex --keep-allele-order --pheno %s --out %s '
             '--memory %d --threads %d')
    score = score % (plinkexe, bfile, sumstats, qrange, qfile, phenofile, ou,
                     maxmem, threads)
    o,e = executeLine(score) 
    # Get the results in dataframe
    profs_written = read_log(ou)
    df  = pd.DataFrame([read_scored_qr('%s.%.2f.profile' % (ou, float(x.label)),
                                       phenofile, alpha, x.Max) if (
        '%s.%.2f.profile' % (ou, float(x.label)) in profs_written) else {}
                        for x in qr.itertuples()]).dropna()
    # Cleanup
    with tarfile.open('Profiles_%s_%.2f.tar.gz' % (prefix, alpha), mode='w:gz'
                      ) as t:
        for fn in glob('%s*.profile' % ou):
            if os.path.isfile(fn):
                try:
                # it has a weird behaviour
                    os.remove(fn)
                    t.add(fn)
                except:
                    pass
    # Return the results dataframe
    return df

#---------------------------------------------------------------------------
def rank_qr(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile, alphastep,
            plinkexe, tar, prunestep=1, qrangefn=None, maxmem=1700,
            threads=1, strategy='sum', every=False):
    """
    Estimate the new rank based on the combination of the cotagging and P+T rank
    
    :param str prefix: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param :class pd.Dataframe sorted_cotag:Filename with results in the sorting
    :param list clumpe: List of tuples with the P+T results
    :param str sumstats: File with the summary statistics in plink format 
    :param str phenofile: Filename with the phenotype
    :param float alphastep: Step of the alpha to be explored
    :param str plinkexe: Path and executable of plink
    :param str tar: Label of target population
    :param int prunestep: Step of the prunning
    :param str qrangefn: Filename of a previously made qrange
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    :param bool every: test one snp at a time
    """
    # Generate the alpha-space
    space = np.concatenate((np.array([0, 0.05]), np.arange(0.1, 1 + alphastep,
                                                        alphastep)))
    # Get the number of valid snps fdrom the sorted cotag dataframe
    nsnps = sorted_cotag.shape[0]
    frac_snps = nsnps / 100
    # Generate the qrange file?
    order = ['label', 'Min', 'Max']
    if qrangefn is None:
        # Define the number of snps per percentage point and generate the range
        percentages = set_first_step(nsnps, prunestep, every=every)
        snps = np.around((percentages * nsnps) / 100).astype(int)
        try:
            # Check if there are repeats in ths set of SNPS
            assert sorted(snps) == sorted(set(snps))
        except AssertionError:
            snps = ((percentages * allsnp) / 100).astype(int)
            assert sorted(snps) == sorted(set(snps))
        labels = ['%.2f' % x for x in percentages]
        # Generate the qrange file
        qrange = '%s.qrange' % prefix
        qr = pd.DataFrame({'label':labels, 'Min':np.zeros(len(percentages)),
                           'Max':snps}).loc[:, order]        
        qr.to_csv(qrange, header=False, index=False, sep =' ')   
    else:
        # Read the previously generate qrange
        qrange = qrangefn
        qr = pd.read_csv(qrange, sep=' ', header=None, names=order)
    # Deal with the P+T results
    if isinstance(clumpe, list):
        premerge = clumpe[0][0].merge(clumpe[1][0], on='SNP', suffixes=[
            '_%sPpT' % clumpe[0][1], '_%sPpT' % clumpe[1][1]])
    else:
        premerge = clumpe.rename(columns={'Index':'Index_%sPpT' % tar}).head()
    # Merge the sorted cotag and the P+T
    merge = sorted_cotag.merge(premerge, on='SNP', suffixes=['Cotag', 'PpT'])
    merge = merge.rename(columns={'Index':'Index_Cotag'})
    # Execute the optimization
    df = Parallel(n_jobs=int(threads))(delayed(single_alpha_qr)(
        prefix, alpha, merge, plinkexe, bfile, sumstats, qrange, phenofile,
        frac_snps, qr, tar, maxmem, threads, strategy) for alpha in tqdm(space))    
    # Return the list of dataframes with the optimization results
    return df

#---------------------------------------------------------------------------
def optimize_alpha(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile, 
                   plinkexe, alphastep, tar, prune_step=1, qrangefn=None,
                   maxmem=1700, threads=1, strategy='sum', every=False):
    """
    Do a line search for the best alpha in nrank = alpha*rankP+T + (1-alpha)*cot
    
    :param str prefix: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param :class pd.Dataframe sorted_cotag:Filename with results in the sorting
    :param list clumpe: List of tuples with the P+T results
    :param str sumstats: File with the summary statistics in plink format 
    :param str phenofile: Filename with the phenotype
    :param float alphastep: Step of the alpha to be explored
    :param str plinkexe: Path and executable of plink
    :param str tar: Label of target population
    :param int prune_step: Step of the prunning
    :param str qrangefn: Filename of a previously made qrange
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    :param bool every: test one snp at a time
    """
    # Set output name
    outfn = '%s_optimized.tsv' % prefix
    # Execute the ranking
    if not os.path.isfile(outfn):
        d = rank_qr(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile, 
                    alphastep, plinkexe, tar, prune_step, qrangefn, maxmem, 
                    threads, strategy, every)
        df = pd.concat(d)
        df.to_csv(outfn, sep='\t', index=False)
    else:
        df = pd.read_table(outfn, sep='\t')
    # Plor the optimazation
    piv = df.loc[:,['SNP kept','alpha', 'R2']]
    piv = piv.pivot(index='SNP kept',columns='alpha', values='R2').sort_index()
    piv.plot()
    plt.savefig('%s_alphas.pdf'%(prefix))   
    # Returned the sorted resulr dataframe
    return df.sort_values('R2', ascending=False).reset_index(drop=True)

#----------------------------------------------------------------------
def read_n_sort_cotag(prefix, cotagfn, freq):
    """
    Smart sort the cotag file

    :param str prefix: Prefix for outputs
    :param str cotagfn: Filename tsv with cotag results
    :param :class pd.DataFrame freq: Dataframe with the plink formatted freqs.
    """
    # Read cotag
    cotags = pd.read_table(cotagfn, sep='\t')
    # Smart sort it
    # Check for previous sort
    prev = glob('*Cotagging.pickle')
    if prev != []:
        shutil.copy(prev[0], '%s_Cotagging.pickle' % prefix)
    df, _ = smartcotagsort(prefix, cotags[cotags.SNP.isin(freq.SNP)])
    # Returned the sorted dataframe
    return df.reset_index()


#----------------------------------------------------------------------
def prankcster(prefix, targetbed, referencebed, cotagfn, ppt_results_tar,
               ppt_results_ref, sumstats, pheno, plinkexe, alpha_step, 
               labels, prune_step, sortresults, freq_threshold=0.1, h2=None,
               qrangefn=None, maxmem=1700, threads=1, strategy='sum', 
               every=False):
    """
    Execute the code and plot the comparison
    
    :param str prefix: Prefix for outputs
    :param str targetbed: Prefix of plink-bed fileset of the target population
    :param str referencebed: Prefix of plink-bed fileset of the ref population
    :param str cotagfn: Filename tsv with cotag results
    :param str ppt_results_tar: path to P+T relsults in the target pop
    :param str ppt_results_ref: path to P+T relsults in the reference pop
    :param str sumstats: File with the summary statistics in plink format 
    :param str pheno: Filename with the phenotype
    :param float alpha_step: Step of the alpha to be explored
    :param str plinkexe: Path and executable of plink
    :param list labels: List with labels of reference and target populations
    :param int prune_step: Step of the prunning
    :param str sortresults: Filename with results in the sorting inlcuding path
    :param float freq_threshold: Threshold for frequency filtering
    :param float h2: Heritability of the phenotype
    :param str qrangefn: Filename of a previously made qrange
    :param int maxmem: Maximum allowed memory
    :param int threads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    :param bool every: test one snp at a time
    """
    # Unpack population labels
    ref, tar = labels
    # Get frequencies of both populations and merge them
    f1 = read_freq(referencebed, plinkexe, freq_threshold=freq_threshold)
    f2 = read_freq(targetbed, plinkexe, freq_threshold=freq_threshold)
    frqs = f1.merge(f2, on=['CHR', 'SNP'], suffixes=['_%s' % ref,
                                                         '_%s' % tar])
    # Read the cotag scores
    if os.path.isfile('%s.sorted_cotag' % prefix):
        sorted_cotag = pd.read_table('%s.sorted_cotag' % prefix, sep='\t')
    else:
        sorted_cotag = read_n_sort_cotag(prefix, cotagfn, f2)
        sorted_cotag.to_csv('%s.sorted_cotag' % prefix, sep='\t', 
                            index=False)    
    # Read and sort the P + T results
    clumpetar = read_n_sort_clumped(ppt_results_tar, frqs.SNP)
    #clumpetar = clumpetar[clumpetar.SNP.isin(frqs.SNP)]
    clumperef = read_n_sort_clumped(ppt_results_ref, frqs.SNP)
    #clumperef = clumperef[clumperef.SNP.isin(frqs.SNP)]
    clumpe = [(clumpetar, tar), (clumperef, ref)]
    # Read summary statisctics
    # if isinstance(sumstats, str):
    #     ss = pd.read_table(sumstats, delim_whitespace=True)
    # else:
    #     assert isinstance(sumstats, pd.DataFrame)
    #     ss = sumstats
    # Optimize the alphas
    df = optimize_alpha(prefix, targetbed, sorted_cotag, clumpe, sumstats,  pheno,
                        plinkexe, alpha_step, tar, prune_step, qrangefn, maxmem,
                        threads, strategy, every)
    # Get the best alpha of the optimization
    grouped = df.groupby('alpha')
    best = grouped.get_group(df.loc[0,'alpha'])
    if isinstance(sortresults, str):
        prevs = pd.read_table(sortresults, sep='\t')
    else:
        prevs = sortresults
    merged = best.merge(prevs, on='SNP kept')
    f, ax = plt.subplots()
    # plot cotagging
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_cotag', label='Cotagging', 
                        c='r', s=2, alpha=0.5, ax=ax)
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_clum%s' % tar, ax=ax,
                        label='Clump Sort %s' % tar, c='k', s=2, alpha=0.5)    
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_clum%s' % ref, ax=ax, s=2,
                        label='Clump Sort %s' % ref, c='0.5', marker='*', 
                        alpha=0.5)    
    merged.plot.scatter(x='SNP kept', y='R2', label='Hybrid', c='g', s=2, 
                        alpha=0.5, ax=ax) 
    if h2 is not None:
        ax.axhline(h2, c='0.5', ls='--')
    plt.savefig('%s_compare.pdf' % prefix)
    # Write resulted marged dataframe to file
    merged.to_csv('%s.tsv' % prefix, sep='\t', index=False)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-b', '--reference', help=('prefix of the bed fileset '
                                                   'in reference'), 
                                             required=True)    
    parser.add_argument('-c', '--target', help=('prefix of the bed fileset in '
                                                'target'), required=True)
    parser.add_argument('-L', '--labels', help=('Space separated string with '
                                                'labels of reference and target'
                                                ' populations'), nargs=2)
    parser.add_argument('-T', '--target_ppt', help=('Filename of the results of'
                                                    ' a PPT run'), default=None)
    parser.add_argument('-r', '--ref_ppt', help=('Filename with results for the'
                                                 ' P+Toptimization in the refer'
                                                 'ence population'), 
                                           default=None)      
    parser.add_argument('-R', '--sortresults', help=('Filename with results in '
                                                     'the sorting inlcuding pat'
                                                     'h'), 
                                               required=True)      
    parser.add_argument('-d', '--cotagfn', help=('Filename tsv with cotag '
                                                 'results'), required=True) 
    parser.add_argument('-s', '--sumstats', help=('Filename of the summary stat'
                                                  'istics in plink format'), 
                                            required=True)    
    parser.add_argument('-f', '--pheno', help=('filename of the true phenotype '
                                               'of the target population'), 
                                         required=True)      
    parser.add_argument('-S', '--alpha_step', help=('Step for the granularity '
                                                    'of the grid search. Defau'
                                                    'lt: .1'), default=0.1,
                                              type=float) 
    parser.add_argument('-E', '--prune_step', help=('Percentage of snps to be '
                                                    'tested at each step is 1'
                                                    ), default=1, type=float)      
    parser.add_argument('-P', '--plinkexe')
    parser.add_argument('-y', '--every', action='store_true', default=False)
    parser.add_argument('-t', '--threads', default=-1, action='store', type=int) 
    parser.add_argument('-H', '--h2', default=0.66, type=float, 
                        help=('Heritability of the simulated phenotype'))     
    parser.add_argument('-M', '--maxmem', default=1700, type=int) 
    parser.add_argument('-F', '--freq_threshold', default=0.1, type=float) 
    parser.add_argument('-Q', '--qrangefn', default=None, help=(
        'Specific pre-made qrange file')) 
    parser.add_argument('-g', '--strategy', default='sum', help=(
        'Strategy to produce the hybrid measure. Currently available is '
        'weighted sum (sum)'))     
    args = parser.parse_args()
    prankcster(args.prefix, args.target, args.reference, args.cotagfn, 
               args.target_ppt, args.ref_ppt, args.sumstats, args.pheno,
               args.plinkexe, args.alpha_step, args.labels, args.prune_step, 
               args.sortresults, h2=args.h2, freq_threshold=args.freq_threshold,
               qrangefn=args.qrangefn, maxmem=args.maxmem, threads=args.threads,
               strategy=args.strategy, every=args.every)   