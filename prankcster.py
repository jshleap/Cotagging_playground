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
from glob import glob
from tqdm import tqdm
from utilities4cotagging import *
from scipy.stats import linregress
from matplotlib import pyplot as plt
from joblib import delayed, Parallel
matplotlib.use('Agg')
plt.style.use('ggplot')

#---------------------------------------------------------------------------
def read_n_sort_clumped(resultsfn, allsnps):
    """
    Read the <target>.results file, get the best clumping in the P+T and do the
    "sorting"
    
    :param str resultsfn: filename of the result file of a P+T.py run
    :param :class pd.Series allsnps: Series with allowed set of snps
    :param :class pf.DataFrame clump: 
    """
    path = os.path.split(resultsfn)[0]
    res = pd.read_table(resultsfn, sep='\t')
    clumpfn = '%s/%s.clumped' % (path if path != '' else '.', 
                                 res.nlargest(1, 'R2').File.iloc[0])
    return parse_sort_clump(clumpfn, allsnps)

#---------------------------------------------------------------------------
def yielder(prefix, bfile, sorted_cotag, clumped,sumstats, phenofile, plinkexe, 
            step):
    space = np.concatenate((np.array([0.05]), np.arange(0.1, 0.9 + step, step)))
    if space[-1] > 1:
        space[-1] = 1
    for i in space:
        d = {'prefix': prefix, 'bfile':bfile, 'sorted_cotag':sorted_cotag, 
             'clumped':clumped, 'sumstats':sumstats, 'phenofile':phenofile, 
             'alpha':i, 'plinkexe':plinkexe}
        yield d

#---------------------------------------------------------------------------
def scoreit(bfile, gwasfn, outpref, phenofile, plinkexe):
    """
    compute the PRS or profile given <score> betas and <bfile> genotype
    """
    score = ('%s --bfile %s --extract %s.extract --score %s 2 4 7 header '
             '--allow-no-sex --keep-allele-order --pheno %s --out %s')
    score = score%(plinkexe, bfile, outpref, gwasfn, phenofile, outpref)
    o,e = executeLine(score)  

#---------------------------------------------------------------------------
def read_scored_qr(profilefn, phenofile, alpha, nsnps):
    """
    Read the profile file a.k.a. PRS file or scoresum
    """
    sc = pd.read_table(profilefn, delim_whitespace=True)
    pheno = pd.read_table(phenofile, delim_whitespace=True, header=None, names=[
    'FID', 'IID', 'pheno'])
    sc = sc.merge(pheno, on=['FID', 'IID'])
    err = sum((sc.pheno - sc.SCORE)**2)
    lr = linregress(sc.pheno, sc.SCORE)
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
                    qrange, phenofile, frac_snps, qr, tar):
    """
    single execution of the alpha loop for paralellization
    """
    qfile = '%s.qfile' 
    ou = '%s_%.2f' % (prefix, alpha)
    qfile = qfile % ou
    cot = merge.Index_Cotag
    ppt = merge.loc[:, 'Index_%sPpT' % tar]
    merge['New_rank'] = strategy_sum(cot, ppt, alpha)
    new_rank = merge.sort_values('New_rank')
    new_rank['New_rank'] = new_rank.reset_index(drop=True).index.tolist()         
    new_rank.loc[:,['SNP', 'New_rank']].to_csv(qfile, sep=' ', header=False,
                                               index=False)    
    score = ('%s --bfile %s --score %s 2 4 7 header --q-score-range %s '
             '%s --allow-no-sex --keep-allele-order --pheno %s --out %s')
    score = score%(plinkexe, bfile, sumstats, qrange, qfile, phenofile, ou)        
    o,e = executeLine(score) 
    df  = pd.DataFrame([read_scored_qr('%s.%s.profile' % (ou, x.label),
                                       phenofile, alpha, x.Max)
                        for x in qr.itertuples()])  
    with tarfile.open('Profiles_%s_%.2f.tar.gz' % (prefix, alpha), mode='w:gz'
                      ) as t:
        for fn in glob('*.profile'):
            if os.path.isfile(fn):
                try:
                # it has a weird behaviour
                    os.remove(fn)
                    t.add(fn)
                except:
                    pass
    return df

#---------------------------------------------------------------------------
def rank_qr(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile, alphastep,
            plinkexe, threads, tar, prunestep=1):
    """
    Estimate the new rank based on the combination of the cotagging and P+T rank
    """
    l=[]
    lappend = l.append
    space = np.concatenate((np.array([0.05]), np.arange(0.1, 0.9 + alphastep, 
                                                        alphastep)))
    qrange = '%s.qrange' % prefix
    nsnps = sorted_cotag.shape[0]
    frac_snps = nsnps/100
    percentages = set_first_step(nsnps, prunestep)
    order = ['label', 'Min', 'Max']
    qr = pd.DataFrame({'label':percentages, 'Min':np.zeros(len(percentages)),
                       'Max':np.around(np.array(percentages, dtype=float) * 
                                       frac_snps, decimals=0).astype(int)}
                      ).loc[:, order]
    qr.to_csv(qrange, header=False, index=False, sep =' ')   
     
    if isinstance(clumpe, list):
        premerge = clumpe[0][0].merge(clumpe[1][0], on='SNP', suffixes=[
            '_%sPpT' % clumpe[0][1], '_%sPpT' % clumpe[1][1]])
    else:
        premerge = clumpe.rename(columns={'Index':'Index_%sPpT' % tar}).head()
    merge = sorted_cotag.merge(premerge, on='SNP', suffixes=['Cotag', 'PpT'])
    merge = merge.rename(columns={'Index':'Index_Cotag'})
    df = Parallel(n_jobs=int(threads))(delayed(single_alpha_qr)(
        prefix, alpha, merge, plinkexe, bfile, sumstats, qrange, phenofile,
        frac_snps, qr, tar) for alpha in tqdm(space))    
    return df

#---------------------------------------------------------------------------
def optimize_alpha(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile, 
                   plinkexe, step, threads, tar, prune_step=1):
    """
    Do a line search for the best alpha in nrank = alpha*rankP+T + (1-alpha)*cot
    """
    outfn = '%s_optimized.tsv' % prefix
    if not os.path.isfile(outfn):
        d = rank_qr(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile, 
                    step, plinkexe, threads, tar, prunestep=prune_step)
        df = pd.concat(d)
        df.to_csv(outfn, sep='\t', index=False)
    else:
        df = pd.read_table(outfn, sep='\t')
    piv = df.loc[:,['SNP kept','alpha', 'R2']]
    piv = piv.pivot(index='SNP kept',columns='alpha', values='R2').sort_index()
    piv.plot()
    plt.savefig('%s_alphas.pdf'%(prefix))   
    
    return df.sort_values('R2', ascending=False).reset_index(drop=True)

#----------------------------------------------------------------------
def read_n_sort_cotag(prefix, cotagfn, freq):
    """
    Smart sort the cotag file
    """
    cotags = pd.read_table(cotagfn, sep='\t')
    df, _ = smartcotagsort(prefix, cotags[cotags.SNP.isin(freq.SNP)])
    return df.reset_index()


#----------------------------------------------------------------------
def prankcster(prefix, targetbed, referencebed, cotagfn, ppt_results_tar,
               ppt_results_ref, sumstats, pheno, plinkexe, alpha_step, threads, 
               labels, prune_step, sortresults, freq_threshold=0.1, h2=None):
    """
    execute the code and plot the comparison
    """
    ref, tar = labels
    f1 = read_freq(referencebed, plinkexe, freq_threshold=freq_threshold)
    f2 = read_freq(targetbed, plinkexe, freq_threshold=freq_threshold)
    frqs = f1.merge(f2, on=['CHR', 'SNP'], suffixes=['_%s' % ref, '_%s' % tar])   
    if os.path.isfile('%s.sorted_cotag' % prefix):
        sorted_cotag = pd.read_table('%s.sorted_cotag' % prefix, sep='\t')
    else:
        sorted_cotag = read_n_sort_cotag(prefix, cotagfn, f2)
        sorted_cotag.to_csv('%s.sorted_cotag' % prefix, sep='\t', 
                            index=False)    
    clumpetar = read_n_sort_clumped(ppt_results_tar, frqs.SNP)
    clumpetar = clumpetar[clumpetar.SNP.isin(frqs.SNP)]
    clumperef = read_n_sort_clumped(ppt_results_ref, frqs.SNP)
    clumperef = clumperef[clumperef.SNP.isin(frqs.SNP)] 
    clumpe = [(clumpetar, tar), (clumperef, ref)]
    ss = pd.read_table(sumstats, delim_whitespace=True)  
    df = optimize_alpha(prefix, targetbed, sorted_cotag, clumpe, sumstats, 
                        pheno, plinkexe, alpha_step, threads, tar, 
                        prune_step=prune_step)
    grouped = df.groupby('alpha')
    best = grouped.get_group(df.loc[0,'alpha'])
    prevs = pd.read_table(sortresults, sep='\t')
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
                                                'labels of reference and target '
                                                'populations'), nargs=2)
    parser.add_argument('-T', '--target_ppt', help=('Filename of the results of '
                                                    'a PPT run'), default=None) 
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
    parser.add_argument('-S', '--alpha_step', help=('Step for the granularity of'
                                                    ' the grid search. Default: '
                                                    '.1'), default=0.1, 
                                              type=float) 
    parser.add_argument('-E', '--prune_step', help=('Percentage of snps to be '
                                                    'tested at each step is 0.1'
                                                    ), default=0.1, type=float)      
    parser.add_argument('-P', '--plinkexe')
    parser.add_argument('-t', '--threads', default=-1, action='store', type=int) 
    parser.add_argument('-H', '--h2', default=0.66, type=float, 
                        help=('Heritability of the simulated phenotype'))     
    parser.add_argument('-M', '--maxmem', default=False, action='store') 
    parser.add_argument('-F', '--freq_threshold', default=0.1, type=float) 
    args = parser.parse_args()
    prankcster(args.prefix, args.target, args.reference, args.cotagfn, 
               args.target_ppt, args.ref_ppt, args.sumstats, args.pheno,
               args.plinkexe, args.alpha_step, args.threads, args.labels,
               args.prune_step, args.sortresults, h2=args.h2,  
               freq_threshold=args.freq_threshold)   