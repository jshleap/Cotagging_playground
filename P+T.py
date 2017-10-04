#!/usr/bin/env python
#coding:utf-8
"""
  Author:   Jose Sergio Hleap --<>
  Purpose: From SumStats get the best combination of R2 and P-thresholding P + T
  Created: 10/01/17
"""
import os
import shutil
import pickle
import tarfile
import argparse
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy import stats
import matplotlib.pyplot as plt
from utilities4cotagging import executeLine, read_pheno
plt.style.use('ggplot')
matplotlib.use('Agg')

#---------------------------------------------------------------------------
def clump_vars(outpref, bfile, sumstats, r2, window, phenofn, plinkexe):
    """
    Use plink to clump variants based on a pval and r2 threshold 

    :param float r2: LD threshold for clumping       
    """
    outfn = '%s_%.2f_%d'%(outpref, r2, window)
    plink = ('%s --bfile %s -clump %s --clump-p1 1 --clump-p2 1 --clump-r2'
            ' %f --clump-kb %d --out %s --allow-no-sex --keep-allele-order'
            ' --pheno %s')
    if not os.path.isfile('%s.clumped'%(outfn)):
        plink = plink%(plinkexe, bfile, sumstats, r2, window, outfn, phenofn)
        o, e = executeLine(plink)
    ## read Clump file
    fn = '%s.clumped'%(outfn)
    try:
        table = pd.read_table(fn, delim_whitespace=True)  
    except FileNotFoundError as err:
        # check if the error is because of lack of significant clums
        if 'No significant --clump results' in open('%s.log'% outfn).read():
            table = None
        else:
            raise FileNotFoundError(err)
    return outfn, table      

#----------------------------------------------------------------------
def range_profiles(name, range_tuple, r2, qfiledf, phenofn):
    """
    read single profile from the q-range option
    """
    pheno = read_pheno(phenofn)
    range_label = range_tuple.name
    assert float(range_tuple.name) == range_tuple.Max
    nsps = qfiledf[qfiledf.P <= range_tuple.Max].shape[0]
    profilefn = '%s.%s.profile' % (name, range_label)
    score = pd.read_table(profilefn, delim_whitespace=True)    
    score = score.loc[:,['FID', 'IID', 'SCORE']].merge(pheno, on=['FID','IID'])
    score.rename(columns={'SCORE':'PRS'}, inplace=True)
    ## check if peno is binary:
    if set(score.Pheno) <= set([0,1]):
        score['pheno'] = score.Pheno - 1
        y, X = dmatrices('pheno ~ PRS', score, return_type = 'dataframe'
                         )
        logit = sm.Logit(y, X)
        logit = logit.fit(disp=0)
        ## get the pseudo r2 (McFadden's pseudo-R-squared.)
        pR2 = logit._results.prsquared
    else:
        ## Linear/quantitative trait
        slope, intercept, pR2, p_value, std_err = stats.linregress(
            score.Pheno, score.PRS)
        score['pheno'] = score.Pheno 
        pR2=pR2**2
    err = ((score.pheno - score.PRS)**2).sum()
    d = r'$\sum(Y_{AFR} - \widehat{Y}_{AFR|EUR})^{2}$'
    ## get parameter from name of clumped file
    return {'File':'%s.%s' % (name, range_label), 'LDthresh':r2, 
            'Pthresh':range_label, 'R2':pR2,'pval':p_value,'SNP kept':nsps,
            d:err}     

#----------------------------------------------------------------------
def qfile_gen(outpref, clumped, r2, pvals_th):
    """
    generate the qfile for --q-score-range
    """
    qrange = '%s_%.2f.qrange' % (outpref, r2)
    qfile = '%s%.2f.qfile' % (outpref, r2)
    qr = pd.DataFrame({'name':[str(x) for x in pvals_th], 'Min':np.zeros(len(
        pvals_th)), 'Max': pvals_th})
    order = ['name', 'Min', 'Max']
    qr.loc[:, order].to_csv(qrange, header=False, index=False, sep =' ')
    qf = clumped.loc[:,['SNP','P']]
    qf.to_csv(qfile, header=False, index=False, sep=' ')
    return qrange, qfile, qr, qf    

#---------------------------------------------------------------------------
def ScoreClumped(outpref, bfile, clumped, phenofn, sumstatsdf, r2, pvals_th, 
                 plinkexe):
    """
    Compute the PRS for the clumped variants

    :param str clumpPref: prefix of the clumpped file (plink format) 
    """
    results = pd.DataFrame(columns=['File', 'LDthresh', 'Pthresh', 'R2'])
    ## merge to get specific clumps
    name, clumped = clumped
    if not os.path.isfile('%s.pickle' % (name)):
        if clumped is None:
            return 
        merge = sumstatsdf[sumstatsdf.SNP.isin(clumped.loc[: , 'SNP'])]
        if not 'OR' in merge.columns:
            cols = ['SNP', 'A1', 'BETA']
        else:
            cols = ['SNP', 'A1', 'OR']
            merge['OR'] = np.log(merge.OR)
        merge = merge.loc[:, cols]
        ## write file for scoring
        merge.to_csv('%s.score'%(name), sep=' ', header=False, 
                     index=False)
        ## Score using plink
        qrange, qfile, qr, qf = qfile_gen(outpref, clumped, r2, pvals_th)
        score = ('%s --bfile %s --score %s.score --q-score-range %s %s '
                 '--allow-no-sex --keep-allele-order --out %s --pheno %s')
        score = score % (plinkexe, bfile, name, qrange, qfile, name, phenofn)        
        o,e = executeLine(score)
        l = [range_profiles(name, range_label, r2, qf, phenofn) for range_label 
             in qr.itertuples()]
        with open('%s.pickle' % name, 'wb') as f:
            pickle.dump(l, f)
    else:
        with open('%s.pickle' % name, 'rb') as f:
            l = pickle.load(f)
    results = results.append(l).reset_index(drop=True)
    top = glob('%s.profile' % results.nlargest(1, 'R2').File.iloc[0])
    with tarfile.open('Profiles_%.2f.tar.gz' % r2, mode='w:gz') as t:
        for fn in glob('*.profile'):
            if fn not in top:
                t.add(fn)
                os.remove(fn)  
    return results

#----------------------------------------------------------------------
def cleanup(results, clean):
    """
    Organize the CWD
    """
    top10 = results.nlargest(10, 'R2').reset_index(drop=True)
    top = top10.nlargest(1, 'R2')
    for i in glob('%s.*' % top.File.iloc[0]):
        shutil.copy(i, '%s.BEST'%(i))        
    if clean:
        print('Cleaning up ...')
        files = results.File
        tocle = files[~files.isin(top10.File)]
        tocle = [x for y in tocle for x in glob('%s*' % y)]
        profiles = glob('*.profile')                
        for fi in tqdm(tocle, total=len(tocle)):
            if os.path.isfile(fi):
                os.remove(fi)
    if not os.path.isdir('LOGs'):
        os.mkdir('LOGs')
    for f in glob('*.log'):
        shutil.move(f, 'LOGs')    

#----------------------------------------------------------------------
def plotppt(outpref, results):
    """
    PLot results of P + T
    """
    results.loc[:,'pval'] = -np.log(results.pval)
    piv = results.pivot_table(index='SNP kept', values=['pval', 'R2'])
    f, ax = plt.subplots()
    ax2 = ax.twinx()
    # plot results for comparisons
    piv.loc[:,'R2'].plot(ax=ax, color='b', alpha=0.5)
    ax.set_ylabel(r'$R^2$', color='b')
    ax.tick_params('y', colors='b')    
    piv.loc[:,'pval'].plot(ax=ax2, color='r', alpha=0.5)
    ax2.set_ylabel('-log(P-Value)', color='r')
    ax2.tick_params('y', colors='r')         
    plt.savefig('%s_PpT.pdf' % outpref)    
    
#----------------------------------------------------------------------
def pplust(outpref, bfile, sumstats, r2range, prange, snpwindow, phenofn,
           plinkexe, plot=False, clean=False):
    """
    Execute P + T
    """
    sumstatsdf = pd.read_table(sumstats, delim_whitespace=True)
    print('Performing clumping in %s...' % outpref)
    results = [ScoreClumped(outpref, bfile, clump_vars(
        outpref, bfile, sumstats, r2, snpwindow, phenofn,plinkexe), phenofn, 
                            sumstatsdf, r2, prange, plinkexe) for r2 in tqdm(
                                r2range, total=len(r2range))] 
    results = pd.concat(results)
    results.sort_values('R2', inplace=True, ascending=False)
    results.to_csv('%s.results'%(outpref), sep='\t', index=False)
    if plot:
        plotppt(outpref, results)
    cleanup(results, clean)
    return results
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bfile', help='plink fileset prefix', 
                        required=True)    
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-s', '--sumstats', help='Filename of sumary statistics',
                        required=True)    
    parser.add_argument('-P', '--pheno', help='Filename of phenotype file',
                        required=True)    
    parser.add_argument('-n', '--plinkexe', help=('Path and executable file of '
                                                  'plink'), required=True) 
    parser.add_argument('-l', '--LDwindow', help='Physical distance threshold '+
                        'for clumping in kb (250kb by default)', type=int, 
                        default=250)
    parser.add_argument('-c', '--rstart', help='minimum R2 threshold. '
                        'Default: 0.1', type=float, default=0.1)   
    parser.add_argument('-d', '--rstop', help='maximum R2 threshold. '
                        'Default: 0.8', type=float, default=0.8)    
    parser.add_argument('-e', '--rstep', help='step for R2 threshold. '
                        'Default: 0.1', type=float, default=0.1)
    parser.add_argument('-v', '--pstart', help='Minimum value for for the Pval'+
                        ' range. Default: 1E-8', type=float, default=1E-8)
    parser.add_argument('-w', '--pstop', help='Maximum value for for the Pval'+
                        ' range. Default: 1', type=float, default=1)    
    parser.add_argument('-C', '--customP', help='Custom pvalue range.' + 
                        'Default: (None)', default=None)
    parser.add_argument('-z', '--clean', help='Cleanup the clump and profiles', 
                        default=False, action='store_true')    
    parser.add_argument('-L', '--label', help='Label of the populations being' +
                        ' analyzed.', default='EUR')   
    parser.add_argument('-t', '--plot', help='Plot results of analysis', 
                        default=False, action='store_true')      
    args = parser.parse_args()
    
    LDs = [x if x <= 0.99 else 0.99 for x in sorted(
        np.arange(args.rstart, args.rstop + args.rstep, args.rstep), 
        reverse=True)]
    if args.customP:
        Ps = [float('%.1g' % float(x)) for x in args.customP.split(',')] 
    else:
        sta, sto = np.log10(pstart), np.log10(pstop)
        Ps = [float('%.1g' % 10**(x)) for x in np.concatenate(
            (np.arange(sta, sto), [sto]), axis=0)]
    Ps = sorted(Ps, reverse=True)    
    pplust(args.prefix, args.bfile, args.sumstats, LDs, Ps, args.LDwindow, 
           args.pheno, args.plinkexe, plot=args.plot, clean=args.clean)