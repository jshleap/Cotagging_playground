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
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy import stats
import matplotlib.pyplot as plt
from utilities4cotagging import executeLine, read_pheno
plt.style.use('ggplot')


#---------------------------------------------------------------------------
def clump_vars(outpref, bfile, sumstats, r2, window, phenofn, plinkexe, maxmem,
               threads):
    """
    Use plink to clump variants based on a pval and r2 threshold 
    
    :param str outpref: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param str sumstats: File with the summary statistics in plink format 
    :param float r2: LD threshold for clumping       
    :param int window: Size of the clumping window
    :param str phenofn: File with the phenotype in plink format
    :param str plinkexe: Path and executable of plink
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    """
    # Define output name
    outfn = '%s_%.2f_%d'%(outpref, r2, window)
    # Prepare plink CLA
    plink = ('%s --bfile %s -clump %s --clump-p1 1 --clump-p2 1 --clump-r2'
            ' %f --clump-kb %d --out %s --allow-no-sex --keep-allele-order'
            ' --pheno %s --memory %d --threads %d')
    # If output is not there yet, execute the command
    if not os.path.isfile('%s.clumped'%(outfn)):
        plink = plink%(plinkexe, bfile, sumstats, r2, window, outfn, phenofn,
                       maxmem, threads)
        o, e = executeLine(plink)
    #=# read Clump file
    fn = '%s.clumped'%(outfn)
    try:
        table = pd.read_table(fn, delim_whitespace=True)  
    except FileNotFoundError as err:
        # check if the error is because of lack of significant clumps
        if 'No significant --clump results' in open('%s.log'% outfn).read():
            table = None
        else:
            raise FileNotFoundError(err)
    # returns the filename of the output and its dataframe
    return outfn, table      

#----------------------------------------------------------------------
def range_profiles(name, range_tuple, r2, qfiledf, phenofn):
    """
    Read single profile from the q-range option
    
    :param str name: Output name of clumping
    :param tuple range_tuple: Namedtuple with row info from the qrange dataframe
    :param float r2: LD threshold for clumping
    :param :class pd.DataFrame qfiledf: Data frame with the qfile information
    :param str phenofn: File with the phenotype in plink format
    """
    # Read phenotype into a pandas dataframe
    pheno = read_pheno(phenofn)
    # Get the lable from the named tuple
    range_label = range_tuple.name
    # Make sure that that everything matches as it supposed to
    assert float(range_tuple.name) == range_tuple.Max
    # Get the number of SNPs
    nsps = qfiledf[qfiledf.P <= range_tuple.Max].shape[0]
    # Set the input file name
    profilefn = '%s.%s.profile' % (name, range_label)
    # Read the profile
    score = pd.read_table(profilefn, delim_whitespace=True)    
    # Merge score and pheno by individual (IID) and family (FID) IDs 
    score = score.loc[:,['FID', 'IID', 'SCORE']].merge(pheno, on=['FID','IID'])
    # Rename SCORE to PRS
    score.rename(columns={'SCORE':'PRS'}, inplace=True)
    # check if peno is binary:
    if set(score.Pheno) <= set([0,1]):
        score['pheno'] = score.Pheno - 1
        y, X = dmatrices('pheno ~ PRS', score, return_type = 'dataframe'
                         )
        logit = sm.Logit(y, X)
        logit = logit.fit(disp=0)
        # get the pseudo r2 (McFadden's pseudo-R-squared.)
        pR2 = logit._results.prsquared
    else:
        # Linear/quantitative trait
        slope, intercept, pR2, p_value, std_err = stats.linregress(
            score.Pheno, score.PRS)
        score['pheno'] = score.Pheno 
        pR2=pR2**2
    # return a dictionary with the filename, r2, Pval-threshold used and nsnps
    return {'File':'%s.%s' % (name, range_label), 'LDthresh':r2, 
            'Pthresh':range_label, 'R2':pR2,'pval':p_value,'SNP kept':nsps}     

#----------------------------------------------------------------------
def qfile_gen(outpref, clumped, r2, pvals_th):
    """
    Generate the qfile for --q-score-range
    
    :param str outpref: Prefix for outputs
    :param :class pd.DataFrame clumped: Dataframe with the clumpfile results 
    :param float r2: LD threshold for clumping
    :param list pvals_th: List with the pvalue thresholds to be tested
    """
    # Set the input/output names
    qrange = '%s_%.2f.qrange' % (outpref, r2)
    qfile = '%s%.2f.qfile' % (outpref, r2)
    # Set the values for the qrange file
    qr = pd.DataFrame({'name':[str(x) for x in pvals_th], 'Min':np.zeros(len(
        pvals_th)), 'Max': pvals_th})
    order = ['name', 'Min', 'Max']
    # Write q-range to file
    qr.loc[:, order].to_csv(qrange, header=False, index=False, sep =' ')
    # Set and write qfile based on clumped values
    qf = clumped.loc[:,['SNP','P']]
    qf.to_csv(qfile, header=False, index=False, sep=' ')
    # return the output filenames and the corresponding dataframes
    return qrange, qfile, qr, qf    

#---------------------------------------------------------------------------
def ScoreClumped(outpref, bfile, clumped, phenofn, sumstatsdf, r2, pvals_th, 
                 plinkexe, maxmem=1700, threads=8):
    """
    Compute the PRS for the clumped variants
    :param str outpref: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param tuple clumped: tuple with name and Pandas Data Frame with clump data
    :param :class pd.DataFrame sumstatsdf: Summary statistics Pandas Data Frame
    :param float r2: LD threshold for clumping
    :param list pvals_th: List with the pvalue thresholds to be tested
    :param str plinkexe: Path and executable of plink
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    """
    # Create empty dataframe with predifined columns to store reults
    results = pd.DataFrame(columns=['File', 'LDthresh', 'Pthresh', 'R2'])
    # Expand the clumped tuple 
    name, clumped = clumped
    if not os.path.isfile('%s.pickle' % (name)):
        if clumped is None:
            return 
        # Merge summary statistics and the clumped df
        merge = sumstatsdf[sumstatsdf.SNP.isin(clumped.loc[: , 'SNP'])]
        if not 'OR' in merge.columns:
            cols = ['SNP', 'A1', 'BETA']
        else:
            cols = ['SNP', 'A1', 'OR']
            merge['OR'] = np.log(merge.OR)
        merge = merge.loc[:, cols]
        # Write file for scoring
        merge.to_csv('%s.score'%(name), sep=' ', header=False, 
                     index=False)
        # Score using plink
        qrange, qfile, qr, qf = qfile_gen(outpref, clumped, r2, pvals_th)
        score = ('%s --bfile %s --score %s.score --q-score-range %s %s '
                 '--allow-no-sex --keep-allele-order --out %s --pheno %s '
                 '--memory %d --threads %d')
        score = score % (plinkexe, bfile, name, qrange, qfile, name, phenofn,
                         maxmem, threads)        
        o,e = executeLine(score)
        # read range results
        l = [range_profiles(name, range_label, r2, qf, phenofn) for range_label 
             in qr.itertuples()]
        with open('%s.pickle' % name, 'wb') as f:
            pickle.dump(l, f)
    else:
        with open('%s.pickle' % name, 'rb') as f:
            l = pickle.load(f)
    # Store results
    results = results.append(l).reset_index(drop=True)
    top = glob('%s.profile' % results.nlargest(1, 'R2').File.iloc[0])
    # Cleanup leaving the top
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
    
    :param :class pd.DataFrame results: Pandas dataframe with p+t results
    :param bool clean: Whether to clean the folder or not
    """
    # Get top 10 results to avoid erasing them
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
    #if not os.path.isdir('LOGs'):
    #    os.mkdir('LOGs')
    #for f in glob('*.log'):
    #    shutil.move(f, 'LOGs')

#----------------------------------------------------------------------
def plotppt(outpref, results):
    """
    Plot results of P + T
    
    :param str outpref: Prefix for outputs
    :param :class pd.DataFrame results: Pandas dataframe with p+t results
    """
    # Fix the grid
    matplotlib.rcParams.update({'figure.autolayout': True})
    # Get the -log of pvalues in the result file
    results.loc[:,'pval'] = -np.log(results.pval)
    # reorganize the data frame to efficiently plot the alpha exploration
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
           plinkexe, plot=False, clean=False, maxmem=1700, threads=1):
    """
    Execute P + T
    
    :param str outpref: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param str sumstats: File with the summary statistics in plink format 
    :param list r2range: List with floats representing LD thresholds
    :param list prange: List wirth pvalue thresholds
    :param int snpwindow: Size of the clumping window
    :param str phenofn: File with the phenotype in plink format
    :param str plinkexe: Path and executable of plink
    :param bool plot: Wether to plot the P + T results
    :param bool clean: Whether to clean the folder or not
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    """
    # Read the summary statistics file
    sumstatsdf = pd.read_table(sumstats, delim_whitespace=True)
    # Ensure the phenotype file contains only individuals from the bfile
    fn = os.path.split(bfile)[-1]
    if os.path.isfile('%s.keep' % fn):
        nphenofn = '%s.pheno' % fn
        keep = pd.read_table('%s.keep' % fn, delim_whitespace=True, 
                             header=None, names=['FID', 'IID'])
        ph = pd.read_table(phenofn, delim_whitespace=True, header=None, 
                           names=['FID', 'IID', 'pheno'])
        ph[ph.IID.isin(keep.IID)].to_csv(nphenofn, sep=' ', header=False, 
                                         index=False)
        phenofn = nphenofn
    print('Performing clumping in %s...' % outpref)
    # Execute the clumping and read the files
    results = [ScoreClumped(outpref, bfile, clump_vars(
        outpref, bfile, sumstats, r2, snpwindow, phenofn, plinkexe, maxmem, 
        threads), phenofn, sumstatsdf, r2, prange, plinkexe, maxmem, threads) 
               for r2 in tqdm(r2range, total=len(r2range))] 
    # Concatenate the results in a single data frame
    results = pd.concat(results)
    # Sort the results by R2 and write them to file
    results.sort_values('R2', inplace=True, ascending=False)
    results.to_csv('%s.results'%(outpref), sep='\t', index=False)
    # Plot results if necessary
    if plot:
        plotppt(outpref, results)
    # Clean folders if necessary
    cleanup(results, clean)
    # Returns the dataframe with the results and the path to its file
    return results, os.path.join(os.getcwd(), '%s.results'%(outpref))
    

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