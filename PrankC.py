'''
Linear combination of rank P+T/cotagging
'''
import os
import shutil
import argparse
import numpy as np
import pandas as pd
from glob import glob
from LinearPipeline import *
from NullSNPPruning import *
from matplotlib import pyplot as plt
from joblib import delayed, parallel
plt.style.use('ggplot')


def read_n_sort_clumped(resultsfn, allsnps, clump=None):
    """
    Read the <target>.results file, get the best clumping in the P+T and do the
    "sorting"
    """
    if clump is None:
        path = os.path.split(resultsfn)[0]
        res = pd.read_table(resultsfn, sep='\t')
        clumpfn = '%s/%s.clumped' % (path, res.nlargest(1, 'pR2').File[0])
    else:
        clumpfn = clump
    return parse_sort_clump(clumpfn, allsnps).reset_index()

def read_n_sort_cotag(prefix, cotagfn, freq):
    """
    Smart sort the cotag file
    """
    cotags = pd.read_table(cotagfn, sep='\t')
    df, _ = smartcotagsort(prefix, cotags[cotags.SNP.isin(freq.SNP)])
    return df.reset_index()

def rank(prefix, bfile, sorted_cotag, clumped, sumstats, phenofile, alpha,
         plinkexe):
    """
    Estimate the new rank based on the combination of the cotagging and P+T rank
    """
    l=[]
    out = '%s_%.2f' % (prefix, alpha)
    merge = sorted_cotag.merge(clumped, on='SNP', suffixes=['Cotag', 'PpT'])
    nsnps = merge.shape[0]
    merge['New_rank'] = (alpha * merge.indexCotag) + ((1 - alpha) * 
                                                      merge.indexPpT)                            
    new_rank = merge.sort_values('New_rank')
    new_rank['New_rank'] = new_rank.reset_index().index
    for i in set_first_step(nsnps, 1):
        n = min(int(round(nsnps * (i/100.))), nsnps)
        out = '%s_%.2f_%d' % (prefix, alpha, n)
        profilefn = '%s.profile' % out
        if not os.path.isfile(profilefn):
            new_rank.SNP[:n].to_csv('%s.extract' % out, sep=' ', index=False,
                                    header=False)
            scoreit(bfile, sumstats, out, phenofile, plinkexe)
        sc, d, lr = read_scored(profilefn, phenofile)
        l += [{'File':profilefn, 'alpha':alpha, 'R2':lr.rvalue**2,'SNP kept':n}]
    df = pd.DataFrame(l)
    #cleanup()
    return df

def cleanup():
    cwd = os.getcwd()
    for i in glob(os.path.join(cwd, '*.nopred')) + glob(os.path.join(cwd, 
                                                                    '*.nosex')):
        os.remove(i)
    if not os.path.isdir('LOGS'):
        os.mkdir('LOGs')
    for fn in glob(os.path.join(cwd, '*.log')):
        shutil.move(fn, 'LOGs')

def yielder(prefix, bfile, sorted_cotag, clumped,sumstats, phenofile, plinkexe, 
            step):
    space = np.concatenate((np.array([0.05]), np.arange(0.1, 0.9 + step, step)))
    for i in space:
        d = {'prefix': prefix, 'bfile':bfile, 'sorted_cotag':sorted_cotag, 
             'clumped':clumped, 'sumstats':sumstats, 'phenofile':phenofile, 
             'alpha':i, 'plinkexe':plinkexe}
        yield d
        
def optimize_alpha(prefix, bfile, sorted_cotag, clumped, sumstats, phenofile, 
                   plinkexe, step, threads):
    """
    Do a line search for the best alpha in nrank = alpha*rankP+T + (1-alpha)*cot
    """
    if threads == 0:
        d = [rank(**alpha) for alpha in yielder(prefix, bfile, sorted_cotag,
                                                clumped, sumstats, phenofile, 
                                                plinkexe, step)]        
    else:
        d = Parallel(n_jobs=int(threads))(delayed(rank)(**alpha) 
                                          for alpha in yielder(prefix, bfile, 
                                                           sorted_cotag,clumped,
                                          sumstats, phenofile, plinkexe, step))
    df = pd.concat(d)
    df.to_csv('optimized.tsv', sep='\t')
    piv = df.loc[:,['SNP kept','alpha', 'R2']]
    piv = piv.pivot(index='SNP kept',columns='alpha', values='R2').sort_index()
    piv.plot()
    plt.savefig('%s_alphas.pdf'%(prefix))    
    return df.sort_values('R2', ascending=False).reset_index(drop=True)

def main(args):
    """
    execute the code
    """
    f1, f2 = read_freqs(args.bfile1, 'ref', args.bfile2, 'tar', args.plinkexe)
    if os.path.isfile('%s.sorted_cotag' % args.prefix):
        sorted_cotag = pd.read_table('%s.sorted_cotag' % args.prefix, sep='\t')
    else:
        sorted_cotag = read_n_sort_cotag(args.prefix, args.cotagfn, f2)
        sorted_cotag.to_csv('%s.sorted_cotag' % args.prefix, sep='\t', 
                            index=False)
    #sorted_cotag = sorted_cotag[sorted_cotag.SNP.isin(f2.SNP)]
    clumped = read_n_sort_clumped(args.PpTresults, f2.SNP, clump=args.clump)
    clumped = clumped[clumped.SNP.isin(f1.SNP)]
    ss = pd.read_table(args.sumstats,delim_whitespace=True)
    clumped = fix_clumped(clumped, ss.SNP)
    df = optimize_alpha(args.prefix, args.bfile2, sorted_cotag, clumped,
                        args.sumstats, args.pheno, args.plinkexe, args.step, 
                        args.threads)
    grouped = df.groupby('alpha')
    best = grouped.get_group(df.loc[0,'alpha'])
    prevs = pd.read_table(args.sortresults, sep='\t')
    merged = best.merge(prevs, on='SNP kept')
    f, ax = plt.subplots()
    # plot cotagging
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_cotag', label='Cotagging', 
                       c='r', s=2, alpha=0.5, ax=ax)
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_clum', label='Clump Sort', 
                           c='k', s=2, alpha=0.5, ax=ax)    
    merged.plot.scatter(x='SNP kept', y='R2', label='Hybrid', c='g', s=2, 
                        alpha=0.5, ax=ax) 
    plt.savefig('%s_compare.pdf' % args.prefix)
    merged.to_csv('%s.tsv' % args.prefix, sep='\t', index=False)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-b', '--bfile1', help=('prefix of the bed fileset in '
                                                'reference'), required=True)    
    parser.add_argument('-c', '--bfile2', help=('prefix of the bed fileset in '
                                                'target'), required=True)
    parser.add_argument('-C', '--clump', help=('Filename of the clump file to '
                                               'use. by default it uses the '
                                               'best of P+T'), default=None) 
    parser.add_argument('-r', '--PpTresults', help=('Filename with results for '
                                                    'the P+Toptimization'), 
                                              default=None)    
    parser.add_argument('-R', '--sortresults', help=('Filename with results in '
                                                     'the sorting inlcuding path'
                                                     ''), 
                                               required=True)      
    parser.add_argument('-d', '--cotagfn', help=('Filename tsv with cotag '
                                                 'results'), required=True) 
    parser.add_argument('-s', '--sumstats', help=('Filename of the summary stat'
                                                  'istics in plink format'), 
                                            required=True)    
    parser.add_argument('-f', '--pheno', help='filename of the true phenotype' +
                        ' of the target population', required=True)   
    parser.add_argument('-v', '--split', help='number of splits for validation',
                        default=0, type=int)     
    parser.add_argument('-S', '--step', help=('Step for the granularity of the '
                                              'grid search. Default is 0.1'), 
                                        default=0.1, type=float)     
    parser.add_argument('-P', '--plinkexe')
    parser.add_argument('-t', '--threads', default=-1, action='store', type=int)
    args = parser.parse_args()
    main(args)     