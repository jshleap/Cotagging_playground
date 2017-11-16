'''
Linear combination of rank P+T/cotagging
'''
import os
import shutil
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from LinearPipeline import *
from NullSNPPruning import *
from matplotlib import pyplot as plt
from joblib import delayed, Parallel
from LinearPipeline import main as lp
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

def single_alpha_qr(prefix, alpha, merge, qfile, plinkexe, bfile, sumstats, 
                    qrange, phenofile, frac_snps, qr, tar):
    """
    single execution of the alpha loop for paralellization
    """
    ou = '%s_%.2f' % (prefix, alpha)
    qfile = qfile % ou
    merge['New_rank'] = (alpha * merge.Index_Cotag) + ((1 - alpha) * merge.loc[
        :, 'Index_%sPpT' % tar])                            
    new_rank = merge.sort_values('New_rank')
    new_rank['New_rank'] = new_rank.reset_index(drop=True).index.tolist()         
    new_rank.loc[:,['SNP', 'New_rank']].to_csv(qfile, sep=' ', header=False,
                                               index=False)    
    score = ('%s --bfile %s --score %s 2 4 7 header --q-score-range %s '
             '%s --allow-no-sex --keep-allele-order --pheno %s --out %s')
    score = score%(plinkexe, bfile, sumstats, qrange, qfile, phenofile, ou)        
    o,e = execute(score) 
    df  = pd.DataFrame([read_scored_qr('%s.%s.profile' % (ou, x.label),
                                       phenofile, alpha, x.Max)
                        for x in qr.itertuples()])  
    with tarfile.open('Profiles_%.2f.tar.gz' % alpha, mode='w:gz') as t:
        for fn in glob('*.profile'):
            if os.path.isfile(fn):
                try:
                    # it has a weird behaviour
                    os.remove(fn)
                    t.add(fn)
                except:
                    pass
    return df

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
    qfile = '%s.qfile'  
    if isinstance(clumpe, list):
        premerge = clumpe[0][0].merge(clumpe[1][0], on='SNP', suffixes=[
            '_%sPpT' % clumpe[0][1], '_%sPpT' % clumpe[1][1]])
    else:
        premerge = clumpe.rename(columns={'Index':'Index_%sPpT' % tar}).head()
    merge = sorted_cotag.merge(premerge, on='SNP', suffixes=['Cotag', 'PpT'])
    merge = merge.rename(columns={'Index':'Index_Cotag'})
    df = Parallel(n_jobs=int(threads))(delayed(single_alpha_qr)(
        prefix, alpha, merge, qfile, plinkexe, bfile,sumstats, qrange,phenofile,
        frac_snps, qr, tar) for alpha in tqdm(space))    
    #df = pd.concat(df)
    #cleanup()
    return df

def rank(prefix, bfile, sorted_cotag, clumped, sumstats, phenofile, alpha,
         plinkexe):
    """
    Estimate the new rank based on the combination of the cotagging and P+T rank
    """
    l=[]
    out = '%s_%.2f' % (prefix, alpha)
    merge = sorted_cotag.merge(clumped, on='SNP', suffixes=['Cotag', 'PpT'])
    nsnps = merge.shape[0]
    merge['New_rank'] = (alpha * merge.Index_Cotag) + ((1 - alpha) * 
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
    if space[-1] > 1:
        space[-1] = 1
    for i in space:
        d = {'prefix': prefix, 'bfile':bfile, 'sorted_cotag':sorted_cotag, 
             'clumped':clumped, 'sumstats':sumstats, 'phenofile':phenofile, 
             'alpha':i, 'plinkexe':plinkexe}
        yield d
        
def optimize_alpha(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile, 
                   plinkexe, step, threads, qr, tar, prune_step=1):
    """
    Do a line search for the best alpha in nrank = alpha*rankP+T + (1-alpha)*cot
    """
    outfn = '%s_optimized.tsv' % prefix
    if not os.path.isfile(outfn):
        if qr:  
            d = rank_qr(prefix, bfile, sorted_cotag, clumpe, sumstats, 
                        phenofile, step, plinkexe, threads, tar, 
                        prunestep=prune_step)
        else:
            if threads == 0:
                d = [rank(**alpha) for alpha in yielder(prefix, bfile, 
                                                        sorted_cotag,
                                                        clumpe, sumstats, 
                                                        phenofile, plinkexe, 
                                                        step)]        
            else:
                args = (prefix, bfile, sorted_cotag, clumpe, sumstats, 
                        phenofile, plinkexe, step)
                d = Parallel(n_jobs=int(threads))(delayed(rank)(**alpha) for 
                                                  alpha in yielder(args))
        df = pd.concat(d)
        df.to_csv(outfn, sep='\t', index=False)
    else:
        df = pd.read_table(outfn, sep='\t')
    piv = df.loc[:,['SNP kept','alpha', 'R2']]
    piv = piv.pivot(index='SNP kept',columns='alpha', values='R2').sort_index()
    piv.plot()
    plt.savefig('%s_alphas.pdf'%(prefix))    
    return df.sort_values('R2', ascending=False).reset_index(drop=True)
       

def prankcster(args):
    """
    execute the code
    """
    cwd = os.getcwd()
    ref, tar = args.labels
    if args.debug:
        args.alpha_step = 0.4
        args.step = 10
    if args.pipeline:
        print("Starting simulation...")
        bestr2ref, args.reference, args.target = lp(args)
        os.chdir(cwd)
    f1, f2 = read_freqs(args.reference, 'ref', args.target, 'tar', 
                        args.plinkexe)
    if os.path.isfile('%s.sorted_cotag' % args.prefix):
        sorted_cotag = pd.read_table('%s.sorted_cotag' % args.prefix, sep='\t')
    else:
        sorted_cotag = read_n_sort_cotag(args.prefix, args.cotagfn, f2)
        sorted_cotag.to_csv('%s.sorted_cotag' % args.prefix, sep='\t', 
                            index=False)
    #sorted_cotag = sorted_cotag[sorted_cotag.SNP.isin(f2.SNP)]
    clumpe = read_n_sort_clumped(args.PpTresults, f2.SNP, clump=args.clump)
    clumpe = clumpe[clumpe.SNP.isin(f1.SNP)]
    ss = pd.read_table(args.sumstats,delim_whitespace=True)
    clumpe = fix_clumped(clumpe, ss.SNP)    
    #if args.PpTref is not None:
        #clumpref = read_n_sort_clumped(args.PpTref, f1.SNP)
        #clumpref = fix_clumped(clumpref, ss.SNP)
        #clumps = [(clumpe, args.labels[1]), (clumpref, args.labels[0])]
    df = optimize_alpha(args.prefix, args.target, sorted_cotag, clumpe,
                        args.sumstats, args.pheno, args.plinkexe,
                        args.alpha_step, args.threads, args.qr, args.labels[1],
                        prune_step=args.step)
    grouped = df.groupby('alpha')
    best = grouped.get_group(df.loc[0,'alpha'])
    prevs = pd.read_table(args.sortresults, sep='\t')
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
    if args.pipeline:
        ax.axhline(bestr2ref, c='0.5', ls='--') 
    plt.savefig('%s_compare.pdf' % args.prefix)
    merged.to_csv('%s.tsv' % args.prefix, sep='\t', index=False)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-b', '--reference', help=('prefix of the bed fileset '
                                                   'in reference'), 
                                             required=True)    
    parser.add_argument('-c', '--target', help=('prefix of the bed fileset in '
                                                'target'), required=True)
    parser.add_argument('-L', '--labels', help=('Labels for the population in t'
                                                'he same order as the reference'
                                                ' and target. This option has t'
                                                'o be passed twice, once with e'
                                                'ach of the populations'),
                                          action='append')
    parser.add_argument('-C', '--clump', help=('Filename of the clump file to '
                                               'use. by default it uses the '
                                               'best of P+T'), default=None) 
    parser.add_argument('-r', '--PpTresults', help=('Filename with results for '
                                                    'the P+Toptimization'), 
                                              default=None)    
    parser.add_argument('-w', '--PpTref', help=('Include the reference ppt sort'
                                                'ing'), default=None)    
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
    parser.add_argument('-v', '--split', help='number of splits for validation',
                        default=0, type=int)     
    parser.add_argument('-S', '--alpha_step', help=('Step for the granularity of'
                                                   ' the grid search. Default: '
                                                   '.1'), default=0.1, 
                                              type=float) 
    parser.add_argument('-E', '--step', help=('Percentage of snps to be tested '
                                              'at each step is 0.1'),
                                        default=0.1, type=float)      
    parser.add_argument('-P', '--plinkexe')
    parser.add_argument('-t', '--threads', default=-1, action='store', type=int)
    parser.add_argument('-Q', '--qr', help='Use q-range', default=False, 
                        action='store_true')
    parser.add_argument('--pipeline', help='run full pipeline', default=False, 
                            action='store_true')   
    parser.add_argument('-D', '--debug', help='debug settings', 
                        action='store_true', default=False)   
    parser.add_argument('-n', '--ncausal', help='Number of causal variants', 
                        default=200) 
    parser.add_argument('-q', '--quality', help=('type of plots (png, pdf)'), 
                        default='png') 
    parser.add_argument('--sortedclump', help='use clump file instead of res', 
                        default='auto')   
    parser.add_argument('-H', '--h2', help='Heritability', default=0.66, 
                        type=float)     
    parser.add_argument('-M', '--maxmem', default=False, action='store') 
    args = parser.parse_args()
    prankcster(args)     