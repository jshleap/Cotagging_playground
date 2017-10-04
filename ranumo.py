#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Jose Sergio Hleap --<>
  Purpose: Analyze the tagging scores vs P+T and a Null random model
  Created: 10/02/17
"""

import os
import shutil
import pickle
import tarfile
import argparse
import matplotlib
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from functools import reduce
import matplotlib.pyplot as plt
from utilities4cotagging import *
from scipy.stats import linregress
from subprocess import Popen, PIPE
from joblib import Parallel, delayed
plt.style.use('ggplot')
matplotlib.use('Agg')

#---------------------------------------------------------------------------
def read_scored_qr(profilefn, phenofile, kind, nsnps):
    """
    Read the profile file a.k.a. PRS file or scoresum
    
    :param str profilefn: filename of scored (.profile) file
    :param str phenoflie: file name of file with phenotype
    :param str kind: label to match the scoring type (e.g cotag, clump, etc..)
    :param int nsps: number of snps that were used to score the profile fn
    """
    sc = pd.read_table(profilefn, delim_whitespace=True)
    pheno = pd.read_table(phenofile, delim_whitespace=True, header=None, names=[
    'FID', 'IID', 'pheno'])
    mer = sc.merge(pheno, on=['FID','IID'])
    lr = linregress(mer.pheno, mer.SCORE)
    dic = {'SNP kept':nsnps, '-log(P)_%s' % kind : -np.log10(lr.pvalue), 
           r'$R^{2}$_%s' % kind : lr.rvalue**2, 'Slope_%s' % kind: lr.slope}
    return dic

#---------------------------------------------------------------------------
def read_gwas_n_cotag(gwasfile, cotagfile):
    """
    Read the GWAS (a.k.a. summary stats) in the reference population and merge
    it with the cotagging info
    """
    cotag = pd.read_table(cotagfile, sep='\t')
    gwas = pd.read_table(gwasfile, delim_whitespace=True)
    gwas = gwas.sort_values('BP').reset_index()
    return gwas.merge(cotag, on='SNP')

#---------------------------------------------------------------------------
def subsetter_qrange(prefix, sortedcota, sortedtagT, sortedtagR, step,
                     phenofile, tarbed, clumped=None):
    """
    Create the files to be used in q-score-range. It will use the index of the
    sorted files as thresholds
    
    :param str prefix: prefix for oututs
    :param :class pd.DataFrame sortedcota: Sorted cotag data frame
    :param :class pd.DataFrame sortedtagT: Sorted target tag data frame
    :param :class pd.DataFrame sortedtagR: Sorted referemce tag data frame
    :param float step: step for the snp range
    :param tuple clumped: list of tuples with clumped dataframe and kind, and 
    phenofile
    """  
    assert sortedcota.shape[0] == sortedtagR.shape[0]
    nsnps = sortedcota.shape[0]  
    randomtagg = sortedcota.copy()
    idxs = randomtagg.Index.tolist()
    np.random.shuffle(idxs)
    randomtagg['Index'] = idxs
    prefix = prefix.replace('_', '')
    qrange = '%s.qrange' % prefix
    percentages = set_first_step(nsnps, step)
    order = ['label', 'Min', 'Max']
    qr = pd.DataFrame({'label':percentages, 'Min':np.zeros(len(percentages)),
                       'Max':np.around(np.array(percentages, dtype=float)*(
                           nsnps/100)).astype(int)}).loc[:, order]
    qr.to_csv(qrange, header=False, index=False, sep =' ')    
    qfile = '%s_%s.qfile'    
    c = (qfile % (prefix, 'cotag'), phenofile, tarbed)
    t = (qfile % (prefix, 'tagt'), phenofile, tarbed)
    r = (qfile % (prefix, 'tagr'), phenofile, tarbed)
    a = (qfile % (prefix, 'rand'), phenofile, tarbed)
    sortedcota.loc[:,['SNP', 'Index']].to_csv(c[0], sep=' ', header=False, 
                                              index=False)
    sortedtagT.loc[:,['SNP', 'Index']].to_csv(t[0], sep=' ', header=False, 
                                              index=False)
    sortedtagR.loc[:,['SNP', 'Index']].to_csv(r[0], sep=' ', header=False, 
                                              index=False)
    randomtagg.loc[:,['SNP', 'Index']].to_csv(a[0], sep=' ', header=False,
                                              index=False)    
    out = (qr, c, t, r, a) 
    if clumped is not None:
        for clumped, kind, pf, bed in clumped:
            p = (qfile % (prefix, 'clum%s' % kind), pf, bed)
            out += (p,)
            clumped.loc[:,['SNP', 'Index']].to_csv(p[0], sep=' ', header=False, 
                                                   index=False)
    return out 

#---------------------------------------------------------------------------
def cleanup():
    """
    Clean up the folder: remove nosex lists, put all logs under the LOGs folder
    and all the extract files under SNPs folder
    """
    if not os.path.isdir('LOGs'):
        os.mkdir('LOGs')
    else:
        shutil.rmtree('LOGs')
        os.mkdir('LOGs')
    if not os.path.isdir('SNP_lists'):
        os.mkdir('SNP_lists')
    else:
        shutil.rmtree('SNP_lists')
        os.mkdir('SNP_lists')        
    for log in glob('*.log'):
        shutil.move(log, 'LOGs')
    for ns in glob('*.nosex'):
        os.remove(ns)
    for snp in glob('*.extract'):
        shutil.move(snp, 'SNP_lists')
    for nopred in glob('*.nopred'):
        os.remove(nopred)

#----------------------------------------------------------------------
def score_qfiles(out, prefix, plinkexe, gwasfn, frac_snps):
    """
    Score the set of qfiles defined in out
    
    :param tuple outs: tuple with the qfiles to be analyzed 
    :param str prefix: prefix of outputs
    """
    qr = out[0]
    qrange = '%s.qrange' % prefix
    frames = []
    for tup in tqdm(out[1:], total=len(out[1:])):
        qfile, phenofile, bfile = tup
        suf = qfile[qfile.find('_') +1 : qfile.rfind('.')]
        ou = '%s_%s' % (prefix, suf)
        score = ('%s --bfile %s --score %s 2 4 7 header --q-score-range %s %s '
                 '--allow-no-sex --keep-allele-order --pheno %s --out %s')
        score = score%(plinkexe, bfile, gwasfn, qrange, qfile, phenofile, ou)
        o,e = executeLine(score)       
        df = pd.DataFrame([read_scored_qr('%s.%s.profile' % (ou, x.label), 
                                          phenofile, suf, 
                                          round(float(x.label) * frac_snps))
                           for x in qr.itertuples()])
        frames.append(df)    
        with tarfile.open('Profiles_%s.tar.gz' % ou, mode='w:gz') as t:
            for fn in glob('*.profile'):
                if os.path.isfile(fn):
                    t.add(fn)
                    os.remove(fn) 
    return frames

#---------------------------------------------------------------------------
def prunebypercentage_qr(prefix, bfile, gwasfn, phenofile, sortedcotag, 
                         sortedtagT, sortedtagR, plinkexe, clumped=None, step=1,
                         causal=None, tar_label='AFR', ref_label='EUR'):
    """
    Execute the prunning in a range from <step> to 100 with step <step> (in %)
    scoring using --q-score-ragne
    :param str gwasfn: filename of the summary stats to get the betas from
    """
    frac_snps = sortedcotag.shape[0]/100
    if os.path.isfile('pbp.pickle'):
        with open('pbp.pickle', 'rb') as f:
            merge = pickle.load(f)
    else:
        print('Performing prunning ...')
        out = subsetter_qrange(prefix, sortedcotag,sortedtagT, sortedtagR, step,
                               phenofile, bfile, clumped=clumped) 
        frames = score_qfiles(out, prefix, plinkexe, gwasfn, frac_snps)
        merge = reduce(lambda x, y: pd.merge(x, y, on = 'SNP kept'), frames)
        merge['Percentage of SNPs used'] = (merge.loc[:, 'SNP kept']/merge.loc[
            :, 'SNP kept'].max()) * 100
        merge.to_csv('%s_merged.tsv' % prefix, sep='\t', index=False)
        cleanup()
        with open('pbp.pickle', 'wb') as f:
            pickle.dump(merge, f)
    return merge


#---------------------------------------------------------------------------
def plotit(prefix, merge, col, labels, ppt=None, line=False, vline=None,
           hline=None, plottype='png', x='SNP kept'):
    """
    Plot the error (difference) vs the proportion of SNPs included for random
    pick and cotagging
    """
    ref, tar = labels
    if line:
        rand = merge.pivot(x).loc[:,col+'_rand']
        cota = merge.pivot(x).loc[:,col+'_cotag']
        f, ax = plt.subplots()
        rand.plot(x=x, y=col+'_rand', label='Random', c='b', s=2, alpha=0.5, 
                  ax=ax)
        cota.plot(x=x, y=col+'_cotag', label='Cotagging', ax=ax, c='r', s=2, 
                  alpha=0.5)
    else:
        f, ax = plt.subplots()
        merge.plot.scatter(x=x, y=col+'_rand', label='Random', c='b', s=2, 
                           alpha=0.5, ax=ax)
        merge.plot.scatter(x=x, y=col+'_cotag', label='Cotagging', ax=ax, c='r',
                           s=2, alpha=0.5)
        merge.plot.scatter(x=x, y=col+'_tagt', label='Tagging %s' % tar, ax=ax, 
                           c='c', s=2, alpha=0.5)   
        merge.plot.scatter(x=x, y=col+'_tagr', label='Tagging %s' % ref, ax=ax, 
                           c='m', s=2, alpha=0.5)  
        #if isinstance(ppt, str):
        merge.plot.scatter(x=x, y=col+'_clum%s' % ref , ax=ax, c='0.5', s=2, 
                           alpha=0.5, label='Sorted Clump %s' % ref, marker='*')             
        merge.plot.scatter(x=x, y=col+'_clum%s' % tar , ax=ax, c='k', s=2, 
                           alpha=0.5, label='Sorted Clump %s' % tar)        
    if vline is not None:
        ax.axvline(float(vline), c='0.5', ls='--')
    if hline is not None:
        ax.axhline(float(hline), c='0.5', ls='--')        
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig('%s_error.%s' % (prefix, plottype)) 
    plt.close()
    
def ranumo(prefix, tarbed, refbed, gwasfn, cotagfn, plinkexe, labels, phenotar, 
           phenoref, pptR=None, pptT=None, check_freqs=None, hline=None,
           step=1, quality='png'):
    """
    execute the code  
    """
    prefix = prefix.replace('_','')
    ref, tar = labels
    gwas = pd.read_table(gwasfn, delim_whitespace=True)        
    if isinstance(cotagfn, str):
        cotags = pd.read_table(cotagfn, sep='\t')
    gwas = gwas.merge(cotags, on='SNP')
    if check_freqs is not None:
        frq = read_freq(tarbed, plinkexe, freq_threshold=check_freqs)
        gwas = gwas[gwas.SNP.isin(frq.SNP)]
    # Cotagging
    sortedcot, beforetail = smartcotagsort(prefix, cotags)
    # Tagging Target
    sortedtagT, beforetailTT = smartcotagsort(prefix, cotags, 
                                              column='Tagging %s' % tar)
    # Tagging Reference
    sortedtagR, beforetailTR = smartcotagsort(args.prefix, cotags,
                                              column='Tagging %s' % ref) 
    # Process clump if required
    clump = []
    if pptR is not None:
        if isinstance(pptR, str):
            resR = pd.read_table(pptR, sep='\t')
        else:
            assert isinstance(pptR, pd.DataFrame)
            resR = pptR
        best_clumpR = resR.nlargest(1, 'R2').File.iloc[0]
        clumR = parse_sort_clump(os.path.join(os.path.split(pptR)[0], 
                                              '%s.clumped' % best_clumpR), 
                                              gwas.SNP)
        clump.append((clumR, ref, phenoref, refbed))
    if pptT is not None:
        if isinstance(pptT, str):
            resT = pd.read_table(pptT, sep='\t')
        else:
            assert isinstance(pptT, pd.DataFrame)
            resT = pptT        
        best_clumpT = resT.nlargest(1, 'R2').File.iloc[0]
        clumT = parse_sort_clump(os.path.join(os.path.split(pptT)[0], 
                                              '%s.clumped' % best_clumpT), 
                                              gwas.SNP)
        clump.append((clumT, tar, phenotar, tarbed))        
    merge = prunebypercentage_qr(prefix, tarbed, gwasfn, phenotar, sortedcot, 
                                 sortedtagT, sortedtagR, plinkexe, 
                                 clumped=clump, step=step)      

    plotit(prefix+'_rval', merge, r'$R^{2}$', labels, ppt=clump, 
           plottype=quality, hline=hline)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-g', '--gwas', help='Filename of gwas results in the' +
                        ' reference population', required=True)
    parser.add_argument('-b', '--tarbed', help=('prefix of the bed fileset for'
                                                 ' the target population'), 
                                           required=True)
    parser.add_argument('-R', '--refbed', help=('prefix of the bed fileset for'
                                                 ' the reference population'), 
                                           required=True)    
    parser.add_argument('-f', '--phenotar', help=('filename of the true '
                                                  'phenotype of the target '
                                                  'population'), required=True) 
    parser.add_argument('-i', '--phenoref', help=('filename of the true '
                                                  'phenotype of the reference '
                                                  'population'), required=True)    
    parser.add_argument('-P', '--plinkexe', default='~/Programs/plink_mac/plink'
                        )    
    parser.add_argument('-c', '--cotagfile', help='Filename of the cotag tsv ' +
                        'file ')
    parser.add_argument('-s', '--step', help='Step in the percentage range to' +
                        ' explore. By deafult is 1', default=1, type=float) 
    parser.add_argument('-l', '--labels', help=('Space separated string with '
                                                'reference and target lables '
                                                '(in that order)'), nargs=2) 
    parser.add_argument('-r', '--ppt_ref', help=('Path to P+T results in the '
                                                 'reference population'), 
                                           default=None)
    parser.add_argument('-t', '--ppt_tar', help=('Path to P+T results in the '
                                                 'reference population'), 
                                           default=None)
    parser.add_argument('-H', '--h2', help=('Heritability of the simulated '
                                            'phenotype'), default=False) 
    parser.add_argument('-F', '--check_freq', help=('Read a frequency file and '
                                                    'filter by this threshold'), 
                                              default=0.1, type=float) 
    parser.add_argument('-q', '--quality', help=('type of plots (png, pdf)'), 
                        default='png')     
    args = parser.parse_args()
    ranumo(args.prefix, args.tarbed, args.refbed, args.gwas, args.cotagfile, 
           args.plinkexe, args.labels, args.phenotar, args.phenoref,  
           pptR=args.ppt_ref, pptT=args.ppt_tar, check_freqs=args.check_freq, 
           hline=args.h2, step=args.step, quality=args.quality)