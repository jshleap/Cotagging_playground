#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Jose Sergio Hleap --<>
  Purpose: Utilities for cottagging
  Created: 09/30/17
"""
from scipy.stats import linregress
from joblib import Parallel, delayed
from subprocess import Popen, PIPE
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import tarfile
import pickle
import mmap
import os

#----------------------------------------------------------------------
def mapcount(filename):
    """
    Efficient line counter courtesy of Ryan Ginstrom answer in stack overflow
    """
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines    

#---------------------------------------------------------------------------
def read_log(prefix):
    """
    Read logfile with the profiles written

    :param prefix: prefix of the logfile
    :return: List of files written
    """
    l = []
    with open('%s.log' % prefix) as F:
        for line in F:
            if 'profile written' not in line:
                continue
            else:
                l.append(line.split()[0])
    return l

#----------------------------------------------------------------------
def executeLine(line):
    """
    Execute line with subprocess
    
    :param str line: Line to be executed in shell
    """
    pl = Popen(line, shell=True, stderr=PIPE, stdout=PIPE)
    o, e = pl.communicate()
    return o, e


#----------------------------------------------------------------------
def read_BimFam(prefix):
    """
    Read a bim/fam files from the plink fileset
    :param str prefix: prefix of the plink bedfileset
    """
    Bnames = ['CHR', 'SNP', 'cM', 'BP', 'A1', 'A2']
    bim = pd.read_table('%s.bim'%(prefix), delim_whitespace=True, header=None,
                        names=Bnames)
    Fnames = ['FID', 'IID', 'father', 'mother', 'Sex', 'Phenotype']    
    fam = pd.read_table('%s.fam'%(prefix), delim_whitespace=True, header=None,
                        names=Bnames)    
    return bim, fam


#----------------------------------------------------------------------
def read_freq(bfile, plinkexe, freq_threshold=0.1, maxmem=1700, threads=1):
    """
    Generate and read frequency files and filter based on threshold
    
    :param str bfile: prefix of the plink bedfileset
    :param str plinkexe: path to plink executable
    :param float freq_threshold: Lower threshold to filter MAF by
    """
    high = 1 - freq_threshold
    low = freq_threshold
    if not os.path.isfile('%s.frq.gz' % bfile):
        nname = os.path.split(bfile)[-1]
        frq = ('%s --bfile %s --freq gz --keep-allele-order --out %s --memory '
               '%d --threads %d')
        line = frq % (plinkexe, bfile, nname, maxmem, threads)
        o,e = executeLine(line)
        frq = pd.read_table('%s.frq.gz' % nname, delim_whitespace=True)
    else:
        frq = pd.read_table('%s.frq.gz' % bfile, delim_whitespace=True)  
    #filter MAFs greater than 1 - freq_threshold and smaller than freq_threshold
    return frq[(frq.MAF < high) & (frq.MAF > low)]


# ----------------------------------------------------------------------
def train_test_gen_only(prefix, bfile, plinkexe, splits=10, maxmem=1700,
                        threads=1):
    """
    Generate a list of individuals for training and a list for validation.
    The list is to be passed to plink. It will take one split as validation and
    the rest as training.

    :param str prefix: profix of output
    :param str bfile: prefix for the bed fileset
    :param str plinkexe: path to plink executable
    :param int splits: Number of splits to be done
    """
    trainthresh = (splits - 1) / splits
    fam = pd.read_table('%s.fam' % bfile, delim_whitespace=True, header=None,
                        names=['FID', 'IID', 'a', 'b', 'c', 'd'])
    msk = np.random.rand(len(fam)) < trainthresh
    train, test = '%s_train' % prefix, '%s_test' % prefix
    opts = dict(header=False, index=False, sep=' ')
    fam.loc[msk, ['FID', 'IID']].to_csv('%s.keep' % train, **opts)
    fam.loc[~msk, ['FID', 'IID']].to_csv('%s.keep' % test, **opts)
    make_bed = ('%s --bfile %s --keep %s.keep --make-bed --out %s --memory %d '
                '--threads %d')
    for i in [train, test]:
        executeLine(make_bed % (plinkexe, bfile, i, i, maxmem, threads))
    return train, test

#----------------------------------------------------------------------
def train_test(prefix, bfile, plinkexe, pheno, splits=10, maxmem=1700, 
               threads=1):
    """
    Generate a list of individuals for training and a list for validation.
    The list is to be passed to plink. It will take one split as validation and
    the rest as training.
    
    :param str prefix: profix of output
    :param str bfile: prefix for the bed fileset
    :param str plinkexe: path to plink executable
    :param int splits: Number of splits to be done
    """
    pheno = read_pheno(pheno)
    trainthresh = (splits - 1) / splits
    fn = os.path.split(bfile)[-1]
    keeps= {'%s_train'% prefix:(os.path.join(os.getcwd(),'%s_train.keep' % fn), 
                                             os.path.join(os.getcwd(),
                                                          '%s_train.pheno' % fn)
                                             ), 
            '%s_test'% prefix: (os.path.join(os.getcwd(),'%s_test.keep' % fn),
                                os.path.join(os.getcwd(),'%s_test.pheno' % fn))}      
    fam = pd.read_table('%s.fam' % bfile, delim_whitespace=True, header=None,
                        names=['FID', 'IID', 'a', 'b', 'c', 'd'])
    msk = np.random.rand(len(fam)) < trainthresh
    fam.loc[msk, ['FID', 'IID']].to_csv(keeps['%s_train'% prefix][0], 
                                        header=False, index=False, sep=' ')
    pheno.loc[msk, ['FID', 'IID', 'Pheno']].to_csv(keeps['%s_train'% prefix][1],
                                                   header=False, index=False, 
                                                   sep=' ')

    fam.loc[~msk,['FID', 'IID']].to_csv(keeps['%s_test' % prefix][0], 
                                        header=False, index=False, sep=' ')
    pheno.loc[~msk,['FID', 'IID', 'Pheno']].to_csv(keeps['%s_test' % prefix][1],
                                                   header=False, index=False, 
                                                   sep=' ')    
    make_bed = ('%s --bfile %s --keep %s --make-bed --out %s --memory %d '
                '--threads %d -pheno %s')
    for k, v in keeps.items():
        executeLine(make_bed % (plinkexe, bfile, v[0], k, maxmem, threads, v[1])
                    )
    return keeps

#----------------------------------------------------------------------
def norm(array, a=0, b=1):
    '''
    normalize an array between a and b
    '''
    ## normilize 0-1
    rang = max(array) - min(array)
    A = (array - min(array)) / rang
    ## scale
    range2 = b - a
    return (A * range2) + a    

#----------------------------------------------------------------------
def read_pheno(pheno):
    """
    Read a phenotype file with plink profile format
    
    :param str pheno: Filename of phenotype
    """
    if 'FID' in open(pheno).readline():
        ## asumes that has 3 columns with the first two with headers FID adn
        ## IID
        pheno = pd.read_table(pheno, delim_whitespace=True)
        pheno.rename(columns={pheno.columns[-1]: 'Pheno'}, inplace=True)
    else:
        Pnames = ['FID', 'IID', 'Pheno']
        pheno = pd.read_table(pheno, delim_whitespace=True, header=None,
                              names=Pnames)
    return pheno

#---------------------------------------------------------------------------
def parse_sort_clump(fn, allsnps):
    """
    Parse and sort clumped file
    
    :param str fn: clump file name
    :param :class pd.Series allsnps: Series with all snps being analyzed
    """
    try:
        df = pd.read_table(fn, delim_whitespace=True)
    except FileNotFoundError:
        spl = fn.split('.')
        if spl[0] == '':
            idx=1
        else:
            idx=0
        fn = '.'.join(np.array(spl)[[idx,1+idx,-1]])
        if idx == 1:
            fn = '.%s' % fn
        df = pd.read_table(fn, delim_whitespace=True)
    SNPs = df.loc[:,'SP2']
    tail = [x.split('(')[0] for y in SNPs for x in y.split(',') if x.split('(')[
        0] != 'NONE']
    full = pd.DataFrame(df.SNP.tolist() + tail, columns=['SNP'])
    full = full[full.SNP.isin(allsnps)]
    rest = allsnps[~allsnps.isin(full.SNP)]
    df = pd.concat((full.SNP,rest)).reset_index(drop=False)
    df.rename(columns={'index':'Index'}, inplace=True)   
    return df


#---------------------------------------------------------------------------
def helper_smartsort(grouped, key):
    """
    helper function to parallelize smartcotagsort
    """
    df = grouped.get_group(key)
    head = df.loc[df.index[0],:]
    tail = df.loc[df.index[1:],:]
    return head, tail

#---------------------------------------------------------------------------
def helper_smartsort2(grouped, key):
    """
    helper function to parallelize smartcotagsort
    """  
    df = grouped.get_group(key)
    return df.loc[df.index[0],:]

#---------------------------------------------------------------------------
def smartcotagsort(prefix, gwaswcotag, column='Cotagging', threads=1):
    """
    perform a 'clumping' based on Cotagging score, but retain all the rest in 
    the last part of the dataframe
    
    :param str prefix: prefix for oututs
    :param :class pd.DataFrame gwaswcotag: merged dataframe of cotag and gwas
    :param str column: name of the column to be sorted by in the cotag file
    """
    gwaswcotag = gwaswcotag.sort_values(by=column, ascending=False)
    picklefile = '%s_%s.pickle' % (prefix, ''.join(column.split()))
    if os.path.isfile(picklefile):
        with open(picklefile, 'rb') as F:
            df, beforetail = pickle.load(F)
    else:
        print('Sorting File based on %s "clumping"...' % column)
        #sorteddf = pd.DataFrame()
        #tail = pd.DataFrame()
        grouped = gwaswcotag.groupby(column)
        keys = sorted(grouped.groups.keys(), reverse=True)
        #for key in tqdm(keys, total=len(keys)):
            #df = grouped.get_group(key)
            #sorteddf = sorteddf.append(df.loc[df.index[0],:])
            #tail = tail.append(df.loc[df.index[1:],:])
        tup = Parallel(n_jobs=int(threads))(delayed(helper_smartsort2)(
            grouped, key) for key in  tqdm(keys, total=len(keys)))
        if isinstance(tup[0], pd.core.series.Series):
            sorteddf = pd.concat(tup, axis=1).transpose()
        else:
            sorteddf = pd.concat(tup)
        tail = gwaswcotag[~gwaswcotag.index.isin(sorteddf.index)]
        #sorteddf, tail = zip(*tup)
        #sorteddf = pd.concat(sorteddf)
        #tail = pd.concat(tail)
        beforetail = sorteddf.shape[0]
        df = sorteddf.copy()
        if not tail.empty:
            df = df.append(tail.sample(frac=1))
        df = df.reset_index(drop=True)
        df['Index'] = df.index.tolist()
        with open(picklefile, 'wb') as F:
            pickle.dump((df,beforetail), F)
    return df, beforetail


#---------------------------------------------------------------------------
def set_first_step(nsnps, step, init_step=2, every=False):
    """
    Define the range starting by adding one snp up the the first step
    
    :param int nsnps: Total number of snps
    :param float step: step for the snp range
    """
    onesnp = 100./float(nsnps)
    if every:
        full = np.arange(onesnp, 100 + onesnp, onesnp)
    else:
        # just include the first 200 snps in step of 2
        initial = np.arange(onesnp, (200 * onesnp) + onesnp, (init_step*onesnp))
        rest = np.arange(initial[-1] + onesnp, 100 + step, step)
        full = np.concatenate((initial, rest))
    if full[-1] < 100:
        full[-1] = 100
    return full

#----------------------------------------------------------------------
def gen_qrange(prefix, nsnps, prunestep, every=False, qrangefn=None):
    """
    Generate qrange file to be used with plink qrange
    """
    order = ['label', 'Min', 'Max']
    if qrangefn is None:
        # Define the number of snps per percentage point and generate the range
        percentages = set_first_step(nsnps, prunestep, every=every)
        snps = np.around((percentages * nsnps) / 100).astype(int)
        try:
            # Check if there are repeats in ths set of SNPS
            assert sorted(snps) == sorted(set(snps))
        except AssertionError:
            snps = ((percentages * nsnps) / 100).astype(int)
            assert sorted(snps) == sorted(set(snps))
        labels = ['%.2f' % x for x in percentages]
        # Generate the qrange file
        qrange = '%s.qrange' % prefix
        qr = pd.DataFrame({'label':labels, 'Min':np.zeros(len(percentages)),
                           'Max':snps}).loc[:, order]        
        qr.to_csv(qrange, header=False, index=False, sep =' ')      
    else:
        qrange = qrangefn
        qr = pd.read_csv(qrange, sep=' ', header=None, names=order)
    return qr, qrange
#---------------------------------------------------------------------------
def read_scored_qr(profilefn, phenofile, alpha, nsnps, score_type='sum'):
    """
    Read the profile file a.k.a. PRS file or scoresum
    
    :param str profilefn: Filename of profile being read
    :param str phenofile: Filename of phenotype
    :param int nsnps: Number of snps that produced the profile
    """
    if score_type == 'sum':
        col = 'SCORESUM'
    else:
        col = 'SCORE'
    # Read the profile
    sc = pd.read_table(profilefn, delim_whitespace=True)
    # Read the phenotype file
    pheno = pd.read_table(phenofile, delim_whitespace=True, header=None, names=[
    'FID', 'IID', 'pheno'])
    # Merge the two dataframes
    sc = sc.merge(pheno, on=['FID', 'IID'])
    # Compute the linear regression between the score and the phenotype
    lr = linregress(sc.pheno, sc.loc[:, col])
    # Return results in form of dictionary
    dic = {'File':profilefn, 'alpha':alpha, 'R2':lr.rvalue**2, 'SNP kept':nsnps}
    return dic

#--------------------------------------------------------------------------- 
def qrscore(plinkexe, bfile, scorefile, qrange, qfile, phenofile, ou, qr, maxmem,
            threads, label, prefix, normalized_geno=True):
    """
    Score using qrange
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str plinkexe: Path and executable of plink
    :param str bfile: Prefix of plink-bed fileset
    :param str scorefile: File with the summary statistics in plink format
    :param str qrange: File with the ranges to be passed to the --q-score-range
    :param str phenofile: Filename with the phenotype
    """
    # Score files with the new ranking
    # score = ('%s --bfile %s --score %s 2 4 7 header --q-score-range %s %s '
    #          '--allow-no-sex --keep-allele-order --pheno %s --out %s --memory '
    #          '%d --threads %d')
    if normalized_geno:
        sc_type = 'sum'
    else:
        sc_type = ''
    score = ('%s --bfile %s --score %s --q-score-range %s %s --allow-no-sex '
             '--keep-allele-order --pheno %s --out %s --memory %d --threads %d')
    score = score % (plinkexe, bfile, '%s %s' % (scorefile, sc_type), qrange,
                     qfile, phenofile, ou, maxmem, threads)
    o,e = executeLine(score) 
    # Get the results in dataframe
    profs_written = read_log(ou)
    df  = pd.DataFrame([read_scored_qr('%s.%.2f.profile' % (ou, float(x.label)),
                                       phenofile, label, x.Max, sc_type) if
                        ('%s.%.2f.profile' % (ou, float(x.label)) in 
                         profs_written) else {} 
                        for x in qr.itertuples()]).dropna()
    # Cleanup
    try:
        label = label if isinstance(label, str) else '%.2f' % label
        tarfn = 'Profiles_%s_%s.tar.gz' % (prefix, label)
    except TypeError:
        tarfn = 'Profiles_%s.tar.gz' % (prefix)
    with tarfile.open(tarfn, mode='w:gz') as t:
        for fn in glob('%s*.profile' % ou):
            if os.path.isfile(fn):
                try:
                # it has a weird behaviour
                    os.remove(fn)
                    t.add(fn)
                except:
                    pass  
    return df