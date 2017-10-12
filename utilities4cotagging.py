#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Jose Sergio Hleap --<>
  Purpose: Utilities for cottagging
  Created: 09/30/17
"""

from subprocess import Popen, PIPE
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os

#---------------------------------------------------------------------------
def read_log(prefix):
    """
    Read logfile with the profiles written

    :param prefix: prefix of the logfile
    :return: List of files writtenß
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

#----------------------------------------------------------------------
def train_test(prefix, bfile, plinkexe, splits=10, maxmem=1700, threads=1):
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
    fn = os.path.split(bfile)[-1]
    keeps= {'%s_train'% prefix:os.path.join(os.getcwd(),'%s_train.keep' % fn), 
            '%s_test'% prefix: os.path.join(os.getcwd(),'%s_test.keep' % fn)
            }      
    fam = pd.read_table('%s.fam' % bfile, delim_whitespace=True, header=None,
                        names=['FID', 'IID', 'a', 'b', 'c', 'd'])
    msk = np.random.rand(len(fam)) < trainthresh
    fam.loc[msk, ['FID', 'IID']].to_csv(keeps['%s_train'% prefix], header=False, 
                                        index=False, sep=' ')
    fam.loc[~msk,['FID', 'IID']].to_csv(keeps['%s_test' % prefix], header=False, 
                                        index=False, sep=' ')
    make_bed = ('%s --bfile %s --keep %s --make-bed --out %s --memory %d '
                '--threads %d')
    for k, v in keeps.items():
        executeLine(make_bed % (plinkexe, bfile, v, k, maxmem, threads))
    return list(keeps.keys())

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
def smartcotagsort(prefix, gwaswcotag, column='Cotagging'):
    """
    perform a 'clumping' based on Cotagging score, but retain all the rest in 
    the last part of the dataframe
    
    :param str prefix: prefix for oututs
    :param :class pd.DataFrame gwaswcotag: merged dataframe of cotag and gwas
    :param str column: name of the column to be sorted by in the cotag file
    """
    picklefile = '%s_%s.pickle' % (prefix, ''.join(column.split()))
    if os.path.isfile(picklefile):
        with open(picklefile, 'rb') as F:
            df, beforetail = pickle.load(F)
    else:
        print('Sorting File based on %s "clumping"...' % column)
        sorteddf = pd.DataFrame()
        tail = pd.DataFrame()
        grouped = gwaswcotag.groupby(column)
        keys = sorted(grouped.groups.keys(), reverse=True)
        for key in tqdm(keys, total=len(keys)):
            df = grouped.get_group(key)
            sorteddf = sorteddf.append(df.loc[df.index[0],:])
            tail = tail.append(df.loc[df.index[1:],:])
        beforetail = sorteddf.shape[0]
        df = sorteddf.append(tail.sample(frac=1)).reset_index(drop=True)
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
    onesnp = 100/nsnps
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
