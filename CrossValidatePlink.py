'''
CrossValidatePlink.py
make folds for crossvalidation in plink format
'''
#from sklearn.cross_validation import train_test_split
import argparse
import numpy as np
import pandas as pd
from subprocess import Popen, PIPE
from collections import defaultdict
#from joblib import Parallel, delayed
from itertools import cycle

def exe(plinkexe, fn, fam, line, phenofile, fold, i):
    keep = '%s_fold%d.keep'%(fn,i)
    filename = '%s_fold%d'%(fn, i)
    subset = fam.loc[fold,['FID','IID']]
    subset.to_csv(keep, header=False, sep=' ', index=False)    
    if phenofile:
        line = line%(plinkexe, fn, keep, filename, phenofile)
    else:
        line = line%(plinkexe, fn, keep, filename)    
    plink = Popen(line, shell=True)
    #plink.communicate()
    return filename
    
def createFolds(fn, plinkexe, phenofile=None, stratified=False, nfolds=10, 
                sample=False):
    '''
    Using the fam file create the "keep" files for subsetting the genotype file
    as folds
    :param str fn: Prefix of the plink bed-related files
    :param str plinkexe: Path to plink program
    :param str phenofile: If stratified is True, a phenotype file must be passed
    :param bool stratified: Whether or not to stratify the cross valudation
    :param int nfolds: number of folds to be created
    '''
    #outfiles = []
    pcall = '%s --bfile %s --keep %s --make-bed --allow-no-sex --1 '
    pcall += '--keep-allele-order --out %s '
    fam = pd.read_table(fn + '.fam', delim_whitespace=True, header=None, 
                        names=['FID', 'IID', 'father', 'mother', 'Sex', 'Phen'])
    if stratified:
        phen =  pd.read_table(phenofile, delim_whitespace=True, names=['FID', 
                                                                       'IID',
                                                                       'pheno'])
        #phenoname = phen.columns[-1]
        merged = fam.merge(phen, on=['FID','IID'])
        prefold = []
        for name, df in merged.groupby('pheno'):#phenoname):
            idx = df.index.values
            np.random.shuffle(idx)
            prefold.extend(np.array_split(idx, nfolds))
        #tuples = pd.DataFrame(prefold).itertuples()
        rang = range(0,len(prefold),2)
        offset = len(rang)
        if sample:
            s0 = min(min([prefold[x].shape[0] for x in range(offset)]), 
                     sample[0])
            s1 = min(min([prefold[x + offset].shape[0] for x in range(offset)]), 
                     sample[1])
            folds = [np.concatenate((np.random.choice(prefold[i], size=s0,
                                                      replace=False), 
                                     np.random.choice(prefold[i + offset], 
                                     size=s1, replace=False))) 
                     for i in range(offset)]
        else:
            folds = [np.concatenate((prefold[i], prefold[i + offset])) for i in 
                     range(offset)]
        [np.random.shuffle(y) for y in folds]
        line = pcall + '--pheno %s'
    else:
        idx = np.arange(fam.shape[0])
        np.random.shuffle(idx)
        folds =  np.array_split(idx, nfolds)
        line = pcall
    gen = (x for x in zip(cycle([plinkexe]), cycle([fn]), cycle([fam]), 
                          cycle([line]),cycle([phenofile]), folds))
    F = [exe(*params, i) for i, params in enumerate(gen)]
        
    #F = Parallel(n_jobs=-1)(delayed(exe)(*params, i) for i, params in enumerate(gen))        
    outfiles = F
    return outfiles


# Get the original prefix for the bed,bim,fam files
def main(args):
    '''
    execute if called
    '''
    files = createFolds(args.prefix, args.plinkexe, phenofile=args.phenofile, 
                        stratified=args.Stratified, nfolds=args.folds, 
                        sample=args.sample)
    
    




if __name__ == '__main__':
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix of input bed/bim/fam', 
                        required=True)
    parser.add_argument('-f', '--folds', help='number of folds to be performed',
                        default=10, required=True, type=int) 
    parser.add_argument('-n', '--plinkexe',  help='Path to plink executable',
                        required=True)
    parser.add_argument('-P', '--phenofile', help='Path to phenotype file',
                        default=None)
    parser.add_argument('-s', '--Stratified', help='Do not make stratified CV '+
                        'based on cases and controls', default=True, 
                        action='store_false')    
    parser.add_argument('-q', '--sample', help='Create each fold with a given '+
                        'sample size. You have to pass this argument twice, ' +
                        'once for controls and one for cases', action='append', 
                        type=int)    
    
    args = parser.parse_args()
    main(args)