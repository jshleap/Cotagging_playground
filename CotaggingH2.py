"""
Test cotagging in a range of heritabilities
"""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob 
from LinearPRS import *
#from subprocess import Popen
from LinearPipeline import *
from NullSNPPruning import *
from scipy.stats import linregress
from matplotlib import pyplot as plt
from joblib import delayed, parallel
plt.style.use('ggplot')

def remove_select():
    """
    remove most files
    """
    profiles = ['profile', 'tsv', 'log', 'clumped', 'linear', 'nosex', 'pheno',
                'gz', 'truebeta', 'score', 'truebeta', 'pickle']
    for p in tqdm(profiles, total=len(profiles)):
        for f in G('*.%s'%p):
            os.remove(f)

def getbest(prefix, h2):
    """
    loop over profiles, and return the higest R2
    """
    res = []
    types = ['cotag', 'rand', 'tagt', 'tagr']
    for t in types:
        ars = []
        for f in glob('%s_*%s.profile' % (prefix, t)):
            df = pd.read_table(f, delim_whitespace=True)
            slope, intercept, r, p_value, std_err = linregress(x = df.PHENO,
                                                               y = df.SCORE)
            ars.append(r**2)
        res.append({'h2':h2, 'R2':max(ars), 'type':t}) 
    return pd.DataFrame(res)
            

def single(prefix, reference, target, labels, h2, ncausal, cotagfn, freqs,
           plinkexe, threads, maxmem, quality):
    print('Processing Heritability %.2f' % h2)
    pthresh=('1.0,0.8,0.5,0.4,0.3,0.2,0.1,0.08,0.05,0.02,0.01,10E-3,10E-4,1'
             '0E-5,10E-6,10E-7,10E-8')    
    picklefile = '%s_h2_%.2f.single' % (prefix, h2)
    if not os.path.isfile(picklefile):
        cwd = os.getcwd()
        ref, tar = labels
        f1, f2 = freqs
        # Simulate reference phenotypes
        gwas2, truebeta2 = simulate_phenos(reference, ref, plinkexe, ncausal,h2, 
                                           threads, maxmem, frq=f1, ext=quality
                                           )    
        truebeta2.to_csv('%s.truebeta' % ref, sep='\t', index=False)
        causalfn = '%s.score'%(ref)    
        causals = pd.read_table(causalfn, delim_whitespace=True, header=None, 
                                names=['SNP', 'A1', 'True_beta'])   
        gwasfn = os.path.join(cwd, '%s_gwas.assoc.linear' % (ref))
        if 'beta' in truebeta2.columns:
            ceff = truebeta2.rename(columns={'beta':'Eff'}).loc[:,['SNP','Eff']]
        elif 'Eff' in truebeta2.columns:
            ceff = truebeta2.loc[:, ['SNP','Eff']]
        else:
            ceff = truebeta2.rename(columns={'eff':'Eff'}).loc[:, ['SNP','Eff']]
        # Simulate target phenotypes
        gwas1, truebeta1,  = simulate_phenos(target, tar, plinkexe, ncausal, h2, 
                                             threads, maxmem,snps=truebeta2.SNP, 
                                             frq=f2, causaleff=ceff, ext=quality
                                             )
        tarphe = '%s.pheno' % (tar)
        # perform P + T on african genotype with european sumstats
        PpT = PplusT(target, plinkexe, '%s-%s' % (tar, ref), 250, gwasfn,tarphe,
                     customPrange=pthresh)
        # read and sort Cotagging score
        cotags = pd.read_table(cotagfn, sep='\t')
        sortedcot, beforetail = smartcotagsort(args.prefix, cotags)
        sortedtagT, beforetailTT = smartcotagsort(args.prefix, cotags, 
                                                  column='Tagging %s' % tar)  
        sortedtagR, beforetailTR = smartcotagsort(args.prefix, cotags,
                                                  column='Tagging %s' % ref) 
        merge, col = prunebypercentage(prefix, target, gwasfn, '%s.pheno' % tar, 
                                       sortedcot, sortedtagT, sortedtagR,
                                       plinkexe, step=1) 
        result = getbest(prefix, h2)
        result = result.append(pd.DataFrame([{'R2':max(PpT.results.pR2),'h2':h2,
                                              'type':'P+T'}]))
        remove_select()
        with open(picklefile, 'wb') as F:
            pickle.dump(result, F)
    else:
        with open(picklefile, 'rb') as F:
            result = pickle.load(F)
    return result

def plot(prefix, results, tar, ref, quality):
    types = [('cotag','Cotagging', 2,'r', None), ('rand','Random', 2, 'b', None), 
             ('tagr', 'Tagging %s' % ref, 2, 'm', None), 
             ('tagt', 'Tagging %s' % tar, 2, 'c', None), 
             ('P+T', 'P+T %s' % tar, 10, 'c', '$%s$' % tar[0])]
    f, ax = plt.subplots()
    for t in types:
        ty, label, s, c, marker = t
        #results[results.type == ty].plot.scatter(x='h2', y='R2', label=label, 
                                                 #s=s, c=c, marker=marker, ax=ax,
                                                 #alpha=0.5)    
        results[results.type == ty].plot(x='h2', y='R2', label=label, ax=ax,
                                         color=c, marker=marker, alpha=0.5)
    plt.ylabel('$R^2$')
    plt.tight_layout()
    plt.savefig('%s_tagvsh2.%s' % (prefix, quality))
    
def main(args):
    """
    execute cotaggingh2
    """
    ref, tar = args.labels
    freqs = read_freqs(args.reference, ref, args.target, tar, args.plinkexe)
    h2s= np.arange(0,1.1,0.1)
    dfs = [single(args.prefix, args.reference, args.target, args.labels, h2, 
                  args.ncausal, args.cotagfn, freqs, args.plinkexe,args.threads,
                  args.maxmem, args.quality) for h2 in h2s]
    #dfs = Parallel(n_jobs=int(args.threads))(delayed(single)(
        #args.prefix, args.reference, args.target, args.labels, h2, args.ncausal, 
        #args.cotagfn, freqs, args.plinkexe, args.threads, args.maxmem, 
        #args.quality) for h2 in h2s)
    df = pd.concat(dfs)
    plot(args.prefix, df, tar, ref, args.quality)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)    
    parser.add_argument('-R', '--reference', help=('Bed fileset prefix of the re'
                                                  'ference population. This is '
                                                  'usually European'))
    parser.add_argument('-T', '--target', help=('Bed fileset prefix of the targ'
                                                'et population. This is usually'
                                                ' African'))
    parser.add_argument('-L', '--labels', help=('Labels for the population in t'
                                                'he same order as the reference'
                                                ' and target. This option has t'
                                                'o be passed twice, once with e'
                                                'ach of the populations'),
                                          action='append')
    parser.add_argument('-d', '--cotagfn', help=('Filename tsv with cotag '
                                                 'results'), required=True) 
    parser.add_argument('-c', '--ncausal', help='number of causal variants', 
                        default=200, type=int)    
    parser.add_argument('-P', '--plinkexe')
    parser.add_argument('-q', '--quality', help=('type of plots (png, pdf)'), 
                        default='png')     
    parser.add_argument('-t', '--threads', default=False, action='store')
    parser.add_argument('-M', '--maxmem', default=False, action='store') 
    
    args = parser.parse_args()
    main(args)    