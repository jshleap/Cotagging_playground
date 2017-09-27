'''
Null model of SNP pruning
'''
import os
import shutil
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from functools import reduce
from dotproductV2 import norm
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from scipy.stats import linregress
plt.style.use('ggplot')


#----------------------------------------------------------------------
def execute(line):
    """
    Execute a given line using Popen.
    :param str line: Line to be executed
    :returns: stderr and stdout
    """
    exe = Popen(line, shell=True, stderr=PIPE, stdout=PIPE)
    o, e = exe.communicate()
    return o.strip(), e.strip() 

def read_pheno(phenofile):
    """
    Read the phenotype file
    """
    pheno = pd.read_table(phenofile, delim_whitespace=True, header=None,
                          names=['FID', 'IID', 'pheno'])
    return pheno

def read_scored(profilefn, phenofile):
    """
    Read the profile file a.k.a. PRS file or scoresum
    """
    d = r'$\sum(Y_{AFR} - \widehat{Y}_{AFR|EUR})^{2}$'
    sc = pd.read_table(profilefn, delim_whitespace=True)
    pheno = pd.read_table(phenofile, delim_whitespace=True, header=None, names=[
    'FID', 'IID', 'pheno'])
    sc = sc.merge(pheno, on=['FID', 'IID'])
    #sc[d] = (sc.PHENO - sc.SCORESUM)**2
    sc[d] = (sc.pheno - sc.SCORE)**2
    #lr = linregress(sc.PHENO, sc.SCORESUM)
    lr = linregress(sc.pheno, sc.SCORE)
    return sc, d, lr

def read_scored_qr(profilefn, phenofile, kind, nsnps, tar_label, ref_label):
    """
    Read the profile file a.k.a. PRS file or scoresum
    """
    d = r'$\sum(Y_{%s} - \widehat{Y}_{%s|%s})^{2}$' % (tar_label, tar_label, 
                                                       ref_label)
    sc = pd.read_table(profilefn, delim_whitespace=True)
    pheno = pd.read_table(phenofile, delim_whitespace=True, header=None, names=[
    'FID', 'IID', 'pheno'])
    sc = sc.merge(pheno, on=['FID', 'IID'])
    err = sum((sc.pheno - sc.SCORE)**2)
    lr = linregress(sc.pheno, sc.SCORE)
    dic = {'SNP kept':nsnps, '%s_%s' % (d, kind): err, 
           '-log(P)_%s' % kind : -np.log10(lr.pvalue), 
           r'$R^{2}$_%s' % kind : lr.rvalue**2, 'Slope_%s' % kind: lr.slope}
    return dic

def read_gwas(gwasfile, cotagfile):
    """
    Read the GWAS (a.k.a. summary stats) in the reference population and merge
    it with the cotagging info
    """
    cotag = pd.read_table(cotagfile, sep='\t')
    gwas = pd.read_table(gwasfile, delim_whitespace=True)
    gwas = gwas.sort_values('BP').reset_index()
    return gwas.merge(cotag, on='SNP')

def scoreit(bfile, gwasfn, outpref, phenofile, plinkexe):
    """
    compute the PRS or profile given <score> betas and <bfile> genotype
    """
    #score = ('%s --bfile %s --score %s.score sum --allow-no-sex '
    score = ('%s --bfile %s --extract %s.extract --score %s 2 4 7 header --allo'
             'w-no-sex --keep-allele-order --pheno %s --out %s')
    score = score%(plinkexe, bfile, outpref, gwasfn, phenofile, outpref)
    #print('executing %s' % score)
    o,e = execute(score)  

def smartcotagsort(prefix, gwaswcotag, column='Cotagging'):
    """
    perform a 'clumping' based on Cotagging score, but retain all the rest in 
    the last part of the dataframe
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
        for key in tqdm(keys,total=len(keys)):
            df = grouped.get_group(key)
            sorteddf = sorteddf.append(df.loc[df.index[0],:])
            tail = tail.append(df.loc[df.index[1:],:])
        beforetail = sorteddf.shape[0]
        df = sorteddf.append(tail.sample(frac=1)).reset_index(drop=True)
        df['Index'] = df.index.tolist()
        with open(picklefile, 'wb') as F:
            pickle.dump((df,beforetail), F)
    return df, beforetail

def set_first_step(nsnps, step):
    """
    Define the range starting by adding one snp up the the first step
    """
    onesnp = 100/nsnps
    initial = np.arange(onesnp, step + onesnp, onesnp)
    rest = np.arange(step + onesnp, 100 + step, step)
    full = np.concatenate((initial, rest))
    if full[-1] < 100:
        full[-1] = 100
    return ['%.2f' % x for x in full]


def subsetter(sortedcotag, sortedtagT, sortedtagR, step, clumped=None):
    """
    Create the file with the SNP subset
    """
    assert sortedcotag.shape[0] == sortedtagR.shape[0]
    nsnp = sortedcotag.shape[0]
    if clumped is not None:
        clumped = clumped[clumped.SNP.isin(sortedcotag.SNP)]
    tagTidx = np.array(sortedtagT.index)
    tagRidx = np.array(sortedtagR.index)
    cotaidx = np.array(sortedcotag.index)
    randidx = np.array(sortedcotag.index.copy(deep=True))
    np.random.shuffle(randidx)
    for i in set_first_step(nsnp, step):
        n = min(int(round(nsnp * (float(i)/100.))), sortedcotag.shape[0])
        rand = sortedcotag.loc[sorted(randidx[:n]),:].SNP
        cota = sortedcotag.loc[sorted(cotaidx[:n]),:].SNP
        tagT = sortedtagT.loc[sorted(tagTidx[:n]),:].SNP
        tagR = sortedtagR.loc[sorted(tagRidx[:n]),:].SNP
        assert ~(rand.tolist() == cota.tolist())
        assert ~(tagT.tolist() == cota.tolist())
        assert ~(rand.tolist() == tagT.tolist())
        assert ~(tagR.tolist() == tagT.tolist())
        assert rand.shape[0] == cota.shape[0]
        assert rand.shape[0] == tagT.shape[0]
        assert tagT.shape[0] == tagR.shape[0]
        if clumped is not None:
            clum = clumped[:n]
            assert cota.shape[0] == n
            assert rand.shape[0] == n
            assert tagT.shape[0] == n
            assert tagR.shape[0] == n
            assert clum.shape[0] == n
            yield i, cota, rand, tagT, tagR, clum, n
        else:
            yield i, cota, rand, tagT, tagR, n

def subsetter_qrange(prefix, sortedcota, sortedtagT, sortedtagR, step,
                     clumped=None):
    """
    Create the files to be used in q-score-range. It will use the index of the
    sorted files as thresholds
    """  
    assert sortedcota.shape[0] == sortedtagR.shape[0]
    nsnps = sortedcota.shape[0]  
    randomtagg = sortedcota.copy()
    idxs = randomtagg.Index.tolist()
    np.random.shuffle(idxs)
    randomtagg['Index'] = idxs
    qrange = '%s.qrange' % prefix
    percentages = set_first_step(nsnps, step)
    order = ['label', 'Min', 'Max']
    qr = pd.DataFrame({'label':percentages, 'Min':np.zeros(len(percentages)),
                      'Max':(np.array(percentages, dtype=float)*(nsnps/100)
                             ).astype(int)}).loc[:, order]
    qr.to_csv(qrange, header=False, index=False, sep =' ')    
    qfile = '%s_%s.qfile'    
    c = qfile % (prefix, 'cotag')
    t = qfile % (prefix, 'tagt')
    r = qfile % (prefix, 'tagr')
    a = qfile % (prefix, 'rand')
    sortedcota.loc[:,['SNP', 'Index']].to_csv(c, sep=' ', header=False, 
                                              index=False)
    sortedtagT.loc[:,['SNP', 'Index']].to_csv(t, sep=' ', header=False, 
                                              index=False)
    sortedtagR.loc[:,['SNP', 'Index']].to_csv(r, sep=' ', header=False, 
                                              index=False)
    randomtagg.loc[:,['SNP', 'Index']].to_csv(a, sep=' ', header=False,
                                              index=False)    
    out = (qr, c, t, r, a) 
    if clumped is not None:
        p = qfile % (prefix, 'clum')
        out += (p,)
        clumped = clumped[clumped.SNP.isin(sortedcota.SNP)]  
        clumped.loc[:,['SNP', 'Index']].to_csv(p, sep=' ', header=False, 
                                               index=False)
    return out 
    
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
        
def process_profile(bfile, gwasfn, profilefn, dataframe, phenofile, plinkexe):
    """
    produce and process the profile file
    """
    if not os.path.isfile(profilefn):
        fn = profilefn[: profilefn.rfind('.')]
        dataframe.to_csv('%s.extract' % fn, sep=' ', index=False, header=False)
        scoreit(bfile, gwasfn, fn, phenofile, plinkexe)
    df, col, lr = read_scored(profilefn, phenofile)
    log, rv, slope = -np.log10(lr.pvalue), lr.rvalue**2, lr.slope    
    error = df.loc[:, col].sum()    
    return  df, col, log, rv, slope, error

def process_profile_qr(bfile, gwasfn, qrdf, dataframe, phenofile, plinkexe
                       ):
    """
    produce and process the profile file with the ---q-score-range
    """
    
    if not os.path.isfile(profilefn):
        fn = profilefn[: profilefn.rfind('.')]
        dataframe.to_csv('%s.extract' % fn, sep=' ', index=False, header=False)
        scoreit(bfile, gwasfn, fn, phenofile, plinkexe)
    df, col, lr = read_scored(profilefn, phenofile)
    log, rv, slope = -np.log10(lr.pvalue), lr.rvalue**2, lr.slope    
    error = df.loc[:, col].sum()    
    return  df, col, log, rv, slope, error
   
def prunebypercentage(prefix, bfile, gwasfn, phenofile, sortedcotag, sortedtagT,
                      sortedtagR, plinkexe, clumped=None, step=1, causal=None):
    """
    Execute the prunning in a range from <step> to 100 with step <step> (in %)
    :param str gwasfn: filename of the summary stats to get the betas from
    """
    if os.path.isfile('pbp.pickle'):
        with open('pbp.pickle', 'rb') as f:
            merge, colc = pickle.load(f)
    else:
        rows = []
        rappend = rows.append
        print('Performing prunning ...')
        oriidx=np.array([])
        for tup in tqdm(subsetter(sortedcotag, sortedtagT, sortedtagR, step, 
                                  clumped)):       
            if causal is not None:
                cause = [all(causal.SNP.isin(x)) for x in tup[1:-1]]
            else:
                cause = [False]*len(tup[1:-1])
            if len(tup) == 6:
                i, cota, rand, tagsT, tagsR, number = tup
            else:
                i, cota, rand, tagsT, tagsR, clum, number = tup
            i = float(i)
            fn = '%s_%.2f' % (prefix, i)
            # Process cotags
            profilecot = '%s_cotag.profile' % fn
            dfc, colc, logc, rvc, slopec, errorC = process_profile(
                bfile, gwasfn, profilecot, cota, phenofile, plinkexe)
            # Process TAGS
            profiletagT = '%s_tagt.profile' % fn
            dft, colt, logt, rvt, slopet, errorT = process_profile(
                        bfile, gwasfn, profiletagT, tagsT, phenofile, plinkexe)   
            profiletagR = '%s_tagr.profile' % fn
            dftr, coltr, logtr, rvtr, slopetr, errorTr = process_profile(
                bfile, gwasfn, profiletagR, tagsR, phenofile, plinkexe)
            # Process Random   
            profilerand = '%s_rand.profile'% fn
            dfr, colr, logr, rvr, sloper, errorR = process_profile(
                        bfile, gwasfn, profilerand, rand, phenofile, plinkexe) 
            # Process Clumped if there are
            if clumped is not None:
                profileclu = '%s_clum.profile' % fn
                dfclu, colclu, logclu, rvclu, slopeclu,errorClu=process_profile(
                    bfile, gwasfn, profileclu, clum, phenofile, plinkexe)        
                # Store results
                rappend({'SNP kept':number, colc + '_cotag':errorC, 
                         '%s_tagt' % colt: errorT, '%s_tagr' % coltr: errorTr, 
                         colr + '_rand':errorR, '-log(P)_rand':logr,
                         '-log(P)_tagt':logt, '-log(P)_cotag':logc, 
                         '-log(P)_tagr':logtr,r'$R^{2}$_rand':rvr, 
                         r'$R^{2}$_tagt':rvt, r'$R^{2}$_tagr':rvtr,
                         r'$R^{2}$_cotag':rvc, 'Slope_cotag':slopec, 
                         'Slope_tagr':slopetr,'Slope_rand':sloper,
                         'Slope_tagt':slopet, 'Slope_clum': slopeclu, 
                         r'$R^{2}$_clum':rvclu, '-log(P)_clum':logclu,
                         '%s_clum' % colclu: errorClu, 'All_causal_cota': cause[
                             0], 'All_causal_rand': cause[1], 
                         'All_causal_tagt': cause[2], 'All_causal_tagr': cause[
                             3], 'All_causal_clum': cause[
                             4]})
                
            else:   
                # Store results
                rappend({'SNP kept':number, colc + '_cotag':errorC, 
                         '%s_tagt' % colt: errorT, '%s_tagr' % coltr: errorTr, 
                         colr + '_rand':errorR, '-log(P)_rand':logr,
                         '-log(P)_tagt':logt, '-log(P)_cotag':logc, 
                         '-log(P)_tagr':logtr,r'$R^{2}$_rand':rvr, 
                         r'$R^{2}$_tagt':rvt, r'$R^{2}$_tagr':rvtr,
                         r'$R^{2}$_cotag':rvc, 'Slope_cotag':slopec, 
                         'Slope_tagr':slopetr,'Slope_rand':sloper,
                         'Slope_tagt':slopet, 'All_causal_cota': cause[
                             0], 'All_causal_rand': cause[1], 
                         'All_causal_tagt': cause[2], 'All_causal_tagr': cause[
                             3]})
        merge = pd.DataFrame(rows)    
        merge['Percentage of SNPs used'] = (merge.loc[:, 'SNP kept']/merge.loc[
            :, 'SNP kept'].max()) * 100
        merge.to_csv('%s_merged.tsv' % prefix, sep='\t', index=False)
        cleanup()
        with open('pbp.pickle', 'wb') as f:
            pickle.dump((merge, colc), f)
    return merge, colc

def prunebypercentage_qr(prefix, bfile, gwasfn, phenofile, sortedcotag, 
                         sortedtagT, sortedtagR, plinkexe, clumped=None, step=1,
                         causal=None, tar_label='AFR', ref_label='EUR'):
    """
    Execute the prunning in a range from <step> to 100 with step <step> (in %)
    scoring using --q-score-ragne
    :param str gwasfn: filename of the summary stats to get the betas from
    """
    col = r'$\sum(Y_{%s} - \widehat{Y}_{%s|%s})^{2}$' % (tar_label, tar_label,
                                                         ref_label)
    frac_snps = sortedcotag.shape[0]/100
    if os.path.isfile('pbp.pickle'):
        with open('pbp.pickle', 'rb') as f:
            merge, colc = pickle.load(f)
    else:
        print('Performing prunning ...')
        out = subsetter_qrange(prefix, sortedcotag,sortedtagT, sortedtagR, step,
                               clumped=clumped) 
        qr = out[0]
        qrange = '%s.qrange' % prefix
        frames = []
        for qfile in out[1:]:
            suf = qfile[qfile.find('_') +1 : qfile.rfind('.')]
            ou = '%s_%s' % (prefix, suf)
            score = ('%s --bfile %s --score %s 2 4 7 header --q-score-range %s '
                     '%s --allow-no-sex --keep-allele-order --pheno %s --out %s'
                     )
            score = score%(plinkexe, bfile, gwasfn, qrange, qfile,  phenofile, 
                           ou)
            o,e = execute(score)       
            df = pd.DataFrame([read_scored_qr('%s.%s.profile' % (ou, x.label), 
                                              phenofile, suf, 
                                              round(float(x.label) * frac_snps), 
                                              tar_label, ref_label)
                               for x in qr.itertuples()])
            frames.append(df)
        merge = reduce(lambda x, y: pd.merge(x, y, on = 'SNP kept'), frames)
        merge['Percentage of SNPs used'] = (merge.loc[:, 'SNP kept']/merge.loc[
            :, 'SNP kept'].max()) * 100
        merge.to_csv('%s_merged.tsv' % prefix, sep='\t', index=False)
        cleanup()
        with open('pbp.pickle', 'wb') as f:
            pickle.dump((merge, col), f)
    return merge, col

def fix_clumped(clumped, allsnps):
    """
    If the clump file does not have all the SNPS put them in the tail
    """
    rest = allsnps[~allsnps.isin(clumped.SNP)]
    df = pd.concat((clumped.SNP,rest)).reset_index(drop=False)
    df['Index'] = df.index.tolist()
    return df

def includePPT(ppt, lab, col, ax, x, color):
    if not 'Percentage of SNPs used' in ppt.columns:
        ppt['Percentage of SNPs used'] =  (ppt.loc[:, 'SNP kept'
                                                   ]/ppt.loc[
        :, 'SNP kept'].max()) * 100 
    if col == '-log(P)':
        ppt['-log(P)'] = -np.log10(ppt.pval)
    elif col == r'$R^{2}$':
        ppt.rename(columns={'pR2':r'$R^{2}$'}, inplace=True)
    ppt.plot.scatter(x=x, y=col, label='P+T %s' % lab, s=10, c=color, 
                     marker='$%s$' % lab[0], ax=ax, alpha=0.5)    

def parse_sort_clump(fn, allsnps, ppt=None):
    """
    Parse and sort clumped file
    """
    if fn == 'auto':
        fn = '%s.clumped' % ppt
    try:
        df = pd.read_table(fn, delim_whitespace=True)
    except FileNotFoundError:
        fn = '.'.join(np.array(fn.split('.'))[[0,1,-1]])
        df = pd.read_table(fn, delim_whitespace=True)
    SNPs = df.loc[:,'SP2']
    tail = [x.split('(')[0] for y in SNPs for x in y.split(',') if x.split('(')[
        0] != 'NONE']
    full = pd.DataFrame(df.SNP.tolist() + tail, columns=['SNP'])
    return fix_clumped(full, allsnps)

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
        if isinstance(ppt, str):
            merge.plot.scatter(x=x, y=col+'_clum', label='Sorted Clump', ax=ax, 
                               c='k', s=2, alpha=0.5)             
        elif ppt is not None:
            for ppt, lab, color in ppt:
                includePPT(ppt, lab, col, ax, x, color)
        
    if vline:
        ax.axvline(vline, c='0.5', ls='--')
    if hline:
        ax.axhline(hline, c='0.5', ls='--')        
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig('%s_error.%s' % (prefix, plottype)) 
    plt.close()

def main(args):
    """
    execute the code  DEPRECATED Will execute directly in linearpipeline  
    """
    #gwas = read_gwas(args.gwas, args.cotagfile)
    cotag = pd.read_table(args.cotagfile, sep='\t')
    if not os.path.isfile('%s_merged.tsv' % args.prefix):
        merge, col = prunebypercentage(args.prefix, args.bedfile, args.gwas, 
                                       args.pheno, cotag, cotag, args.plinkexe, 
                                       step=args.step
                                       )
    else:
        merge = pd.read_table('%s_merged.tsv' % args.prefix, sep='\t')
        col = merge.columns[1][:merge.columns[1].rfind('_')]
    plotit(args.prefix, merge, col)
    plotit(args.prefix+'_pval', merge, '-log(P)')
    plotit(args.prefix+'_rval', merge, r'$R^{2}$')

if __name__ == '__main__':   
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-g', '--gwas', help='Filename of gwas results in the' +
                        ' reference population', required=True)
    parser.add_argument('-b', '--bedfile', help='prefix of the bed fileset for'+
                        ' the target population', required=True)
    parser.add_argument('-f', '--pheno', help='filename of the true phenotype' +
                        ' of the target population', required=True)    
    parser.add_argument('-P', '--plinkexe', default='~/Programs/plink_mac/plink'
                        )    
    parser.add_argument('-c', '--cotagfile', help='Filename of the cotag tsv ' +
                        'file ')
    parser.add_argument('-s', '--step', help='Step in the percentage range to' +
                        ' explore. By deafult is 1', default=1, type=float)    
    args = parser.parse_args()
    main(args)    