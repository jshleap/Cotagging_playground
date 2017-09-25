"""
Compute the Cov(g_i, Yp) and plot it
"""
import argparse
import numpy as np
import pandas as pd
import pickle as P
import socket
if 'abacus' in socket.gethostname():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
else:
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy import stats
from subprocess import Popen
from joblib import Parallel, delayed

#----------------------------------------------------------------------
def readFreq(fn):
    """
    read the plink frequency file (with --freq or --freqx) and compute the 
    variant variance
    """
    freq = pd.read_table(fn, delim_whitespace=True)
    freq['Var'] = freq.MAF * (1 - freq.MAF)
    return freq

#----------------------------------------------------------------------
def GWASnCOV(freqfn, gwasfn):
    """
    Read the GWAS file (fn; plink format), execute readFreq and compute
    """
    freq = readFreq(freqfn)
    gwas = pd.read_table(gwasfn, delim_whitespace=True)
    gwas['Cov'] = gwas.BETA * freq.Var
    gwas['-log(P)'] = -np.log(gwas.P)
    return gwas


#----------------------------------------------------------------------
def plot(subsetgwas, label, causal, out):
    """
    Cov(g_i, Y) vs -log(pval)
    """
    # Get indices where causal
    subsetgwas.reset_index(inplace=True)
    causalidx = subsetgwas[subsetgwas.SNP.isin(causal.SNP)].index
    abso = '$|Cov(g_{i},PRS_{%s})|$' % label
    subsetgwas[abso] = abs(subsetgwas.Cov)
    absB = '|$\beta_{%s}$|' % label
    subsetgwas[absB] = abs(subsetgwas.BETA)    
    # Plot Covariance
    x = '$-log(P)_{%s}$' % label
    y = '$Cov(g_{i},PRS_{%s})$' % label
    subsetgwas.rename(columns={'Cov': y , '-log(P)': x}, inplace=True)
    subsetgwas = subsetgwas.dropna()
    slope, intercept, r2, p_value, std_err = stats.linregress(
        subsetgwas.loc[:, x], subsetgwas.loc[:, y])  
    plt.figure()
    ax = subsetgwas.plot.scatter(x=x, y=y)
    ax.add_artist(AnchoredText('$R^{2} = %.3f $' % r2, 2))
    subsetgwas.loc[causalidx, :].plot.scatter(x=x, y=y, ax=ax, marker='*', s=2)
    plt.tight_layout()
    plt.savefig(out + '.png')
    plt.close()
    # Plot BETA
    slope, intercept, r2, p_value, std_err = stats.linregress(
            subsetgwas.loc[:, x], subsetgwas.loc[:, 'BETA'])    
    plt.figure()
    ax = subsetgwas.plot.scatter(x=x, y='BETA')
    ax.add_artist(AnchoredText('$R^{2} = %.3f $' % r2, 2))
    subsetgwas.loc[causalidx, :].plot.scatter(x=x, y=y, ax=ax, marker='*', s=2)
    plt.tight_layout()
    plt.savefig(out + '_beta.png')    
    # Plot Absolute value of Cov
    slope, intercept, r2, p_value, std_err = stats.linregress(
            subsetgwas.loc[:, x], subsetgwas.loc[:, abso])    
    plt.figure()
    ax = subsetgwas.plot.scatter(x=x, y=abso)
    ax.add_artist(AnchoredText('$R^{2} = %.3f $' % r2, 2))
    subsetgwas.loc[causalidx, :].plot.scatter(x=x, y=y, ax=ax, marker='*', s=2)
    plt.tight_layout()
    plt.savefig(out + '_absCov.png')     
    # Plot Absolute value of Beta
    slope, intercept, r2, p_value, std_err = stats.linregress(
            subsetgwas.loc[:, x], subsetgwas.loc[:, absB])    
    plt.figure()
    ax = subsetgwas.plot.scatter(x=x, y=absB)
    ax.add_artist(AnchoredText('$R^{2} = %.3f $' % r2, 2))
    subsetgwas.loc[causalidx, :].plot.scatter(x=x, y=y, ax=ax, marker='*', s=2)
    plt.tight_layout()
    plt.savefig(out + '_absBeta.png')     

#----------------------------------------------------------------------
def dotproduct(fn, top):
    """
    load the cotagging score and return the top scoring snps for comparison
    """
    with open(fn, 'rb') as F:
        dens, dots, merged = P.load(F)
    dp = pd.DataFrame({'SNP':list(dots[0].keys()), 
                       'Cotagging':list(dots[0].values())})
    d12 = pd.DataFrame({'SNP':list(dots[1].keys()), 
                       '$Tagging Score 1$':list(dots[1].values())})
    if top == 'All':
        return dp
    else:
        return dp.nlargest(top, 'Cotagging')


#----------------------------------------------------------------------
def clumped(fn):
    """
    Reads the clumped file and returns the SNPs
    """
    clump = pd.read_table(fn, delim_whitespace=True, header=None, 
                          names=['SNP', 'A1', 'beta'])
    return clump

#----------------------------------------------------------------------
def betaBeta(prefix, dp, gwas1, gwas2, nbins, labels, causals):
    """
    plot correlation of betas colorcoded by dp
    """
    # merge dataframes
    gwases = gwas1.merge(gwas2, on=['CHR', 'SNP', 'BP', 'A1', 'TEST'],
                         suffixes=tuple(['_%s' % x for x in labels]))
    merged = gwases.merge(dp, on='SNP').reset_index()
    # get the indiced of causals
    causalidx = merged[merged.SNP.isin(causals.SNP)].index        
    # sort merged dataframe by cotagging score
    merged.sort_values(by='Cotagging', inplace=True)
    # Get the normalizer
    N = Normalize(vmin=merged.Cotagging.min(), vmax=merged.Cotagging.max())
    merged['Cotagging Score'] = N(merged.Cotagging)
    # get cotag inidices
    cotagidx = merged[merged.loc[:, 'Cotagging Score'] > 0.6].index    
    # loop over chunks and plot coloring by beta
    #f, ax = plt.subplots()
    #for k, df in merged.groupby(np.arange(len(merged))//nbins):
    x , y = 'BETA_%s' % labels[0], 'BETA_%s' % labels[1]
    # compute the stats
    full = stats.linregress(merged.loc[:, x], merged.loc[:, y])
    truelr = stats.linregress(merged.loc[causalidx, x], merged.loc[causalidx, 
                                                                   y])    
    cotalr = stats.linregress(merged.loc[cotagidx, x], merged.loc[cotagidx, y])     
    plt.figure()
    ax = merged.plot.scatter(x=x, y=y, alpha=0.5, colormap='YlOrBr', 
                             c='Cotagging Score', colorbar=True, s=0.8)
    merged.loc[causalidx, :].plot.scatter(x=x, y=y, ax=ax, marker='*', s=5, 
                                          c='Cotagging Score', colorbar=False, 
                                          colormap='YlOrBr', alpha=0.8, 
                                          edgecolor='black', linewidth=0.1)
    label = ('$R^{2}_{full} = %.3f$\n$R^{2}_{True} = %.3f$ \n$R^{2}_{Cotag > '
             '0.6} = %.3f$') % (full.rvalue**2, truelr.rvalue**2, 
                                cotalr.rvalue**2)
    ax.add_artist(AnchoredText(label, 2))
    plt.tight_layout()
    plt.savefig(prefix + '_BetavsBeta.pdf')  
    plt.close()

#----------------------------------------------------------------------
def snpCrosPRSvsSnpWithinPRS(gwas1, gwas2, labels):
    """
    Plot the B_ig_i cross population vs B_ig_i within population
    """
    pass

#----------------------------------------------------------------------
def CovsnpPRS_phenovsbeta(gwas1, gwas2, pheno, labels):
    """
    Plot the B_ig_i cross population vs B_ig_i within population
    """
    # Read pheno file
    pass

#----------------------------------------------------------------------
def snpPRS(plinkexe, bedfile1, snpallelebetatuple, phenofn, freqfile, labels):
    """
    Compute the B_i*g_i from a single SNP
    """
    # write score to file
    with open('temp.score', 'w') as t:
        t.write(' '.join(snpallelebetatuple))
    line = '%s --bfile %s -score temp.score sum --pheno %s --allow-no-sex '
    line += '--keep-allele-order --read-freq %s --out temp'
    line = line % (plinkexe, bedfile1, phenofn, freqfile)
    # Execute plink
    exe = Popen(line, shell=True)
    exe.communicate()
    # read the profile and cleanup
    profile = pd.read_table('temp.profile', delim_whitespace=True)
    rm = Popen('rm temp*', shell=True)
    rm.communicate()
    snp = snpallelebetatuple[0]
    #cov = np.diag(profile.loc[:,['PHENO','SCORESUM']].cov(),k=1)[0]
    cov = np.diag(profile.loc[:,['PHENO','SCORE']].cov(),k=1)[0]
    covname = '$Cov(\beta_{i}^{%s}g_{i}^{%s}, Y)$' % tuple(labels)
    return pd.DataFrame([{'SNP': snp, covname: cov}])
    
#----------------------------------------------------------------------
def Cov_snpPRS_pheno(plinkexe, bedfile1, pheno1fn, gwas1, gwas2, labels, dp, 
                     freq1file, causals, cpus=-1):
    """
    Compute the covariance of the phenotype and snpPRS
    """
    gwas1['-log(P)'] = np.log(gwas1.P)
    gwas1 = gwas1.loc[ : , ['SNP', '-log(P)']]
    # get the snpPRS
    covname = '$Cov(\beta_{i}^{%s}g_{i}^{%s}, Y)$' % (labels[0], labels[1])
    #snpPRSes = pd.DataFrame([],columns=['SNP', covname])
    snpPRSes = pd.concat(Parallel(n_jobs=cpus)(delayed(snpPRS)(
        plinkexe, bedfile1, (i.SNP, i.A1, str(i.BETA)), pheno1fn, freq1file, 
        labels) for i in gwas2.itertuples()))     
    #for i in gwas2.itertuples():
        #thetuple = (i.SNP, i.A1, str(i.BETA))
        #snpPRSes.append(snpPRS(plinkexe, bedfile1, thetuple, pheno1fn, labels, 
                               #freq1file))
    snpPRSes.reset_index(inplace=True)
    # merge it with dp
    merged = snpPRSes.merge(dp, on='SNP').reset_index()
    merged = merged.merge(gwas1, on='SNP')
    # get the color scheme
    N = Normalize(vmin=merged.Cotagging.min(), vmax=merged.Cotagging.max())
    merged['Cotagging Score'] = N(merged.Cotagging)
    # get the indiced of causals
    causalidx = merged[merged.SNP.isin(causals.SNP)].index 
    # compute the stats
    x = '-log(P)'
    y = covname
    slope, intercept, r2, p_value, std_err = stats.linregress(merged.loc[:, x], 
                                                              merged.loc[:, y])     
    plt.figure()
    ax = merged.plot.scatter(x=x, y=y, alpha=0.5, colormap='YlOrBr', 
                             c='Cotagging Score', colorbar=True, s=0.8)
    merged.loc[causalidx, :].plot.scatter(x=x, y=y, ax=ax, marker='*', s=5, 
                                          c='Cotagging Score', colorbar=False, 
                                          colormap='YlOrBr', alpha=0.8, 
                                          edgecolor='black', linewidth=0.05)
    ax.add_artist(AnchoredText('$R^{2} = %.3f $' % r2, 2))
    plt.tight_layout()
    plt.savefig(prefix + '_CovsnpPRSvslogP.pdf')    

        
#----------------------------------------------------------------------
def main(args):
    """
    Executes the code
    """
    # get causal snps
    causals = pd.read_table(args.causalfn, delim_whitespace=True, header=None, 
                            names=['SNP', 'A1', 'True_beta'])    
    # Get GWAS and compute the covariance
    gwas1 = GWASnCOV(args.freq, args.gwasfn1)
    gwas2 = GWASnCOV(args.freq, args.gwasfn2)
    # get the subset of snps from P+T
    clump = clumped(args.clumped)
    cSNPs = clump.SNP
    # get the subset of snps from dotproduct
    top = cSNPs.shape[0]
    if args.cotagfn.endswith('.dotproduct'):
        cotags = dotproduct(args.cotagfn, top)
    else:
        cotagDF = pd.read_table(args.cotagfn, sep='\t')#delim_whitespace=True)
        #pd.read_pickle(args.cotagfn)
        cotags = cotagDF.nlargest(top, 'Cotagging')
    cotSNP = cotags.SNP
    # plot beta vs beta
    betaBeta(args.prefix, cotags, gwas1, gwas2, 10, args.labels, causals)
    # plot covariance of PRS and phenotype vs logpval
    Cov_snpPRS_pheno(args.plinkexe, args.bedfile1, args.pheno1fn, gwas1, gwas2, 
                     args.labels, cotags, causals, args.freq, args.cpus)
    # plot cov vs -logp for P + T
    #out = '%s_PpT' % args.prefix
    #plot(gwas1[gwas1.SNP.isin(cSNPs)], args.labels[0], causals, out)
    # plot cov vs -logp for P + T
    #out = '%s_cotag' % args.prefix
    #plot(gwas1[gwas1.SNP.isin(cotSNP)], args.labels[0], causals, out)
    

if __name__ == '__main__':   
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)    
    parser.add_argument('-f', '--freq', help='Filename of the frequencies', 
                        required=True)
    parser.add_argument('-g', '--gwasfn1', help='Filename of the GWAS of pop1',
                        required=True)
    parser.add_argument('-i', '--gwasfn2', help='Filename of the GWAS of pop2',
                            required=True)    
    parser.add_argument('-c', '--clumped', help='Filename of the clumped result',
                        required=True)    
    parser.add_argument('-d', '--cotagfn', help='Filename pickle with cotag ' +
                        'results', required=True)    
    parser.add_argument('-l', '--labels', help='name of the populations being' + 
                        ' in order of pop1 pop2',# default=['AFR', 'EUR'], 
                        action='append')  
    parser.add_argument('-C', '--causalfn', help='Filename of causal/score', 
                        required=True) 
    parser.add_argument('-n', '--plinkexe', help='path to plink executable', 
                        required=True)
    parser.add_argument('-b', '--bedfile1', help='Filename of bedfile for ' + 
                        'population 1', required=True)     
    parser.add_argument('-P', '--pheno1fn', help='Filename of phenotype file ' +
                        'for pop 1', required=True)    
    parser.add_argument('-u', '--cpus', help='Number of processors', default=-1,
                        type=int)     
    args = parser.parse_args()
    main(args)     