"""
Compute the expected squared effect
"""
import shutil
import numpy as np
#from collections import defaultdict
from utilities4cotagging import *

def single_ese(mu, B_hat, v_j, suma):
    """
    Compute the effect for a single marker j
    :param j: index in B_hat of variant being analized
    """
    ese = ((( 2 * mu ) + ( v_j**2 )) / ( 4 * ( mu**2 ))) 
    ese *= ( np.exp(( v_j**2 ) / ( 4 * mu) ))
    return ese/suma
    
def ese_locus(h2_total, lenght_locus, all_markers, N, B_hat):
    """
    Compute the expected squared effect
    
    :param h2_total: total heritability of the trait
    :param lenght_locus: number of markers in locus
    :param all_markers: number of markers in genome
    :param N: Number of individuals
    :param B_hat: vector or estimated betas
    
    """
    M = lenght_locus
    h2 = h2_total
    avh2 = h2 / all_markers
    h2_l = avh2 * M
    vs = lambda x: ( N*B_hat[x] ) / (2 * ( 1 - h2_l ) )
    v_j = vs[j]
    mu = ( (N /(2 * ( 1 - h2_l ) ) ) + ( M / ( 2 * h2 ) ) )
    suma = sum([vs[x] for x in range(B_hat)]) 
    ese = np.array([single_ese(mu, B_hat, v_j, suma) for j in B_hat])
    return ese

def get_centimorgans(bfile, rmap, plinkexe, chromosome=None, 
                     cmthresh=1):
    """
    Compute centimorgan distances based on a recombination map. If Chromosome
    is passed it will assume that this is the chromosome number replacing 
    the current bim file. It will generate a new bimfile with the centimorgan
    info.
    :param bfile: prefix for plink fileset
    :param rmap: filename of the recombination map
    :param plinkexe: path to plink executable
    :param chromosome: chromosome being analysed. It shoudl match the rmap
    :param cmthresh: threshold for centimorgan grouping
    """
    prefix = '%s_cm' % os.path.split(bfile)[-1]
    if chromosome is not None:
        with open('%s.bim' % bfile) as F:
            curr_chr = int(F.readline().split()[0])
        gsub = '{gsub("%d","%d",$1)}1' % (curr_chr, chromosome)
        awk = "awk '%s' OFS='\t' %s.bim > %s_new.bim" % (gsub, bfile, bfile)
        o, e = executeLine(awk)
    else:
        shutil.copyfile('%s.bim', '%_new.bim')
    # Get the cm distance
    plink = '%s --bfile %s --bim %s_new --cm-map %s %d --make-just-bim --out %s'
    plink = plink % (plinkexe, bfile, bfile, rmap, chromosome, prefix)
    o,e = executeLine(plink)
    Bnames = ['CHR', 'SNP', 'cM', 'BP', 'A1', 'A2']
    bim = pd.read_table('%s.bim'%(prefix), delim_whitespace=True, header=None,
                        names=Bnames)
    bim['diffe']= bim.cM - bim.cM[0]
    # get the locus
    locus={}
    curr=0
    for i in range(cmthresh, int(bim.diffe.iloc[-1])+1, cmthresh):
        diff = bim[bim.diffe <= i].iloc[curr:]
        curr = diff.index[-1]+1
        first_snp = diff.iloc[0].SNP
        locus[first_snp] = diff.SNP
    return locus

def integrateB2(loci, sumstats, h2, N):
    """
    execute the integral (equation 11)
    :param dict: dictionary with snps per locus 
    :param sumstats: dataframe or string with the summary statistics
    :param h2: float with the total heritability of the trait
    :param N: Number of individuals a.k.a. sample size
    """
    if isinstance(sumstats, str):
        sumstats = pd.read_table(sumstats, delim_whitespace=True)
    all_markers = sumstats.shape[0]
    res = pd.DataFrame(columns='SNP', 'ese')
    for js in loci.values():
        locus = sumstats[sumstats.SNP.isin(js)]
        B_hat = locus.BETA
        ese_locus(h2, locus.shape[0], all_markers, N, B_hat)
        
        
        