"""
Compute the expected squared effect
"""

import shutil
import argparse
import numexpr
import numpy as np
#from collections import defaultdict
import matplotlib.pyplot as plt
from utilities4cotagging import *
plt.style.use('ggplot')

def vs(x, N, h2_l):
    return (( N * x ) / (2 * ( 1 - h2_l ) ))

def single_ese(v_j, mu, suma):
    """
    Compute the effect for a single marker j
    :param j: index in B_hat of variant being analized
    """
    #v_j = vs(j, N, h2_l)
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
    mu = ( (N /(2 * ( 1 - h2_l ) ) ) + ( M / ( 2 * h2 ) ) )
    suma = sum([vs(x, N, h2_l) for x in B_hat]) 
    ese = np.array([single_ese(vs(j, N, h2_l), mu, suma) for j in B_hat])
    return ese

def get_centimorgans(bfile, rmap, plinkexe, chromosome=None, 
                     cmthresh=1, threads=1, maxmem=3000):
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
    plink = ('%s --bfile %s --bim %s_new.bim --cm-map %s %d --make-just-bim '
             '--out %s --threads %d --memory  %d')
    plink = plink % (plinkexe, bfile, bfile, rmap, chromosome, prefix, threads,
                     maxmem)
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

def integrateB2(locus, sumstats, h2, N):
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
    res = pd.DataFrame(columns=['SNP', 'ese'])
    locus = sumstats[sumstats.SNP.isin(locus)]
    B_hat = locus.BETA
    for j in locus.itertuples():
        eses = ese_locus(h2, locus.shape[0], all_markers, N, B_hat)
        res = res.append(pd.DataFrame({'SNP':j.SNP, 'ese':eses}))
    return res
        
def expected_covariance(integral, cotagfn, sumstats, every=False):
    """
    Compute equation 4
    :param integral: vector with squared effects
    :param cotagfn: filename with cotagging score (THIS MIGT NOT BE IT!!)
    """
    cotags = pd.read_table(cotagfn, sep='\t')
    cotags = cotags.merge(integral,on='SNP')    
    cotags['product'] = cotags.Cotagging * cotags.ese
    return cotags

#----------------------------------------------------------------------
def read_lds(refld, tarld):
    """
    read LD matrix
    """
    refld = pd.read_table(refld, delim_whitespace=True)
    tarld = pd.read_table(tarld, delim_whitespace=True)
    return refld, tarld

#----------------------------------------------------------------------
def get_D(ld, i, j):
    """
    given an LD matrix retun D_ij
    """
    logic = ('((SNP_A == @i) & (SNP_B == @j)) | ((SNP_A == @j) & (SNP_B == @i))'
             ) 
    return ld.query(logic).D.iloc[0]
    
    
def main(args):
    """
    execute the code
    """
    loci = get_centimorgans(args.reference, args.rmap, args.plinkexe, 
                             chromosome=args.chrom, cmthresh=args.cm, 
                             threads=args.threads, maxmem=args.maxmem)
    integral = integrateB2(loci, args.sumstats, args.h2, args.N)
    res = expected_covariance(integral, args.cotagfile, args.sumstats)
    product, _ = smartcotagsort(args.prefix, res, column='product')
    nsnps = product.shape[0]
    percentages = set_first_step(nsnps, 5, every=args.every)
    snps = np.around((percentages * nsnps) / 100).astype(int) 
    qfile = '%s.qfile' % args.prefix
    qrange= '%s.qrange' % args.prefix
    qr = gen_qrange(args.prefix, nsnps, 5, qrange, every=args.every)
    product.loc[:,['SNP', 'Index']].to_csv(qfile, sep=' ', header=False,
                                           index=False)   
    df = qrscore(args.plinkexe, args.target, args.sumstats, qrange, qfile, 
                 args.pheno, args.prefix, qr, args.maxmem, args.threads, 'None', 
                 args.prefix)
    #get ppt results
    ppts=[]
    for i in glob('*.results'):
        three_code = i[:4]
        results = pd.read_table(i, sep='\t')
        R2 = results.nlargest(1, 'R2').R2.iloc[0]
        ppts.append((three_code, R2))
    ppts = sorted(ppts, key=lambda x: x[1], reverse=True)
    aest = [('0.5', '*'), ('k', '.')]
    f, ax = plt.subplots()
    df.plot.scatter(x='SNP kept', y='R2', alpha=0.5, c='purple', s=2, ax=ax,
                    label='Transferability', linestyle=':')
    for i, item in enumerate(ppts):
        pop, r2 = item 
        ax.axhline(r2, label='%s P + T Best' % pop, color=aest[i][0], ls='--',
                   marker=aest[i][1], markevery=10)
    plt.ylabel('$R^2$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s_transferability.pdf' % args.prefix)    
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-b', '--reference', required=True, 
                        help=('prefix of the bed fileset in reference'))  
    parser.add_argument('-g', '--target', required=True, 
                            help=('prefix of the bed fileset in target'))     
    parser.add_argument('-l', '--refld', required=True,
                        help=('plink LD matrix for the target population'))
    parser.add_argument('-d', '--tarld', required=True,
                        help=('plink LD matrix for the reference population'))
    parser.add_argument('-c', '--cotagfile', help=('Filename of the cotag tsv '
                                                   'file '))
    parser.add_argument('-s', '--sumstats', help='Filename of sumstats',
                        required=True)
    parser.add_argument('-n', '--N', type=int, help='Number of individuals',
    required=True)
    parser.add_argument('-H', '--h2', type=float, help='Heritability of trait',
                        required=True)
    parser.add_argument('-r', '--rmap', help='Path to recombination map',
                        required=True)
    parser.add_argument('-f', '--pheno', required=True, 
                        help=('Filename of the true phenotype of the target '
                              'population'))    
    parser.add_argument('-o', '--chrom', help=('Chromosome number, should match'
                                               ' the recombination map'),
                                         default=None, type=int)
    parser.add_argument('-C', '--cm', help=('Threshold for locus determination '
                                            'in centimorgans'),
                                      default=1)
    parser.add_argument('-P', '--plinkexe', required=True)
    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=3000, type=int)
    parser.add_argument('-y', '--every', action='store_true', default=False)
    args = parser.parse_args()
    main(args)