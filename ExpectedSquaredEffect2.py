"""
Compute the expected squared effect
"""

import shutil
import argparse
import numexpr
import numpy as np
from scipy.sparse import csr_matrix, triu
import pickle
import mmap
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utilities4cotagging import *
from itertools import permutations
from joblib import delayed, Parallel
import multiprocessing as mp
#import bigfloat
plt.style.use('ggplot')

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

#----------------------------------------------------------------------
def vs(B_hat_x, N, h2_l):
    """
    compute v_x
    """
    return (( N * B_hat_x ) / (2 * ( 1 - h2_l ) ))   
    
#----------------------------------------------------------------------
def integral_b(vs, mu, snps):
    """
    Compute the expected beta square
    :param vs: vector of v
    :param mu: mean
    :param all_markers: number of markers in genome
    :param N: Number of individuals
    :param snps: names of snps in order
    """
    snps = snps.tolist()
    exp = np.exp( np.power(vs,2) / (4 * mu) )
    exp.index = snps
    lhs = ( (2 * mu) + np.power(vs,2) ) / ( 4 * np.power(mu,2) )
    rhs = exp / exp.sum()
    vec = lhs * rhs
    vec.index = snps
    return vec

#----------------------------------------------------------------------
def expected_cov(integral, i, snps_in_locus):
    """
    Compute the expected covariance for one i
    """
    D_i = get_Ds(ld1, ld2, i, snps_in_locus)
    return D_i.p.dot(integral)
    
#----------------------------------------------------------------------
def get_centimorgans(bfile, rmap, plinkexe, chromosome=None, cmthresh=1, 
                     threads=1, maxmem=3000):
    """
    Compute centimorgan distances based on a recombination map. If 
    Chromosome is passed it will assume that this is the chromosome number
    replacing the current bim file. It will generate a new bimfile with 
    the centimorganinfo.
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

CHUNK_SIZE = 3000000
def getdf(df):
    return df
#----------------------------------------------------------------------
def read_one_parallel(fn, cpus):
    """
    Read one LD matrix but parallel
    """
    dtypes = {'SNP_A':str, 'SNP_B':str, 'D':float}
    reader = pd.read_table(fn, engine='c', delim_whitespace=True, 
                           usecols=['SNP_A', 'SNP_B', 'D'], dtype=dtypes, 
                           chunksize=CHUNK_SIZE)    
    res = Parallel(n_jobs=cpus)(delayed(getdf)(df) for df in reader)
    res = pd.concat(res)
    return res
#----------------------------------------------------------------------
def read_lds(refld, tarld, cpus=mp.cpu_count()):
    """
    read LD matrix
    """
    dtypes = {'SNP_A':str, 'SNP_B':str, 'D':float}
    print('Reading reference ld from %s' % refld)
    if not os.path.isfile('referenceLD.hdf'):
        refld = read_one_parallel(refld, cpus)
        #refld = pd.read_table(refld, engine='c', delim_whitespace=True, 
                              #usecols=['SNP_A', 'SNP_B', 'D'], dtype=dtypes,
                              #memory_map=True)
        refld.to_hdf('referenceLD.hdf', 'refld')
    else:
        refld = pd.read_hdf('referenceLD.hdf', 'refld')
    print('Reading target ld from %s' % tarld)
    if not os.path.isfile('targetLD.hdf'):  
        tarld = read_one_parallel(tarld, cpus)
        #tarld = pd.read_table(tarld, engine='c', delim_whitespace=True,
                              #usecols=['SNP_A', 'SNP_B', 'D'], dtype=dtypes,
                              #memory_map=True)
        tarld.to_hdf('targetLD.hdf', 'tarld')
    else:
        tarld = pd.read_hdf('targetLD.hdf', 'tarld')
    #intersect the LDs
    refsnps = set(refld.loc[:,['SNP_A', 'SNP_B']].unstack())
    tarsnps = set(tarld.loc[:,['SNP_A', 'SNP_B']].unstack())
    available_snps = list(refsnps.intersection(tarsnps))
    refld = refld[refld.SNP_A.isin(available_snps) & 
                  refld.SNP_B.isin(available_snps)]
    tarld = tarld[tarld.SNP_A.isin(available_snps) & 
                  tarld.SNP_B.isin(available_snps)]    
    return refld, tarld

#----------------------------------------------------------------------
def get_all_is(ld1, ld2, avail, integral):
    """
    Given a locus' SNPs, compute the expected covariance for all is
    """
    # Fetch D info for the locus
    query = ('((SNP_A == @avail) | (SNP_B == @avail))')
    ld1 = ld1.query(query)
    ld2 = ld2.query(query)
    # symetrize ld matrices
    D1 = pd.DataFrame(ld1[ld1.SNP_A.isin(avail) & ld1.SNP_B.isin(avail)].pivot(
        columns='SNP_B', index='SNP_A', values='D'), index=avail, 
                      columns=avail)#.fillna(0)
    D1 = csr_matrix(D1.values, shape=(len(avail), len(avail)))
    del ld1
    #D1.update(piv1)
    D1 = (triu(D1) + triu(D1,1).T)
    D1.setdiag(1)
    #D1 = pd.DataFrame((np.triu(D1) + np.triu(D1,1).T), index=D1.index, 
                      #columns=D1.columns)
    #np.fill_diagonal(D1.values, 1) 
    D2 = pd.DataFrame(ld2[ld2.SNP_A.isin(avail) & ld2.SNP_B.isin(avail)].pivot(
        columns='SNP_B', index='SNP_A', values='D'), index=avail, 
                      columns=avail)#.fillna(0)
    del ld2
    D2 = (triu(D2) + triu(D2,1).T)
    D2.setdiag(1)
    #D2.update(piv2)
    #D2 = pd.DataFrame((np.triu(D2) + np.triu(D2, 1).T), index=D2.index, 
                      #columns=D2.columns)
    #np.fill_diagonal(D2.values, 1)
    # compute the dot product for the locus
    ec_i = pd.DataFrame({'SNP':avail, 'ese': list(D1.multiply(D2).dot(integral)
                                                  )})
    #ec_i = (D1 * D2).dot(integral).to_frame(name='ese')
    #ec_i['SNP'] = ec_i.index.tolist() 
    return ec_i
    
#----------------------------------------------------------------------
def get_Ds(ld1, ld2, i, avail, integral):
    """
    given an LD matrix retun the D vectors Di for all Js in both pops
    """
    query = ('((SNP_A == @i) & (SNP_B == @avail)) | ((SNP_B == @i) & '
             '(SNP_A == @avail))')
    D1 = ld1.query(query)
    sub = D1[(D1.SNP_A == i) & (D1.SNP_B != i)]
    s1 = sub.D
    s1.index = sub.SNP_B
    sub = D1[(D1.SNP_A != i) & (D1.SNP_B == i)]
    st = sub.D
    st.index = sub.SNP_A
    s1 = s1.append(st)
    D2 = ld2.query(query)
    sub = D2[(D2.SNP_A == i) & (D2.SNP_B != i)]
    s2 = sub.D
    s2.index = sub.SNP_B
    sub = D2[(D2.SNP_A != i) & (D2.SNP_B == i)]
    st = sub.D
    st.index = sub.SNP_A
    s2 = s2.append(st)
    names = ['D1','D2']
    s1.name, s2.name = names
    D = pd.concat((s1[~s1.index.duplicated()], s2[~s2.index.duplicated()]), 
                  axis=1)
    D['p'] = D.D1 * D.D2
    D = pd.concat((D,integral), axis=1).dropna()
    return D.p.dot(D.BETA)

#----------------------------------------------------------------------
def ese(i, j, M, all_markers, N, h2_l, denominator, mu, ld1, ld2):
    """
    compute single expected value of i over js
    """
    b2 = (integral_b(j.BETA, M, all_markers, N, h2_l, mu) / 
          denominator )
    dp = get_D(ld1, i.SNP, j.SNP) * get_D(ld2, i.SNP, j.SNP)
    return dp * b2   

#----------------------------------------------------------------------
def per_locus(locus, sumstats, avh2, h2,  N, ld1, ld2):
    """
    compute the per-locus expectation
    """
    locus = sumstats[sumstats.SNP.isin(locus)].loc[:,['SNP', 'BETA']]
    #locus = locus.astype({'BETA':np.longfloat})
    locus.index = locus.SNP.tolist()
    M = locus.shape[0]
    h2_l = avh2 * M
    mu = ( (N /(2 * ( 1 - h2_l ) ) ) + ( M / ( 2 * h2 ) ) )
    vjs = (( N * locus.BETA ) / (2 * ( 1 - h2_l ) ))
    I = integral_b(vjs, mu, locus.SNP)
    del vjs
    expcovs = get_all_is(ld1, ld2, sorted(locus.SNP.tolist()), I)
    return expcovs
        
#----------------------------------------------------------------------
def transferability(args):
    """
    Execute trasnferability code
    """
    ld1, ld2 = read_lds(args.refld, args.tarld, cpus=int(args.threads))
    sumstats = pd.read_table(args.sumstats, delim_whitespace=True)
    avh2 = args.h2 / sumstats.shape[0]
    loci = get_centimorgans(args.reference, args.rmap, args.plinkexe, 
                            args.chrom, args.cm, args.threads, args.maxmem)
    with open('loci.pickle','wb') as L:
        pickle.dump(loci, L)
    #res = pd.DataFrame(columns=['SNP', 'ese'])
    N = mapcount('%s.fam' % args.reference)
    resfile = '%s_res.pickle' % args.prefix
    if not os.path.isfile(resfile):
        res = Parallel(n_jobs=int(args.threads))(delayed(per_locus)(
            locus, sumstats, avh2, args.h2, N, ld1, ld2) for locus in tqdm(
                loci.values(), total=len(loci)))    
        res = pd.concat(res)
        with open(resfile, 'wb') as R:
            pickle.dump(res, R)
    else:
        with open(resfile, 'rb') as R:
            res = pickle.load(R)        
    product, _ = smartcotagsort(args.prefix, res, column='ese')
    nsnps = product.shape[0]
    percentages = set_first_step(nsnps, 5, every=args.every)
    snps = np.around((percentages * nsnps) / 100).astype(int) 
    qfile = '%s.qfile' % args.prefix
    qrange= '%s.qrange' % args.prefix
    qr = gen_qrange(args.prefix, nsnps, 5, qrange, every=args.every)
    product.loc[:,['SNP', 'Index']].to_csv(qfile, sep=' ', header=False,
                                           index=False)   
    df = qrscore(args.plinkexe, args.target, args.sumstats, qrange, qfile, 
                 args.pheno, args.prefix, qr, args.maxmem, args.threads, 
                 'None', args.prefix)
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
    #parser.add_argument('-n', '--N', type=int, help='Number of individuals',
    #required=True)
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
    transferability(args)