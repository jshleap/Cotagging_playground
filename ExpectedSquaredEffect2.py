"""
Compute the expected squared effect
"""

import shutil
import argparse
import numexpr
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utilities4cotagging import *
from itertools import permutations
from joblib import delayed, parallel
#import bigfloat
plt.style.use('ggplot')

#----------------------------------------------------------------------
def vs(B_hat_x, N, h2_l):
    """
    compute v_x
    """
    return (( N * B_hat_x ) / (2 * ( 1 - h2_l ) ))
    
    
#----------------------------------------------------------------------
def integral(vs, mu, snps):
    """
    Compute the expected beta square
    :param vs: vector of v
    :param mu: mean
    :param all_markers: number of markers in genome
    :param N: Number of individuals
    :param snps: names of snps in order
    """
    exp = np.exp( (vs**2) / (4 * mu) )
    lhs = ( (2 * mu) + (vs**2) ) / ( 4 * (mu**2) )
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

#----------------------------------------------------------------------
def read_lds(refld, tarld):
    """
    read LD matrix
    """
    if not os.path.isfile('referenceLD.pickle'):
        refld = pd.read_table(refld, delim_whitespace=True)
        refld.to_pickle('referenceLD.pickle')#, compression='gzip')
    else:
        refld = pd.read_pickle('referenceLD.pickle', compression='gzip')
    if not os.path.isfile('targetLD.pickle'):
        tarld = pd.read_table(tarld, delim_whitespace=True)
        tarld.to_pickle('targetLD.pickle')#, compression='gzip')
    else:
        tarld = pd.read_pickle('targetLD.pickle')#, compression='gzip')
    return refld, tarld

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
    D = pd.concat((s1[~s1.index.duplicated()],s2[~s2.index.duplicated()]), 
                  axis=1)
    D['p'] = D.D1 * D.D2
    D = pd.concat((D,integral), axis=1).dropna()
    return D.p.dot(D.BETA)

#----------------------------------------------------------------------
def ese(i, j, M, all_markers, N, h2_l, denominator, mu, ld1, ld2):
    """
    compute single expected value of i over js
    """
    b2 = (integral(j.BETA, M, all_markers, N, h2_l, mu) / 
          denominator )
    dp = get_D(ld1, i.SNP, j.SNP) * get_D(ld2, i.SNP, j.SNP)
    return dp * b2   

#----------------------------------------------------------------------
def per_locus(locus, sumstats, avh2, h2,  N, ld1, ld2):
    """
    compute the per-locus expectation
    """
    locus = sumstats[sumstats.SNP.isin(locus)].reset_index(drop=True)
    locus = locus.astype({'BETA':np.longfloat})
    M = locus.shape[0]
    h2_l = avh2 * M
    mu = ( (N /(2 * ( 1 - h2_l ) ) ) + ( M / ( 2 * h2 ) ) )
    vs = np.exp( ((( N * locus.BETA ) / (2 * ( 1 - h2_l ) ))**2)/(4 * mu) )
    vs.index = locus.SNP.tolist()
    I = integral(vs, mu, locus.SNP)
    expcovs = [get_Ds(ld1, ld2, i, locus.SNP.tolist(), I) for i in locus.SNP]
    return pd.DataFrame({'SNP':locus.SNP, 'ese':expcovs})
        
#----------------------------------------------------------------------
def transferability(args):
    """
    Execute trasnferability code
    """
    ld1, ld2 = read_lds(args.refld, args.tarld)
    sumstats = pd.read_table(args.sumstats, delim_whitespace=True)
    h2 = args.h2
    all_markers = sumstats.shape[0]
    avh2 = h2 / all_markers
    loci = get_centimorgans(args.reference, args.rmap, args.plinkexe, 
                            args.chrom, args.cm, args.threads, args.maxmem)
    with open('loci.pickle','wb') as L:
        pickle.dump(loci, L)
    #res = pd.DataFrame(columns=['SNP', 'ese'])
    N = args.N
    res = Parallel(n_jobs=int(args.threads))(delayed(per_locus)(
        locus, sumstats, avh2, h2, N, ld1, ld2) for index, locus in tqdm(
            loci.items(), total=len(loci)))    
    
    #for index, locus in tqdm(loci.items(), total=len(loci)):
        ##im = np.zeros((len(locus), len(locus)))
        #locus = sumstats[sumstats.SNP.isin(locus)].reset_index(drop=True)
        #M = locus.shape[0]
        #print('Processing locus %s with M=%d' %(index, M))
        #h2_l = avh2 * M
        #mu = ( (N /(2 * ( 1 - h2_l ) ) ) + ( M / ( 2 * h2 ) ) )
        #vs = np.exp(((( N * locus.BETA ) / (2 * ( 1 - h2_l ) ))**2)/(4 * mu))
        #I = integral(vs, mu, locus.SNP)
        #expcovs = [get_Ds(ld1, ld2, i, locus.SNP) for i in locus.SNP]
        #denominator = np.sum([np.exp((vs(k, N, h2_l)**2)/(4 * mu)) for k in 
        #                      locus.BETA])
        #if not os.path.isfile('vec_%s.pickle'%index):
            #vec = [ese(i, j, M, all_markers, N, h2_l, denominator, mu, ld1, ld2) 
                   #for i, j in permutations(locus.itertuples(), 2)]
            #with open('vec_%s.pickle'%index, 'wb') as v:
                #pickle.dump(vec, v)
        #else:
            #with open('vec_%s.pickle'%index, 'rb') as v:
                #vec = pickle.load(v)
        #vec =  np.apply_along_axis(sum, 1, np.split(np.array(vec), M-1))
        #vec = [np.sum([ese(i, j, M, all_markers, N, h2_l, denominator,
        #                   mu, ld1, ld2)
        #               for j in locus.itertuples() if j != i]) 
        #       for i in locus.itertuples()
        #       ]
        #res = res.append(pd.DataFrame({'SNP':locus.SNP, 'ese':expcovs}))
    res = pd.concat(res)
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
    transferability(args)