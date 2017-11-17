"""
Expected beta
"""
from time import time
import argparse
import matplotlib
import pandas as pd
import numpy as np
import gzip
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utilities4cotagging import *
from joblib import delayed, Parallel
plt.style.use('ggplot')


# read the LD matrices locus by locus ans store them in list
#----------------------------------------------------------------------
def make_symmetrical(matrix, avail):
    """
    make matrix symmetrical
    """
    matrix = matrix.pivot(columns='SNP_B', index='SNP_A', values='D')
    m = pd.DataFrame(None, index=avail, columns=avail)
    m.update(matrix)
    m.update(matrix.transpose())
    np.fill_diagonal(m.values, 1) 
    return m

#----------------------------------------------------------------------
def get_next_group(snp, df, group, app):
    """
    get the LD block and next index snp
    """
    gr = group.get_group(snp)
    done = sorted(set(gr.loc[:,['SNP_A', 'SNP_B']].unstack()))
    sub = df[df.SNP_A.isin(done) & df.SNP_B.isin(done)]
    last = gr.SNP_B.iloc[-1]
    sub = make_symmetrical(sub, done)
    try:
        nextsnp = group.get_group(last).iloc[0].SNP_B
    except KeyError:
        print ('Seems done')
        nextsnp = None
    app(done)
    return sub, last, nextsnp
    
#----------------------------------------------------------------------
def get_blocks(df, available_snps, label, sliding=False, cpus=1):
    """
    process LD matrix and store submatrices of size locus
    """
    print('Getting LD blocks from', label)
    group = df.groupby('SNP_A')
    keys = sorted(group.groups.keys())
    ngroups = pd.Series(keys).isin(available_snps).sum()
    grouping = (x for x in keys if x in available_snps)
    if sliding:
        print('Using sliding window (this takes longer!! just FYI)')
        mats = Parallel(n_jobs=cpus)(delayed(sliding_block)(
            group.get_group(k), df) for k in tqdm(grouping, total=ngroups))    
    else:
        snp = df.SNP_A.iloc[0]
        mats = []
        alls = []
        app = alls.extend
        mapp = mats.append        
        while (df.SNP_A.iloc[-1] not in alls) and (snp is not None):
            sub, last, snp  = get_next_group(snp, df[~((df.SNP_A.isin(alls) & 
                                                        df.SNP_B.isin(alls)))], 
                                             group, app)
            mapp(sub)                          
    return mats

# make the integral locus by locus and store them in list
#----------------------------------------------------------------------
def integral_b(vs, mu, snps):
    """
    Compute the expected beta square
    :param vs: vector of v
    :param mu: mean
    :param snps: names of snps in order
    """
    exp = np.exp( np.power(vs,2) / (4 * mu) )
    lhs = ( ((2 * mu) + np.power(vs,2)) ) / ( 4 * np.power(mu,2) )
    rhs = exp / exp.sum()
    vec = lhs * rhs
    return pd.Series(vec, index=snps)

#----------------------------------------------------------------------
def per_locus(locus, sumstats, avh2, h2,  N, ld1, ld2, M):
    """
    compute the per-locus expectation
    """
    locus = sumstats[sumstats.SNP.isin(locus)].loc[:,['SNP', 'BETA']]
    locus.index = locus.SNP.tolist()
    snps = locus.SNP.tolist()
    #M = locus.shape[0]
    h2_l = avh2 * M
    mu = ( (N /(2 * ( 1 - h2_l ) ) ) + ( M / ( 2 * h2 ) ) )
    vjs = (( N * locus.BETA.values ) / (2 * ( 1 - h2_l ) ))
    I = integral_b(vjs, mu, snps)
    expcovs = (ld1.loc[snps, snps].multiply(ld2.loc[snps, snps]).dot(I))
    return pd.DataFrame({'SNP': expcovs.index.tolist(), 
                         'ese': expcovs.values.tolist()})

#----------------------------------------------------------------------
def compute_ld(bfile, prefix, plinkexe, window=1000):
    """
    Compute plink ld matrices
    """
    print('Computing LD matrix for file', bfile)
    out = prefix if 'ld.gz' not in prefix else prefix.split('.')[0]
    plink = ('%s --bfile %s -r gz dprime-signed with-freqs --ld-window-kb %d '
             '--ld-window %d --out %s')
    plink = plink % (plinkexe, bfile, window, int(window*1E3) + 1, out)
    o, e = executeLine(plink)

#----------------------------------------------------------------------
def sliding_block(gr, fulldf):
    """
    create the matrices one group at a time (sliding)
    """
    avail = pd.Series(pd.unique(gr.loc[:,['SNP_A', 'SNP_B']].unstack())
                      ).sort_values()
    df = fulldf.join(avail, how='inner')
    #sub = fulldf[fulldf.SNP_A.isin(avail) & fulldf.SNP_B.isin(avail)]
    #start = gr.index[0]
    #sub = pd.DataFrame()
    #iterator = (i for i in range(-1,-(gr.shape[0]+1),-1)) #compatibility 2 and 3
    #while sub.empty:
        #if gr.shape[0] == 1:
            #sub = gr.index 
            #break
        #condition = fulldf.SNP_A == gr.SNP_B.iloc[next(iterator)]          
        #sub = fulldf.SNP_A.index[condition]
    #stop = sub[-1]
    #sub = fulldf.loc[start:stop,:]
    
    return make_symmetrical(df, avail)
    
#----------------------------------------------------------------------
def readLD(fn):
    """
    Read LD matrix
    """
    print('Reading LD from file', fn)
    dtypes = {'SNP_A':str, 'SNP_B':str, 'D':float}
    cols = ['SNP_A', 'SNP_B', 'D']
    df = pd.read_table(fn, engine='c', delim_whitespace=True, usecols=cols, 
                       dtype=dtypes).dropna()
    snps = pd.unique(df.loc[:, ['SNP_A', 'SNP_B']].unstack())
    return df, snps

#----------------------------------------------------------------------
def thelocus(i, ld1, ld2, sum_snps):
    """
    return the  intersection between the allowed snps
    """
    return sorted(set(ld1[i].index.tolist()).intersection(ld2[i].index.tolist(
        )).intersection(sum_snps))    
            
#----------------------------------------------------------------------
def transferability(args):
    """
    Execute trasnferability code
    """
    sumstats = pd.read_table(args.sumstats, delim_whitespace=True)
    sum_snps = sumstats.SNP.tolist()
    if not os.path.isfile(args.refld):
        compute_ld(args.reference, args.refld, args.plinkexe, 
                   window=args.window)
    if not os.path.isfile(args.tarld):
        compute_ld(args.target, args.tarld, args.plinkexe, window=args.window)
    df1, snps1 = readLD(args.refld)
    df2, snps2 = readLD(args.tarld)
    available_snps = set(snps1).intersection(snps2).intersection(sum_snps)
    matfile = '%s_matrices.pickle' % args.prefix
    if not os.path.isfile(matfile):        
        ld1 = get_blocks(df1, available_snps, args.refld, sliding=args.sliding, 
                         cpus=args.threads)
        ld2 = get_blocks(df2, available_snps, args.tarld, sliding=args.sliding, 
                         cpus=args.threads)
        pick = pickle.dumps((ld1,ld2))
        with gzip.open(matfile, 'w') as F:
            F.write(pick)
    else:
        print('Loading previously computed blocks')
        with gzip.open(matfile, 'r') as F:
            ld1, ld2 = pickle.loads(F.read())
    print('Setting the loci')
    #loci = Parallel(n_jobs=int(args.threads))(delayed(thelocus)(i, ld1, ld2, 
    #                                                            sum_snps)
    #                                          for i in range(len(ld1)))
    loci = [thelocus(index, ld1, ld2, sum_snps) for index in range(len(ld1))]
    avh2 = args.h2 / len(sum_snps)
    with open('%s_loci.pickle' % args.prefix,'wb') as L:
        pickle.dump(loci, L)
    N = mapcount('%s.fam' % args.target)
    resfile = '%s_res.tsv' % args.prefix
    print('Compute expected beta square per locus...')
    if not os.path.isfile(resfile):
        res = Parallel(n_jobs=int(args.threads))(delayed(per_locus)(
            locus, sumstats, avh2, args.h2, N, ld1[i], ld2[i], len(loci)
            ) for i, locus in tqdm(enumerate(loci), total=len(loci)))    
        res = pd.concat(res)
        res.to_csv(resfile, index=False, sep='\t')
    else:
        res = pd.read_csv(resfile, sep='\t')  
    if args.sliding:
        res = res.groupby('SNP').mean()
        res['SNP'] = res.index.tolist()
    #product, _ = smartcotagsort(args.prefix, res, column='ese')
    product = res.sort_values('ese', ascending=False).reset_index(drop=True)
    product['Index'] = product.index.tolist()
    nsnps = product.shape[0]
    percentages = set_first_step(nsnps, 5, every=False)
    snps = np.around((percentages * nsnps) / 100).astype(int) 
    qfile = '%s.qfile' % args.prefix
    if args.qrange is None:
        qrange= '%s.qrange' % args.prefix
        qr = gen_qrange(args.prefix, nsnps, 5, qrange, every=False)
    else:
        qrange = args.qrange
        order = ['label', 'Min', 'Max']
        qr = pd.read_csv(qrange, sep=' ', header=None, names=order) 
    product.loc[:,['SNP', 'Index']].to_csv(qfile, sep=' ', header=False,
                                           index=False)   
    df = qrscore(args.plinkexe, args.target, args.sumstats, qrange, qfile, 
                 args.pheno, args.prefix, qr, args.maxmem, args.threads, 
                 'None', args.prefix)
    #get ppt results
    #ppts=[]
    #for i in glob('*.results'):
        #three_code = i[:4]
        #results = pd.read_table(i, sep='\t')
        #R2 = results.nlargest(1, 'R2').R2.iloc[0]
        #ppts.append((three_code, R2))
    #ppts = sorted(ppts, key=lambda x: x[1], reverse=True)
    #aest = [('0.5', '*'), ('k', '.')]
    if args.merged is not None:
        merged = pd.read_table(args.merged, sep='\t')
    merged = merged.merge(df, on='SNP kept')
    f, ax = plt.subplots()
    merged.plot.scatter(x='SNP kept', y='R2', alpha=0.5, c='purple', s=5,
                        ax=ax, label='Transferability', linestyle=':')
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_cotag', label='Cotagging', 
                        c='r', s=2, alpha=0.5, ax=ax)    
    merged.plot.scatter(x='SNP kept', y='R2_hybrid', c='g',s=5, alpha=0.5, 
                        ax=ax, label='Hybrid (COT & P+T)')
    merged.plot.scatter(x='SNP kept', y='$R^{2}$_clumEUR', c='0.5', s=5, 
                        alpha=0.5, marker='*', ax=ax, label='EUR P+T') 
    merged.plot.scatter(x='SNP kept', y='$R^{2}$_clumAFR', c='k', s=5, 
                        alpha=0.5, marker='.', ax=ax, label='AFR P+T')     
    #for i, item in enumerate(ppts):
        #pop, r2 = item 
        #ax.axhline(r2, label='%s P + T Best' % pop, color=aest[i][0], ls='--',
                   #marker=aest[i][1], markevery=10)
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
    parser.add_argument('-s', '--sumstats', help='Filename of sumstats',
                        required=True)
    parser.add_argument('-H', '--h2', type=float, help='Heritability of trait',
                        required=True)    
    parser.add_argument('-f', '--pheno', required=True, 
                        help=('Filename of the true phenotype of the target '
                              'population')) 
    parser.add_argument('-S', '--sliding', default=False, action='store_true',
                        help=('Use a sliding window instead of hard block'))    
    
    parser.add_argument('-w', '--window', default=1000, type=int, 
                        help=('Size of the LD window. a.k.a locus')) 
    parser.add_argument('-P', '--plinkexe', required=True)
    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=3000, type=int) 
    parser.add_argument('-m', '--merged', default=None, help=('Merge file of '
                                                              'prankcster run'
                                                              ))
    parser.add_argument('-q', '--qrange', default=None, 
                        help="File of previous qrange. e.g. ranumo's qrange")    
    args = parser.parse_args()
    transferability(args)    