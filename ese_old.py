"""
Expected beta
"""
import matplotlib
import gzip
from prankcster import *
from simple_GWAS import *
from joblib import delayed, Parallel
from tqdm import tqdm

matplotlib.use('Agg')
plt.style.use('ggplot')


# read the LD matrices locus by locus ans store them in list
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
def get_next_group(snp, df, group, app):
    """
    get the LD block and next index snp
    """
    gr = group.get_group(snp)
    done = sorted(set(gr.loc[:, ['SNP_A', 'SNP_B']].unstack()))
    sub = df[df.SNP_A.isin(done) & df.SNP_B.isin(done)]
    last = gr.SNP_B.iloc[-1]
    sub = make_symmetrical(sub, done)
    try:
        nextsnp = group.get_group(last).iloc[0].SNP_B
    except KeyError:
        print('Seems done')
        nextsnp = None
    app(done)
    return sub, last, nextsnp


# ----------------------------------------------------------------------
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
            sub, last, snp = get_next_group(snp, df[~((df.SNP_A.isin(alls) &
                                                       df.SNP_B.isin(alls)))],
                                            group, app)
            mapp(sub)
    return mats


def log_aprox_exp(a):
    k = a.max()
    l = a - k
    t = np.exp(l).sum()
    return np.exp(k) * t


# make the integral locus by locus and store them in list
# ----------------------------------------------------------------------
def integral_b(vs, mu, snps):
    """
    Compute the expected beta square
    :param vs: vector of v
    :param mu: mean
    :param snps: names of snps in order
    """
    # k = vs.max()
    # exp = np.exp(vs - k)
    # lhs = ((2 * mu) + (vs * vs)) / (4 * (mu * mu))
    # rhs = exp / exp.sum()
    # vec = lhs * rhs
    # return pd.Series(vec, index=snps)
    exp = (vs ** 2)/ (4 * mu)
    k = exp.max()
    e = np.exp(exp - k)
    lhs = ((2 * mu) + (vs * vs)) / (4 * (mu * mu))
    rhs = e / e.sum()
    vec = lhs * rhs
    return pd.Series(vec, index=snps)


# ----------------------------------------------------------------------
def integral_b2(v, mu, e, esum):
    """
    Compute the expected beta square
    :param vs: vector of v
    :param mu: mean
    :param snps: names of snps in order
    """
    v = mp.mpf(v)
    lhs = ((2 * mu) + (v * v)) / (4 * (mu * mu))
    rhs = e / esum
    integral = lhs * rhs
    return integral


# ----------------------------------------------------------------------
#@jit(parallel=True)
def per_locus2(locus, sumstats, avh2, h2, n, l_number, within=False,
              integral_only=False):
    """
    compute the per-locus expectation
    """
    snps, D_r, D_t = locus
    locus = sumstats[sumstats.snp.isin(snps)].reindex(columns=['snp', 'slope'])
    m = snps.shape[0]
    h2_l = avh2 * m
    den = np.clip((1 - h2_l), 1E-10, 1)
    mu = ((n / (2 * den)) + (m / (2 * h2)))
    assert np.all(mu >= 0)
    vjs = ((n * locus.slope.values) / den)
    exp = [(v ** 2) / mp.mpf(4 * mu) for v in vjs]
    es = np.array([mp.exp(x) for x in exp])
    esum = es.sum()
    I = np.array([integral_b2(v, mu, es[i], esum) for i, v in enumerate(vjs)])
    assert np.all(I >= 0)  # make sure integral is positive
    if integral_only:
        return pd.DataFrame({'snp': snps, 'ese': I, 'locus': l_number})
    assert max(I) > 0 # check if at least one is different than 0
    if within == 1:
        p = (D_r * D_r)
    elif within == 2:
        p = (D_t * D_t)
    else:
        p = (D_r * D_t)
    Eejb = []
    for i in range(p.shape[0]):
        expcovs = 0
        for j in range(p.shape[1]):
            expcovs += p[i,j] * I[j]
        Eejb.append(expcovs)
    Eejb = np.array(Eejb)
    return pd.DataFrame({'snp': snps, 'ese': abs(Eejb), 'locus': l_number})


# ----------------------------------------------------------------------
@jit(parallel=True)
def per_locus(locus, sumstats, avh2, h2, n, l_number, within=False,
              integral_only=False):
    """
    compute the per-locus expectation
    """
    snps, D_r, D_t = locus
    locus = sumstats[sumstats.snp.isin(snps)].reindex(columns=['snp', 'slope'])
    m = snps.shape[0]
    h2_l = avh2 * m
    den = np.clip((1 - h2_l), 1E-10, 1)
    mu = ((n / (2 * den)) + (m / (2 * h2)))
    assert np.all(mu >= 0)
    vjs = ((n * locus.slope.values) / den) #(2 * (1 - h2_l)))
    I = integral_b(vjs, mu, snps)
    assert np.all(I >= 0)  # make sure integral is positive
    if integral_only:
        return pd.DataFrame({'snp': I.index, 'ese': I.values, 'locus': l_number}
                            )
    assert max(I) > 0 # check if at least one is different than 0
    if within == 1:
        p = (D_r * D_r)
    elif within == 2:
        p = (D_t * D_t)
    else:
        p = (D_r * D_t)
    # TODO: check if p or p'
    expcovs = p.dot(I)
    return pd.DataFrame({'snp': snps, 'ese': abs(expcovs), 'locus': l_number})


# ----------------------------------------------------------------------
def compute_ld(bfile, prefix, plinkexe, window=1000):
    """
    Compute plink ld matrices
    """
    print('Computing LD matrix for file', bfile)
    out = prefix if 'ld.gz' not in prefix else prefix.split('.')[0]
    plink = ('%s --bfile %s -r gz dprime-signed with-freqs --ld-window-kb %d '
             '--ld-window %d --out %s')
    plink = plink % (plinkexe, bfile, window, int(window * 1E3) + 1, out)
    o, e = executeLine(plink)


# ----------------------------------------------------------------------
def sliding_block(gr, fulldf):
    """
    create the matrices one group at a time (sliding)
    """
    avail = pd.Series(pd.unique(gr.loc[:, ['SNP_A', 'SNP_B']].unstack())
                      ).sort_values()
    df = fulldf.join(avail, how='inner')
    # sub = fulldf[fulldf.SNP_A.isin(avail) & fulldf.SNP_B.isin(avail)]
    # start = gr.index[0]
    # sub = pd.DataFrame()
    # iterator = (i for i in range(-1,-(gr.shape[0]+1),-1)) #compatibility 2 and 3
    # while sub.empty:
    # if gr.shape[0] == 1:
    # sub = gr.index
    # break
    # condition = fulldf.SNP_A == gr.SNP_B.iloc[next(iterator)]
    # sub = fulldf.SNP_A.index[condition]
    # stop = sub[-1]
    # sub = fulldf.loc[start:stop,:]

    return make_symmetrical(df, avail)


# ----------------------------------------------------------------------
def readLD(fn):
    """
    Read LD matrix
    """
    print('Reading LD from file', fn)
    dtypes = {'SNP_A': str, 'SNP_B': str, 'D': float}
    cols = ['SNP_A', 'SNP_B', 'D']
    df = pd.read_table(fn, engine='c', delim_whitespace=True, usecols=cols,
                       dtype=dtypes).dropna()
    snps = pd.unique(df.loc[:, ['SNP_A', 'SNP_B']].unstack())
    return df, snps


# ----------------------------------------------------------------------
def thelocus(i, ld1, ld2, sum_snps):
    """
    return the  intersection between the allowed snps
    """
    return sorted(set(ld1[i].index.tolist()).intersection(ld2[i].index.tolist(
    )).intersection(sum_snps))


# ----------------------------------------------------------------------
def transferability_plink(args):
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
        pick = pickle.dumps((ld1, ld2))
        with gzip.open(matfile, 'w') as F:
            F.write(pick)
    else:
        print('Loading previously computed blocks')
        with gzip.open(matfile, 'r') as F:
            ld1, ld2 = pickle.loads(F.read())
    print('Setting the loci')
    # loci = Parallel(n_jobs=int(args.threads))(delayed(thelocus)(i, ld1, ld2,
    #                                                            sum_snps)
    #                                          for i in range(len(ld1)))
    loci = [thelocus(index, ld1, ld2, sum_snps) for index in range(len(ld1))]
    avh2 = args.h2 / len(sum_snps)
    with open('%s_loci.pickle' % args.prefix, 'wb') as L:
        pickle.dump(loci, L)
    N = map_count('%s.fam' % args.target)
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
    # product, _ = smartcotagsort(args.prefix, res, column='ese')
    product = res.sort_values('ese', ascending=False).reset_index(drop=True)
    product['Index'] = product.index.tolist()
    nsnps = product.shape[0]
    percentages = set_first_step(nsnps, 5, every=False)
    percentages = set_first_step(nsnps, 5, every=False)
    snps = np.around((percentages * nsnps) / 100).astype(int)
    qfile = '%s.qfile' % args.prefix
    if args.qrange is None:
        # qrange= '%s.qrange' % args.prefix
        qr, qrange = gen_qrange(args.prefix, nsnps, 5, qrange, every=False)
    else:
        qrange = args.qrange
        order = ['label', 'Min', 'Max']
        qr = pd.read_csv(qrange, sep=' ', header=None, names=order)
    product.loc[:, ['SNP', 'Index']].to_csv(qfile, sep=' ', header=False,
                                            index=False)
    df = qrscore(args.plinkexe, args.target, args.sumstats, qrange, qfile,
                 args.allele_file, args.pheno, args.prefix, qr, args.maxmem,
                 args.threads, 'None', args.prefix)
    # get ppt results
    # ppts=[]
    # for i in glob('*.results'):
    # three_code = i[:4]
    # results = pd.read_table(i, sep='\t')
    # R2 = results.nlargest(1, 'R2').R2.iloc[0]
    # ppts.append((three_code, R2))
    # ppts = sorted(ppts, key=lambda x: x[1], reverse=True)
    # aest = [('0.5', '*'), ('k', '.')]
    if args.merged is not None:
        merged = pd.read_table(args.merged, sep='\t')
    merged = merged.merge(df, on='Number of SNPs')
    f, ax = plt.subplots()
    merged.plot.scatter(x='Number of SNPs', y='R2', alpha=0.5, c='purple', s=5,
                        ax=ax, label='Transferability', linestyle=':')
    merged.plot.scatter(x='Number of SNPs', y=r'$R^{2}$_cotag', label='Cotagging',
                        c='r', s=2, alpha=0.5, ax=ax)
    merged.plot.scatter(x='Number of SNPs', y='R2_hybrid', c='g', s=5, alpha=0.5,
                        ax=ax, label='Hybrid (COT & P+T)')
    merged.plot.scatter(x='Number of SNPs', y='$R^{2}$_clumEUR', c='0.5', s=5,
                        alpha=0.5, marker='*', ax=ax, label='EUR P+T')
    merged.plot.scatter(x='Number of SNPs', y='$R^{2}$_clumAFR', c='k', s=5,
                        alpha=0.5, marker='.', ax=ax, label='AFR P+T')
    # for i, item in enumerate(ppts):
    # pop, r2 = item
    # ax.axhline(r2, label='%s P + T Best' % pop, color=aest[i][0], ls='--',
    # marker=aest[i][1], markevery=10)
    plt.ylabel('$R^2$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s_transferability.pdf' % args.prefix)
    plt.close()
    return res


# ----------------------------------------------------------------------
def transferability(prefix, refgeno, refpheno, targeno, tarpheno, h2, labels,
                    LDwindow, sumstats, refld=None, tarld=None, seed=None,
                    max_memory=None, threads=1, merged=None, within=False,
                    pedestrian=False, **kwargs):
    """
    Execute trasnferability code
    """
    if pedestrian:
        per_locus = per_locus2
    seed = np.random.randint(1e4) if seed is None else seed
    now = time.time()
    print('Performing expected square effect (ESE)!')
    refl, tarl = labels
    # If pheno is None for the reference, make simulation
    if isinstance(refpheno, str):
        rpheno = dd.read_table(refpheno, blocksize=25e6, delim_whitespace=True)
        tpheno = dd.read_table(tarpheno, blocksize=25e6, delim_whitespace=True)
    elif refpheno is None:
        # make simulation for reference
        print('Simulating phenotype for reference population %s \n' % refl)
        opts = {'outprefix': refl, 'bfile': refgeno, 'h2': h2,
                'ncausal': kwargs['ncausal'], 'normalize': kwargs['normalize'],
                'uniform': kwargs['uniform'], 'snps': None, 'seed': seed,
                'bfile2': targeno, 'flip':kwargs['gflip'],
                'max_memory': max_memory}
        rpheno, h2, (rgeno, rbim, rtruebeta, rvec) = qtraits_simulation(**opts)
        # make simulation for target
        print('Simulating phenotype for target population %s \n' % tarl)
        opts.update(
            dict(outprefix=tarl, bfile=targeno, validate=kwargs['split'],
                 causaleff=rbim.dropna(subset=['beta']), bfile2=refgeno))
        tpheno, h2, (tgeno, tbim, ttruebeta, tvec) = qtraits_simulation(**opts)
        opts.update(dict(prefix='ranumo_gwas', pheno=rpheno, geno=rgeno,
                         validate=kwargs['split'], threads=threads, bim=rbim,
                         flip=kwargs['flip']))
    elif isinstance(refgeno, str):
        (rbim, rfam, rgeno) = read_geno(refgeno, kwargs['freq_thresh'], threads,
                                        check=kwargs['check'])
        (tbim, tfam, tgeno) = read_geno(targeno, kwargs['freq_thresh'], threads,
                                        check=kwargs['check'])
    if isinstance(sumstats, str):
        sumstats = pd.read_table(sumstats, delim_whitespace=True)
    else:
        sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    print("reference bim's shape: %d, %d" % (rbim.shape[0], rbim.shape[1]))
    print("target bim's shape: %d, %d" % (tbim.shape[0], tbim.shape[1]))
    sum_snps = sumstats.snp.tolist()
    if refld is None:
        # Compute Ds
        loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=LDwindow,
                      threads=threads, justd=True)
    else:
        raise NotImplementedError
    avh2 = h2 / len(sum_snps)
    n = tgeno.shape[0]
    resfile = '%s_res.tsv' % prefix
    print('Compute expected beta square per locus...')
    if not os.path.isfile(resfile):
        # res = Parallel(n_jobs=int(threads))(
        #     delayed(per_locus)(locus, sumstats, avh2, h2, n) for
        #     i, locus in tqdm(enumerate(loci), total=len(loci)))
        delayed_results = [
            dask.delayed(per_locus)(locus, sumstats, avh2, h2, n, i,
                                    within=within) for i, locus in
            enumerate(loci)]
        res = list(dask.compute(*delayed_results, num_workers=threads))
        res = pd.concat(res)
        result = res.merge(sumstats.reindex(columns=['slope', 'snp', 'beta']),
                           on='snp')
        result.to_csv(resfile, index=False, sep='\t')
    else:
        result = pd.read_csv(resfile, sep='\t')
    #result = res.sort_values('ese', ascending=False).reset_index(drop=True)
    #result['Index'] = result.index.tolist()
    if 'i' not in result.columns:
        result = result.merge(tbim.reindex(columns=['snp', 'i']), on='snp')
    if 'slope' not in result.columns:
        result = result.merge(sumstats.reindex(columns=['snp', 'slope']),
                              on='snp')
    prod, _ = smartcotagsort(prefix, result, column='ese')
    trans = prune_it(prod, tgeno, tpheno, 'ese', threads=threads)
    if merged is not None:
        #merged = pd.read_table(merged, sep='\t')
        with open(merged, 'rb') as F:
            merged = pickle.load(F)
    else:
        # do prancster
        merged = prankcster('%s_prancster' % prefix, tgeno, rgeno, tpheno,
                            (refl, tarl), 0.1, 5, splits=kwargs['split'],
                            threads=threads, seed=seed, graph=kwargs['graph'],
                            r_range=kwargs['r_range'],
                            p_tresh=kwargs['p_tresh'], sumstats=sumstats,
                            tbim=tbim, rbim=rbim, window=LDwindow,
                            X_test=X_test, Y_test=y_test,
                            ld_operator=kwargs['ld_operator'])
    res = pd.concat([trans] + merged)
    # plot
    colors = iter(['r', 'b', 'm', 'g', 'c', 'k','y'])
    f, ax = plt.subplots()
    for t, df in res.groupby('type'):
        df.plot(x='Number of SNPs', y='R2', kind='scatter', legend=True,
                s=3, c=next(colors), ax=ax, label=t)
    plt.tight_layout()
    plt.savefig('%s_transferability.pdf' % args.prefix)
    plt.close()
    print('ESE done after %.2f minutes' % ((time.time() - now) / 60.))
    return res


# ----------------------------------------------------------------------
if __name__ == '__main__':
    class Store_as_arange(argparse._StoreAction):
        def __call__(self, parser, namespace, values, option_string=None):
            values = np.arange(values[0], values[1], values[2])
            return super().__call__(parser, namespace, values, option_string)


    class Store_as_array(argparse._StoreAction):
        def __call__(self, parser, namespace, values, option_string=None):
            values = np.array(values)
            return super().__call__(parser, namespace, values, option_string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-a', '--allele_file', default='EUR.allele',
                        help='File with the allele order. A1 in position 3 and '
                             'id in position2')
    parser.add_argument('-b', '--reference', required=True,
                        help=('prefix of the bed fileset in reference'))
    parser.add_argument('-g', '--target', required=True,
                        help=('prefix of the bed fileset in target'))
    parser.add_argument('-L', '--labels', help=('Populations labels'), nargs=2)
    parser.add_argument('-l', '--refld', default=None,
                        help=('plink LD matrix for the target population'))
    parser.add_argument('-d', '--tarld', default=None,
                        help=('plink LD matrix for the reference population'))
    parser.add_argument('-s', '--sumstats', help='Filename of sumstats',
                        default=None)
    parser.add_argument('-H', '--h2', type=float, help='Heritability of trait',
                        required=True)
    parser.add_argument('-f', '--tarpheno', default=None,
                        help=('Filename of the true phenotype of the target '
                              'population'))
    parser.add_argument('-r', '--refpheno', default=None,
                        help=('Filename of the true phenotype of the reference '
                              'population'))
    parser.add_argument('-S', '--sliding', default=False, action='store_true',
                        help=('Use a sliding window instead of hard block'))

    parser.add_argument('-w', '--window', default=1000, type=int,
                        help=('Size of the LD window. a.k.a locus'))
    parser.add_argument('-P', '--plinkexe', default=None)
    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=None, type=int)
    parser.add_argument('-m', '--merged', default=None, help=('Merge file of '
                                                              'prankcster run'
                                                              ))
    parser.add_argument('-q', '--qrange', default=None,
                        help="File of previous qrange. e.g. ranumo's qrange")
    parser.add_argument('--ncausal', default=200, type=int)
    parser.add_argument('--normalize', default=True, action='store_false')
    parser.add_argument('--uniform', default=True, action='store_false')
    parser.add_argument('--split', default=2, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--r_range', default=None, nargs=3,
                        action=Store_as_arange, type=float)
    parser.add_argument('--p_tresh', default=None, nargs='+',
                        action=Store_as_array, type=float)
    parser.add_argument('--flip', action='store_true', help='flip sumstats')
    parser.add_argument('--gflip', action='store_true', help='flip genotype')
    parser.add_argument('--freq_thresh', type=float, help='filter by mafs')
    parser.add_argument('--within', default=0, type=int,
                        help='0=cross; 1=reference; 2=target')
    parser.add_argument('--ld_operator', default='lt')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--check', action='store_true',
                        help='check and clean invariant sites')
    parser.add_argument('--pedestrian', default=False, action='store_true')

    args = parser.parse_args()
    transferability(args.prefix, args.reference, args.refpheno, args.target,
                    args.tarpheno, args.h2, args.labels, args.window,
                    args.sumstats, refld=args.refld, tarld=args.tarld,
                    seed=args.seed, threads=args.threads, merged=args.merged,
                    ncausal=args.ncausal, normalize=True, uniform=args.uniform,
                    r_range=args.r_range, p_tresh=args.p_tresh,
                    max_memory=args.maxmem, split=args.split, flip=args.flip,
                    gflip=args.gflip, within=args.within,
                    ld_operator=args.ld_operator, graph=args.graph,
                    check=args.check, pedestrian=args.pedestrian)
