#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<2017>
  Purpose: Compute the expected squared effect size
  Created: 09/30/17
"""

from prankcster import prankcster
from simple_GWAS import *

# ----------------------------------------------------------------------
def integral_b(vs, mu, snps):
    """
    Compute the expected beta square for all j_s in this locus

    :param vs: vector of v (see equation 8)
    :param mu: mean
    :param snps: names of snps in order
    """
    exp = (vs * vs)/ (4 * mu)  # Compute the exponent of RHS of eq. 12
    # Get the maximum of the exponent to apply the log strategy
    k = exp.max()
    e = np.exp(exp - k)
    # Compute the left hand side portion of equation 12
    lhs = ((2 * mu) + (vs * vs)) / (4 * (mu * mu))
    # Compute the right hand side portion of equation 12
    rhs = e / e.sum()
    vec = lhs * rhs         # Compute the i\integral
    return pd.Series(vec, index=snps, dtype=np.float64)


# ----------------------------------------------------------------------
def per_locus(locus, sumstats, avh2, h2, n, l_number, within=False,
              integral_only=False):
    """
    Compute the per-locus expectation

    :param locus: Tuple with source an target LD matrices and the list of snps
    :param sumstats: Dataframe with the sumary statistics
    :param avh2: Average heritability of the trait
    :param h2: Heritability pf the trait
    :param n: Number of individuals in the analysis
    :param l_number: Number of the locus being analyzed
    :param within: Type of LD product to use
    :param integral_only: Return only the integral without weight
    :return: Dataframe with the weighted expected squared effect
    """
    snps, D_r, D_t = locus
    locus = sumstats[sumstats.snp.isin(snps)].reindex(columns=['snp', 'slope'])
    m = snps.shape[0]
    h2_l = avh2 * m
    assert isinstance(h2_l, float)
    den = np.clip((1 - h2_l), 1E-10, 1)
    mu = ((n / (2 * den)) + (m / (2 * h2)))
    assert np.all(mu >= 0)
    vjs = ((n * locus.slope.values.astype(np.float64)) / den)
    I = integral_b(vjs, mu, snps)
    assert np.all(I >= 0)  # make sure integral is positive
    if integral_only:
        return pd.DataFrame({'snp': snps, 'ese': I.values, 'locus': l_number})
    assert max(I) > 0 # check if at least one is different than 0
    # set the appropriate D to work with
    if within == 1:
        p = (D_r * D_r)
    elif within == 2:
        p = (D_t * D_t)
    else:
        p = (D_r * D_t)
    expcovs = p.dot(I)
    return pd.DataFrame({'snp': snps, 'ese': expcovs, 'locus': l_number})


# ----------------------------------------------------------------------
def transferability(prefix, refgeno, refpheno, targeno, tarpheno, h2, labels,
                    LDwindow, sumstats, refld=None, tarld=None, seed=None,
                    max_memory=None, threads=1, merged=None, within=False,
                    **kwargs):
    """
    Execute trasnferability code
    """
    # Set CPU limits
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    resource.setrlimit(resource.RLIMIT_NPROC, (threads, hard))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print('Soft limit changed to :', soft)

    # set Cache to protect memory spilling
    if max_memory is not None:
        available_memory = max_memory
    else:
        available_memory = psutil.virtual_memory().available
    cache = Chest(available_memory=available_memory)

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
                                        check=kwargs['check'],
                                        max_memory=max_memory)
        (tbim, tfam, tgeno) = read_geno(targeno, kwargs['freq_thresh'], threads,
                                        check=kwargs['check'],
                                        max_memory=max_memory)
    if isinstance(sumstats, str):
        sumstats = pd.read_table(sumstats, delim_whitespace=True)
    else:
        sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    print("reference bim's shape: %d, %d" % (rbim.shape[0], rbim.shape[1]))
    print("target bim's shape: %d, %d" % (tbim.shape[0], tbim.shape[1]))
    sum_snps = sumstats.snp.tolist()
    if refld is None:
        # Compute Ds
        loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=LDwindow, justd=True,
                      threads=threads, max_memory=max_memory)
    else:
        raise NotImplementedError
    avh2 = h2 / len(sum_snps)
    n = tgeno.shape[0]
    resfile = '%s_res.tsv' % prefix
    print('Compute expected beta square per locus...')
    if not os.path.isfile(resfile):
        delayed_results = [
            dask.delayed(per_locus)(locus, sumstats, avh2, h2, n, i,
                                    within=within) for i, locus in
            enumerate(loci)]
        res = list(dask.compute(*delayed_results, num_workers=threads,
                                cache=cache, pool=ThreadPool(threads)))
        res = pd.concat(res)
        result = res.merge(sumstats.reindex(columns=['slope', 'snp', 'beta']),
                           on='snp')
        result.to_csv(resfile, index=False, sep='\t')
    else:
        result = pd.read_csv(resfile, sep='\t')
    if 'i' not in result.columns:
        result = result.merge(tbim.reindex(columns=['snp', 'i']), on='snp')
    if 'slope' not in result.columns:
        result = result.merge(sumstats.reindex(columns=['snp', 'slope']),
                              on='snp')
    prod, _ = smartcotagsort(prefix, result, column='ese')
    trans = prune_it(prod, tgeno, tpheno, 'ese', threads=threads,
                     max_memory=max_memory)
    if merged is not None:
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