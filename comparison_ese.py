'''
Code to execute parts of ese and plot the different sorts
'''

from ese import *

within_dict = {0: 'ese cotag', 1: 'ese EUR', 2: 'ese AFR'}
prunestep = 30


def sortbylocus(prefix, df, column='ese', title=None):
    picklefile = '%s.pickle' % prefix
    if os.path.isfile(picklefile):
        with open(picklefile, 'rb') as F:
            sorteddf = pickle.load(F)
    else:
        df['m_size'] = norm(abs(df.slope), 20, 200)
        # df.sort_values(by=column,  ascending=False, inplace=True)
        grouped = df.groupby('locus', as_index=False)
        grouped = grouped.apply(lambda grp: grp.nlargest(1, column))
        sorteddf = grouped.sort_values(by=column, ascending=False)
        tail = df[~df.snp.isin(sorteddf.snp)]
        # grouped = tail.groupby('locus', as_index=False)
        if not tail.empty:
            sorteddf = sorteddf.append(
                tail.sort_values(by=column, ascending=False))
        sorteddf = sorteddf.reset_index(drop=True)
        sorteddf['index'] = sorteddf.index.tolist()
        with open(picklefile, 'wb') as F:
            pickle.dump(sorteddf, F)
    size = sorteddf.m_size
    # make sure x and y are numeric
    sorteddf['pos'] = pd.to_numeric(sorteddf.pos)
    sorteddf['index'] = pd.to_numeric(sorteddf.index)
    idx = sorteddf.dropna(subset=['beta']).index.tolist()
    f, ax = plt.subplots()
    sorteddf.plot.scatter(x='pos', y='index', ax=ax, label=column, s=size.values
                          )
    sorteddf.loc[idx, :].plot.scatter(x='pos', y='index', c='k', marker='*',
                                      ax=ax, label='Causals',
                                      s=size[idx].values)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig('%s.pdf' % prefix)
    return sorteddf


def individual_ese(sumstats, avh2, h2, n, within, loci, tgeno, tpheno, threads,
                   tbim, prefix):
    within_str = within_dict[within]
    prefix = '%s_%s' % (prefix, '_'.join(within_str.split()))
    print('Compute expected beta square per locus...')
    resfile = '%s_res.tsv' % prefix
    if not os.path.isfile(resfile):
        delayed_results = [
            dask.delayed(per_locus)(locus, sumstats, avh2, h2, n, i,
                                    within=within) for i, locus in
            enumerate(loci)]
        with ProgressBar():
            res = list(dask.compute(*delayed_results, num_workers=args.threads))
        res = pd.concat(res)  # , ignore_index=True)
        result = res.merge(
            sumstats.reindex(columns=['slope', 'snp', 'beta', 'pos']), on='snp')
        result = result.rename(columns={'ese': within_str})
        if 'i' not in result.columns:
            result = result.merge(tbim.reindex(columns=['snp', 'i']), on='snp')
        if 'slope' not in result.columns:
            result = result.merge(sumstats.reindex(columns=['snp', 'slope']),
                                  on='snp')
        result.to_csv(resfile, index=False, sep='\t')
    else:
        result = pd.read_csv(resfile, sep='\t')
    # prod, _ = smartcotagsort(prefix, result, column=within_str, ascending=False)
    prod = sortbylocus(prefix, result, column=within_str,
                       title=r'Realized $h^2$: %f' % h2)
    prod = prune_it(prod, tgeno, tpheno, within_str, step=prunestep,
                    threads=threads)
    # prod['type'] = within
    return prod


def get_tagged(snp_list, D_r, ld_thr, p_thresh, sumstats):
    index = []
    ippend = index.append
    tag = []
    text = tag.extend
    while snp_list != []:
        # get lowest pvalue snp in the locus
        curr_high = sumstats.nsmallest(1, 'pvalue')  # TODO: change to min
        if curr_high.pvalue.values[0] < p_thresh:
            curr_high = curr_high.snp.values[0]
            ippend(curr_high)
            chidx = np.where(D_r.columns == curr_high)[0]
            # get snps in LD
            vec = D_r.loc[:, chidx]
            tagged = vec[vec > ld_thr].columns.tolist()
            if curr_high in tagged: # TODO: check if necesary
                tagged.pop(tagged.index(curr_high))
            text(tagged)
            snp_list = [snp for snp in snp_list if snp not in tagged]
            snp_list.pop(snp_list.index(curr_high))
        else:
            low = sumstats.snp.tolist()
            text(low)
            snp_list = [snp for snp in snp_list if snp not in low]
        sumstats = sumstats[sumstats.snp.isin(snp_list)]
    return index, tag


@jit
def loop_pairs(snp_list, D_r, l, p, sumstats, pheno, geno):
    index, tag = get_tagged(snp_list, D_r, l, p, sumstats)
    clump = sumstats[sumstats.snp.isin(index)]
    idx = clump.i.tolist()
    prs = geno[:, idx].dot(clump.slope)
    est = np.corrcoef(prs, pheno.PHENO)[1, 0] ** 2
    return est, (index, tag)


def dirty_ppt(loci, sumstats, geno, pheno, threads):
    now = time.time()
    print('Starting dirty PPT...')
    index, tag = [], []
    for r, locus in enumerate(loci):
        snps, D_r, D_t = locus
        snp_list = snps.tolist()
        D_r = dd.from_dask_array(D_r, columns=snps) ** 2
        sub = sumstats[sumstats.snp.isin(snps)].reindex(
            columns=['snp', 'slope', 'pvalue', 'i'])
        # filter pvalue
        pvals = [1, 0.5, 0.2, 0.05, 10E-3, 10E-5, 10E-7, 1E-9]
        ldval = np.arange(0.1, 0.8, 0.1)  # [0.1, 0.2, 0.4, 0.8]
        pairs = product(pvals, ldval)
        delayed_results = [
            dask.delayed(loop_pairs)(snp_list, D_r, l, p, sub, pheno, geno)
            for p, l in pairs]
        with ProgressBar():
            print('    Locus', r)
            d = dict(list(dask.compute(*delayed_results, num_workers=threads)))
        best_key = max(d.keys())
        i, t = d[best_key]
        index += i
        tag += t
    pre = sumstats[sumstats.snp.isin(index)]
    pos = sumstats[sumstats.snp.isin(tag)]
    ppt = pre.append(pos, ignore_index=True).reset_index(drop=True)
    ppt['index'] = ppt.index.tolist()
    print('Dirty ppt done after %.2f minutes' % ((time.time() - now) / 60.))
    return ppt, pre, pos


def main(args):
    now = time.time()
    seed = np.random.randint(1e4) if args.seed is None else args.seed
    refl, tarl = args.labels
    # make simulations
    print('Simulating phenotype for reference population %s \n' % refl)
    opts = {'outprefix': refl, 'bfile': args.refgeno, 'h2': args.h2,
            'ncausal': args.ncausal, 'normalize': args.normalize,
            'uniform': args.uniform, 'snps': None, 'seed': seed,
            'bfile2': args.targeno, 'flip': args.gflip,
            'max_memory': args.maxmem, 'freqthreshold': args.freq_thresh}
    rpheno, h2, (rgeno, rbim, rtruebeta, rvec) = qtraits_simulation(**opts)
    # make simulation for target
    print('Simulating phenotype for target population %s \n' % tarl)
    opts.update(dict(outprefix=tarl, bfile=args.targeno, bfile2=args.refgeno,
                     causaleff=rbim.dropna(subset=['beta']), validate=args.split
                     ))
    if args.reference:
        tpheno, tgeno = rpheno, rgeno
    else:
        tpheno, h2, (tgeno, tbim, truebeta, tvec) = qtraits_simulation(**opts)
        opts.update(dict(prefix='ranumo_gwas', pheno=rpheno, geno=rgeno,
                         validate=args.split, threads=args.threads, bim=rbim,
                         flip=args.flip))

    # perform GWAS
    sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    sumstats['beta_sq'] = sumstats.slope ** 2
    # plot correlation between pval and beta^2
    # ax = sumstats.plot.scatter(x='pvalue', y='beta_sq')
    # ax.set_xscale('log')
    # plt.savefig('%s_pval_b2.pdf' % args.prefix)
    sum_snps = sumstats.snp.tolist()
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=args.window,
                  threads=args.threads, justd=True)
    # include ppt
    scs_title = r'Realized $h^2$: %f' % h2
    pptfile = '%s_%s_res.tsv' % (args.prefix, 'ppt')
    if not os.path.isfile(pptfile):
        ppt, _, _ = dirty_ppt(loci, sumstats, tgeno, tpheno, args.threads)
        ppt, _ = smartcotagsort('%s_ppt' % args.prefix, ppt, column='index',
                                ascending=True, title=scs_title)
        ppt = prune_it(ppt, tgeno, tpheno, 'P + T', step=prunestep,
                       threads=args.threads)
        ppt.to_csv(pptfile, index=False, sep='\t')
    else:
        ppt = pd.read_csv(pptfile, sep='\t')
    ppt.plot(x='Number of SNPs', y='R2', kind='scatter', legend=True, s=3)
    plt.savefig('ppt_afr.pdf')
    avh2 = h2 / len(sum_snps)
    n = tgeno.shape[0]
    # compute ese only integral
    delayed_results = [dask.delayed(per_locus)(locus, sumstats, avh2, h2, n, i,
                                               integral_only=True) for i, locus
                       in enumerate(loci)]
    with ProgressBar():
        integral = list(dask.compute(*delayed_results, num_workers=args.threads)
                        )
    integral = pd.concat(integral, ignore_index=True)
    integral = integral.merge(sumstats.reindex(
        columns=['snp', 'pvalue', 'beta_sq', 'slope', 'pos', 'i', 'beta']),
        on='snp')
    intfile = '%s_%s_res.tsv' % (args.prefix, 'integral')
    if not os.path.isfile(intfile):
        integral = sortbylocus('%s_integral' % args.prefix, integral,
                               column='ese', title='Integral; %s' % scs_title)
        integral = prune_it(integral, tgeno, tpheno, 'Integral', step=prunestep,
                            threads=args.threads)
        integral.to_csv(intfile, index=False, sep='\t')
    else:
        integral = pd.read_csv(intfile, sep='\t')
    # expecetd square effect
    eses = [individual_ese(sumstats, avh2, h2, n, x, loci, tgeno, tpheno,
                           args.threads, tbim, args.prefix) for x in [0, 1, 2]]
    # prune by pval
    resfile = '%s_%s_res.tsv' % (args.prefix, 'pvalue')
    if not os.path.isfile(resfile):
        pval = sumstats.sort_values('pvalue', ascending=True).reset_index(
            drop=True)
        pval['index'] = pval.index
        pval, _ = smartcotagsort('%s_pval' % args.prefix, pval, column='index',
                                 ascending=True)
        pval = prune_it(pval, tgeno, tpheno, 'pval', step=prunestep,
                        threads=args.threads)
        pval.to_csv(resfile, index=False, sep='\t')
    else:
        pval = pd.read_csv(resfile, sep='\t')
    # prune by estimated beta
    betafile = '%s_%s_res.tsv' % (args.prefix, 'slope')
    if not os.path.isfile(betafile):
        beta = sumstats.copy()
        beta, _ = smartcotagsort('%s_slope' % args.prefix, beta,
                                 column='beta_sq', ascending=False,
                                 title=scs_title)
        beta = prune_it(beta, tgeno, tpheno, r'$\beta^2$', step=prunestep,
                        threads=args.threads)
        beta.to_csv(betafile, index=False, sep='\t')
    else:
        beta = pd.read_csv(betafile, sep='\t')
    # include causals
    causals = sumstats.dropna(subset=['beta'])
    caus, _ = smartcotagsort('%s_causal' % args.prefix, causals, column='beta',
                             ascending=False, title='Causals; %s' % scs_title)
    caus = prune_it(caus, tgeno, tpheno, 'Causals', step=prunestep,
                    threads=args.threads)
    caus.to_csv('%s_%s_res.tsv' % (args.prefix, 'causal'), index=False,
                sep='\t')
    # plot them
    res = pd.concat(eses + [pval, beta, caus, ppt, integral])
    # best_r2 = lr(
    #     tgeno[:, sumstats.dropna().i.values].dot(sumstats.dropna().slope),
    #     tpheno.PHENO).rvalue ** 2
    res.to_csv('%s_finaldf.tsv' % args.prefix, sep='\t', index=False)
    colors = iter([dict(s=10, c='r', marker='p'), dict(s=15, c='b', marker='*'),
                   dict(s=7, c='m', marker=','), dict(s=6, c='g', marker='o'),
                   dict(s=5, c='c', marker='v'), dict(s=4, c='k', marker='^'),
                   dict(s=3, c='y', marker='>'), dict(s=2, c='0.6', marker='<')]
                  )
    f, ax = plt.subplots()
    for t, df in res.groupby('type'):
        color = next(colors)
        df.plot(x='Number of SNPs', y='R2', kind='scatter', legend=True, ax=ax,
                label=t, **color)
                #s=3, c=color)
    # ax.axhline(best_r2, ls='--ls', c='0.5')
    plt.tight_layout()
    plt.savefig('%s_transferability.pdf' % args.prefix)
    print(
        'ESE comparison done after %.2f minutes' % ((time.time() - now) / 60.))


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
    parser.add_argument('-b', '--refgeno', required=True,
                        help=('prefix of the bed fileset in reference'))
    parser.add_argument('-g', '--targeno', required=True,
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
    parser.add_argument('--freq_thresh', type=float, help='filter by mafs',
                        default=-1)
    parser.add_argument('--within', default=0, type=int,
                        help='0=cross; 1=reference; 2=target')
    parser.add_argument('--ld_operator', default='lt')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--reference', action='store_true',
                        help='use reference for computations')

    args = parser.parse_args()
    main(args)
