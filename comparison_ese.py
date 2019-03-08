#!/usr/bin/env python
# coding:utf-8
"""
  Author: Jose Sergio Hleap  --<2017>
  Purpose: pipeline to test the expected squared effect strategies
  on the optimized ranking
  Created: 10/02/17
"""
from itertools import product
from operator import itemgetter
from ese import *

np.seterr(all='raise')  # Debugging
within_dict = {0: 'ese cotag', 1: 'ese EUR', 2: 'ese AFR'}
prunestep = 30


def sortbylocus(prefix, df, column='ese', title=None, ascending=False,
                plot=False):
    #picklefile = '%s.pickle' % prefix
    sort_columns = [column, 'beta_sq', 'pos']
    # if os.path.isfile(picklefile):
    #     with open(picklefile, 'rb') as F:
    #         sorteddf = pickle.load(F)
    # else:
    df['m_size'] = norm(abs(df.slope.copy()), 20, 200)
    if 'beta_sq' not in df.columns:
        df['beta_sq'] = df.slope**2
    df = df.sort_values(by=sort_columns, ascending=[ascending, False, True])
    try:
        grouped = df.groupby('locus', as_index=False)
    except:
        grouped = df.groupby('locus_ese', as_index=False)
    try:
        if ascending:
            grouped = grouped.apply(lambda grp: grp.nsmallest(1, column))
        else:
            grouped = grouped.apply(lambda grp: grp.nlargest(1, column))
    except TypeError:
        grouped = grouped.apply(lambda grp: grp.sort_values(
            by=column, ascending=ascending).iloc[0])
    sorteddf = grouped.sort_values(by=sort_columns, ascending=[ascending,
                                                               False, True])
    tail = df[~df.snp.isin(sorteddf.snp)]
    # grouped = tail.groupby('locus', as_index=False)
    if not tail.empty:
        sorteddf = sorteddf.append(
            tail.sort_values(by=sort_columns, ascending=[ascending, False,
                                                         True]))
    sorteddf = sorteddf.reset_index(drop=True)
    sorteddf['index'] = sorteddf.index.tolist()
    # with open(picklefile, 'wb') as F:
    #     pickle.dump(sorteddf, F)
    size = sorteddf.m_size
    # make sure x and y are numeric
    sorteddf['pos'] = pd.to_numeric(sorteddf.pos)
    sorteddf['index'] = pd.to_numeric(sorteddf.index)
    #print(sorteddf.head())
    if plot:
        idx = sorteddf.dropna(subset=['beta']).index.tolist()
        causals = sorteddf.loc[idx, :]
        f, ax = plt.subplots()
        sorteddf.plot.scatter(x='pos', y='index', ax=ax, label=column,
                              s=size.values
                              )
        if not causals.empty:
            causals.plot.scatter(x='pos', y='index', c='k', marker='*', ax=ax,
                                 label='Causals', s=size[idx].values)
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        plt.savefig('%s.pdf' % prefix)
        plt.close()
    #
    return sorteddf


def individual_ese(sumstats, avh2, h2, n, within, loci, tgeno, tpheno, threads,
                   tbim, prefix, memory, pedestrian=False, prune_n=None):
    cache = Chest(available_memory=memory)
    # if pedestrian:
    #     func_per_locus = per_locus
    # else:
    #     func_per_locus = per_locus
    within_str = within_dict[within]
    prefix = '%s_%s' % (prefix, '_'.join(within_str.split()))
    print('Compute expected beta square per locus...')
    resfile = '%s_res.tsv' % prefix
    if not os.path.isfile(resfile):
        delayed_results = [
            dask.delayed(per_locus)(locus, sumstats, avh2, h2, n, i,
                                                within=within) for i, locus in
            enumerate(loci)]
        dask_options = dict(num_workers=threads, memory_limit=memory,
                            cache=cache, pool=ThreadPool(threads))
        with ProgressBar(), dask.set_options(**dask_options):
            res = list(dask.compute(*delayed_results))
        res = pd.concat(res)  # , ignore_index=True)
        result = res.merge(
            sumstats.reindex(columns=['slope', 'snp', 'beta', 'pos']), on='snp')
        result = result.rename(columns={'ese': within_str})
        if 'i' not in result.columns:
            result = result.merge(tbim.reindex(columns=['snp', 'i']), on='snp')
        if 'slope' not in result.columns:
            result = result.merge(sumstats.reindex(columns=['snp', 'slope']),
                                  on='snp')
        prod = sortbylocus(prefix, result, column=within_str,
                           title=r'Realized $h^2$: %f' % h2)
        prod.to_csv('df' + resfile, index=False, sep='\t')
        prod = prune_it(prod, tgeno, tpheno, within_str, step=prunestep,
                        threads=threads, max_memory=memory, n=prune_n)
        prod.to_csv(resfile, index=False, sep='\t')
    else:
        prod = pd.read_csv(resfile, sep='\t')
    return prod


def get_tagged(snp_list, D_r, ld_thr, p_thresh, sumstats):
    index = []
    high=[]
    ippend = index.append
    tag = []
    text = tag.extend
    # sort just once
    sumstats = sumstats[sumstats.snp.isin(snp_list)].sort_values(
        ['pvalue', 'beta_sq', 'pos'], ascending=[True, False, True])
    if any([isinstance(x,str) for x in sumstats.pvalue]):
        sumstats.loc[:, 'pvalue'] = [mp.mpf(i) for i in  sumstats.pvalue]
    total_snps = sumstats.shape[0]
    while len(index + tag) != total_snps:
        curr_high = sumstats.iloc[0]
        if mp.mpf(curr_high.pvalue) < p_thresh:
            curr_high = curr_high.snp
            high.append(curr_high)
            chidx = np.where(D_r.columns == curr_high)[0]
            # get snps in LD
            vec = D_r.loc[chidx, :] # Is in the row since is square and rows are
            #  index while columns aren't
            tagged = vec[vec > ld_thr].columns.tolist()
            if curr_high in tagged: # TODO: it is necessary.. possible bug
                tagged.pop(tagged.index(curr_high))
            text(tagged)
            ippend((curr_high, tagged))
        else:
            low = sumstats.snp.tolist()
            text(low)
        sumstats = sumstats[~sumstats.snp.isin(high + tag)]
    return index, tag, high


@jit
def loop_pairs(snp_list, D_r, l, p, sumstats, pheno, geno):
    index, tag, high = get_tagged(snp_list, D_r, l, p, sumstats)
    clump = sumstats[sumstats.snp.isin(high)]
    idx = clump.i.values.astype(int)
    prs = geno[:, idx].dot(clump.slope)
    est = np.corrcoef(prs, pheno.PHENO)[1, 0] ** 2
    return est, (index, tag, l, p)

def dirty_ppt(loci, sumstats, geno, pheno, threads, split, seed, memory,
              pvals=None, lds=None):
    cache = Chest(available_memory=memory)
    now = time.time()
    if not 'beta_sq' in sumstats.columns:
        try:
            sumstats.loc[: , 'beta_sq'] = sumstats.slope ** 2
        except TypeError:
            sumstats['beta_sq'] = [mp.mpf(x) ** 2 for x in sumstats.slope]

    print('Starting dirty PPT...')
    index, tag = [], []
    if split > 1:
        x_train, x_test, y_train, y_test = train_test_split(geno, pheno,
                                                            test_size=1 / split,
                                                            random_state=seed)
    else:
        x_train, x_test, y_train, y_test = geno, geno, pheno, pheno
    pos =[]
    for r, locus in enumerate(loci):
        snps, D_r, D_t = locus
        snp_list = snps.tolist()
        D_r = dd.from_dask_array(D_r, columns=snps) ** 2
        sub = sumstats[sumstats.snp.isin(snps)].reindex(
            columns=['snp', 'slope', 'beta_sq', 'pvalue', 'i', 'pos', 'beta'])
        sub['locus'] = r
        # filter pvalue
        if pvals is None:
            pvals = [1, 0.5, 0.2, 0.05, 10E-3, 10E-5, 10E-7, 1E-9]
        if lds is None:
            lds = np.arange(0.1, 0.8, 0.1)
        pairs = product(pvals, lds)
        delayed_results = [
            dask.delayed(loop_pairs)(snp_list, D_r, l, p, sub, y_train, x_train)
            for p, l in pairs]
        dask_options = dict(num_workers=threads, cache=cache,
                            pool=ThreadPool(threads))
        with ProgressBar(), dask.set_options(**dask_options):
            print('    Locus', r)
            d = dict(list(dask.compute(*delayed_results)))
            best_key = max(d.keys())
            i, t, ld, pv = d[best_key]
            clump = sub[sub.snp.isin([x[0] for x in i])]
            clump['tag'] = ';'.join([';'.join(x[1]) for x in i])
            index.append(clump)
            tag += t
        pos.append(sub[sub.snp.isin(tag)])
    pre = pd.concat(index).sort_values(['pvalue', 'pos'], ascending=True)
    pre.to_csv('PPT.clumped.tsv', sep='\t', index=False)
    pos = pd.concat(pos).sort_values(['pvalue', 'pos'], ascending=True)
    ppt = pre.append(pos, ignore_index=True).reset_index(drop=True)
    ppt['index'] = ppt.index.tolist()
    print('Dirty ppt done after %.2f minutes' % ((time.time() - now) / 60.))
    return ppt, pre, pos, x_test, y_test


def main(args):
    # Set CPU limits
    # soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    # print('Processor limits are:', soft, hard)
    # resource.setrlimit(resource.RLIMIT_NPROC, (args.threads, hard))
    # soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    # print('Soft limit changed to :', soft)

    now = time.time()
    seed = np.random.randint(1e4) if args.seed is None else args.seed
    # set Cache to protect memory spilling
    if args.maxmem is not None:
        memory = args.maxmem
    else:
        memory = psutil.virtual_memory().available
    cache = Chest(available_memory=memory)
    refl, tarl = args.labels
    # make simulations
    print('Simulating phenotype for reference population %s \n' % refl)
    opts = {'outprefix': refl, 'bfile': args.refgeno, 'h2': args.h2,
            'ncausal': args.ncausal, 'normalize': args.normalize,
            'uniform': args.uniform, 'snps': None, 'seed': seed,
            'bfile2': args.targeno, 'flip': args.gflip,
            'max_memory': args.maxmem, 'freq_thresh': args.freq_thresh,
            'remove_causals':args.avoid_causals}
    rpheno, h2, (rgeno, rbim, rtruebeta, rcausals) = qtraits_simulation(**opts)
    # make simulation for target
    print('Simulating phenotype for target population %s \n' % tarl)
    opts.update(dict(outprefix=tarl, bfile=args.targeno, bfile2=args.refgeno,
                     causaleff=rcausals, validate=args.split))
    if args.reference:
        tpheno, tgeno = rpheno, rgeno
    else:
        tpheno, h2, (tgeno, tbim, truebeta, tcausals) = qtraits_simulation(
            **opts)
    opts.update(dict(prefix='ranumo_gwas', pheno=rpheno, geno=rgeno, bim=rbim,
                     validate=None if args.split <= 1 else args.split,
                     threads=args.threads, flip=args.flip,
                     high_precision=args.high_precision))
    # perform GWAS
    sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    try:
        sumstats['beta_sq'] = sumstats.slope ** 2
    except TypeError:
        sumstats['beta_sq'] = [mp.mpf(x) ** 2 for x in sumstats.slope]
        sumstats['pvalue'] = [mp.mpf(x) for x in sumstats.pvalue]
    sum_snps = sumstats.snp.tolist()
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=args.window, justd=True,
                  max_memory=memory, threads=args.threads, extend=args.extend)
    avh2 = h2 / len(sum_snps)
    n = tgeno.shape[0]
    scs_title = r'Realized $h^2$: %f' % h2
    # compute ese only integral
    intfile = '%s_%s_res.tsv' % (args.prefix, 'integral')
    if not os.path.isfile(intfile):
        print('Compute ese only integral')
        delayed_results = [
            dask.delayed(per_locus)(locus, sumstats, avh2, h2, n, i,
                                    integral_only=True) for i, locus in
            enumerate(loci)]
        dask_options = dict(num_workers=args.threads, cache=cache,
                            pool=ThreadPool(args.threads))
        with ProgressBar(), dask.set_options(**dask_options):
            integral = list(dask.compute(*delayed_results))
        integral = pd.concat(integral, ignore_index=True)
        integral = integral.merge(sumstats.reindex(
            columns=['snp', 'pvalue', 'beta_sq', 'slope', 'pos', 'i', 'beta']),
            on='snp')
        integral_df = sortbylocus('%s_integral' % args.prefix, integral,
                                  column='ese', title='Integral; %s' % scs_title
                                  )
        assert integral_df.shape[0] == sumstats.shape[0]
        integral = prune_it(integral_df, tgeno, tpheno, 'Integral',
                            step=prunestep, threads=args.threads,
                            max_memory=memory)
        integral_df.to_csv('df_' + intfile, index=False, sep='\t')
        integral.to_csv(intfile, index=False, sep='\t')
        # plot beta_sq vs integral
        inte = integral_df.reindex(columns=['snp', 'ese', 'beta_sq']).rename(
            columns={'ese': 'integral'})
        f, ax = plt.subplots()
        try:
            inte.plot.scatter(x='beta_sq', y='integral', ax=ax)
        except ValueError:
            ax.scatter(x=inte.beta_sq.values, y=inte.integral.values.astype(
                np.float64))
        plt.tight_layout()
        plt.savefig('%s_betasqvsintegral.pdf' % args.prefix)
        plt.close()
    else:
        integral_df = pd.read_csv('df_' + intfile, sep='\t')
        integral = pd.read_csv(intfile, sep='\t')
    # include ppt
    pptfile = '%s_%s_res.tsv' % (args.prefix, 'ppt')
    if not os.path.isfile(pptfile):
        out = dirty_ppt(loci, integral_df, rgeno, rpheno, args.threads,
                        args.split, seed, memory, pvals=args.p_thresh,
                        lds=args.r_range)
        ppt_df, _, _, x_test, y_test = out
        assert ppt_df.shape[0] == sumstats.shape[0]
        ppt = prune_it(ppt_df, tgeno, tpheno, 'P + T', step=prunestep,
                       threads=args.threads, max_memory=memory)
        ppt.to_csv(pptfile, index=False, sep='\t')
    else:
        ppt = pd.read_csv(pptfile, sep='\t')
    ppt.plot(x='Number of SNPs', y='R2', kind='scatter', legend=True, s=3)
    plt.savefig('ppt_afr.pdf')
    plt.close()
    # expecetd square effect
    eses = [individual_ese(sumstats, avh2, h2, n, x, loci, tgeno, tpheno,
                           args.threads, tbim, args.prefix, memory,
                           pedestrian=args.pedestrian) for x in [0, 1, 2]]
    # prune by pval
    resfile = '%s_%s_res.tsv' % (args.prefix, 'pvalue')
    if not os.path.isfile(resfile):
        sub = integral_df.reindex(columns=['locus', 'snp', 'pvalue', 'beta_sq',
                                           'slope', 'pos', 'beta', 'i'])
        pval = sub.sort_values(['pvalue', 'beta_sq'],
                               ascending=[True, False]).reset_index(drop=True)
        pval['index'] = pval.index
        # pval = sortbylocus('%s_pvalue' % args.prefix, pval, column='pvalue',
        #                       title='P-value; %s' % scs_title, ascending=True)
        # pval, _ = smartcotagsort('%s_pval' % args.prefix, pval, column='index',
        #                          ascending=True)
        pval.to_csv('df_%s' % resfile, index=False, sep='\t')
        assert pval.shape[0] == sumstats.shape[0]
        pval = prune_it(pval, tgeno, tpheno, 'pval', step=prunestep,
                        threads=args.threads, max_memory=memory)
        pval.to_csv(resfile, index=False, sep='\t')
    else:
        pval = pd.read_csv(resfile, sep='\t')
    # prune by estimated beta
    betafile = '%s_%s_res.tsv' % (args.prefix, 'slope')
    if not os.path.isfile(betafile):
        m = integral_df.reindex(columns=['snp', 'locus'])
        beta = sumstats.merge(m, on='snp')
        # beta, _ = smartcotagsort('%s_slope' % args.prefix, beta,
        #                          column='beta_sq', ascending=False,
        #                          title=scs_title)
        beta_df = sortbylocus('%s_betasq' % args.prefix, beta, column='beta_sq',
                              title=r'$\hat{\beta}^2$; %s' % scs_title)
        beta_df.to_csv('df_' + betafile, index=False, sep='\t')
        assert beta_df.shape[0] == sumstats.shape[0]
        #assert beta_df.iloc[0].snp == integral_df.iloc[0].snp
        beta = prune_it(beta_df, tgeno, tpheno, r'$\hat{\beta}^2$',
                        step=prunestep, threads=args.threads, max_memory=memory)
        beta.to_csv(betafile, index=False, sep='\t')
        # plot beta_sq vs pval
        try:
            sumstats['-log(pval)'] = -np.log10(sumstats.pvalue.values)
        except AttributeError:
            sumstats['-log(pval)'] = [float(-mp.log10(x)) for x in
                                      sumstats.pvalue]
        f, ax = plt.subplots()
        sumstats.plot.scatter(x='beta_sq', y='-log(pval)', ax=ax)
        plt.tight_layout()
        plt.savefig('%s_betasqvspval.pdf' % args.prefix)
        plt.close()
    else:
        beta = pd.read_csv(betafile, sep='\t')
    # include causals
    causals = integral_df.dropna(subset=['beta'])
    if not causals.empty:
        # caus, _ = smartcotagsort('%s_causal' % args.prefix, causals, column='beta',
        #                          ascending=False, title='Causals; %s' % scs_title)
        caus = sortbylocus('%s_causals' % args.prefix, causals, column='beta',
                          title='Causals; %s' % scs_title)
        caus = prune_it(caus, tgeno, tpheno, 'Causals', step=prunestep,
                        threads=args.threads, max_memory=memory)
        caus.to_csv('%s_%s_res.tsv' % (args.prefix, 'causal'), index=False,
                    sep='\t')
    else:
        caus = None
    # plot them
    res = pd.concat(eses + [pval, beta, caus, ppt, integral])
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
    plt.close()
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
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--uniform', default=False, action='store_true')
    parser.add_argument('--split', default=2, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--r_range', default=None, nargs=3,
                        action=Store_as_arange, type=float)
    parser.add_argument('--p_thresh', default=None, nargs='+',
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
    parser.add_argument('-A', '--avoid_causals', default=False,
                        action='store_true', help='Remove causals from set')
    parser.add_argument('--pca', default=None, type=int)
    parser.add_argument('-E', '--extend', default=False, action='store_true')
    parser.add_argument('--high_precision', default=False, action='store_true')
    parser.add_argument('--pedestrian', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
