#!/usr/bin/env python
# coding:utf-8
"""
  Author: Jose Sergio Hleap  --<2017>
  Purpose: Optimize the mixing of cotagging and P + T and chose the snps based
  on the optimized ranking
  Created: 10/02/17
"""

from tqdm import tqdm

from ppt import pplust
from simple_GWAS import *

matplotlib.use('Agg')
plt.style.use('ggplot')


# ----------------------------------------------------------------------
def strategy_hyperbola(x, y, alpha):
    """
    strategy for new rank with summation

    :param :class pd.Series x: Series with the first range to be combined
    :param :class pd.Series y: Series with the second range to be combined
    :param float alpha: Float with the weight to be combined by
    """
    den = (alpha / x) + ((1 - alpha) / y)
    return 1 / den


# ----------------------------------------------------------------------
@jit
def strategy_sum(x, y, alpha):
    """
    strategy for new rank with hypergeometry

    :param :class pd.Series x: Series with the first range to be combined
    :param :class pd.Series y: Series with the second range to be combined
    :param float alpha: Float with the weight to be combined by
    """
    rank = (alpha * x) + ((1 - alpha) * y)
    alphas = [alpha] * len(rank)
    return pd.DataFrame({'New_rank': rank, 'alpha': alphas})


# ---------------------------------------------------------------------------
def optimize_alpha(prefix, bfile, pheno, merged, alphastep, prune_step=1,
                   threads=1, index_type='cotag'):
    """
    Do a line search for the best alpha in nrank = alpha*rankP+T + (1-alpha)*cot

    :param str prefix: Prefix for outputs
    :param dask bfile: Dask array with genotype
    :param :class pd.Dataframe sorted_cotag:Filename with results in the sorting
    :param :class pd.Dataframe clumpe: Dataframe of merged P+T results
    :param str sumstats: File with the summary statistics
    :param :class pd.Dataframe phenofile: Dataframe with the phenotype
    :param float alphastep: Step of the alpha to be explored
    :param str tar: Label of target population
    :param int prune_step: Step of the prunning
    :param int trheads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    """
    # Set output name
    outfn = '%s_optimized.tsv' % prefix
    picklfn = '%s_optimized.pickle' % prefix
    # Execute the ranking
    if not os.path.isfile(picklfn):
        # Generate the alpha-space
        space = np.concatenate(
            (np.array([0, 0.05]), np.arange(0.1, 1 + alphastep,
                                            alphastep)))
        x = merged.loc[:, 'index_PpT']
        y = merged.loc[:, 'index_%s' % index_type]
        print('computing weighted sum')
        res = Parallel(n_jobs=int(threads))(
            delayed(strategy_sum)(x, y, alpha) for alpha in tqdm(space))
        df = pd.concat(res)
        # score it
        opts = dict(step=int(prune_step), threads=threads)
        scored = pd.DataFrame()
        datas = pd.DataFrame()
        grouped = df.groupby('alpha')
        for label, d in grouped:
            data = pd.concat((merged, d), axis=1).sort_values(by='New_rank')
            data['alpha'] = label
            datas = datas.append(data)
            scored = scored.append(
                prune_it(data, bfile, pheno, label, **opts))

        scored.sort_values(by='R2', ascending=False).to_csv(outfn, sep='\t',
                                                            index=False)
        with open(picklfn, 'wb') as F:
            pickle.dump((df, scored, grouped, datas), F)
    else:
        # df = pd.read_table(outfn, sep='\t')
        with open(picklfn, 'rb') as F:
            df, scored, grouped, datas = pickle.load(F)
    # Plot the optimization
    scored.rename(columns={'type': 'alpha'}, inplace=True)
    piv = scored.reindex(columns=['Number of SNPs', 'alpha', 'R2'])
    piv = piv.pivot(index='Number of SNPs', columns='alpha',
                    values='R2').sort_index()
    piv.plot(colormap='copper', alpha=0.5)
    plt.ylabel('$R^2$')
    plt.tight_layout()
    plt.savefig('%s_alphas.pdf' % (prefix))
    plt.close()
    # Returned the sorted result dataframe
    results = scored.sort_values('R2', ascending=False).reset_index(drop=True)
    gr = datas.groupby('alpha')
    best = gr.get_group(results.iloc[0].alpha)
    return results, best


# ----------------------------------------------------------------------
def prankcster(prefix, tbed, rbed, tpheno, labels, alpha_step, prune_step,
               cotag=None, freq_threshold=0.01, splits=3, threads=1, seed=None,
               max_memory= None, **kwargs):
    seed = np.random.randint(1e4) if seed is None else seed
    print('Performing prankcster')
    # Unpack population labels
    refl, tarl = labels
    # check if phenotype is provided
    if tpheno is None:
        # make simulation for reference
        print('Simulating phenotype for reference population %s \n' % refl)
        opts = {'outprefix': refl, 'bfile': rbed, 'h2': kwargs['h2'],
                'ncausal': kwargs['ncausal'], 'normalize': kwargs['normalize'],
                'uniform': kwargs['uniform'], 'snps': None, 'seed': seed,
                'bfile2': tbed, 'f_thr': freq_threshold, 'max_memory':max_memory
                }
        rpheno, h2, (rgeno, rbim, rtruebeta, rvec) = qtraits_simulation(**opts)
        # make simulation for target
        print('Simulating phenotype for target population %s \n' % tarl)
        opts.update(dict(outprefix=tarl, bfile=tbed, causaleff=rbim.dropna(),
                         bfile2=rbed))
        tpheno, h2, (tgeno, tbim, ttruebeta, tvec) = qtraits_simulation(**opts)
        opts.update(dict(prefix='ranumo_gwas', pheno=rpheno, geno=rgeno,
                         validate=3, threads=threads, bim=rbim))
        sumstats, X_train, X_test, Y_train, Y_test = plink_free_gwas(**opts)
    # Read summary statistics
    elif ('sumstats' in kwargs) and isinstance(kwargs['sumstats'], str):
        sumstats = pd.read_table(kwargs['sumstats'], delim_whitespace=True)
        rgeno = rbed
        tgeno = tbed
    else:
        rgeno = rbed
        tgeno = tbed
        sumstats = kwargs['sumstats']
        tbim = kwargs['tbim']
        rbim = kwargs['rbim']
        X_test = kwargs['X_test']
        Y_test = kwargs['Y_test']

    # Read the cotag scores
    if os.path.isfile('%s_cotags.tsv' % prefix):
        cotags = pd.read_table('%s_cotags.tsv' % prefix, sep='\t')
    elif isinstance(cotag, str):
        sorted_cotag = pd.read_table(cotag, sep='\t')
    elif cotag is None:
        try:
            assert 'sumstats' in kwargs
        except AssertionError:
            print('Please provide the sumary statistics as a keyword argument')
        cotags = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=kwargs['window'],
                        threads=threads)
        cotags.to_csv('%s_cotags.tsv' % prefix, sep='\t', index=False)
    sorted_cotag, _ = smartcotagsort(prefix, cotags, column='cotag')
    sorted_cotag = sorted_cotag.merge(
        sumstats.reindex(columns=['snp', 'i', 'slope']), on='snp')
    sorted_tagr, _ = smartcotagsort(prefix, cotags, column='ref')
    sorted_tagr = sorted_tagr.merge(
        sumstats.reindex(columns=['snp', 'i', 'slope']), on='snp')
    sorted_tagt, _ = smartcotagsort(prefix, cotags, column='tar')
    sorted_tagt = sorted_tagt.merge(
        sumstats.reindex(columns=['snp', 'i', 'slope']), on='snp')
    # Read and sort the P + T results
    if os.path.isfile('%s.sorted_ppt' % tarl):
        clumpetar = pd.read_table('%s.sorted_ppt' % tarl, sep='\t')
    else:
        clumpetar = \
        pplust('%s_ppt' % tarl, tgeno, tpheno, sumstats, kwargs['r_range'],
               kwargs['p_tresh'], bim=tbim, ld_operator=kwargs['ld_operator'],
               graph=kwargs['graph'])[-1]
    if os.path.isfile('%s.sorted_ppt' % refl):
        clumperef = pd.read_table('%s.sorted_ppt' % refl, sep='\t')
    else:
        clumperef = \
        pplust('%s_ppt' % refl, X_test, Y_test, sumstats, kwargs['r_range'],
               kwargs['p_tresh'], bim=tbim, ld_operator=kwargs['ld_operator'],
               graph=kwargs['graph'])[-1]
    # clumperef = clumperef[clumperef.SNP.isin(frqs.SNP)]
    clumpe = clumpetar.merge(clumperef, on='snp', suffixes=['_%sPpT' % tarl,
                                                            '_%sPpT' % refl])
    # Merge the sorted cotag and the P+T
    cols = ['snp']
    if 'i' in clumpetar.columns and 'i' in sorted_cotag.columns:
        cols += ['i']
    if 'slope' in clumpetar.columns and 'slope' in sorted_cotag.columns:
        cols += ['slope']
        test = sorted_cotag.merge(clumpetar, on='snp', suffixes=['_cotag',
                                                                 '_PpT'])
        assert all(test.slope_PpT == test.slope_cotag)
    i_s = tbim.reindex(columns=['snp', 'i'])
    merge = sorted_cotag.merge(clumpetar, on=cols, suffixes=['_cotag', '_PpT'])
    if 'i' not in merge.columns:
        merge = merge.merge(i_s, on='snp')
    mtagr = sorted_tagr.merge(clumpetar, on=cols, suffixes=['_tagr', '_PpT'])
    if 'i' not in mtagr.columns:
        mtagr = mtagr.merge(i_s, on='snp')
    mtagt = sorted_tagt.merge(clumpetar, on=cols, suffixes=['_tagt', '_PpT'])
    if 'i' not in mtagt.columns:
        mtagt = mtagt.merge(i_s, on='snp')
    # merge = merge.rename(columns={'Index': 'Index_Cotag'})
    # Create crossvalidation
    x_train, x_test, y_train, y_test = train_test_split(tgeno, tpheno,
                                                        test_size=1 / splits,
                                                        random_state=seed)
    # Optimize the alphas
    z = zip(['cotag', 'tagr', 'tagt'], [merge, mtagr, mtagt])
    todos = {
        t[0]: optimize_alpha('%s_%s' % (prefix, t[0]), x_train, y_train, t[1],
                             alpha_step, prune_step, threads, t[0]) for t in z}
    # results, best_alpha = optimize_alpha(prefix, x_train, y_train, merge,
    #                                      alpha_step, prune_step, threads)
    # Score with test-set
    opts = dict(step=prune_step, threads=threads)
    # prune tags
    # TODO: Check P + T seems that complete LD is not being clumped
    pre = [prune_it(clumperef, X_test, Y_test, 'PPT %s' % refl, **opts),
           prune_it(clumpetar, x_test, y_test, 'PPT %s' % tarl, **opts),
           prune_it(sorted_cotag, x_test, y_test, 'CoTagging', **opts),
           prune_it(sorted_tagr, x_test, y_test, 'Tagging %s' % refl, **opts),
           prune_it(sorted_tagt, x_test, y_test, 'Tagging %s' % tarl, **opts)]
    with open('%s_merged.pickle' % prefix, 'wb') as F:
        pickle.dump(pre, F)
    dfs = []
    for lab, best_alpha in todos.items():
        df = best_alpha[1]
        if 'i' not in df.columns:
            df = df.merge(i_s, on='snp')
        alpha = prune_it(df, x_test, y_test, 'hybrid %s' % lab,
                         **opts)
        res = pd.concat([alpha] + pre)
        dfs.append((lab, res))
    # plot
    for lab, res in dfs:
        colors = iter(['r', 'b', 'm', 'g', 'c', 'k', 'y'])
        f, ax = plt.subplots()
        for t, df in res.groupby('type'):
            df.plot(x='Number of SNPs', y='R2', kind='scatter', legend=True,
                    s=3, c=next(colors), ax=ax, label=t)
        plt.tight_layout()
        plt.savefig('%s_%s_prankcster.pdf' % (prefix, lab))
        plt.close()
        res.to_csv('%s_%s_cotag.tsv' % (prefix, lab), sep='\t', index=False)
    return pre


if __name__ == '__main__':
    class Store_as_arange(argparse._StoreAction):
        def __call__(self, parser, namespace, values, option_string=None):
            values = np.arange(values[0], values[1], values[2])
            return super().__call__(parser, namespace, values, option_string)


    class Store_as_array(argparse._StoreAction):
        def __call__(self, parser, namespace, values, option_string=None):
            values = np.array(values)
            return super().__call__(parser, namespace, values, option_string)


    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-a', '--allele_file', default='EUR.allele',
                        help='File with the allele order. A1 in position 3 and '
                             'id in position2')
    parser.add_argument('-b', '--reference', required=True,
                        help=('prefix of the bed fileset in reference'))
    parser.add_argument('-c', '--target', required=True,
                        help=('prefix of the bed fileset in target'))
    parser.add_argument('-L', '--labels', nargs=2,
                        help=('Space separated string with labels of reference '
                              'and target populations'))
    parser.add_argument('-T', '--target_ppt', default=None,
                        help=('Filename of the results of a PPT run'))
    parser.add_argument('-r', '--ref_ppt', default=None,
                        help=('Filename with results for the P+Toptimization in'
                              ' the reference population'))
    parser.add_argument('-R', '--sortresults', help=(
        'Filename with results in the sorting inlcuding path'))
    parser.add_argument('-d', '--cotagfn', default=None,
                        help=('Filename tsv with cotag results'))
    parser.add_argument('-s', '--sumstats', default=None,
                        help=('Filename of the summary statistics in plink '
                              'format'))
    parser.add_argument('-f', '--pheno', default=None,
                        help=('Filename of the true phenotype of the target '
                              'population'))
    parser.add_argument('-S', '--alpha_step', default=0.1, type=float,
                        help=('Step for the granularity of the grid search.'))
    parser.add_argument('-E', '--prune_step', default=1, type=float,
                        help=('Percentage of snps to be tested at each step'))
    parser.add_argument('-v', '--splits', help='Number of folds for cross-val',
                        default=5, type=int)
    parser.add_argument('-C', '--column', help='Column to sort by',
                        default='Cotagging')
    parser.add_argument('-w', '--weight', default=False, action='store_true',
                        help=('Perform the sorting based on the weighted square'
                              ' effect sizes'))
    parser.add_argument('-y', '--every', action='store_true', default=False)
    parser.add_argument('-t', '--threads', default=1, action='store', type=int)
    parser.add_argument('-H', '--h2', default=0.66, type=float,
                        help=('Heritability of the simulated phenotype'))
    parser.add_argument('-M', '--maxmem', default=None, type=int)
    parser.add_argument('-F', '--freq_threshold', default=0.1, type=float)
    parser.add_argument('-Q', '--qrangefn', default=None, help=(
        'Specific pre-made qrange file'))
    parser.add_argument('-g', '--strategy', default='sum', help=(
        'Strategy to produce the hybrid measure. Currently available is '
        'weighted sum (sum)'))
    parser.add_argument('--window', default=1000, help='kbwindow for ld',
                        type=int)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--ncausal', default=200, type=int)
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--uniform', default=False, action='store_true')
    parser.add_argument('--r_range', default=None, nargs=3,
                        action=Store_as_arange, type=float)
    parser.add_argument('--p_tresh', default=None, nargs='+',
                        action=Store_as_array, type=float)
    parser.add_argument('--ld_operator', default='lt')
    parser.add_argument('--graph', action='store_true')
    args = parser.parse_args()

    prankcster(args.prefix, args.target, args.reference, args.pheno,
               args.labels, args.alpha_step, args.prune_step,
               cotag=args.cotagfn, freq_threshold=args.freq_threshold,
               splits=args.splits, seed=args.seed, h2=args.h2,
               ncausal=args.ncausal, normalize=args.normalize,
               uniform=args.uniform, r_range=args.r_range, p_tresh=args.p_tresh,
               window=args.window, max_memory=args.maxmem,
               ld_operator=args.ld_operator, graph=args.graph)