#!/usr/bin/env python
# coding:utf-8
"""
  Author:   Jose Sergio Hleap --<2017>
  Purpose: From SumStats get the best combination of R2 and P-thresholding P + T
  Created: 10/01/17
"""
from itertools import product
from operator import itemgetter

import igraph
import matplotlib

matplotlib.use('Agg')
from qtraitsimulation_old import *
from simple_GWAS import *
import gc
from Chest import Chest
from tempfile import tempdir
import operator
plt.style.use('ggplot')


# ----------------------------------------------------------------------
@jit
def single_clump(df, R2, block, r_thresh, field='pvalue', ld_operation='lt'):
    """
    Process single clump to get index and tagged snps

    :param df: Dataframe with the sorting field per snp
    :param R2: LD information for the snps in df
    :param block: number of the block being processed
    :param r_thresh: ld threshold to be used
    :param field: Field in df to be thresholded
    :param ld_operation: Operation on the ld threshold (greater than, less than)
    :return:
    """
    print('\t Processing block', block)
    op = {'lt': operator.lt, 'gt': operator.gt}
    out = {}
    r2 = R2[block]
    # generate a graph
    g = igraph.Graph.Adjacency((op[ld_operation](r2,  r_thresh)).values.astype(
        int).tolist(), mode='UPPER')
    g.vs['label'] = r2.columns.tolist()
    sg = g.components().subgraphs()
    for l in sg:
        snps = l.vs['label']
        sub = df[df.snp.isin(snps)].sort_values(by=field, ascending=True)
        if not sub.empty:
            if sub.shape[0] > 1:
                values = sub.iloc[1:].snp.tolist()
            else:
                values = []
            out[sub.iloc[0].snp] = values
    # Help garbage collection
    gc.collect()
    return out


# ----------------------------------------------------------------------
def clump(R2, sumstats, r_thr, p_thr, threads, field='pvalue', max_memory=None,
          ld_operator='lt'):
    print('    Clumping with R2 %.2g and P_thresh of %.2g, with operator %s' % (
        r_thr, p_thr, ld_operator))
    sub = sumstats[sumstats.loc[:, field] < p_thr]
    delayed_results = [
        dask.delayed(single_clump)(df, R2, block, r_thr, field, ld_operator) for
        block, df in sub.groupby('block')]
    if max_memory is not None:
        cache = Chest(path=tempdir, available_memory=max_memory)
        r = list(dask.compute(*delayed_results, num_workers=threads,
                              cache=cache))
    else:
        r = list(dask.compute(*delayed_results, num_workers=threads))
    clumps = dict(pair for d in r for pair in d.items())
    del r
    gc.collect()
    # set dataframe
    cols = ['snp', field, 'slope', 'i']
    df = sub[sub.snp.isin(list(clumps.keys()))].reindex(columns=cols)
    # set a dataframe compatible with ranumo
    df2 = df.sort_values(by=field)
    df['Tagged'] = [';'.join(clumps[v]) for v in df.snp]
    del clumps
    gc.collect()
    tail = sumstats[~sumstats.snp.isin(df2.snp)].reindex(columns=cols)
    df2 = df2.append(tail).reset_index(drop=True)
    df2['index'] = df2.index.tolist()
    return df, df2


# ----------------------------------------------------------------------
def new_plot(prefix, ppt, geno, pheno, threads):
    params = dict(column='index', ascending=True)
    ppt, _ = smartcotagsort(prefix, ppt, **params)
    ppt = prune_it(ppt, geno, pheno, 'P+T %s' % prefix, threads=threads)
    f, ax = plt.subplots()
    ppt.plot(x='Number of SNPs', y='R2', kind='scatter', s=5, ax=ax,
             legend=True)
    ax.set_ylabel(r'$R^2$')
    plt.tight_layout()
    plt.savefig('%s_ppt.pdf' % prefix)
    plt.close()


# ----------------------------------------------------------------------
def score(geno, pheno, sumstats, r_t, p_t, R2, threads, approx=False,
          field='pvalue', max_memory=None, ld_operator='lt'):
    print('Scoring with p-val %.2g and R2 %.2g' % (p_t, r_t))
    if isinstance(pheno, pd.core.frame.DataFrame):
        pheno = pheno.PHENO.values
    clumps, df2 = clump(R2, sumstats, r_t, p_t, threads, field=field,
                        max_memory=max_memory, ld_operator=ld_operator)
    index = clumps.snp.tolist()
    idx = sumstats[sumstats.snp.isin(index)].i.tolist()
    betas = sumstats[sumstats.snp.isin(index)].slope
    prs = geno[:, idx].dot(betas)
    if approx:
        r_value = np.corrcoef(prs, pheno)[1, 0]
    else:
        slope, intercept, r_value, p_value, std_err = lr(pheno, prs)
    print('Done clumping for this configuration. R2=%.3f\n' % r_value ** 2)
    return r_t, p_t, r_value ** 2, clumps, prs, df2


# ----------------------------------------------------------------------
def pplust(prefix, geno, pheno, sumstats, r_range, p_thresh, split=3, seed=None,
           threads=1, window=250, pv_field='pvalue', max_memory=None,
           ld_operator='lt', graph=False, **kwargs):
    print(kwargs)
    X_train = None
    now = time.time()
    print ('Performing P + T!')
    seed = np.random.randint(1e4) if seed is None else seed
    # Read required info (sumstats, genfile)
    if 'bim' in kwargs:
        bim = kwargs['bim']
    if 'fam' in kwargs:
        fam = kwargs['fam']
    if isinstance(pheno, str):
        pheno = pd.read_table(pheno, delim_whitespace=True, header=None,
                              names=['fid', 'iid', 'PHENO'])
    elif pheno is None:
        # make simulation
        opts = {'outprefix': 'ppt_simulation', 'bfile': geno,
                'h2': kwargs['h2'], 'ncausal': kwargs['ncausal'],
                'normalize': kwargs['normalize'], 'uniform': kwargs['uniform'],
                'seed': seed, 'max_memory': max_memory}
        pheno, h2, (geno, bim, truebeta, vec) = qtraits_simulation(**opts)
        assert bim.shape[0] == geno.shape[1]
        opt2 = {'prefix': 'ppt_simulation', 'pheno': pheno, 'geno': geno,
                'validate': 2, 'seed': seed, 'threads': threads, 'bim':bim}
        sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opt2)
    if isinstance(geno, str) and pheno is not None:
        (bim, fam, geno) = read_plink(geno)
        geno = geno.T
    if isinstance(sumstats, str):
        sumstats = pd.read_table(sumstats, delim_whitespace=True)
    bim, R2 = blocked_R2(bim, geno, window)
    bim['gen_index'] = bim.i.tolist()
    # Create training and testing set
    if X_train is None:
        X_train, X_test, y_train, y_test = train_test_split(geno, pheno,
                                                            test_size=1 / split,
                                                            random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_test, y_test,
                                                            test_size=1 / split,
                                                            random_state=seed)
    # Sort sumstats by pvalue and clump by R2
    sumstats = sumstats.sort_values(by=pv_field, ascending=True)
    # do clumping
    sumstats = sumstats.merge(bim.reindex(columns=['snp', 'block']),
                              on='snp').dropna(subset=[pv_field])
    print('Largest p-value in summary statistics\n', sumstats.iloc[-1])
    print('Smallest p-value in summary statistics\n', sumstats.iloc[0])
    if os.path.isfile('%s_ppt.results.tsv' % prefix):
        r = pd.read_csv('%s_ppt.results.tsv' % prefix, sep='\t')
    else:
        r = []
        rapp = r.append
        for r_t, p_t in product(r_range, p_thresh):
            rapp(score(X_train, y_train, sumstats, r_t, p_t, R2, threads,
                       field=pv_field, approx=True, max_memory=max_memory,
                       ld_operator=ld_operator))
        r = sorted(r,  key=itemgetter(2), reverse=True)
        get = itemgetter(0,1,2)
        r = pd.DataFrame.from_records([get(x) for x in r], columns=[
            'LD threshold', 'P-value threshold', 'R2'])
        r.to_csv('%s_ppt.results.tsv' % prefix, sep='\t', index=False)
    best_rt, best_pt, best_r2 = r.nlargest(n=1, columns='R2').values.flat
    # score in test set
    r_t, p_t, fit, clumps, sc, df2 = score(X_test, y_test, sumstats, best_rt,
                                          best_pt, R2, threads, field=pv_field,
                                           max_memory=max_memory,
                                           ld_operator=ld_operator)
    if isinstance(sc, np.ndarray):
        if isinstance(y_test, pd.core.frame.DataFrame):
            prs = y_test.reindex(columns=['fid', 'iid'])
        else:
            raise NotImplementedError
        prs['prs'] = sc
    else:
        prs = y_test.reindex(columns=['fid','iid'])
        prs['prs'] = sc.compute(num_workers=threads)
    print('P+T optimized with pvalue %.4g and LD value of %.3f: R2 = %.3f in '
          'the test set' % (p_t, r_t, fit))
    clumps.to_csv('%s.clumps' % prefix, sep='\t', index=False)
    prs.to_csv('%s.prs' % prefix, sep='\t', index=False)
    df2.to_csv('%s.sorted_ppt' % prefix, sep='\t', index=False)
    # plot the prune version:
    new_plot(prefix, df2, X_test, y_test, threads)
    print ('P + T Done after %.2f minutes' % ((time.time() - now) / 60.))
    return p_t, r_t, fit, clumps, prs, df2


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
    parser.add_argument('-b', '--bfile', help='plink fileset prefix',
                        required=True)
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-s', '--sumstats', help='Filename of Sumstats',
                        default=None)
    parser.add_argument('-P', '--pheno', help='Filename of phenotype file',
                        default=None)
    parser.add_argument('-A', '--allele_file', default='EUR.allele',
                        help='File with the allele order. A1 in position 3 and '
                             'id in position2')
    parser.add_argument('-l', '--LDwindow',
                        help='Physical distance threshold ' +
                             'for clumping in kb (250kb by default)', type=int,
                        default=250)
    parser.add_argument('--r_range', default=None, nargs=3,
                        action=Store_as_arange, type=float)
    parser.add_argument('--p_thresh', default=None, nargs='+',
                        action=Store_as_array, type=float)
    parser.add_argument('-z', '--clean', help='Cleanup the clump and profiles',
                        default=False, action='store_true')
    parser.add_argument('-L', '--label', help='Label of the populations being' +
                                              ' analyzed.', default='EUR')
    parser.add_argument('-t', '--plot', help='Plot results of analysis',
                        default=False, action='store_true')
    parser.add_argument('-f', '--clump_field', default='P',
                        help=('field in the summary stats to clump with'))
    parser.add_argument('-a', '--sort_file', default=None,
                        help='File to sort the snps by instead of pvalue')
    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=None, type=float)
    parser.add_argument('-S', '--score_type', default='sum')
    parser.add_argument('--h2', default=None, type=float)
    parser.add_argument('--ncausal', default=None, type=int)
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--uniform', default=False, action='store_true')
    parser.add_argument('--nsplits', default=2, type=int)
    parser.add_argument('--pvalue_field', default='pvalue')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--ld_operator', default='lt')
    args = parser.parse_args()

    prs = pplust(args.prefix, args.bfile, args.pheno, args.sumstats,
                 args.r_range, args.p_thresh, seed=args.seed, split=args.nsplits,
                 threads=args.threads, h2=args.h2, ncausal=args.ncausal,
                 uniform=args.uniform, pv_field=args.pvalue_field,
                 normalize=args.normalize, max_memory=args.maxmem,
                 ld_operator=args.ld_operator)