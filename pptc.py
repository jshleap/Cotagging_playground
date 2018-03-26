#!/usr/bin/env python
# coding:utf-8
"""
  Author: Jose Sergio Hleap  --<2017>
  Purpose: Implementation of P + T with espected squared effect estimations
  on the optimized ranking
  Created: 10/02/17
"""
from itertools import product
from ese import *
from comparison_ese import dirty_ppt


def get_tagged2(locus, ld_thr, ese_thresh, sumstats, avh2, h2, n):
    snp_list, d_r, d_t = locus
    d_r = pd.DataFrame(d_r, index=snp_list, columns=snp_list) ** 2
    d_t = pd.DataFrame(d_t, index=snp_list, columns=snp_list) ** 2
    index = []
    ippend = index.append
    tag = []
    text = tag.extend
    # sort just once
    sumstats = sumstats[sumstats.snp.isin(snp_list)].sort_values(
        ['ese', 'pvalue', 'beta_sq', 'pos'], ascending=[False, True, False,
                                                        True])
    if any([isinstance(x, str) for x in sumstats.pvalue]):
        sumstats.loc[:, 'pvalue'] = [mp.mpf(i) for i in sumstats.pvalue]
    total_snps = sumstats.shape[0]
    r = 0
    while len(index + tag) != total_snps:
        curr_high = sumstats.iloc[0]
        #if mp.mpf(curr_high.pvalue) < p_thresh:
        if curr_high.ese > ese_thresh:
            # get snps in LD
            vec = d_r.loc[curr_high.snp, :]
            tagged = vec[vec > ld_thr].index.tolist()
            # re-score ese for the clump
            ss = sumstats[sumstats.snp.isin(tagged)]
            n_locus = (tagged, d_r.loc[tagged, tagged], d_t.loc[tagged, tagged])
            # TODO: include within as option?
            df_ese = per_locus(n_locus, ss, avh2, h2, n, r, within=0)
            ss.merge(df_ese.reindex(columns=['snp', 'ese']), on='snp')
            r += 1
            largest = ss.nlargest(1, 'ese')
            if largest.ese > curr_high.ese:
                curr_high = largest
            ippend(curr_high.snp)
            if curr_high.snp in tagged:
                tagged.pop(tagged.index(curr_high.snp))
            text(tagged)
        else:
            low = sumstats.snp.tolist()
            text(low)
        sumstats = sumstats[~sumstats.snp.isin(index + tag)]
    return index, tag


@jit
def loop_pairs2(locus, l, p, e, sumstats, pheno, geno, avh2, h2):
    # clean with soft pvalue
    n, m = geno.shape
    ss = sumstats[sumstats.pvalue < p]
    index, tag = get_tagged2(locus, l, e, ss, avh2, h2, n)
    clump = sumstats[sumstats.snp.isin(index)]
    idx = clump.i.values.astype(int)
    prs = geno[:, idx].dot(clump.slope)
    est = np.corrcoef(prs, pheno.PHENO)[1, 0] ** 2
    return est, (index, tag, l, p)


@jit
def just_score(index_snp, sumstats, pheno, geno):
    clump = sumstats[sumstats.snp.isin(index_snp)]
    idx = clump.i.values.astype(int)
    prs = geno[:, idx].dot(clump.slope)
    est = np.corrcoef(prs, pheno.PHENO)[1, 0] ** 2
    return est


def pptc(loci, sumstats, geno, pheno, h2, threads, memory, pvals=None, lds=None,
         within=0):
    cache = Chest(available_memory=memory)
    now = time.time()
    if not 'beta_sq' in sumstats.columns:
        try:
            sumstats.loc[:, 'beta_sq'] = sumstats.slope ** 2
        except TypeError:
            sumstats['beta_sq'] = [mp.mpf(x) ** 2 for x in sumstats.slope]

    print('Starting PPTC...')
    index, tag = [], []
    n, m = geno.shape
    avh2 = h2 / m
    pre = []
    pos = []
    big_sumstats = []
    for r, locus in enumerate(loci):
        snps, d_r, d_t = locus
        ese_df = per_locus(locus, sumstats, avh2, h2, n, r, within=within)
        snp_list = snps.tolist()
        d_r = dd.from_dask_array(d_r, columns=snps) ** 2
        sub = sumstats[sumstats.snp.isin(snps)].reindex(
            columns=['snp', 'slope', 'beta_sq', 'pvalue', 'i', 'pos', 'beta'])
        sub['locus'] = r
        sub = sub.merge(ese_df.reindex(columns=['snp', 'ese']), on='snp')
        big_sumstats.append(sub)
        # TODO: include ese threshold?
        ese_threshold = sub.ese.quantile(np.arange(.0, 1, .1))
        # filter pvalue
        if pvals is None:
            pvals = [1, 0.5, 0.05, 5E-4, 10E-8]
        if lds is None:
            lds = np.arange(0.1, 0.8, 0.1)
        pairs = product(pvals, lds, ese_threshold)
        delayed_results = [
            dask.delayed(loop_pairs2)(locus, l, p, e, sub, geno, pheno, avh2, h2
                                      ) for p, l, e in pairs]
        with ProgressBar():
            print('    Locus', r)
            d = dict(list(dask.compute(*delayed_results, num_workers=threads,
                                       memory_limit=memory, cache=cache,
                                       pool=ThreadPool(threads))))
            best_key = max(d.keys())
            i, t, ld, pv = d[best_key]
            index += i
            tag += t
        pre.append(sub[sub.snp.isin(index)])
        pos.append(sub[sub.snp.isin(tag)])
    cols = ['ese', 'pvalue', 'pos']
    asc = [False, True, True]
    pre = pd.concat(pre).sort_values(cols, ascending=asc)
    pos = pd.concat(pos).sort_values(cols, ascending=asc)
    ppt = pre.append(pos, ignore_index=True).reset_index(drop=True)
    ppt['index'] = ppt.index.tolist()
    big_sumstats.to_csv('%s.big_sumstats.tsv', sep='\t', index=False)
    print('Dirty ppt done after %.2f minutes' % ((time.time() - now) / 60.))
    return ppt, pre, pos


def main(prefix, refgeno, refpheno, targeno, tarpheno, h2, labels,
         LDwindow, sumstats, refld=None, tarld=None, seed=None,
         max_memory=None, threads=1, merged=None, within=False,
         **kwargs):
    """
    Execute trasnferability code
    """
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
                'bfile2': targeno, 'flip': kwargs['gflip'],
                'max_memory': max_memory}
        rpheno, h2, (rgeno, rbim, rtruebeta, rvec) = qtraits_simulation(**opts)
        # make simulation for target
        print('Simulating phenotype for target population %s \n' % tarl)
        opts.update(
            dict(outprefix=tarl, bfile=targeno, validate=kwargs['split'],
                 causaleff=rbim.dropna(subset=['beta']), bfile2=refgeno))
        tpheno, h2, (tgeno, tbim, ttruebeta, tvec) = qtraits_simulation(**opts)
        opts.update(dict(prefix='ranumo_gwas', pheno=rpheno, geno=rgeno,
                         validate=0, threads=threads, bim=rbim,
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
    # run pptc
    out = pptc(loci, sumstats, X_train, y_train, h2, threads, max_memory,
               pvals=None, lds=None, within=within)
    merged, index, tagged, x_test, y_test = out
    out_ppt = dirty_ppt(loci, sumstats, X_train, y_train, threads, 2, seed,
                        max_memory, pvals=None, lds=None)

    merged_ppt, index_ppt, tagged_ppt, x_test_ppt, y_test_ppt = out_ppt
    R2_ppt = just_score(index_ppt.snp, index_ppt, tpheno, tgeno)
    R2_pptc = just_score(index.snp, index, tpheno, tgeno)
    pd.DataFrame({'ppt': R2_ppt, 'pptc': R2_pptc}).to_csv('%s.pptcres.tsv',
                                                          sep='\t', index=False)

    # f, ax = plt.subplots()
    # for t, df in res.groupby('type'):
    #     df.plot(x='Number of SNPs', y='R2', kind='scatter', legend=True,
    #             s=3, c=next(colors), ax=ax, label=t)
    # plt.tight_layout()
    # plt.savefig('%s_transferability.pdf' % args.prefix)
    # plt.close()
    # print('ESE done after %.2f minutes' % ((time.time() - now) / 60.))
    # return res


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
    main(args.prefix, args.reference, args.refpheno, args.target,
         args.tarpheno, args.h2, args.labels, args.window,
         args.sumstats, refld=args.refld, tarld=args.tarld,
         seed=args.seed, threads=args.threads, merged=args.merged,
         ncausal=args.ncausal, normalize=True, uniform=args.uniform,
         r_range=args.r_range, p_tresh=args.p_tresh,
         max_memory=args.maxmem, split=args.split, flip=args.flip,
         gflip=args.gflip, within=args.within, check=args.check,
         ld_operator=args.ld_operator, graph=args.graph)
