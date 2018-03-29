from itertools import product
import operator
from ese import *
from comparison_ese import just_score


np.seterr(all='raise')  # Debugging


def clumps(locus, sum_stats, ld_threshold, h2, avh2, n, select_by='pvalue',
           clump_with='d_reference'):
    """
    Get clumps from locus
    :param r2_reference: the r2 (LD) matrix for the subset
    :param sum_stats: subset of the summary statistics for the locus
    :param ld_threshold: the threshold for this run
    :return:
    """
    # unpack the locus tuple
    snp_list, d_reference, d_target = locus
    # Name the rows and columns
    d_reference = pd.DataFrame(d_reference, index=snp_list, columns=snp_list)
    d_target = pd.DataFrame(d_target, index=snp_list, columns=snp_list)
    # subset sum_stats
    sum_stats = sum_stats[sum_stats.snp.isin(snp_list)]
    # Get the clumps pfr this locus
    clumps = {}
    while not sum_stats.empty:
        # get the index snp
        if select_by == 'pvalue':
            index = sum_stats.nsmallest(1, select_by)
        else:
            index = sum_stats.nlargest(1, select_by)
        # get the clump around index for
        vec = (locals()[clump_with] ** 2).loc[index.snp, :]
        tag = vec[vec > ld_threshold].index.tolist()
        # Subset the sumary statistic dataframe with the snps in the clump
        sub_stats = sum_stats[sum_stats.snp.isin(tag)]
        # Store the sub matrices in a tuple for ESE estimation
        n_locus = (tag, d_reference.loc[tag, tag], d_target.loc[tag, tag])
        # Compute ESE and include it into the main dataframe for this clump
        df_ese = per_locus(n_locus, sub_stats, avh2, h2, n, 0, within=0)
        ss = sub_stats.merge(df_ese.reindex(columns=['snp', 'ese']), on='snp')
        # make sure it matches
        assert ss.pvalue.nsmallest(1).values ==  index.pvalue.values
        # Get the highest ESE of the clump
        max_ese = ss.nlargest(1, 'ese')
        # Store the results in clump dictionary
        key = (index.snp.iloc[0], index.pvalue.iloc[0], max_ese.snp.iloc[0],
               max_ese.ese.iloc[0])
        clumps[key] = ss
        # remove the clumped snps from the summary statistics dataframe
        sum_stats = sum_stats[~sum_stats.snp.isin(tag)]
    return clumps


def compute_clumps(loci, sum_stats, ld_threshold, h2, avh2, n, threads, cache,
                   memory, select_by='pvalue', clump_with='d_reference'):
    """

    :param loci: list of tuples with the LDs and snps per locus
    :param sum_stats: sumary statistics
    :param ld_threshold: trheshold for cumping
    :param h2: heritability of the trait
    :param avh2: average heritability
    :param n: number of samples
    :param threads: number of threads to use in multiprocessing
    :param cache: chest dictionary to avoid memory overflow
    :param memory: max memory to use
    :return: dictionary with the clumps
    """
    delayed_results = [
        dask.delayed(clumps)(locus, sum_stats, ld_threshold, h2, avh2, n,
                             select_by, clump_with) for locus in loci]
    with ProgressBar():
        print("Identifying clumps with R2 threshold of %.3f" % ld_threshold)
        l = list(dask.compute(*delayed_results, num_workers=threads,
                              memory_limit=memory, cache=cache,
                              pool=ThreadPool(threads)))
    return dict(pair for d in l for pair in d.items())


def optimize_it(loci, ld_range, by_range, h2, avh2, n, threads, cache, memory,
                sum_stats, test_geno, test_pheno, by='pvalue',
                clump_with='d_reference'):
    """
    Optimize the R2 based on summary statistics

    :param loci: List of tuples with the LDs and snps per locus
    :param ld_range: Range of ld thresholds
    :param by_range: Range of the ranking strategy (pvalue or ese)
    :param h2: Heritability of the trait
    :param avh2: Average heritability per snp
    :param n: Sample size
    :param threads: Number of threads to use in multithread computations
    :param cache: A dictionary that spills to disk. chest instance
    :param memory: Maximum memory to use
    :param sum_stats: Dataframe with the sumary statistics of an association
    :param test_geno: Test set genotype array
    :param test_pheno: Test set genotype vector (or series)
    :param by: Ranking strategy (ESE or Pvalue)
    :return: Tuple with the list of index snps and their R2
    """
    if by == 'pvalue':
        rank = getattr(operator, 'lt')
        snp_index = 0
    else:
        rank = getattr(operator, 'gt')
        snp_index = 2
    curr_best = ([], 0)
    for ld_threshold in ld_range:
        all_clumps = compute_clumps(loci, sum_stats, ld_threshold, h2, avh2, n,
                                    threads, cache, memory, by, clump_with)
        if by_range is None:
            by_range = pd.concat(all_clumps.values()).ese.quantile(
                np.arange(.0, 1, .1))
        index_snps = [k[snp_index] for by_threshold in by_range for k in
                      all_clumps.keys() if rank(k[snp_index + 1], by_threshold)]
        r2 = just_score(index_snps, sum_stats, test_pheno, test_geno)
        if r2 > curr_best[1]:
            curr_best = (index_snps, r2)
    return r2


def run_optimization_by(by_range, by, loci, h2, m, n, threads, cache, sum_stats,
                        available_memory,test_geno, test_pheno, tpheno, tgeno,
                        prefix, select_by='pvalue', clump_with='d_reference'):
    avh2 = h2 / m
    ld_range = np.arange(0.1, 0.8, 0.1)
    if by_range is None and (by == 'pvalue'):
            by_range = [1, 0.5, 0.05, 10E-3, 10E-5, 10E-7, 1E-9]
    r2_tuple = optimize_it(loci, ld_range, by_range, h2, avh2, n, threads,
                           cache, available_memory, sum_stats, test_geno,
                           test_pheno, select_by, clump_with)
    # score in target
    r2 = just_score(r2_tuple[0], sum_stats, tpheno, tgeno)
    ascending = True if by == 'pvalue' else False
    pre = sum_stats[sum_stats.snp.isin(r2_tuple[0])].sort_values(
        by, ascending=ascending)
    pos = sum_stats[~sum_stats.snp.isin(r2_tuple[0])].sort_values(
        by, ascending=ascending)
    pre.append(pos).to_csv('%s_full.tsv' % prefix, index=False, sep='\t')
    return pre, pos, r2


def main(prefix, refgeno, refpheno, targeno, tarpheno, h2, labels, LDwindow,
         sum_stats, seed=None, max_memory=None, threads=1, by='pvalue',
         by_range=None, clump_with='d_reference', **kwargs):
    """
    Execute trasnferability code
    """
    now = time.time()
    # set Cache to protect memory spilling
    if max_memory is not None:
        available_memory = max_memory
    else:
        available_memory = psutil.virtual_memory().available
    cache = Chest(available_memory=available_memory)

    seed = np.random.randint(1e4) if seed is None else seed
    print('Performing P + T + C!')
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
                         validate=None, threads=threads, bim=rbim,
                         flip=kwargs['flip']))
    elif isinstance(refgeno, str):
        (rbim, rfam, rgeno) = read_geno(refgeno, kwargs['freq_thresh'], threads,
                                        check=kwargs['check'],
                                        max_memory=max_memory)
        (tbim, tfam, tgeno) = read_geno(targeno, kwargs['freq_thresh'], threads,
                                        check=kwargs['check'],
                                        max_memory=max_memory)
    if isinstance(sum_stats, str):
        sum_stats = pd.read_table(sum_stats, delim_whitespace=True)
    else:
        sum_stats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    print("Reference bim's shape: %d, %d" % (rbim.shape[0], rbim.shape[1]))
    print("Target bim's shape: %d, %d" % (tbim.shape[0], tbim.shape[1]))
    sum_snps = sum_stats.snp.tolist()
    # Compute Ds
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=LDwindow, justd=True,
                  threads=threads, max_memory=max_memory)
    # optimize R2
    n, m = X_train.shape
    if by is None:
        # run pvalue
        pvalue = run_optimization_by(by_range, 'pvalue', loci, h2, m, n,
                                     threads, cache, sum_stats,
                                     available_memory, X_test, y_test, tpheno,
                                     tgeno, '%s_pval' % prefix, 'pvalue',
                                     clump_with)
        # run ese
        ese = run_optimization_by(by_range, 'ese', loci, h2, m, n, threads,
                                  cache, sum_stats, available_memory, X_test,
                                  y_test, tpheno, tgeno, '%s_pval' % prefix,
                                  'ese', clump_with)
        pd.DataFrame([{r'R^{2}_{ese}': ese[-1], r'R^{2}_{pvalue}': pvalue[-1],
                       'prefix': prefix}]).to_csv('%s.tsv' % prefix, sep='\t',
                                                  index=False)
    else:
        index, tagged, r2 = run_optimization_by(
            by_range, by, loci, h2, m, n, threads, cache, sum_stats,
            available_memory, X_test, y_test, tpheno, tgeno, '%s_pval' % prefix,
            by, clump_with)
        print('R2 is %.3f' % r2)
    print('pptc done after %.2f minutes' % ((time.time() - now) / 60.))


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
    parser.add_argument('-s', '--sumstats', help='Filename of sumstats',
                        default=None)
    parser.add_argument('-H', '--h2', type=float, help='Heritability of trait',
                        required=True)
    parser.add_argument('-f', '--tarpheno', default=None,
                        help='Filename of the true phenotype of the target '
                              'population')
    parser.add_argument('-r', '--refpheno', default=None,
                        help='Filename of the true phenotype of the reference '
                              'population')
    parser.add_argument('-S', '--sliding', default=False, action='store_true',
                        help='Use a sliding window instead of hard block')

    parser.add_argument('-w', '--window', default=1000, type=int,
                        help='Size of the LD window. a.k.a locus')
    parser.add_argument('-P', '--plinkexe', default=None)
    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=None, type=int)
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
    parser.add_argument('--by', default=None, help='ese or pvalue (none '
                                                       'will do both)')
    parser.add_argument('--check', action='store_true',
                        help='check and clean invariant sites')
    parser.add_argument('--clump_with', default='d_reference',
                        help='ld matrix to clump with: d_reference or d_target')

    args = parser.parse_args()
    main(args.prefix, args.reference, args.refpheno, args.target, args.tarpheno,
         args.h2, args.labels, args.window, args.sumstats, seed=args.seed,
         threads=args.threads, ncausal=args.ncausal, normalize=True, by=args.by,
         uniform=args.uniform, by_range=args.r_range, max_memory=args.maxmem,
         split=args.split, flip=args.flip, gflip=args.gflip, within=args.within,
         check=args.check)