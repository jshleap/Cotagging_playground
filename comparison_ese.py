'''
Code to execute parts of ese and plot the different sorts
'''

from ese import *

within_dict={0:'ese cotag', 1:'ese EUR', 2:'ese AFR'}


def individual_ese(sumstats, avh2, h2, n, within, loci):
    within = within_dict[within]
    resfile = '%s_res.tsv' % args.prefix
    print('Compute expected beta square per locus...')
    delayed_results = [dask.delayed(per_locus)(locus, sumstats, avh2, h2, n,
                                               within=within) for i, locus
                       in enumerate(loci)]
    res = list(dask.compute(*delayed_results, num_workers=args.threads))
    res = pd.concat(res)
    result = res.merge(sumstats.reindex(columns=['slope', 'snp', 'beta']),
                       on='snp')
    result.to_csv(resfile, index=False, sep='\t')
    prod, _ = smartcotagsort(args.prefix, result, column='ese')


def main(args):
    seed = np.random.randint(1e4) if args.seed is None else args.seed
    refl, tarl = args.labels
    # make simulations
    print('Simulating phenotype for reference population %s \n' % refl)
    opts = {'outprefix': refl, 'bfile': args.refgeno, 'h2': args.h2,
            'ncausal': args.ncausal, 'normalize': args.normalize,
            'uniform': args.uniform, 'snps': None, 'seed': seed,
            'bfile2': args.targeno, 'flip': args.gflip,
            'max_memory': args.max_memory}
    rpheno, h2, (rgeno, rbim, rtruebeta, rvec) = qtraits_simulation(**opts)
    # make simulation for target
    print('Simulating phenotype for target population %s \n' % tarl)
    opts.update(dict(outprefix=tarl, bfile=args.targeno,
                     causaleff=rbim.dropna(), bfile2=args.refgeno,
                     validate=args.split))
    tpheno, h2, (tgeno, tbim, ttruebeta, tvec) = qtraits_simulation(**opts)
    opts.update(dict(prefix='ranumo_gwas', pheno=rpheno, geno=rgeno,
                     validate=args.split, threads=args.threads, bim=rbim,
                     flip=args.flip))
    # perform GWAS
    sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    sum_snps = sumstats.snp.tolist()
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=args. LDwindow,
                  threads=args.threads, justd=True)
    avh2 = h2 / len(sum_snps)
    n = tgeno.shape[0]
    # expecetd square effect
    eses = [individual_ese(sumstats, avh2, h2, n, x, loci) for x in [0, 1, 2]]
    # prune by pval
    pval, _ = smartcotagsort('%s_pval' % args.prefix, sumstats, column='pvalue')
    # plot them
    res = pd.concat(eses + [pval])
    colors = iter(['r', 'b', 'm', 'g', 'c', 'k', 'y'])
    f, ax = plt.subplots()
    for t, df in res.groupby('type'):
        df.plot(x='Number of SNPs', y='R2', kind='scatter', legend=True,
                s=3, c=next(colors), ax=ax, label=t)
    plt.tight_layout()
    plt.savefig('%s_transferability.pdf' % args.prefix)



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
    parser.add_argument('--freq_thresh', type=float, help='filter by mafs')
    parser.add_argument('--within', default=0, type=int,
                        help='0=cross; 1=reference; 2=target')
    parser.add_argument('--ld_operator', default='lt')
    parser.add_argument('--graph', action='store_true')

    args = parser.parse_args()
    main(args)