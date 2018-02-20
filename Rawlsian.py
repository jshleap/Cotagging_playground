from comparison_ese import *


def single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno, threads, split,
           seed, memory):
    r_idx = np.arange(0, i)
    prefix = '%d_gwas' % i
    opts.update(
        dict(prefix=prefix, pheno=rpheno.iloc[:i], geno=rgeno[r_idx, :],
             validate=None, threads=threads, bim=rbim, seed=None))
    sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    out = dirty_ppt(loci, sumstats, X_train, y_train, threads, split, seed,
                    memory)
    ppt, selected, tail, _, _ = out
    idx = selected.i.values
    prs = tgeno[:, idx].dot(selected.slope)
    est = np.corrcoef(prs, tpheno.PHENO)[1, 0] ** 2
    return {r'$R^2_{ppt}$': est, 'EUR_n': i}


def main(args):
    seed = np.random.randint(1e4) if args.seed is None else args.seed
    memory = 1E9 if args.maxmem is None else args.maxmem
    refl, tarl = args.labels
    # make simulations
    print('Simulating phenotype for reference population %s \n' % refl)
    opts = {'outprefix': refl, 'bfile': args.refgeno, 'h2': args.h2,
            'ncausal': args.ncausal, 'normalize': args.normalize,
            'uniform': args.uniform, 'snps': None, 'seed': seed,
            'bfile2': args.targeno, 'flip': args.gflip,
            'freqthreshold': args.freq_thresh}
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
    max_r2 = np.corrcoef(tpheno.gen_eff.values, tpheno.PHENO)[1, 0] ** 2
    # perform GWASes
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=args.window,
                  threads=args.threads, justd=True)
    array = [10, 20, 40, 80, 160, 320, 640, 1280, 3000, 5000, 10000, 20000,
             45000]
    res = []
    for i in array:
        res.append(single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno,
                          args.threads, args.split, seed, memory))
    res = pd.DataFrame(res)
    f, ax = plt.subplots()
    res.plot(x='EUR_n', y=r'$R^2_{ppt}$', marker='o', ax=ax)
    plt.ylabel(r'AFR $R^2_{ppt}$')
    ax.axhline(max_r2, ls='-.', color='0.5')
    plt.tight_layout()
    plt.savefig('%s.pdf' % args.prefix)
    plt.close()


# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-b', '--refgeno', required=True,
                        help=('prefix of the bed fileset in reference'))
    parser.add_argument('-g', '--targeno', required=True,
                        help=('prefix of the bed fileset in target'))
    parser.add_argument('-L', '--labels', help=('Populations labels'), nargs=2)
    parser.add_argument('-H', '--h2', type=float, help='Heritability of trait',
                        required=True)
    parser.add_argument('-m', '--ncausal', default=200, type=int)
    parser.add_argument('--normalize', default=True, action='store_false')
    parser.add_argument('--uniform', default=True, action='store_false')
    parser.add_argument('--split', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--flip', action='store_true', help='flip sumstats')
    parser.add_argument('--gflip', action='store_true', help='flip genotype')
    parser.add_argument('--freq_thresh', type=float, help='filter by mafs',
                        default=-1)
    parser.add_argument('--reference', action='store_true',
                        help='use reference for computations')
    parser.add_argument('-w', '--window', default=1000, type=int,
                        help='Size of the LD window. a.k.a locus')
    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=None, type=int)

    args = parser.parse_args()
    main(args)
