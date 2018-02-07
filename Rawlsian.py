from comparison_ese import *


def process_beta_sq(sumstats, geno, pheno, prunestep):
    sumstats['beta_sq'] = sumstats.slope ** 2
    beta, _ = smartcotagsort('%s_slope' % args.prefix, sumstats,
                             column='beta_sq', ascending=False)
    pruned = prune_it(beta, geno, pheno, r'$\beta^2$', step=prunestep,
                    threads=args.threads)
    m = pruned.nlargest(1, 'R2').loc[:,'Number of SNPs'].values[0]
    return beta.iloc[: m]


def single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno, run, threads):
    r_idx = np.arange(0, i)
    prefix = '%d_gwas' % i
    opts.update(
        dict(prefix=prefix, pheno=rpheno.iloc[:i], geno=rgeno[r_idx, :],
             validate=None, threads=threads, bim=rbim))
    sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    ppt, selected, tail = dirty_ppt(loci, sumstats, tgeno, tpheno,
                                    args.threads)
    idx = selected.i.values
    prs = tgeno[:, idx].dot(selected.slope)
    est = np.corrcoef(prs, tpheno.PHENO)[1, 0] ** 2
    # get selection by beta^2
    betas = process_beta_sq(sumstats, tgeno, tpheno, 10)
    b_idx = betas.i.values
    b_prs = tgeno[:, b_idx].dot(betas.slope)
    b_est = np.corrcoef(b_prs, tpheno.PHENO)[1, 0] ** 2
    return {r'$R^2_{ppt}$': est, r'$R^2_{\beta^2}$': b_est, 'EUR_n': i,
            'run': run}


def main(args):
    seed = np.random.randint(1e4) if args.seed is None else args.seed
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
    # perform GWASes
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=args.window,
                  threads=args.threads, justd=True)
    result = pd.DataFrame()
    for run in range(50):
        results = [
            single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno, run,
                   args.threads) for i in
            np.linspace(200, rgeno.shape[0], 50, dtype=int)]
        [os.remove(fn) for fn in glob('./*') if
         (os.path.isfile(fn) and fn != 'Rawlsian.tsv')]
        result = result.append(pd.DataFrame(results))
        result.to_csv('Rawlsian.tsv', sep='\t', index=False)
    cols = [c for c in result.columns if c != 'run']
    gp3 = result.loc[:, cols].groupby('EUR_n')
    means = gp3.mean()
    errors = gp3.std()
    f, ax = plt.subplots()
    means.plot(yerr=errors, ax=ax)
    plt.tight_layout()
    plt.savefig('%s.pdf' % args.prefix)


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

    args = parser.parse_args()
    main(args)
