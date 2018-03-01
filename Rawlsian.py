from comparison_ese import *

#@jit(parallel=True)
def single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno, threads, split,
           seed, memory, pvals, lds):
    r_idx = np.arange(0, i)
    prefix = '%d_gwas' % i
    opts.update(
        dict(prefix=prefix, pheno=rpheno.iloc[r_idx], geno=rgeno[r_idx, :],
             validate=None, threads=threads, bim=rbim, seed=seed))
    sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    out = dirty_ppt(loci, sumstats, X_train, y_train, threads, split, seed,
                    memory, pvals=pvals, lds=lds)
    ppt, selected, tail, _, _ = out
    ppt.to_csv('%s_ppt.tsv' % prefix, sep='\t', index=False)
    selected.to_csv('%s_selected.tsv' % prefix, sep='\t', index=False)
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
    # shuffle rerefence individuals
    idx = np.arange(rgeno.shape[0])
    np.random.shuffle(idx)
    assert not np.array_equal(idx , rpheno.i.values)
    rgeno = rgeno[idx, :]
    rpheno = rpheno.iloc[idx].reset_index()
    rpheno['i'] = rpheno.index
    # make simulation for target
    print('Simulating phenotype for target population %s \n' % tarl)
    opts.update(dict(outprefix=tarl, bfile=args.targeno, bfile2=args.refgeno,
                     causaleff=rbim.dropna(subset=['beta']),
                     validate=args.split))
    if args.reference:
        tpheno, tgeno = rpheno, rgeno
    else:
        tpheno, h2, (tgeno, tbim, truebeta, tvec) = qtraits_simulation(**opts)
        idx = np.arange(tgeno.shape[0])
        np.random.shuffle(idx)
        assert not np.array_equal(idx, tpheno.i.values)
        tgeno = tgeno[idx, :]
        tpheno = tpheno.iloc[idx].reset_index()
        tpheno = tpheno.iloc[idx].reset_index()
        tpheno['i'] = tpheno.index
    max_r2 = np.corrcoef(tpheno.gen_eff.values, tpheno.PHENO)[1, 0] ** 2
    # perform GWASes
    ldfiles = glob('%s_locus*.pickle' % args.prefix)
    if not ldfiles:
        loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=args.window,
                      threads=args.threads, justd=True)
        # Pickle loci??
        for i, l in enumerate(loci):
            with open('%s_locus%d.pickle' % (args.prefix, i), 'wb') as F:
                pickle.dump(l, F)
    else:
        print('Loading LD per locus')
        loci = [pickle.load(open(l, 'rb')) for l in ldfiles]
    # do ppt in AFR
    o = dict(prefix='Target_sumstats', pheno=tpheno, geno=tgeno, validate=2,
             threads=args.threads, bim=tbim, seed=None)
    out = plink_free_gwas(**o)
    t_sumstats, t_X_train, t_X_test, t_y_train, t_y_test = out
    out = dirty_ppt(loci, t_sumstats, t_X_test, t_y_test, args.threads, 2,
                    None, memory, pvals=args.pvals, lds=args.lds)
    t_ppt, t_pre, t_pos, t_x_test, t_y_test = out
    idx_tar = t_pre.i.values
    t_prs = t_x_test[:, idx_tar].dot(t_pre.slope)
    t_r2 = np.corrcoef(t_prs, t_y_test.PHENO)[1, 0] ** 2
    array = [10, 40, 160, 640, 2560, 10240, 40.960, 45000]
    # randomize individuals to check if it changes the result
    res = []
    for i in array:
        res.append(single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno,
                          args.threads, 1, seed, memory, args.pvals,
                          args.lds))
    res = pd.DataFrame(res)
    res.to_csv('%s_finaldf.tsv' % args.prefix, sep='\t', index=False)
    f, ax = plt.subplots()
    res.plot(x='EUR_n', y=r'$R^2_{ppt}$', marker='.', s=5, ax=ax)
    plt.ylabel(r'AFR $R^2_{ppt}$')
    ax.axhline(max_r2, ls='-.', color='0.5', label='Causals')
    ax.axhline(t_r2, ls='-.', color='r', label=r'$%s_{P + T}$' % tarl)
    plt.title('P + T with 1 and 0.6 pvalue and ld, respectively')
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
    parser.add_argument('--pvals', default=None, nargs='+', type=float)
    parser.add_argument('--lds', default=None, nargs='+', type=float)

    args = parser.parse_args()
    main(args)
