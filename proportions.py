from comparison_ese import *
plt.style.use('ggplot')


def single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno, test_geno,
           test_pheno, threads, memory, full):
    seed = np.random.randint(54321)
    prefix = '%.2f_gwas' % i
    total_n = rgeno.shape[0] # ASSUMES THAT BOTH SOURCE POPS ARE THE SAME SIZE
    frac_n = int(total_n * i)
    if i == 0:
        pheno = tpheno
        geno = tgeno
    elif i == 1:
        pheno = rpheno
        geno = rgeno
    else:
        ref_pheno = rpheno.iloc[: frac_n]
        ref_geno = rgeno[:frac_n, :]
        tar_pheno = tpheno.iloc[: (total_n - frac_n)]
        tar_geno = tgeno[:(total_n - frac_n), :]
        pheno = pd.concat([ref_pheno, tar_pheno]).reset_index(drop=True)
        pheno['i'] = pheno.index.values
        geno = da.concatenate([ref_geno, tar_geno], axis=0)
        pheno = pheno.sample(frac=1)
        geno = geno[pheno.i.values, :]
        pheno['i'] = list(range(pheno.shape[0]))
    assert geno.shape[0] == full == pheno.shape[0]
    opts.update(
        dict(prefix=prefix, pheno=pheno, geno=geno, validate=None,
             threads=threads, bim=rbim, seed=None, pca=args.pca))
    sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    # P+T scored in Target
    out = dirty_ppt(loci, sumstats, X_test, y_test, args.threads, 1, seed,
                    memory)
    ppt_tar, sel_tar, tail_tar, _, _ = out
    idx_tar = sel_tar.i.values
    prs_tar = test_geno[:, idx_tar].dot(sel_tar.slope)
    r2_tar = np.corrcoef(prs_tar, test_pheno.PHENO)[1, 0] ** 2
    return {r'$R^2_{ppt}$': r2_tar, 'EUR_frac': i, 'AFR_frac': 1 - i}


def main(args):
    if os.path.isfile('proportions.tsv'):
        results = pd.read_table('proportions.tsv', sep='\t')
        t_r2 = np.load('t_r2')
        max_r2 = np.load('max_r2')
    else:
        seed = np.random.randint(1e4) if args.seed is None else args.seed
        memory = 1E9 if args.maxmem is None else args.maxmem
        refl, tarl = args.labels
        # make simulations
        print('Simulating phenotype for reference population %s \n' % refl)
        opts = {'outprefix': refl, 'bfile': args.refgeno, 'h2': args.h2,
                'ncausal': args.ncausal, 'normalize': args.normalize,
                'uniform': args.uniform, 'snps': None, 'seed': seed,
                'bfile2': args.targeno, 'flip': args.gflip,
                'freq_thresh': args.freq_thresh, 'max_memory': memory}
        rpheno, h2, (rgeno, rbim, rtruebeta, rvec) = qtraits_simulation(**opts)
        # make simulation for target
        print('Simulating phenotype for target population %s \n' % tarl)
        opts.update(
            dict(outprefix=tarl, bfile=args.targeno, bfile2=args.refgeno,
                 causaleff=rbim.dropna(subset=['beta']), validate=args.split))
        if args.reference:
            tpheno, tgeno = rpheno, rgeno
        else:
            tpheno, h2, (tgeno, tbim, truebeta, tvec) = qtraits_simulation(
                **opts)
        # # Get the diverse sample to be test on
        opts2 = dict(test_size=1 / args.split, random_state=seed)
        r_out = train_test_split(rgeno, rpheno, **opts2)
        rgeno, rgeno_test, rpheno, rpheno_test = r_out
        rpheno = rpheno.reset_index(drop=True)
        t_out = train_test_split(tgeno, tpheno, **opts2)
        tgeno, tgeno_test, tpheno, tpheno_test = t_out
        max_r2 = np.corrcoef(tpheno.gen_eff.values, tpheno.PHENO)[1, 0] ** 2
        np.save('max_r2', max_r2)
        # Get LD info
        loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=args.window,
                      justd=True, threads=args.threads, max_memory=args.maxmem)
        # do ppt in AFR
        o = dict(prefix='Target_sumstats', pheno=tpheno, geno=tgeno, validate=2,
                 threads=args.threads, bim=tbim, seed=None, pca=args.pca,
                 normalize=opts['normalize'])
        out = plink_free_gwas(**o)
        t_sumstats, t_X_train, t_X_test, t_y_train, t_y_test = out
        out = dirty_ppt(loci, t_sumstats, t_X_test, t_y_test, args.threads, 2,
                        None, memory,pvals=args.pvals, lds=args.lds)
        t_ppt, t_pre, t_pos, t_x_test, t_y_test = out
        idx_tar = t_pre.i.values
        t_prs = t_x_test[:, idx_tar].dot(t_pre.slope)
        t_r2 = np.corrcoef(t_prs, t_y_test.PHENO)[1, 0] ** 2
        np.save('t_r2', t_r2)
        full = tgeno.shape[0]
        results = pd.DataFrame([
            single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno, tgeno_test,
                   tpheno_test, args.threads, memory, full=full) for i in
            np.clip(np.arange(0, 1.1, 0.1), a_min=0, a_max=1)])
        results.to_csv('proportions.tsv', sep='\t', index=False)
    f, ax = plt.subplots()
    ax.axhline(max_r2, ls='-.', color='0.5', label='Causals in %s' % tarl)
    ax.axhline(t_r2, ls='-.', color='r', label='%s P + T' % tarl)
    results.plot(x='EUR_frac', y=r'$R^2_{ppt}$', ax=ax, marker='o')
    plt.ylabel(r'AFR $R^2_{ppt}$')
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
    parser.add_argument('--pca', default=20, type=int)
    parser.add_argument('--pvals', default=[1], nargs='+', type=float)
    parser.add_argument('--lds', default=[0.6], nargs='+', type=float)
    args = parser.parse_args()
    main(args)
