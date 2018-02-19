from comparison_ese import *
plt.style.use('ggplot')

def single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno, diverse_geno,
           diverse_pheno, run, threads, memory):
    prefix = '%d_gwas' % i
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
        pheno['i'] = list(range(pheno.shape[0]))
        geno = da.concatenate([ref_geno, tar_geno], axis=0)
    opts.update(
        dict(prefix=prefix, pheno=pheno, geno=geno, validate=None,
             threads=threads, bim=rbim, seed=None))
    sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    # P+T scored in Target
    ppt_tar, sel_tar, tail_tar = dirty_ppt(loci, sumstats, X_test, y_test,
                                           diverse_geno, diverse_pheno,
                                           args.threads, memory)
    idx_tar = sel_tar.i.values
    prs_tar = diverse_geno[:, idx_tar].dot(sel_tar.slope)
    r2_tar = np.corrcoef(prs_tar, diverse_pheno.PHENO)[1, 0] ** 2
    return {r'$R^2_{ppt}$': r2_tar, 'EUR_frac': i, 'run': run}


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

    # Get the diverse sample to be test on
    opts = dict(test_size=1 / args.split, random_state=seed)
    r_out = train_test_split(rgeno, rpheno, **opts)
    rgeno, rgeno_test, rpheno, rpheno_test = r_out
    rpheno = rpheno.reset_index(drop=True)
    rpheno_test = rpheno_test.reset_index(drop=True)
    t_out = train_test_split(tgeno, tpheno, **opts)
    tgeno, tgeno_test, tpheno, tpheno_test = t_out
    tpheno = tpheno.reset_index(drop=True)
    tpheno_test = tpheno_test.reset_index(drop=True)
    diverse_geno = da.concatenate([rgeno_test, tgeno_test])
    diverse_pheno = pd.concat([rpheno_test, tpheno_test]).reset_index().sample(
        frac=1)
    diverse_pheno['i'] = diverse_pheno.index.values
    diverse_geno = diverse_geno[diverse_pheno.index.values, :]
    # Get LD info
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=args.window, justd=True,
                  threads=args.threads)
    result = pd.DataFrame()
    for run in range(25):
        results = [single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno,
                          diverse_geno, diverse_pheno, run, args.threads,
                          memory) for i in
                   np.clip(np.arange(0, 1.15, 0.15), a_min=0, a_max=1)]
        [os.remove(fn) for fn in glob('./*') if
         (os.path.isfile(fn) and fn != 'Rawlsian.tsv')]
        result = result.append(pd.DataFrame(results))
        result.to_csv('Rawlsian.tsv', sep='\t', index=False)
    cols = [c for c in result.columns if c != 'run']
    gp3 = result.loc[:, cols].groupby('EUR_frac')
    means = gp3.mean()
    errors = gp3.std()
    f, ax = plt.subplots()
    means.plot(yerr=errors, ax=ax)
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