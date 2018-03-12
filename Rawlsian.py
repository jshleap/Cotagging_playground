from comparison_ese import *


#@jit(parallel=True)
def tagged(D_r, snp_list, lds, pvals, sumstats):
    try:
        D_r = dd.from_dask_array(D_r, columns=snp_list) ** 2
    except AttributeError:
        D_r = pd.DataFrame(D_r, columns=snp_list) ** 2
    index, tag = get_tagged(snp_list, D_r, lds, pvals, sumstats)
    return index


def single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno, threads, seed,
           pvals, lds, memory):
    r_idx = np.arange(0, i)
    prefix = '%d_gwas' % i
    opts.update(
        dict(prefix=prefix, pheno=rpheno.iloc[r_idx], geno=rgeno[r_idx, :],
             validate=None, threads=threads, bim=rbim, seed=seed, check=False))
    sumstats, _, _, _, _ = plink_free_gwas(**opts)
    sumstats['beta_sq'] = sumstats.slope * sumstats.slope
    #index_snps = []
    delayed_results = [dask.delayed(tagged)(D_r, snp_list, lds, pvals, sumstats)
                       for snp_list, D_r, D_t in loci]
    with ProgressBar():
        print('Getting tag for', i, "EUR")
        d = list(dask.compute(*delayed_results, num_workers=threads,
                              memory_limit=memory))
        gc.collect()
    index_snps = [item for sublist in d for item in sublist]
    score = just_score(index_snps, sumstats, tpheno, tgeno)
    del sumstats, tpheno, tgeno, d
    gc.collect()
    # out = dirty_ppt(loci, sumstats, X_test, y_test, threads, split, seed,
    #                 memory, pvals=pvals, lds=lds)
    # ppt, selected, tail, _, _ = out
    # ppt.to_csv('%s_ppt.tsv' % prefix, sep='\t', index=False)
    # selected.to_csv('%s_selected.tsv' % prefix, sep='\t', index=False)
    # idx = selected.i.values
    # prs = tgeno[:, idx].dot(selected.slope)
    # est = np.corrcoef(prs, tpheno.PHENO)[1, 0] ** 2
    return {r'$R^2_{ppt}$': score, 'EUR_n': i, 'SNPs_n':len(index_snps)}


def main(args):
    rawls_final = '%s_finaldf.tsv' % args.prefix
    if not os.path.isfile(rawls_final):
        seed = np.random.randint(1e4) if args.seed is None else args.seed
        memory = 1E9 if args.maxmem is None else args.maxmem
        refl, tarl = args.labels
        # make simulations
        print('Simulating phenotype for reference population %s \n' % refl)
        opts = {'outprefix': refl, 'bfile': args.refgeno, 'h2': args.h2,
                'ncausal': args.ncausal, 'normalize': args.normalize,
                'uniform': args.uniform, 'snps': None, 'seed': seed,
                'bfile2': args.targeno, 'flip': args.gflip,
                'freq_thresh': args.freq_thresh, 'threads': args.threads}
        rpheno, h2, (rgeno, rbim, rtruebeta, rvec) = qtraits_simulation(**opts)
        # randomize individuals to check if it changes the result
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
                         causaleff=rbim.dropna(subset=['beta'])))
        if args.reference:
            tpheno, tgeno = rpheno, rgeno
        else:
            tpheno, h2, (tgeno, tbim, truebeta, tvec) = qtraits_simulation(
                **opts)
            # randomize individuals to check if it changes the result
            idx = np.arange(tgeno.shape[0])
            np.random.shuffle(idx)
            assert not np.array_equal(idx, tpheno.i.values)
            tgeno = tgeno[idx, :]
            tpheno = tpheno.iloc[idx].reset_index()
            tpheno['i'] = tpheno.index
        max_r2 = np.corrcoef(tpheno.gen_eff.values, tpheno.PHENO)[1, 0] ** 2
        # perform GWASes
        loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=args.window,
                      threads=args.threads, justd=True)
        # do ppt in AFR
        o = dict(prefix='Target_sumstats', pheno=tpheno, geno=tgeno, validate=2,
                 threads=args.threads, bim=tbim, seed=None, flip=args.gflip,
                 freq_thresh=args.freq_thresh, check=args.check)
        out = plink_free_gwas(**o)
        t_sumstats, t_X_train, t_X_test, t_y_train, t_y_test = out
        out = dirty_ppt(loci, t_sumstats, t_X_test, t_y_test, args.threads, 2,
                        None, memory, pvals=args.pvals, lds=args.lds)
        t_ppt, t_pre, t_pos, t_x_test, t_y_test = out
        idx_tar = t_pre.i.values
        t_prs = t_x_test[:, idx_tar].dot(t_pre.slope)
        t_r2 = np.corrcoef(t_prs, t_y_test.PHENO)[1, 0] ** 2
        with open('t_r2.pickle', 'wb') as F:
            pickle.dump(t_r2, F)
        array = [10, 40, 160, 640, 2560, 10240, 40960, 45000]
        #array = [40000]
        res = []
        for i in array:
            res.append(single(opts, i, rpheno, rbim, rgeno, loci, tpheno, tgeno,
                              args.threads, seed, 1, 0.6, memory))
            gc.collect()
        res = pd.DataFrame(res)
        res.to_csv(rawls_final, sep='\t', index=False)
    else:
        res = pd.read_table(rawls_final, sep='\t')
        tpheno = pd.read_table('%s.prs_pheno.gz' % args.prefix, sep='\t')
        max_r2 = np.corrcoef(tpheno.gen_eff.values, tpheno.PHENO)[1, 0] ** 2
        with open('t_r2.pickle', 'rb') as F:
            pickle.dump(F)
    f, ax = plt.subplots()
    ax2 = ax.twinx()
    r = res.plot(x='EUR_n', y=r'$R^2_{ppt}$', marker='.', ms=5, ax=ax)
    s = res.plot(x='EUR_n', y='SNPs_n', c='b', marker='.', ms=5, ax=ax2)
    plt.ylabel(r'AFR $R^2_{ppt}$')
    ax.axhline(max_r2, ls='-.', color='0.5', label='Causals')
    ax.axhline(t_r2, ls='-.', color='r', label=r'$%s_{P + T}$' % tarl)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(s.get_color())
    ax2.yaxis.label.set_color(s.get_color())
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
    parser.add_argument('--check', default=False, type=float)

    args = parser.parse_args()
    main(args)
