#!/usr/bin/env python
# coding:utf-8
"""
  Author: Jose Sergio Hleap  --<2017>
  Purpose: pipeline to test the expected squared effect in single locus
  Created: 19/04/18
"""
from pptc2 import run_optimization_by
from comparison_ese import *
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


def random_locus_yielder(rbfile, tbfile, window, f_thr, threads, flip,
                         check, max_memory):
    """
    Read genotype and yield random locus of width window

    :param rbfile: Prefix for reference genotype fileset
    :param tbfile: Prefix for target genotype fileset
    :param window: size of the window in kbp
    :param f_thr:
    :param threads:
    :param flip:
    :param check:
    :param max_memory:
    :return:
    """
    (rbim, rfam, rg) = read_geno(rbfile, f_thr, threads, flip=flip, check=check,
                              max_memory=max_memory)
    (tbim, tfam, tg) = read_geno(tbfile, f_thr, threads, check=check,
                                 max_memory=max_memory)
    tsnps = tbim.snp
    rsnps = rbim.snp
    # subset the genotype file
    rindices = rbim[rbim.snp.isin(tsnps)].i
    rg = rg[:, rindices.tolist()]
    rbim = rbim[rbim.i.isin(rindices)].reset_index(drop=True)
    rbim['i'] = rbim.index.tolist()
    tindices = tbim[tbim.snp.isin(rsnps)].i
    tg = tg[:, tindices.tolist()]
    tbim = tbim[tbim.i.isin(tindices)].reset_index(drop=True)
    tbim['i'] = tbim.index.tolist()
    # Get the overlapping snps and their info
    shared = ['chrom', 'snp', 'pos', 'cm']
    mbim = rbim.merge(tbim, on=shared, suffixes=['_ref', '_tar'])
    # Get the number of bins or loci to be computed
    nbins = np.ceil(max(mbim.pos) / (window * 1000)).astype(int)
    # Get the limits of the loci
    bins = np.linspace(0, max(mbim.pos) + 1, num=nbins, endpoint=True,
                       dtype=int)
    if bins.shape[0] == 1:
        # Fix the special case in which the window is much bigger than the
        # range
        bins = np.append(bins, window * 1000)
    # Get the proper intervals into the dataframe
    mbim['windows'] = pd.cut(mbim['pos'], bins, include_lowest=True)
    grouped = mbim.groupby('windows')
    while True:
        df = grouped.get_group(random.sample(grouped.groups.keys(), 1)[0])
        ridx, tidx = df.i_ref.values, df.i_tar.values
        # Subset the genotype arrays
        nrg, ntg = rg[:, ridx], tg[:, tidx]
        df['i'] = list(range(nrg.shape[1]))
        assert df.shape[0] == nrg.shape[1]
        yield nrg, rfam, ntg, tfam, df


def compute_individual(tgeno, tpheno, loci, intfile, sumstats, avh2, h2, n,
                       scs_title, integral_only, column, memory, threads):
    typ = 'Integral' if integral_only else r'$\beta^2$'
    if not os.path.isfile(intfile):
        print('Computing %s' % typ)
        delayed_results = [
            dask.delayed(per_locus)(locus, sumstats, avh2, h2, n, i,
                                    integral_only=integral_only) for i, locus in
            enumerate(loci)]
        dask_options = dict(num_workers=args.threads,
                            pool=ThreadPool(args.threads))
        with ProgressBar(), dask.set_options(**dask_options):
            df = list(dask.compute(*delayed_results))
        df = pd.concat(df, ignore_index=True)
        df = df.merge(sumstats.reindex(
            columns=['snp', 'pvalue', 'beta_sq', 'slope', 'pos', 'i', 'beta']),
            on='snp')
        sorted_df = sortbylocus('%s_integral' % args.prefix, df, column=column,
                                title='Integral; %s' % scs_title)
        try:
            assert sorted_df.shape[0] == sumstats.shape[0]
        except AssertionError:
            print(sorted_df.head(), '\n', sumstats.head())
        df = prune_it(sorted_df, tgeno, tpheno, typ, step=1, n=1,
                      threads=threads, max_memory=memory)
    else:
        sorted_df = pd.read_csv('df_' + intfile, sep='\t')
        df = pd.read_csv(intfile, sep='\t')
    #sorted_df.to_csv('df_' + intfile, index=False, sep='\t')
    #df.to_csv(intfile, index=False, sep='\t')
    return df, sorted_df


def compute_all(prefix, index, locus, refl, tarl, seed, maxmem, avoid_causals,
                threads, window):
    if maxmem is not None:
        available_memory = maxmem
    else:
        available_memory = psutil.virtual_memory().available
    cache = Chest(available_memory=available_memory)
    # Unpack locus
    rg, rfam, tg, tfam, df = locus
    # compute phenotypes
    re_columns = {'a0_ref': 'a0', 'a1_ref': 'a1',
                  'mafs_ref': 'mafs'}
    rbim = df.reindex(columns=['chrom', 'snp', 'cm', 'pos', 'a0_ref', 'a1_ref',
                               'i_ref', 'mafs_ref', 'flip', 'i']).rename(
        columns=re_columns)
    re_columns = {'a0_tar': 'a0', 'a1_tar': 'a1',
                  'mafs_tar': 'mafs'}
    tbim = df.reindex(columns=['chrom', 'snp', 'cm', 'pos', 'a0_tar', 'a1_tar',
                               'i_tar', 'mafs_tar', 'flip', 'i']).rename(
        columns=re_columns)
    opts = dict(outprefix='%s_%d' % (refl, index), bfile=rg, h2=0.05, ncausal=1,
                normalize=True, uniform=False, snps=None, seed=seed, bfile2=tg,
                flip=False, max_memory=maxmem, freq_thresh=0, bim=rbim,fam=rfam,
                remove_causals=avoid_causals,)
    rpheno, h2, (rgeno, rbim, rtruebeta, rcausals) = qtraits_simulation(**opts)
    # make simulation for target
    opts.update(dict(outprefix='%s_%d' % (tarl, index), bfile=tg,  bfile2=rg,
                     causaleff=rcausals, fam=tfam, bim=tbim))
    tpheno, h2, (tgeno, tbim, truebeta, tcausals) = qtraits_simulation(**opts)
    # Compute summary statistics
    name = '%s_Locus%s' % (prefix, str(index))
    if os.path.isfile('%s.gwas' % name):
        c = 0
        new_name = name + '_%d' % c
        while os.path.isfile('%s.gwas' % new_name):
            new_name = name + '_%d' % c
            c += 1
        name = new_name
    opts.update(dict(prefix=name, pheno=rpheno, geno=rgeno,
                     bim=rbim, validate=2, threads=threads, flip=False,
                     high_precision=False))
    sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    assert sorted(sumstats.snp) == sorted(rbim.snp) == sorted(tbim.snp)
    # compute loci
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=window, justd=True,
                  max_memory=maxmem, threads=threads, extend=False)
    # compute all the methods in this run
    sum_snps = sumstats.snp.tolist()
    avh2 = h2 / len(sum_snps)
    n = tgeno.shape[0]
    scs_title = r'Realized $h^2$: %f' % h2
    # compute ese only integral
    intfile = '%s_%s_res.tsv' % (name, 'integral')
    integral, integral_df = compute_individual(tgeno, tpheno, loci, intfile,
                                               sumstats, avh2, h2, n, scs_title,
                                               True, 'ese', maxmem, threads)
    # include ppt
    pptfile = '%s_%s_res.tsv' % (name, 'ppt')
    n, m = tgeno.shape
    if not os.path.isfile(pptfile):
        opts = dict(by_range=None, sort_by='pvalue', loci=loci, h2=h2, m=m, n=n,
                    threads=threads, sum_stats=sumstats,available_memory=maxmem,
                    test_geno=X_test, test_pheno=y_test, tpheno=tpheno,
                    tgeno=tgeno, prefix='%s_pval_all' % name,
                    select_index_by='pvalue', clump_with='d_reference',
                    do_locus_ese=False, normalize=True, cache=cache)
        # run standard P + T
        ppt = run_optimization_by(**opts)
        snps = ppt['index_snps'].snp.tolist()
        ppt = prune_it(sumstats[sumstats.snp.isin(snps)], tgeno, tpheno, 'P + T',
                       step=1, n=1, threads=threads, max_memory=maxmem)
        #ppt.to_csv(pptfile, index=False, sep='\t')
    else:
        ppt = pd.read_csv(pptfile, sep='\t')
    # expecetd square effect
    eses = [
        individual_ese(sumstats, avh2, h2, n, x, loci, tgeno, tpheno, threads,
                       tbim, name, maxmem, prune_n=1) for x in [0, 1, 2]]

    # prune by pval
    resfile = '%s_%s_res.tsv' % (name, 'pvalue')
    if not os.path.isfile(resfile):
        sub = integral_df.reindex(columns=['locus', 'snp', 'pvalue', 'beta_sq',
                                        'slope', 'pos', 'beta', 'i'])
        pval = sub.sort_values(['pvalue', 'beta_sq'],
                               ascending=[True, False]).reset_index(drop=True)
        pval['index'] = pval.index
        #pval.to_csv('df_%s' % resfile, index=False, sep='\t')
        #assert pval.shape[0] == sumstats.shape[0]
        pval = prune_it(pval, tgeno, tpheno, 'pval', step=prunestep,
                        threads=threads, max_memory=maxmem, n=1)
        #pval.to_csv(resfile, index=False, sep='\t')
    else:
        pval = pd.read_csv(resfile, sep='\t')

    # prune by estimated beta
    betafile = '%s_%s_res.tsv' % (name, 'slope')
    beta, beta_df = compute_individual(tgeno, tpheno, loci, betafile,
                                       sumstats, avh2, h2, n, scs_title, False,
                                       'beta_sq', maxmem, threads)
    # include causals
    causals = integral_df.dropna(subset=['beta'])
    if not causals.empty:
        caus = sortbylocus('%s_causals' % name, causals, column='beta',
                           title='Causals; %s' % scs_title)
        caus = prune_it(caus, tgeno, tpheno, 'Causals', step=prunestep,
                        threads=threads, max_memory=maxmem, n=1)
        # caus.to_csv('%s_%s_res.tsv' % (name, 'causal'), index=False,
        #             sep='\t')
    else:
        caus = None
    res = pd.concat(eses + [pval, beta, caus, ppt, integral])
    res.to_csv('%s_finaldf.tsv' % name, sep='\t', index=False)
    return res


def plot(prefix, dfs):
    columns_my_order = ['Causals', 'P + T', 'pval', r'$\hat{\beta}^2$',
                        'Integral', 'ese AFR', 'ese EUR', 'ese cotag']
    dfs2 = []
    for d in dfs:
        d = d.groupby(['run', 'type'], as_index=False).agg({'R2': max})
        d[r'$R^2$ difference'] = d.R2.values - d[(d.type == 'P + T')].R2.values
        dfs2.append(d)
    df = pd.concat(dfs2)
    df.rename(columns={'type': 'Method'}, inplace=True)
    # df['Method'] = df['Method'].map({r'$\beta^2$': r'$\hat{\beta}^2$'})
    df.replace(to_replace=r'$\beta^2$', value=r'$\hat{\beta}^2$', inplace=True)
    df.replace(to_replace=r'$\hat{\beta^2}$', value=r'$\hat{\beta}^2$',
               inplace=True)
    a = df[df.Method == 'ese cotag'].loc[:, r'$R^2$ difference']
    for i, col in enumerate(columns_my_order[:-1]):
        b = df[df.Method == col].loc[:, r'$R^2$ difference']
        res = scipy.stats.ttest_ind(a, b, equal_var=False)
        if res.pvalue <= 0.05:
            new = col + '*'
            columns_my_order[i] = new
            df.replace(to_replace=col, value=new, inplace=True)
    df.to_csv('plotted_data.tsv', sep='\t')
    fig, ax = plt.subplots()
    sns.boxplot(x='Method', y=r'$R^2$ difference', data=df,
                order=columns_my_order,
                ax=ax)
    # df.boxplot(column=r'R^2 difference', by='Method')
    plt.ylabel(r'$R^2$ difference')
    plt.title(r'$R^2$ difference with P + T')
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig('%s_difference_1_scored.pdf' % prefix)


def main(args):
    now = time.time()
    seed = np.random.randint(1e4) if args.seed is None else args.seed
    # set Cache to protect memory spilling
    if args.maxmem is not None:
        memory = args.maxmem
    else:
        memory = psutil.virtual_memory().available
    cache = Chest(available_memory=memory)
    refl, tarl = args.labels
    # Get the locus
    locus = random_locus_yielder(args.refgeno, args.targeno, args.window, 0.01,
                                 args.threads, False, False, memory)
    cwd = os.getcwd()
    res=[]
    for i in range(args.n_runs):
        if not os.path.isdir('run%d' % i):
            os.mkdir('run%d' % i)
        os.chdir('run%d' % i)
        result = compute_all(args.prefix, i, next(locus), refl, tarl, seed,
                             memory, args.avoid_causals, args.threads,
                             args.window)
        result['run'] = i
        res.append(result)
        os.chdir(cwd)
    plot(args.prefix, res)


# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-b', '--refgeno', required=True,
                        help='prefix of the bed fileset in reference')
    parser.add_argument('-g', '--targeno', required=True,
                        help='prefix of the bed fileset in target')
    parser.add_argument('-L', '--labels', help=('Populations labels'), nargs=2)
    parser.add_argument('-H', '--h2', type=float, help='Heritability of trait',
                        default=0.05)
    parser.add_argument('-w', '--window', default=250, type=int,
                        help=('Size of the LD window. a.k.a locus'))
    parser.add_argument('-r', '--n_runs', default=10, type=int,
                        help='number of runs')
    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('-A', '--avoid_causals', default=False,
                        action='store_true', help='Remove causals from set')
    args = parser.parse_args()
    main(args)