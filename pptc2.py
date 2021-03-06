from ese import *
from comparison_ese import sortbylocus
from utilities4cotagging import *
from subprocess import run, PIPE

np.seterr(all='raise')  # Debugging


def clumps(locus, sum_stats, ld_threshold, h2, avh2, n, do_locus_ese=False,
           select_index_by='pvalue', clump_with='d_reference'):
    """
    Get clumps from locus
    :param r2_reference: the r2 (LD) matrix for the subset
    :param sum_stats: subset of the summary statistics for the locus
    :param ld_threshold: the threshold for this run
    :return:
    """
    ascend = True if select_index_by == 'pvalue' else False
    # unpack the locus tuple
    snp_list, d_reference, d_target = locus
    # Name the rows and columns
    d_reference = pd.DataFrame(d_reference, index=snp_list, columns=snp_list)
    d_target = pd.DataFrame(d_target, index=snp_list, columns=snp_list)
    # subset sum_stats
    sum_stats = sum_stats[sum_stats.snp.isin(snp_list)]
    if h2 > 0:
        locus_ese = per_locus(locus, sum_stats, avh2, h2, n, 0, within=0)
        locus_ese = locus_ese.reindex(columns=['snp', 'ese']).rename(columns={
            'ese': 'locus_ese'})
        sum_stats = sum_stats.merge(locus_ese, on='snp')
    # Get the clumps pfr this locus
    clumps = {}
    while not sum_stats.empty:
        # get the index snp
        if do_locus_ese:
            #clump by locus ese
            #index = sum_stats.sort_values('locus_ese', ascending=ascend).iloc[0]
            index = sortbylocus('locus_ese', sum_stats, column='locus_ese',
                                ascending=ascend).iloc[0]
        else:
            # clump by locus pval
            #index = sum_stats.sort_values('pvalue', ascending=ascend).iloc[0]
            index = sortbylocus('pvalue', sum_stats, column='pvalue',
                                ascending=ascend).iloc[0]
        # get the clump around index for
        vec = (locals()[clump_with] ** 2).loc[index.snp, :]
        tag = vec[vec > ld_threshold].index.tolist()
        if tag:
            # Subset the sumary statistic dataframe with the snps in the clump
            sub_stats = sum_stats[sum_stats.snp.isin(tag)]
            if not sub_stats.empty:
                # Store the sub matrices in a tuple for ESE estimation
                n_locus = (tag, d_reference.loc[tag, tag], d_target.loc[tag,
                                                                        tag])
                # Compute ESE and include it into the main dataframe for this
                # clump
                try:
                    df_ese = per_locus(n_locus, sub_stats, avh2, h2, n, 0,
                                       within=0)
                except:
                    df_ese = per_locus(n_locus, sub_stats, avh2, h2, n, 0,
                                       within=0)
                ss = sub_stats.merge(df_ese.reindex(columns=['snp', 'ese']),
                                     on='snp')
                # Get the highest ESE of the clump
                max_ese = ss.nlargest(1, 'ese')
                if select_index_by == 'locus_ese':
                    max_l_ese = ss.nlargest(1, 'locus_ese')
                else:
                    max_l_ese = pd.DataFrame([{'snp': 'None',
                                               'locus_ese': 'None'}])
                try:
                    key = (index.snp, index.pvalue, max_ese.snp.iloc[0],
                           max_ese.ese.iloc[0], max_l_ese.snp.iloc[0],
                           max_l_ese.locus_ese.iloc[0])
                except:
                    key = (index.snp, index.pvalue)
                clumps[key] = ss
                # except:
                #     pass
                # remove the clumped snps from the summary statistics dataframe
                sum_stats = sum_stats[~sum_stats.snp.isin(tag)]
            else:
                #Tagged SNPS has been tagged already
                sum_stats = sum_stats[~sum_stats.snp.isin([index.snp])]
        else:
            # none element tagged
            sum_stats = sum_stats[~sum_stats.snp.isin([index.snp])]
    return clumps


def compute_clumps(loci, sum_stats, ld_threshold, h2, avh2, n, threads, cache,
                   memory, select_index_by='pvalue', clump_with='d_reference',
                   do_locus_ese=False):
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
    opts = dict(sum_stats=sum_stats, ld_threshold=ld_threshold, h2=h2,
                avh2=avh2, n=n, do_locus_ese=do_locus_ese,
                select_index_by=select_index_by, clump_with=clump_with)
    delayed_results = [dask.delayed(clumps)(locus=locus, **opts) for locus in
                       loci]
    with ProgressBar():
        print("Identifying clumps with R2 threshold of %.3f" % ld_threshold)
        l = list(dask.compute(*delayed_results, num_workers=threads,
                              memory_limit=memory, cache=cache,
                              pool=ThreadPool(threads)))
    return dict(pair for d in l for pair in d.items())


def main(prefix, refgeno, refpheno, targeno, tarpheno, h2, labels, LDwindow,
         sum_stats, seed=None, max_memory=None, threads=1, by='pvalue',
         by_range=None, clump_with='d_reference', **kwargs):
    """
    Execute transferability code
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
    picklefile = '%s.pckl' % prefix
    loaded = False
    if os.path.isfile(picklefile):
        with open(picklefile, 'rb') as pckl:
            (opts, ffam, fbim, fgeno, tbim, tfam, tgeno, rbim, rfam, rgeno,
             seed) = pickle.load(pckl)
        loaded = True
        common_snps = list(set(rbim.snp).intersection(tbim.snp))
        causals = pd.read_table('merged.causaleff', delim_whitespace=True)
    else:
        # Merge target and reference
        (rbim, rfam, rgeno) = read_geno(refgeno, kwargs['freq_thresh'],
                                        threads, check=kwargs['check'],
                                        max_memory=max_memory)
        (tbim, tfam, tgeno) = read_geno(targeno, kwargs['freq_thresh'],
                                        threads, check=kwargs['check'],
                                        max_memory=max_memory)
        common_snps = list(set(rbim.snp).intersection(tbim.snp))
        plink_args = ['plink', '--bfile', refgeno, '--bmerge',
                      '%s.bed' % targeno, '%s.bim' % targeno,
                      '%s.fam' % targeno, '--make-bed', '--out',
                      '%s_merged' % prefix]
        if not os.path.isfile('%s.bed' % plink_args[-1]):
            run(plink_args)
        (fbim, ffam, fgeno) = read_geno("%s.bed" % plink_args[-1],
                                        kwargs['freq_thresh'], threads,
                                        check=kwargs['check'],
                                        usable_snps=common_snps,
                                        max_memory=max_memory)
        # split them again
        rbim = tbim = fbim
        tfam = ffam[ffam.iid.isin(tfam.iid.tolist())].reset_index(drop=True)
        tidx = tfam.i.tolist()
        tfam.i = tfam.index.tolist()
        tgeno = fgeno[tidx, :]
        rfam = ffam[ffam.iid.isin(rfam.iid.tolist())].reset_index(drop=True)
        ridx = rfam.i.tolist()
        rfam.i = rfam.index.tolist()
        rgeno = fgeno[ridx, :]

        opts = dict(outprefix="merged", bfile=fgeno, bim=fbim, fam=ffam, h2=h2,
                    ncausal=kwargs['ncausal'], normalize=kwargs['normalize'],
                    uniform=kwargs['uniform'], snps=None, seed=seed,
                    flip=kwargs['gflip'], max_memory=max_memory,
                    high_precision_on_zero=kwargs['highp'],
                    freq_thresh=0.0)
        with open('%s.pckl' % prefix, 'wb') as pckl:
            pickle.dump((opts, ffam, fbim, fgeno, tbim, tfam, tgeno, rbim,
                         rfam, rgeno, seed), pckl)
    plink_args = ['plink', '--bfile', refgeno, '--bmerge', '%s.bed' % targeno,
                  '%s.bim' % targeno, '%s.fam' % targeno, '--make-bed',
                  '--out', '%s_merged' % prefix]
    if not os.path.isfile('%s.bed' % plink_args[-1]):
        run(plink_args)
    # if not loaded:
    #     (fbim, ffam, fgeno) = read_geno(targeno, kwargs['freq_thresh'],
    #                                     threads, check=kwargs['check'],
    #                                     usable_snps=common_snps,
    #                                     max_memory=max_memory)

    opts = dict(outprefix="merged", bfile=fgeno, bim=fbim, fam=ffam, h2=h2,
                ncausal=kwargs['ncausal'], normalize=kwargs['normalize'],
                uniform=kwargs['uniform'], snps=None, seed=seed,
                flip=kwargs['gflip'], max_memory=max_memory,
                high_precision_on_zero=kwargs['highp'],
                freq_thresh=0.0)
    # If pheno is None for the reference, make simulation
    if isinstance(refpheno, str):
        rpheno = dd.read_table(refpheno, blocksize=25e6, delim_whitespace=True)
        tpheno = dd.read_table(tarpheno, blocksize=25e6, delim_whitespace=True)
    elif refpheno is None:
        # make simulation for reference
        print('Simulating phenotype for reference population %s \n' % refl)
        fpheno, h2, (fgeno, fbim, ftruebeta, fvec) = qtraits_simulation(**opts)
        tpheno = fpheno[fpheno.iid.isin(tfam.iid)]
        rpheno = fpheno[fpheno.iid.isin(rfam.iid)]
    causal_betas = fvec.reindex(columns=['snp', 'beta'])
    if isinstance(sum_stats, str):
        sum_stats = pd.read_table(sum_stats, delim_whitespace=True)
    else:
        opts.update({'bim': rbim, 'fam': rfam})
        out = plink_free_gwas('train', rpheno, rgeno, **opts)
        sum_stats, X_train, X_test, y_train, y_test = out
    sum_stats = sum_stats.merge(causal_betas, on='snp', how='outer')
    print("Reference bim's shape: %d, %d" % (rbim.shape[0], rbim.shape[1]))
    print("Target bim's shape: %d, %d" % (tbim.shape[0], tbim.shape[1]))
    # process causals
    causal_snps = fvec.snp.tolist()
    r2_causal = just_score(causal_snps, sum_stats, tpheno, tgeno)
    print('R2_causal for Target', r2_causal)
    r2_causal_ref = just_score(causal_snps, sum_stats, rpheno, rgeno)
    print('R2_causal for reference', r2_causal_ref)
    assert np.allclose(r2_causal, r2_causal_ref, atol=0.1, rtol=0.1)
    # Compute Ds
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=LDwindow, justd=True,
                  threads=threads, max_memory=max_memory)
    # optimize R2
    n, m = X_train.shape
    if by is None:
        opts = dict(by_range=None, sort_by='pvalue', loci=loci, h2=h2, m=m,
                    threads=threads, cache=cache, sum_stats=sum_stats, n=n,
                    available_memory=available_memory, test_geno=X_test,
                    test_pheno=y_test, tpheno=tpheno, tgeno=tgeno,
                    prefix='%s_pval_all' % prefix, select_index_by='pvalue',
                    clump_with=clump_with, do_locus_ese=False,
                    normalize=kwargs['normalize'],
                    clump_function=compute_clumps
                    )
        print('Running standard P + T')
        if os.path.isfile('pvalue.pickle'):
            with open('pvalue.pickle', 'rb') as pckl:
                pvalue = pickle.load(pckl)
        else:
            pvalue = run_optimization_by(**opts)
            with open('pvalue.pickle', 'wb') as pckl:
                pickle.dump(pvalue, pckl)
        ##
        print('Clumping with pval, select index with pval, select across with '
              'ese')
        if os.path.isfile('ppe.pickle'):
            with open('ppe.pickle', 'rb') as pckl:
                pval_pval_ese = pickle.load(pckl)
        else:
            opts.update(sort_by='ese', prefix='%s_pval_pval_ese' % prefix,
                        select_index_by='pvalue')
            pval_pval_ese = run_optimization_by(**opts)
            with open('ppe.pickle', 'wb') as pckl:
                pickle.dump(pval_pval_ese, pckl)
        ##
        print('Clumping with pval, select index with pval, select across with '
              'locus ese')
        if os.path.isfile('ppl.pickle'):
            with open('ppl.pickle', 'rb') as pckl:
                pval_pval_lese = pickle.load(pckl)
        else:
            opts.update(sort_by='locus_ese', prefix='%s_pval_pval_lese'%prefix,
                        select_index_by='pvalue')
            pval_pval_lese = run_optimization_by(**opts)
            with open('ppl.pickle', 'wb') as pckl:
                pickle.dump(pval_pval_lese, pckl)
        #
        print('Clumping with pval, select index with ese, select across with '
              'ese')
        if os.path.isfile('pep.pickle'):
            with open('pep.pickle', 'rb') as pckl:
                pval_ese_pval = pickle.load(pckl)
        else:
            opts.update(sort_by='pvalue', prefix='%s_pval_ese_pval' % prefix,
                        select_index_by='ese')
            pval_ese_pval = run_optimization_by(**opts)
            with open('pep.pickle', 'wb') as pckl:
                pickle.dump(pval_ese_pval, pckl)
        #
        print('Clumping with pval, select index with ese, select across with '
              'ese')
        if os.path.isfile('pee.pickle'):
            with open('pee.pickle', 'rb') as pckl:
                pval_ese_ese = pickle.load(pckl)
        else:
            opts.update(sort_by='ese', prefix='%s_pval_ese_ese' % prefix,
                        select_index_by='ese')
            pval_ese_ese = run_optimization_by(**opts)
            with open('pee.pickle', 'wb') as pckl:
                pickle.dump(pval_ese_ese, pckl)
        #
        print('Clumping with pval, select index with ese, select across with '
              'locus ese')
        if os.path.isfile('pel.pickle'):
            with open('pel.pickle', 'rb') as pckl:
                pval_ese_lese = pickle.load(pckl)
        else:
            opts.update(sort_by='locus_ese', prefix='%s_pval_ese_lese' %prefix,
                        select_index_by='ese')
            pval_ese_lese = run_optimization_by(**opts)
            with open('pel.pickle', 'wb') as pckl:
                pickle.dump(pval_ese_lese, pckl)
        #
        print('Clumping with pval, select index with lpcus ese, select across '
              'with pval')
        if os.path.isfile('plp.pickle'):
            with open('plp.pickle', 'rb') as pckl:
                pval_lese_pval = pickle.load(pckl)
        else:
            opts.update(sort_by='pvalue', prefix='%s_pval_lese_pval' % prefix,
                        select_index_by='locus_ese')
            pval_lese_pval = run_optimization_by(**opts)
            with open('plp.pickle', 'wb') as pckl:
                pickle.dump(pval_lese_pval, pckl)
        #
        print('Clumping with pval, select index with locus ese, select across '
              'with ese')

        # Clump pval, select index with lese, select across with ese
        opts.update(sort_by='ese', prefix='%s_pval_lese_ese' % prefix,
                    select_index_by='locus_ese')
        pval_lese_ese = run_optimization_by(**opts)

        print('Clumping with  pval, select index with locus ese, select across'
              ' with locus ese')
        opts.update(sort_by='locus_ese', prefix='%s_pval_lese_lese' % prefix,
                    select_index_by='locus_ese')
        pval_lese_lese = run_optimization_by(**opts)

        print('Clump locus ese, select index with pval, select across with '
              'pval')
        opts.update(sort_by='pvalue', prefix='%s_lese_pval_pval' % prefix,
                    select_index_by='pvalue', do_locus_ese=True)
        lese_pval_pval = run_optimization_by(**opts)

        print('Clumping with locus ese, select index with pval, select across '
              'with ese')
        opts.update(sort_by='ese', prefix='%s_lese_pval_ese' % prefix,
                    select_index_by='pvalue', do_locus_ese=True)
        lese_pval_ese = run_optimization_by(**opts)

        print('Clumping with locus ese, select index with pval, select across '
              'with locus ese')
        opts.update(sort_by='locus_ese', prefix='%s_lese_pval_lese' % prefix,
                    select_index_by='pvalue', do_locus_ese=True)
        lese_pval_lese = run_optimization_by(**opts)

        print('Clumping with locus ese, select index with ese, select across '
              'with pval')
        opts.update(sort_by='pvalue', prefix='%s_lese_ese_pval' % prefix,
                    select_index_by='ese', do_locus_ese=True)
        lese_ese_pval = run_optimization_by(**opts)

        print('Clumping with locus ese, select index with ese, select across '
              'with ese')
        opts.update(sort_by='ese', prefix='%s_lese_ese_ese' % prefix,
                    select_index_by='ese', do_locus_ese=True)
        lese_ese_ese = run_optimization_by(**opts)

        print('Clumping with locus ese, select index with ese, select across '
              'with locus ese')
        opts.update(sort_by='locus_ese', prefix='%s_lese_ese_lese' % prefix,
                    select_index_by='ese', do_locus_ese=True)
        lese_ese_lese = run_optimization_by(**opts)

        print('Clumping with locus ese, select index with locus ese, select '
              'across with pval')
        opts.update(sort_by='pvalue', prefix='%s_lese_lese_pval' % prefix,
                    select_index_by='locus_ese', do_locus_ese=True)
        lese_lese_pval = run_optimization_by(**opts)

        print('Clumping with locus ese, select index with locus ese, select '
              'across with ese')
        opts.update(sort_by='ese', prefix='%s_lese_lese_ese' % prefix,
                    select_index_by='locus_ese', do_locus_ese=True)
        lese_lese_ese = run_optimization_by(**opts)

        print('Clumping with locus ese, select index with locus ese, select '
              'across with locus ese')
        opts.update(sort_by='locus_ese', prefix='%s_lese_lese_lese' % prefix,
                    select_index_by='locus_ese', do_locus_ese=True)
        lese_lese_lese = run_optimization_by(**opts)

        cols = [r'$R^{2}_{pvalue}$', r'$R^{2}_{pvalue}$ ref',
                r'$R^{2}_{pval pval ese}$', r'$R^{2}_{pval pval ese}$ ref',
                r'$R^{2}_{pval pval lese}$', r'$R^{2}_{pval pval lese}$ ref',
                r'$R^{2}_{pval ese pval}$', r'$R^{2}_{pval ese pval}$ ref',
                r'$R^{2}_{pval ese ese}$', r'$R^{2}_{pval ese ese}$ ref',
                r'$R^{2}_{pval ese lese}$', r'$R^{2}_{pval ese lese}$ ref',
                r'$R^{2}_{pval lese pval}$', r'$R^{2}_{pval lese pval}$ ref',
                r'$R^{2}_{pval lese ese}$', r'$R^{2}_{pval lese ese}$ ref',
                r'$R^{2}_{pval lese lese}$', r'$R^{2}_{pval lese lese}$ ref',
                r'$R^{2}_{lese pval ese}$', r'$R^{2}_{lese pval ese}$ ref',
                r'$R^{2}_{lese pval ese}$', r'$R^{2}_{lese pval ese}$ ref',
                r'$R^{2}_{lese pval lese}$', r'$R^{2}_{lese pval lese}$ ref',
                r'$R^{2}_{lese ese pval}$', r'$R^{2}_{lese ese pval}$ ref',
                r'$R^{2}_{lese ese ese}$', r'$R^{2}_{lese ese ese}$ ref',
                r'$R^{2}_{lese ese lese}$', r'$R^{2}_{lese ese lese}$ ref',
                r'$R^{2}_{lese lese pval}$', r'$R^{2}_{lese lese pval}$ ref',
                r'$R^{2}_{lese lese ese}$', r'$R^{2}_{lese lese ese}$ ref',
                r'$R^{2}_{lese lese lese}$', r'$R^{2}_{lese lese lese}$ ref',
                r'$R^{2}_{causals}']

        vals = [pvalue['R2'], pvalue['R2_ref'],
                pval_pval_ese['R2'], pval_pval_ese['R2_ref'],
                pval_pval_lese['R2'], pval_pval_lese['R2_ref'],
                pval_ese_pval['R2'], pval_ese_pval['R2_ref'],
                pval_ese_ese['R2'],  pval_ese_ese['R2_ref'],
                pval_ese_lese['R2'], pval_ese_lese['R2_ref'],
                pval_lese_pval['R2'], pval_lese_pval['R2_ref'],
                pval_lese_ese['R2'], pval_lese_ese['R2_ref'],
                pval_lese_lese['R2'], pval_lese_lese['R2_ref'],
                lese_pval_pval['R2'], lese_pval_pval['R2_ref'],
                lese_pval_ese['R2'], lese_pval_ese['R2_ref'],
                lese_pval_lese['R2'], lese_pval_lese['R2_ref'],
                lese_ese_pval['R2'], lese_ese_pval['R2_ref'],
                lese_ese_ese['R2'], lese_ese_ese['R2_ref'],
                lese_ese_lese['R2'], lese_ese_lese['R2_ref'],
                lese_lese_pval['R2'], lese_lese_pval['R2_ref'],
                lese_lese_ese['R2'], lese_lese_ese['R2_ref'],
                lese_lese_lese['R2'], lese_lese_lese['R2_ref'],
                r2_causal]

        pd.DataFrame([dict(zip(cols, vals))]).to_csv('%s.tsv' % prefix,
                                                     sep='\t', index=False)
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
    parser.add_argument('--highp', action='store_true',
                        help='Use arbitrary precision')

    args = parser.parse_args()
    main(args.prefix, args.reference, args.refpheno, args.target,
         args.tarpheno, args.h2, args.labels, args.window, args.sumstats,
         seed=args.seed, threads=args.threads, ncausal=args.ncausal,
         normalize=True, by=args.by, uniform=args.uniform,  within=args.within,
         by_range=args.r_range, max_memory=args.maxmem, split=args.split,
         flip=args.flip, gflip=args.gflip, check=args.check, highp=args.highp,
         freq_thresh=args.freq_thresh)
