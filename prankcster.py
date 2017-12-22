#!/usr/bin/env python
# coding:utf-8
"""
  Author: Jose Sergio Hleap  --<>
  Purpose: Optimize the mixing of cotagging and P + T and chose the snps based 
  pn the optimized ranking
  Created: 10/02/17
"""

import matplotlib

matplotlib.use('Agg')
from ppt import *
from matplotlib import pyplot as plt
from joblib import delayed, Parallel

plt.style.use('ggplot')


# ---------------------------------------------------------------------------
def read_n_sort_clumped(resultsfn, allsnps):
    """
    Read the <target>.results file, get the best clumping in the P+T and do the
    "sorting"
    
    :param str resultsfn: filename of the result file of a P+T.py run
    :param :class pd.Series allsnps: Series with allowed set of snps
    """
    # Get the path to the results
    path = os.path.split(resultsfn)[0]
    # Read the Clumped results file
    res = pd.read_table(resultsfn, sep='\t')
    # Set the path and filename of the best clumped file
    clumpfn = '%s/%s.clumped' % (path if path != '' else '.',
                                 res.nlargest(1, 'R2').File.iloc[0])
    # Return the dataftame of the parse of the clumpled file
    return parse_sort_clump(clumpfn, allsnps)


# ----------------------------------------------------------------------
def strategy_hyperbola(x, y, alpha):
    """
    strategy for new rank with summation
    
    :param :class pd.Series x: Series with the first range to be combined
    :param :class pd.Series y: Series with the second range to be combined
    :param float alpha: Float with the weight to be combined by
    """
    den = (alpha / x) + ((1 - alpha) / y)
    return 1 / den


# ----------------------------------------------------------------------
def strategy_sum(x, y, alpha):
    """
    strategy for new rank with hypergeometry
    
    :param :class pd.Series x: Series with the first range to be combined
    :param :class pd.Series y: Series with the second range to be combined
    :param float alpha: Float with the weight to be combined by
    """
    return (alpha * x) + ((1 - alpha) * y)


# ---------------------------------------------------------------------------
def single_alpha_qr(prefix, alpha, merge, plinkexe, bfile, sumstats, qrange, qr,
                    tar, allele_file, maxmem=1700, threads=8, strategy='sum',
                    score_type='SUM'):
    """
    Single execution of the alpha loop for paralellization
    
    :param str prefix: Prefix for outputs
    :param float alpha: proportion of cotagging included
    :param :class pd.DataFrame: Dataframe with the merge cotagging and P + T
    :param str plinkexe: Path and executable of plink
    :param str bfile: Prefix of plink-bed fileset
    :param str sumstats: File with the summary statistics in plink format 
    :param str qrange: File with the ranges to be passed to the --q-score-range
    :param str phenofile: Filename with the phenotype
    :param float frac_snps: Numer of SNPs in one percent
    :param :class pd.DataFrame qr: DataFrame with the qranges
    :param str tar: label of target population
    :param str allele_file: Filename with allele order
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    """
    # Define qifile name
    qfile = '%s.qfile'
    # Define output prefix including the alpha
    ou = '%s_%.2f' % (prefix, alpha)
    qfile = qfile % ou
    # Get the apropriate rankings
    cot = merge.Index_Cotag
    ppt = merge.loc[:, 'Index_%sPpT' % tar]
    # Compute the new ranking with the given strategy
    strategy_fnc = globals()['strategy_%s' % strategy]
    merge['New_rank'] = strategy_fnc(cot, ppt, alpha)
    new_rank = merge.sort_values('New_rank')
    new_rank['New_rank'] = new_rank.reset_index(drop=True).index.tolist()
    # Wirte results of the new ranking to file
    new_rank.loc[:, ['SNP', 'New_rank']].to_csv(qfile, sep=' ', header=False,
                                                index=False)
    # scorefile = '%s.score' % prefix
    normalize_geno = True if score_type == 'SUM' else False
    df = qrscore(plinkexe, bfile, sumstats, qrange, qfile, allele_file, ou, qr,
                 maxmem, threads, alpha, prefix, normalized_geno=normalize_geno)
    # Return the results dataframe
    return df


# ---------------------------------------------------------------------------
def rank_qr(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile, alphastep,
            plinkexe, tar, allele_file, prunestep=1, qrangefn=None, maxmem=1700,
            threads=1, strategy='sum', every=False, score_type='SUM'):
    """
    Estimate the new rank based on the combination of the cotagging and P+T rank
    
    :param str allele_file: Filename with the allele order
    :param str prefix: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param :class pd.Dataframe sorted_cotag:Filename with results in the sorting
    :param list clumpe: List of tuples with the P+T results
    :param str sumstats: File with the summary statistics in plink format 
    :param str phenofile: Filename with the phenotype
    :param float alphastep: Step of the alpha to be explored
    :param str plinkexe: Path and executable of plink
    :param str tar: Label of target population
    :param int prunestep: Step of the prunning
    :param str qrangefn: Filename of a previously made qrange
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    :param bool every: test one snp at a time
    """
    # Generate the alpha-space
    space = np.concatenate((np.array([0, 0.05]), np.arange(0.1, 1 + alphastep,
                                                           alphastep)))
    # Get the number of valid snps fdrom the sorted cotag dataframe
    nsnps = sorted_cotag.shape[0]
    frac_snps = nsnps / 100
    # Generate the qrange file?
    qr, qrange = gen_qrange(prefix, nsnps, prunestep, every=every,
                            qrangefn=qrangefn)
    # Deal with the P+T results
    if isinstance(clumpe, list):
        premerge = clumpe[0][0].merge(clumpe[1][0], on='SNP', suffixes=[
            '_%sPpT' % clumpe[0][1], '_%sPpT' % clumpe[1][1]])
    else:
        premerge = clumpe.rename(columns={'Index': 'Index_%sPpT' % tar}).head()
    # Merge the sorted cotag and the P+T
    merge = sorted_cotag.merge(premerge, on='SNP', suffixes=['Cotag', 'PpT'])
    merge = merge.rename(columns={'Index': 'Index_Cotag'})
    # Execute the optimization
    df = Parallel(n_jobs=int(threads))(delayed(single_alpha_qr)(
        prefix, alpha, merge, plinkexe, bfile, sumstats, qrange, qr, tar,
        allele_file, maxmem, threads, strategy, score_type) for alpha in tqdm(
        space))
    # Return the list of dataframes with the optimization results
    return df, qrange, qr, merge


# ---------------------------------------------------------------------------
def optimize_alpha_plink(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile,
                   plinkexe, alphastep, tar, prune_step=1, qrangefn=None,
                   maxmem=1700, threads=1, strategy='sum', every=False,
                   score_type='SUM'):
    """
    Do a line search for the best alpha in nrank = alpha*rankP+T + (1-alpha)*cot
    
    :param str prefix: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param :class pd.Dataframe sorted_cotag:Filename with results in the sorting
    :param list clumpe: List of tuples with the P+T results
    :param str sumstats: File with the summary statistics in plink format 
    :param str phenofile: Filename with the phenotype
    :param float alphastep: Step of the alpha to be explored
    :param str plinkexe: Path and executable of plink
    :param str tar: Label of target population
    :param int prune_step: Step of the prunning
    :param str qrangefn: Filename of a previously made qrange
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    :param bool every: test one snp at a time
    """
    # Set output name
    outfn = '%s_optimized.tsv' % prefix
    picklfn = '%s_optimized.pickle' % prefix
    # Execute the ranking
    if not os.path.isfile(picklfn):
        d, r, qr, merge = rank_qr(prefix, bfile, sorted_cotag, clumpe, sumstats,
                                  phenofile, alphastep, plinkexe, tar,
                                  prune_step,
                                  qrangefn, maxmem, threads, strategy, every,
                                  score_type)
        df = pd.concat(d)
        df.to_csv(outfn, sep='\t', index=False)
        with open(picklfn, 'wb') as F:
            pickle.dump((df, r, qr, merge), F)
    else:
        # df = pd.read_table(outfn, sep='\t')
        with open(picklfn, 'rb') as F:
            df, r, qr, merge = pickle.load(F)
    # Plot the optimization
    piv = df.loc[:, ['SNP kept', 'alpha', 'R2']]
    piv = piv.pivot(index='SNP kept', columns='alpha', values='R2').sort_index()
    piv.plot(colormap='copper', alpha=0.5)
    plt.ylabel('$R^2$')
    plt.tight_layout()
    plt.savefig('%s_alphas.pdf' % (prefix))
    # Returned the sorted result dataframe
    results = df.sort_values('R2', ascending=False).reset_index(drop=True)
    return results, r, qr, merge


# ----------------------------------------------------------------------
def read_n_sort_cotag(prefix, cotagfn, freq, threads=1, weight=None,
                      column='Cotagging'):
    """
    Smart sort the cotag file

    :param str prefix: Prefix for outputs
    :param str cotagfn: Filename tsv with cotag results
    :param :class pd.DataFrame freq: Dataframe with the plink formatted freqs.
    :param str column: Column to sort by
    """
    # Read cotag
    cotags = pd.read_table(cotagfn, sep='\t')
    if weight is not None:
        cotags = cotags.merge(weight, on='SNP')
        cotags['weighted'] = cotags.preweighted * cotags.loc[:, column]
        column = 'weighted'
    # Smart sort it
    # Check for previous sort
    prev = glob('*Cotagging.pickle')
    if prev != []:
        if not os.path.isfile('%s_Cotagging.pickle' % prefix):
            shutil.copy(prev[0], '%s_Cotagging.pickle' % prefix)
    df, _ = smartcotagsort(prefix, cotags[cotags.SNP.isin(freq.SNP)],
                           column=column, threads=threads)
    # Returned the sorted dataframe
    return df.reset_index()


# ----------------------------------------------------------------------
def weighted_squares(out, sortedcotag):
    """
    Perform the ranking based on the weigted squared effect sizes as an 
    approximation for equation 4 in transferability of PRS across populations
    :param out: prefix of outputs
    :param sortedcotag: a dataframe with cotagging and sumstats information
    """
    # Compute the expected square effect
    sortedcotag['b2'] = (sortedcotag.BETA ** 2) * (1 - sortedcotag.P)
    # Compute the wieghted score
    sortedcotag['prod'] = sortedcotag.b2 * sortedcotag.Cotagging
    # Sort by the newly created score
    df, bt = smartcotagsort(out, sortedcotag, column='prod')
    df = df.loc[:, ['SNP', 'Index']]
    # Define qifile name and write it
    qfile = '%s.qfile' % out
    df.to_csv(qfile, sep=' ', header=False, index=False)
    return qfile


# ----------------------------------------------------------------------
def prankcster_plink(prefix, targetbed, referencebed, cotagfn, ppt_results_tar,
                     ppt_results_ref, sumstats, pheno, plinkexe, alpha_step,
                     labels, prune_step, sortresults, allele_file,
                     freq_threshold=0.1,
                     h2=None, qrangefn=None, maxmem=1700, threads=1,
                     strategy='sum',
                     every=False, column='Cotagging', splits=5, weight=False,
                     score_type='SUM'):
    """
    Execute the code and plot the comparison
    
    :param str prefix: Prefix for outputs
    :param str targetbed: Prefix of plink-bed fileset of the target population
    :param str referencebed: Prefix of plink-bed fileset of the ref population
    :param str cotagfn: Filename tsv with cotag results
    :param str ppt_results_tar: path to P+T relsults in the target pop
    :param str ppt_results_ref: path to P+T relsults in the reference pop
    :param str sumstats: File with the summary statistics in plink format 
    :param str pheno: Filename with the phenotype
    :param float alpha_step: Step of the alpha to be explored
    :param str plinkexe: Path and executable of plink
    :param list labels: List with labels of reference and target populations
    :param int prune_step: Step of the prunning
    :param str sortresults: Filename with results in the sorting inlcuding path
    :param str allele_file: Filename with the allele order information
    :param float freq_threshold: Threshold for frequency filtering
    :param float h2: Heritability of the phenotype
    :param str qrangefn: Filename of a previously made qrange
    :param int maxmem: Maximum allowed memory
    :param int threads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    :param bool every: test one snp at a time
    :param str column: Column to sort by
    :param str split: number of folds for crossvalidation
    :param bool weight: Perform the sorting based on the weighted square effect
    sizes
    """
    print('Performing prankcster')
    # Unpack population labels
    ref, tar = labels
    # Get frequencies of both populations and merge them
    f1 = read_freq(referencebed, plinkexe, freq_threshold=freq_threshold)
    f2 = read_freq(targetbed, plinkexe, freq_threshold=freq_threshold)
    frqs = f1.merge(f2, on=['CHR', 'SNP'], suffixes=['_%s' % ref,
                                                     '_%s' % tar])
    # Read summary statistics
    if isinstance(sumstats, str):
        ss = pd.read_table(sumstats, delim_whitespace=True)
    else:
        assert isinstance(sumstats, pd.DataFrame)
        ss = sumstats
    # Normalize betas and create score file
    maf = 'MAF_%s' % tar
    a1 = 'A1_%s' % tar
    ss = ss.merge(frqs.loc[:, ['SNP', a1, maf]], on='SNP')
    ss['norm'] = np.sqrt((2 * ss.loc[:, maf]) * (1 - ss.loc[:, maf]))
    ss['BETA_norm'] = ss.BETA / ss.norm
    scorefn = '%s.score' % prefix
    ss.loc[:, ['SNP', a1, 'BETA_norm']].to_csv(scorefn, sep=' ', index=False,
                                               header=False)
    # get weighted squares 
    if weight:
        ws = ss.loc[:, ['SNP', 'BETA_norm', 'P']]
        ws['preweighted'] = ((ws.BETA_norm ** 2) * (1 - ws.P))
        ws = ws.loc[:, ['SNP', 'preweighted']]
        weight = ws
    else:
        weight = None
    # Read the cotag scores
    if os.path.isfile('%s.sorted_cotag' % prefix):
        sorted_cotag = pd.read_table('%s.sorted_cotag' % prefix, sep='\t')
    else:
        sorted_cotag = read_n_sort_cotag(prefix, cotagfn, f2, threads=threads,
                                         column=column, weight=weight)
        sorted_cotag.to_csv('%s.sorted_cotag' % prefix, sep='\t', index=False)
    nsnps = sorted_cotag.shape[0]
    frac_snps = nsnps / 100
    # Read and sort the P + T results
    clumpetar = read_n_sort_clumped(ppt_results_tar, frqs.SNP)
    # clumpetar = clumpetar[clumpetar.SNP.isin(frqs.SNP)]
    clumperef = read_n_sort_clumped(ppt_results_ref, frqs.SNP)
    # clumperef = clumperef[clumperef.SNP.isin(frqs.SNP)]
    clumpe = [(clumpetar, tar), (clumperef, ref)]
    # Create crossvalidation
    cv = train_test(prefix, targetbed, plinkexe, pheno)
    train, test = cv.keys()
    phe_tr, phe_te = [x[1] for x in cv.values()]
    # Optimize the alphas
    df, qrange, qr, merged = optimize_alpha_plink(prefix, train, sorted_cotag, clumpe,
                                            scorefn, phe_tr, plinkexe,
                                            alpha_step, tar, prune_step,
                                            qrangefn, maxmem, threads, strategy,
                                            every, score_type=score_type)
    best_alpha = df.alpha.iloc[0]
    # Score with test-set
    res_pr = '%s_testset' % prefix
    best = single_alpha_qr(res_pr, best_alpha, merged, plinkexe, test, scorefn,
                           qrange, qr, tar, allele_file, maxmem=maxmem,
                           threads=threads, strategy=strategy,
                           score_type=score_type)
    # Get the best alpha of the optimization
    # grouped = df.groupby('alpha')
    # best = grouped.get_group(df.loc[0,'alpha'])
    if isinstance(sortresults, str):
        prevs = pd.read_table(sortresults, sep='\t')
    else:
        prevs = sortresults
    merged = best.merge(prevs, on='SNP kept')
    # Get the weighted squares
    wout = '%s_weighted' % prefix
    if 'BETA' not in sorted_cotag.columns:
        gwas = pd.read_table(sumstats, delim_whitespace=True)
        sortedcotag = sorted_cotag.merge(gwas, on='SNP')
    else:
        sortedcotag = sorted_cotag
    wqfile = weighted_squares(wout, sortedcotag)
    qdf = qrscore(plinkexe, targetbed, scorefn, qrange, wqfile, allele_file,
                  wout, qr, maxmem, threads, 'weighted', prefix)
    merged = merged.merge(qdf, on='SNP kept', suffixes=['_hybrid', '_weighted'])
    f, ax = plt.subplots()
    # plot cotagging
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_cotag', label=column,
                        c='r', s=2, alpha=0.5, ax=ax)
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_clum%s' % tar, ax=ax,
                        label='Clump Sort %s' % tar, c='k', s=2, alpha=0.5)
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_clum%s' % ref, ax=ax, s=2,
                        label='Clump Sort %s' % ref, c='0.5', marker='*',
                        alpha=0.5)
    merged.plot.scatter(x='SNP kept', y='R2_hybrid', label='Hybrid', c='g', s=2,
                        alpha=0.5, ax=ax)
    merged.plot.scatter(x='SNP kept', y='R2_weighted', label='Weighted', s=2,
                        c='purple', alpha=0.5, ax=ax)
    if h2 is not None:
        ax.axhline(h2, c='0.5', ls='--')
    plt.ylabel('$R^2$')
    plt.tight_layout()
    plt.savefig('%s_compare.pdf' % prefix)
    # Write resulted marged dataframe to file
    merged.to_csv('%s.tsv' % prefix, sep='\t', index=False)


# ---------------------------------------------------------------------------
def optimize_alpha(prefix, bfile, sorted_cotag, clumpe, sumstats, phenofile,
                   plinkexe, alphastep, tar, prune_step=1, qrangefn=None,
                   maxmem=1700, threads=1, strategy='sum', every=False,
                   score_type='SUM'):
    """
    Do a line search for the best alpha in nrank = alpha*rankP+T + (1-alpha)*cot

    :param str prefix: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param :class pd.Dataframe sorted_cotag:Filename with results in the sorting
    :param list clumpe: List of tuples with the P+T results
    :param str sumstats: File with the summary statistics in plink format
    :param str phenofile: Filename with the phenotype
    :param float alphastep: Step of the alpha to be explored
    :param str plinkexe: Path and executable of plink
    :param str tar: Label of target population
    :param int prune_step: Step of the prunning
    :param str qrangefn: Filename of a previously made qrange
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str strategy: Suffix of the function with the selection strategy
    :param bool every: test one snp at a time
    """
    # Set output name
    outfn = '%s_optimized.tsv' % prefix
    picklfn = '%s_optimized.pickle' % prefix
    # Execute the ranking
    if not os.path.isfile(picklfn):
        d, r, qr, merge = rank_qr(prefix, bfile, sorted_cotag, clumpe, sumstats,
                                  phenofile, alphastep, plinkexe, tar,
                                  prune_step,
                                  qrangefn, maxmem, threads, strategy, every,
                                  score_type)
        df = pd.concat(d)
        df.to_csv(outfn, sep='\t', index=False)
        with open(picklfn, 'wb') as F:
            pickle.dump((df, r, qr, merge), F)
    else:
        # df = pd.read_table(outfn, sep='\t')
        with open(picklfn, 'rb') as F:
            df, r, qr, merge = pickle.load(F)
    # Plot the optimization
    piv = df.loc[:, ['SNP kept', 'alpha', 'R2']]
    piv = piv.pivot(index='SNP kept', columns='alpha', values='R2').sort_index()
    piv.plot(colormap='copper', alpha=0.5)
    plt.ylabel('$R^2$')
    plt.tight_layout()
    plt.savefig('%s_alphas.pdf' % (prefix))
    # Returned the sorted result dataframe
    results = df.sort_values('R2', ascending=False).reset_index(drop=True)
    return results, r, qr, merge

# ----------------------------------------------------------------------
def prankcster(prefix, tbed, rbed, tpheno, labels, sumstats=None, cotag=None,
               freq_threshold=0.01, splits=3, threads=1, seed=None, **kwargs):
    seed = np.random.randint(1e4) if seed is None else seed
    print('Performing prankcster')
    # Unpack population labels
    refl, tarl = labels
    # check if phenotype is provided
    if tpheno is None:
        # make simulation for reference
        print('Simulating phenotype for reference population %s \n' % refl)
        opts = {'outprefix': refl, 'bfile': rbed, 'h2': kwargs['h2'],
                'ncausal': kwargs['ncausal'], 'normalize': kwargs['normalize'],
                'uniform': kwargs['uniform'], 'snps': None, 'seed': seed,
                'bfile2': tbed, 'f_thr': freq_threshold}
        rpheno, (rgeno, rbim, rtruebeta, rvec) = qtraits_simulation(**opts)
        # make simulation for target
        print('Simulating phenotype for target population %s \n' % tarl)
        opts.update(dict(outprefix=tarl, bfile=tbed, causaleff=rbim.dropna(),
                         bfile2=rbed))
        tpheno, (tgeno, tbim, ttruebeta, tvec) = qtraits_simulation(**opts)
        opts.update(dict(prefix='ranumo_gwas', pheno=rpheno, geno=rgeno,
                         validate=3, threads=threads, bim=rbim))
        sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    # Read summary statistics
    elif isinstance(sumstats, str):
        sumstats = pd.read_table(sumstats, delim_whitespace=True)
    else:
        assert isinstance(sumstats, pd.DataFrame)
    # Read the cotag scores
    if os.path.isfile('%s_cotags.tsv' % prefix):
        sorted_cotag = pd.read_table('%s_cotags.tsv' % prefix, sep='\t')
    elif isinstance(cotag, str):
        sorted_cotag = pd.read_table(cotag, sep='\t')
    elif cotag is None:
        cotags = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=kwargs['window'],
                        threads=threads)
        # D_r = da.dot(rgeno.T, rgeno) / rgeno.shape[0]
        # D_t = da.dot(tgeno.T, tgeno) / tgeno.shape[0]
        # cot = da.diag(da.dot(D_r, D_t))
        # ref = da.diag(da.dot(D_r, D_r))
        # tar = da.diag(da.dot(D_t, D_t))
        # stacked = da.stack([tbim.snp, ref, tar, cot], axis=1)
        # cotags = dd.from_dask_array(stacked, columns=['snp', 'ref', 'tar',
        #                                               'cotag']).compute(
        #     num_tasks=threads)
        cotags.to_csv('%s_cotags.tsv' % prefix, sep='\t', index=False)

    nsnps = sorted_cotag.shape[0]
    frac_snps = nsnps / 100
    # Read and sort the P + T results
    if os.path.isfile('%s.sorted_ppt' % tarl):
        clumpetar = pd.read_table('%s.sorted_ppt' % tarl, sep='\t')
    else:
        clumpetar = pplust('%s_ppt' % tarl, tgeno, tpheno, sumstats,
                           kwargs['r_range'], kwargs['p_tresh'], bim=tbim)[-1]
    if os.path.isfile('%s.sorted_ppt' % refl):
        clumperef = pd.read_table('%s.sorted_ppt' % refl, sep='\t')
    else:
        clumperef =  pplust('%s_ppt' % tarl, tgeno, tpheno, sumstats,
                           kwargs['r_range'], kwargs['p_tresh'], bim=tbim)[-1]
    # clumperef = clumperef[clumperef.SNP.isin(frqs.SNP)]
    clumpe = [(clumpetar, tarl), (clumperef, refl)]
    # Create crossvalidation
    x_train, x_test, y_train, y_test = train_test_split(tgeno, tpheno,
                                                        test_size=1 / splits,
                                                        random_state=seed)

    # Optimize the alphas
    df, qrange, qr, merged = optimize_alpha(prefix, train, sorted_cotag, clumpe,
                                            scorefn, phe_tr, plinkexe,
                                            alpha_step, tar, prune_step,
                                            qrangefn, maxmem, threads, strategy,
                                            every, score_type=score_type)
    best_alpha = df.alpha.iloc[0]
    # Score with test-set
    res_pr = '%s_testset' % prefix
    best = single_alpha_qr(res_pr, best_alpha, merged, plinkexe, test, scorefn,
                           qrange, qr, tar, allele_file, maxmem=maxmem,
                           threads=threads, strategy=strategy,
                           score_type=score_type)
    # Get the best alpha of the optimization
    # grouped = df.groupby('alpha')
    # best = grouped.get_group(df.loc[0,'alpha'])
    if isinstance(sortresults, str):
        prevs = pd.read_table(sortresults, sep='\t')
    else:
        prevs = sortresults
    merged = best.merge(prevs, on='SNP kept')
    # Get the weighted squares
    wout = '%s_weighted' % prefix
    if 'BETA' not in sorted_cotag.columns:
        gwas = pd.read_table(sumstats, delim_whitespace=True)
        sortedcotag = sorted_cotag.merge(gwas, on='SNP')
    else:
        sortedcotag = sorted_cotag
    wqfile = weighted_squares(wout, sortedcotag)
    qdf = qrscore(plinkexe, targetbed, scorefn, qrange, wqfile, allele_file,
                  wout, qr, maxmem, threads, 'weighted', prefix)
    merged = merged.merge(qdf, on='SNP kept', suffixes=['_hybrid', '_weighted'])
    f, ax = plt.subplots()
    # plot cotagging
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_cotag', label=column,
                        c='r', s=2, alpha=0.5, ax=ax)
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_clum%s' % tar, ax=ax,
                        label='Clump Sort %s' % tar, c='k', s=2, alpha=0.5)
    merged.plot.scatter(x='SNP kept', y=r'$R^{2}$_clum%s' % ref, ax=ax, s=2,
                        label='Clump Sort %s' % ref, c='0.5', marker='*',
                        alpha=0.5)
    merged.plot.scatter(x='SNP kept', y='R2_hybrid', label='Hybrid', c='g', s=2,
                        alpha=0.5, ax=ax)
    merged.plot.scatter(x='SNP kept', y='R2_weighted', label='Weighted', s=2,
                        c='purple', alpha=0.5, ax=ax)
    if h2 is not None:
        ax.axhline(h2, c='0.5', ls='--')
    plt.ylabel('$R^2$')
    plt.tight_layout()
    plt.savefig('%s_compare.pdf' % prefix)
    # Write resulted marged dataframe to file
    merged.to_csv('%s.tsv' % prefix, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-a', '--allele_file', default='EUR.allele',
                        help='File with the allele order. A1 in position 3 and '
                             'id in position2', required=True)
    parser.add_argument('-b', '--reference', required=True,
                        help=('prefix of the bed fileset in reference'))
    parser.add_argument('-c', '--target', required=True,
                        help=('prefix of the bed fileset in target'))
    parser.add_argument('-L', '--labels', nargs=2,
                        help=('Space separated string with labels of reference '
                              'and target populations'))
    parser.add_argument('-T', '--target_ppt', default=None,
                        help=('Filename of the results of a PPT run'))
    parser.add_argument('-r', '--ref_ppt', default=None,
                        help=('Filename with results for the P+Toptimization in'
                              ' the reference population'))
    parser.add_argument('-R', '--sortresults', required=True,
                        help=('Filename with results in the sorting inlcuding '
                              'path'))
    parser.add_argument('-d', '--cotagfn', required=True,
                        help=('Filename tsv with cotag results'))
    parser.add_argument('-s', '--sumstats', required=True,
                        help=('Filename of the summary statistics in plink '
                              'format'))
    parser.add_argument('-f', '--pheno', required=True,
                        help=('Filename of the true phenotype of the target '
                              'population'))
    parser.add_argument('-S', '--alpha_step', default=0.1, type=float,
                        help=('Step for the granularity of the grid search.'))
    parser.add_argument('-E', '--prune_step', default=1, type=float,
                        help=('Percentage of snps to be tested at each step'))
    parser.add_argument('-P', '--plinkexe')
    parser.add_argument('-v', '--splits', help='Number of folds for cross-val',
                        default=5, type=int)
    parser.add_argument('-C', '--column', help='Column to sort by',
                        default='Cotagging')
    parser.add_argument('-w', '--weight', default=False, action='store_true',
                        help=('Perform the sorting based on the weighted square'
                              ' effect sizes'))
    parser.add_argument('-y', '--every', action='store_true', default=False)
    parser.add_argument('-t', '--threads', default=1, action='store', type=int)
    parser.add_argument('-H', '--h2', default=0.66, type=float,
                        help=('Heritability of the simulated phenotype'))
    parser.add_argument('-M', '--maxmem', default=1700, type=int)
    parser.add_argument('-F', '--freq_threshold', default=0.1, type=float)
    parser.add_argument('-Q', '--qrangefn', default=None, help=(
        'Specific pre-made qrange file'))
    parser.add_argument('-g', '--strategy', default='sum', help=(
        'Strategy to produce the hybrid measure. Currently available is '
        'weighted sum (sum)'))
    parser.add_argument('--window', default=1000, help='kbwindow for ld',
                        type=int)
    args = parser.parse_args()
    prankcster(args.prefix, args.target, args.reference, args.cotagfn,
               args.target_ppt, args.ref_ppt, args.sumstats, args.pheno,
               args.plinkexe, args.alpha_step, args.labels, args.prune_step,
               args.sortresults, args.allele_file, h2=args.h2,
               freq_threshold=args.freq_threshold,
               qrangefn=args.qrangefn, maxmem=args.maxmem, threads=args.threads,
               strategy=args.strategy, every=args.every, column=args.column,
               splits=args.splits, weight=args.weight)
