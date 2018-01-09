#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<>
  Purpose: Analyze the tagging scores vs P+T and a Null random model
  Created: 10/02/17
"""

import matplotlib
import random

matplotlib.use('Agg')
from itertools import filterfalse
from ppt import *

plt.style.use('ggplot')
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
from numba import double
from numba.decorators import jit


# ---------------------------------------------------------------------------
@jit
def read_scored_qr(profilefn, phenofile, kind, nsnps, profiles):
    """
    Read the profile file a.k.a. PRS file or scoresum
    
    :param str profilefn: filename of scored (.profile) file
    :param str phenoflie: file name of file with phenotype
    :param str kind: label to match the scoring type (e.g cotag, clump, etc..)
    :param int nsps: number of snps that were used to score the profile fn
    """
    if not profilefn in profiles:
        return {}
    # Read the profile into a pandas dataframe
    sc = pd.read_table(profilefn, delim_whitespace=True)
    # Read the phenotype file into a pandas dataframe
    pheno = pd.read_table(phenofile, delim_whitespace=True, header=None, names=[
        'FID', 'IID', 'pheno'])
    # Merge the two data frames
    mer = sc.merge(pheno, on=['FID', 'IID'])
    # Compute the linear regression between the phenotype and the scored PRS
    lr = linregress(mer.pheno, mer.SCORESUM)
    # Store and return result in dictionary format
    dic = {'SNP kept': nsnps, '-log(P)_%s' % kind: -np.log10(lr.pvalue),
           r'$R^{2}$_%s' % kind: lr.rvalue ** 2, 'Slope_%s' % kind: lr.slope}
    return dic


# ---------------------------------------------------------------------------
def read_gwas_n_cotag(gwasfile, cotagfile):
    """
    Read the GWAS (a.k.a. summary stats) in the reference population and merge
    it with the cotagging info
    
    :param str gwasfile: Filename of the summary statistics file
    :param str cotagfile: Filename of the cotagging file
    """
    # Read cotagging file into a pandas dataframe
    cotag = pd.read_table(cotagfile, sep='\t')
    # Read the summary statistics into a pandas dataframe
    gwas = pd.read_table(gwasfile, delim_whitespace=True)
    gwas = gwas.sort_values('BP').reset_index()
    # Return the merged data frame
    return gwas.merge(cotag, on='SNP')


# ---------------------------------------------------------------------------
def subsetter_qrange(prefix, sortedcota, sortedtagT, sortedtagR, step,
                     phenofile, tarbed, allsnp, clumped=None, every=False):
    """
    Create the files to be used in q-score-range. It will use the index of the
    sorted files as thresholds
    
    :param int allsnp: Maximum number of snps or shared snps
    :param str prefix: Prefix for oututs
    :param :class pd.DataFrame sortedcota: Sorted cotag data frame
    :param :class pd.DataFrame sortedtagT: Sorted target tag data frame
    :param :class pd.DataFrame sortedtagR: Sorted referemce tag data frame
    :param float step: step for the snp range
    :param tuple clumped: list of tuples with clumped dataframe and kind, and 
    phenofile
    :param bool every: test one snp at a time
    """
    # Make sure that the shapes coincide
    assert sortedcota.shape[0] == sortedtagR.shape[0]
    # get the number of possible snps
    nsnps = sortedcota.shape[0]
    # Create a copy to randomize
    randomtagg = sortedcota.copy()
    idxs = randomtagg.Index.tolist()
    np.random.shuffle(idxs)
    # Store randomized snp rank into Index column
    randomtagg['Index'] = idxs
    # Fix prefixes
    prefix = prefix.replace('_', '')
    # Set qrange filename
    qrange = '%s.qrange' % prefix
    # Get a series with the percentages to be explore with emphasis in the first
    # 200
    percentages = set_first_step(nsnps, step, every=every)
    snps = np.around((percentages * allsnp) / 100).astype(int)
    try:
        # Check if there are repeats in ths set of SNPS
        assert sorted(snps) == sorted(set(snps))
    except AssertionError:
        snps = ((percentages * allsnp) / 100).astype(int)
        assert sorted(snps) == sorted(set(snps))

    labels = ['%.2f' % x for x in percentages]
    # Generate the qrange file?
    order = ['label', 'Min', 'Max']
    qr = pd.DataFrame({'label': labels, 'Min': np.zeros(len(percentages)),
                       'Max': snps}).loc[:, order]
    # np.around(np.array(percentages, dtype=float)*(
    #    nsnps/100)).astype(int)}).loc[:, order]
    qr.to_csv(qrange, header=False, index=False, sep=' ')
    # Set qfile filename
    qfile = '%s_%s.qfile'
    # Set the output tuples with the qfile,  phenotype file and matching bed 
    c = (qfile % (prefix, 'cotag'), phenofile, tarbed)
    t = (qfile % (prefix, 'tagt'), phenofile, tarbed)
    r = (qfile % (prefix, 'tagr'), phenofile, tarbed)
    a = (qfile % (prefix, 'rand'), phenofile, tarbed)
    # Write the sortings to file
    sortedcota.loc[:, ['SNP', 'Index']].to_csv(c[0], sep=' ', header=False,
                                               index=False)
    sortedtagT.loc[:, ['SNP', 'Index']].to_csv(t[0], sep=' ', header=False,
                                               index=False)
    sortedtagR.loc[:, ['SNP', 'Index']].to_csv(r[0], sep=' ', header=False,
                                               index=False)
    randomtagg.loc[:, ['SNP', 'Index']].to_csv(a[0], sep=' ', header=False,
                                               index=False)
    out = (qr, c, t, r, a)
    # Do the same as before but for clumps if any
    if clumped is not None:
        for clumped, kind, pf, bed in clumped:
            p = (qfile % (prefix, 'clum%s' % kind), pf, bed)
            out += (p,)
            clumped.loc[:, ['SNP', 'Index']].to_csv(p[0], sep=' ', header=False,
                                                    index=False)
    # return the tuple of tuples, with the first element being the qr data frame
    return out, qrange


# ---------------------------------------------------------------------------
def cleanup():
    """
    Clean up the folder: remove nosex lists, put all logs under the LOGs folder
    and all the extract files under SNPs folder
    """
    if not os.path.isdir('LOGs'):
        os.mkdir('LOGs')
    else:
        shutil.rmtree('LOGs')
        os.mkdir('LOGs')
    if not os.path.isdir('SNP_lists'):
        os.mkdir('SNP_lists')
    else:
        shutil.rmtree('SNP_lists')
        os.mkdir('SNP_lists')
    for log in glob('*.log'):
        shutil.move(log, 'LOGs')
    for ns in glob('*.nosex'):
        os.remove(ns)
    for snp in glob('*.extract'):
        shutil.move(snp, 'SNP_lists')
    for nopred in glob('*.nopred'):
        os.remove(nopred)


# ----------------------------------------------------------------------
def score_qfiles(out, prefix, plinkexe, gwasfn, frac_snps, maxmem=1700,
                 threads=8):
    """
    Score the set of qfiles defined in out
    
    :param tuple outs: Tuple with the qfiles and matching partners to analyze
    :param str prefix: Prefix of outputs
    :param str plinkexe: Path and executable of plink
    :param str gwasfn: Filename of the summary statistics file
    :param float frac_snps: Number of snps per one percentage
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    """
    # Get the qrange dataframe from out tuple
    qr = out[0]
    # Set qrange name 
    qrange = '%s.qrange' % prefix
    # Score files with plink's --q-score-range option and read them into a 
    # pandas data frame ... and do it in parallel
    frames = Parallel(n_jobs=int(threads))(delayed(single_score)(
        prefix, qr, tup, plinkexe, gwasfn, qrange, frac_snps, maxmem, threads)
                                           for tup in tqdm(out[1:],
                                                           total=len(out[1:])))
    return filterfalse(lambda df: df.empty, frames)


# ---------------------------------------------------------------------------
def prunebypercentage_qr(prefix, bfile, gwasfn, phenofile, sortedcotag, allsnp,
                         sortedtagT, sortedtagR, plinkexe, clumped=None, step=1,
                         tar_label='AFR', ref_label='EUR', maxmem=1700,
                         threads=8, every=False):
    """
    Execute the prunning in a range from <step> to 100 with step <step> (in %)
    scoring using --q-score-ragne
    
    :param str prefix: Prefix of outputs
    :param str bfile: Prefix of plink-bed fileset
    :param str gwasfn: Filename of the summary stats to get the betas from
    :param str phenofile: File name of file with phenotype
    :param :class pd.DataFrame sortedcotag: Dataframe with cotag sorting
    :param int allsnp: Maximum number of snps or shared snps
    :param :class pd.DataFrame sortedtagT: Dataframe with tagT sorting
    :param :class pd.DataFrame sortedtagR: Dataframe with tagR sorting
    :param str plinkexe: Path and executable of plink
    :param list clumped: list with clump tuples (dataframe,label,phenofile,bed)
    :param float step: Step for prunning in percentage
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param bool every: test one snp at a time
    """
    # Get the fraction of snps per percentage
    frac_snps = sortedcotag.shape[0] / 100
    if os.path.isfile('pbp.pickle'):
        with open('pbp.pickle', 'rb') as f:
            merge = pickle.load(f)
            qrangefn = '%s.qrange' % prefix
    else:
        print('Performing prunning ...')
        # Execute the qrange scoring and read it into a dataframe
        out, qrangefn = subsetter_qrange(prefix, sortedcotag, sortedtagT,
                                         sortedtagR, step, phenofile, bfile,
                                         allsnp, clumped=clumped, every=every)
        frames = score_qfiles(out, prefix, plinkexe, gwasfn, frac_snps, maxmem,
                              threads)
        # Merge all the frames by the number of snps kept
        merge = reduce(lambda x, y: pd.merge(x, y, on='SNP kept'), frames)
        merge['Percentage of SNPs used'] = (
                                               merge.loc[:,
                                               'SNP kept'] / merge.loc[
                                                             :,
                                                             'SNP kept'].max()) * 100
        # write results to file
        merge.to_csv('%s_merged.tsv' % prefix, sep='\t', index=False)
        # Cleanup
        cleanup()
        with open('pbp.pickle', 'wb') as f:
            # Store results for relaunch
            pickle.dump(merge, f)
    # return the merged data frame
    return merge, qrangefn


# ---------------------------------------------------------------------------
def plotit(prefix, merge, col, labels, ppt=None, line=False, vline=None,
           hline=None, plottype='png', x='SNP kept'):
    """
    Plot the R2 vs the proportion of SNPs included for random pick and cotagging
    
    :param str prefix: prefix of outputs
    :param :class pd.DataFrame merge: Data Frame with the merge data of prunning
    :param str col: Name of Column to ploy against x
    :param list labels: labels of the populations  (reference, target) 
    :param list ppt: List with the clumped P + T results' tuples
    :param bool line: Make it a line plot instead of scatter
    :param float vline: Whether (and where) to put a vertical line
    :param float hline: Whether (and where) to put an horizontal line
    :param str plottype: Format to save the plot in (follows matplotlib formats)
    :param str x: Name of column to be use in x axis
    """
    # ensure numeric
    for c in merge.columns:
        merge.loc[:, c] = pd.to_numeric(merge.loc[:, c])
    # Unpack the labels
    ref, tar = labels
    if line:
        # If line plot, format dataframes accordingly an plot them
        rand = merge.pivot(x).loc[:, col + '_rand']
        cota = merge.pivot(x).loc[:, col + '_cotag']
        f, ax = plt.subplots()
        rand.plot(x=x, y=col + '_rand', label='Random', c='b', s=2, alpha=0.5,
                  ax=ax)
        cota.plot(x=x, y=col + '_cotag', label='Cotagging', ax=ax, c='r', s=2,
                  alpha=0.5)
    else:
        # If scatter, plot each type
        f, ax = plt.subplots()
        merge.plot.scatter(x=x, y=col + '_rand', label='Random', c='b', s=2,
                           alpha=0.5, ax=ax)
        merge.plot.scatter(x=x, y=col + '_cotag', label='Cotagging', ax=ax,
                           c='r',
                           s=2, alpha=0.5)
        merge.plot.scatter(x=x, y=col + '_tagt', label='Tagging %s' % tar,
                           ax=ax,
                           c='c', s=2, alpha=0.5)
        merge.plot.scatter(x=x, y=col + '_tagr', label='Tagging %s' % ref,
                           ax=ax,
                           c='m', s=2, alpha=0.5)
        # if isinstance(ppt, str):
        merge.plot.scatter(x=x, y=col + '_clum%s' % ref, ax=ax, c='0.5', s=2,
                           alpha=0.5, label='Sorted Clump %s' % ref, marker='*')
        merge.plot.scatter(x=x, y=col + '_clum%s' % tar, ax=ax, c='k', s=2,
                           alpha=0.5, label='Sorted Clump %s' % tar)
    if vline is not None:
        ax.axvline(float(vline), c='0.5', ls='--')
    if hline is not None:
        ax.axhline(float(hline), c='0.5', ls='--')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig('%s_error.%s' % (prefix, plottype))
    plt.close()


# ----------------------------------------------------------------------
def ranumo_plink(prefix, tarbed, refbed, gwasfn, cotagfn, plinkexe, labels,
                 phenotar,
                 phenoref, pptR=None, pptT=None, check_freqs=0.1, hline=None,
                 step=1, quality='png', maxmem=1700, threads=8, every=False):
    """
    execute the code  
    
    :param str prefix: prefix of outputs
    :param str tarbed: Prefix of plink-bed fileset for target population
    :param str refbed: Prefix of plink-bed fileset for reference population
    :param str gwasfn: Filename of the summary stats to get the betas from
    :param str cotagfn: Filename of the cotagging file
    :param str plinkexe: Path and executable of plink
    :param list labels: labels of the populations  (reference, target) 
    :param str phenotar: File name of file with phenotype of the target
    :param str phenoref: File name of file with phenotype of the reference
    :param str pptR: Filename or data frame with the P+T result in reference
    :param str pptT: Filename or data frame with the P+T result in Target
    :param float check_freqs: Frequency to filter MAF 
    :param float hline: Where to put an horizontal line in the resulting plot
    :param float step: Step for prunning in percentage
    :param str quality: Format to save the plot in (follows matplotlib formats)
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param bool every: test one snp at a time
    """
    print('Performing ranumo')
    # make sure the prefix does not have any _
    prefix = prefix.replace('_', '')
    # Unpack labels
    ref, tar = labels
    # Read summary statistics file
    gwas = pd.read_table(gwasfn, delim_whitespace=True)
    # read cotagfile (if needed) and mer it with the summary stats
    if isinstance(cotagfn, str):
        cotags = pd.read_table(cotagfn, sep='\t')
    gwas = gwas.merge(cotags, on='SNP')
    # Filter the merged file by MAF if required
    frqT = read_freq(tarbed, plinkexe, freq_threshold=check_freqs)
    frqR = read_freq(refbed, plinkexe, freq_threshold=check_freqs)
    frq = frqT.merge(frqR, on=['CHR', 'SNP'], suffixes=['_%s' % ref,
                                                        '_%s' % tar])
    gwas = gwas[gwas.SNP.isin(frq.SNP)]
    # normalize betas and write a scoring file
    maf = 'MAF_%s' % tar
    a1 = 'A1_%s' % tar
    mer = frq.loc[:, ['SNP', a1, maf]].merge(gwas, on='SNP')
    mer['norm'] = np.sqrt((2 * mer.loc[:, maf]) * (1 - mer.loc[:, maf]))
    mer['BETA_norm'] = mer.BETA / mer.norm
    scorefn = '%s.score' % prefix
    mer.loc[:, ['SNP', a1, 'BETA_norm']].to_csv(scorefn, sep=' ', index=False,
                                                header=False)
    allsnp = gwas.shape[0]
    # Cotagging
    sortedcot, beforetail = smartcotagsort(prefix, gwas,
                                           threads=threads)  # cotags)
    # Tagging Target
    sortedtagT, beforetailTT = smartcotagsort(prefix, gwas,
                                              column='Tagging %s' % tar,
                                              threads=threads)
    # Tagging Reference
    sortedtagR, beforetailTR = smartcotagsort(prefix, gwas,  # cotags,
                                              column='Tagging %s' % ref,
                                              threads=threads)
    # Process clump if required
    clump = []
    # Include the P + T of the reference population if required
    if pptR is not None:
        if isinstance(pptR, str):
            resR = pd.read_table(pptR, sep='\t')
        else:
            assert isinstance(pptR, pd.DataFrame)
            resR = pptR
        best_clumpR = resR.nlargest(1, 'R2').File.iloc[0]
        pptR = os.path.join(os.path.split(phenoref)[0],
                            '%s.clumped' % best_clumpR)
        clumR = parse_sort_clump(pptR, gwas.SNP)
        clump.append((clumR, ref, phenoref, refbed))
    # Include the P + T of the target population if required
    if pptT is not None:
        if isinstance(pptT, str):
            resT = pd.read_table(pptT, sep='\t')
        else:
            assert isinstance(pptT, pd.DataFrame)
            resT = pptT
        best_clumpT = resT.nlargest(1, 'R2').File.iloc[0]
        pptT = os.path.join(os.path.split(phenotar)[0],
                            '%s.clumped' % best_clumpT)
        clumT = parse_sort_clump(pptT, gwas.SNP)
        clump.append((clumT, tar, phenotar, tarbed))

    # Perform the prunning and scoring
    merge, qrangefn = prunebypercentage_qr(prefix, tarbed, scorefn, phenotar,
                                           sortedcot, allsnp, sortedtagT,
                                           sortedtagR, plinkexe, clumped=clump,
                                           step=step, tar_label=tar,
                                           ref_label=ref, maxmem=maxmem,
                                           threads=threads, every=every)
    # Plot reults
    plotit(prefix + '_rval', merge, r'$R^{2}$', labels, ppt=clump,
           plottype=quality, hline=hline)
    # Return the merged data frame
    return merge, qrangefn


# ----------------------------------------------------------------------
def read_clump(fn):
    clump = {}
    with open(fn) as F:
        for line in F:
            bl = line.split()
            clump[bl[0]] = bl[1].split(';')
    return clump


# ----------------------------------------------------------------------
def ranumo(prefix, refgeno, refpheno, sumstats, targeno, tarpheno, cotagfn,
           labels, prunestep=10, seed=None, threads=1, ppts=None, **kwargs):
    seed = np.random.randint(1e4) if seed is None else seed
    now = time.time()
    print('Performing random null model (RANUMO)!')
    refl, tarl = labels
    # If pheno is None for the reference, make simulation
    if isinstance(refpheno, str):
        rpheno = dd.read_table(refpheno, blocksize=25e6, delim_whitespace=True)
        tpheno = dd.read_table(tarpheno, blocksize=25e6, delim_whitespace=True)
    elif refpheno is None:
        # make simulation for reference
        print('Simulating phenotype for reference population %s \n' % refl)
        opts = {'outprefix': refl, 'bfile': refgeno, 'h2': kwargs['h2'],
                'ncausal': kwargs['ncausal'], 'normalize': kwargs['normalize'],
                'uniform': kwargs['uniform'], 'snps': None, 'seed': seed,
                'bfile2': targeno}
        rpheno, h2, (rgeno, rbim, rtruebeta, rvec) = qtraits_simulation(**opts)
        # make simulation for target
        print('Simulating phenotype for target population %s \n' % tarl)
        opts.update(dict(outprefix=tarl, bfile=targeno, causaleff=rbim.dropna(),
                         bfile2=refgeno, validate=kwargs['split']))
        tpheno, h2, (tgeno, tbim, ttruebeta, tvec) = qtraits_simulation(**opts)
        opts.update(dict(prefix='ranumo_gwas', pheno=rpheno, geno=rgeno,
                         validate=kwargs['split'], threads=threads, bim=rbim))
        sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opts)
    elif isinstance(refgeno, str):
        (rbim, rfam, rgeno) = read_plink(refgeno)
        rgeno = rgeno.T
        sumstats = pd.read_table(sumstats, delim_whitespace=True)
        (tbim, tfam, tgeno) = read_plink(targeno)
        tgeno = tgeno.T
    # subset the geno files to shared among the reference and target
    mbim = rbim.merge(tbim, on='snp')
    ridx = rbim.snp[rbim.snp.isin(mbim.snp)].index.tolist()
    tidx = tbim.snp[tbim.snp.isin(mbim.snp)].index.tolist()
    rgeno = rgeno[:, ridx]
    tgeno = tgeno[:, tidx]
    # read cotagfile (if needed) and mer it with the summary stats
    if isinstance(cotagfn, str):
        cotags = pd.read_table(cotagfn, sep='\t')
    elif isinstance(cotagfn, pd.core.frame.DataFrame):
        cotags = cotagfn
    else:
        cotags = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=kwargs['window'],
                        threads=threads)
        cotags.to_csv('%s_cotags.tsv' % prefix, sep='\t', index=False)
    gwas = cotags.merge(sumstats, on='snp')
    # Sort the sumstats based on scores
    results = []
    # Cotagging
    sortedcot, beforetail = smartcotagsort(prefix, gwas, column='cotag',
                                           ascending=False)
    assert sortedcot[sortedcot.snp.isin(tbim)].i.equals(tbim[tbim.snp.isin(
        sortedcot)].i)
    sortedcot = sortedcot.merge(tbim.reindex(columns=['snp', 'i']), on=['snp',
                                                                        'i'])

    # Tagging Target
    params = dict(column='tar', ascending=False)
    sortedtagT, beforetailTT = smartcotagsort(prefix, gwas, **params)
    # sortedtagT['gen_index'] = [tbim[tbim.snp == i].i.values[0] for i in
    #                            sortedtagT.snp]
    assert sortedtagT[sortedtagT.snp.isin(tbim)].i.equals(
        tbim[tbim.snp.isin(sortedtagT)].i)
    sortedtagT = sortedtagT.merge(tbim.reindex(columns=['snp', 'i']), on=['snp',
                                                                          'i'])
    # Tagging Reference
    params.update(dict(column='ref'))
    sortedtagR, beforetailR = smartcotagsort(prefix, gwas, **params)
    assert sortedtagR[sortedtagR.snp.isin(tbim)].i.equals(
        tbim[tbim.snp.isin(sortedtagR)].i)
    sortedtagR = sortedtagR.merge(rbim.reindex(columns=['snp', 'i']), on=['snp',
                                                                          'i'])
    # Random model
    rand = gwas.sample(frac=1).reset_index(drop=True)
    rand['rand_index'] = rand.index.tolist()
    random, beforetailrand = smartcotagsort(prefix, rand, column='rand_index',
                                           ascending=False)
    random = random.merge(tbim.reindex(columns=['snp', 'i']), on=['snp', 'i'])
    # make sure that all have the same total SNPS
    assert sorted(sortedtagT.snp) == sorted(sortedcot.snp)
    assert sorted(sortedtagT.snp) == sorted(sortedtagR.snp)
    assert sorted(sortedtagT.snp) == sorted(random.snp)

    if isinstance(ppts, tuple):
        # read ppt from files
        ppt_r = pd.read_table(ppts[0], delim_whitespace=True)
        ppt_t = pd.read_table(ppts[1], delim_whitespace=True)
    else:
        # perform p+t in reference
        ppt_r = pplust('%s_ppt' % refl, X_test, y_test, sumstats,
                       kwargs['r_range'], kwargs['p_tresh'], bim=rbim,
                       split=kwargs['split'])[-1]
        params.update(dict(column='index', ascending=True))
        ppt_r, _ = smartcotagsort('%s_%s' % (prefix, refl), ppt_r, **params)
        # # get the full lenght with apropriate inidces
        # tagged_r = [y for x in ppt_r for y in x if y]
        # tagged_r = sumstats[sumstats.snp.isin(tagged_r)].reindex(
        #     columns=['snp', 'p_value', 'slope']).sample(frac=1)
        # ppt_r = ppt_r.reindex(colums=['snp', 'pvalue', 'slope']).sort_values(
        #     'p_value')
        # ppt_r = pd.concat((ppt_r, tagged_r), ignore_index=True)
        # ppt_r['gen_index'] = [rbim[rbim.snp == i].i[0] for i in ppt_r.snp]
        # perform p+t in target
        ppt_t = pplust('%s_ppt' % tarl, tgeno, tpheno, sumstats,
                       kwargs['r_range'], kwargs['p_tresh'], bim=tbim,
                       split=kwargs['split'])[-1]
        ppt_t, _ = smartcotagsort('%s_%s' % (prefix, tarl), ppt_t, **params)
        # # get the full lenght with apropriate inidces
        # tagged_t = [y for x in ppt_r for y in x if y]
        # tagged_t = sumstats[sumstats.snp.isin(tagged_t)].reindex(
        #     columns=['snp', 'p_value', 'slope']).sample(frac=1)
        #     columns=['snp', 'p_value', 'slope']).sample(rac=1)
        # ppt_t = ppt_t.reindex(colums=['snp', 'pvalue', 'slope']).sort_values(
        #     'p_value')
        # ppt_t = pd.concat((ppt_t, tagged_t), ignore_index=True)
        # ppt_r['gen_index'] = [tbim[tbim.snp == i].i[0] for i in ppt_t.snp]

    # prune and process all
    # prune cotagging
    results.append(prune_it(sortedcot, tgeno, tpheno, 'Cotagging',
                            step=prunestep))
    # Prune tag tar
    results.append(prune_it(sortedtagT, tgeno, tpheno, 'Tagging %s' % tarl,
                            step=prunestep))
    # prune tag ref
    results.append(prune_it(sortedtagR, X_test, y_test, 'Tagging %s' % refl,
                            step=prunestep, threads=threads))
    results.append(
        prune_it(sortedtagR, tgeno, tpheno, 'Tagging %s on tar' % refl,
                 step=prunestep, threads=threads))
    # prune and score the random model
    results.append(prune_it(random, tgeno, tpheno, 'Random', step=prunestep,
                            threads=threads))
    # prune and process P+Ts
    results.append(prune_it(ppt_t, tgeno, tpheno, 'P+T %s' % tarl,
                            step=prunestep, threads=threads))
    results.append(prune_it(ppt_r, X_test, y_test, 'P+T %s' % refl,
                            step=prunestep, threads=threads))
    results = pd.concat(results)
    results.to_csv('%s.ranumo.tsv' % prefix, sep='\t', index=False)
    # plot
    colors = iter(['r', 'b', 'm', 'g', 'c', 'k','y'])
    f, ax = plt.subplots()
    for t, df in results.groupby('type'):
        df.plot(x='Number of SNPs', y='R2', kind='scatter', legend=True, s=3,
                c=next(colors), ax=ax, label=t)
    plt.tight_layout()
    plt.savefig('%s_ranumo.pdf' % prefix)
    plt.close()
    print('Ranumo done after %.2f minutes' % ((time.time() - now) / 60.))
    return results


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
    parser.add_argument('-g', '--gwas', help='Filename of gwas results in the' +
                                             ' reference population',
                        default=None)
    parser.add_argument('-b', '--tarbed', help=('prefix of the bed fileset for'
                                                ' the target population'),
                        required=True)
    parser.add_argument('-R', '--refbed', help=('prefix of the bed fileset for'
                                                ' the reference population'),
                        required=True)
    parser.add_argument('-f', '--phenotar', help=('filename of the true '
                                                  'phenotype of the target '
                                                  'population'), default=None)
    parser.add_argument('-i', '--phenoref', help=('filename of the true '
                                                  'phenotype of the reference '
                                                  'population'), default=None)
    parser.add_argument('-P', '--plinkexe', default='~/Programs/plink_mac/plink'
                        , help='DEPRECATED, does nothing now')
    parser.add_argument('-c', '--cotagfile', help=('Filename of the cotag tsv'
                                                   ' file'), default=None)
    parser.add_argument('-s', '--step', help=('Step in the percentage range to '
                                              'explore. By deafult is 1'),
                        default=1, type=float)
    parser.add_argument('-l', '--labels', help=('Space separated string with '
                                                'reference and target lables '
                                                '(in that order)'), nargs=2)
    parser.add_argument('-r', '--ppt_ref', help=('Path to P+T results in the '
                                                 'reference population'),
                        default=None)
    parser.add_argument('-t', '--ppt_tar', help=('Path to P+T results in the '
                                                 'reference population'),
                        default=None)
    parser.add_argument('-H', '--h2', help=('Heritability of the simulated '
                                            'phenotype'), default=False,
                        type=float)
    parser.add_argument('-F', '--check_freq', help=('Read a frequency file and '
                                                    'filter by this threshold'),
                        default=0.1, type=float)
    parser.add_argument('-q', '--quality', help=('type of plots (png, pdf)'),
                        default='png')
    parser.add_argument('-y', '--every', action='store_true', default=False)
    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=3000, type=int)
    parser.add_argument('--ncausal', default=200, type=int)
    parser.add_argument('--normalize', default=True, action='store_false')
    parser.add_argument('--uniform', default=True, action='store_false')
    parser.add_argument('--r_range', default=None, nargs=3,
                        action=Store_as_arange, type=float)
    parser.add_argument('--p_tresh', default=None, nargs='+',
                        action=Store_as_array, type=float)
    parser.add_argument('--split', default=None, type=int)
    parser.add_argument('--window', default=1000, type=int)

    args = parser.parse_args()
    ranumo(args.prefix, args.refbed, args.phenoref,  args.gwas, args.tarbed,
           args.phenotar, args.cotagfile, args.labels, pptR=args.ppt_ref,
           pptT=args.ppt_tar, check_freqs=args.check_freq, hline=args.h2,
           step=args.step, quality=args.quality, every=args.every,
           threads=args.threads, maxmem=args.maxmem, h2=args.h2,
           ncausal=args.ncausal, normalize=args.normalize, uniform=args.uniform,
           r_range=args.r_range, p_tresh=args.p_tresh, split=args.split,
           window=args.window)
