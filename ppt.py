#!/usr/bin/env python
# coding:utf-8
"""
  Author:   Jose Sergio Hleap --<>
  Purpose: From SumStats get the best combination of R2 and P-thresholding P + T
  Created: 10/01/17
"""
import shutil
from itertools import product
from operator import itemgetter

import dask.dataframe as dd
import matplotlib

matplotlib.use('Agg')
from qtraitsimulation import *
from plinkGWAS import *

plt.style.use('ggplot')


# ---------------------------------------------------------------------------
def clump_vars_plink(outpref, bfile, sumstats, r2, window, phenofn, al_file,
                     plinkexe,
                     maxmem, threads, clump_field='P'):
    """
    Use plink to clump variants based on a pval and r2 threshold 
    
    :param al_file: File with the allele order
    :param str outpref: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param str sumstats: File with the summary statistics in plink format 
    :param float r2: LD threshold for clumping       
    :param int window: Size of the clumping window
    :param str phenofn: File with the phenotype in plink format
    :param str plinkexe: Path and executable of plink
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    """
    # Define output name
    outfn = '%s_%.2f_%d' % (outpref, r2, window)
    # Prepare plink CLA
    plink = ('%s --bfile %s -clump %s --clump-field %s --clump-p1 1 --clump-p2'
             ' 1 --clump-r2 %f --clump-kb %d --out %s --allow-no-sex '
             '--a1-allele %s 3 2 --pheno %s --memory %d --threads %d')
    # If output is not there yet, execute the command
    if not os.path.isfile('%s.clumped' % (outfn)):
        plink = plink % (plinkexe, bfile, sumstats, clump_field, r2,
                         window, outfn, al_file, phenofn, maxmem, threads)
        o, e = executeLine(plink)
    # =# read Clump file
    fn = '%s.clumped' % (outfn)
    try:
        table = pd.read_table(fn, delim_whitespace=True)
    except FileNotFoundError as err:
        # check if the error is because of lack of significant clumps
        if 'No significant --clump results' in open('%s.log' % outfn).read():
            table = None
        else:
            raise FileNotFoundError(err)
    # returns the filename of the output and its dataframe
    return outfn, table


# ----------------------------------------------------------------------
def range_profiles(name, range_tuple, r2, qfiledf, phenofn, score_type='sum'):
    """
    Read single profile from the q-range option
    
    :param str name: Output name of clumping
    :param tuple range_tuple: Namedtuple with row info from the qrange dataframe
    :param float r2: LD threshold for clumping
    :param :class pd.DataFrame qfiledf: Data frame with the qfile information
    :param str phenofn: File with the phenotype in plink format
    """
    if score_type == 'sum':
        col = 'SCORESUM'
    else:
        col = 'SCORE'
    # Read phenotype into a pandas dataframe
    pheno = read_pheno(phenofn)
    # Get the lable from the named tuple
    range_label = range_tuple.name
    # Make sure that that everything matches as it supposed to
    assert np.around(float(range_tuple.name), 5) == np.around(range_tuple.Max,
                                                              5)
    # Get the number of SNPs
    nsps = qfiledf[qfiledf.P <= range_tuple.Max].shape[0]
    # Set the input file name
    profilefn = '%s.%s.profile' % (name, range_label)
    # Read the profile
    score = pd.read_table(profilefn, delim_whitespace=True)
    # Merge score and pheno by individual (IID) and family (FID) IDs 
    score = score.loc[:, ['FID', 'IID', col]].merge(pheno, on=['FID', 'IID'])
    # Rename SCORE to PRS
    score.rename(columns={col: 'PRS'}, inplace=True)
    # check if peno is binary:
    if set(score.Pheno) <= set([0, 1]):
        score['pheno'] = score.Pheno - 1
        y, X = dmatrices('pheno ~ PRS', score, return_type='dataframe'
                         )
        logit = sm.Logit(y, X)
        logit = logit.fit(disp=0)
        # get the pseudo r2 (McFadden's pseudo-R-squared.)
        pR2 = logit._results.prsquared
    else:
        # Linear/quantitative trait
        slope, intercept, pR2, p_value, std_err = stats.linregress(
            score.Pheno, score.PRS)
        score['pheno'] = score.Pheno
        pR2 = pR2 ** 2
    # return a dictionary with the filename, r2, Pval-threshold used and nsnps
    return {'File': '%s.%s' % (name, range_label), 'LDthresh': r2,
            'Pthresh': range_label, 'R2': pR2, 'pval': p_value,
            'SNP kept': nsps}


# ----------------------------------------------------------------------
def qfile_gen(outpref, clumped, r2, pvals_th, clump_field='P'):
    """
    Generate the qfile for --q-score-range
    
    :param str outpref: Prefix for outputs
    :param :class pd.DataFrame clumped: Dataframe with the clumpfile results 
    :param float r2: LD threshold for clumping
    :param list pvals_th: List with the pvalue thresholds to be tested
    :param str clump_field: field to based the clumping on
    """
    # Set the input/output names
    qrange = '%s_%.2f.qrange' % (outpref, r2)
    qfile = '%s%.2f.qfile' % (outpref, r2)
    # Set the values for the qrange file
    qr = pd.DataFrame({'name': [str(x) for x in pvals_th], 'Min': np.zeros(len(
        pvals_th)), 'Max': pvals_th})
    order = ['name', 'Min', 'Max']
    # Write q-range to file
    qr.loc[:, order].to_csv(qrange, header=False, index=False, sep=' ')
    # Set and write qfile based on clumped values
    qf = clumped.loc[:, ['SNP', clump_field]]
    qf.to_csv(qfile, header=False, index=False, sep=' ')
    # return the output filenames and the corresponding dataframes
    return qrange, qfile, qr, qf


# ---------------------------------------------------------------------------
def ScoreClumped_plink(outpref, bfile, clumped, phenofn, sumstatsdf, r2,
                       pvals_th,
                       plinkexe, allele_file, clump_field='P', maxmem=1700,
                       threads=8,
                       score_type='sum'):
    """
    Compute the PRS for the clumped variants
    :param allele_file: A1 allele file
    :param str outpref: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param tuple clumped: tuple with name and Pandas Data Frame with clump data
    :param :class pd.DataFrame sumstatsdf: Summary statistics Pandas Data Frame
    :param float r2: LD threshold for clumping
    :param list pvals_th: List with the pvalue thresholds to be tested
    :param str plinkexe: Path and executable of plink
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str clump_field: field to based the clumping on
    """
    # Create empty dataframe with predifined columns to store reults
    results = pd.DataFrame(columns=['File', 'LDthresh', 'Pthresh', 'R2'])
    # Expand the clumped tuple 
    name, clumped = clumped
    if not os.path.isfile('%s.pickle' % (name)):
        if clumped is None:
            return
            # Merge summary statistics and the clumped df
        merge = sumstatsdf[sumstatsdf.SNP.isin(clumped.loc[:, 'SNP'])]
        if not 'OR' in merge.columns:
            cols = ['SNP', 'A1', 'BETA']
        else:
            cols = ['SNP', 'A1', 'OR']
            merge['OR'] = np.log(merge.OR)
        merge = merge.loc[:, cols]
        # Write file for scoring
        merge.to_csv('%s.score' % (name), sep=' ', header=False,
                     index=False)
        # Score using plink
        if score_type != 'sum':
            score_type = ''
        qrange, qfile, qr, qf = qfile_gen(outpref, clumped, r2, pvals_th)
        score = ('%s --bfile %s --score %s.score %s --q-score-range %s %s '
                 '--allow-no-sex --a1-allele %s 3 2 --out %s --pheno %s '
                 '--memory %d --threads %d')
        score = score % (plinkexe, bfile, name, score_type, qrange, qfile,
                         allele_file, name, phenofn, maxmem, threads)
        o, e = executeLine(score)
        # read range results
        profs_written = read_log(name)
        l = [range_profiles(name, range_label, r2, qf, phenofn, score_type)
             for range_label in qr.itertuples() if '%s.%s.profile' % (
                 name, range_label.name) in profs_written]
        with open('%s.pickle' % name, 'wb') as f:
            pickle.dump(l, f)
    else:
        with open('%s.pickle' % name, 'rb') as f:
            l = pickle.load(f)
    # Store results
    results = results.append(l).reset_index(drop=True)
    top = glob('%s.profile' % results.nlargest(1, 'R2').File.iloc[0])
    # Cleanup leaving the top
    with tarfile.open('Profiles_%.2f.tar.gz' % r2, mode='w:gz') as t:
        for fn in glob('*.profile'):
            if fn not in top:
                t.add(fn)
                os.remove(fn)
    return results


# ----------------------------------------------------------------------
def cleanup(results, clean):
    """
    Organize the CWD
    
    :param :class pd.DataFrame results: Pandas dataframe with p+t results
    :param bool clean: Whether to clean the folder or not
    """
    # Get top 10 results to avoid erasing them
    top10 = results.nlargest(10, 'R2').reset_index(drop=True)
    top = top10.nlargest(1, 'R2')
    for i in glob('%s.*' % top.File.iloc[0]):
        shutil.copy(i, '%s.BEST' % (i))
    if clean:
        print('Cleaning up ...')
        files = results.File
        tocle = files[~files.isin(top10.File)]
        tocle = [x for y in tocle for x in glob('%s*' % y)]
        profiles = glob('*.profile')
        for fi in tqdm(tocle, total=len(tocle)):
            if os.path.isfile(fi):
                os.remove(fi)
                # if not os.path.isdir('LOGs'):
                #    os.mkdir('LOGs')
                # for f in glob('*.log'):
                #    shutil.move(f, 'LOGs')


# ----------------------------------------------------------------------
def plotppt(outpref, results):
    """
    Plot results of P + T
    
    :param str outpref: Prefix for outputs
    :param :class pd.DataFrame results: Pandas dataframe with p+t results
    """
    # Fix the grid
    matplotlib.rcParams.update({'figure.autolayout': True})
    # Get the -log of pvalues in the result file
    results.loc[:, 'pval'] = -np.log(results.pval)
    # reorganize the data frame to efficiently plot the alpha exploration
    piv = results.pivot_table(index='SNP kept', values=['pval', 'R2'])
    f, ax = plt.subplots()
    ax2 = ax.twinx()
    # plot results for comparisons
    piv.loc[:, 'R2'].plot(ax=ax, color='b', alpha=0.5)
    ax.set_ylabel(r'$R^2$', color='b')
    ax.tick_params('y', colors='b')
    piv.loc[:, 'pval'].plot(ax=ax2, color='r', alpha=0.5)
    ax2.set_ylabel('-log(P-Value)', color='r')
    ax2.tick_params('y', colors='r')
    plt.savefig('%s_PpT.pdf' % outpref)


# ----------------------------------------------------------------------
def compute_prange(values, p=0.01):
    """
    If the range of values to clump is set to infer, set a range of thresholds
    :param values: Sorted numpy array of values
    :param p: porportion of thresholds to be sampled
    """
    n = np.around(values.shape[0] * p)
    return values[np.linspace(0, values.shape[0], n, endpoint=False, dtype=int)]


# ----------------------------------------------------------------------
def pplust_plink(outpref, bfile, sumstats, r2range, prange, snpwindow, phenofn,
                 plinkexe, allele_file, clump_field='P', sort_file=None,
                 f_thr=0.1,
                 plot=False, clean=False, maxmem=1700, threads=1,
                 score_type='sum'):
    """
    Execute P + T
    
    :param str outpref: Prefix for outputs
    :param str bfile: Prefix of plink-bed fileset
    :param str sumstats: File with the summary statistics in plink format 
    :param list r2range: List with floats representing LD thresholds
    :param list prange: List wirth pvalue thresholds
    :param int snpwindow: Size of the clumping window
    :param str phenofn: File with the phenotype in plink format
    :param str plinkexe: Path and executable of plink
    :param bool plot: Wether to plot the P + T results
    :param bool clean: Whether to clean the folder or not
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param str clump_field: field in the summary stats to clump with
    :param str sort_file: File with the values for SNP sorting
    """
    print('Performing ppt')
    # Read the summary statistics file
    sumstatsdf = pd.read_table(sumstats, delim_whitespace=True)
    frq = read_freq(bfile, plinkexe, freq_threshold=f_thr, maxmem=maxmem,
                    threads=threads)
    sumstatsdf = sumstatsdf.merge(frq, on=['SNP', 'A1', 'CHR'])
    sumstatsdf.rename(columns={'BETA': 'BETA_UNNORM'}, inplace=True)
    sumstatsdf['norm'] = np.sqrt((2 * sumstatsdf.MAF) * (1 - sumstatsdf.MAF))
    sumstatsdf['BETA'] = sumstatsdf.BETA_UNNORM / sumstatsdf.norm
    # Ensure the phenotype file contains only individuals from the bfile
    fn = os.path.split(bfile)[-1]
    if os.path.isfile('%s.keep' % fn):
        nphenofn = '%s.pheno' % fn
        keep = pd.read_table('%s.keep' % fn, delim_whitespace=True,
                             header=None, names=['FID', 'IID'])
        ph = pd.read_table(phenofn, delim_whitespace=True, header=None,
                           names=['FID', 'IID', 'pheno'])
        ph[ph.IID.isin(keep.IID)].to_csv(nphenofn, sep=' ', header=False,
                                         index=False)
        phenofn = nphenofn
    print('Performing clumping in %s...' % outpref)
    if prange == 'infer':
        sort_file = pd.read_table(sort_file, delim_whitespace=True)
        if any(sort_file.loc[:, clump_field] < 0):
            sort_file[clump_field] = sort_file.loc[:, clump_field].abs()
        sumstatsdf = sumstatsdf.merge(sort_file, on='SNP')
        sumstats = '%s_abs.txt' % sumstats[:sumstats.rfind('.')]
        vals = sumstatsdf.loc[:, clump_field].sort_values(ascending=False)
        sumstatsdf.to_csv(sumstats, sep=' ', index=False)
        prange = compute_prange(vals.values)
    # Execute the clumping and read the files
    results = [ScoreClumped_plink(outpref, bfile, clump_vars_plink(
        outpref, bfile, sumstats, r2, snpwindow, phenofn, allele_file, plinkexe,
        maxmem, threads, clump_field=clump_field), phenofn, sumstatsdf, r2,
                                  prange, plinkexe, allele_file, clump_field,
                                  maxmem,
                                  threads, score_type)
               for r2 in tqdm(r2range, total=len(r2range))]
    # Concatenate the results in a single data frame
    results = pd.concat(results)
    # Sort the results by R2 and write them to file
    results.sort_values('R2', inplace=True, ascending=False)
    results.to_csv('%s.results' % (outpref), sep='\t', index=False)
    # clumped = parse_sort_clump(results.iloc[0], sumstatsdf.SNP)
    # Plot results if necessary
    if plot:
        plotppt(outpref, results)
    # Clean folders if necessary
    cleanup(results, clean)
    # Returns the dataframe with the results and the path to its file
    return results, os.path.join(os.getcwd(), '%s.results' % (outpref))
    return clum


# ----------------------------------------------------------------------
def single_clump(snp, R2, done, r_thresh, extnd, clumps):
    if not snp in done:
        vec = R2.loc[:, snp]
        idx = (vec > r_thresh) & (~vec.index.isin(done))
        clum = vec.index[idx].tolist()
        clum.pop(clum.index(snp))
        extnd(clum)
        clumps[snp] = clum
        # return clum


# ----------------------------------------------------------------------
def clump(R2, sumstats, r_thresh, p_thresh):
    sub = sumstats[sumstats.p_value < p_thresh]
    done = []
    clumps = {}
    extnd = done.extend
    avail = sub.snp.tolist()
    r2 = R2.loc[avail, avail].compute()
    [single_clump(row.snp, r2, done, r_thresh, extnd, clumps)
     for row in sub.itertuples()]
    # set dataframe
    df = sub[sub.snp.isin(list(clumps.keys()))].reindex(columns=['snp',
                                                                 'p_value'])
    # set a dataframe compatible with ranumo
    df2 = df.copy()
    df['Tagged'] = [';'.join(clumps[v]) for v in df.snp]
    tagged = [x for v in df.snp for x in clumps[v]]
    tail = sub[sub.snp.isin(tagged)].reindex(columns=['snp', 'p_value'])
    df2 = df2.append(tail).reset_index()
    del clump, tagged, avail, done
    return df, df2


# ----------------------------------------------------------------------
def new_plot(prefix, ppt, geno, pheno, threads):
    params = dict(column='index', threads=threads)
    ppt, _ = smartcotagsort(prefix, ppt, **params)
    ppt = prune_it(ppt, geno, pheno, 'P+T %s' % prefix, step=5, threads=threads)
    f, ax = plt.subplots()
    ppt.plot(x='Number of SNPs', y='R2', kind='scatter',ax=ax, legend=True)
    plt.tight_layout()
    plt.savefig('%s_ppt.pdf' % prefix)
    plt.close()


# ----------------------------------------------------------------------
def score(geno, bim, pheno, sumstats, r_t, p_t, R2):
    print('Scoring with p-val %.2g and R2 %.2g' % (p_t, r_t))
    if isinstance(pheno, pd.core.frame.DataFrame):
        pheno = pheno.PHENO.values
    assert isinstance(pheno, np.ndarray)
    clumps, df2 = clump(R2, sumstats, r_t, p_t)
    assert isinstance(clumps, pd.core.frame.DataFrame)
    index = clumps.snp.tolist()
    idx = bim[bim.snp.isin(index)].i.tolist()
    betas = sumstats[sumstats.snp.isin(index)].slope
    prs = geno[:, idx].dot(betas)
    slope, intercept, r_value, p_value, std_err = lr(pheno, prs)
    del slope, intercept, p_value, std_err
    return r_t, p_t, r_value ** 2, clumps, prs, df2


# ----------------------------------------------------------------------
def pplust(prefix, geno, pheno, sumstats, r_range, p_thresh, split=3, seed=None,
           threads=1, **kwargs):
    X_train = None
    now = time.time()
    print ('Performing P + T!')
    seed = np.random.randint(1e4) if seed is None else seed
    # Read required info (sumstats, genfile)
    if 'bim' in kwargs:
        bim = kwargs['bim']
    if 'fam' in kwargs:
        fam = kwargs['fam']
    if isinstance(pheno, str):
        pheno = dd.read_table(pheno, blocksize=25e6,
                              delim_whitespace=True)
    elif pheno is None:
        # make simulation
        opts = {'outprefix': 'ppt_simulation', 'bfile': geno,
                'h2': kwargs['h2'], 'ncausal': kwargs['ncausal'],
                'normalize': kwargs['normalize'], 'uniform': kwargs['uniform'],
                'seed': seed}
        pheno, (geno, bim, truebeta, vec) = qtraits_simulation(**opts)
        opt2 = {'prefix': 'ppt_simulation', 'pheno': pheno, 'geno': geno,
                'validate': 3, 'seed': seed, 'threads': threads, 'bim':bim}
        sumstats, X_train, X_test, y_train, y_test = plink_free_gwas(**opt2)
    if isinstance(geno, str):
        (bim, fam, geno) = read_plink(geno)
        geno = geno.T
    if isinstance(sumstats, str):
        sumstats = pd.read_table(sumstats, delim_whitespace=True)
    # Compute LD (R2) in dask format
    R2 = dd.from_dask_array(geno, columns=bim.snp).corr() ** 2
    # Create training and testing set
    if X_train is None:
        X_train, X_test, y_train, y_test = train_test_split(geno, pheno,
                                                            test_size=1 / split,
                                                            random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_test, y_test,
                                                            test_size=1 / split,
                                                            random_state=seed)
    # Sort sumstats by pvalue and clump by R2
    sumstats = sumstats.sort_values(by='p_value', ascending=True)
    # do clumping
    delayed_results = [
        dask.delayed(score)(X_train, bim, y_train, sumstats, r_t, p_t, R2) for
        r_t, p_t in product(r_range, p_thresh)]
    r = list(dask.compute(*delayed_results, num_workers=threads))
    best = sorted(r, key=itemgetter(2), reverse=True)[0]
    with open('%s_ppt.results.tsv' % prefix, 'w') as F:
        line = ['\t'.join(['LD threshold', 'P-value threshold', 'R2'])]
        line += ['\t'.join(map(str, i[:-2])) for i in r]
        F.write('\n'.join(line))
    # score in test set
    r_t, p_t, r2, clumps, sc, df2 = score(X_test, bim, y_test, sumstats, best[0],
                                     best[1], R2)
    if isinstance(sc, np.ndarray):
        if isinstance(y_test, pd.core.frame.DataFrame):
            prs = y_test.reindex(columns=['fid', 'iid'])
        else:
            raise NotImplementedError
        prs['prs'] = sc
    else:
        prs = dd.from_dask_array(sc, columns='PRS')
        prs.loc[:, ['FID', 'IID']] = fam.loc[:, ['FID', 'IID']]
    print('P+T optimized with pvalue %.4g and LD value of %.3f: R2 = %.3f in '
          'the test set' % (p_t, r_t, r2))
    clumps.to_csv('%s.clumps' % prefix, sep='\t', index=False)
    prs.to_csv('%s.prs' % prefix, sep='\t', index=False)
    # plot the prune version:
    new_plot(prefix, df2, geno, pheno, threads)
    print ('P + T Done after %.2f minutes' % ((time.time() - now) / 60.))
    return p_t, r_t, r2, clumps, prs, df2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bfile', help='plink fileset prefix',
                        required=True)
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-s', '--sumstats', help='Filename of Sumstats',
                        default=None)
    parser.add_argument('-P', '--pheno', help='Filename of phenotype file',
                        default=None)
    parser.add_argument('-A', '--allele_file', default='EUR.allele',
                        help='File with the allele order. A1 in position 3 and '
                             'id in position2')
    parser.add_argument('-n', '--plinkexe', help=('Path and executable file of '
                                                  'plink'))
    parser.add_argument('-l', '--LDwindow',
                        help='Physical distance threshold ' +
                             'for clumping in kb (250kb by default)', type=int,
                        default=250)
    parser.add_argument('-c', '--rstart', help='minimum R2 threshold. '
                                               'Default: 0.1', type=float,
                        default=0.1)
    parser.add_argument('-d', '--rstop', help='maximum R2 threshold. '
                                              'Default: 0.8', type=float,
                        default=0.8)
    parser.add_argument('-e', '--rstep', help='step for R2 threshold. '
                                              'Default: 0.1', type=float,
                        default=0.1)
    parser.add_argument('-v', '--pstart',
                        help='Minimum value for for the Pval' +
                             ' range. Default: 1E-8', type=float, default=1E-8)
    parser.add_argument('-w', '--pstop', help='Maximum value for for the Pval' +
                                              ' range. Default: 1', type=float,
                        default=1)
    parser.add_argument('-C', '--customP', help='Custom pvalue range.' +
                                                'Default: (None)', default=None)
    parser.add_argument('-z', '--clean', help='Cleanup the clump and profiles',
                        default=False, action='store_true')
    parser.add_argument('-L', '--label', help='Label of the populations being' +
                                              ' analyzed.', default='EUR')
    parser.add_argument('-t', '--plot', help='Plot results of analysis',
                        default=False, action='store_true')
    parser.add_argument('-f', '--clump_field', default='P',
                        help=('field in the summary stats to clump with'))
    parser.add_argument('-a', '--sort_file', default=None,
                        help=('File to sort the snps by instead of pvalue'))
    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=3000, type=int)
    parser.add_argument('-S', '--score_type', default='sum')
    parser.add_argument('--h2', default=None, type=float)
    parser.add_argument('--ncausal', default=None, type=int)
    parser.add_argument('--normalize', default=None, type=bool)
    parser.add_argument('--uniform', default=None, type=bool)

    args = parser.parse_args()

    LDs = [x if x <= 0.99 else 0.99 for x in sorted(
        np.arange(args.rstart, args.rstop + args.rstep, args.rstep),
        reverse=True)]
    if args.customP:
        if args.customP != 'infer':
            Ps = [float('%.1g' % float(x)) for x in args.customP.split(',')]
            Ps = sorted(Ps, reverse=True)
        else:
            Ps = 'infer'
    else:
        sta, sto = np.log10(args.pstart), np.log10(args.pstop)
        Ps = sorted([float('%.1g' % 10 ** (x)) for x in np.concatenate(
            (np.arange(sta, sto), [sto]), axis=0)], reverse=True)
    prs = pplust(args.prefix, args.bfile, args.pheno, args.sumstats,
                 LDs, Ps, threads=args.threads, h2=args.h2,
                 ncausal=args.ncausal, normalize=args.normalize,
                 uniform=args.uniform)
    # pplust_plink(args.prefix, args.bfile, args.sumstats, LDs, Ps, args.LDwindow,
    #              args.pheno, args.plinkexe, args.allele_file,
    #              clump_field=args.clump_field, sort_file=args.sort_file,
    #              plot=args.plot, clean=args.clean, maxmem=args.maxmem,
    #              threads=args.threads, score_type=args.score_type)
