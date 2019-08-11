#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<2017>
  Purpose: Run regression analyses
  Created: 09/30/17
"""

import argparse
import time
from itertools import cycle
from multiprocessing import cpu_count
import gc
import dask.array as da
import h5py
import matplotlib
import mpmath as mp
import statsmodels.formula.api as smf
from dask.array.core import Array
from sklearn.decomposition import PCA


from qtraitsimulation import qtraits_simulation
from utilities4cotagging import *
from dask.distributed import Client, LocalCluster
# require gmpy
mp.dps = 25
mp.pretty = True
matplotlib.use('Agg')
plt.style.use('ggplot')


# ----------------------------------------------------------------------
def t_sf(t, df):
    """
    Student t distribution cumulative density function or survival function

    :param t: t statistic
    :param df: degrees of freedom
    :return: area under the PDF from -inf to t
    """
    t = -mp.fabs(t)
    lhs = mp.gamma((df + 1) / 2) / (mp.sqrt(df * mp.pi) * mp.gamma(df / 2))
    rhs = mp.quad(lambda x: (1 + (x * x) / df) ** (-df / 2 - 1 / 2),
                  [-mp.inf, t])
    gc.collect()
    return lhs * rhs


# ----------------------------------------------------------------------
@jit
def nu_linregress(x, y):
    """
    Refactor of the scipy linregress with mpmath in the estimation of the
    pvalue, numba, and less checks for speed sake

    :param x: array for independent variable
    :param y: array for the dependent variable
    :return: dictionary with slope, intercept, r, pvalue and stderr
    """
    cols = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
    # Make sure x and y are arrays
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    # means in vector form
    xmean = np.mean(x, None)
    ymean = np.mean(y, None)
    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=True).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    # Estimate correlation
    r = r_num / r_den
    # test for numerical error propagation
    if r > 1.0:
        r = 1.0
    elif r < -1.0:
        r = -1.0
    # estimate degrees of freedom
    df = n - 2
    slope = r_num / ssxm
    intercept = ymean - slope * xmean
    # Estimate t-statistic
    t = r * np.sqrt(df / ((1.0 - r) * (1.0 + r)))
    # Get the pvalue
    prob = 2 * t_sf(t, df)
    # get the estimated standard error
    sterrest = np.sqrt((1 - r * r) * ssym / ssxm / df)
    return dict(zip(cols, [slope, intercept, r, prob, sterrest]))


# ----------------------------------------------------------------------
@jit
def high_precision_pvalue(df, r):
    r = r if np.abs(r) != 1.0 else mp.mpf(0.9999999999999999) * mp.sign(r)
    den = ((1.0 - r) * (1.0 + r))
    t = r * np.sqrt(df / den)
    return t_sf(t, df) * 2


# ----------------------------------------------------------------------
def manhattan_plot(outfn, p_values, causal_pos=None, alpha=0.05, title=''):
    """
    Generates a manhattan plot for a list of p-values. Overlays a horizontal
    line indicating the Bonferroni significance threshold assuming all p-values
    derive from independent test.

    :param outfn: Outfilename
    :param causal_pos: List with floats to include vertical lines as causal
    :param p_values: A list of single-test p-values
    :param alpha: The single-site significance threshold
    :param str title: Include a title in the plot
    """
    # Compute the bonferrony corrected threshold
    bonferroni_threshold = alpha / len(p_values)
    # Make it log
    log_b_t = -np.log10(bonferroni_threshold)
    # Fix infinites
    p_values = np.array(p_values)
    p_values[np.where(p_values < 1E-10)] = 1E-10
    # Make the values logaritm
    vals = -np.log10(p_values)
    # Plot it
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    # Add threshold line
    ax2.axhline(y=log_b_t, linewidth=1, color='r', ls='--')
    # Add shaded regions on the causal positions
    if causal_pos is not None:
        [ax2.axvspan(x - 0.2, x + 0.2, facecolor='0.8', alpha=0.8) for x in
         causal_pos]
    # Plot one point per value
    ax2.plot(vals, '.', ms=1)
    # Zoom-in / limit the view to different portions of the data
    ymax = max(vals)
    ax2.set_ylim(0, ymax + 0.2)  # most of the data
    ax2.set_xlim([-0.2, len(vals) + 1])
    plt.xlabel(r"marker index")
    plt.ylabel(r"-log10(p-value)")
    plt.title(title)
    plt.savefig(outfn)
    plt.close()


# ----------------------------------------------------------------------
def st_mod(x, y, covs=None):
    """
    Linear regression using stats models. This module is very slow but allows to
    include covariates in the estimation.

    :param x: array for independent variable
    :param y: array for dependent variable
    :param covs: array for covariates
    :return: Regression results
    """
    df = pd.DataFrame({'geno': x, 'pheno': y})
    cols = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'b_pval',
            'b_std_err']
    if np.allclose(x.var(), 0.0):
        linregress_result = dict(zip(cols, cycle([np.nan])))
    else:
        if covs is not None:
            c = []
            for col in range(covs.shape[1]):
                df['Cov%d' % col] = covs[:, col]
                c.append('Cov%d' % col)
            formula = 'pheno ~ geno + %s' % ' + '.join(c)
        else:
            formula = 'pheno ~ geno'
        model = smf.ols(formula=formula, data=df)
        results = model.fit()
        vals = [results.params.Intercept, results.params.geno,
                results.pvalues.Intercept, results.pvalues.geno,
                results.rsquared, results.bse.Intercept, results.bse.geno]
        linregress_result = dict(zip(cols, vals))
    return linregress_result


# ----------------------------------------------------------------------
@jit(parallel=True)
def do_pca(g, n_comp):
    """
    Perform a PCA on the genetic array and return n_comp of it

    :param g: Genotype array
    :param n_comp: Number of components sought
    :return: components array
    """
    pca = PCA(n_components=n_comp)
    pca = pca.fit_transform(g)
    return pca


# ----------------------------------------------------------------------
def load_previous_run(prefix, threads):
    """
    Load a previos GWAS run

    :param prefix: The prefix of the output files from the previous run
    :param threads: Number of threads to be used in the estimations
    :return: previous gwas results
    """
    # Get the file names
    pfn = '%s_phenos.hdf5' % prefix
    gfn = '%s.geno.hdf5' % prefix

    f = h5py.File(gfn, 'r')  # Read the genotype h5 file
    chunks = np.load('chunks.npy')  # Load the chunks stored
    # Estimate chunk sizes given the number of threads
    chunks = [estimate_chunks(tuple(i), threads) for i in chunks]
    # Get training set of the genotype array
    x_train = da.from_array(f.get('x_train'), chunks=tuple(chunks[0]))
    # Get the test set of the genotype array
    x_test = da.from_array(f.get('x_test'), chunks=tuple(chunks[1]))
    # Get the training set of the phenotype
    y_train = pd.read_hdf(pfn, key='y_train')
    # Get the testing set of the phenotype
    y_test = pd.read_hdf(pfn, key='y_test')
    # Read the resulting gwas table
    res = pd.read_csv('%s.gwas' % prefix, sep='\t')
    return res, x_train, x_test, y_train, y_test


# ----------------------------------------------------------------------
def plink_free_gwas(prefix, pheno, geno, validate=None, seed=None, plot=False,
                    causal_pos=None, threads=8, pca=None, stmd=False,
                    high_precision=False, max_memory=None,
                    high_precision_on_zero=False, **kwargs):
    """
    Compute the least square regression for a genotype in a phenotype. This
    assumes that the phenotype has been computed from a nearly independent set
    of variants to be accurate (I believe that that is the case for most
    programs but it is not "advertised")

    :param max_memory: Memory limit
    :param prefix: Prefix of outputs
    :param pheno: Filename or dataframe wiith phenotype
    :param geno: Filename or array (numpy or dask) with the genotype
    :param validate: Number of training/test set splits
    :param seed: Random seed
    :param plot: Whether to make a manhatan plot or not
    :param causal_pos: Location of causal variants (for plotting purposes only)
    :param threads: Number of threads to use in the computation
    :param pca: Perform PCA
    :param stmd: Use statsmodels insted of linregress (is set to True if pca)
    :param high_precision: Use arbitrary precision in the pvalue estimation
    :param kwargs: Keyword arguments for qtraits_simulation and read_geno
    :return: regression results and validation sets
    """
    # # Set CPU limits
    # soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    # resource.setrlimit(resource.RLIMIT_NPROC, (threads, hard))
    # soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    # print('Soft limit changed to :', soft)

    # set Cache to protect memory spilling
    if max_memory is not None:
        available_memory = max_memory
    else:
        available_memory = psutil.virtual_memory().available
    cache = Chest(available_memory=available_memory)
    seed = np.random.randint(10000) if seed is None else seed
    print('Performing GWAS\n    Using seed', seed)
    now = time.time()
    pfn = '%s_phenos.hdf5' % prefix
    gfn = '%s.geno.hdf5' % prefix
    if os.path.isfile(pfn):
        res, x_train, x_test, y_train, y_test = load_previous_run(prefix,
                                                                  threads)
    else:
        np.random.seed(seed=seed)
        if isinstance(geno, str):
            # read the genotype files if the provided geno is a string
            options = dict(bfile=geno, freq_thresh=kwargs['f_thr'],
                           threads=threads, flip=kwargs['flip'],
                           check=kwargs['check'], max_memory=max_memory)
            (bim, fam, x) = read_geno(**options)
        else:
            # Check that is in an apropriate format of dask or numpy arrays
            if isinstance(kwargs['bim'], str):
                bim = pd.read_table(kwargs['bim'], delim_whitespace=True)
            else:
                bim = kwargs['bim']
            try:
                assert isinstance(geno, Array)
                x = geno.rechunk((geno.shape[0], geno.chunks[1]))
            except AssertionError:
                assert isinstance(geno, np.ndarray)
                x = geno
        del geno
        gc.collect()
        if pheno is None:
            # If pheno is not provided, simulate it using qtraits_simulation
            options.update(kwargs)
            pheno, h2, gen = qtraits_simulation(prefix, **options)
            (x, bim, truebeta, vec) = gen
        elif isinstance(pheno, str):
            # If pheno is provided as a string, read it
            pheno = pd.read_table(pheno, delim_whitespace=True, header=None,
                                  names=['fid', 'iid', 'PHENO'])
        try:
            y = pheno.compute(num_workers=threads, cache=cache)
        except AttributeError:
            y = pheno
        del pheno
        gc.collect()
        if validate is not None:
            print('making the crossvalidation data')
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=1 / validate, random_state=seed)
        else:
            # If validation is not require just return duplicates
            x_train, x_test, y_train, y_test = x, x, y, y
        del x, y
        gc.collect()
        # write test and train IDs
        opts = dict(sep=' ', index=False, header=False)
        y_test.to_csv('%s_testIDs.txt' % prefix, **opts)
        y_train.to_csv('%s_trainIDs.txt' % prefix, **opts)
        if isinstance(x_train, dask.array.core.Array):
            x_train = x_train.rechunk(
                estimate_chunks(x_train.shape, threads, max_memory))
        if 'normalize' in kwargs:
            if kwargs['normalize']:
                print('Normalizing train set to variance 1 and mean 0')
                x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0
                                                                         )
                print('Normalizing test set to variance 1 and mean 0')
                x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)
        # Get apropriate function for linear regression
        func = nu_linregress if high_precision else st_mod if stmd else lr
        daskpheno = dask.delayed(y_train.PHENO)
        daskgeno = dask.delayed(x_train)
        if pca is not None:
            print('Using %d PCs' % pca)
            #Perform PCA
            func = st_mod                   # Force function to statsmodels
            covs = do_pca(x_train, pca)     # Estimate PCAs
            delayed_results = [
                dask.delayed(func)(daskgeno[:, i], daskpheno, covs=covs) for
                i in range(x_train.shape[1])]
        else:
            delayed_results = [dask.delayed(func)(daskgeno[:, i], daskpheno)
                               for i in range(x_train.shape[1])]
        dask_options = dict(num_workers=threads, cache=cache,
                            pool=ThreadPool(threads))
        with ProgressBar(), dask.config.set(**dask_options):
            print('Performing regressions')
            r = list(dask.compute(*delayed_results))
            gc.collect()
        try:
            res = pd.DataFrame.from_records(r, columns=r[0]._fields)
        except AttributeError:
            res = pd.DataFrame(r)
        assert res.shape[0] == bim.shape[0]  # Make sure no missing data
        # Combine mapping and gwas
        res = pd.concat((res, bim.reset_index()), axis=1)
        # check precision issues and re-run the association
        zeros = res[res.pvalue == 0.0]
        if not zeros.empty and not stmd and high_precision_on_zero:
            print('    Processing zeros with arbitrary precision')
            df = x_train.shape[0] - 2
            dr = [dask.delayed(high_precision_pvalue)(df, r) for r in
                  zeros.rvalue.values]
            with ProgressBar(), dask.config.set(**dask_options):
                zero_res = np.array(dask.compute(*dr))
            res.loc[res.pvalue == 0.0, 'pvalue'] = zero_res
            res['pvalue'] = [mp.mpf(z) for z in res.pvalue]
        # Make a manhatan plot
        if plot:
            manhattan_plot('%s.manhatan.pdf' % prefix, res.slope, causal_pos,
                           alpha=0.05)
        # write files
        res.to_csv('%s.gwas' % prefix, sep='\t', index=False)

        labels = ['/x_train', '/x_test']
        arrays = [x_train, x_test]
        y_train.to_hdf(pfn, 'y_train', table=True, mode='a', format="table")
        y_test.to_hdf(pfn, 'y_test', table=True, mode='a', format="table")
        assert len(x_train.shape) == 2
        assert len(x_test.shape) == 2
        chunks = np.array([x_train.shape, x_test.shape])
        np.save('chunks.npy', chunks)
        data = dict(zip(labels, arrays))
        da.to_hdf5(gfn, data)
    print('GWAS DONE after %.2f seconds !!' % (time.time() - now))
    return res, x_train, x_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-g', '--geno', default='EUR', required=True)
    parser.add_argument('-f', '--pheno', help='Phenotype file', default=None)
    parser.add_argument('-v', '--validate', default=None, type=int)
    parser.add_argument('-V', '--validsnpsfile', default=None)
    parser.add_argument('-C', '--covs', default=None, action='store')
    parser.add_argument('-s', '--nosex', default=False, action='store_true')
    parser.add_argument('-l', '--plot', help='Generate a manhatan plot',
                        default=False, action='store_true')
    parser.add_argument('-S', '--seed', help='Random seed', default=None)
    parser.add_argument('-t', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=None, type=int)
    parser.add_argument('--use_statsmodels', action='store_true')
    parser.add_argument('--pca', default=None, type=int)
    parser.add_argument('--f_thr', type=float, default=0,
                        help=('Keyword argument for read_geno. The frequency '
                              'threshold to cleanup the genotype file'))
    parser.add_argument('--flip', action='store_true', default=False,
                        help=('Keyword argument for read_geno. Whether to flip '
                              'the genotypes when MAF is > 0.5'))
    parser.add_argument('--check', action='store_true', default=False,
                        help=('Keyword argument for read_geno. Whether to check'
                              ' the genotypes for invariant columns.'))
    parser.add_argument('--h2', default=0.5, type=float,
                        help=('Keyword argument for qtraits_simulation. '
                              'Heritability of the trait.'))
    parser.add_argument('--ncausal', default=1, type=int,
                        help=('Keyword argument for qtraits_simulation. Number '
                              'of causal variats.'))
    parser.add_argument('--bim', help=('Filename of the mapping file if geno is'
                                       ' an array'))
    parser.add_argument('--normalize', action='store_true',
                        help=('Keyword argument for qtraits_simulation. Whether '
                              'to normalize the genotype or not.'))
    parser.add_argument('--snps', type=list, nargs='+',
                        help=('Keyword argument for qtraits_simulation. List of'
                              ' causal snps.'))
    parser.add_argument('--freqthreshold', default=0, type=float,
                        help=('Keyword argument for qtraits_simulation. Lower '
                              'threshold to filter MAF by'))

    args = parser.parse_args()
    plink_free_gwas(args.prefix, args.pheno, args.geno, validate=args.validate,
                    plot=args.plot, threads=args.threads, seed=args.seed,
                    stmd=args.use_statsmodels, flip=args.flip, pca=args.pca,
                    max_memory=args.maxmem, f_thr=args.freqthreshold,
                    check=args.check)
