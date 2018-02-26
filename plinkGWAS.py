#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<>
  Purpose: Run plink gwas analyses
  Created: 09/30/17
"""
import argparse
import os
import time
import dask.dataframe as dd
import dask.array as da
import matplotlib
import numpy as np
import pandas as pd
from dask.array.core import Array
import dask
from pandas_plink import read_plink
from collections import namedtuple
from scipy import stats
import h5py
import statsmodels.api as sm
import statsmodels.formula.api as smf
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utilities4cotagging import *
from sklearn.model_selection import train_test_split
from qtraitsimulation import qtraits_simulation
from multiprocessing import Pool, cpu_count
from collections import Counter
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from numba import jit, prange
import mpmath as mp
# require gmpy

lr = jit(stats.linregress)


# ----------------------------------------------------------------------
def t_sf(t, df):
    a = df / 2
    b = 1 / 2
    x = df / ((t ** 2) + df)
    I = mp.betainc(a, b, 0, x)
    #mp.dps = 1
    #f = lambda x: (1 + x ** 2 / df) ** (-df / 2 - 1 / 2)
    #p = 1 / 2 + (t * mp.gamma((df+1)/2)/(mp.sqrt(df * mp.pi) * mp.gamma(df/2)))
    #p *= mp.hyp2f1(1/2,(df+1)/2, 3/2, (-t**2 / df))
    #mp.quad(f, [-mp.inf, t])
    return I


# ----------------------------------------------------------------------
@jit
def nu_linregress(x, y):
    """
    Refactor of the scipy linregress with numba and less checks for speed sake

    :param x: array for independent
    :param y:
    :return:
    """
    cols = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']
    LinregressResult = namedtuple('LinregressResult', cols)
    #TINY = 1.0e-20
    x = np.asarray(x)
    y = np.asarray(y)
    arr = np.array([x, y])
    n = len(x)
    xmean = np.mean(x, None)
    ymean = np.mean(y, None)
    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = (np.dot(arr, arr.T) / n).flat
    r_num = ssxym * mp.mpf(1)
    r_den = mp.sqrt(ssxm * ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
    df = n - 2
    slope = r_num / ssxm
    intercept = ymean - slope * xmean
    t = r * np.sqrt(df / ((1.0 - r) * (1.0 + r)))
    prob = t_sf(np.abs(t), df)
    sterrest = np.sqrt((1 - r ** 2) * ssym / ssxm / df)
    return LinregressResult(slope, intercept, r, prob, sterrest)


# ----------------------------------------------------------------------
@jit
def da_linregress(x, y):
    """
    Refactor of the scipy linregress with numba, less checks for speed sake and
    done with dask arrays

    :param x: array for independent
    :param y:
    :return:
    """
    TINY = 1.0e-20
    # x = np.asarray(x)
    # y = np.asarray(y)
    arr = da.stack([x, y], axis=1)
    n = len(x)
    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = (da.dot(arr.T, arr) / n).ravel()
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
    df = n - 2
    slope = r_num / ssxm
    r_t = r + TINY
    t = r * da.sqrt(df / ((1.0 - r_t) * (1.0 + r_t)))
    prob = 2 * stats.distributions.t.sf(np.abs(t), df)
    return slope, r ** 2, prob


# ----------------------------------------------------------------------
def gwas(plinkexe, bfile, outprefix, allele_file, covs=None, nosex=False,
         threads=False, maxmem=False, validsnpsfile=None):
    """
    Execute plink gwas. This assumes continuos phenotype and that standard qc 
    has been done already

    :param str plinkexe: Path and executable of plink
    :param str bfile: Tuple with Prefix for the bed fileset and phenotype file
    :param str outprefix: Prefix for the outputs
    :param str pheno: Filename to phenotype in plink format
    :param bool nosex: If option --allow-no-sex should be used
    :param str covs: Filename with the covariates to use
    :param str validsnpsfile: file with valid snps to pass to --extract 
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    """
    ## for plinkgwas string:
    ## 1) plink path and executable
    ## 2) prefix for the bed fileset
    ## 3) Name of covariate file
    ## 4) names of the columns to use in the covariance file separated by "," 
    ## or '-' if range
    ## 5) prefix for outputs
    bfile, pheno = bfile
    # Format CLA
    plinkgwas = ("%s --bfile %s --assoc fisher-midp --linear --pheno %s --prune"
                 " --out %s_gwas --ci 0.95 --a1-allele %s 3 2 --vif 100")
    plinkgwas = plinkgwas % (plinkexe, bfile, pheno, outprefix, allele_file)
    # Include a subset of snps file if required
    if validsnpsfile is not None:
        plinkgwas += " --extract %s" % validsnpsfile
    # Include Covariates
    if covs:
        plinkgwas += " --covar %s keep-pheno-on-missing-cov" % covs
    # Ignore sex
    if nosex:
        plinkgwas += ' --allow-no-sex'
    else:
        plinkgwas += ' --sex'
    # Inlcude threads and memory
    if threads:
        plinkgwas += ' --threads %s' % threads
    if maxmem:
        plinkgwas += ' --memory %s' % maxmem
        # execulte CLA
    out = executeLine(plinkgwas)
    # Set the Outfile
    fn = '%s_gwas.assoc.linear' % outprefix
    # Return the dataframe of the GWAS and its filename
    return pd.read_table(fn, delim_whitespace=True), fn


# ----------------------------------------------------------------------
def manhattan_plot(outfn, p_values, causal_pos=None, alpha=0.05, title=''):
    """ 
    Generates a manhattan plot for a list of p-values. Overlays a horizontal 
    line indicating the Bonferroni significance threshold assuming all p-values 
    derive from independent test.
    
    :param str outfn: Outfilename
    :param list causal_pos: List with floats to include vertical lines as causal
    :param list p-values: A list of single-test p-values
    :param float alpha: The single-site significance threshold
    :param str title: Include a title in the plot
    """
    # Get the lenght of the p-values array
    L = len(p_values)
    # Compute the bonferrony corrected threshold
    bonferroni_threshold = alpha / L
    # Make it log
    logBT = -np.log10(bonferroni_threshold)
    # Fix infinites
    p_values = np.array(p_values)
    p_values[np.where(p_values < 1E-10)] = 1E-10
    # Make the values logaritm
    vals = -np.log10(p_values)
    # Plot it
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    # Add threshold line
    ax2.axhline(y=logBT, linewidth=1, color='r', ls='--')
    # Add shaded regions on the causal positions
    if causal_pos is not None:
        [ax2.axvspan(x - 0.2, x + 0.2, facecolor='0.8', alpha=0.8) for x in
         causal_pos]
    # Plot one point per value
    ax2.plot(vals, '.', ms=1)
    # Zoom-in / limit the view to different portions of the data
    Ymin = min(vals)
    Ymax = max(vals)
    ax2.set_ylim(0, Ymax + 0.2)  # most of the data
    ax2.set_xlim([-0.2, len(vals) + 1])
    plt.xlabel(r"marker index")
    plt.ylabel(r"-log10(p-value)")
    plt.title(title)
    plt.savefig(outfn)
    plt.close()


# ----------------------------------------------------------------------
def plink_gwas(plinkexe, bfile, outprefix, pheno, allele_file, covs=None,
               nosex=False, threads=cpu_count(), maxmem=False, validate=None,
               validsnpsfile=None, plot=False, causal_pos=None):
    """
    execute the gwas
    
    :param str plinkexe: Path and executable of plink
    :param str bfile: Prefix for the bed fileset
    :param str outprefix: Prefix for the outputs
    :param bool nosex: If option --allow-no-sex should be used
    :param str covs: Filename with the covariates to use
    :param int/bool validate: To split the bed into test and tGraining
    """
    print('Performing plinkGWAS')
    # Create a test/train validation sets
    if validate is not None:
        keeps = train_test(outprefix, bfile, plinkexe, pheno, validate, maxmem,
                           threads)
        for k, v in keeps.items():
            if 'train' in k:
                train = k, v[1]
            elif 'test' in k:
                test = k, v[1]
            else:
                raise TypeError
    else:
        train, test = (bfile, pheno), (None, pheno)
    # Run the GWAS
    # training, pheno = train
    gws, fn = gwas(plinkexe, train, outprefix, allele_file, covs, nosex=nosex,
                   threads=threads, maxmem=maxmem, validsnpsfile=validsnpsfile)
    # Make a manhatan plot
    if plot:
        manhattan_plot('%s.manhatan.pdf' % outprefix, gws.P, causal_pos,
                       alpha=0.05)
    # Return the gwas dataframe, its filename, and the train and test sets 
    # prefix
    return gws, os.path.join(os.getcwd(), fn), train, test


def matrix_reg(X, Y):
    bs_hat, sse, rank, sing = da.linalg.lstsq(X, Y)
    se = np.sqrt(da.diag(sse * da.linalg.inv(np.dot(X.T, X))))
    t = bs_hat / se
    pval = stats.t.sf(np.abs(t), X.shape[0] - X.shape[1] - 1) * 2
    return bs_hat, pval


def regression_iter(x, y, threads):
    for i in range(x.shape[1]):
        print('processing', i)
        yield x[:, i].compute(num_workers=threads), y.compute(
            num_workers=threads)

# ----------------------------------------------------------------------
@jit(parallel=True)
def st_mod(x, y, covs=None):
    df = pd.DataFrame({'geno':x, 'pheno':y})
    cols = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'b_pval',
           'b_std_err']
    LinregressResult = namedtuple('LinregressResult', cols)
    if np.allclose(x.var(), 0.0):
        slope = intercept = p_value = b_pval = r_value = std_err = np.nan
        b_std_err = np.nan
    else:
        if covs is not None:
            c = []
            for col in range(covs.shape[1]):
                df['Cov%d' % col] = covs[:, col]
                c.append('Cov%d' % col)
            formula = 'pheno ~ geno + %s' % ' + '.join(c)
        else:
            formula = 'pheno ~ geno'
        # X = sm.add_constant(x)
        # model = sm.OLS(y, X)
        model = smf.ols(formula=formula, data=df)
        results = model.fit()
        intercept= results.params.Intercept
        slope = results.params.geno
        b_pval = results.pvalues.Intercept
        p_value = results.pvalues.geno
        r_value = results.rsquared
        b_std_err = results.bse.Intercept
        std_err = results.bse.geno
    return LinregressResult(slope, intercept, r_value, p_value, std_err, b_pval,
                            b_std_err)


# ----------------------------------------------------------------------
@jit(parallel=True)
def do_pca(G, n_comp):
    pca = PCA(n_components=n_comp)
    pca = pca.fit_transform(G)
    return pca


# ----------------------------------------------------------------------
def load_previous_run(prefix, threads):
    # TODO: not working check chinks
    pfn = '%s_phenos.hdf5' % prefix
    gfn = '%s.geno.hdf5' % prefix
    f = h5py.File(gfn, 'r')
    chunks = np.load('chunks.npy')
    chunks = [estimate_chunks(i, threads) for i in chunks]
    x_train = da.from_array(f.get('x_train'), chunks=tuple(chunks[0]))
    x_test = da.from_array(f.get('x_test'), chunks=tuple(chunks[1]))
    y_train = pd.read_hdf(pfn, key='y_train')
    y_test = pd.read_hdf(pfn, key='y_test')
    res = pd.read_csv('%s.gwas' % prefix, sep='\t')
    return res, x_train, x_test, y_train, y_test


# ----------------------------------------------------------------------
@jit(parallel=True)
def func(snp):
    if np.unique(snp).shape[0] < 3:
        nsnp = snp
    else:
        c = np.bincount(snp) / snp.shape[0]
        frq = c[0] + (0.5 * c[1])
        ofrq = c[2] + (0.5 * c[1])
        if frq > ofrq:
            nsnp = da.from_array((snp - 2) * -1, chunks=snp.chunks)
        else:
            nsnp = snp
    return nsnp


# ----------------------------------------------------------------------
@jit(parallel=True)
def flip(G):
    print('fixing possible flips (this might take a while)')
    #delayed_results = [dask.delayed(func)(G[i, :]) for i in range(G.shape[0])]
    #r = list(dask.compute(*delayed_results, num_workers=threads))
    #p = Pool(threads); r = p.map(func, (G[i,:] for i in range(G.shape[0])))
    r = []
    rppend = r.append
    for i in prange(G.shape[0]):
        rppend(func(G[i, :]))
    darray = da.stack(r)
    return darray


# ----------------------------------------------------------------------
def plink_free_gwas(prefix, pheno, geno, validate=None, seed=None, flip=False,
                    causal_pos=None, plot=False, threads=cpu_count(), pca=None,
                    stmd=False, high_precision=False, **kwargs):
    """
    Compute the least square regression for a genotype in a phenotype. This
    assumes that the phenotype has been computed from a nearly independent set
    of variants to be accurate (I believe that that is the case for most
    programs but it is not "advertised")

    :param pheno: phenotype file or dataframe
    :param kwargs: key word arguments with either bfile or array parameters
    :return: betas and pvalues
    """
    print('Performing GWAS')
    now = time.time()
    pfn = '%s_phenos.hdf5' % prefix
    gfn = '%s.geno.hdf5' % prefix
    if os.path.isfile(pfn):
        res, x_train, x_test, y_train, y_test = load_previous_run(prefix,
                                                                  threads)
    else:
        if 'bfile' in kwargs:
            bfile = kwargs['bfile']
        if 'bim' in kwargs:
            bim = kwargs['bim']
        seed = np.random.randint(1e4) if seed is None else seed
        print('using seed %d' % seed)
        np.random.seed(seed=seed)
        if isinstance(geno, str):
            (bim, fam, geno) = read_plink(geno)
            geno = geno.T
            geno = (geno - geno.mean(axis=0)) / geno.std(axis=0)
        else:
            try:
                assert isinstance(geno, Array)
            except AssertionError:
                assert isinstance(geno, np.ndarray)
        if pheno is None:
            pheno, h2, gen = qtraits_simulation(prefix, **kwargs)
            (geno, bim, truebeta, vec) = gen
        elif isinstance(pheno, str):
            pheno = pd.read_table(pheno, delim_whitespace=True, header=None,
                                  names=['fid', 'iid', 'PHENO'])
        if isinstance(geno, dask.array.core.Array):
            x = geno.rechunk((geno.shape[0], geno.chunks[1]))
        else:
            x = geno
        try:
            y = pheno.compute(num_workers=threads)
        except AttributeError:
            y = pheno
        if validate is not None:
            print('making the crossvalidation data')
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=1 / validate, random_state=seed)
        else:
            x_train, x_test, y_train, y_test = x, x, y, y
        # write test and train IDs
        opts = dict(sep=' ', index=False, header=False)
        y_test.to_csv('%s_testIDs.txt' % prefix, **opts)
        y_train.to_csv('%s_trainIDs.txt' % prefix, **opts)
        chunks = tuple(np.ceil(np.array(x_train.shape) * np.array([0.6, 0.1])
                               ).astype(int))
        if isinstance(x_train, dask.array.core.Array):
            x_train = x_train.rechunk(chunks)
        print('using dask delayed')
        func = nu_linregress if high_precision else st_mod if stmd else lr
        if pca is not None:
            func = st_mod
            covs = do_pca(x_train, pca)
            delayed_results = [
                dask.delayed(func)(x_train[:, i], y_train.PHENO, covs=covs) for
                i in range(x_train.shape[1])]
        else:
            delayed_results = [dask.delayed(func)(x_train[:, i], y_train.PHENO)
                               for i in range(x_train.shape[1])]
        with ProgressBar():
            r = list(dask.compute(*delayed_results, num_workers=threads))
        res = pd.DataFrame.from_records(r, columns=r[0]._fields)
        assert res.shape[0] == bim.shape[0]
        res = pd.concat((res, bim), axis=1)
        # check if flips
        if flip:
            res.loc[res.flip, 'slope'] = res.loc[res.flip].slope * -1
        # if 'flip' in res.columns:
        #     res['slope_old'] = res.slope
        #     res.loc[res.flip, 'slope'] = res[res.flip].slope * -1
        # # Make a manhatan plot
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
        #arrays = [chunks] + prearrays
        data = dict(zip(labels, arrays))
        da.to_hdf5(gfn, data)
    print('GWAS DONE after %.2f seconds !!' % (time.time() - now))
    return res, x_train, x_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-B', '--bfile', default='EUR')
    parser.add_argument('-f', '--pheno', help='Phenotype file', required=True)
    parser.add_argument('-P', '--plinkexe', default='~/Programs/plink_mac/plink'
                        )
    parser.add_argument('-a', '--allele_file', default='EUR.allele',
                        help='File with the allele order. A1 in position 3 and '
                             'id in position2')
    parser.add_argument('-v', '--validate', default=None, type=int)
    parser.add_argument('-V', '--validsnpsfile', default=None)
    parser.add_argument('-C', '--covs', default=None, action='store')
    parser.add_argument('-s', '--nosex', default=False, action='store_true')
    parser.add_argument('-l', '--plot', help=('Generate a manhatan plot'),
                        default=False, action='store_true')
    parser.add_argument('-S', '--seed', help=('Random seed'),
                        default=None)
    parser.add_argument('-t', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=None, type=int)
    parser.add_argument('--use_statsmodels', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--pca', default=None, type=int)
    args = parser.parse_args()
    plink_free_gwas(args.prefix, args.pheno, args.bfile, validate=args.validate,
                    plot=args.plot, threads=args.threads, seed=args.seed,
                    stmd=args.use_statsmodels, flip=args.flip, pca=args.pca,
                    max_memory=args.maxmem)
    # plink_gwas(args.plinkexe, args.bfile, args.prefix, args.pheno,
    #            args.allele_file,  covs=args.covs, nosex=args.nosex,
    #            threads=args.threads, maxmem=args.maxmem, validate=args.validate,
    #            validsnpsfile=args.validsnpsfile, plot=args.plot)
