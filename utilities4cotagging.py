#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<>
  Purpose: Utilities for cottagging
  Created: 09/30/17
"""
import mmap
import os
import pickle
import tarfile
from collections import ChainMap
from functools import reduce
from glob import glob
from subprocess import Popen, PIPE
import dask
import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
from numba import jit
from pandas_plink import read_plink
from scipy.stats import linregress
from multiprocessing.pool import ThreadPool

lr = jit(linregress)
# ----------------------------------------------------------------------
def mapcount(filename):
    """
    Efficient line counter courtesy of Ryan Ginstrom answer in stack overflow
    """
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines


# ---------------------------------------------------------------------------
def read_log(prefix):
    """
    Read logfile with the profiles written

    :param prefix: prefix of the logfile
    :return: List of files written
    """
    l = []
    with open('%s.log' % prefix) as F:
        for line in F:
            if 'profile written' not in line:
                continue
            else:
                l.append(line.split()[0])
    return l


# ----------------------------------------------------------------------
def executeLine(line):
    """
    Execute line with subprocess
    
    :param str line: Line to be executed in shell
    """
    pl = Popen(line, shell=True, stderr=PIPE, stdout=PIPE)
    o, e = pl.communicate()
    return o, e


# ----------------------------------------------------------------------
def read_BimFam(prefix):
    """
    Read a bim/fam files from the plink fileset
    :param str prefix: prefix of the plink bedfileset
    """
    Bnames = ['CHR', 'SNP', 'cM', 'BP', 'A1', 'A2']
    bim = pd.read_table('%s.bim' % (prefix), delim_whitespace=True, header=None,
                        names=Bnames)
    Fnames = ['FID', 'IID', 'father', 'mother', 'Sex', 'Phenotype']
    fam = pd.read_table('%s.fam' % (prefix), delim_whitespace=True, header=None,
                        names=Bnames)
    return bim, fam


# ----------------------------------------------------------------------
def read_freq(bfile, plinkexe, freq_threshold=0.1, maxmem=1700, threads=1):
    """
    Generate and read frequency files and filter based on threshold
    
    :param str bfile: prefix of the plink bedfileset
    :param str plinkexe: path to plink executable
    :param float freq_threshold: Lower threshold to filter MAF by
    :param int maxmem: Maximum allowed memory
    :param int threads: Maximum number of threads to use

    """
    high = 1 - freq_threshold
    low = freq_threshold
    if not os.path.isfile('%s.frq.gz' % bfile):
        nname = os.path.split(bfile)[-1]
        frq = ('%s --bfile %s --freq gz --keep-allele-order --out %s --memory '
               '%d --threads %d')
        line = frq % (plinkexe, bfile, nname, maxmem, threads)
        o, e = executeLine(line)
        frq = pd.read_table('%s.frq.gz' % nname, delim_whitespace=True)
    else:
        frq = pd.read_table('%s.frq.gz' % bfile, delim_whitespace=True)
        # filter MAFs greater than 1 - freq_threshold and smaller than freq_threshold
    return frq[(frq.MAF < high) & (frq.MAF > low)]



# ----------------------------------------------------------------------
@jit
def single_block(geno, df, block):
    idx = df.i.tolist()
    sub = geno[:, idx]
    assert sub.shape[1] == len(idx)
    r2 = dd.from_dask_array(sub, columns=df.snp.tolist()).corr() ** 2
    return {block: r2}



# ----------------------------------------------------------------------
def blocked_R2(bim, geno, kbwindow=1000, threads=1):
    # subset = lambda x, d: x[:, d.i.tolist()]
    assert bim.shape[0] == geno.shape[1]
    nbins = np.ceil(max(bim.pos) / (kbwindow * 1000)).astype(int)
    bins = np.linspace(0, max(bim.pos) + 1, num=nbins, endpoint=True, dtype=int)
    bim['block'] = pd.cut(bim['pos'], bins, include_lowest=True)
    # delayed_results = [dask.delayed(single_block)(df, geno) for block, df in
    #                    bim.groupby('block')]
    r = Parallel(n_jobs=int(threads))(
        delayed(single_block)(geno, df, block) for block, df in
        bim.groupby('block'))
    #r = list(dask.compute(*delayed_results, num_workers=threads))
    R2 = dict(ChainMap(*r))#dd.concat(r, interleave_partitions=True)
    return bim, R2


# # ----------------------------------------------------------------------
# def blocked_R2(bim, geno, kbwindow):
#     window = kbwindow * 1000
#     assert bim.shape[0] == geno.shape[1]
#     # bim['pos_kb'] = bim.pos #/ 1000
#     ma = bim.pos.max()
#     mi = bim.pos.min()
#     r2_block = {}
#     curr = 0
#     for i, b in enumerate(np.arange(mi, ma, window, dtype=int)):
#         boole = (bim.pos < b + window - 1) & (bim.pos > curr)
#         bim.loc[boole, 'block'] = i
#         curr = bim.pos[boole].max()
#         idx = bim.i[boole].tolist()
#         sub = geno[:, idx]
#         assert sub.shape[1] == len(idx)
#         r2_block[i] = dd.from_dask_array(sub, columns=bim.snp[boole].tolist()
#                                 betafile         ).corr()**2
#     return bim, r2_block


# ----------------------------------------------------------------------
def train_test_gen_only(prefix, bfile, plinkexe, splits=10, maxmem=1700,
                        threads=1):
    """
    Generate a list of individuals for training and a list for validation.
    The list is to be passed to plink. It will take one split as validation and
    the rest as training.

    :param str prefix: profix of output
    :param str bfile: prefix for the bed fileset
    :param str plinkexe: path to plink executable
    :param int splits: Number of splits to be done
    """
    fam = pd.read_table('%s.fam' % bfile, delim_whitespace=True, header=None,
                        names=['FID', 'IID', 'a', 'b', 'c', 'd'])
    fold = int(np.ceil(fam.shape[0] / splits))
    msk = fam.IID.isin(fam.IID.sample(n=fold))
    train, test = '%s_train' % prefix, '%s_test' % prefix
    opts = dict(header=False, index=False, sep=' ')
    fam.loc[~msk, ['FID', 'IID']].to_csv('%s.keep' % train, **opts)
    fam.loc[msk, ['FID', 'IID']].to_csv('%s.keep' % test, **opts)
    make_bed = ('%s --bfile %s --keep %s.keep --make-bed --out %s --memory %d '
                '--threads %d')
    for i in [train, test]:
        executeLine(make_bed % (plinkexe, bfile, i, i, maxmem, threads))
    return train, test


# ----------------------------------------------------------------------
def train_test(prefix, bfile, plinkexe, pheno, splits=10, maxmem=1700,
               threads=1):
    """
    Generate a list of individuals for training and a list for validation.
    The list is to be passed to plink. It will take one split as validation and
    the rest as training.
    
    :param str prefix: profix of output
    :param str bfile: prefix for the bed fileset
    :param str plinkexe: path to plink executable
    :param int splits: Number of splits to be done
    """
    pheno = read_pheno(pheno)
    # trainthresh = (splits - 1) / splits
    fn = os.path.split(bfile)[-1]
    keeps = {
        '%s_train' % prefix: (os.path.join(os.getcwd(), '%s_train.keep' % fn),
                              os.path.join(os.getcwd(),
                                           '%s_train.pheno' % fn)
                              ),
        '%s_test' % prefix: (os.path.join(os.getcwd(), '%s_test.keep' % fn),
                             os.path.join(os.getcwd(), '%s_test.pheno' % fn))}
    fam = pd.read_table('%s.fam' % bfile, delim_whitespace=True, header=None,
                        names=['FID', 'IID', 'a', 'b', 'c', 'd'])
    fold = int(np.ceil(fam.shape[0] / splits))
    # msk = np.random.rand(len(fam)) < trainthresh
    msk = fam.IID.isin(fam.IID.sample(n=fold))
    opts = dict(header=False, index=False, sep=' ')
    fam.loc[~msk, ['FID', 'IID']].to_csv(keeps['%s_train' % prefix][0], **opts)
    pheno.loc[~msk, ['FID', 'IID', 'Pheno']].to_csv(keeps['%s_train' % prefix][1
                                                    ], **opts)
    fam.loc[msk, ['FID', 'IID']].to_csv(keeps['%s_test' % prefix][0], **opts)
    pheno.loc[msk, ['FID', 'IID', 'Pheno']].to_csv(keeps['%s_test' % prefix][1],
                                                   **opts)
    make_bed = ('%s --bfile %s --keep %s --make-bed --out %s --memory %d '
                '--threads %d -pheno %s')
    for k, v in keeps.items():
        executeLine(make_bed % (plinkexe, bfile, v[0], k, maxmem, threads, v[1])
                    )
    return keeps


# ----------------------------------------------------------------------
def norm(array, a=0, b=1):
    '''
    normalize an array between a and b
    '''
    # make sure is an array
    array = np.array(array, dtype=float)
    ## normilize 0-1
    rang = max(array) - min(array)
    A = (array - min(array)) / rang
    ## scale
    range2 = b - a
    return (A * range2) + a


# ----------------------------------------------------------------------
def read_pheno(pheno):
    """
    Read a phenotype file with plink profile format
    
    :param str pheno: Filename of phenotype
    """
    if 'FID' in open(pheno).readline():
        ## asumes that has 3 columns with the first two with headers FID adn
        ## IID
        pheno = pd.read_table(pheno, delim_whitespace=True)
        pheno.rename(columns={pheno.columns[-1]: 'Pheno'}, inplace=True)
    else:
        Pnames = ['FID', 'IID', 'Pheno']
        pheno = pd.read_table(pheno, delim_whitespace=True, header=None,
                              names=Pnames)
    return pheno


# ---------------------------------------------------------------------------
def parse_sort_clump(fn, allsnps):
    """
    Parse and sort clumped file
    
    :param str fn: clump file name
    :param :class pd.Series allsnps: Series with all snps being analyzed
    """
    # make sure allsnps is a series
    allsnps = pd.Series(allsnps)
    try:
        df = pd.read_table(fn, delim_whitespace=True)
    except FileNotFoundError:
        spl = fn.split('.')
        if spl[0] == '':
            idx = 1
        else:
            idx = 0
        fn = '.'.join(np.array(spl)[[idx, 1 + idx, -1]])
        if idx == 1:
            fn = '.%s' % fn
        df = pd.read_table(fn, delim_whitespace=True)
    SNPs = df.loc[:, 'SP2']
    tail = [x.split('(')[0] for y in SNPs for x in y.split(',') if x.split('(')[
        0] != 'NONE']
    full = pd.DataFrame(df.SNP.tolist() + tail, columns=['SNP'])
    full = full[full.SNP.isin(allsnps)]
    rest = allsnps[~allsnps.isin(full.SNP)]
    df = pd.concat((full.SNP, rest)).reset_index(drop=False)
    df.rename(columns={'index': 'Index'}, inplace=True)
    return df


# ---------------------------------------------------------------------------
def helper_smartsort(grouped, key):
    """
    helper function to parallelize smartcotagsort
    """
    df = grouped.get_group(key)
    head = df.loc[df.index[0], :]
    tail = df.loc[df.index[1:], :]
    return head, tail


# ---------------------------------------------------------------------------
def helper_smartsort2(grouped, key):
    """
    helper function to parallelize smartcotagsort
    """
    df = grouped.get_group(key)
    return df.loc[df.index[0], :]


# ---------------------------------------------------------------------------
def read_geno(bfile, freq_thresh, threads, flip=False, check=False):

    (bim, fam, G) = read_plink(bfile)
    m, n = G.shape
    # remove invariant sites
    if check:
        # remove constant variants
        G_std = G.std(axis=1)  #
        with ProgressBar():
            print('Removing invariant sites')
            with dask.set_options(pool=ThreadPool(threads)):
                idx = (G_std != 0).compute()
        G = G[idx, :]
        bim = bim[idx].copy()
    mafs = G.sum(axis=1) / (2 * n)
    if flip:
        # check possible flips
        flips = np.zeros(bim.shape[0], dtype=bool)
        flips[np.where(mafs > 0.5)[0]] = True
        bim['flip'] = flips
        vec = np.zeros(flips.shape[0])
        vec[flips] = 2
        # perform the flipping
        G = abs(G.T - vec)
    else:
        G = G.T
    # Filter MAF
    if freq_thresh > 0:
        print('Filtering MAFs smaller than', freq_thresh)
        print('    Genotype matrix shape before', G.shape)
        good = (mafs < (1 - float(freq_thresh))) & (mafs > float(freq_thresh))
        with ProgressBar():
            with dask.set_options(pool=ThreadPool(threads)):
                good, mafs = dask.compute(good, mafs)
                # good = good.compute(num_workers=threads)
        G = G[:, good]
        bim = bim[good]
        bim['mafs'] = mafs[good]
        print('    Genotype matrix shape after', G.shape)
    bim = bim.reset_index(drop=True)
    bim['i'] = bim.index.tolist()
    return bim, fam, G


# ---------------------------------------------------------------------------
def smartcotagsort(prefix, gwascotag, column='Cotagging', ascending=False,
                   title=None):
    """
    perform a 'clumping' based on Cotagging score, but retain all the rest in 
    the last part of the dataframe
    
    :param str prefix: prefix for outputs
    :param :class pd.DataFrame gwascotag: merged dataframe of cotag and gwas
    :param str column: name of the column to be sorted by in the cotag file
    """
    picklefile = '%s_%s.pickle' % (prefix, ''.join(column.split()))
    if os.path.isfile(picklefile):
        with open(picklefile, 'rb') as F:
            df, beforetail = pickle.load(F)
    else:
        print('Sorting File based on %s "clumping"...' % column)
        gwascotag.loc[:, 'm_size'] = norm(abs(gwascotag.slope),  10, 150)
        grouped = gwascotag.sort_values(by=column, ascending=ascending).groupby(
            column, as_index=False, sort=False).first()
        sorteddf = grouped.sort_values(by=column, ascending=ascending)
        tail = gwascotag[~gwascotag.snp.isin(sorteddf.snp)]
        beforetail = sorteddf.shape[0]
        df = sorteddf.copy()
        if not tail.empty:
            df = df.append(tail.sample(frac=1), ignore_index=True)
        df = df.reset_index(drop=True)
        df['index'] = df.index.tolist()
        with open(picklefile, 'wb') as F:
            pickle.dump((df, beforetail), F)
    idx = df.dropna(subset=['beta']).index.tolist()
    size = df.m_size
    f, ax = plt.subplots()
    df.plot.scatter(x='pos', y='index', ax=ax, label=column)
    df.loc[idx, :].plot.scatter(x='pos', y='index', marker='*', c='k', ax=ax,
                                s=size[idx].values, label='Causals')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig('%s_%s.pdf' % (prefix, '_'.join(column.split())))
    plt.close()
    return df, beforetail


# ---------------------------------------------------------------------------
def set_first_step(nsnps, step, init_step=2, every=False):
    """
    Define the range starting by adding one snp up the the first step
    
    :param int nsnps: Total number of snps
    :param float step: step for the snp range
    """
    # Fool proofing
    if nsnps < 20:
        print('Really? running with less than 20 snps? I am setting your step '
              'to 1, and making every equals True')
        every = True
        step = 1
        init_step = 1
    onesnp = 100. / float(nsnps)
    if every:
        full = np.arange(onesnp, 100 + onesnp, onesnp)
    else:
        # just include the first 5% snps in step of init_step
        initial = np.arange(onesnp, (nsnps * 0.05 * onesnp) + onesnp,
                            (init_step * onesnp))
        rest = np.arange(initial[-1] + onesnp, 100 + step, step)
        full = np.concatenate((initial, rest))
    if full[-1] < 100:
        full[-1] = 100
    return full


# ----------------------------------------------------------------------
def gen_qrange(prefix, nsnps, prunestep, every=False, qrangefn=None):
    """
    Generate qrange file to be used with plink qrange
    :param prefix: prefix for output
    :param nsnps: Total number of SNPs being analyzed
    :param prunestep: percentage to test at a time
    :param every: test one snp at a time
    :param qrangefn: Name for a pre-ran rangefile
    """
    order = ['label', 'Min', 'Max']
    # dtype = {'label': object, 'Min': float, 'Max': float}
    if qrangefn is None:
        # Define the number of snps per percentage point and generate the range
        percentages = set_first_step(nsnps, prunestep, every=every)
        snps = np.around((percentages * nsnps) / 100).astype(int)
        try:
            # Check if there are repeats in ths set of SNPS
            assert sorted(snps) == sorted(set(snps))
        except AssertionError:
            snps = ((percentages * nsnps) / 100).astype(int)
            assert sorted(snps) == sorted(set(snps))
        labels = ['%.2f' % x for x in percentages]
        if float(labels[-1]) > 100.:
            labels[-1] = '100.00'
        if snps[-1] != nsnps:
            snps[-1] = nsnps
        assert snps[-1] == nsnps
        assert labels[-1] == '100.00'
        # Generate the qrange file
        qrange = '%s.qrange' % prefix
        qr = pd.DataFrame({'label': labels, 'Min': np.zeros(len(percentages)),
                           'Max': snps}).loc[:, order]
        qr.to_csv(qrange, header=False, index=False, sep=' ')
    else:
        qrange = qrangefn
        qr = pd.read_csv(qrange, sep=' ', header=None,
                         names=order)  # , dtype=dtype)
    return qr, qrange


# ---------------------------------------------------------------------------
def read_scored_qr(profilefn, phenofile, alpha, nsnps, score_type='sum'):
    """
    Read the profile file a.k.a. PRS file or scoresum
    
    :param str profilefn: Filename of profile being read
    :param str phenofile: Filename of phenotype
    :param int nsnps: Number of snps that produced the profile
    """
    if score_type == 'sum':
        col = 'SCORESUM'
    else:
        col = 'SCORE'
    # Read the profile
    sc = pd.read_table(profilefn, delim_whitespace=True)
    # Read the phenotype file
    pheno = pd.read_table(phenofile, delim_whitespace=True, header=None, names=[
        'FID', 'IID', 'pheno'])
    # Merge the two dataframes
    sc = sc.merge(pheno, on=['FID', 'IID'])
    # Compute the linear regression between the score and the phenotype
    lr = linregress(sc.pheno, sc.loc[:, col])
    # Return results in form of dictionary
    dic = {'File': profilefn, 'alpha': alpha, 'R2': lr.rvalue ** 2,
           'SNP kept': nsnps}
    return dic


# ---------------------------------------------------------------------------
def qrscore(plinkexe, bfile, scorefile, qrange, qfile, phenofile, ou, qr,
            maxmem, threads, label, prefix, allele_file, normalized_geno=True):
    """
    Score using qrange

    :param qr: Dataframe with qrange information
    :param str ou: output prefix for scoring
    :param str prefix: Prefix for outputs
    :param bool normalized_geno: If normalizing the genotype is required
    :param str label: Label of population being analyzed
    :param str allele_file: Filename with allelic info. VarID pos 3, A1 pos 2
    :param int maxmem: Maximum allowed memory
    :param int threads: Maximum number of threads to use
    :param str plinkexe: Path and executable of plink
    :param str bfile: Prefix of plink-bed fileset
    :param str scorefile: File with the summary statistics in plink format
    :param str qrange: File with the ranges to be passed to the --q-score-range
    :param str phenofile: Filename with the phenotype
    """
    # Score files with the new ranking score = ('%s --bfile %s --score %s 2 4
    #  7 header --q-score-range %s %s ' '--allow-no-sex --keep-allele-order
    # --pheno %s --out %s --memory ' '%d --threads %d')
    if normalized_geno:
        sc_type = 'sum'
    else:
        sc_type = ''
    score = ('%s --bfile %s --score %s --q-score-range %s %s --allow-no-sex '
             '--a1-allele %s 3 2 --pheno %s --out %s --memory %d --threads %d')
    score = score % (plinkexe, bfile, '%s %s' % (scorefile, sc_type), qrange,
                     qfile, allele_file, phenofile, ou, maxmem, threads)
    _, _ = executeLine(score)
    # Get the results in dataframe
    profs_written = read_log(ou)
    df = pd.DataFrame([read_scored_qr('%s.%.2f.profile' % (ou, float(x.label)),
                                      phenofile, label, x.Max, sc_type) if
                       ('%s.%.2f.profile' % (ou, float(x.label)) in
                        profs_written) else {}
                       for x in qr.itertuples()]).dropna()
    # Cleanup
    try:
        label = label if isinstance(label, str) else '%.2f' % label
        tarfn = 'Profiles_%s_%s.tar.gz' % (prefix, label)
    except TypeError:
        tarfn = 'Profiles_%s.tar.gz' % prefix
    with tarfile.open(tarfn, mode='w:gz') as t:
        for fn in glob('%s*.profile' % ou):
            if os.path.isfile(fn):
                try:
                    # it has a weird behaviour
                    os.remove(fn)
                    t.add(fn)
                except:
                    pass
    return df


# ----------------------------------------------------------------------
def estimate_size(shape):
    """
    Estimate the potential size of an array
    :param shape: shape of the resulting array
    :return: size in Mb
    """
    total_bytes = reduce(np.multiply, shape) * 8
    return total_bytes / 1E6


# ----------------------------------------------------------------------
def estimate_chunks(shape, threads, memory=None):
    total = psutil.virtual_memory().available / 1E7 # a tenth of the memory
    avail_mem = total if memory is None else memory
    usage = estimate_size(shape) * threads
    if usage < avail_mem:
        return shape
    else:
        n_chunks = np.floor(usage / avail_mem).astype(int)
        return tuple([int(max([1, i])) for i in np.array(shape) / n_chunks])


# ----------------------------------------------------------------------
def nearestPD(A, threads=1):
    """
    Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2] from Ahmed Fasih

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    isPD = lambda x: da.all(np.linalg.eigvals(x) > 0).compute()
    B = (A + A.T) / 2
    _, s, V = da.linalg.svd(B)
    H = da.dot(V.T, da.dot(da.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3
    spacing = da.spacing(da.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    eye_chunk = estimate_chunks((A.shape[0], A.shape[0]), threads=threads)[0]
    I = da.eye(A.shape[0], chunks=eye_chunk)
    k = 1
    while not isPD(A3):
        mineig = da.min(da.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
    return A3


# ----------------------------------------------------------------------
def single_score_plink(prefix, qr, tup, plinkexe, gwasfn, qrange, frac_snps,
                 maxmem, threads):
    """
    Helper function to paralellize score_qfiles
    """
    qfile, phenofile, bfile = tup
    suf = qfile[qfile.find('_') + 1: qfile.rfind('.')]
    ou = '%s_%s' % (prefix, suf)
    # score = ('%s --bfile %s --score %s 2 4 7 header --q-score-range %s %s '
    #          '--allow-no-sex --keep-allele-order --pheno %s --out %s '
    #          '--memory %d --threads %d')
    score = (
        '%s --bfile %s --score %s sum --q-score-range %s %s --allow-no-sex '
        '--keep-allele-order --pheno %s --out %s --memory %d --threads %d')
    score = score % (plinkexe, bfile, gwasfn, qrange, qfile, phenofile, ou,
                     maxmem, threads)
    o, e = executeLine(score)
    profs = read_log(ou)
    df = pd.DataFrame([read_scored_qr('%s.%s.profile' % (ou, x.label),
                                      phenofile, suf, round(float(x.label)
                                                            * frac_snps), profs)
                       for x in qr.itertuples()])
    # frames.append(df)
    with tarfile.open('Profiles_%s.tar.gz' % ou, mode='w:gz') as t:
        for fn in glob('%s*.profile' % ou):
            if os.path.isfile(fn):
                t.add(fn)
                os.remove(fn)
    return df


# ----------------------------------------------------------------------
@jit
def single_score(subdf, geno, pheno, label):
    indices = subdf.i.tolist()#gen_index.tolist()
    prs = geno[:, indices].dot(subdf.slope)
    regress = np.corrcoef(prs, pheno.PHENO)[1, 0] ** 2 #lr(pheno.PHENO, prs)
    return {'Number of SNPs': subdf.shape[0], 'R2': regress**2,#.rvalue **2,
            'type': label}


# ----------------------------------------------------------------------
def prune_it(df, geno, pheno, label, step=10, threads=1):
    """
    Prune and score a dataframe of sorted snps

    :param pheno: Phenotype array
    :param geno: Genotype array
    :param random: randomize the snps
    :param df: sorted dataframe
    :return: scored dataframe
    """
    print('Prunning %s...' % label)
    print('First 200')
    gen = ((df.iloc[:i], geno, pheno, label) for i in
           range(1, min(201, df.shape[0] + 1), 1))
    delayed_results = [dask.delayed(single_score)(*i) for i in gen]
    with ProgressBar():
        res = list(dask.compute(*delayed_results, num_workers=threads))
    # process the first two hundred every 2
    print('Processing the rest of variants')
    if df.shape[0] > 200:
        ngen = ((df.iloc[: i], geno, pheno, label) for i in
                range(201, df.shape[0] + 1, int(step)))
        delayed_results = [dask.delayed(single_score)(*i) for i in ngen]
        with ProgressBar():
            res += list(dask.compute(*delayed_results, num_workers=threads))
    return pd.DataFrame(res)


# ----------------------------------------------------------------------
@jit
def single_window(df, rgeno, tgeno, threads=1, max_memory=None, justd=False):
    ridx = df.i_ref.values
    tidx = df.i_tar.values
    rg = rgeno[:, ridx]
    tg = tgeno[:, tidx]
    D_r = da.dot(rg.T, rg) / rg.shape[0]
    D_t = da.dot(tg.T, tg) / tg.shape[0]
    if justd:
        return df.snp, D_r, D_t
    cot = da.diag(da.dot(D_r, D_t))
    ref = da.diag(da.dot(D_r, D_r))
    tar = da.diag(da.dot(D_t, D_t))
    stacked = da.stack([df.snp, ref, tar, cot], axis=1)
    c_h_u_n_k_s = estimate_chunks(stacked.shape, threads, max_memory)
    stacked = da.rechunk(stacked, chunks=c_h_u_n_k_s)
    columns=['snp', 'ref', 'tar', 'cotag']
    return dd.from_dask_array(stacked, columns=columns).compute()


# ----------------------------------------------------------------------
def large_pickle(obj, file_path):
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    data = bytearray(n_bytes)
    bytes_out = pickle.dumps(obj)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])



# ----------------------------------------------------------------------
def large_unpickle(file_path):
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj


# ----------------------------------------------------------------------
def get_ld(rgeno, rbim, tgeno, tbim, kbwindow=1000, threads=1, max_memory=None,
           justd=False):
    # pickle_file = 'ld_perwindow.pickle'
    # if os.path.isfile(pickle_file):
    #     r = large_unpickle(pickle_file)
    # else:
    print('Computing LD score per window')
    mbim = rbim.merge(tbim, on=['chrom', 'snp', 'pos'],
                      suffixes=['_ref', '_tar'])
    nbins = np.ceil(max(mbim.pos)/(kbwindow * 1000)).astype(int)
    bins = np.linspace(0, max(mbim.pos) + 1, num=nbins, endpoint=True, dtype=int
                       )
    mbim['windows'] = pd.cut(mbim['pos'], bins, include_lowest=True)
    delayed_results = [
        dask.delayed(single_window)(df, rgeno, tgeno, threads, max_memory,
                                    justd) for window, df in
        mbim.groupby('windows')]
    with ProgressBar():
        r = list(dask.compute(*delayed_results, num_workers=threads))
        # large_pickle(r, pickle_file)
    if justd:
        return r
    r = pd.concat(r)
    return r
