#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<2017>
  Purpose: Utilities for cottagging
  Created: 09/30/17
"""
import os
import pickle
from collections import ChainMap
from collections import Counter
from functools import reduce
from multiprocessing.pool import ThreadPool
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

# Numbafy linregress (makes it a bit faster)
lr = jit(linregress)


# ----------------------------------------------------------------------
def execute_line(line):
    """
    Execute line with subprocess

    :param str line: Line to be executed in shell
    """
    pl = Popen(line, shell=True, stderr=PIPE, stdout=PIPE)
    o, e = pl.communicate()
    return o, e


# ----------------------------------------------------------------------
@jit
def single_block(geno, df, block):
    """
    Process a single window of genotype to infer R2

    :param da.core.Array geno: Genotype array (dask array)
    :param pd..core.frame.DataFrame df: Subset of the bim file in dataframe form
    :param block: Number of the block
    :return: dictionary with block as ke
    """
    idx = df.i.tolist()  # Get the positions in the genotype array
    sub = geno[:, idx]   # Subset the genotype array
    assert sub.shape[1] == len(idx)  # make sure the subset is performed
    # Compute the squared correlation in dask dataframe form
    r2 = dd.from_dask_array(sub, columns=df.snp.tolist()).corr() ** 2
    return {block: r2}


# ----------------------------------------------------------------------
def blocked_r2(bim, geno, kb_window=1000, threads=1):
    """
    Compute the ld (R2) statistic per window
    :param bim: bim file from a plink bed fileset
    :param geno: genotypic array (dask) as read from pandas_plink
    :param kb_window: size of the window in kb
    :param threads: number of threads to be used
    :return: bim dataframe with the bloacks and an r2 dictionary
    """
    assert bim.shape[0] == geno.shape[1]  # Make sure the dimensions fit
    # Get the bins by the window size
    bins = np.arange(bim.pos.min(), bim.pos.max() + 1, kb_window * 1000,
                     dtype=int)
    # Cut the dataframe into apropriate bins
    bim['block'] = pd.cut(bim['pos'], bins, include_lowest=True)
    # Include the last interval by replacing the NA created by cut with the
    # apropriate interval
    fill = pd.Interval(bim.iloc[-2].block.right, bim.pos.max())
    bim.block = bim.block.astype(object)
    bim = bim.fillna(fill)
    # Run the individual blocks in parallel
    r = Parallel(n_jobs=int(threads))(
        delayed(single_block)(geno, df, block) for block, df in
        bim.groupby('block'))
    r2 = dict(ChainMap(*r))  # Convert to dictionary
    return bim, r2


# ----------------------------------------------------------------------
def norm(array, a=0, b=1):
    """
    Normalize an array between a and b

    :param np.ndarray array: array to be normalized
    :param float a: Lower limit of the normalization
    :param floar b: Upper limit of the normalization
    """
    tiny = 1E-20
    # make sure is an array
    array = np.array(array, dtype=float)
    # normilize 0-1
    rang = max(array) - min(array)
    c = (array - min(array)) / (rang + tiny)
    # scale
    range2 = b - a
    return (c * range2) + a


# ---------------------------------------------------------------------------
def read_geno(bfile, freq_thresh, threads, flip=False, check=False):
    """
    Read the plink bed fileset, restrict to a given frequency (optional,
    freq_thresh), flip the sequence to match the MAF (optional; flip), and check
    if constant variants present (optional; check)

    :param bfile: Prefix of the bed (plink) fileset
    :param freq_thresh: If greater than 0, limit MAF to at least freq_thresh
    :param threads: Number of threads to use in computation
    :param flip: Whether to check for flips and to fix the genotype file
    :param check: Wheather to check for constant sites
    :return: Dataframes (bim, fam) and array corresponding to the bed fileset
    """
    (bim, fam, g) = read_plink(bfile)   # read the files using pandas_plink
    m, n = g.shape                      # get the dimensions of the genotype
    # remove invariant sites
    if check:
        g_std = g.std(axis=1)
        with ProgressBar():
            print('Removing invariant sites')
            with dask.set_options(pool=ThreadPool(threads)):
                idx = (g_std != 0).compute()
        g = g[idx, :]
        bim = bim[idx].copy()
    # compute the mafs if required
    mafs = g.sum(axis=1) / (2 * n) if flip or freq_thresh > 0 else None
    if flip:
        # check possible flips
        flips = np.zeros(bim.shape[0], dtype=bool)
        flips[np.where(mafs > 0.5)[0]] = True
        bim['flip'] = flips
        vec = np.zeros(flips.shape[0])
        vec[flips] = 2
        # perform the flipping
        g = abs(g.T - vec)
    else:
        g = g.T
    # Filter MAF
    if freq_thresh > 0:
        print('Filtering MAFs smaller than', freq_thresh)
        print('    Genotype matrix shape before', g.shape)
        assert freq_thresh < 0.5
        good = (mafs < (1 - float(freq_thresh))) & (mafs > float(freq_thresh))
        with ProgressBar():
            with dask.set_options(pool=ThreadPool(threads)):
                good, mafs = dask.compute(good, mafs)
        g = g[:, good]
        bim = bim[good]
        bim['mafs'] = mafs[good]
        print('    Genotype matrix shape after', g.shape)
    bim = bim.reset_index(drop=True)    # Get the indices in order
    # Fix the i such that it matches the genotype indices
    bim['i'] = bim.index.tolist()
    return bim, fam, g


# ---------------------------------------------------------------------------
def smartcotagsort(prefix, gwascotag, column='Cotagging', ascending=False,
                   title=None, beta='slope', position='pos'):
    """
    perform a 'clumping' based on Cotagging score, but retain all the rest in
    the last part of the dataframe

    :param str position: Label of column with the bp position
    :param srt beta: Name of the column with the effect size
    :param str title: Title of the figure
    :param bool ascending: Order of the sort, by default reversed.
    :param str prefix: prefix for outputs
    :param pd.core.frame.DataFrame gwascotag: merged dataframe of cotag and gwas
    :param str column: name of the column to be sorted by in the cotag file
    """
    # if done before, load it
    picklefile = '%s_%s.pickle' % (prefix, ''.join(column.split()))
    if os.path.isfile(picklefile):
        with open(picklefile, 'rb') as F:
            df, beforetail = pickle.load(F)
    else:
        print('Sorting File based on %s "clumping"...' % column)
        # Add the slope as normed size for plotting
        gwascotag.loc[:, 'm_size'] = norm(abs(gwascotag.loc[:, beta]), 10, 150)
        # Get groups in column (pseudo clump) from a sorted dataframe and retain
        # the first occurence per group
        grouped = gwascotag.sort_values(by=column, ascending=ascending).groupby(
            column, as_index=False, sort=False).first()
        # re-sort the resulting dataframe
        sorteddf = grouped.sort_values(by=column, ascending=ascending)
        # Add the rest of lines at the end
        try:
            tail = gwascotag[~gwascotag.snp.isin(sorteddf.snp)]
        except AttributeError:
            tail = gwascotag[~gwascotag.SNP.isin(sorteddf.SNP)]
        beforetail = sorteddf.shape[0]
        df = sorteddf.copy()  # Work on a copy of the sorted dataframe
        if not tail.empty:
            # Include the tail lines in a random order
            df = df.append(tail.sample(frac=1), ignore_index=True)
        df = df.reset_index()
        with open(picklefile, 'wb') as F:
            pickle.dump((df, beforetail), F)
    try:
        # If causals in the dataframe
        idx = df.dropna(subset=['beta']).index.tolist()
        causals = df.loc[idx, :]

    except KeyError:
        causals = pd.DataFrame([])
    size = df.m_size
    # Plot the scheme with position in x and rank (a.k.a Index) in y
    gwascotag.loc[:,position] = gwascotag.loc[:,position].astype(int)
    f, ax = plt.subplots()
    df.plot.scatter(x=position, y='index', ax=ax, label=column)
    if not causals.empty:
        causals.plot.scatter(x=position, y='index', marker='*', c='k', ax=ax,
                             s=size[idx].values, label='Causals')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig('%s_%s.pdf' % (prefix, '_'.join(column.split())))
    plt.close()
    return df, beforetail


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
    """
    Estimate the appropriate chunks to split arrays in the dask format to made
    them fit in memory. If Memory is None, it will be set to a tenth of the
    total memory. It also takes into account the number of threads

    :param tuple shape: Shape of the array to be chunkenized
    :param threads: Number of threads intended to be used
    :param memory: Memory limit
    :return: The appropriate chunk in tuple form
    """
    total = psutil.virtual_memory().available / 1E7  # a tenth of the memory
    avail_mem = total if memory is None else memory  # Set available memory
    usage = estimate_size(shape) * threads     # Compute threaded estimated size
    # Determine number of chunks given usage and available memory
    n_chunks = np.ceil(usage / avail_mem).astype(int)
    # Mute divided by zero error only for this block of code
    with np.errstate(divide='ignore',invalid='ignore'):
        estimated = tuple(np.array(shape) / n_chunks)  # Get chunk estimation
    chunks = min(shape, tuple(estimated))            # Fix if n_chunks is 0
    return tuple(int(i) for i in chunks)  # Assure chunks is a tuple of integers


# ----------------------------------------------------------------------
@jit
def single_score(subdf, geno, pheno, label, beta='slope'):
    """
    Execute single score per subset of snps prunned. This is a
    helper to parallelized part of prune_it function.

    :param beta: Column with the effect size
    :param subdf: Subset of summary statistics dataframe
    :param geno: genotype array
    :param pheno: True phenotype
    :param label: Name of the strategy being prunned
    :return:
    """
    # Get the indices of the genotype array corresponding to the subset
    indices = subdf.i.tolist()
    # Generate the PRS
    prs = geno[:, indices].dot(subdf.loc[:, beta]).astype(float)
    # Rapid estimation of R2 with the true phenotype
    regress = np.corrcoef(prs, pheno.PHENO)[1, 0] ** 2
    return {'Number of SNPs': subdf.shape[0], 'R2': regress, 'type': label}


# ----------------------------------------------------------------------
def prune_it(df, geno, pheno, label, step=10, threads=1, beta='slope'):
    """
    Prune and score a dataframe of sorted snps

    :param str beta: Column with the effect size
    :param int threads: Number of threads to use
    :param int step: Step of the pruning
    :param str label: Name of the current prunning
    :param pheno: Phenotype array
    :param geno: Genotype array
    :param df: sorted dataframe
    :return: scored dataframe
    """
    print('Prunning %s...' % label)
    print('First 200')
    gen = ((df.iloc[:i], geno, pheno, label, beta) for i in
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
@jit(parallel=True)
def single_window(df, rgeno, tgeno, threads=1, max_memory=None, justd=False,
                  extend=False):
    ridx = df.i_ref.values
    tidx = df.i_tar.values
    rg = rgeno[:, ridx]
    tg = tgeno[:, tidx]
    if extend:
        # extend the Genotpe at both end to avoid edge effects
        ridx_a, ridx_b = np.array_split(ridx, 2)
        tidx_a, tidx_b = np.array_split(tidx, 2)
        rg = da.concatenate([rgeno[:, (ridx_a[::-1][:-1])], rg,
                             rgeno[:, (ridx_b[::-1][1:])]], axis=1)
        tg = da.concatenate([tgeno[:, (tidx_a[::-1][:-1])], tg,
                             tgeno[:, (tidx_b[::-1][1:])]], axis=1)
        rho_r = da.dot(rg.T, rg) / rg.shape[0]
        rho_t = da.dot(tg.T, tg) / tg.shape[0]
        # remove the extras
        rho_r = rho_r[:, (ridx_a.shape[0] + 1):][:, :(ridx.shape[0])]
        rho_r = rho_r[(ridx_a.shape[0] + 1):, :][:(ridx.shape[0]), :]
        rho_t = rho_t[:, (tidx_a.shape[0] + 1):][:, :(tidx.shape[0])]
        rho_t = rho_t[(tidx_a.shape[0] + 1):, :][:(tidx.shape[0]), :]
        assert rho_r.shape[1] == ridx.shape[0]
        assert rho_t.shape[1] == tidx.shape[0]
    else:
        rho_r = da.dot(rg.T, rg) / rg.shape[0]
        rho_t = da.dot(tg.T, tg) / tg.shape[0]
    if justd:
        return df.snp, rho_r, rho_t
    cot = da.diag(da.dot(rho_r, rho_t))
    ref = da.diag(da.dot(rho_r, rho_r))
    tar = da.diag(da.dot(rho_t, rho_t))
    stacked = da.stack([df.snp, ref, tar, cot], axis=1)
    c_h_u_n_k_s = estimate_chunks(stacked.shape, threads, max_memory)
    stacked = da.rechunk(stacked, chunks=c_h_u_n_k_s)
    columns = ['snp', 'ref', 'tar', 'cotag']
    return dd.from_dask_array(stacked, columns=columns).compute()


# ----------------------------------------------------------------------
def compute_maf(dask_column, keep_allele_order=False):
    c = Counter(dask_column)
    if keep_allele_order:
        k = sorted(c.keys(), reverse=False)[:2]
        maf = sum(c[i] for i in k) / (sum(c.values()) * 2)
    else:
        maf = sum(sorted(c.values(), reverse=False)[:2]) / (sum(c.values()) * 2)
    return maf
