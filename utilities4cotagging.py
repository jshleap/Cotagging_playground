#!/usr/bin/env python
# coding:utf-8
"""
  Utilitarian function for most of the cotagging work
  Author:  Jose Sergio Hleap --<2017>
  Purpose: Utilities for cottagging
  Created: 09/30/17
"""
import gc
import operator
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
from dask.distributed import Client
import matplotlib
import numpy as np
import pandas as pd
import psutil
from chest import Chest
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
from numba import jit
from pandas_plink import read_plink
from scipy.stats import linregress
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')
from matplotlib import pyplot as plt


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
    Compute the ld (R2) statistic per window in single population

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
def read_geno(bfile, freq_thresh, threads, flip=False, check=False,
              max_memory=None, usable_snps=None):
    """
    Read the plink bed fileset, restrict to a given frequency (optional,
    freq_thresh), flip the sequence to match the MAF (optional; flip), and check
    if constant variants present (optional; check)

    :param max_memory: Maximum allowed memory
    :param bfile: Prefix of the bed (plink) fileset
    :param freq_thresh: If greater than 0, limit MAF to at least freq_thresh
    :param threads: Number of threads to use in computation
    :param flip: Whether to check for flips and to fix the genotype file
    :param check: Whether to check for constant sites
    :return: Dataframes (bim, fam) and array corresponding to the bed fileset
    """
    # set Cache to protect memory spilling
    if max_memory is not None:
        available_memory = max_memory
    else:
        available_memory = psutil.virtual_memory().available
    cache = Chest(available_memory=available_memory)
    (bim, fam, g) = read_plink(bfile)   # read the files using pandas_plink
    m, n = g.shape                      # get the dimensions of the genotype
    # remove invariant sites
    if check:
        g_std = g.std(axis=1)
        with ProgressBar():
            print('Removing invariant sites')
            with dask.config.set(pool=ThreadPool(threads)):
                idx = (g_std != 0).compute(cache=cache)
        g = g[idx, :]
        bim = bim[idx].copy().reset_index(drop=True)
        bim.i = bim.index.tolist()
        del g_std, idx
        gc.collect()
    if usable_snps is not None:
        idx = bim[bim.snp.isin(usable_snps)].i.tolist()
        g = g[idx, :]
        bim = bim[bim.i.isin(idx)].copy().reset_index(drop=True)
        bim.i = bim.index.tolist()
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
        del flips
        gc.collect()
    else:
        g = g.T
    # Filter MAF
    if freq_thresh > 0:
        print('Filtering MAFs smaller than', freq_thresh)
        print('    Genotype matrix shape before', g.shape)
        assert freq_thresh < 0.5
        good = (mafs < (1 - float(freq_thresh))) & (mafs > float(freq_thresh))
        with ProgressBar():
            with dask.config.set(pool=ThreadPool(threads)):
                good, mafs = dask.compute(good, mafs, cache=cache)
        g = g[:, good]
        print('    Genotype matrix shape after', g.shape)
        print(bim.shape)
        bim = bim[good]
        bim['mafs'] = mafs[good]
        del good
        gc.collect()
    bim = bim.reset_index(drop=True)    # Get the indices in order
    # Fix the i such that it matches the genotype indices
    bim['i'] = bim.index.tolist()
    # Get chunks apropriate with the number of threads
    g = g.rechunk(estimate_chunks(g.shape, threads, memory=available_memory))
    del mafs
    gc.collect()
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
        df = df.reset_index(drop=False)
        with open(picklefile, 'wb') as F:
            pickle.dump((df, beforetail), F)
    try:
        # If causals in the dataframe
        idx = df.dropna(subset=['beta']).index.tolist()
        causals = df.loc[idx, :]

    except KeyError:
        causals = pd.DataFrame([])
        idx = []
    size = df.m_size
    # Plot the scheme with position in x and rank (a.k.a Index) in y
    gwascotag.loc[:, position] = gwascotag.loc[:, position].astype(int)
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
    total = psutil.virtual_memory().available  # a tenth of the memory
    avail_mem = total if memory is None else memory  # Set available memory
    usage = estimate_size(shape) * threads     # Compute threaded estimated size
    # Determine number of chunks given usage and available memory
    n_chunks = np.ceil(usage / avail_mem).astype(int)
    # Mute divided by zero error only for this block of code
    with np.errstate(divide='ignore', invalid='ignore'):
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
def prune_it(df, geno, pheno, label, step=10, threads=1, beta='slope',
             max_memory=None, n=None):
    """
    Prune and score a dataframe of sorted snps

    :param n: Max number of records to prune
    :param max_memory: Maximum available memory
    :param str beta: Column with the effect size
    :param int threads: Number of threads to use
    :param int step: Step of the pruning
    :param str label: Name of the current prunning
    :param pheno: Phenotype array
    :param geno: Genotype array
    :param df: sorted dataframe
    :return: scored dataframe
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
    print('Prunning %s...' % label)
    opts = dict(num_workers=threads, cache=cache, pool=ThreadPool(threads))
    if n is not None:
        print('Just prunning', n)
        tup = (df.iloc[: n], geno, pheno, label, beta)
        delayed_results = [dask.delayed(single_score)(*tup)]
        with ProgressBar(), dask.config.set(**opts):
            res = list(dask.compute(*delayed_results))
    else:
        # Process the first 200 snps at one step regardless of the step passed.
        # This is done to have a finer grain in the first part of the prunning
        # where most of the causals should be captured
        print('First 200')
        # Create a generator with the subset and the arguments for the single
        # function
        gen = ((df.iloc[:i], geno, pheno, label, beta) for i in
               range(1, min(201, df.shape[0] + 1), 1))
        # Run the scoring in parallel threads
        delayed_results = [dask.delayed(single_score)(*i) for i in gen]
        with ProgressBar(), dask.config.set(**opts):
            res = list(dask.compute(*delayed_results))
        print('Processing the rest of variants')
        if df.shape[0] > 200:
            ngen = ((df.iloc[: i], geno, pheno, label) for i in
                    range(201, df.shape[0] + 1, int(step)))
            delayed_results = [dask.delayed(single_score)(*i) for i in ngen]
            with ProgressBar(), dask.config.set(**opts):
                res += list(dask.compute(*delayed_results))
    return pd.DataFrame(res)


# ----------------------------------------------------------------------
def single_window(df, rg, tg, threads=1, max_memory=None, justd=False,
                  extend=False):
    """
    Helper function to compute the correlation between variants from a genotype
    array
    :param df: Merged dataframe mapping of the positions in the genotypes
    :param rg: slice of Genotype array of the reference population
    :param tg: slice of Genotype array of the target population
    :param threads: Number of threads to estimate memory use
    :param max_memory: Memory limit
    :param justd: Return the raw LD matrices insted of its dot product
    :param extend: 'Circularize' the genome by extending both ends
    :return:
    """
    if not df.empty:
        # set Cache to protect memory spilling
        if max_memory is not None:
            available_memory = max_memory
        else:
            available_memory = psutil.virtual_memory().available/2
        cache = Chest(available_memory=available_memory)
        # Make sure chunks make sense
        chunk_opts = dict(threads=threads, memory=available_memory)
        if not isinstance(rg, np.ndarray):
            rg = rg.rechunk(estimate_chunks(shape=rg.shape, **chunk_opts))
            tg = tg.rechunk(estimate_chunks(shape=tg.shape, **chunk_opts))
        # extend the genotype at both end to avoid edge effects
        if extend:
            # get the indices of the subset genotype array
            nidx = np.arange(rg.shape[1])
            # Split the array in half (approximately)
            idx_a, idx_b = np.array_split(nidx, 2)
            # Get the extednded indices
            i = np.concatenate([idx_a[::-1][:-1], nidx, idx_b[::-1][1:]])
            # Re-subset the genotype arrays with the extensions
            rg, tg = rg[:, i], tg[:, i]
            assert rg.shape[1] == tg.shape[1]
            # Compute the correltion as X'X/N
            rho_r = da.dot(rg.T, rg) / rg.shape[0]
            rho_t = da.dot(tg.T, tg) / tg.shape[0]
            # remove the extras
            idx = np.arange(i.shape[0])[idx_a.shape[0]-1: (nidx.shape[0] +
                                                            idx_b.shape[0])]
            rho_r, rho_t = rho_r[idx, :], rho_t[idx, :]
            rho_r, rho_t = rho_r[:, idx], rho_t[:, idx]
            # Make sure the shape match
            assert rho_r.shape[1] == rho_r.shape[0]
            assert rho_t.shape[1] == rho_t.shape[0]
        else:
            # Just compute the correlations
            rho_r = da.dot(rg.T, rg) / rg.shape[0]
            rho_t = da.dot(tg.T, tg) / tg.shape[0]
        if justd:
            # return the raw LD matrices
            return df.snp, rho_r, rho_t
        gc.collect()
        # compute the cotagging/tagging scores
        cot = da.diag(da.dot(rho_r, rho_t))
        ref = da.diag(da.dot(rho_r, rho_r))
        tar = da.diag(da.dot(rho_t, rho_t))
        stacked = da.stack([df.snp, ref, tar, cot], axis=1)
        c_h_u_n_k_s = estimate_chunks(stacked.shape, threads, max_memory)
        stacked = da.rechunk(stacked, chunks=c_h_u_n_k_s)
        columns = ['snp', 'ref', 'tar', 'cotag']
        return dd.from_dask_array(stacked, columns=columns).compute(
            cache=cache)


# ----------------------------------------------------------------------
def window_yielder(rgeno, tgeno, mbim):
    """
    TODO: include in pytest
    :param rgeno: Genotype for reference
    :param tgeno: Genotype from target
    :param mbim: Merged bim files from reference and target
    :return: subsets of genotypes, their indices and subset dataframe
    """
    for window, df in mbim.groupby('windows'):
        if not df.snp.empty:
            # Get the mapping indices
            ridx, tidx = df.i_ref.values, df.i_tar.values
            assert ridx.shape == tidx.shape
            # Subset the genotype arrays
            rg, tg = rgeno[:, ridx], tgeno[:, tidx]
            yield dask.delayed(rg), dask.delayed(tg), ridx, tidx, \
                  dask.delayed(df)


# ----------------------------------------------------------------------
def get_ld(rgeno, rbim, tgeno, tbim, kbwindow=1000, threads=1, max_memory=None,
           justd=False, extend=False):
    """
    Get the LD blocks from snp overlap between two populations

    :param rgeno: Genotype array of the reference populartion
    :param rbim: Mapping variant info and the genotype array position for ref
    :param tgeno: Genotype array of the target populartion
    :param tbim: Mapping variant info and the genotype array position for tar
    :param kbwindow: Size of the window in KB
    :param threads: Number of threads to use for computation
    :param max_memory: Memory limit
    :param justd: Return only the raw LD matrices or the tagging/cotagging
    :param extend: 'Circularize' the genome by extending both ends
    :return: A list of tuples (or dataframe if not justd) with the ld per block
    """
    # # Set CPU limits
    # soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    # resource.setrlimit(resource.RLIMIT_NPROC, (threads, hard))
    # soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # print('Soft limit changed to :', soft)

    # set Cache to protect memory spilling
    rp = 'r.pickle'
    if os.path.isfile(rp):
        with open(rp, 'rb') as pckl:
            r = pickle.load(pckl)
    else:
        if max_memory is not None:
            available_memory = max_memory
        else:
            available_memory = psutil.virtual_memory().available
        cache = Chest(available_memory=available_memory)
        if os.path.isfile('ld.matrix'):
            print('Loading precomputed LD matrix')
            r = dd.read_parquet('ld.matrix')
        else:
            print('Computing LD score per window')
            # Get the overlapping snps and their info
            shared = ['chrom', 'snp', 'pos']
            mbim = rbim.merge(tbim, on=shared, suffixes=['_ref', '_tar'])
            assert mbim.i_ref.values.shape == mbim.i_tar.values.shape
            # Get the number of bins or loci to be computed
            nbins = np.ceil(max(mbim.pos)/(kbwindow * 1000)).astype(int)
            # Get the limits of the loci
            bins = np.linspace(0, max(mbim.pos) + 1, num=nbins, endpoint=True,
                               dtype=int)
            if bins.shape[0] == 1:
                # Fix the special case in which the window is much bigger than
                # the range
                bins = np.append(bins, kbwindow * 1000)
            # Get the proper intervals into the dataframe
            mbim['windows'] = pd.cut(mbim['pos'], bins, include_lowest=True)
            # Compute each locus in parallel
            dask_rgeno = dask.delayed(rgeno)
            dask_tgeno = dask.delayed(tgeno)
            delayed_results = [
                dask.delayed(single_window)(df, rg, tg, threads, max_memory,
                                            justd, extend) for
                rg, tg, ridx, tidx, df in
                window_yielder(dask_rgeno, dask_tgeno, mbim)]
            opts = dict(num_workers=threads, cache=cache,
                        pool=ThreadPool(threads))
            with ProgressBar(), dask.config.set(**opts), open(rp, 'wb') as pck:
                r = tuple(dask.compute(*delayed_results))
                pickle.dump(r, pck)
    r = tuple(x for x in r if x is not None)
    if justd:
        return r
    r = pd.concat(r)
    dd.to_parquet(r, 'ld.matrix')
    return r


# ----------------------------------------------------------------------
def compute_maf(column, keep_allele_order=False):
    c = Counter(column)
    if keep_allele_order:
        k = sorted(c.keys(), reverse=False)[:2]
        maf = sum(c[i] for i in k) / (sum(c.values()) * 2)
    else:
        maf = sum(sorted(c.values(), reverse=False)[:2]) / (sum(c.values()) * 2)
    return maf


# ----------------------------------------------------------------------
def optimize_it(loci, ld_range, by_range, h2, avh2, n, threads, cache, memory,
                sum_stats, test_geno, test_pheno, clump_function,
                select_index_by='pvalue', clump_with='d_reference',
                do_locus_ese=False, normalize=True):
    """
    Optimize the R2 based on summary statistics

    :param normalize: Whether to normalize the genotype or not
    :param do_locus_ese: Clump with ESE strategy instead of pvalue
    :param clump_with: LD matrix to clump with (d_reference, t_reference)
    :param clump_function: Function to clump with.
    :param loci: List of tuples with the LDs and snps per locus
    :param ld_range: Range of ld thresholds
    :param by_range: Range of the ranking strategy (pvalue or ese)
    :param h2: Heritability of the trait
    :param avh2: Average heritability per snp
    :param n: Sample size
    :param threads: Number of threads to use in multithread computations
    :param cache: A dictionary that spills to disk. chest instance
    :param memory: Maximum memory to use
    :param sum_stats: Dataframe with the sumary statistics of an association
    :param test_geno: Test set genotype array
    :param test_pheno: Test set genotype vector (or series)
    :param select_index_by: Ranking strategy (ESE or Pvalue)
    :return: Tuple with the list of index snps and their R2
    """
    if select_index_by == 'pvalue':
        rank = getattr(operator, 'lt')
        snp_index = 0
    else:
        rank = getattr(operator, 'gt')
        if select_index_by == 'ese':
            snp_index = 2
        else:
            snp_index = 4
    curr_best = ([], 0)
    # Optimize with one split, return reference score with the second
    out = train_test_split(test_geno, test_pheno, test_size=0.5)
    train_g, test_g, train_p, test_p = out
    if normalize:
        # re-normalize the genotypes
        train_g = (train_g - train_g.mean(axis=0)) / train_g.std(axis=0)
        test_g = (test_g - test_g.mean(axis=0)) / test_g.std(axis=0)
    opt_dic = {}  # for debugging purposes
    for ld_threshold in ld_range:
        all_clumps = clump_function(loci, sum_stats, ld_threshold, h2, avh2, n,
                                    threads, cache, memory, select_index_by,
                                    clump_with, do_locus_ese)
        if by_range is None:
            by_range = pd.concat(all_clumps.values()).ese.quantile(
                np.arange(.0, 1, .1))
        for by_threshold in by_range:
            index_snps = [k[snp_index] for k in all_clumps.keys() if
                          rank(k[snp_index + 1], by_threshold)]
            if not index_snps:
                continue
            try:
                r2 = just_score(index_snps, sum_stats, train_p, train_g)
            except Exception:
                with open('failed.pickle', 'wb') as F:
                    pickle.dump((index_snps, sum_stats, train_p, train_g), F)
                    raise
            opt_dic[ld_threshold] = (r2, index_snps)
            if r2 > curr_best[1]:
                curr_best = (index_snps, r2, pd.concat(all_clumps.values()))
    r2 = just_score(curr_best[0], sum_stats, test_p, test_g)
    return curr_best[0], r2, curr_best[-1]


# ----------------------------------------------------------------------
def run_optimization_by(by_range, sort_by, loci, h2, m, n, threads, cache,
                        sum_stats, available_memory, test_geno, test_pheno,
                        tpheno, tgeno, prefix, clump_function,
                        select_index_by='pvalue', normalize=True,
                        clump_with='d_reference', do_locus_ese=False):
    """
    Run the optimzation of byrange and select by ranges

    :param normalize: Whether to normalize the genotype or not
    :param clump_function: Function to clump with.
    :param by_range: Range to optimize the R2 during P+T strategy
    :param sort_by: Name of variable to sort SNPs between clumps (set of index)
    :param loci: List of tuples with the LDs and snps per locus
    :param h2: Heritability of the trait
    :param m: Number of snps
    :param n: Number of individuals
    :param threads: Number of threads to use in parallelization
    :param cache: A chest dictionary to spill to disk if not enough memory
    :param sum_stats: Dataframe with sumary statistics of the association
    :param available_memory: Max available memory for the program
    :param test_geno: Dask array with test subset genotype of the reference
    :param test_pheno: Dataframe with test subset phenotype of the reference
    :param tpheno: Dask array with target genotype
    :param tgeno: Dataframe with target phenotype
    :param prefix: Prefix for the outputs
    :param select_index_by: Name of variable (column) to do index snp selection
    :param clump_with: LD matrix to clump with (d_reference, t_reference)
    :param do_locus_ese: Clump with ESE strategy instead of pvalue
    :return: Dictionary with index snps, tagged_snpsand the R2 for target and
    reference
    """
    avh2 = h2 / m
    ld_range = np.concatenate([np.array([0.1]), np.arange(.2, .8, .2)])
    if by_range is None and (sort_by == 'pvalue'):
            by_range = [1, 0.5, 0.05, 10E-3, 10E-5, 10E-7, 1E-9]
    opt = dict(loci=loci, ld_range=ld_range, by_range=by_range, h2=h2,
               avh2=avh2, n=n, threads=threads, cache=cache,
               memory=available_memory, sum_stats=sum_stats,
               test_geno=test_geno, test_pheno=test_pheno,
               select_index_by=select_index_by, clump_with=clump_with,
               do_locus_ese=do_locus_ese, normalize=normalize,
               clump_function=clump_function)
    r2_tuple = optimize_it(**opt)
    index_snps, r2_ref, sum_stats = r2_tuple
    with open('%s.index_snps' % select_index_by, 'w') as F:
        F.write('\n'.join(index_snps))
    # score in target
    r2 = just_score(index_snps, sum_stats, tpheno, tgeno)
    ascending = True if sort_by == 'pvalue' else False
    pre = sum_stats[sum_stats.snp.isin(index_snps)].sort_values(
        sort_by, ascending=ascending)
    pos = sum_stats[~sum_stats.snp.isin(index_snps)].sort_values(
        sort_by, ascending=ascending)
    pd.concat([pre, pos]).to_csv('%s_full.tsv' % prefix, index=False, sep='\t')
    result = dict(index_snps=pre, tagged_snps=pos, R2=r2, R2_ref=r2_ref)
    with open('result_%s.pickle' % sort_by, 'wb') as P:
        pickle.dump(result, P)
    return result


# ----------------------------------------------------------------------
def just_score(index_snp, sumstats, pheno, geno):
    clump = sumstats[sumstats.snp.isin(index_snp)]
    idx = clump.i.values.astype(int)
    prs = geno[:, idx].dot(clump.slope)
    est = np.corrcoef(prs, pheno.PHENO)[1, 0] ** 2
    return est

