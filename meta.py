#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<2019>
  Purpose: Run transethnic metaanalysis varying sample size
  Created: 21/04/19
"""
import scipy
import pandas as pd
import numpy as np


header_names=['CHR', 'SNP', 'BP', 'A1', 'TEST', 'NMISS', 'BETA', 'STAT', 'P']


def get_se(x, y):
    return x/abs(y)


def get_std(x, y):
    return x * np.sqrt(y)

def get_weigtedb(x,y):
    return (1/x) * (y**2)


def read_gwas(gwas1, gwas2, suffixes=('_EUR', '_ASN'), names=header_names):
    df1 = pd.read_csv(gwas1, sep='\t', names=names)
    df2 = pd.read_csv(gwas2, sep='\t', names=names)
    merged = df1.merge(df2, on=['CHR', 'SNP'], suffixes=suffixes)
    for s in suffixes:
        merged['SE%s' % s] = np.vectorize(get_se)(merged['BETA%s' % s],
                                                  merged['STAT%s' % s])
        merged['SD%s' % s] = np.vectorize(get_std)(merged['SE%s' % s],
                                                   merged['NMISS%s' % s])
    return merged


def scores(mean_bar, comb_se):
    low_lim = mean_bar - 1.96 * comb_se
    hig_lim = mean_bar + 1.96 * comb_se
    z_scores = mean_bar / comb_se
    p_values = scipy.stats.norm.sf(abs(z_scores)) * 2
    return low_lim, hig_lim, p_values


def fixed_effects(source_b, target_b, source_v, target_v):
    w_source = 1 / source_v
    w_target = 1 / target_v
    v_dot = 1 / (w_source + w_target)
    comb_se = np.sqrt(v_dot)
    mean_bar = ((source_b * w_source) + (target_b * w_target)) / v_dot
    low_lim, hig_lim, p_values = scores(mean_bar, comb_se)
    return pd.concat([mean_bar, low_lim, hig_lim, p_values], ignore_index=True,
                     name=['BETA_meta', '95% CI low', '95% CI high', 'P'])


def get_betarandom(q, c):
    if q > 1:
        return (q - 1) / c
    else:
        return 0

def random_effects(source_b, target_b, source_v, target_v):
    df = 1 # for two pop
    w_source = 1 / source_v
    w_target = 1 / target_v
    sumofwei = sum(w_source, w_target)
    sqsumofw = (w_source**2, w_target**2)
    c = sumofwei - (sqsumofw/sumofwei)
    w_sqsumb = (w_source * source_b**2) + (w_target * target_b**2)
    w_sumbsq = ((w_source * source_b) + (w_target * target_b))**2
    q = w_sqsumb * (w_sumbsq / sumofwei)
    tau_squared = np.vectorize(get_betarandom)(q, c)
    w_sourcestar = 1 / (source_v + tau_squared)
    w_targetstar = 1 / (target_v + tau_squared)
    v_star = 1 / (w_sourcestar + w_targetstar)
    mean_barstar = ((source_b * w_sourcestar) + (target_b * w_targetstar))
    mean_barstar /= v_star
    comb_se = np.sqrt(v_star)
    low_lim, hig_lim, p_values = scores(mean_barstar, comb_se)
    return pd.concat([mean_bar, low_lim, hig_lim, p_values], ignore_index=True,
                     name=['BETA_meta', '95% CI low', '95% CI high', 'P'])


def adjust_vars_in_one(frac_n, ori_n, merged_df, suffix):
    betas = merged_df['BETA%s' % suffix]
    sigma = merged_df['SD%s' % suffix]
    noise = (((sigma**2) * (ori_n - 1)) / (ori_n - frac_n -1))
    eta_s = np.random.normal(scale=(noise-sigma))
    b_mod = betas + eta_s
    return b_mod, noise

    
def compute_one_n(source_n, max_n, merged_df, suffixes):
    source, target = suffixes
    target_n = max_n - source_n
    source_b, source_v = adjust_vars_in_one(source_n, max_n, merged_df, source)
    target_b, target_v = adjust_vars_in_one(target_n, max_n, merged_df, target)
    r_meta_gwas = random_effects(source_b, target_b, source_v, target_v)
    r_fixe_gwas = fixed_effects(source_b, target_b, source_v, target_v)
    return r_meta_gwas, r_fixe_gwas


def ppt():
    opts = dict(by_range=None, sort_by='pvalue', loci=loci, h2=h2, m=m,
                threads=threads, cache=cache, sum_stats=sum_stats, n=n,
                available_memory=available_memory, test_geno=X_test,
                test_pheno=y_test, tpheno=tpheno, tgeno=tgeno,
                prefix='%s_pval_all' % prefix, select_index_by='pvalue',
                clump_with=clump_with, do_locus_ese=False,
                normalize=kwargs['normalize'],
                clump_function=compute_clumps
                )
def main(source_gwas, target_gwas, labels):
    suffixes = tuple('_%s' % x for x in labels)
    merged = read_gwas(source_gwas, target_gwas, suffixes=suffixes)
    max_n = merged.NMISS.max()
    for source_n in np.linspace(0, max_n, 10):
        r_meta_gwas, r_fixe_gwas = compute_one_n(source_n, max_n, merged_df,
                                                 suffixes)
        # do PPT here

    
    
    # 1. Read each individual association file
    # 2. Compute the variance based on the SE
    # 3. feed "sample corrected betas" to the meta analysis functions
    # 4. compute PRS with PPT
    # 5. assess and output plot?
    
    
    







