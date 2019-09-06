#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<2019>
  Purpose: Run transethnic metaanalysis varying sample size
  Created: 21/04/19
  Requires pyrs in path
"""
import matplotlib
import scipy
import pandas as pd
import numpy as np
from graph_clump import main as pyrs
from utilities4cotagging import read_geno, just_score
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
plt.style.use('ggplot')
np.seterr(all='raise')

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
        try:
            merged['SE%s' % s] = np.vectorize(get_se)(merged['BETA%s' % s],
                                                      merged['STAT%s' % s])
            merged['SD%s' % s] = np.vectorize(get_std)(merged['SE%s' % s],
                                                       merged['NMISS%s' % s])
        except TypeError:
            print(merged.head())
            raise
    return merged


def scores(mean_bar, comb_se):
    low_lim = mean_bar - 1.96 * comb_se
    hig_lim = mean_bar + 1.96 * comb_se
    z_scores = mean_bar / comb_se
    p_values = pd.Series(scipy.stats.norm.sf(abs(z_scores)) * 2)
    return low_lim, hig_lim, p_values


def fixed_effects(source_b, target_b, source_v, target_v):
    try:
        w_source = 1 / source_v
    except ZeroDivisionError:
        w_source = np.repeat(0, source_v.shape[0])
    try:
        w_target = 1 / target_v
    except ZeroDivisionError:
        w_target = np.repeat(0, target_v.shape[0])
    v_dot = 1 / (w_source + w_target)
    comb_se = np.sqrt(v_dot)
    mean_bar = ((source_b * w_source) + (target_b * w_target)) / v_dot
    low_lim, hig_lim, p_values = scores(mean_bar, comb_se)
    return pd.concat([mean_bar, low_lim, hig_lim, p_values], ignore_index=True,
                     names=['BETA_meta', '95% CI low', '95% CI high', 'P'])


def get_betarandom(q, c, df):
    if q > df:
        return (q - df) / c
    else:
        return 0


def random_effects(source_b, target_b, source_v, target_v):
    df = 1 # for two pop
    try:
        w_source = 1 / source_v
    except FloatingPointError:
        w_source = np.repeat(0, source_v.shape[0])
    try:
        w_target = 1 / target_v
    except FloatingPointError:
        w_target = np.repeat(0, target_v.shape[0])
    sumofwei = w_source + w_target
    sqsumofw = w_source**2 + w_target**2
    c = sumofwei - (sqsumofw/sumofwei)
    w_sqsumb = (w_source * source_b**2) + (w_target * target_b**2)
    w_sumbsq = ((w_source * source_b) + (w_target * target_b))**2
    q = w_sqsumb * (w_sumbsq / sumofwei)
    tau_squared = np.vectorize(get_betarandom)(q, c, df)
    try:
        w_sourcestar = 1 / (source_v + tau_squared)
    except FloatingPointError:
        w_sourcestar = np.repeat(0, source_v.shape[0])
    try:
        w_targetstar = 1 / (target_v + tau_squared)
    except FloatingPointError:
        w_targetstar = np.repeat(0, target_v.shape[0])
    v_star = 1 / (w_sourcestar + w_targetstar)
    mean_barstar = ((source_b * w_sourcestar) + (target_b * w_targetstar))
    mean_barstar /= v_star
    comb_se = np.sqrt(v_star)
    low_lim, hig_lim, p_values = scores(mean_barstar, comb_se)
    df = pd.concat([mean_barstar, low_lim, hig_lim, p_values],
                   ignore_index=True, axis=1).rename(columns=dict(
        zip(range(4), ['slope', '95% CI low', '95% CI high', 'pvalue'])))
    return df


def adjust_vars_in_one(frac_n, ori_n, merged_df, suffix):
    betas = merged_df['BETA%s' % suffix]
    sigma = merged_df['SD%s' % suffix]
    noise = (((sigma**2) * (ori_n - 1)) / (ori_n - frac_n - 1))
    try:
        eta_s = np.random.normal(scale=(noise - sigma))
        b_mod = betas + eta_s
    except ValueError:
        noise = b_mod = np.repeat(0, sigma.shape[0])
    return b_mod, noise

    
def compute_one_n(source_n, target_n, ori_source, ori_target, merged_df,
                  suffixes):
    cols = dict(zip(range(8),['chr', 'snp', 'bp', 'a1', 'slope', '95% CI low',
                              '95% CI high', 'pvalue']))
    target, source = suffixes
    p = source_n/ori_source
    chr = merged_df.CHR
    snps = merged_df.SNP
    bp = merged_df['BP%s' % source]
    a1 = merged_df['A1%s' % source]
    if p == 1:
        source_b = merged_df['BETA%s' % source]
        #source_se = merged_df['SD%s' % source]
        #low_lim, hig_lim, p_values = scores(source_b, source_se)
        low_lim = pd.Series(np.nan, index=merged_df.index)
        hig_lim = pd.Series(np.nan, index=merged_df.index)
        p_values = merged_df['P%s' % source]
        data = [chr, snps, bp, a1, source_b, low_lim, hig_lim, p_values]
    elif p == 0:
        target_b = merged_df['BETA%s' % target]
        target_se = merged_df['SD%s' % target]
        #low_lim, hig_lim, p_values = scores(target_b, target_se)
        low_lim = pd.Series(np.nan, index=merged_df.index)
        hig_lim = pd.Series(np.nan, index=merged_df.index)
        p_values = merged_df['P%s' % target]
        data = [chr, snps, bp, a1, target_b, low_lim, hig_lim, p_values]
    else:
        source_b, source_v = adjust_vars_in_one(source_n, ori_source,
                                                merged_df, source)
        target_b, target_v = adjust_vars_in_one(target_n, ori_target,
                                                merged_df, target)
        r_meta_gwas = random_effects(source_b, target_b, source_v, target_v)
        f_meta_gwas = fixed_effects(source_b, target_b, source_v, target_v)
        return r_meta_gwas, f_meta_gwas
    r_meta_gwas = f_meta_gwas = pd.concat(data, ignore_index=True, axis=1
                                          ).rename(columns=cols)
    return r_meta_gwas, f_meta_gwas


def plot_it(intended_labels, df, outprefix):
    source_label, target_label = intended_labels
    time = '%s (%)' % source_label
    value = r"$R^2$"
    f, ax = plt.subplots()
    sns.tsplot(time=time, value=value, unit="run", data=df, ax=ax,
               condition='Pop', ci=[25, 50, 75, 95])
    plt.tight_layout()
    plt.savefig('%s.pdf' % outprefix)
    plt.close()


def main(geno_prefix, source_gwas, target_gwas, labels, outprefix,
         source_gwas_n, pheno, threads=8, unintended_tuples=None,
         ld_range=None, pval_range=None, freq_thr=0.01, index_snps=None):
    # unintended_tuples are a list of tuples with: 1) name of population,
    # 2) bed fileset prefix, and 3) file with phenotype
    run = 0 # intended to extend to simulations later on
    (bim, fam, geno) = read_geno(geno_prefix, freq_thr, threads)
    suffixes = tuple('_%s' % x for x in labels)
    merged = read_gwas(source_gwas, target_gwas, suffixes=suffixes)
    df_rand = []
    df_fixe = []
    space = np.linspace(0, fam.shape[0], 10)
    for target_n in tqdm(space, total=len(space)):
        source_n = source_gwas_n - target_n
        percentage = source_n/source_gwas_n
        r_meta_gwas, f_meta_gwas = compute_one_n(
            source_n, fam.shape[0], source_gwas_n, fam.shape[0], merged,
            suffixes)
        opt = dict(
            geno=(bim, fam, geno), pheno=pheno, outpref=outprefix,
            pval_range=pval_range, ld_range=ld_range, gwas=r_meta_gwas,
            threads=threads, freq_thresh=freq_thr, covs=None, memory=None,
            validate=3, check=None, snp_subset=None
        )
        r2_rand, ppt_rand = pyrs(**opt)

        # PRS((bim, fam, geno), r_meta_gwas, ld_range=ld_range,
        #                 pval_range=pval_range, pheno=pheno,  threads=threads)
        # r2_rand = ppt_rand.compute_prs()
        ppt_fixe = PRS((bim, fam, geno), f_meta_gwas, ld_range=ld_range,
                       pval_range=pval_range, pheno=pheno, threads=threads)
        r2_rand = ppt_rand.compute_prs()
        r2_fixe = ppt_fixe.compute_prs()
        best_rand = ppt_rand.best
        best_fixe = ppt_fixe.best
        df_rand.append((labels[1], r2_rand, percentage, run))
        df_fixe.append((labels[1], r2_fixe, percentage, run))
        # if pheno is not None and index_snps is None:
        #     assert geno.shape[0] == pheno.shape[0]
        #     # optimize ppt
        #     ppt_rand.optimize_it(geno, pheno)
        #     ppt_fixe.optimize_it(geno, pheno)
        #     best_rand = ppt_rand.best
        #     best_fixe = ppt_fixe.best
        #     df_rand.append((labels[1], best_rand.r2, percentage, run))
        #     df_fixe.append((labels[1], best_fixe.r2, percentage, run))
        for tup in unintended_tuples:
            (c_bim, c_fam, c_geno) = read_geno(tup[1], freq_thr, threads)
            c_pheno = pd.read_csv(tup[2], blocksize=25e6,
                                  delim_whitespace=True)
            r2_rand = just_score(best_rand.indices, r_meta_gwas, c_pheno,
                                 c_geno)
            r2_fixe = just_score(best_fixe.indices, f_meta_gwas, c_pheno,
                                 c_geno)
            df_rand.append((tup[0], r2_rand, percentage, run))
            df_fixe.append((tup[0], r2_fixe, percentage, run))
    df_rand = pd.DataFrame(df_rand, columns=['%s (%)' % labels[0],  r'$R^2$',
                                             'Pop', 'run'])
    df_fixe = pd.DataFrame(df_fixe, columns=['%s (%)' % labels[0], r'$R^2$',
                                             'Pop', 'run'])
    plot_it(labels, df_rand, '%s_meta_randomeffects' % outprefix)
    plot_it(labels, df_fixe, '%s_meta_fixedeffects' % outprefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='meta.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('geno', help='Genotype file (bed filename)')
    parser.add_argument('source_gwas', help='GWAS of source, plink format')
    parser.add_argument('target_gwas', help='GWAS of target, plink format')
    parser.add_argument('labels', help='Labels of source, target populations',
                        nargs=2)
    parser.add_argument('outprefix', help='prefix for outputs')
    parser.add_argument('source_gwas_n', help='Number of individuals in source',
                        type=int)
    parser.add_argument('-f', '--pheno', default=None,
                        help='Tuple with Phenotype files of the training sets')
    parser.add_argument('-t', '--threads', default=1, type=int)
    parser.add_argument('-u', '--unintended_pops', default=None, type=tuple,
                        nargs='*', help='Tuple with label, bed fileset prefix '
                                        'and phenotype filename')
    parser.add_argument('-p', '--pval_range', default=None,
                        help='Range of pvalues to explore')
    parser.add_argument('-r', '--ld_range', default=None, help='Range of R2 to'
                                                               ' explore')

    parser.add_argument('--f_thr', type=float, default=0,
                        help='Keyword argument for read_geno. The frequency '
                             'threshold to cleanup the genotype file')
    parser.add_argument('-i', '--indices', default=None, nargs='*',
                        help='Index snps')

    args = parser.parse_args()
    main(args.geno, args.source_gwas, args.target_gwas, args.labels,
         args.outprefix, args.source_gwas_n, pheno=args.pheno,
         threads=args.threads, unintended_tuples=args.unintended_pops,
         ld_range=args.ld_range, pval_range=args.pval_range,
         freq_thr=args.f_thr, index_snps=args.indices)






