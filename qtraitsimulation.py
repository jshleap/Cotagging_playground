#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<>
  Purpose: Simulate quantitative phenotypes given 
  Created: 09/30/17
"""
import argparse
import dask.array as da
import matplotlib
from pandas_plink import read_plink

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')
from utilities4cotagging import *
from numba import double
from numba.decorators import jit


# ----------------------------------------------------------------------

def get_SNP_dist(bfile, causals):
    """
    compute the distance between snps (for now only for uniformly distributed)
    :param str bfile: prefix of the plink bedfileset
    :param list causals: Causal variants to simulate
    """
    bim = pd.read_table('%s.bim' % bfile, delim_whitespace=True, header=None,
                        names=['CHR', 'SNP', 'cm', 'BP', 'A1', 'A2'])
    bim = bim[bim.SNP.isin(causals)]
    return int(np.around(bim.BP.diff().mean() / 1000))


# ----------------------------------------------------------------------
def true_prs(prefix, bfile, h2, ncausal, normalize=False, bfile2=None,
             f_thr=0.1, seed=None, causaleff=None, uniform=False, usepi=False,
             snps=None, threads=1, flip=False, max_memory=None, check=False):
    """
    Generate TRUE causal effects and the genetic effect equal to the h2 (a.k.a
    setting Vp = 1)

    :param prefix: Prefix of outputs
    :param bfile: Plink 1.9 binary fileset prefix for the first (or only) pop
    :param h2: Desired heritability
    :param ncausal: Number of causal variants
    :param normalize: Whether or not to normalize the genotype
    :param bfile2: Second population (optional)
    :param seed: Random seed
    :param causaleff: Dataframe with previos causal effects
    :param uniform: Whether to sample causal snps uniformingly or randomly
    :param snps: list of snps to subset the casuals to
    :return: genetic matrix, bim and fam dataframes and the causal vector
    """
    # set random seed
    seed = np.random.randint(1e4) if seed is None else seed
    print('using seed %d' % seed)
    np.random.seed(seed=seed)
    snps2 = None
    # get indices of second pop if needed
    if bfile2 is not None:
        # merge the bim files of tw populations to use common snps
        (bim2, fam2, G2) = read_geno(bfile2, f_thr, threads, check=check)
        snps2 = bim2.snp
        del bim2, fam2, G2
    # read rhe genotype files
    (bim, fam, G) = read_geno(bfile, f_thr, threads, flip=flip, check=check)
    if snps2 is not None:
        print('Filtering current population with second set')
        print('Genotype matrix shape before', G.shape)
        # subset the genotype file
        indices = bim[bim.snp.isin(snps2)].i
        G = G[:, indices.tolist()]
        bim = bim[bim.i.isin(indices)].reset_index(drop=True)
        bim['i'] = bim.index.tolist()
        print('Genotype matrix shape after', G.shape)
    # get MAFs
    m, n = G.shape

    pi = G.var(axis=0).mean() if usepi else 1
    # Normalize G to variance 1 and mean 0 if required
    if normalize:
        print('Normalizing genotype to variance 1 and mean 0')
        # G = (G.T - G.mean(axis=1)) / G.std(axis=1)
        G = (G - G.mean(axis=0)) / G.std(axis=0)
    # else:
    #     # Transpose G so is n x m
    #     G = G.transpose()
    # Set some local variables
    allele = '%s.alleles' % prefix
    totalsnps = '%s.totalsnps' % prefix
    allsnps = G.shape[1]
    h2_snp = h2 / (ncausal*pi)
    std = np.sqrt(h2_snp)
    # write the possible snps and the allele file
    par = dict(sep=' ', index=False, header=False)
    bim.snp.to_csv(totalsnps, **par)
    bim.loc[:, ['snp', 'a0']].to_csv(allele, **par)
    # Get causal mutation indices randomly distributed
    if ncausal > allsnps:
        print('More causals than available snps. Setting it to %d' % allsnps)
        ncausal = allsnps
    if causaleff is not None:
        print('using causal effect')
        cols = ['snp', 'beta']
        causals = bim[bim.snp.isin(causaleff.snp)].copy()
        pre_beta = causaleff.beta
        c = cols if 'beta' in bim else 'snp'
        causals = causals.merge(causaleff.reindex(columns=cols), on=c)
        print(causals.head())
    elif uniform:
        idx = np.linspace(0, bim.shape[0] - 1, num=ncausal, dtype=int)
        causals = bim.iloc[idx].copy()
        # making sure is a copy of the DF
        # causals = bim[bim.snp.isin(causals)].copy()
        av_dist = (np.around(causals.pos.diff().mean() / 1000)).astype(int)
        print('Causal SNPs are %d kbp apart on average' % av_dist)
    elif snps is None:
        causals = bim.sample(ncausal, replace=False, random_state=seed).copy()
    else:
        causals = bim[bim.snp.isin(snps)].copy()
    # If causal effects are provided use them, otherwise get them
    if causaleff is None:
        # chunks = estimate_chunks((ncausal,), threads)
        pre_beta = np.repeat(std, ncausal)#np.random.normal(loc=0, scale=std, size=ncausal)
        # Store them
        causals['beta'] = pre_beta  # .compute()
        causals = causals.dropna()
    nc = causals.reindex(columns=['snp', 'beta'])
    bidx = bim[bim.snp.isin(nc.snp)].index.tolist()
    bim = bim.reindex(columns=['chrom', 'snp', 'cm', 'pos', 'a0', 'a1', 'i',
                               'mafs', 'flip'])
    bim.loc[bidx, 'beta'] = nc.beta.values.tolist()
    print(bim.dropna(subset=['beta']).head())
    idx = bim.dropna(subset=['beta']).i.values
    # Score
    assert np.allclose(sorted(causals.beta.values), sorted(pre_beta))
    g_eff = G[:, idx].dot(causals.beta).compute(num_workers=threads)

    if causaleff is not None:
        assert sorted(bim.dropna(subset=['beta']).snp) == sorted(causaleff.snp)
    fam['gen_eff'] = g_eff
    print('Variance in beta is', bim.beta.var())
    print('Variance of genetic component', g_eff.var())
    print(bim.head())
    # write full table
    fam.to_csv('%s.full' % prefix, sep=' ', index=False)
    return G, bim, fam, causals.beta


# ----------------------------------------------------------------------
def TruePRS(outprefix, bfile, h2, ncausal, plinkexe, snps=None, frq=None,
            causaleff=None, bfile2=None, freqthreshold=0.1, maxmem=1700,
            threads=1, seed=None, uniform=False):
    """
    Generate TRUE causal effects and PRS. Also, subset the SNPs based on
    frequency


    ??

    :param seed: Random seed for the random functions
    :param threads: Number of threads to use in plink
    :param outprefix: Prefix for outputs
    :param bfile: prefix of the plink bedfileset
    :param h2: Desired heritability
    :param ncausal: Number of causal variants to simulate
    :param plinkexe: path to plink executable
    :param snps: Series with the names of causal snps
    :param frq: DataFrame with the MAF frequencies
    :param causaleff: DataFrame with the true causal effects
    :param bfile2: prefix of the plink bedfileset on a second population
    :param float freqthreshold: Lower threshold to filter MAF by
    :param uniform: pick uniformingly distributed causal variants
    """
    # set the seed
    seed = np.random.randint(1e4) if seed is None else seed
    print('using seed %d' % seed)
    np.random.seed(seed=seed)
    # Get the per snp heritability
    h2_snp = h2 / ncausal
    # Set some local variables
    maf = 'MAF'
    a1 = 'A1'
    allele = '%s.alleles' % outprefix
    totalsnps = '%s.totalsnps' % outprefix
    if not os.path.isfile('%s.full' % outprefix):
        # Read freq file
        if frq is None:
            frq = read_freq(bfile, plinkexe, freq_threshold=freqthreshold)
            if bfile2 is not None:
                frq = frq.merge(read_freq(bfile2, plinkexe,
                                          freq_threshold=freqthreshold),
                                on=['CHR', 'SNP'])
                maf = 'MAF_x'
                a1 = 'A1_x'
        allsnps = frq.shape[0]
        print('Total Number of variants available: %d' % allsnps)
        frq.SNP.to_csv(totalsnps, sep=' ', header=False, index=False)
        frq.to_csv(allele, sep=' ', header=False, index=False)
        # Get causal mutation indices randomly distributed
        if ncausal > allsnps:
            print('More causals than available snps. Setting it to %d' %
                  allsnps)
            ncausal = allsnps
        if causaleff is not None:
            cols = ['SNP', 'eff', 'norm', 'beta']
            causals = frq[frq.SNP.isin(causaleff.SNP)]
            causals = causals.merge(causaleff.loc[:, cols], on='SNP')
        elif uniform:
            causals = frq.SNP.values[np.linspace(0, frq.shape[0] - 1,
                                                 num=ncausal, dtype=int)]
            av_dist = get_SNP_dist(bfile, causals)
            causals = frq[frq.SNP.isin(causals)]
            print('Causal SNPs are %d kbp apart on average' % av_dist)
        elif snps is None:
            causals = frq.sample(ncausal, replace=False, random_state=seed)
        else:
            causals = frq[frq.SNP.isin(snps)]
        # If causal effects are provided use them, otherwise get them
        if causaleff is None:
            std = np.sqrt(h2_snp)
            g_eff = np.random.normal(loc=0, scale=std, size=ncausal)
            # make sure is the correct variance when samples are small
            # while not np.allclose(g_eff.var(), h2_snp, rtol=0.05):
            #    g_eff = np.random.normal(loc=0, scale=std, size=ncausal)
            causals.loc[:, 'eff'] = g_eff
        # write snps and effect to score file
        mafs = causals.maf
        causals.loc[:, 'norm'] = np.sqrt(2 * mafs * (1 - mafs))
        causals.loc[:, 'beta'] = causals.loc[:, 'eff'] / causals.norm
        scfile = causals.sort_index()
        # Write score to file
        keep = ['SNP', a1, 'beta', 'eff']
        scfile.loc[:, keep].to_csv('%s.score' % outprefix, sep=' ',
                                   header=False, index=False)
        scfile.to_csv('%s.full' % outprefix, sep=' ', index=False)
    else:
        scfile = pd.read_table('%s.full' % outprefix, delim_whitespace=True,
                               header=None, names=['SNP', 'Allele', 'beta'])
    # Score using plink
    score = ('%s --bfile %s --score %s.full 2 3 9 sum header --allow-no-sex '
             '--extract %s --a1-allele %s 3 2 --out %s --memory %d --threads '
             '%d')
    if not os.path.isfile('%s.profile' % outprefix):
        executeLine(score % (plinkexe, bfile, outprefix, totalsnps, allele,
                             outprefix, maxmem, threads))
    if not os.path.isfile('%s_unnorm.profile' % outprefix):
        score = (
            '%s --bfile %s --score %s.full 2 3 7 header sum --allow-no-sex '
            '--extract %s --a1-allele %s 3 2 --out %s --memory %d --threads %d')
        executeLine(score % (plinkexe, bfile, outprefix, totalsnps, allele,
                             '%s_unnorm' % outprefix, maxmem, threads))
    # Read scored and rename SCORE column,
    score = pd.read_table('%s.profile' % outprefix, delim_whitespace=True)
    score = score.rename(columns={'SCORESUM': 'gen_eff'})

    return score, scfile, totalsnps


# ----------------------------------------------------------------------

def create_pheno(prefix, h2, prs_true, noenv=False):
    """
    Generate phenotypes and real betas.

    :param prefix: Prefix for outputs
    :type prefix: str
    :param h2: Desired heritability
    :type h2: float
    :param prs_true: First Dataframe outputted by TruePRS with the scored geno
    :type prs_true: :class pd.DataFrame
    :param noenv: whether or not environmental effect should be added
    :type noenv: bool
    """
    # Deal with no environment
    if h2 == 1:
        noenv = True
    nind = prs_true.shape[0]
    if noenv:
        env_effect = np.zeros(nind)
    else:
        va = prs_true.gen_eff.var()
        # std = np.sqrt((va / h2) - va)
        std = np.sqrt(max(1 - va, 0))
        env_effect = np.random.normal(loc=0, scale=std, size=nind)
        # for small sample sizes force to be close to the expected heritability
        # while not np.allclose(env_effect.var(), 1 - h2, rtol=0.05):
        # env_effect = np.random.normal(loc=0, scale=std, size=nind)
    # Include environmental effects into the dataframe
    prs_true['env_eff'] = env_effect
    # Generate the phenotype from the model Phenotype = genetics + environment
    prs_true['PHENO'] = prs_true.gen_eff + prs_true.env_eff
    print('Phenotype variance: %.3f' % prs_true.PHENO.var())
    # Check that the estimated heritability matches the expected one
    realized_h2 = prs_true.gen_eff.var() / prs_true.PHENO.var()
    line = 'Estimated heritability (Va/Vp) : %.4f' % realized_h2
    with open('realized_h2.txt', 'w') as F:
        F.write(line)
        print(line)
    den = prs_true.gen_eff.var() + prs_true.env_eff.var()
    est_h2 = prs_true.gen_eff.var() / den
    print('Estimated heritability (Va/(Va + Ve)) : %.4f' % est_h2)
    if not np.allclose(h2, est_h2, rtol=0.05):
        print(Warning('Estimated heritability is different than expected'))
    # Write it to file
    outfn = '%s.prs_pheno.gz' % prefix
    prs_true.to_csv(outfn, sep='\t', compression='gzip', index=False)
    ofn = '%s.pheno' % prefix
    opts = dict(sep=' ', header=False, index=False)
    prs_true.reindex(columns=['fid', 'iid', 'PHENO']).to_csv(ofn, **opts)
    # return the dataframe
    return prs_true, realized_h2


# ----------------------------------------------------------------------
def plot_pheno(prefix, prs_true, quality='pdf'):
    """
    Plot phenotype histogram
    
    :param :class pd.DataFrame prs_true: Output of the create_pheno function 
    with true PRSs
    :param str prefix: prefix for outputs
    :param str quality: quality of the plot (e.g. pdf, png, jpg)
    """
    prs_true.loc[:, ['PHENO', 'gen_eff', 'env_eff']].hist(alpha=0.5)
    plt.savefig('%s.%s' % (prefix, quality))
    plt.close()


# ----------------------------------------------------------------------
# TODO: include test of correlation of variants (LD)??

def qtraits_simulation(outprefix, bfile, h2, ncausal, snps=None,
                       causaleff=None, noenv=False, plothist=False, bfile2=None,
                       freqthreshold=0.01, quality='png', seed=None, flip=False,
                       uniform=False, normalize=False, max_memory=None,
                       check=False, remove_causals=False, **kwargs):
    """
    Execute the code. This code should output a score file, a pheno file, and 
    intermediate files with the dataframes produced
    
    :param normalize: Normalize the genotype
    :param validate: prepare train/test sets for crossvalidation
    :param str outprefix: Prefix for outputs
    :param str bfile: prefix of the plink bedfileset
    :param float h2: Desired heritability
    :param int ncausal: Number of causal variants to simulate
    :param str plinkexe: path to plink executable
    :param :class pd.Series snps: Series with the names of causal snps
    :param :class pd.DataFrame frq: DataFrame with the MAF frequencies
    :param causaleff: File with DataFrame with the true causal effects
    :param bool noenv: whether or not environmental effect should be added
    :param float freqthreshold: Lower threshold to filter MAF by
    :param str bfile2: prefix of the plink bedfileset on a second population
    :param str quality: quality of the plot (e.g. pdf, png, jpg)
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    :param int seed: random seed to use in sampling
    :param bool uniform: pick uniformingly distributed causal variants
    """
    now = time.time()
    line = "Performing simulation with h2=%.2f, and %d causal variants"
    print(line % (h2, ncausal))
    if causaleff is not None:
        if isinstance(causaleff, str):
            causaleff = pd.read_table('%s' % causaleff, delim_whitespace=True)
        causaleff = causaleff.reindex(columns=['snp', 'beta'])
        assert causaleff.shape[0] == ncausal
    picklefile = '%s.pickle' % outprefix
    if not os.path.isfile(picklefile):
        opts = dict(prefix=outprefix, bfile=bfile, h2=h2, ncausal=ncausal,
                    normalize=normalize, bfile2=bfile2, seed=seed, snps=snps,
                    causaleff=causaleff, uniform=uniform, f_thr=freqthreshold,
                    flip=flip, max_memory=max_memory, check=check)
        G, bim, truebeta, vec = true_prs(**opts)
        with open(picklefile, 'wb') as F:
            pickle.dump((G, bim, truebeta, vec), F)
    else:
        with open(picklefile, 'rb') as F:
            G, bim, truebeta, vec = pickle.load(F)
    if not os.path.isfile('%s.prs_pheno.gz' % outprefix):
        pheno, realized_h2 = create_pheno(outprefix, h2, truebeta, noenv=noenv)
    else:
        pheno = pd.read_table('%s.prs_pheno.gz' % outprefix, sep='\t')
        realized_h2 = float(open('realized_h2.txt').read().strip().split()[-1])
    if plothist:
        plot_pheno(outprefix, pheno, quality=quality)
    causals = bim.dropna(subset=['beta'])
    causals.to_csv('%s.causaleff' % outprefix, index=False, sep='\t')
    if remove_causals:
        print('Removing causals from files!!')
        bim = bim[~bim.snp.isin(causals.snp)]
        G = G[:, bim.i.values]
        bim.loc[:, 'i'] = list(range(G.shape[1]))
        bim.reset_index(drop=True, inplace=True)
    print('Simulation Done after %.2f seconds!!\n' % (time.time() - now))
    return pheno, realized_h2, (G, bim, truebeta, causals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-m', '--ncausal', type=int, default=200)
    parser.add_argument('-b', '--h2', type=float, default=0.66)
    parser.add_argument('-B', '--bfile', default='EUR')
    parser.add_argument('-P', '--plinkexe', default='~/Programs/plink_mac/plink'
                        )
    parser.add_argument('-e', '--noenv', default=False, action='store_true')
    parser.add_argument('-q', '--quality', help=('type of plots (png, pdf)'),
                        default='png')
    parser.add_argument('--plothist', help=('Plot histogram of phenotype'),
                        default=False, action='store_true')
    parser.add_argument('--causal_eff', help=('Provide causal effects file as'
                                              'produced by a previous run of '
                                              'this code with the extension '
                                              'full'), default=None)
    parser.add_argument('-f', '--freqthreshold', help=('Lower threshold to filt'
                                                       'er MAF by'), type=float,
                        default=0.1)
    parser.add_argument('-2', '--bfile2', help=('prefix of the plink bedfileset'
                                                'o n a second population'))
    parser.add_argument('-u', '--uniform', default=False, action='store_true')
    parser.add_argument('-t', '--threads', default=False, action='store',
                        type=int)
    parser.add_argument('-M', '--maxmem', default=None, action='store')
    parser.add_argument('-s', '--seed', default=None, type=int)
    parser.add_argument('-F', '--flip', default=False, action='store_true')
    parser.add_argument('-C', '--check', default=False, action='store_true')
    parser.add_argument('-a', '--avoid_causals', default=False,
                        action='store_true', help='Remove causals from set')


    args = parser.parse_args()
    qtraits_simulation(args.prefix, args.bfile, args.h2, args.ncausal,
                       args.plinkexe, plothist=args.plothist,
                       causaleff=args.causal_eff, quality=args.quality,
                       freqthreshold=args.freqthreshold, bfile2=args.bfile2,
                       maxmem=args.maxmem, threads=args.threads,
                       seed=args.seed, uniform=args.uniform, flip=args.flip,
                       max_memory=args.maxmem, check=args.check,
                       remove_causals=args.avoid_causals)
