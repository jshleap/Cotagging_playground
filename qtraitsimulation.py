#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Jose Sergio Hleap --<2017>
  Purpose: Simulate quantitative phenotypes given a genotype
  Created: 09/30/17
"""
import argparse
import time
import gc
from utilities4cotagging import *

plt.style.use('ggplot')


# ----------------------------------------------------------------------
def true_prs(prefix, bfile, h2, ncausal, normalize=False, bfile2=None,
             f_thr=0.01, seed=None, causaleff=None, uniform=False, usepi=False,
             snps=None, threads=1, flip=False, check=False, max_memory=None,
             **kwargs):
    """
    Generate TRUE causal effects and the genetic effect equal to the h2 (a.k.a
    setting Vp = 1)

    :param max_memory: Maximum allowed memory
    :param check: Whether to check for constant sites or not
    :param flip: Whether to flip the genotype where MAF > 0.5 or not
    :param threads: Number of thread to use in the parallelizing
    :param usepi: Use the genotype variance to weight on the h2_snp
    :param f_thr: MAF threshold
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
    gc.collect()
    cache = Chest(available_memory=int(max_memory))
    # set random seed
    seed = np.random.randint(10000) if seed is None else seed
    print('using seed %d' % seed)
    np.random.seed(seed=seed)
    if isinstance(bfile, str):
        # read the genotype files of the reference population
        (bim, fam, g) = read_geno(bfile, f_thr, threads, flip=flip, check=check,
                                  max_memory=max_memory)
        # get indices of second pop if needed
        if bfile2 is not None:
            # merge the bim files of tw populations to use common snps
            #bim2 = pd.read_table('%s.bim' % bfile2, delim_whitespace=True)
            (bim2, fam2, G2) = read_geno(bfile2, f_thr, threads, check=check,
                                         max_memory=max_memory)
            snps2 = bim2.snp
            # Save some memory
            del bim2, fam2, G2
            gc.collect()
            print('Filtering current population with second set:')
            print('    Genotype matrix shape before', g.shape)
            # subset the genotype file
            indices = bim[bim.snp.isin(snps2)].i
            g = g[:, indices.tolist()]
            bim = bim[bim.i.isin(indices)].reset_index(drop=True)
            bim['i'] = bim.index.tolist()
            print('    Genotype matrix shape after', g.shape)
    else:
        bim, fam, g = kwargs['bim'], kwargs['fam'], bfile
    # Define pi as 1 or as the mean variance to weight on the ncausals for the
    # computation of the h2 per locus
    pi = g.var(axis=0).mean() if usepi else 1
    # Normalize G to variance 1 and mean 0 if required
    if normalize:
        print('Normalizing genotype to variance 1 and mean 0')
        g = (g - g.mean(axis=0)) / g.std(axis=0)
    # Set some local variables
    allele = '%s.alleles' % prefix
    totalsnps = '%s.totalsnps' % prefix
    allsnps = g.shape[1]
    # Compute heritability per snp
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
        c = cols if 'beta' in bim else 'snp'
        causals = causals.merge(causaleff.reindex(columns=cols), on=c)
        bim = bim.merge(causaleff, on='snp', how='outer')
        print(bim.head())
        # print(causals.head())
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
        if ncausal <= 5:
            pre_beta = np.repeat(std, ncausal)
        else:
            pre_beta = np.random.normal(loc=0, scale=std, size=ncausal)
        # Store them
        causals['beta'] = pre_beta
        causals = causals.dropna(subset=['beta'])
        # make sure we have the right causals
        assert np.allclose(sorted(causals.beta.values), sorted(pre_beta))
    nc = causals.reindex(columns=['snp', 'beta'])
    bidx = bim[bim.snp.isin(nc.snp)].index.tolist()
    bim = bim.reindex(columns=['chrom', 'snp', 'cm', 'pos', 'a0', 'a1', 'i',
                               'mafs', 'flip'])
    bim.loc[bidx, 'beta'] = nc.beta.values.tolist()
    print(bim.dropna(subset=['beta']).head())
    idx = bim.dropna(subset=['beta']).i.values
    # Score
    g_eff = g[:, idx].dot(causals.beta).compute(num_workers=threads, cache=cache
                                                )
    if causaleff is not None:
        assert sorted(bim.dropna(subset=['beta']).snp) == sorted(causaleff.snp)
    fam['gen_eff'] = g_eff
    del g_eff
    gc.collect()
    print('Variance in beta is', bim.beta.var())
    print('Variance of genetic component', fam.gen_eff.var())
    print(bim.head())
    # write full table
    fam.to_csv('%s.full' % prefix, sep=' ', index=False)

    return g, bim, fam, causals.beta


# ----------------------------------------------------------------------
def create_pheno(prefix, h2, prs_true, noenv=False, covs=None):
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
    gc.collect()
    # Deal with no environment
    if h2 == 1:
        noenv = True
    nind = prs_true.shape[0]
    if noenv:
        env_effect = np.zeros(nind)
    else:
        # Compute the enviromental effect as with a variance of 1 - Va, thereby
        # guaranting that Vp = 1
        va = prs_true.gen_eff.var()
        std = np.sqrt(max(1 - va, 0))
        env_effect = np.random.normal(loc=0, scale=std, size=nind)

    # Include environmental effects into the dataframe
    prs_true['env_eff'] = env_effect
    # Generate the phenotype from the model Phenotype = genetics + environment
    prs_true['PHENO'] = prs_true.gen_eff + prs_true.env_eff
    dim1 = prs_true.shape[0]
    # check covariates
    if covs is not None:
        cov = pd.read_table(covs, delim_whitespace=True, header=None)
        # change names for ease
        covs_names = ['Cov%d' % x for x in range(len(cov.columns) - 2)]
        columns = dict(zip(cov.columns, ['fid', 'iid'] + covs_names))
        cov = cov.rename(columns=columns)
        prs_true.merge(cov, on=['fid', 'iid'])
        assert prs_true.shape[0] == dim1
        prs_true['PHENO'] = prs_true[:, ['PHENO'] + covs_names].sum(axis=1)
        del cov
        gc.collect()
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
    gc.collect()
    return prs_true, realized_h2


# ----------------------------------------------------------------------
def plot_pheno(prefix, prs_true, quality='pdf'):
    """
    Plot phenotype histogram

    :param prs_true: Output of the create_pheno function with true PRSs
    :param prefix: prefix for outputs
    :param quality: quality of the plot (e.g. pdf, png, jpg)
    """
    prs_true.loc[:, ['PHENO', 'gen_eff', 'env_eff']].hist(alpha=0.5)
    plt.savefig('%s.%s' % (prefix, quality))
    plt.close()


# ----------------------------------------------------------------------
def qtraits_simulation(outprefix, bfile, h2, ncausal, snps=None, noenv=False,
                       causaleff=None, plothist=False, bfile2=None, flip=False,
                       freq_thresh=0.01, quality='png', check=False, seed=None,
                       uniform=False, normalize=False, remove_causals=False,
                       threads=1, max_memory=None, covs=None, **kwargs):
    """
    Execute the code. This code should output a score file, a pheno file, and
    intermediate files with the dataframes produced

    :param max_memory: Maximum allowed memory
    :param threads: Number of threads to use
    :param remove_causals: Remove the causal variants from the genotype file
    :param check: Whether to check for constant sites or not
    :param flip: Whether to flip the genotype where MAF > 0.5 or not
    :param plothist: Whether to plot the phenotype histogram or not
    :param snps: List or array of causal snps
    :param normalize: Normalize the genotype
    :param outprefix: Prefix for outputs
    :param bfile: prefix of the plink bedfileset
    :param h2: Desired heritability
    :param ncausal: Number of causal variants to simulate
    :param causaleff: File with DataFrame with the true causal effects
    :param noenv: whether or not environmental effect should be added
    :param freq_thresh: Lower threshold to filter MAF by
    :param bfile2: prefix of the plink bedfileset on a second population
    :param quality: quality of the plot (e.g. pdf, png, jpg)
    :param seed: random seed to use in sampling
    :param uniform: pick uniformingly distributed causal variants
    """
    if max_memory is not None:
        available_memory = max_memory
    else:
        available_memory = psutil.virtual_memory().available
    now = time.time()  # record time
    line = "Performing simulation of %s\n" % outprefix
    line += "\tPerforming simulation with h2=%.2f, and %d causal variants"
    print(line % (h2, ncausal))
    # If causal effect, read it into pandas dataframe
    if causaleff is not None:
        if isinstance(causaleff, str):
            causaleff = pd.read_table('%s' % causaleff, sep='\t')
        causaleff = causaleff.reindex(columns=['snp', 'beta'])
        assert causaleff.shape[0] == ncausal
    # If another run has been performed, load it if not compute it
    picklefile = '%s.pickle' % outprefix
    if not os.path.isfile(picklefile):
        opts = dict(prefix=outprefix, bfile=bfile, h2=h2, ncausal=ncausal,
                    normalize=normalize, bfile2=bfile2, seed=seed, snps=snps,
                    causaleff=causaleff, uniform=uniform, f_thr=freq_thresh,
                    flip=flip, check=check, threads=threads,
                    max_memory=available_memory)
        opts.update(kwargs)
        g, bim, truebeta, vec = true_prs(**opts)  # Get true PRS
        with open(picklefile, 'wb') as F:
            pickle.dump((g, bim, truebeta, vec), F)
        gc.collect()
    else:
        g, bim, truebeta, vec = pd.read_pickle(picklefile)
        gc.collect()
        # with open(picklefile, 'rb') as F:
        #     g, bim, truebeta, vec = pickle.load(F)
    if not os.path.isfile('%s.prs_pheno.gz' % outprefix):
        # Get phenotype
        pheno, realized_h2 = create_pheno(outprefix, h2, truebeta, noenv=noenv,
                                          covs=covs)
        gc.collect()
    else:
        pheno = pd.read_table('%s.prs_pheno.gz' % outprefix, sep='\t')
        realized_h2 = float(open('realized_h2.txt').read().strip().split()[-1])
        gc.collect()
    if plothist:
        # Plot phenotype histogram
        plot_pheno(outprefix, pheno, quality=quality)
    # Write outfiles
    causals = bim.dropna(subset=['beta'])
    causals.to_csv('%s.causaleff' % outprefix, index=False, sep='\t')
    gc.collect()
    if remove_causals:
        print('Removing causals from files!!')
        bim = bim[~bim.snp.isin(causals.snp)]
        g = g[:, bim.i.values]
        bim.loc[:, 'i'] = list(range(g.shape[1]))
        bim.reset_index(drop=True, inplace=True)
    print('Simulation Done after %.2f seconds!!\n' % (time.time() - now))
    gc.collect()
    return pheno, realized_h2, (g, bim, truebeta, causals)


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
    parser.add_argument('-q', '--quality', help='type of plots (png, pdf)',
                        default='png')
    parser.add_argument('--plothist', help='Plot histogram of phenotype',
                        default=False, action='store_true')
    parser.add_argument('--causal_eff', help=('Provide causal effects file as'
                                              'produced by a previous run of '
                                              'this code with the extension '
                                              'full'), default=None)
    parser.add_argument('-f', '--freqthreshold', default=0.1, type=float,
                        help='Lower threshold to filter MAF by')
    parser.add_argument('-2', '--bfile2', help=('prefix of the plink bedfileset'
                                                'o n a second population'))
    parser.add_argument('-u', '--uniform', default=False, action='store_true')
    parser.add_argument('-t', '--threads', default=False, action='store',
                        type=int)
    parser.add_argument('-M', '--maxmem', default=None, action='store', type=int
                        )
    parser.add_argument('-s', '--seed', default=None, type=int)
    parser.add_argument('-F', '--flip', default=False, action='store_true')
    parser.add_argument('-C', '--check', default=False, action='store_true')
    parser.add_argument('-a', '--avoid_causals', default=False,
                        action='store_true', help='Remove causals from set')
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--covs', default=None)

    args = parser.parse_args()
    qtraits_simulation(args.prefix, args.bfile, args.h2, args.ncausal,
                       plothist=args.plothist, causaleff=args.causal_eff,
                       quality=args.quality, freq_thresh=args.freqthreshold,
                       bfile2=args.bfile2, seed=args.seed, uniform=args.uniform,
                       flip=args.flip, check=args.check, max_memory=args.maxmem,
                       remove_causals=args.avoid_causals, covs=args.covs)
