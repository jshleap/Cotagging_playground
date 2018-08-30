"""
Unit testing for qtraitsimulation
"""
import pytest
from glob import glob
from qtraitsimulation import *

# Constants for tests
script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
bed1 = os.path.join(test_folder, 'toy_bed')
bed2 = os.path.join(test_folder, 'toy_bed2')
snps = ['SNP1', 'SNP59246']

# Tests
# TODO: include texts for uni
@pytest.mark.parametrize("prefix,h2,ncausal,snp_list,normed,uni,bfile2", [
    ('toy_trueprs_0.5_5_0.01_norm', 0.5, 5, None, True, None, None),
    ('toy_trueprs_0.5_5_0.01_unorm', 0.5, 5, None, False, None, None),
    ('toy_trueprs_0.5_5_0.01_norm_snps', 0.5, 2, snps, True, None, None),
    ('toy_trueprs_0.5_5_0.01_unorm_snps', 0.5, 2, snps, False, None, None),
    ('toy_trueprs_0.5_2_0.01_unorm_snps_bfile2', 0.5, 2, snps, False, None, True
     )])
def test_true_prs(prefix, h2, ncausal, normed, snp_list, uni, bfile2):
    bfile = os.path.join(test_folder, 'toy200K')
    if bfile2:
        bfile2 = os.path.join(test_folder, 'toy200K_2')
    seed = 12345
    expected = os.path.join(test_folder, prefix)
    with open(expected, 'rb') as F:
        g, bim, fam, causals = pickle.load(F)
    if isinstance(snp_list, list):
        ceff = None
    else:
        ceff = bim.dropna(subset=['beta'])
    g, b, f, v = true_prs('prefix', bfile, h2, ncausal, normalize=normed,
                          bfile2=bfile2, seed=seed, causaleff=ceff, uniform=uni,
                          snps=snp_list)
    assert v.shape[0] == ncausal
    if normed:
        np.testing.assert_allclose(g.mean(axis=0).compute(), 0, atol=1e-07)
        np.testing.assert_allclose(g.var(axis=0).compute(), 1)

    np.testing.assert_allclose(f.gen_eff.values, fam.gen_eff.values)
    [os.remove(x) for x in glob('prefix.*')]


# TODO: include noenv. This requires another true PRS as well
@pytest.mark.parametrize("prefix,h2,noenv", [
    ('toy_trueprs_0.5_5_0.01_norm', 0.5, False),
    ('toy_trueprs_0.5_5_0.01_unorm', 0.5, False)
])
def test_create_pheno(prefix, h2, noenv):
    expected = os.path.join(test_folder, prefix)
    with open(expected, 'rb') as F:
        _, _, fam, _ = pickle.load(F)
    pheno, realized_h2 = create_pheno("prefix", h2, fam, noenv)
    den = pheno.gen_eff.var() + pheno.env_eff.var()
    est_h2 = pheno.gen_eff.var() / den
    np.testing.assert_allclose(pheno.PHENO.var(), 1, rtol=1E-2, atol=1E-2)
    np.testing.assert_allclose(realized_h2, est_h2, rtol=1E-3, atol=1E-3)
    [os.remove(x) for x in glob('prefix.*')]


@pytest.mark.parametrize("prefix,h2,ncausals,pop2,uni,normed,ceff", [
    ('toy_trueprs_0.5_5_0.01_norm', 0.5, 5, False, False, True, None),
    ('toy_trueprs_0.5_5_0.01_unorm', 0.5, 5, False, False, False, None),
    ('toy_trueprs_0.2_5_0.01_norm', 0.2, 5, False, False, False, None),
    ('toy_trueprs_0.2_5_0.01_norm', 0.2, 5, False, True, False, None),
    ('toy_trueprs_0.5_5_0.01_norm', 0.5, 5, False, False, True, True)])
def test_qtraits_simulation(prefix, h2, ncausals, pop2, uni, normed, ceff):
    bfile = os.path.join(test_folder, 'toy200K')
    bfile2 = os.path.join(test_folder, 'toy200K2') if pop2 else None
    seed = 12345
    expected = os.path.join(test_folder, prefix)
    if ceff is not None:
        with open(expected, 'rb') as F:
            _, bim, _, _ = pickle.load(F)
        ceff = bim.dropna(subset=['beta'])
    else:
        ceff = None
    out = qtraits_simulation('prefix', bfile, h2, ncausals, causaleff=ceff,
                             bfile2=bfile2, seed=seed, uniform=uni,
                             normalize=normed)
    pheno, realized_h2, (g, bim, truebeta, causals) = out
    # Phenotype should have variance of approximately 1
    np.testing.assert_allclose(pheno.PHENO.var(), 1, rtol=0.01, atol=0.01)
    # Genetic effect should have variance equal to h2 and mean 0 N(0,h2)
    if normed:
        np.testing.assert_allclose(pheno.gen_eff.var(), h2, rtol=0.05,
                                   atol=0.05)
    if ncausals > 5:
        np.testing.assert_allclose(pheno.gen_eff.mean(), 0, rtol=0.05, atol=0.05
                                   )
    # Error effect should have variance equal to 1 - va and mean 0 N(0,1-h2)
    va = pheno.gen_eff.var()
    np.testing.assert_allclose(pheno.env_eff.var(), max(1 - va, 0), rtol=0.05)

    np.testing.assert_allclose(pheno.env_eff.mean(), 0, rtol=0.05, atol=0.05)
    esth2 = pheno.gen_eff.var() / pheno.PHENO.var()
    np.testing.assert_allclose(esth2, realized_h2, rtol=0.05, atol=0.05)
    [os.remove(x) for x in glob('prefix.*')]