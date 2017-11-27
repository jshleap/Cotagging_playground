"""
Unit testing for qtraitsimulation
"""
import pytest
import filecmp
from itertools import cycle
from qtraitsimulation import *

script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
plink = os.path.join(test_folder, 'plink')
bfile = os.path.join(test_folder, 'toy_bed')


@pytest.mark.parametrize("causals,expected", [
    (['SNP1', 'SNP67','SNP120'], 28), (['SNP54', 'SNP80', 'SNP1', 'SNP14'], 12),
    (['SNP80', 'SNP120', 'SNP54', 'SNP94', 'SNP27'], 10), ([
        'SNP1', 'SNP14', 'SNP27', 'SNP40', 'SNP54', 'SNP67', 'SNP80', 'SNP94',
        'SNP107', 'SNP120'], 6)])
def test_get_SNP_dist(causals, expected):
    result = get_SNP_dist(bfile, causals)
    assert result == expected

# TODO: include texts for uni, bfile2 and snps
@pytest.mark.parametrize("prefix,h2,ncausal,snps,normed,uni,bfile2,ceff", [
    ('toy_trueprs_0.5_5_0.01_norm', 0.5, 5, None, True, None, None, False),
    # ('toy_trueprs_0.5_5_0.01_unorm', 0.5, 5, None, False, None, None, False),
    # ('toy_trueprs_0.5_5_0.01_norm', 0.5, 5, None, True, None, None, True)
])
def test_true_prs(prefix, h2, ncausal, normed, snps,uni,bfile2,ceff):
    bfile = os.path.join(test_folder, 'toy_bed_10K')
    seed = 12345
    expected = os.path.join(test_folder, prefix)
    with open(expected, 'rb') as F:
        bim, fam, vec = pickle.load(F)
    if ceff:
        ceff = bim.dropna()
    else:
        ceff = None
    g, b, f, v = true_prs(prefix, bfile, h2, ncausal, normalize=normed,
                          bfile2=bfile2, seed=seed, causaleff=ceff, uniform=uni,
                          snps=snps)
    np.testing.assert_allclose(f.gen_eff.var(), h2, rtol=1E-3)

@pytest.mark.parametrize("prefix,h2,ncausals,noenv", [
    ('toy_trueprs_0.5_5_0.01_norm', 0.5, 5, False)])
def test_create_pheno(prefix, h2, ncausals, noenv):
    expected = os.path.join(test_folder, prefix)
    with open(expected, 'rb') as F:
        _, fam, _ = pickle.load(F)
    pheno = create_pheno(prefix, h2, fam, noenv)
    den = pheno.gen_eff.var() + pheno.env_eff.var()
    est_h2 = pheno.gen_eff.var() / den
    np.testing.assert_allclose(h2, est_h2, rtol=0.05)
    np.testing.assert_allclose(h2, pheno.gen_eff.var(), rtol=0.05)
    np.testing.assert_allclose(1 - h2, pheno.env_eff.var(), rtol=0.05)

@pytest.mark.parametrize("prefix,h2,ncausals,pop2,uni,normed,ceff", [
    ('toy_trueprs_0.5_5_0.01_norm', 0.5, 5, False, False, True, None),
    ('toy_trueprs_0.5_5_0.01_unorm', 0.5, 5, False, False, False, None),
    ('toy_trueprs_0.2_5_0.01_norm', 0.2, 5, False, False, False, None),
    ('toy_trueprs_0.2_5_0.01_norm', 0.2, 5, False, True, False, None),
    ('toy_trueprs_0.5_5_0.01_norm', 0.5, 5, False, False, True, True)])
def test_qtraits_simulation(prefix, h2, ncausals, pop2, uni, normed, ceff):
    bfile = os.path.join(test_folder, 'toy5k')#'toy_bed_10K')
    bfile2 = os.path.join(test_folder, 'toy_bed_10K') if pop2 else None
    seed = 12345
    expected = os.path.join(test_folder, prefix)
    if ceff is not None:
        with open(expected, 'rb') as F:
            bim, _, _ = pickle.load(F)
        ceff = bim.dropna().rename(columns={'beta':'eff'})
    else:
        ceff = None
    pheno, Gs = qtraits_simulation(prefix, bfile, h2, ncausals, causaleff=ceff,
                                   bfile2=bfile2, seed=seed, uniform=uni,
                                   normalize=normed)
    # Phenotype should have variance of approximately 1
    np.testing.assert_allclose(pheno.PHENO.var(), 1, rtol=0.05)
    # Genetic effect should have variance equal to h2 and mean 0 N(0,h2)
    np.testing.assert_allclose(pheno.gen_eff.var(), h2, rtol=0.05)
    if normed:
        np.testing.assert_allclose(pheno.gen_eff.mean(), 0, rtol=0.05, atol=0.05)
    # Error effect should have variance equal to 1 - h2 and mean 0 N(0,1-h2)
    np.testing.assert_allclose(pheno.env_eff.var(), 1-h2, rtol=0.05)
    np.testing.assert_allclose(pheno.env_eff.mean(), 0, rtol=0.05, atol=0.05)



# @pytest.mark.parametrize("prefix,h2,ncausal,snps,frq,fth,uni,bfile2,ceff", [
#     ('toy_trueprs_0.5_5_0.01', 0.5, 5, None, None, 0.01, False, None, None),
#     # ('toy_trueprs_0.5_2_0.01_snps', 0.5, 2, pd.Series(['SNP1', 'SNP14']), None,
#     #  0.01, False, None, None), ('toy_trueprs_0.5_5_0.01_frq', 0.5, 5, None,
#     #                             'check', 0.01, False, None, None),
#     # ('toy_trueprs_0.5_3_0.01_uni', 0.5, 3, None, None, 0.01, True, None, None),
#     # ('toy_trueprs_0.5_3_0.01_bf2', 0.5, 3, None, None, 0.01, False,
#     #  os.path.join(test_folder, 'toy_bed2_10K'), None)
# ])
# def test_TruePRS(prefix, h2, ncausal, snps, frq, fth, uni, bfile2, ceff):
#     bfile = os.path.join(test_folder, 'toy_bed_10K')
#     seed = 12345
#     if frq is not None:
#         frq = read_freq(bfile, plink, freq_threshold=fth)
#     params = {'outprefix': prefix, 'bfile': bfile, 'h2': h2, 'ncausal': ncausal,
#               'plinkexe': plink, 'snps': snps, 'frq':frq, 'causaleff': ceff,
#               'bfile2': bfile2, 'freqthreshold': fth, 'seed': seed,
#               'uniform':uni}
#     a, b, c = TruePRS(**params)
#     files = ['%s.alleles' % prefix, '%s.full' % prefix, '%s.profile' % prefix,
#              '%s.score' % prefix, '%s.totalsnps' % prefix]
#     expected = [os.path.join(i, j) for i, j in zip(cycle([test_folder]),files)]
#     check = all([filecmp.cmp(curr, exp) for curr, exp in zip(files, expected)])
# #    for i in glob('*toy*'):
# #        os.remove(i)
#     assert check


# @pytest.mark.parametrize("prefix,h2,prs_true,ncausal,noenv", [
#     ('pheno_0.5', 0.5, os.path.join(test_folder,'expected_create_pheno'), 3,
#      False)
# ])
# def test_create_pheno(prefix, h2, prs_true, ncausal, noenv):
#     prs_true = pd.read_csv(prs_true, index_col=0)
#     new = create_pheno(prefix, h2, prs_true, ncausal, noenv=noenv)
