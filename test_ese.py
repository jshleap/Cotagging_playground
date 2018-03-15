"""
Unit testing for qtraitsimulation
"""
import pytest
from glob import glob
from ese import *


# Constants for tests
script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
bed = os.path.join(test_folder, 'EUR_single_locus')
bed2 = os.path.join(test_folder, 'AFR_single_locus')
sumstats = os.path.join(test_folder, 'EUR_single_locus.gwas')
rpicklefile = '%s.pickle' % bed
tpicklefile = '%s.pickle' % bed2
(EUR_bim, EUR_fam, EUR_g) = read_geno(bed, 0.01, 1, False, False)
(AFR_bim, AFR_fam, AFR_g) = read_geno(bed2, 0.01, 1, False, False)

def per_locus2(locus, sumstats, avh2, h2, n, l_number):
    """
    compute the per-locus expectation on double loop
    """
    snps, D_r, D_t = locus
    locus = sumstats[sumstats.snp.isin(snps)].reindex(columns=['snp', 'slope'])
    m = snps.shape[0]
    h2_l = avh2 * m
    den = np.clip((1 - h2_l), 1E-10, 1)
    mu = ((n / (2 * den)) + (m / (2 * h2)))
    vjs = ((n * locus.slope.values) / den)
    I = integral_b(vjs, mu, snps)
    assert (I.values >= 0).all()
    p = (D_r * D_t)
    Eejb = []
    for i in range(p.shape[0]):
        expcovs = 0
        for j in range(p.shape[1]):
            expcovs += p[i,j] * I[j]
        Eejb.append(expcovs)
    Eejb = np.array(Eejb)
    return pd.DataFrame({'snp': snps, 'ese': abs(Eejb), 'locus': l_number})



@pytest.mark.parametrize("h2_snp,m", [(0.01, 10), (0.05, 2)])
def test_integral_b(h2_snp, m):
    snps = ['SNP%d' % i for i in range(m)]
    for _ in range(10):
        vector = np.clip(np.random.normal(0, np.sqrt(h2_snp), m), 0.35, -0.35)
        den = np.clip((1 - h2_snp), 1E-10, 1)
        mu = ((10000 / (2 * den)) + (m / (2 * h2_snp * m)))
        vs = ((10000 * vector) / den)
        exp = (vs ** 2) / (4 * mu)
        e = np.exp(exp)
        lhs = ((2 * mu) + (vs * vs)) / (4 * (mu * mu))
        rhs = e / e.sum()
        expected = lhs * rhs
        estimated = integral_b(vs, mu, snps)
        assert np.allclose(expected, estimated.values)


@pytest.mark.parametrize("sumstats,rpicklefile,tpicklefile", [
    (sumstats, rpicklefile, tpicklefile)])
def test_per_locus(sumstats, rpicklefile, tpicklefile):#rgeno, rbim, tgeno, tbim, sumstats):
    sumstats = pd.read_table(sumstats, sep='\t')
    with open(rpicklefile, 'rb') as r, open(tpicklefile, 'rb') as t:
        rgeno, rbim, truebeta, vec = pickle.load(r)
        tgeno, tbim, ttruebeta, tvec =  pickle.load(t)
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=250,
                  threads=8, justd=True)
    n = tgeno.shape[0]
    result = per_locus2(loci[0], sumstats, 0.4, 0.4, n, 0) # double for loop
    estimated = per_locus(loci[0], sumstats, 0.4, 0.4, n, 0)
    #assert (result.values.ravel() == estimated.values.ravel()).all()
    np.testing.assert_allclose(result.ese.values, estimated.ese.values)
    pd.testing.assert_series_equal(result.snp, estimated.snp)
    #pd.testing.assert_frame_equal(result, estimated, check_names=False,
    #                              check_like=True)


