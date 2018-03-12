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

(EUR_bim, EUR_fam, EUR_g) = read_geno(bed, 0.01, 1, False, False)
(AFR_bim, AFR_fam, AFR_g) = read_geno(bed2, 0.01, 1, False, False)

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

def test_per_locus(rgeno, rbim, tgeno, tbim, sumstats):
    loci = get_ld(rgeno, rbim, tgeno, tbim, kbwindow=250,
                  threads=8, justd=True)

