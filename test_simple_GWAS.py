"""
Unittesting for plinkGWAS module
"""
from glob import glob

import pytest
from scipy.stats import distributions
from simple_GWAS import *

# ----------------------------------------------------------------------
def isclose(a, b, rtol=1e-3, atol=1e-3):
    """
    Returns True a and b are equal within a tolerance.
    :param a: mpmath float or float
    :param b: mpmath float or float
    :param rtol: The relative tolerance parameter (See numpy allclose)
    :param atol: The absolute tolerance parameter (See numpy allclose)
    :return: True if a and b are equal within the tolerance; False otherwise.
    """
    return mp.fabs(a - b) <= (atol + rtol * mp.fabs(b))

# variables
script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
plink = os.path.join(test_folder, 'plink')
bfile = os.path.join(test_folder, 'toy200k_first')


@pytest.mark.parametrize("t,df", [
    (4,45998), (1,45998), (10,45998), (20,45998), (4,1000), (2,1000)])
def test_t_sf(t,df):
    estimated = t_sf(t, df)
    expected =  distributions.t.sf(t,df)
    assert isclose(estimated, expected)


@pytest.mark.parametrize("cov", [0.1, 0.2, 0.5, 0.8])
def test_nu_linregress(cov):
    for _ in range(10):
        a, b = np.random.multivariate_normal([0, 0], [[1, cov], [cov, 1]],
                                             1000).T
        expected = lr(a,b)
        estimated = nu_linregress(a,b)
        for i, k in enumerate(expected._fields):
            if k in estimated:
                assert isclose(expected[i], estimated[k], rtol=1E-5, atol=1E-5)


def test_high_precision_pvalue():
    for r in np.arange(0.1,0.9,0.1):  # beyond 0.8 the traditional is set to 0
        t = r * np.sqrt(1000 / ((1.0 - r) * (1.0 + r)))
        expected = distributions.t.sf(t, 1000) * 2
        estimated = high_precision_pvalue(1000, r)
        assert isclose(expected, estimated, rtol=1E-5, atol=1E-5)


#TODO: include tests for the covariates
@pytest.mark.parametrize("cov", [0.1, 0.2, 0.5, 0.8])
def test_st_mod(cov):
    for _ in range(10):
        a, b = np.random.multivariate_normal([0, 0], [[1, cov], [cov, 1]],
                                             1000).T
        expected = lr(a,b)
        estimated = st_mod(a,b)
        for i, k in enumerate(expected._fields):
            if k in estimated:
                assert isclose(expected[i], estimated[k], rtol=1E-5, atol=1E-5)


@pytest.mark.parametrize("pheno,geno,val", [(None, 'toy5k', 2)])
def test_plink_free_gwas(pheno, geno, val):
    # TODO: include external validation of results?
    seed = 54321
    prefix = 'toy5k_gwas'
    bfile = os.path.join(test_folder, geno)
    if pheno is None:
        kwargs = dict(h2=1, ncausal=3, uniform=True, seed=seed, bfile=bfile,
                      snps=['SNP1', 'SNP59246', 'SNP133303'], freq_thresh=0,
                      flip=False, check=False, validate=val, prefix=prefix,
                      pheno=pheno, geno=bfile)
    tup = plink_free_gwas(**kwargs)
    res, X_train, X_test, y_train, y_test = tup
    vals = res.pvalue.values[[0, 4, 9]]
    nvals = res.pvalue.values[[1, 2, 3, 5, 6, 7, 8]]
    sigs = vals < 1E-10
    nsigs = nvals > 1E-3
    [os.remove(i) for i in glob('%s*' % prefix)]
    assert all(sigs) and all(nsigs)
