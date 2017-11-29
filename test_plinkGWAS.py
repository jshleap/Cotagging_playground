"""
Unittesting for plinkGWAS module
"""
from glob import glob

import pytest

from plinkGWAS import *

script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
plink = os.path.join(test_folder, 'plink')
bfile = os.path.join(test_folder, 'toy_bed')

@pytest.mark.parametrize("pheno,geno,val", [
    (None, 'bfile', 3)#, [ 0.12931427,  0.42988976, -0.19299026])
])
def test_plink_free_gwas(pheno,geno,val):#,expected):
    #expected = np.array(expected)
    #threads = 8
    seed = 54321
    bfile = os.path.join(test_folder, 'toy5k')
    prefix = 'toy5k_gwas'
    if pheno is None:
        kwargs = {'h2': 1, 'ncausal': 3, 'uniform': True, 'seed': seed,
                  #'threads':threads,
                  'snps':['SNP1', 'SNP59246', 'SNP133303']}
    tup = plink_free_gwas(prefix, bfile, pheno, geno, validate=val, **kwargs)
    res, X_train, X_test, y_train, y_test = tup
    vals = res.p_value.values[[0,4,9]]
    sigs = (vals < 0.005)
    for i in glob('%s*' % prefix):
        os.remove(i)
    assert all(sigs)
    #np.testing.assert_allclose(expected, vals, rtol=0.05)
