"""
Unittesting for plinkGWAS module
"""
from glob import glob

import pytest

from plinkGWAS import *

script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
plink = os.path.join(test_folder, 'plink')
bfile = os.path.join(test_folder, 'toy200k_first')

@pytest.mark.parametrize("pheno,geno,val", [(None, 'bfile', 3)])
def test_plink_free_gwas(pheno, geno, val):
    seed = 54321
    prefix = 'toy5k_gwas'
    if pheno is None:
        kwargs = {'h2': 1, 'ncausal': 3, 'uniform': True, 'seed': seed,
                  'snps':['SNP1', 'SNP59246', 'SNP133303'], 'bfile':bfile}
    tup = plink_free_gwas(prefix, pheno, geno, validate=val,
                          **kwargs)
    res, X_train, X_test, y_train, y_test = tup
    vals = res.p_value.values[[0,4,9]]
    sigs = (vals < 0.005)
    for i in glob('%s*' % prefix):
        os.remove(i)
    assert all(sigs)
