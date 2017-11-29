"""
Unittesting of the ppt module
"""
import os
import pytest
from ppt import *
from pandas_plink import read_plink

script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
(bim, fam, geno) = read_plink(os.path.join(test_folder, 'toy200k_first'))
R2 = (dd.from_dask_array(geno.T, columns=bim.snp.tolist()).corr()**2).compute()


@pytest.mark.parametrize("snp,r_thresh,done, expected", [
    ('SNP1', 0.5, [], ['SNP1', 'SNP4', 'SNP7', 'SNP8', 'SNP12', 'SNP13',
                       'SNP14','SNP15']),
    ('SNP1', 0.5, ['SNP4', 'SNP7'], ['SNP1', 'SNP8', 'SNP12', 'SNP13', 'SNP14',
                                     'SNP15']),
    ('SNP5', 0.5, [], ['SNP5', 'SNP11']),
    ('SNP5', 0.2, ['SNP1', 'SNP4', 'SNP5', 'SNP7', 'SNP8'],
     ['SNP11', 'SNP12', 'SNP13', 'SNP14', 'SNP15'])

])
def test_single_clump(snp,r_thresh, done, expected):
    clum = single_clump(snp, R2, done, r_thresh)
    assert clum == expected


@pytest.mark.parametrize("t_thresh,p_thresh", [()])
def test_clump():
    clumps = clump(R2, sumstats, t_thresh, p_thresh)