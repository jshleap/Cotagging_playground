"""
Unittesting of the ppt module
"""
import os
import pytest
from ppt import *
import numpy as np
from pandas_plink import read_plink

np.random.seed(seed=54321)
script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
(bim, fam, geno) = read_plink(os.path.join(test_folder, 'toy200k_first'))
y = fam.reindex(columns=['fid', 'iid'])
R2 = dd.from_dask_array(geno.T, columns=bim.snp.tolist()).corr() ** 2
betas = np.array([0.18682959, 0.11082782, -0.11142298])
# np.random.normal(0,1/3,size=3)
idx = np.linspace(0, geno.shape[0] - 1, num=3, dtype=int)
X = (geno.T - geno.mean(axis=1)) / geno.std(axis=1)  # .compute()
y['PHENO'] = X[:, idx].dot(betas)
recs = [linregress(X[:, i], y.PHENO) for i in range(X.shape[1])]
ss = pd.DataFrame.from_records(recs, columns=['slope', 'intercept', 'r_value',
                                              'p_value', 'std_err'])
ss['snp'] = bim.snp.tolist()


@pytest.mark.parametrize("snp,r_thresh,done, expected", [
    (
    'SNP1', 0.5, [], ['SNP4', 'SNP7', 'SNP8', 'SNP12', 'SNP13', 'SNP14', 'SNP15'
                      ]),
    ('SNP1', 0.5, ['SNP4', 'SNP7'], ['SNP8', 'SNP12', 'SNP13', 'SNP14', 'SNP15']
     ),
    ('SNP5', 0.5, [], ['SNP11']),
    ('SNP5', 0.2, ['SNP1', 'SNP4', 'SNP7', 'SNP8'], ['SNP11', 'SNP12', 'SNP13',
                                                     'SNP14', 'SNP15'])
])
def test_single_clump(snp, r_thresh, done, expected):
    clumps = {}
    extnd = done.extend
    r2 = R2.compute()
    single_clump(snp, r2, done, r_thresh, extnd, clumps)
    assert clumps[snp] == expected


@pytest.mark.parametrize("r_thresh,p_thresh,expected", [
    (0.5, 1E-16,
     dict(Tagged={0: 'SNP4;SNP7;SNP8;SNP12;SNP13;SNP14;SNP15', 2: '',
                  4: 'SNP11', 9: ''}, p_value={0: 0.0, 2: 0.0, 4: 0.0, 9: 0.0},
          snp={0: 'SNP1', 2: 'SNP3', 4: 'SNP5', 9: 'SNP10'}))])
def test_clump(r_thresh, p_thresh, expected):
    clumps, df2 = clump(R2, ss, r_thresh, p_thresh)
    assert clumps.to_dict() == expected


@pytest.mark.parametrize("r_t,p_t,expected", [(0.2, 1E-15, 0.74698),
                                              (0.2, 1, 0.74734),
                                              (0.5, 1E-15, 0.72663),
                                              (0.5, 1, 0.72688)])
def test_score(r_t, p_t, expected):
    r, p, r2, clumps, prs, df2 = score(X, bim, y, ss, r_t, p_t, R2)
    assert (r, p) == (r_t, p_t)
    np.testing.assert_allclose(r2, expected, rtol=1E-4, atol=1E-4)


@pytest.mark.parametrize("r_range,p_range,expected", [
    ([0.2, 0.3, 0.4, 0.5], [1, 1E-15, 1E-16], (1E-15, 0.2, 0.745)),
    ([0.3, 0.4, 0.5], [1, 1E-16], (1E-16, 0.4, 0.7295))
])
def test_pplust(r_range, p_range, expected):
    p_t, r_t, r2, clumps, prs, df2 = pplust('testing', X, y, bim, ss, r_range,
                                       p_range)
    assert (p_t, r_t) == expected[:2]
    np.testing.assert_allclose(r2, expected[-1], rtol=0.01, atol=0.01)
