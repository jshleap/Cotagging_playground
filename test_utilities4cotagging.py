"""
Unit testing for utilities4cotagging
"""
import pytest

from pptc2 import compute_clumps
from qtraitsimulation import qtraits_simulation
from utilities4cotagging import *

# Global Variables
script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
cov3 = [[1, 0.8, 0.6], [0.8, 1, 0.4], [0.6, 0.4, 1]]
geno = da.from_array(np.random.multivariate_normal([0, 0, 0], cov3, 1000),
                     chunks=(1000, 3))
bim = pd.DataFrame({'snp':['SNP0', 'SNP1', 'SNP3'], 'i': [0, 1, 2],
                    'pos': [0, 1000, int(1.1E6)]})
df = bim[bim.snp.isin(['SNP0', 'SNP1'])]
sub = geno[:, df.i.values]
corr = pd.DataFrame(np.corrcoef(geno.T)**2, columns=bim.snp.tolist(),
                    index=bim.snp.tolist())
bed = os.path.join(test_folder, 'toy_bed')
with open(os.path.join(test_folder, 'toy_bed_maf.pickle'), 'rb') as F:
    (b, f, g) = pickle.load(F)
with open(os.path.join(test_folder, 'toy_Cotagging.pickle'), 'rb') as F:
    sumstats, tail = pickle.load(F)
shared_snps = sumstats[sumstats.SNP.isin(b.snp)].SNP.tolist()
sumstats_subset = sumstats[sumstats.SNP.isin(shared_snps)]
idx = b[b.snp.isin(shared_snps)].i.tolist()
sumstats_subset.loc[:, 'i'] = idx
pheno = f.copy()
pheno['PHENO'] = g[:,idx].dot(sumstats_subset.BETA.values.astype(float)
                              ).compute()
(EUR_bim, EUR_fam, EUR_g) = read_geno(bed, 0.01, 1, False, False)
(AFR_bim, AFR_fam, AFR_g) = read_geno(bed + '2', 0.01, 1, False, False)
EUR_g = (EUR_g - EUR_g.mean(axis=0)) / EUR_g.std(axis=0)
AFR_g = (AFR_g - AFR_g.mean(axis=0)) / AFR_g.std(axis=0)
o = qtraits_simulation(os.path.join(test_folder, 'AFR'), bfile=AFR_g,
                       bim=AFR_bim, fam=AFR_fam)
AFR_pheno, realized_h2, (g, bim, truebeta, causals) = o

toy_sumstats = pd.read_table(os.path.join(test_folder, 'toy_test.gwas'),
                             sep='\t')
mbim = EUR_bim.merge(AFR_bim, on=['chrom', 'snp', 'pos'], suffixes=['_ref',
                                                                    '_tar'])

# Tests
@pytest.mark.parametrize("test_input,expected",
                         [('ls test_utilities4cotagging.py',
                           (b'test_utilities4cotagging.py\n', b''))])
def test_execute_line(test_input, expected):
    assert execute_line(test_input) == expected


@pytest.mark.parametrize("test_input,df,block,expected", [
    (sub, df, 0, corr.loc[bim.snp.iloc[:2], bim.snp.iloc[:2]])])
def test_single_block(test_input, df, block, expected):
    d = single_block(test_input, df, block)
    pd.testing.assert_frame_equal(d[block].compute(), expected)


@pytest.mark.parametrize("bim,geno,kbwindow,threads,expected", [
    # TODO: include different window sizes
    (bim, geno, 1000, 1, corr), (bim, geno, 1000, 2, corr)
])
def test_blocked_r2(bim, geno, kbwindow, threads, expected):
    bim, r2 = blocked_r2(bim, geno, kb_window=kbwindow, threads=threads)
    pd.testing.assert_frame_equal(r2[bim.loc[0, 'block']].compute(),
                                  expected.loc[
                                      bim.snp.iloc[:2], bim.snp.iloc[:2]])
    pd.testing.assert_frame_equal(r2[bim.loc[2, 'block']].compute(),
                                  expected.loc[
                                      bim.snp.iloc[2:], bim.snp.iloc[2:]])


@pytest.mark.parametrize("test_input,a,b,expected", [
    (range(10), 0, 1, np.array([0., 0.11111111, 0.22222222, 0.33333333,
                                0.44444444, 0.55555556, 0.66666667,
                                0.77777778, 0.88888889, 1.]))])
def test_norm_array(test_input, a, b, expected):
    nr = norm(test_input, a, b)
    assert np.isclose(expected, nr, rtol=1e-05).all()


@pytest.mark.parametrize("bfile,freq_thresh,threads,flip,check,pickled", [
    (bed, 0, 1, False, False, os.path.join(test_folder,'toy_bed_plain.pickle')),
    (bed, 0, 2, False, False, os.path.join(test_folder,'toy_bed_plain.pickle')),
    (bed, 0, 8, False, True, os.path.join(test_folder, 'toy_bed_check.pickle')),
    (bed, 0, 8, True, False, os.path.join(test_folder, 'toy_bed_flip.pickle')),
    (bed, 0.01, 8, False, False, os.path.join(test_folder, 'toy_bed_maf.pickle')
     )])
def test_read_geno(bfile, freq_thresh, threads, flip, check, pickled):
    with open(pickled,'rb') as F:
        expected = pickle.load(F)
    (bim, fam, g) = read_geno(bfile, freq_thresh, threads, flip, check)
    pd.testing.assert_frame_equal(bim, expected[0])
    pd.testing.assert_frame_equal(fam, expected[1])
    np.testing.assert_allclose(g.compute(), expected[2].compute())


@pytest.mark.parametrize("ascending,dataf,column", [(True, sumstats, 'Index'),
                                                    (False, sumstats, 'Index')])
# TODO: Inlcude a test case for other columns
def test_smartcotagsort(ascending, dataf, column):
    out = smartcotagsort('prefix', dataf, column=column, ascending=ascending,
                         beta='BETA', position='BP')
    result, before_tail = out
    if ascending:
        np.testing.assert_array_equal(result.index.values, dataf.Index.values)
    else:
        np.testing.assert_array_equal(np.flip(result.Index.values,0),
                                      dataf.Index.values)
    execute_line('rm prefix_*')


@pytest.mark.parametrize("shape,expected", [
    ((10, 10), 0.0008), ((10, 1000), 0.08), ((45000,3000), 1080.0)])
def test_estimate_size(shape, expected):
    actual = estimate_size(shape)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape,threads,memory,expected", [
    ((10, 10), 1, None, (10,10)), ((10, 10), 1, 0.0007, (5,5)),
    ((10, 1000), 1, None, (10, 1000)), ((10, 1000), 8, 0.1, (1, 142))])
def test_estimate_chunks(shape, threads, memory, expected):
    result = estimate_chunks(shape, threads, memory)
    assert result == expected


expected = {'Number of SNPs': 4, 'R2':0.9999999999999998 , 'type': 'label'}


@pytest.mark.parametrize("subdf,geno,pheno,expected", [
    (sumstats_subset, g, pheno, expected),
])
def test_single_score(subdf, geno, pheno, expected):
    label = "label"
    d = single_score(subdf, geno, pheno, label, beta='BETA')
    assert d == expected


expected = {'Number of SNPs': {0: 1, 1: 2, 2: 3, 3: 4}, 'R2': {
    0: 0.9931961793474825, 1: 0.9931961793474827, 2: 0.9986418678440357,
    3: 0.9999999999999998}, 'type': {0: 'label', 1: 'label', 2: 'label',
                                     3: 'label'}}


@pytest.mark.parametrize("df,geno,pheno,step,threads,expected", [
    (sumstats_subset, g, pheno, 1, 1, expected),
    (sumstats_subset, g, pheno, 1, 4, expected)
])
def test_prune_it(df, geno, pheno, step, threads, expected):
    # TODO: test different steps
    a = prune_it(df, geno, pheno, 'label', step, threads, beta='BETA')
    assert a.to_dict() == expected


@pytest.mark.parametrize("df,rgeno,tgeno,threads,max_memory,justd,extend,exp", [
    (mbim, EUR_g, AFR_g, 1, None, True, False, 'toy_test_ds.pickle'),
    (mbim, EUR_g, AFR_g, 1, None, False, False, 'toy_test_cotd.pickle'),
    (mbim, EUR_g, AFR_g, 4, None, True, False, 'toy_test_ds.pickle'),
    (mbim, EUR_g, AFR_g, 4, None, False, False, 'toy_test_cotd.pickle'),
    (mbim, EUR_g, AFR_g, 4, None, False, True, 'toy_test_cotd_ext.pickle'),
    (mbim, EUR_g, AFR_g, 4, None, True, True, 'toy_test_d_ext.pickle')
])
def test_single_window(df, rgeno, tgeno, threads, max_memory, justd, extend,
                       exp):
    cwd=os.getcwd()
    os.chdir(test_folder)
    with open(exp, 'rb') as F:
        expected = pickle.load(F)
    ridx, tidx = df.i_ref.values, df.i_tar.values
    rg, tg = rgeno[:, ridx], tgeno[:, tidx]
    out = single_window(df, rg, tg, threads, max_memory, justd, extend)
    if isinstance(out, tuple):
        assert (out[0] == expected[0]).all()
        np.testing.assert_allclose(out[1].compute(), expected[1].compute())
        np.testing.assert_allclose(out[2].compute(), expected[2].compute())
    else:
        pd.testing.assert_frame_equal(out, expected)
    os.chdir(cwd)


@pytest.mark.parametrize(
    "rgeno,rbim,tgeno,tbim,kbwindow,threads,max_memory,justd,extend,exp", [
        (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 1, None, True, False,
         'toy_test_ds.pickle'),
        (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 1, None, False, False,
         'toy_test_cotd.pickle'),
        (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 4, None, True, False,
         'toy_test_ds.pickle'),
        (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 4, None, False, False,
         'toy_test_cotd.pickle'),
        (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 4, None, True, True,
         'toy_test_d_ext.pickle'),
        (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 4, None, False, True,
         'toy_test_cotd_ext.pickle')
])
def test_get_ld(rgeno, rbim, tgeno, tbim, kbwindow, threads, max_memory, justd,
                extend, exp):
    print('rg', rgeno.shape, 'tg', tgeno.shape, 'rbim', rbim.shape,
          'tbim', tbim.shape)
    cwd=os.getcwd()
    os.chdir(test_folder)
    with open(exp, 'rb') as F:
        expected = pickle.load(F)
    out = get_ld(rgeno, rbim, tgeno, tbim, kbwindow, threads, max_memory, justd,
                 extend)
    if isinstance(out, tuple) or isinstance(out, list):
        assert (out[0][0] == expected[0]).all()
        np.testing.assert_allclose(out[0][1].compute(), expected[1].compute()
                                   )
        np.testing.assert_allclose(out[0][2].compute(), expected[2].compute()
                                   )
    else:
        pd.testing.assert_frame_equal(out, expected)
    os.chdir(cwd)


loci = get_ld(EUR_g, EUR_bim, AFR_g, AFR_bim, kbwindow=1000, justd=True,
              threads=8, max_memory=None)
result = [0.25, 0.25, 0.5, 0.5, 0.2, 0.5, 0.5]

@pytest.mark.parametrize("geno,keep_allele_order,result", [
    (EUR_g, False, result)
                         ])
def test_compute_maf(geno, keep_allele_order, result):
    assert [compute_maf(geno[:, i].compute(), keep_allele_order) for i in
            range(EUR_g.shape[1])] == result


@pytest.mark.parametrize(
    "select_index_by,do_locus_ese,normalize,exp", [
        ('pvalue', False,True,),
        # (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 1, None, False, False,
        #  'toy_test_cotd.pickle'),
        # (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 4, None, True, False,
        #  'toy_test_ds.pickle'),
        # (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 4, None, False, False,
        #  'toy_test_cotd.pickle'),
        # (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 4, None, True, True,
        #  'toy_test_d_ext.pickle'),
        # (EUR_g, EUR_bim, AFR_g, AFR_bim, 1000, 4, None, False, True,
        #  'toy_test_cotd_ext.pickle')
    ])
def test_optimize_it(select_index_by, do_locus_ese, normalize):
    ld_range = [0.5]
    by_range = [1]
    h2 = 0.5
    cache = Chest(available_memory=psutil.virtual_memory().available)
    memory = None
    test_geno = AFR_g
    clump_function = compute_clumps

