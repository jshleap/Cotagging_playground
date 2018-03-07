"""
Unit testing for utilities4cotagging
"""
import pytest
import filecmp
from utilities4cotagging import *

script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')


@pytest.mark.parametrize("test_input,expected",
                         [('ls test_utilities4cotagging.py',
                           (b'test_utilities4cotagging.py\n', b''))])
def test_execute_line(test_input, expected):
    assert execute_line(test_input) == expected


cov3 = [[1, 0.8, 0.6], [0.8, 1, 0.4], [0.6, 0.4, 1]]
geno = da.from_array(np.random.multivariate_normal([0, 0, 0], cov3, 1000),
                     chunks=(1000, 3))
bim = pd.DataFrame({'snp':['SNP0', 'SNP1', 'SNP3'], 'i': [0, 1, 2],
                    'pos': [0, 1000, int(1.1E6)]})
df = bim[bim.snp.isin(['SNP0', 'SNP1'])]
sub = geno[:, df.i.values]
corr = pd.DataFrame(np.corrcoef(geno.T)**2, columns=bim.snp.tolist(),
                    index=bim.snp.tolist())


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


bed = os.path.join(test_folder, 'toy_bed')


@pytest.mark.parametrize("bfile,freq_thresh,threads,flip,check,pickled", [
    (bed, 0, 1, False, False, os.path.join(test_folder, 'toy_bed_plain.pickle')
     ), (bed, 0, 2, False, False, os.path.join(test_folder,
                                               'toy_bed_plain.pickle')),
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


@pytest.mark.parametrize("ascending,column", [(True, 'Index'), (False, 'Index')]
                         )
# TODO: Inlcude a test case for other columns
def test_smartcotagsort(ascending, column):
    with open(os.path.join(test_folder, 'toy_Cotagging.pickle'), 'rb') as F:
        df, tail = pickle.load(F)
    out = smartcotagsort('prefix', df, column=column, ascending=ascending,
                         beta='BETA', position='BP')
    result, before_tail = out
    if ascending:
        np.testing.assert_array_equal(result.index.values, df.Index.values)
    else:
        np.testing.assert_array_equal(np.flip(result.Index.values,0),
                                      df.Index.values)
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
