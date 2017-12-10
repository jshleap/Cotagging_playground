"""
Unit testing for utilities4cotagging
"""
import pytest
import filecmp
from utilities4cotagging import *

script_path = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(script_path, 'testfiles')
plink = os.path.join(test_folder, 'plink')


@pytest.mark.parametrize("test_input,expected", [
    (os.path.join(test_folder, 'file_with_10_lines.txt'), 10),
    (os.path.join(test_folder, 'file_with_20_lines.txt'), 20)])
def test_mapcount(test_input, expected):
    assert mapcount(test_input) == expected


@pytest.mark.parametrize("test_input,expected",
                         [(os.path.join(test_folder, 'test_log'), 197)])
def test_read_log(test_input, expected):
    profs = read_log(test_input)
    assert len(profs) == expected


@pytest.mark.parametrize("test_input,expected",
                         [('ls test_utilities4cotagging.py',
                           (b'test_utilities4cotagging.py\n', b''))])
def test_executeLine(test_input, expected):
    assert executeLine(test_input) == expected


@pytest.mark.parametrize("bed,f_thresh,With,plink,expected_snps", [
    (os.path.join(test_folder, 'toy_bed'), 0.1, False, plink, ['SNP1', 'SNP14',
                                                               'SNP27',
                                                               'SNP54']),
    (os.path.join(test_folder, 'toy_bed'), 0.1, True, plink, ['SNP1', 'SNP14',
                                                              'SNP27', 'SNP54']
     ),
    (os.path.join(test_folder, 'toy_bed'), 0.01, False, plink, [
        'SNP1', 'SNP14', 'SNP27', 'SNP40', 'SNP54', 'SNP67', 'SNP107']),
    (os.path.join(test_folder, 'toy_bed'), 0.01, True, plink, [
        'SNP1', 'SNP14', 'SNP27', 'SNP40', 'SNP54', 'SNP67', 'SNP107'])
])
def test_read_freq(bed, f_thresh, With, plink, expected_snps):
    out = '%s.frq.gz' % os.path.split(bed)[-1]
    if not With and os.path.isfile(out):
        os.remove(out)
    frq = read_freq(bed, plink, freq_threshold=f_thresh)
    os.remove(out)
    low = f_thresh
    high = 1 - low
    assert frq[(frq.MAF < high) & (frq.MAF > low)].SNP.tolist() == expected_snps


@pytest.mark.parametrize("test_input,splits,expected",
                         [(os.path.join(test_folder, 'toy_bed'), 2, (5, 5)),
                          (os.path.join(test_folder, 'toy_bed'), 3, (6, 4)),
                          (os.path.join(test_folder, 'toy_bed'), 5, (8, 2)),
                          (os.path.join(test_folder, 'toy_bed'), 10, (9, 1))
                          ])
def test_train_test_gen_only(test_input, splits, expected):
    params = dict(names=['FID', 'IID', 'a', 'b', 'c', 'd'],
                  delim_whitespace=True, header=None)
    tr, te = train_test_gen_only('testing_function', test_input, plink,
                                 splits=splits)
    fam_tr = pd.read_table('%s.fam' % tr, **params)
    fam_te = pd.read_table('%s.fam' % te, **params)
    for f in glob('testing_function*'):
        os.remove(f)
    assert (fam_tr.shape[0], fam_te.shape[0]) == expected


@pytest.mark.parametrize("test_input,pheno_input,splits,expected",
                         [(os.path.join(test_folder, 'toy_bed'),
                           os.path.join(test_folder, 'toy.pheno'), 2, [
                               ((5, 6), (5, 3)), ((5, 6), (5, 3))]),
                          (os.path.join(test_folder, 'toy_bed'),
                           os.path.join(test_folder, 'toy.pheno'), 3, [
                               ((6, 6), (6, 3)), ((4, 6), (4, 3))]),
                          (os.path.join(test_folder, 'toy_bed'),
                           os.path.join(test_folder, 'toy.pheno'), 5, [
                               ((8, 6), (8, 3)), ((2, 6), (2, 3))]),
                          (os.path.join(test_folder, 'toy_bed'),
                           os.path.join(test_folder, 'toy.pheno'), 10, [
                               ((9, 6), (9, 3)), ((1, 6), (1, 3))])
                          ])
def test_train_test(test_input, pheno_input, splits, expected):
    fam_opts = dict(names=['FID', 'IID', 'a', 'b', 'c', 'd'],
                    delim_whitespace=True, header=None)
    phe_opts = dict(names=['FID', 'IID', 'Pheno'],
                    delim_whitespace=True, header=None)
    keeps = train_test('testing_function', test_input, plink, pheno=pheno_input,
                       splits=splits)
    asserts = [(pd.read_table('%s.fam' % k, **fam_opts).shape,
                pd.read_table('%s' % v[1], **phe_opts).shape)
               for k, v in keeps.items()
               ]
    for f in glob('testing_function*') + glob('toy*'):
        os.remove(f)
    assert asserts == expected


@pytest.mark.parametrize("test_input,a,b,expected", [
    (range(10), 0, 1, np.array([0., 0.11111111, 0.22222222, 0.33333333,
                                0.44444444, 0.55555556, 0.66666667,
                                0.77777778, 0.88888889, 1.]))])
def test_norm_array(test_input, a, b, expected):
    nr = norm(test_input, a, b)
    assert np.isclose(expected, nr, rtol=1e-05).all()


@pytest.mark.parametrize("test_input,expected", [
    (os.path.join(test_folder, 'toy.pheno'), np.array([-3.51134, -2.80063,
                                                       -4.52143, -4.25842,
                                                       -1.83618, -2.78718,
                                                       -3.05361, -0.909709,
                                                       -2.47623, -4.12518]))])
def test_read_pheno(test_input, expected):
    phe = read_pheno(test_input)
    v = phe.Pheno.values
    assert np.isclose(expected, v, rtol=1e-05).all() and (phe.shape == (10, 3))


@pytest.mark.parametrize("test_input,expected", [
    (os.path.join(test_folder, 'toy.clumped'),
     os.path.join(test_folder, 'toy_clump_result')),
])
def test_parse_sort_clump(test_input, expected):
    result = pd.read_csv(expected, names=['Index', 0], header=0)
    df = parse_sort_clump(test_input, ['SNP80', 'SNP67', 'SNP54', 'SNP1',
                                       'SNP14'])
    pd.util.testing.assert_frame_equal(df, result)


# TODO: change the tests for the new gwas and cotaggings
@pytest.mark.parametrize("test_input_stats,test_input_cotag,expected", [
    (os.path.join(test_folder, 'toy.sumstats'),
     os.path.join(test_folder, 'toy_taggingscores.tsv'),
     os.path.join(test_folder, 'toy_smartsort_res'))])
def test_smartcotagsort(test_input_stats, test_input_cotag, expected):
    cols = ['Cotagging', 'CHR', 'SNP', 'BP', 'A1', 'TEST', 'NMISS', 'BETA',
            'SE', 'L95', 'U95', 'STAT', 'P', 'Tagging AFR', 'Tagging EUR',
            'Index']
    result = pd.read_hdf(expected, key='a').loc[:,cols]
    sumstats = pd.read_table(test_input_stats, delim_whitespace=True)
    cotag = pd.read_table(test_input_cotag, sep='\t')
    gwascotag = sumstats.merge(cotag, on='SNP')
    df, tail = smartcotagsort('toy', gwascotag)
    assert all([result.SNP.tolist() == df.SNP.tolist(), tail == 9])


@pytest.mark.parametrize("nsnps,step,init_step,every,expected", [
    (10, 1, 1, True, np.arange(10, 110, 10, dtype=float)),
    (10, 2, 2, False, np.arange(10, 110, 10, dtype=float)),
    (100, 2, 1, False, np.array([1., 2., 3., 4., 5., 6., 8., 10., 12., 14., 16.,
                                 18., 20., 22., 24., 26., 28., 30., 32., 34.,
                                 36., 38., 40., 42., 44., 46., 48., 50., 52.,
                                 54., 56., 58., 60., 62., 64., 66., 68., 70.,
                                 72., 74., 76., 78., 80., 82., 84., 86., 88.,
                                 90., 92., 94., 96., 98., 100.])),
    (200, 5, 2, False, np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5., 10., 15., 20.,
                                 25., 30., 35., 40., 45., 50., 55., 60., 65.,
                                 70., 75., 80., 85., 90., 95., 100.]))
])
def test_set_first_step(nsnps, step, init_step, every, expected):
    arr = set_first_step(nsnps, step, init_step=init_step, every=every)
    assert (arr == expected).all()


@pytest.mark.parametrize("nsnps,step,every,qrangefn,expected", [
    (200, 2, False, None, os.path.join(test_folder, 'toy_200_2.qrange')),
    (400, 5, False, None, os.path.join(test_folder, 'toy_400_5.qrange')),
    (100, 5, True, None, os.path.join(test_folder, 'toy_100_5_every.qrange')),
    (1000, 10, False, os.path.join(test_folder, 'toy_400_5.qrange'),
     os.path.join(test_folder, 'toy_400_5.qrange'))
])
def test_gen_qrange(nsnps, step, every, qrangefn, expected):
    order = ['label', 'Min', 'Max']
    expecteddf = pd.read_csv(expected, sep=' ', header=None, names=order)
    qr, qrange = gen_qrange('toy', nsnps, step, every=every, qrangefn=qrangefn
                            )
    expecteddf.Max = expecteddf.Max.astype(qr.Max.dtype)
    expecteddf.Min = expecteddf.Min.astype(qr.Min.dtype)
    expecteddf.label = expecteddf.label.astype(qr.label.dtype)
    boo = filecmp.cmp(qrange, expected)
    # for i in glob('toy*'):
    #     os.remove(i)
    assert boo


@pytest.mark.parametrize("scoretype,profilefn,expected", [
    (' ', os.path.join(test_folder, 'toy.profile'), {
        'File': os.path.join(test_folder, 'toy.profile'),
        'R2': 0.37902598903634782, 'SNP kept': 200, 'alpha': 0.1}),
    ('sum', os.path.join(test_folder, 'toy_sum.profile'), {
        'File': os.path.join(test_folder, 'toy_sum.profile'),
        'R2': 0.37902598903634782, 'SNP kept': 200, 'alpha': 0.1})])
def test_read_scored_qr(scoretype, profilefn, expected):
    phenofile, alpha, nsnps = os.path.join(test_folder, 'toy.pheno'), 0.1, 200
    d = read_scored_qr(profilefn, phenofile, alpha, nsnps, score_type=scoretype)
    assert d == expected


@pytest.mark.parametrize("norm_geno,expected", [
    (False, 'toy_unnorm'), (True, 'toy')])
def test_qrscore(norm_geno, expected):
    order = ['label', 'Min', 'Max']
    qrange = os.path.join(test_folder, 'toy.qrange')
    qr = pd.read_csv(qrange, sep=' ', header=None, names=order)
    params = {'plinkexe': plink, 'bfile': os.path.join(test_folder, 'toy_bed'),
              'scorefile': os.path.join(test_folder, 'toy.score'), 'qr': qr,
              'qrange': qrange, 'qfile': os.path.join(test_folder, 'toy.qfile'),
              'phenofile': os.path.join(test_folder, 'toy.pheno'),
              'ou':expected, 'maxmem': 1700, 'threads': 1, 'label':'EUR',
              'prefix': 'toy', 'normalized_geno': norm_geno,
              'allele_file': os.path.join(test_folder, 'toy.alleles')
              }
    df = qrscore(**params)
    ex = pd.read_csv(os.path.join(test_folder, '%s_expected_qrscore' % expected
                                  ), index_col=0)
    pd.testing.assert_frame_equal(df, ex)
    profs = glob('%s.*.profile')
    check = all([filecmp.cmp(i, os.path.join(test_folder,i)) for i in profs])
    for i in glob('*toy*'):
        os.remove(i)
    assert check

