"""
Compute the square difference of betas ves different LD scores
"""
from plinkGWAS import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')

prefix = 'betavsbeta'
bfile1 = ('/Volumes/project/gravel/hleap_projects/Cotagging/newsim/test/'
          'multiSNP/Linear10K/EUR10K_5K')
bfile2 = ('/Volumes/project/gravel/hleap_projects/Cotagging/newsim/test/'
          'multiSNP/Linear10K/AFR10K_5K')

opts = dict(h2=0.6, ncausal=200, uniform=False, normalize=True, validate=3,
            threads=8)
# simulate Pop1
out1 = qtraits_simulation('%s_pop1' % prefix, bfile1, bfile2=bfile2, **opts)
pheno1, (G1, bim1, truebeta1, vec1) = out1
# GWAS Pop1
gout1 = plink_free_gwas('%s_pop1' % prefix, pheno1, G1, bim=bim1, **opts)
res1, x_train1, X_test1, y_train1, y_test1 = gout1
# simulate Pop2 with same effects
out2 = qtraits_simulation('%s_pop2' % prefix, bfile2, bfile2=bfile1,
                          causaleff=bim1.dropna(), **opts)
pheno2, (G2, bim2, truebeta2, vec2) = out2
# GWAS POP2
gout2 = plink_free_gwas('%s_pop2' % prefix, pheno2, G2, bim=bim2, **opts)
res2, x_train2, X_test2, y_train2, y_test2 = gout2

# difference in betas
d2_eq = (res1.slope - res2.slope)**2

# get LDscores
print('getting scores')
D_r = da.dot(G1.T, G1) / G2.shape[0]
D_t = da.dot(G2.T, G2) / G2.shape[0]
cot = da.diag(da.dot(D_r, D_t)).compute(num_workers=8)
ref = da.diag(da.dot(D_r, D_r)).compute(num_workers=8)
tar = da.diag(da.dot(D_t, D_t)).compute(num_workers=8)
ldscorer = (da.corrcoef(G1.T)**2).sum(axis=0).compute(num_workers=8)
ldscoret = (da.corrcoef(G2.T)**2).sum(axis=0).compute(num_workers=8)

print('plotting dependent betas')
y = np.log10(d2_eq)
markers = ['r--', 'b--', 'm--', 'g--', 'c--', 'k--', 'y--']
la = ['Cotagging', 'Tagging EUR', 'Tagging AFR', 'LDscore EUR', 'LDscore AFR',
      'LDscore difference']
f, ax = plt.subplots()
for m, x in enumerate([cot, ref, tar, ldscorer, ldscoret, ldscorer-ldscoret]):
    idx = ~(np.isnan(x) | np.isnan(y))
    x = np.log10(x[idx])
    z = np.polyfit(x, y[idx], 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), markers[m], label=la[m])
plt.legend()
plt.tight_layout()
plt.savefig('/Users/jshleap/Desktop/betavsbeta_equal.pdf')

# correlated effects
print('processing pseudo independent betas')
x, y = np.random.multivariate_normal((0, 0), np.array([[1, 0.2], [0.2, 1]]),
                                     200).T
nbim1 = bim1.dropna().copy()
nbim1['beta'] = x
nbim2 = bim2.dropna().copy()
nbim2['beta'] = y
# simulate Pop1
out1 = qtraits_simulation('%s_pop1_ind' % prefix, bfile1, bfile2=bfile2,
                          causaleff=nbim1, **opts)
pheno1, (G1, bim1, truebeta1, vec1) = out1
# GWAS Pop1
gout1 = plink_free_gwas('%s_pop1_ind' % prefix, pheno1, G1, **opts)
res1, x_train1, X_test1, y_train1, y_test1 = gout1
# simulate Pop2 with same effects
out2 = qtraits_simulation('%s_pop2_ind' % prefix, bfile2, bfile2=bfile1,
                          causaleff=nbim2, **opts)
pheno2, (G2, bim2, truebeta2, vec2) = out2
# GWAS POP2
gout2 = plink_free_gwas('%s_pop2_ind' % prefix, pheno2, G2, **opts)
res2, x_train2, X_test2, y_train2, y_test2 = gout2

d2_ind = (res1.slope - res2.slope)**2

print('plotting')
y = np.log10(d2_ind)
f, ax = plt.subplots()
for m, x in enumerate([cot, ref, tar, ldscorer, ldscoret, ldscorer-ldscoret]):
    idx = ~(np.isnan(x) | np.isnan(y))
    x = np.log10(x[idx])
    z = np.polyfit(x, y[idx], 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), markers[m], label=la[m])
plt.legend()
plt.tight_layout()
plt.savefig('/Users/jshleap/Desktop/betavsbeta_ind.pdf')
