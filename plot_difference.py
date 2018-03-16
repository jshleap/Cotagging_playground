import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
import scipy
import sys

plt.style.use('ggplot')

if len(sys.argv) > 1:
    one = int(sys.argv[1])
else:
    one = False
fs = glob('run*/*_finaldf.tsv')
integrals = glob('run*/df_*_integral_res.tsv')
betas = glob('run*/df_*_slope_res.tsv')
assert len(fs) == len(betas) == len(integrals)
for j in range(len(betas)):
    if pd.read_csv(betas[j], sep='\t').iloc[0].snp != \
            pd.read_csv(integrals[j], sep='\t').iloc[0].snp:
        print('Mismatch in', betas[j], 'and', integrals[j])

print('Processing %d files' % len(fs))
# if one:
#     dfs = []
#     for fn in fs:
#         d = pd.read_table(fn, sep='\t')
#         d = d[d.loc[:, 'Number of SNPs'] == one]
#         d['run'] = fn
#         dfs.append(d)
# else:
dfs = []
for fn in fs:
    # df = pd.read_table(fn, sep='\t').groupby('type', as_index=False).max()
    d = pd.read_table(fn, sep='\t')
    d['run'] = fn
    dfs.append(d)
df = pd.concat(dfs)
grouped = df.groupby(['Number of SNPs', 'type'], as_index=False).mean()
fig, ax = plt.subplots()
for typ, gr in grouped.groupby('type'):
    gr.plot(x='Number of SNPs', y='R2', label=typ, ax=ax, marker='.')
plt.ylabel(r'$\bar{R^{2}}$')
plt.savefig('average%d_transferability_plot.pdf' % len(fs))

dfs2 = []
for d in dfs:
    if one:
        d = d[d.loc[:, 'Number of SNPs'] == 1]
    d = d.groupby(['run', 'type'], as_index=False).agg({'R2': max})
    d[r'$R^2$ difference'] = d.R2.values - d[(d.type == 'P + T')].R2.values
    dfs2.append(d)
df = pd.concat(dfs2)
df.rename(columns={'type':'Method'}, inplace=True)
#df['Method'] = df['Method'].map({r'$\beta^2$': r'$\hat{\beta}^2$'})
df.replace(to_replace=r'$\beta^2$', value=r'$\hat{\beta}^2$', inplace=True)
df.replace(to_replace=r'$\hat{\beta^2}$', value=r'$\hat{\beta}^2$', inplace=True
           )
print('smallest difference with P + T')
print(df.nsmallest(1, r'$R^2$ difference'))
# print('Largest difference between beta and pval')
# print(df.nlargest(1, r'$/beta$ - pvalue $R^2$'))
columns_my_order = ['Causals', 'P + T',  'pval', r'$\hat{\beta}^2$', 'Integral',
                    'ese AFR', 'ese EUR', 'ese cotag']
a = df[df.Method == 'ese cotag'].loc[:,r'$R^2$ difference']
for i, col in enumerate(columns_my_order[:-1]):
    b = df[df.Method == col].loc[:,r'$R^2$ difference']
    res = scipy.stats.ttest_ind(a, b, equal_var=False)
    if res.pvalue <= 0.05:
        new =  col + '*'
        columns_my_order[i] = new
        df.replace(to_replace=col, value=new, inplace=True)

fig, ax = plt.subplots()
sns.boxplot(x='Method', y=r'$R^2$ difference', data=df, order=columns_my_order,
            ax=ax)
#df.boxplot(column=r'R^2 difference', by='Method')
plt.ylabel(r'$R^2$ difference')
plt.title(r'$R^2$ difference with P + T')
plt.suptitle("")
plt.tight_layout()
if one:
    plt.savefig('difference_%d_scored.pdf' % one)
else:
    plt.savefig('difference.pdf')