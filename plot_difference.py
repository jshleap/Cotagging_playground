import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
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
if one:
    dfs = [pd.read_table(fn, sep='\t') for fn in fs]
    dfs = [df[df.loc[:, 'Number of SNPs'] == one] for df in dfs]
else:
    dfs = [pd.read_table(fn, sep='\t').groupby('type', as_index=False).max()
           for fn in fs]
l = []
for i, df in enumerate(dfs):
    df[r'R^2 difference'] = df.loc[:,'R2'] - df[(df.type == 'P + T')].R2.values
    df['run'] = fs[i]
    l.append(df)
df = pd.concat(l)
df.rename(columns={'type':'Method'}, inplace=True)
#df['Method'] = df['Method'].map({r'$\beta^2$': r'$\hat{\beta}^2$'})
df.replace(to_replace=r'$\beta^2$', value=r'$\hat{\beta}^2$', inplace=True)
print(df.nsmallest(1, r'R^2 difference'))
columns_my_order = ['Causals', 'P + T',  'pval', r'$\hat{\beta}^2$', 'Integral',
                    'ese AFR', 'ese EUR', 'ese cotag']
fig, ax = plt.subplots()
sns.boxplot(x='Method', y=r'R^2 difference', data=df, order=columns_my_order,
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