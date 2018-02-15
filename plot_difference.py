import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import sys

plt.style.use('ggplot')

if len(sys.argv) > 1:
    one = int(sys.argv[1])
else:
    one = False
fs = glob('run*/*_finaldf.tsv')
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
print(df.nsmallest(1, r'R^2 difference'))
df.boxplot(column=r'R^2 difference', by='Method')
plt.ylabel(r'$R^2$ difference')
plt.title(r'$R^2$ difference with P + T')
plt.suptitle("")
plt.tight_layout()
if one:
    plt.savefig('difference_%d_scored.pdf' % one)
else:
    plt.savefig('difference.pdf')