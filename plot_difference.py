import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import sys

if len(sys.argv) > 1:
    one = True
else:
    one = False
fs = glob('run*/*_finaldf.tsv')
if one:
    dfs = [pd.read_table(fn, sep='\t') for fn in fs]
    dfs = [df[df.loc[:, 'Number of SNPs'] == 1] for df in dfs]
else:
    dfs = [pd.read_table(fn, sep='\t').groupby('type', as_index=False).max()
           for fn in fs]
l = []
for i, df in enumerate(dfs):
    df[r'R^2 difference'] = df.loc[:,'R2'] - df[(df.type == 'P + T')].R2.values
    df['run'] = i
    l.append(df)
df = pd.concat(l)
df.rename(columns={'type':'Method'}, inplace=True)
df.boxplot(column=r'R^2 difference', by='Method')
plt.ylabel(r'$R^2$ difference')
plt.title(r'$R^2$ difference with P + T')
plt.suptitle("")
if one:
    plt.savefig('difference_one_scored.pdf')
else:
    plt.savefig('difference.pdf')