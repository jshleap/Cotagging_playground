"""
Based on previous runs of proportions.py plot a join plot
"""
import sys
import os
#import numpy as np
from glob import glob
import pandas as pd
import seaborn as sns
#mport matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
files = glob('run*/proportions.tsv')

todas = []
#t_r2s = []
#max_r2s = []
if sys.argv[1] == 'TRUE':
    read_opts = dict(delim_whitespace=True, header=None, names=['number',
                                                                r'$R^2$'])
    time = 'EUR (%)'
    value = r"$R^2$"

else:
    read_opts = dict(delim_whitespace=True)
    time = "EUR_frac"
    value = r"$R^2_{ppt}$"

for i, fn in enumerate(files):
    path = os.path.split(fn)
    df = pd.read_table(fn, **read_opts)
    df['run'] = i
    df['EUR (%)'] = (df.number.max() - df.number.values) / df.number.max()
    todas.append(df)
    #t_r2s.append(np.load(os.path.join(path[0], 't_r2.npy')))
    #max_r2s.append(np.load(os.path.join(path[0], 'max_r2.npy')))
df = pd.concat(todas)
#t_r2 = np.mean(t_r2s)
#max_r2 = np.mean(max_r2s)
f, ax = plt.subplots()
sns.tsplot(time=time, value=value, unit="run", data=df, ax=ax,  ci="sd")
df.plot.scatter(x=time, y=value, marker='.', s=3, ax=ax)
#ax.axhline(t_r2, ls='-.', color='r', label=r'$\bar{AFR_{P + T}}$')
#ax.axhline(max_r2, ls='-.', color='0.5', label='Causals in AFR')
plt.title('Sample size: %d' % df.number.max())
plt.savefig('Proportions_%sruns_withdots.pdf' % str(df.run.max() + 1))
plt.tight_layout()
plt.close()

f, ax = plt.subplots()
sns.tsplot(time=time, value=value, unit="run", data=df, ax=ax,  ci=[25, 50, 75,
                                                                    95])
plt.title('Sample size: %d' % df.number.max())
plt.savefig('Proportions_%sruns.pdf' % str(df.run.max() + 1))
plt.tight_layout()
plt.close()

f, ax = plt.subplots()
sns.tsplot(time=time, value=value, unit="run", data=df, ax=ax,
           err_style="unit_traces")
plt.title('Sample size: %d' % df.number.max())
plt.savefig('Proportions_%sruns_traces.pdf' % str(df.run.max() + 1))
plt.tight_layout()
plt.close()
