"""
Based on previous runs of proportions.py plot a join plot
"""
import sys
import os
import numpy as np
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
files = glob('run*/proportions.tsv')

todas = []
t_r2s = []
max_r2s = []
for i, fn in enumerate(files):
    path = os.path.split(fn)
    df = pd.read_table(fn, delim_whitespace=True)
    df['run'] = i
    todas.append(df)
    t_r2s.append(np.load(os.path.join(path[0], 't_r2.npy')))
    max_r2s.append(np.load(os.path.join(path[0], 'max_r2.npy')))
df = pd.concat(todas)
t_r2 = np.mean(t_r2s)
max_r2 = np.mean(max_r2s)
f, ax = plt.subplots()
sns.tsplot(time="EUR_frac", value=r"$R^2_{ppt}$", unit="run", data=df,
           ax=ax,  ci="sd")#ci=[25, 50, 75, 95])
df.plot.scatter(x="EUR_frac", y=r"$R^2_{ppt}$", marker='.', s=3, ax=ax)
ax.axhline(t_r2, ls='-.', color='r', label=r'$\bar{AFR_{P + T}}$')
ax.axhline(max_r2, ls='-.', color='0.5', label='Causals in AFR')
plt.tight_layout()
plt.title('Sample size: %s' % sys.argv[1])
plt.savefig('Proportions_%druns.pdf' % i)
plt.close()
