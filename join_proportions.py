"""
Based on previous runs of proportions.py plot a join plot
"""

from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
files = glob('run*/proportions.tsv')

all=[]
for i, fn in enumerate(files):
    df = pd.read_table(fn, delim_whitespace=True)
    df['run'] = i
    all.append(df)
df = pd.concat(all)
f, ax = plt.subplots()
sns.tsplot(time="EUR_frac", value=r"$R^2_{ppt}$", unit="run", data=df, ax=ax)
plt.tight_layout()
plt.savefig('Proportions_%druns.pdf' % i)
plt.close()