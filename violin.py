"""
Given a list of files resulting from prancster make a violin plot. You have to 
be in the base folder of a series of runs with subfolders called with the number
of causal variants:
+-Range
      |
      +-run1
           |
           +-1
           |
           +-10
           .
           .
           .
      |
      +-run2
"""
from matplotlib.offsetbox import AnchoredText
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from glob import glob
#import seaborn as sns
import pandas as pd

files = glob('./run*/*/prancster.tsv')
d=defaultdict(list)
for f in files:
    nc = int(f.split('/')[-2])
    df = pd.read_table(f, sep='\t')
    afr = df.nlargest(1,'$R^{2}$_clumAFR').loc[:,'$R^{2}$_clumAFR'].iloc[0]
    eur = df.nlargest(1,'$R^{2}$_clumEUR').loc[:,'$R^{2}$_clumEUR'].iloc[0]
    hyb = df.nlargest(1,'R2_hybrid').loc[:,'R2_hybrid'].iloc[0]
    totdiff = eur - afr 
    try:
        assert totdiff >= 0
    except AssertionError:
        print('EUR P+T is SMALLER THAN AFR!!!', f)
    hybdiff = hyb - afr
    perc = (hybdiff * 100) / afr#totdiff
    if perc < 0:
        print('SMALL')
        print(f, perc)
    elif perc > 100:
        print('TOO BIG')
        print(f)
        
    d[nc].append(perc)


df = pd.DataFrame([{'causals':k, 'improvement':i} for k, v in d.items() 
                   for i in v], columns=['causals', 'improvement', 'Index'])
c = Counter(df.causals)
for k, v in c.items():
    df.loc[df.causals == k, 'Index'] = list(range(v))    
df = df.pivot(columns='causals', values='improvement', index='Index')
f, ax = plt.subplots()
#ax = sns.violinplot(x="causals", y="improvement", data=df)
plt.axhline(0, alpha=0.5, c='0.4', ls=':')
df.plot.box(ax=ax)
ax.add_artist(AnchoredText('Overall mean %.2f' % df.mean().mean(), 1))
plt.xlabel('Number of causal variants')
plt.ylabel('Percentage of improvement (AFR)')
plt.tight_layout()
plt.savefig('Runs_boxplot.pdf')