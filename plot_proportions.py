"""
PLot proportions
"""

from glob import iglob

import matplotlib
import pandas as pd
import sys
import numpy as np

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def tsplot(ax, data, x, y, **kw):
    # x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


def get_dataframe(pattern, prefix, lines, plink):
    todas = []
    if 'constant' in pattern:
        time = 'ASN (%)'
    else:
        time = 'EUR (%)'
    if plink:
        names = ['number', r'$R^2$', 'Pop', 'run']
        types = dict(zip(names, [int, float, str]))
    else:
        names = ['number', r'$R^2$', 'TP', 'FP', 'ncausal', 'Pop', 'run', time]
        types = dict(zip(names, [int, float, int, int, int, str, int, float]))
    read_opts = dict(sep='\t', header=None,
                     names=names)  # , delim_whitespace=True)
    value = r"$R^2$"
    files = iglob(pattern)
    for i, fn in enumerate(files):
        df = pd.read_table(fn, **read_opts)
        if df.shape[0] != lines:
            continue
        elif df.loc[:, value].isnull().any():
            continue
        else:
            df['run'] = i
            df['EUR (%)'] = ((df.number.max() - df.number.values) * 100
                             ) / df.number.max()
            if (time != 'EUR (%)') or 'cost' in pattern:
                df[time] = 100 - df.loc[:, 'EUR (%)']
            todas.append(df)
    df = pd.concat(todas).dropna().reset_index()
    # pops = df.Pop.unique().tolist()
    # cols = ['r', 'b', 'purple']
    # gr = df.groupby('Pop')
    f, ax = plt.subplots()
    # for i, pop in enumerate(pops):
    #    tsplot(ax, gr.get_group(pop), x=time, color=cols[i], label=pop)
    sns.tsplot(time=time, value=value, unit="run", data=df, ax=ax,
               condition='Pop', ci=[25, 50, 75, 95])
    # plt.title('Sample size: %d' % df.number.max())
    plt.tight_layout()
    plt.savefig('%s_%sruns.pdf' % (prefix, str(df.run.max() + 1)))
    plt.close()


if len(sys.argv) > 1:
    plink = True
else:
    plink = False
n = 44
get_dataframe('run*/proportions.tsv', 'Proportions', n, plink)
# get_dataframe('run*/constant.tsv', 'Constant', 44, plink)
get_dataframe('run*/init.tsv', 'init', n, plink)
get_dataframe('run*/cost.tsv', 'Cost', n, plink)
