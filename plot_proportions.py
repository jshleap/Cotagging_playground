"""
PLot proportions
"""

from glob import iglob

import matplotlib

matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def get_dataframe(pattern, prefix):
    todas = []
    names = ['number', r'$R^2$', 'TP', 'FP', 'ncausal', 'Pop']
    read_opts = dict(delim_whitespace=True, header=None, names=names)
    time = r'$AFR_{frac}$' if 'constant' in pattern else 'EUR (%)'
    value = r"$R^2$"
    files = iglob(pattern)
    for i, fn in enumerate(files):
        df = pd.read_table(fn, **read_opts)
        df['run'] = i
        df['EUR (%)'] = ((df.number.max() - df.number.values) * 100
                         ) / df.number.max()
        df[r'$AFR_{frac}$'] = 100 - df.loc[:, 'EUR (%)']
        todas.append(df)
    df = pd.concat(todas)
    f, ax = plt.subplots()
    sns.tsplot(time=time, value=value, unit="run", data=df, ax=ax,
               condition='Pop', ci=[25, 50, 75, 95])
    plt.title('Sample size: %d' % df.number.max())
    plt.tight_layout()
    plt.savefig('%s_%sruns.pdf' % (prefix, str(df.run.max() + 1)))
    plt.close()


get_dataframe('run*/proportions.tsv', 'Proportions')
get_dataframe('run*/constant.tsv', 'Constant')
