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
import argparse
plt.style.use('ggplot')


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
    read_opts = dict(sep='\t', header=None, names=names)
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
    f, ax = plt.subplots()
    if len(files) == 1:
        sns.tsplot(time=time, value=value, data=df, ax=ax, condition='Pop',
                   ci=[25, 50, 75, 95])
    else:
        sns.tsplot(time=time, value=value, unit="run", data=df, ax=ax,
               condition='Pop', ci=[25, 50, 75, 95])
    # plt.title('Sample size: %d' % df.number.max())
    plt.tight_layout()
    plt.savefig('%s_%sruns.pdf' % (prefix, str(df.run.max() + 1)))
    plt.close()


def main(plink, folder, n):
    get_dataframe('%s/proportions.tsv' % folder, 'Proportions', n, plink)
    # get_dataframe('run*/constant.tsv', 'Constant', 44, plink)
    get_dataframe('%s/init.tsv' % folder, 'init', n, plink)
    get_dataframe('%s/cost.tsv' % folder, 'Cost', n, plink)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plink', default=True, action='store_false')
    parser.add_argument('-f', '--folder', default='run*')
    parser.add_argument('-n', '--expected_n', default=44, type=int,
                        help='expected number of lines in each tsv file')
    args = parser.parse_args()
    main(args.plink, args.folder, args.expected_n)