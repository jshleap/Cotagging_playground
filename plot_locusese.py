import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from utilities4cotagging import prune_it, read_geno
import argparse
import numpy as np
import os
plt.style.use('ggplot')


def main(args):
    fp = 'run*/%s*.tsv' % args.run
    allf = glob(fp)
    files = [x for x in allf if '_' not in x]
    filesall = [x for x in allf if '_' in x]
    df = pd.concat(pd.read_table(f, sep='\t') for f in files)
    ndf = df.rename(columns={c: '$%s$' % c for c in df.columns})
    # plot Pvalue vs eses
    f, ax = plt.subplots()
    ndf.plot.scatter(x=r'$R^{2}_{pvalue}$', y=r'$R^{2}_{ese}$', c='b', ax=ax,
                     label=r'$R^{2}_{ese}$')
    ndf.plot.scatter(x=r'$R^{2}_{pvalue}$', y=r'$R^{2}_{locus ese}$', c='r',
                     ax=ax, label=r'$R^{2}_{locus ese}$')
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    plt.ylabel(r'$R^2_{target}$')
    plt.tight_layout()
    plt.savefig('%s.pdf' % args.prefix)

    # process and plot the full tsvs
    (bim, fam, geno) = read_geno(args.bfile, 0.01, 8, check=True)
    concat = []
    for fn in filesall:
        folder = os.path.split(fn)[0]
        pheno = pd.read_table(os.path.join(folder, args.pheno), header=None,
                              delim_whitespace=True, names=['fid', 'iid',
                                                            'PHENO'])
        alldf = pd.read_table(fn, sep='\t')
        # process by pvalue
        pval = alldf.sort_values('pvalue')
        p_res = prune_it(pval, geno, pheno, 'pvalue')
        p_res['run'] = fn
        # process by clumped ese
        ese = alldf.sort_values('ese', ascending=False)
        e_res = prune_it(ese, geno, pheno, 'ese')
        e_res['run'] = fn
        # process by locus ese
        ese_locus = alldf.sort_values('locus_ese', ascending=False)
        le_res = prune_it(ese_locus, geno, pheno, 'ese')
        le_res['run'] = fn
        concat.append(pd.concat([p_res, e_res, le_res]))
    df = pd.concat(concat)
    grouped = df.groupby(['Number of SNPs', 'type'], as_index=False).mean()
    fig, ax = plt.subplots()
    for typ, gr in grouped.groupby('type'):
        gr.plot(x='Number of SNPs', y='R2', label=typ, ax=ax, marker='.')
    plt.ylabel(r'$\bar{R^{2}}$')
    plt.savefig('average_transferability_plot.pdf')




# ----------------------------------------------------------------------
if __name__ == '__main__':
    class Store_as_arange(argparse._StoreAction):
        def __call__(self, parser, namespace, values, option_string=None):
            values = np.arange(values[0], values[1], values[2])
            return super().__call__(parser, namespace, values, option_string)


    class Store_as_array(argparse._StoreAction):
        def __call__(self, parser, namespace, values, option_string=None):
            values = np.array(values)
            return super().__call__(parser, namespace, values, option_string)


    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-r', '--run', help='prefix of the tsv files',
                        required=True)
    parser.add_argument('-f', '--pheno', required=True, help='phenotype file')
    parser.add_argument('-b', '--bfile', required=True,
                        help='prefix of the bed fileset of the target')

    parser.add_argument('-T', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=None, type=int)
    parser.add_argument('-q', '--qrange', default=None,
                        help="File of previous qrange. e.g. ranumo's qrange")
    parser.add_argument('--ncausal', default=200, type=int)
    parser.add_argument('--normalize', default=True, action='store_false')
    parser.add_argument('--uniform', default=True, action='store_false')
    parser.add_argument('--split', default=2, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--r_range', default=None, nargs=3,
                        action=Store_as_arange, type=float)
    parser.add_argument('--p_tresh', default=None, nargs='+',
                        action=Store_as_array, type=float)
    parser.add_argument('--flip', action='store_true', help='flip sumstats')
    parser.add_argument('--gflip', action='store_true', help='flip genotype')
    parser.add_argument('--freq_thresh', type=float, help='filter by mafs')
    parser.add_argument('--within', default=0, type=int,
                        help='0=cross; 1=reference; 2=target')

    args = parser.parse_args()
    main(args)