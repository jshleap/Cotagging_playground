from sklearn.decomposition import PCA
from utilities4cotagging import read_geno
import pandas as pd
import sys
import argparse

def do_pca(g, n_comp):
    """
    Perform a PCA on the genetic array and return n_comp of it

    :param g: Genotype array
    :param n_comp: Number of components sought
    :return: components array
    """
    pca = PCA(n_components=n_comp)
    pca = pca.fit_transform(g)
    return pca


def main(bfile, n_comps, cpus, mem):
    (bim, fam, g) = read_geno(bfile, 0, cpus, max_memory=mem)
    cols = ['PC%d' % (x + 1) for x in range(n_comps)]
    pca = pd.DataFrame(do_pca(g, n_comps), columns=cols)
    cols = pca.columns.tolist()
    pca['iid'] = fam.iid.tolist()
    pca['fid'] = fam.fid.tolist()
    ordered_cols = ['fid','iid'] + cols
    pca = pca.loc[:, ordered_cols]
    pca.to_csv('%s.pca' % bfile, sep=' ', header=False, index=False)
    return pca

# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bfile')
    parser.add_argument('-t', '--cpus', type=int)
    parser.add_argument('-m', '--mem', type=int)
    parser.add_argument('-c', '--comps', type=int)
    args = parser.parse_args()
    main(args.bfile, args.comps, args.cpus, args.mem)
